import argparse

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_func
from torch.distributed import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask

from model_training.net.context_parallel import all_gather_cat

# Context-parallel video data flow:
# * Datasets shard raw video by latent-frame range. Trainers encode the local
#   VAE window and pass local latent frames into the transformer.
# * The transformer keeps MLP/modulation/projection work local to that frame
#   shard. Self-attention performs Ulysses sequence/head exchange and therefore
#   builds masks in the all-to-all token layout, while outputs stay local during
#   training and CP-local validation.


def latent_frame_count(num_frames: int, temporal_scale: int) -> int:
    assert num_frames > 0, f"num_frames must be positive, got {num_frames}"
    assert temporal_scale > 0, f"temporal_scale must be positive, got {temporal_scale}"
    assert (
        num_frames - 1
    ) % temporal_scale == 0, (
        f"num_frames - 1 must be divisible by temporal_scale {temporal_scale}, got num_frames={num_frames}"
    )
    return (num_frames - 1) // temporal_scale + 1


def validate_latent_frame_layout(
    *,
    num_frames: int,
    temporal_scale: int,
    context_parallel_size: int,
    frames_per_block: int | None = None,
    require_cp_rank_frames_per_block: bool = False,
) -> int:
    latent_num_frames = latent_frame_count(num_frames, temporal_scale)
    assert context_parallel_size >= 1, "context_parallel_size must be positive"
    if frames_per_block is not None:
        assert frames_per_block > 0, f"frames_per_block must be positive, got {frames_per_block}"
        assert (
            latent_num_frames % frames_per_block == 0
        ), f"Latent frame count {latent_num_frames} must be divisible by frames_per_block {frames_per_block}"
    if context_parallel_size > 1:
        assert latent_num_frames % context_parallel_size == 0, (
            f"Latent frame count {latent_num_frames} must be divisible by context_parallel_size "
            f"{context_parallel_size}"
        )
        if require_cp_rank_frames_per_block:
            assert frames_per_block is not None, "frames_per_block is required for CP block-shard validation"
            frames_per_cp_rank = latent_num_frames // context_parallel_size
            assert frames_per_cp_rank % frames_per_block == 0, (
                f"CP shard has {frames_per_cp_rank} latent frames/rank, which must be divisible by "
                f"frames_per_block {frames_per_block}"
            )
    return latent_num_frames


def latent_range_to_raw_frame_slice(
    latent_start: int,
    latent_end: int,
    *,
    total_raw_frames: int,
    temporal_scale: int,
    halo_latent_frames: int,
) -> tuple[int, int, int, int]:
    """Map a latent-frame shard to the raw frames needed for Wan VAE encoding.

    The mapping follows LongLive's left-halo Wan VAE chunking
    (https://github.com/NVlabs/LongLive): nonzero chunks include the
    pseudo-prefix latent created by preceding raw frames, then crop back to
    the requested latent interval after encoding.
    """
    assert latent_start >= 0, f"latent_start must be non-negative, got {latent_start}"
    assert latent_end > latent_start, f"latent_end must be > latent_start, got {latent_start}, {latent_end}"
    assert total_raw_frames > 0, f"total_raw_frames must be positive, got {total_raw_frames}"
    assert temporal_scale > 0, f"temporal_scale must be positive, got {temporal_scale}"
    assert halo_latent_frames >= 0, f"halo_latent_frames must be non-negative, got {halo_latent_frames}"
    total_latent_frames = latent_frame_count(total_raw_frames, temporal_scale)
    assert (
        latent_end <= total_latent_frames
    ), f"latent range [{latent_start}, {latent_end}) exceeds {total_latent_frames} latent frames"

    halo_start = max(0, latent_start - halo_latent_frames)
    raw_start = 0 if halo_start == 0 else temporal_scale * (halo_start - 1)
    raw_end = min(total_raw_frames, 1 + temporal_scale * (latent_end - 1))

    crop_start = (0 if halo_start == 0 else 1) + latent_start - halo_start
    crop_end = crop_start + latent_end - latent_start
    return raw_start, raw_end, crop_start, crop_end


def conditioning_raw_frame_bounds(latent_start: int, latent_end: int, temporal_scale: int) -> tuple[int, int]:
    """Raw (pixel) frame range conditioning a latent frame range under the
    VAE's 1+4N temporal contract; closed-form so a CP rank's non-contiguous
    local chunks do not need running counters.

    Unlike ``latent_range_to_raw_frame_slice`` this covers exactly the
    conditioning-aligned raw frames: no VAE halo, and the start includes the
    +1 for the first-frame latent.
    """
    raw_start = 0 if latent_start == 0 else 1 + temporal_scale * (latent_start - 1)
    raw_end = 1 + temporal_scale * (latent_end - 1)
    return raw_start, raw_end


def cp_rank_temporal_ranges(
    cp_rank: int,
    cp_size: int,
    *,
    total_raw_frames: int,
    temporal_scale: int,
    halo_latent_frames: int,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int]]:
    """One CP rank's temporal ranges in the three active time bases.

    * latent: transformer/VAE latent frames, evenly sharded by CP.
    * conditioning_raw: raw frames aligned to exactly those latent frames.
    * vae_raw: conditioning_raw plus optional left halo for Wan VAE chunking.
    """
    total_latent_frames = latent_frame_count(total_raw_frames, temporal_scale)
    assert (
        total_latent_frames % cp_size == 0
    ), f"latent frame count {total_latent_frames} must divide by context_parallel_size {cp_size}"
    local_latent_frames = total_latent_frames // cp_size
    latent_start = cp_rank * local_latent_frames
    latent_end = latent_start + local_latent_frames
    return {
        "latent": (latent_start, latent_end),
        "conditioning_raw": conditioning_raw_frame_bounds(latent_start, latent_end, temporal_scale),
        "vae_raw": latent_range_to_raw_frame_slice(
            latent_start,
            latent_end,
            total_raw_frames=total_raw_frames,
            temporal_scale=temporal_scale,
            halo_latent_frames=halo_latent_frames,
        ),
    }


def _flex_block_sizes(block_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(block_size, tuple):
        return block_size
    return block_size, block_size


def _block_padded_length(length: int, block: int) -> int:
    return ((length + block - 1) // block) * block


Interval = tuple[int, int]


def _merge_intervals(intervals: list[Interval], total_length: int) -> list[Interval]:
    normalized: list[Interval] = []
    for start, end in intervals:
        start = max(0, start)
        end = min(total_length, end)
        if start < end:
            normalized.append((start, end))
    normalized.sort()

    merged: list[Interval] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _intersect_intervals(
    left: list[Interval],
    right: list[Interval],
) -> list[Interval]:
    intersections: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            intersections.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def _overlapping_tiles(intervals: list[tuple[int, int]], tile_size: int) -> set[int]:
    tiles: set[int] = set()
    for start, end in intervals:
        for tile in range(start // tile_size, (end - 1) // tile_size + 1):
            tiles.add(tile)
    return tiles


def _full_tiles(intervals: list[tuple[int, int]], *, tile_size: int, total_length: int) -> set[int]:
    tiles: set[int] = set()
    for start, end in intervals:
        for tile in range(start // tile_size, (end - 1) // tile_size + 1):
            tile_start = tile * tile_size
            tile_end = min((tile + 1) * tile_size, total_length)
            if start <= tile_start and tile_end <= end:
                tiles.add(tile)
    return tiles


def _rows_to_block_tensors(
    rows: list[list[int]],
    *,
    q_blocks: int,
    kv_blocks: int,
    device: torch.device,
    attention_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.zeros((q_blocks, kv_blocks), device=device, dtype=torch.int32)
    num_blocks = torch.empty((q_blocks,), device=device, dtype=torch.int32)
    for q_block, row in enumerate(rows):
        num_blocks[q_block] = len(row)
        if row:
            indices[q_block, : len(row)] = torch.tensor(row, device=device, dtype=torch.int32)
    # Every backend needs batch/head axes. One head broadcasts the sparse index tables.
    attention_heads = 1 if attention_heads is None else attention_heads
    indices = indices.view(1, 1, q_blocks, kv_blocks).expand(1, attention_heads, -1, -1).contiguous()
    num_blocks = num_blocks.view(1, 1, q_blocks).expand(1, attention_heads, -1).contiguous()
    return num_blocks, indices


def _sparse_block_mask_from_intervals(
    *,
    device: torch.device,
    total_length: int,
    kv_length: int | None = None,
    block_size: int | tuple[int, int],
    query_segments,
    allowed_intervals,
    mask_mod,
    attention_heads: int | None = None,
) -> BlockMask:
    if kv_length is None:
        kv_length = total_length
    q_block_size, kv_block_size = _flex_block_sizes(block_size)
    q_blocks = (total_length + q_block_size - 1) // q_block_size
    kv_blocks = (kv_length + kv_block_size - 1) // kv_block_size

    full_rows: list[list[int]] = []
    partial_rows: list[list[int]] = []
    for q_block in range(q_blocks):
        q_start = q_block * q_block_size
        q_end = min(total_length, q_start + q_block_size)
        segment_intervals = [
            _merge_intervals(allowed_intervals(segment), kv_length) for segment in query_segments(q_start, q_end)
        ]
        union = _merge_intervals([interval for intervals in segment_intervals for interval in intervals], kv_length)
        intersection = segment_intervals[0]
        for intervals in segment_intervals[1:]:
            intersection = _intersect_intervals(intersection, intervals)

        full = _full_tiles(intersection, tile_size=kv_block_size, total_length=kv_length)
        partial = _overlapping_tiles(union, kv_block_size) - full
        full_rows.append(sorted(full))
        partial_rows.append(sorted(partial))

    kv_num_blocks, kv_indices = _rows_to_block_tensors(
        partial_rows,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        device=device,
        attention_heads=attention_heads,
    )
    full_kv_num_blocks, full_kv_indices = _rows_to_block_tensors(
        full_rows,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        device=device,
        attention_heads=attention_heads,
    )
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(total_length, kv_length),
    )


def frame_block_causal_mask(
    *,
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    frames_per_block: int,
    block_size: int | tuple[int, int],
    attention_heads: int | None = None,
) -> BlockMask:
    """Build a block-causal self-attention mask for frame blocks.

    Used for block-causal diffusion-forcing forwards:
    diffusion-forcing training/validation and DMD generator/critic scoring.
    Each query frame block attends to all tokens through the end of that block.
    Under Ulysses CP the attention processor gathers sequence tokens before
    FlexAttention, so callers must pass the global post-patch frame count.
    """
    assert num_frames > 0, f"num_frames must be positive, got {num_frames}"
    assert frame_seqlen > 0, f"frame_seqlen must be positive, got {frame_seqlen}"
    assert frames_per_block > 0, f"frames_per_block must be positive, got {frames_per_block}"
    sequence_length = num_frames * frame_seqlen
    block_token_count = frame_seqlen * frames_per_block
    # attention_mask closures must not capture Python scalars: a captured
    # value is re-traced as symbolic (SymInt) once the same closure sees a
    # second in-process value -- validation interleaved with training does
    # this -- and the FLASH backend cannot compile symbolic captures. So
    # every value a closure needs is precomputed as a tensor indexed by
    # q_idx, sized to the block-aligned padded length flex_attention
    # evaluates mask_mod on.
    q_block_size, _ = _flex_block_sizes(block_size)
    padded_q = _block_padded_length(sequence_length, q_block_size)
    q = torch.arange(padded_q, dtype=torch.long, device=device)
    block_end_vec = (q // block_token_count + 1) * block_token_count

    def attention_mask(b, h, q_idx, kv_idx):
        return kv_idx < block_end_vec[q_idx]

    def query_segments(q_start: int, q_end: int) -> list[int]:
        segments: list[int] = []
        current = q_start
        while current < q_end:
            block = current // block_token_count
            next_boundary = min(q_end, (block + 1) * block_token_count, sequence_length)
            segments.append(block)
            current = next_boundary
        return segments

    def allowed_intervals(block: int) -> list[tuple[int, int]]:
        return [(0, min(sequence_length, (block + 1) * block_token_count))]

    return _sparse_block_mask_from_intervals(
        device=device,
        total_length=sequence_length,
        block_size=block_size,
        query_segments=query_segments,
        allowed_intervals=allowed_intervals,
        mask_mod=attention_mask,
        attention_heads=attention_heads,
    )


class _ContiguousGradient(torch.autograd.Function):
    """Keep gather gradients contiguous to avoid incorrect Gloo SDPA gradients.

    See tests/test_causal_training.py::test_all_gather_cat_with_grad_sdpa_backward_matches_global_oracle.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return gradient.contiguous()


def all_gather_cat_with_grad(
    tensor: torch.Tensor,
    dim: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return tensor
    gathered = dist_nn_func.all_gather(tensor.contiguous(), group=group)
    return torch.cat([_ContiguousGradient.apply(part) for part in gathered], dim=dim)


def all_to_all_with_grad(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return tensor
    world_size = dist.get_world_size(group=group)
    assert tensor.shape[scatter_dim] % world_size == 0, (
        f"cannot split dimension {scatter_dim} of size {tensor.shape[scatter_dim]} "
        f"across {world_size} context-parallel ranks"
    )

    inputs = [chunk.contiguous() for chunk in tensor.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(chunk) for chunk in inputs]
    return torch.cat(list(dist_nn_func.all_to_all(outputs, inputs, group=group)), dim=gather_dim).contiguous()


def gather_kv_cat(tensor: torch.Tensor, mesh, dim: int) -> torch.Tensor:
    """Gather cached K/V rows across the CP group, concatenated in rank order
    along ``dim``, picking the collective for the current grad mode: DMD
    student rollouts backprop through the exit denoising step's gathered K/V,
    so grad-enabled reads must use the autograd-aware gather, while no-grad
    (inference) reads stay on the plain collective — whose loud grad-raise
    keeps guarding that no-grad-only helper.
    """
    if torch.is_grad_enabled():
        return all_gather_cat_with_grad(tensor, dim=dim, group=mesh.get_group())
    return all_gather_cat(tensor, mesh, dim)


def configure_context_parallel_shard_size(args: argparse.Namespace) -> None:
    if args.context_parallel_size <= 1:
        return
    if args.shard_size is None:
        args.shard_size = args.context_parallel_size
        return
    assert args.shard_size == args.context_parallel_size, (
        "shard_size must equal context_parallel_size for context-parallel training "
        "so FSDP shard groups align with CP groups"
    )


def create_context_parallel_mesh(context_parallel_size: int, device_type: str) -> DeviceMesh | None:
    if context_parallel_size <= 1:
        return None
    if not dist.is_initialized():
        raise RuntimeError("context_parallel_size > 1 requires distributed training")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert (
        world_size % context_parallel_size == 0
    ), f"world size {world_size} must be divisible by CP size {context_parallel_size}"

    local_mesh = None
    for start in range(0, world_size, context_parallel_size):
        ranks = list(range(start, start + context_parallel_size))
        mesh = DeviceMesh(device_type, ranks)
        if rank in ranks:
            local_mesh = mesh

    if local_mesh is None:
        raise RuntimeError(f"rank {rank} did not find a CP mesh")
    return local_mesh


def enable_training_context_parallel(pipe, context_parallel_size: int, device_type: str) -> None:
    local_mesh = create_context_parallel_mesh(context_parallel_size, device_type)
    if local_mesh is not None:
        pipe.transformer.enable_context_parallel(local_mesh)
