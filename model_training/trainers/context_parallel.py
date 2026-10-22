from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from accelerate import Accelerator

from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.prompt_formats import prompt_payload_for_text_dropout
from model_training.trainers.trainer import Trainer


class ContextParallelVideoMixin:
    """Shared frame-sharded training helpers for video training objectives.

    Training datasets slice CP temporal shards in ``__getitem__`` so DataLoader
    workers never materialize full 1533-frame tensors; under active CP,
    ``process_inputs`` requires the resulting per-rank batch metadata.
    """

    def _cp_transformer(self):
        transformer = self.pipe.transformer
        module = getattr(transformer, "module", None)
        return module if module is not None else transformer

    def _cp_mesh(self):
        cp_mesh = self._cp_transformer().cp_mesh
        return cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None

    def _cp_rank_size(self) -> tuple[int, int]:
        transformer = self._cp_transformer()
        cp_rank = transformer.cp_rank
        cp_size = transformer.cp_world_size
        assert 0 <= cp_rank < cp_size, f"context parallel rank {cp_rank} must be in [0, {cp_size})"
        return cp_rank, cp_size

    def _is_cp_source_rank(self) -> bool:
        return self._cp_rank_size()[0] == 0

    def _uses_context_parallel(self) -> bool:
        return self._cp_rank_size()[1] > 1

    def _local_frame_start(self, local_latent_frames: int) -> int:
        cp_rank, cp_size = self._cp_rank_size()
        if cp_size == 1:
            return 0
        global_latent_frames = (self.args.num_frames - 1) // self.pipe.vae.config.scale_factor_temporal + 1
        assert global_latent_frames == local_latent_frames * cp_size, (
            f"local latent frames ({local_latent_frames}) do not match global num_frames "
            f"({global_latent_frames}) and CP size ({cp_size})"
        )
        return cp_rank * local_latent_frames

    def _broadcast_cp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        cp_mesh = self._cp_mesh()
        if cp_mesh is not None:
            ranks = [int(rank) for rank in cp_mesh.mesh.view(-1).tolist()]
            cp_rank, _ = self._cp_rank_size()
            if dist.is_initialized():
                assert (
                    dist.get_rank() == ranks[cp_rank]
                ), f"CP rank {cp_rank} must map to global rank {ranks[cp_rank]}, got {dist.get_rank()}"
            src = ranks[0]
            dist.broadcast(tensor, src=src, group=cp_mesh.get_group())
        return tensor

    def _broadcast_cp_data_tensor(self, value: torch.Tensor, group, src: int) -> torch.Tensor:
        if value.device != self.accelerator.device:
            value = value.to(self.accelerator.device, non_blocking=True)

        ndim = torch.empty((), device=value.device, dtype=torch.long)
        if self._is_cp_source_rank():
            ndim.fill_(value.ndim)
        dist.broadcast(ndim, src=src, group=group)

        shape = torch.empty(int(ndim.item()), device=value.device, dtype=torch.long)
        if self._is_cp_source_rank():
            shape.copy_(torch.tensor(value.shape, device=value.device, dtype=torch.long))
        dist.broadcast(shape, src=src, group=group)

        target_shape = tuple(int(dim) for dim in shape.tolist())
        if tuple(value.shape) != target_shape:
            value = torch.empty(target_shape, device=value.device, dtype=value.dtype)
        elif not value.is_contiguous():
            value = value.contiguous()

        dist.broadcast(value, src=src, group=group)
        return value

    @staticmethod
    def _cp_batch_scalar(value) -> int:
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, f"Expected scalar CP metadata, got shape {tuple(value.shape)}"
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            assert len(value) == 1, f"Expected one CP metadata value, got {value!r}"
            return ContextParallelVideoMixin._cp_batch_scalar(value[0])
        return int(value)

    @staticmethod
    def _cp_batch_bool(value) -> bool:
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, f"Expected scalar CP metadata flag, got shape {tuple(value.shape)}"
            return bool(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            assert len(value) == 1, f"Expected one CP metadata flag, got {value!r}"
            return ContextParallelVideoMixin._cp_batch_bool(value[0])
        return bool(value)

    @staticmethod
    def _cp_batch_pair(value) -> tuple[int, int]:
        if isinstance(value, torch.Tensor):
            assert value.numel() == 2, f"Expected pair CP metadata, got shape {tuple(value.shape)}"
            flattened = value.reshape(-1)
            return int(flattened[0].item()), int(flattened[1].item())
        if isinstance(value, (list, tuple)):
            assert len(value) == 2, f"Expected pair CP metadata, got {value!r}"
            return (
                ContextParallelVideoMixin._cp_batch_scalar(value[0]),
                ContextParallelVideoMixin._cp_batch_scalar(value[1]),
            )
        raise TypeError(f"Unsupported CP metadata pair type: {type(value)!r}")

    def _is_data_sharded_for_cp(self, data: dict) -> bool:
        value = data.get("_cp_data_sharded")
        return False if value is None else self._cp_batch_bool(value)

    def _normalise_cp_sharded_batch_metadata(self, data: dict) -> dict:
        local = dict(data)
        local["_cp_vae_crop"] = self._cp_batch_pair(local["_cp_vae_crop"])
        local["_cp_vae_raw_start"] = self._cp_batch_scalar(local["_cp_vae_raw_start"])
        local["_cp_conditioning_raw_start"] = self._cp_batch_scalar(local["_cp_conditioning_raw_start"])
        local["_cp_global_raw_frames"] = self._cp_batch_scalar(local["_cp_global_raw_frames"])
        local["_cp_data_sharded"] = True
        return local

    def _encode_video(self, frames: torch.Tensor, crop: tuple[int, int] | None = None) -> torch.Tensor:
        """Encode a full video or this CP rank's halo raw-frame window."""
        halo_latent_frames = self.args.vae_chunk_halo_latent_frames
        if crop is not None:
            encoded = self.pipe.encode_video_frames(frames)
            return encoded[:, :, crop[0] : crop[1]].contiguous()
        if halo_latent_frames is None:
            return self.pipe.encode_video_frames(frames)
        return self.pipe.encode_video_frames_chunked_for_cp(
            frames,
            halo_latent_frames,
            gather=not self._uses_context_parallel(),
        )

    def _encode_condition_zeros_cached(self, reference: torch.Tensor, crop: tuple[int, int] | None) -> torch.Tensor:
        """Cached zero-condition encode: the frozen-VAE encoding of an all-zeros
        video is deterministic per (shape, crop), so compute it once."""
        key = ("cp_condition", tuple(reference.shape), reference.dtype, reference.device, crop)
        cached = self._zero_encode_cache.get(key)
        if cached is None:
            cached = self._encode_video(torch.zeros_like(reference), crop)
            self._zero_encode_cache[key] = cached
        return cached

    def process_inputs(self, data: dict) -> dict[str, torch.Tensor | bool]:
        if self._uses_context_parallel():
            assert (
                self.args.vae_chunk_halo_latent_frames is not None
            ), "context parallel video training requires local VAE chunking"
            assert self._is_data_sharded_for_cp(data), (
                "context parallel training requires batches pre-sharded per CP rank by the dataset "
                "(missing _cp_data_sharded metadata; see DL3DVPairedDatasetTrain)"
            )
            data = self._normalise_cp_sharded_batch_metadata(data)
        num_frames = data.get("_cp_global_raw_frames", data["rgb_gt"].shape[1])
        processed_data = {"rgb_gt": data["rgb_gt"]}
        # Dropout choices are sampled once per CP group.  Otherwise different
        # ranks would train on inconsistent conditioning for adjacent frame
        # shards of the same video.
        drop_rendered, frames_to_drop, drop_text, ignore_neighbors = self._sample_input_dropout(
            num_frames,
            device=self.accelerator.device,
            is_source_rank=self._is_cp_source_rank(),
            broadcast=self._broadcast_cp_tensor,
        )

        opacity = data["opacity"].to(dtype=self.pipe.vae.dtype)
        if drop_rendered:
            condition = self._encode_condition_zeros_cached(data["rgb_rendered"], data.get("_cp_vae_crop"))
            opacity = torch.zeros_like(opacity)
        else:
            if frames_to_drop > 0:
                dropped_rendered = data["rgb_rendered"].clone()
                # frames_to_drop is defined in the global raw-video timeline;
                # translate that suffix boundary into this rank's local VAE and
                # conditioning windows before zeroing.
                drop_start = num_frames - frames_to_drop
                local_drop_start = max(0, drop_start - data.get("_cp_vae_raw_start", 0))
                if local_drop_start < dropped_rendered.shape[1]:
                    dropped_rendered[:, local_drop_start:] = 0
                condition = self._encode_video(dropped_rendered, data.get("_cp_vae_crop"))
                opacity_drop_start = max(0, drop_start - data.get("_cp_conditioning_raw_start", 0))
                if opacity_drop_start < opacity.shape[1]:
                    if opacity is data["opacity"]:
                        opacity = opacity.clone()
                    opacity[:, opacity_drop_start:] = 0
            else:
                condition = self._encode_video(data["rgb_rendered"], data.get("_cp_vae_crop"))

        processed_data["condition"] = condition
        processed_data["opacity"] = opacity
        if "_cp_vae_crop" in data:
            processed_data["_cp_vae_crop"] = data["_cp_vae_crop"]

        # Text dropout goes through the shared payload helper so token-id
        # prompts (Cosmos3) substitute their stored unconditional instead of
        # being zeroed; the drop decision itself is the CP-broadcast one.
        processed_data.update(prompt_payload_for_text_dropout(data, drop_text=drop_text))

        # Wan zeroes the reference output; Cosmos3 omits reference tokens.
        # A cached placeholder avoids encoding images that will not affect the output.
        if ignore_neighbors:
            processed_data["neighbors_condition"] = self._encode_zeros_cached(data["rgb_neighbors"], neighbors=True)
        else:
            processed_data["neighbors_condition"] = self.pipe.encode_neighbors(data["rgb_neighbors"])
        processed_data["ignore_neighbors"] = ignore_neighbors
        processed_data["neighbor_w2cs"] = data["neighbor_w2cs"]
        processed_data["neighbor_Ks"] = data["neighbor_Ks"]
        processed_data["camera_rays"] = data["camera_rays"].to(dtype=self.pipe.vae.dtype)
        if data.get("neighbor_camera_rays") is not None:
            processed_data["neighbor_camera_rays"] = data["neighbor_camera_rays"].to(dtype=self.pipe.vae.dtype)
        processed_data["w2cs"] = data["w2cs"]
        processed_data["Ks"] = data["Ks"]
        return processed_data


# The mixin must stay first so its CP process_inputs overrides the non-CP
# TrainerBase.process_inputs.
class ContextParallelTrainer(ContextParallelVideoMixin, Trainer):
    """Stage-1 supervised trainer: Trainer's flow-matching objective with
    CP-aware preprocessing and per-rank latent shards; at CP size 1 it
    reduces to the same computation.

    The stage is bidirectional, so frames_per_block is always None.
    """

    pipe: ArtifixerPipelineBase

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        pipe: ArtifixerPipelineBase,
        step_offset: int = 0,
    ):
        super().__init__(
            args, accelerator, optimizer, scheduler, train_dataloader, val_datasets, pipe, None, step_offset
        )

    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        data = self.process_inputs(data)
        batch_size = data["rgb_gt"].shape[0]
        # CP-local training stores one VAE crop/raw-offset tuple per rank.  The
        # launchers use per-rank batch size 1; supporting >1 would require
        # per-sample CP metadata and separate temporal slicing for each scene.
        assert batch_size == 1, "ContextParallelTrainer expects the default per-rank batch size of 1"
        latent_gt = self._encode_video(data["rgb_gt"], data.get("_cp_vae_crop"))
        local_cp_latents = self._uses_context_parallel()
        local_frame_start = self._local_frame_start(latent_gt.shape[2]) if local_cp_latents else 0
        # Flow-matching source z_mix (paper eq. 3): the degraded latent where
        # opacity is high, Gaussian noise elsewhere — not pure noise.
        source_latent = self.pipe.prepare_latents(
            data["condition"],
            data["opacity"],
            local_frame_start == 0,
            broadcast_cp_noise=not local_cp_latents,
        )

        timestep_id = (
            torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.accelerator.device)
            if self._is_cp_source_rank()
            else torch.empty(batch_size, device=self.accelerator.device, dtype=torch.long)
        )
        self._broadcast_cp_tensor(timestep_id)
        timestep = self.train_scheduler.timesteps[timestep_id]
        model_input = self.train_scheduler.add_noise(latent_gt, source_latent, timestep).to(latent_gt.dtype)

        with torch.no_grad():
            training_target = self.train_scheduler.training_target(latent_gt, source_latent)

        transformer_timestep = self.pipe.prepare_transformer_timestep(
            timestep,
            model_input,
            frames_per_block=None,
        )

        prompt_kwargs = self._prompt_kwargs(data)
        noise_pred = self.pipe.transformer(
            hidden_states=model_input,
            timestep=transformer_timestep,
            neighbor_hidden_states=data["neighbors_condition"],
            ignore_neighbors=data["ignore_neighbors"],
            opacity=data["opacity"],
            camera_rays=data["camera_rays"],
            w2cs=data["w2cs"],
            neighbor_w2cs=data["neighbor_w2cs"],
            Ks=data["Ks"],
            neighbor_Ks=data["neighbor_Ks"],
            **self._neighbor_camera_kwargs(data.get("neighbor_camera_rays")),
            return_dict=False,
            frame_offset=local_frame_start,
            local_cp_input=local_cp_latents,
            **prompt_kwargs,
        )[0]

        mse_loss = torch.nn.functional.mse_loss(
            noise_pred.float(),
            training_target.float(),
            reduction="mean",
        )
        # Report a detached tensor, not .item(): the train loop converts it to
        # a float after backward is queued, so the host does not block on the
        # forward pass.
        metrics = {"mse_loss": mse_loss.detach()}
        loss = mse_loss * self.train_scheduler.training_weight(timestep).mean()

        return loss, metrics
