# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


class RolloutKvCache:
    """Per-layer K/V cache for ONE chunked AR rollout, shared by the Wan and
    Cosmos3 wrappers.

    Three write-once slot kinds per layer:

    * ``text_kv``: prompt-derived K/V, cached once per rollout, read by every
      later pass.
    * ``neighbor_kv``: reference-view K/V, same lifecycle. Both context slots
      store detached: neither family has a trainable parameter upstream of a
      cached context write (Cosmos3's text stream is its frozen reasoner; Wan
      fills these slots only on no-grad passes), so detaching changes neither
      values nor gradients.
    * gen chunks: one (K, V) entry per FINISHED chunk of generated frames,
      appended in temporal order with a (start_frame, num_frames) record.
      Entries are never mutated, and reads slice ``[0, num_chunks)``
      explicitly, so a gradient-checkpointing recompute during backward
      (after later chunks were appended) still sees exactly the history its
      forward saw. Eviction (the sliding attention window) drops whole
      chunks after a fixed anchor; coverage keeps counting evicted frames,
      so the contiguity contract rejects duplicate, gapped, or re-run chunk
      passes even after eviction.

    Tensor shapes and CP coordinates are the caller's: under CP both families
    store this rank's frame-shard rows (with all KV heads) and gather across
    the CP group at read time. Frame records diverge on purpose — Cosmos3
    records GLOBAL latent frames (its pass contract, ``kv_cache_start_frame``,
    is the global chunk start and also feeds mRoPE and opacity packing), while
    Wan under CP records this rank's local frames. Coverage arithmetic reads
    only the records, and every rank commits the same chunks in the same
    order, so eviction stays in cross-rank lockstep either way. Lifetime is
    one generate call; never reuse a cache across rollouts (positions, prompt,
    and conditioning are all rollout-specific).
    """

    def __init__(self, num_layers: int, *, truncate_rollout_gradients: bool):
        assert num_layers > 0, f"RolloutKvCache needs at least one layer, got {num_layers}"
        self.num_layers = int(num_layers)
        # Gen-chunk storage policy, one value per training run
        # (--truncate_rollout_gradients). False: entries keep their autograd
        # graphs, so a later chunk's loss backpropagates through the cached
        # context into the passes that produced it — training teaches the
        # model how its frames serve as context for later frames, at the
        # cost of holding every chunk's activations until backward. True:
        # entries are stored detached — credit is cut at chunk boundaries
        # and a chunk's activations free after its own backward. Values are
        # identical either way; only gradients and memory differ.
        self.truncate_rollout_gradients = bool(truncate_rollout_gradients)
        # Per layer: write-once context (K, V) slots.
        self._text_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        self._neighbor_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        # Per layer, per chunk: (K, V) of the finished chunk's frames.
        self._gen_chunks: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(self.num_layers)]
        # Per layer, per chunk: (start_frame, num_latent_frames);
        # store_gen_chunk asserts contiguous coverage against these records.
        self._gen_chunk_frames: list[list[tuple[int, int]]] = [[] for _ in range(self.num_layers)]
        # Per layer: latent frames evicted by the sliding attention window
        # (see total_stored_frames for the coverage contract).
        self._evicted_gen_frames: list[int] = [0] * self.num_layers

    @property
    def num_cached_chunks(self) -> int:
        chunk_counts = {len(chunks) for chunks in self._gen_chunks}
        assert len(chunk_counts) == 1, (
            f"gen K/V cache is ragged across layers ({sorted(chunk_counts)} chunks): a chunk store "
            "must write every layer exactly once"
        )
        return chunk_counts.pop()

    @property
    def total_stored_frames(self) -> int:
        """Contiguously stored latent-frame count — the exclusive end of the
        frame range ``[0, total)`` the gen cache has EVER covered (evicted
        frames included: coverage never rewinds). The next chunk pass must
        start exactly here."""
        frame_totals = {
            evicted + sum(num_frames for _, num_frames in records)
            for evicted, records in zip(self._evicted_gen_frames, self._gen_chunk_frames, strict=True)
        }
        assert len(frame_totals) == 1, (
            f"gen K/V frame coverage is ragged across layers ({sorted(frame_totals)} frames): a chunk "
            "store must write every layer exactly once"
        )
        return frame_totals.pop()

    def assert_pass_starts_at(self, start_frame: int, pass_label: str = "pass") -> None:
        """Every cache pass carries the chunk that starts exactly where stored
        coverage ends; a duplicate, gapped, or re-run chunk pass fails here
        instead of silently mis-positioning history."""
        assert start_frame == self.total_stored_frames, (
            f"kv-cache {pass_label} starts at latent frame {start_frame}, but the cache has stored "
            f"frames [0, {self.total_stored_frames}): chunks run contiguously in temporal order, and "
            "a RolloutKvCache lives for ONE rollout — never reuse it across generate calls."
        )

    def evict_to_window(self, window_chunks: int | None, sink_chunks: int) -> None:
        """Evict retained gen chunks until a ``window_chunks``-chunk attention
        window has room for the next chunk, keeping the first ``sink_chunks``
        as the fixed anchor; ``None`` (unlimited window) never evicts."""
        if window_chunks is None:
            return
        while self.num_cached_chunks > window_chunks - 1:
            self.evict_oldest_gen_chunk(sink_chunks)

    def evict_oldest_gen_chunk(self, num_anchor_chunks: int) -> None:
        """Drop the oldest retained gen chunk after the first
        ``num_anchor_chunks`` anchor chunks on every layer, freeing its K/V;
        the retained list stays in temporal order. Reads shrink but coverage
        does not (see total_stored_frames)."""
        assert num_anchor_chunks >= 0, f"num_anchor_chunks must be non-negative, got {num_anchor_chunks}"
        for layer_index in range(self.num_layers):
            chunks = self._gen_chunks[layer_index]
            frame_records = self._gen_chunk_frames[layer_index]
            assert (
                len(chunks) > num_anchor_chunks
            ), f"layer {layer_index}: no retained gen chunk to evict after the {num_anchor_chunks}-chunk anchor"
            chunks.pop(num_anchor_chunks)
            start_frame, num_frames = frame_records.pop(num_anchor_chunks)
            anchor_frames = sum(record_frames for _, record_frames in frame_records[:num_anchor_chunks])
            assert start_frame == anchor_frames + self._evicted_gen_frames[layer_index], (
                f"layer {layer_index}: eviction bookkeeping drifted — the evicted chunk starts at "
                f"{start_frame}, not right after the {anchor_frames} anchor frames and "
                f"{self._evicted_gen_frames[layer_index]} already-evicted frames"
            )
            self._evicted_gen_frames[layer_index] += num_frames

    def _store_context_kv(
        self,
        slots: list[tuple[torch.Tensor, torch.Tensor] | None],
        slot_name: str,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        assert (
            slots[layer_index] is None
        ), f"layer {layer_index}: {slot_name} K/V are cached exactly once per rollout (write-once)"
        slots[layer_index] = (key.detach().clone(), value.detach().clone())

    def has_text_kv(self, layer_index: int) -> bool:
        return self._text_kv[layer_index] is not None

    def store_text_kv(self, layer_index: int, k_text: torch.Tensor, v_text: torch.Tensor) -> None:
        self._store_context_kv(self._text_kv, "text", layer_index, k_text, v_text)

    def read_text_kv(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_kv[layer_index]
        assert cached is not None, f"layer {layer_index}: text K/V have not been cached yet"
        return cached

    def has_neighbor_kv(self, layer_index: int) -> bool:
        return self._neighbor_kv[layer_index] is not None

    def store_neighbor_kv(self, layer_index: int, k_neighbor: torch.Tensor, v_neighbor: torch.Tensor) -> None:
        self._store_context_kv(self._neighbor_kv, "neighbor", layer_index, k_neighbor, v_neighbor)

    def read_neighbor_kv(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._neighbor_kv[layer_index]
        assert cached is not None, f"layer {layer_index}: neighbor K/V have not been cached yet"
        return cached

    def store_gen_chunk(
        self,
        layer_index: int,
        chunk_index: int,
        start_frame: int,
        num_frames: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> None:
        """Append one layer's (K, V) for a finished chunk. Call from outside
        gradient-checkpointed code: a backward-time recompute would repeat
        the store and trip the write-once contract — forwards stage per-layer
        K/V and commit once per pass (commit_staged_chunk)."""
        chunks = self._gen_chunks[layer_index]
        frame_records = self._gen_chunk_frames[layer_index]
        assert chunk_index == len(chunks), (
            f"layer {layer_index}: gen K/V chunks are write-once and sequential — expected chunk "
            f"{len(chunks)}, got {chunk_index}"
        )
        covered_frames = self._evicted_gen_frames[layer_index] + sum(
            record_frames for _, record_frames in frame_records
        )
        assert num_frames > 0 and start_frame == covered_frames, (
            f"layer {layer_index}: gen K/V must be stored contiguously in temporal order — this chunk "
            f"covers latent frames [{start_frame}, {start_frame + num_frames}) but the layer has stored "
            f"[0, {covered_frames})"
        )
        if self.truncate_rollout_gradients:
            k_chunk, v_chunk = k_chunk.detach().clone(), v_chunk.detach().clone()
        chunks.append((k_chunk, v_chunk))
        frame_records.append((int(start_frame), int(num_frames)))

    def commit_staged_chunk(
        self,
        stage: dict[int, tuple[torch.Tensor, torch.Tensor]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        """Store a forward's staged chunk on every layer at once, at the
        chunk index the cache is up to; a stage that does not cover every
        layer exactly once fails as a cache contract error instead of a
        KeyError."""
        assert sorted(stage) == list(range(self.num_layers)), (
            f"staged chunk covers layers {sorted(stage)}: a chunk store must write every layer of "
            f"[0, {self.num_layers}) exactly once"
        )
        chunk_index = self.num_cached_chunks
        for layer_index in range(self.num_layers):
            key, value = stage[layer_index]
            self.store_gen_chunk(layer_index, chunk_index, start_frame, num_frames, key, value)

    def read_gen_history(self, layer_index: int, num_chunks: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        chunks = self._gen_chunks[layer_index]
        assert (
            0 <= num_chunks <= len(chunks)
        ), f"layer {layer_index}: requested {num_chunks} history chunks, cached {len(chunks)}"
        history = chunks[:num_chunks]
        return [entry[0] for entry in history], [entry[1] for entry in history]
