<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Half-covisibility view sampling

`run_sparse_reconstruction` divides each scene's cameras into two subsets. It
finds the farthest camera pair, ranks all cameras by pose distance from the
first, then assigns the nearer half to subset 0 and the rest to subset 1.

Within each subset, farthest-point sampling starts at its camera of the pair.
Each next camera maximizes its distance to the nearest camera already selected.
The ordered indices are written to `half_covisibility_sampled_indices_0.json`
and `half_covisibility_sampled_indices_1.json`. Reconstruction uses the first
`num_selected_indices` entries from each order.
