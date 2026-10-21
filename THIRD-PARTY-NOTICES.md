<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-Party Notices

This file lists third-party code and licenses distributed with this project.

Open-source components distributed inside the pre-built container images
(everything installed on top of the NGC base image, including transitive
dependencies) are listed in
[the versioned, per-variant container notices](docker/notices/README.md).
The [September CUDA12 table](THIRD-PARTY-NOTICES-CONTAINER.md) is historical.

## LongLive VAE window mapping

- File: `model_training/common/context_parallel.py`
- Source: [NVlabs/LongLive](https://github.com/NVlabs/LongLive),
  `wan_5b/distributed/sp_training.py`
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES

The latent-to-raw-frame mapping follows LongLive's left-halo VAE chunking,
expressed here as a shared helper with explicit bounds and crop offsets.


## PRoPE Relative Positional Encoding

- File: `model_training/net/prope.py`
- Source: "Cameras as Relative Positional Encoding" (https://arxiv.org/pdf/2507.10496)
- License: MIT

```text
MIT License

Copyright (c) Authors of
"Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GenFusion Utility Code

- Files: `model_eval/genfusion_lossutil.py`, `model_eval/lpipsPytorch/`
- Source: Inception3D/GenFusion (https://github.com/Inception3D/GenFusion)
- License: MIT

```text
MIT License

Copyright (c) 2025 Sibo Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## LPIPS / PerceptualSimilarity

- Files: `model_eval/lpipsPytorch/`
- Source: richzhang/PerceptualSimilarity (https://github.com/richzhang/PerceptualSimilarity)
- License: BSD-2-Clause

```text
Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 3DGRUT

- Path: `thirdparty/3DGRUT-ArtiFixer`
- Repository: https://github.com/nv-tlabs/3DGRUT-ArtiFixer
- Upstream project: https://github.com/nv-tlabs/3dgrut
- ArtiFixer integration commit: 9fce681577334709078098206f6f414f36e1ad61
- License: Apache-2.0

The Apache License, Version 2.0 text is reproduced in this repository's
`LICENSE` file.
