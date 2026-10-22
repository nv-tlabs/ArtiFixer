<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing

This project will only accept contributions under Apache-2.0.

By submitting a contribution, you agree that your contribution is licensed under the Apache License, Version 2.0.

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.
To sign off on a commit you simply use the --signoff (or -s) option when committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

## Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Development and testing

Use the [project environment](docs/setup.md) and install pytest. From the
repository root, run:

```bash
python -m pytest tests -q
python -m black --check .
python -m isort --check-only .
```

Black 24.8.0 and isort 5.13.2 are pinned in `docker/requirements.txt`; settings
are in [pyproject.toml](pyproject.toml). Keep CPU-runnable `slow` tests in the
full run. `-m 'not slow'` provides quicker feedback but is only a subset.

The suite needs ShellCheck on `PATH` and Cosmos3-Nano tokenizer assets at the
revision in [pins.py](model_training/pins.py). Tokenizer parity needs no model
weights. Video tests need [host FFmpeg](docs/setup.md#ffmpeg). Pretrained LPIPS
comparison also needs `lpips==0.1.4` and its model weights.

Report failures and skips explicitly. A CPU run skips CUDA cases, and ordinary
pytest does not cover the standalone distributed or smoke commands. See the
[testing reference](docs/testing.md) for assets, four-process normalization, GPU
checks, and smoke runs. Match verification to the changed behavior; skipped
checks are not passes.

Inside Slurm, run the single-process suite without inherited `SLURM_*`,
`PMI_*`, or `PMIX_*` variables; they can make Accelerate initialize distributed mode.
