<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# InstantNuRec: Feed-Forward 3D Gaussian Reconstruction from Driving Logs

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.00602)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt)
[![Model](https://img.shields.io/badge/HF-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/instant-nurec-kelvin)

**NVIDIA**

### Abstract

Reconstructing dynamic outdoor scenes from autonomous-vehicle driving
logs traditionally requires lengthy per-scene optimization. InstantNuRec
takes a different route: a feed-forward transformer directly infers a
dynamic 3D-Gaussian scene representation in a single forward pass.
Given a short window of multi-camera observations from an AV log, the
model emits a Gaussian primitive per pixel — covering geometry,
appearance, and per-Gaussian motion — which can be rendered in real
time and interchanged with existing simulation pipelines.

This repository is the public reference implementation of the predict
side of the **Kelvin** model: ncorev4 ingest → frame batch prep →
forward pass → 3D-Gaussian PLY export. It is sufficient to reproduce
the paper's reconstruction outputs from a recorded ncorev4 sequence.

## Pipeline Overview

NCore V4 Sequence ─► Frame Batching ─► Kelvin Forward Pass ─► 3D Gaussians ─► PLY (per-chunk or merged)

## User Guide

<details>
<summary><b>Setup</b></summary>

#### Prerequisites

- **Python** 3.12
- **NVIDIA driver** >= 570 (CUDA 12.8 compatible)
- **GPU VRAM** ≥ 16 GB

```bash
git clone https://github.com/NVIDIA/instant-nurec.git
cd instant-nurec
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a Python venv and runs `pip install -e .`. The only
CUDA dependency is whatever the pinned `torch` wheel ships with.

The pretrained model `kelvin_full.pt` is fetched on first inference run
from the Hugging Face repo `nvidia/instant-nurec-kelvin` and cached at
`~/.cache/instant_nurec/`. Set `INSTANT_NUREC_FULL_PT` to a local path
to override the auto-download.

</details>

<details>
<summary><b>Inference</b></summary>

The two canonical invocations:

```bash
# Per-chunk PLYs.
./run.sh \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out/no_merge \
    --merge none

# Single merged PLY per sequence.
./run.sh \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out/merge \
    --merge frustum-ownership
```

`run.sh` validates the inputs and execs `python run_inference.py`; you
can also call the CLI directly:

```bash
python run_inference.py \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out \
    --merge none
```

#### CLI reference

| flag | purpose |
| --- | --- |
| `--ncore-path` | ncorev4 dataset root containing `debug.lst`. Required. |
| `--output-dir` | Directory the pipeline writes PLYs (and the resolved config) into. Required. |
| `--merge` | `none` (default) for per-chunk PLYs, `frustum-ownership` for a single merged PLY. |
| `--log-level` | `DEBUG` / `INFO` (default) / `WARNING` / `ERROR` / `CRITICAL`. |

#### Environment variables

| variable | purpose |
| --- | --- |
| `INSTANT_NUREC_FULL_PT` | Absolute path to a local `kelvin_full.pt`. Takes priority over the auto-downloaded copy. |

</details>

<details>
<summary><b>Repository Structure</b></summary>

```
instant-nurec/
├── instant_nurec/                  # main package
│   ├── cli.py                      # argparse entrypoint
│   ├── _hf_mock.py                 # HF resolver (auto-downloads on first run)
│   ├── config_schema/              # pydantic schemas + defaults
│   ├── datasets/                   # ncorev4 ingest + cuboid-track helpers
│   ├── model/                      # GaussiansInstantNuRecSystem + KelvinInstantNuRec + blocks
│   ├── predict/                    # predict loop + PLY export + merge
│   ├── primitives/                 # KelvinInstantNuRecPrimitive
│   └── utils/                      # batch / geometry / sensors / nn-extensions
├── tests/                          # branch-coverage tests
│   └── tolerance.json
├── run_inference.py                # main inference entry point
├── run.sh                          # input-validation wrapper
├── setup.sh                        # venv bootstrap
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE.txt
└── THIRD_PARTY_LICENSE.txt
```

</details>

<details>
<summary><b>Development</b></summary>

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
```

</details>

## License

This project is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt)
and individual file headers for details. Third-party attributions are
in [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yang2025storm,
  title   = {STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes},
  author  = {Yang, Jiawei and Huang, Jiahui and Chen, Yuxiao and Wang, Yan
             and Li, Boyi and You, Yurong and Sharma, Apoorva and Igl, Maximilian
             and Karkus, Peter and Xu, Danfei and Ivanovic, Boris
             and Wang, Yue and Pavone, Marco},
  journal = {arXiv preprint arXiv:2501.00602},
  year    = {2025}
}
```

## Disclaimer

InstantNuRec is trained for the autonomous-vehicle domain; results
outside that domain are not guaranteed.

AI models generate responses and outputs based on complex algorithms
and machine-learning techniques, and those responses or outputs may be
inaccurate or offensive. By downloading a model, you assume the risk of
any harm caused by any response or output of the model. By using this
software or model, you are agreeing to the terms and conditions of the
license, acceptable-use policy, and privacy policy as applicable.
