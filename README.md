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
forward pass → 3D-Gaussian PLY export. It is sufficient to
reproduce the paper's reconstruction outputs from a recorded ncorev4
sequence.

## Pipeline Overview

NCore V4 Sequence ─► Frame Batching ─► Kelvin Forward Pass (JIT) ─► 3D Gaussians ─► PLY (per-chunk or merged)

## User Guide

<details>
<summary><b>Setup</b></summary>

#### Prerequisites

- **Python** 3.11
- **NVIDIA driver** >= 570 (CUDA 12.8 compatible)
- **GPU VRAM** ≥ 16 GB
- **uv** — the [Astral Python package manager](https://docs.astral.sh/uv/).
  Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or
  `pip install uv`.

```bash
git clone https://github.com/NVIDIA/instant-nurec.git
cd instant-nurec
./setup.sh
source .venv/bin/activate
```

`setup.sh` runs `uv sync --frozen`, which installs the locked dependency
tree from `uv.lock` into `.venv/`. The only CUDA dependency is whatever
the pinned `torch` wheel ships with.

The pretrained model `kelvin_jit.pt` (a TorchScript archive of the
Kelvin static-only forward) is fetched on first inference run from the
Hugging Face repo `nvidia/instant-nurec-kelvin` and cached at
`~/.cache/huggingface/nvidia/instant_nurec/kelvin/`. Set
`INSTANT_NUREC_FULL_PT` to a local path to override the auto-download.

</details>

<details>
<summary><b>Inference</b></summary>

`--ncore-path` accepts two input shapes:

##### Mode 1 — single sequence `.json` (NuRec-aligned)

The path is treated as one ncorev4 sequence metadata file.
This matches NuRec's own input convention.

```bash
./run.sh \
    --ncore-path /path/to/clips/<uuid>/pai_<uuid>.json \
    --output-dir /tmp/out \
    --merge none
```

##### Mode 2 — `.lst` manifest (batch)

The path is treated as a list of sequence JSON paths, one per line.
Each line may be absolute, relative-to-the-LST-file's directory, or
`~/`-prefixed; lines starting with `#` and blank lines are skipped;
mixed absolute + relative entries in a single LST are supported.

```
# example_manifest.lst
/abs/path/to/clips/<uuid_a>/pai_<uuid_a>.json
relative/path/to/clips/<uuid_b>/pai_<uuid_b>.json
~/symlinked/clips/<uuid_c>/pai_<uuid_c>.json
```

```bash
./run.sh \
    --ncore-path /path/to/example_manifest.lst \
    --output-dir /tmp/out \
    --merge frustum-ownership
```

`run.sh` validates the input + output paths and execs
`python run_inference.py`. You can also call the CLI directly:

```bash
python run_inference.py \
    --ncore-path /path/to/sequence.json \
    --output-dir /tmp/out \
    --merge none
```

Output layout: PLYs only, under `out_dir/<run_id>/ply/<sequence_id>/...ply`.

#### CLI reference

| flag | default | purpose |
| --- | --- | --- |
| `--ncore-path` | (required) | A `.json` file (single sequence) or a `.lst` manifest (one JSON path per line). |
| `--output-dir` | (required) | Directory the pipeline writes PLYs into. |
| `--merge` | `none` | `none` writes per-chunk PLYs (`<seq>_chunk{N}.ply`); `frustum-ownership` writes a single merged PLY per sequence (`<seq>.ply`). |
| `--camera-id` | `camera_front_wide_120fov` | ncorev4 context-camera id used as model input. Exactly one camera is required. |
| `--lidar-id` | `lidar_top_360fov` | ncorev4 LiDAR sensor id used to source cuboid tracks for dynamic-mask refinement. Must exist in the sequence's `lidar_sensors`. |
| `--max-chunks` | `8` | Maximum number of time-chunks processed per clip. One chunk spans up to 13.5 s, so the default covers 8 × 13.5 = 108 s. Clips longer than that are silently truncated unless this is increased — bump to `ceil(clip_seconds / 13.5)` for longer clips. |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |

#### Environment variables

| variable | purpose |
| --- | --- |
| `INSTANT_NUREC_FULL_PT` | Absolute path to a local `kelvin_jit.pt`. Takes priority over the auto-downloaded copy. |
| `INSTANT_NUREC_RUN_ID` | Override the per-run shortuuid; useful when scripting reproducible output paths. |

</details>

<details>
<summary><b>Repository Structure</b></summary>

```
instant-nurec/
├── instant_nurec/                  # main package (what ships in the wheel)
│   ├── cli.py                      # argparse entrypoint
│   ├── pretrained.py               # auto-downloads kelvin_jit.pt from HF on first run
│   ├── config_schema/              # pydantic schemas + defaults (post-JIT runtime knobs only)
│   ├── datasets/                   # ncorev4 ingest + cuboid-track helpers
│   ├── model/
│   │   ├── __init__.py             # make() — torch.jit.load + JITKelvinAdapter wiring
│   │   ├── jit_adapter.py          # KelvinInstantNuRec-shaped wrapper around the JIT module
│   │   └── system.py               # GaussiansInstantNuRecSystem (predict-loop harness)
│   ├── predict/                    # predict loop + PLY export + merge
│   ├── primitives/                 # KelvinInstantNuRecPrimitive
│   └── utils/                      # batch / geometry / sensors / nn-extensions
├── tests/                          # branch-coverage tests
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
