<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# InstantNuRec: Feed-Forward 3D Gaussian Reconstruction from Driving Logs

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://research.nvidia.com/labs/sil/projects/instant-nurec/) [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2607.14203) [![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt) [![Model](https://img.shields.io/badge/HF-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/instant-nurec) [![Data](https://img.shields.io/badge/NCore-0d9488?logo=database&logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)

**NVIDIA**

## Announcements

- **July 2026:** The [InstantNuRec project page](https://research.nvidia.com/labs/sil/projects/instant-nurec/) and [paper](https://arxiv.org/pdf/2607.14203) are now available.

## Abstract

3D simulation platforms are critical for autonomous driving because they enable end-to-end policy evaluation, thereby reducing development costs and improving safety. In recent years, neural simulation has become predominant, with methods such as NuRec playing a central role; however, these methods remain relatively slow and typically require per-scene tuning. In this work, we present Instant NuRec, a feed-forward neural reconstruction model that turns a short multi-view driving log into a fully simulatable 3D Gaussian Splatting (3DGS) world in a single forward pass. The model accepts multi-view input from a calibrated camera rig and emits a layered output consisting of static and dynamic 3DGS layers, a sky cubemap, and per-camera ISP corrections, while providing native support for non-pinhole camera models via 3DGUT. It reconstructs a 10–20-second multi-camera scene in roughly 1.5 seconds and achieves a PSNR on the Waymo Open Dataset that is 2.01 dB above the strongest evaluated baseline. Instant NuRec is deeply integrated into NuRec and is compatible with AlpaSim for closed-loop simulation.

> **Repository scope:** This standalone CLI exports only the static scene
> Gaussians to PLY. The abstract above describes the complete research model.

![InstantNuRec demo](docs/demo.gif)

## Pipeline Overview

This repo goes from ncorev4 ingest → frame batch prep → forward pass
→ 3D-Gaussian PLY export. The PLY output is usable directly as a
static reconstruction, and can also serve as initialization for
downstream NuRec training to reach higher fidelity.

Instant-NuRec and
[NuRec](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html)
share the same input (NCore V4 clip / HF dataset / sequence `.json`)
but run on different runtimes: Instant-NuRec is a native-Python
feed-forward preview (seconds per clip); NuRec is a Docker-based
per-scene refinement pipeline that produces a high-fidelity USDZ.

![InstantNuRec demo](docs/demo.gif)

## Support

For common errors and fixes (HF auth, driver / CUDA mismatch, OOM at
chunk-prep, `--max-chunks` truncation), see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

- **Usage questions and discussion:** post on the
  [NVIDIA Developer Forum (Omniverse / NuRec)](https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752).
- **Code-level bugs, documentation issues, and feature requests:** file a
  [GitHub issue](../../issues/new/choose) using the appropriate template. For
  bugs, include the full traceback, `nvidia-smi`, and `python --version`.
- **Security vulnerabilities:** use
  [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
  Do not file security issues publicly.

### Background

Instant NuRec is a feed-forward reconstruction model that converts
driving logs into 3D Gaussian Splatting (3DGS) representations. Its
vision-transformer backbone and DPT-decoders output a high-fidelity
3D environment that's ready for simulations.

Instant NuRec leverages the following foundational technologies:
[Depth-Anything-V3](https://github.com/ByteDance-Seed/depth-anything-3),
[STORM](https://github.com/NVlabs/GaussianSTORM), and
[BTimer](https://research.nvidia.com/labs/toronto-ai/bullet-timer/).

## Pipeline Overview

NCore V4 Sequence ─► Frame Batching ─► Eager PyTorch Model ─► 3D Gaussians ─► PLY (per-chunk or merged)

## User Guide

<details>
<summary><b>Setup</b></summary>

#### Prerequisites

- **Python** 3.11
- **NVIDIA driver and GPU VRAM** — see the
  [NuRec Hardware Setup and Requirements](https://docs.nvidia.com/nurec/basics/hardware.html#hardware-setup-and-requirements)
  page; Instant-NuRec inherits the same minimums.
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

This repo is native-Python only — no Docker required. If you want a
container, use the standard
[NuRec](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html)
image as a generic CUDA environment.

#### Download Model Checkpoints [optional]

> **Note:** The checkpoint selected by `--model` is auto-downloaded into
> the Hugging Face hub cache on the first inference run. PA-front remains
> the default.

However, you can also manually download the model into a directory of
your choice:

```bash
pip install huggingface_hub[cli]
hf auth login
hf download nvidia/instant-nurec --local-dir checkpoints
```

This places the following files in `checkpoints/`:

    checkpoints/
    └── pth/
        ├── instant_nurec_pa_front_1.1.0.pth
        ├── instant_nurec_pa_multiview_1.1.0.pth
        └── instant_nurec_pq_road_1.0.0.pth

Point the pipeline at this local copy by exporting:

```bash
export INSTANT_NUREC_FULL_PT="$(pwd)/checkpoints/pth/instant_nurec_pa_front_1.1.0.pth"
```

When using a local override, make sure the file matches the `--model`
selection.

</details>

<details>
<summary><b>Inference</b></summary>

> **Note:** The selected pretrained weights are fetched on first inference
> from the Hugging Face repo `nvidia/instant-nurec` and cached locally;
> subsequent runs read them from the cache. Set `INSTANT_NUREC_FULL_PT`
> to a matching local checkpoint to override the auto-download.

The following inference profiles are available:

| model | description | default input |
| --- | --- | --- |
| `pa-front` | Front-camera profile. **Dense pixel-aligned** Gaussians. | 18 × `camera_front_wide_120fov`, 784×448 |
| `pa-multiview` | 1, 3, or 5 cameras. **Dense pixel-aligned** Gaussians. | 18 frames per camera across front-wide, cross-left, and cross-right, 504×280 (54 images total) |
| `pq-front` | Fixed front-wide camera. **Selective point-query** Gaussians (fewer outputs). | 18 × `camera_front_wide_120fov`, 784×448 |

##### First run — end-to-end on a public demo clip

The clip lives in a gated HF dataset. Accept the terms at
[nvidia/PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)
while logged into Hugging Face, then `hf auth login` locally; the same
auth covers the `nvidia/instant-nurec` model auto-download on first run.

```bash
# Download the clip (~2 GB)
hf download \
    nvidia/PhysicalAI-Autonomous-Vehicles-NCore --repo-type dataset \
    --include "clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/*" \
    --local-dir ./demo_clip

# Reconstruct it
python run_inference.py \
    --model pa-front \
    --ncore-path ./demo_clip/clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.json \
    --output-dir ./demo_output \
    --merge
```

For the default `pa-front` command above, success looks like a single PLY at
`./demo_output/<run_id>/ply/pai_000da9de-.../pai_000da9de-....ply` —
~1.88 M Gaussians, kl-optimal voxelized from 2.87 M merged (3.18 M
pre-merge across 2 chunks) to land in `[0.9 * --n-gaussians,
--n-gaussians]` (default target 2 M). Omit `--merge` to write
per-chunk PLYs instead (voxelization is bundled with merge and
runs only when the flag is set).

To run another model on the same clip, select its profile, such as
`--model pa-multiview` or `--model pq-front`.

##### View your output

The PLY is a **3DGS** PLY (Gaussian Splatting), not a point cloud —
generic viewers like MeshLab / macOS Preview will fail to open it.
Use one of:

- [SuperSplat](https://playcanvas.com/supersplat/editor) — browser, no install.
- `ply_viewer` — shipped in the NuRec container.

`--ncore-path` accepts two input shapes:

##### Mode 1 — single sequence `.json` (NuRec-aligned)

The path is treated as one ncorev4 sequence metadata file.
This matches NuRec's own input convention.

```bash
./run.sh \
    --ncore-path /path/to/clips/<uuid>/pai_<uuid>.json \
    --output-dir /tmp/out
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
    --merge
```

`run.sh` validates the input + output paths and execs
`python run_inference.py`. You can also call the CLI directly:

```bash
python run_inference.py \
    --ncore-path /path/to/sequence.json \
    --output-dir /tmp/out
```

Output layout: PLYs only, under `out_dir/<run_id>/ply/<sequence_id>/...ply`.

#### CLI reference

| flag | default | purpose |
| --- | --- | --- |
| `--model` | `pa-front` | Input/checkpoint profile: `pa-front`, `pa-multiview`, or `pq-front`. |
| `--ncore-path` | (required) | A `.json` file (single sequence) or a `.lst` manifest (one JSON path per line). |
| `--output-dir` | (required) | Directory the pipeline writes PLYs into. |
| `--merge` | absent (false) | Boolean flag. When set, merges per-chunk primitives into a single frustum-ownership PLY per sequence (`<seq>.ply`) and runs kl-optimal voxelization (target count from `--n-gaussians`). Absent (default): per-chunk PLYs (`<seq>_chunk{N}.ply`), no voxelization. |
| `--n-gaussians` | `2000000` | Target number of static Gaussians after voxelization. Only consulted when `--merge` is set. The voxel size is searched iteratively via bracketed binary search to land the count in `[0.9 * target, target]`. |
| `--camera-id` | profile-dependent | Override a context camera. Repeat once per camera in canonical order. `pa-front` requires 1; `pa-multiview` supports 1, 3, or 5; `pq-front` is fixed to `camera_front_wide_120fov`. |
| `--max-chunks` | `8` | Maximum number of time-chunks processed per clip. One chunk spans up to 13.5 s, so the default covers 108 s. Longer clips are truncated and a `WARNING` logs the required value. |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |

#### Environment variables

| variable | purpose |
| --- | --- |
| `INSTANT_NUREC_FULL_PT` | Absolute path to a local weights-only checkpoint matching `--model`. Takes priority over the auto-downloaded copy. |
| `INSTANT_NUREC_RUN_ID` | Override the per-run shortuuid; useful when scripting reproducible output paths. |

</details>

<details>
<summary><b>Repository Structure</b></summary>

```
instant-nurec/
├── instant_nurec/                  # main package (what ships in the wheel)
│   ├── cli.py                      # argparse entrypoint
│   ├── pretrained.py               # auto-downloads weight checkpoint from HF on first run
│   ├── config_schema/              # pydantic schemas + public architecture defaults
│   ├── datasets/                   # ncorev4 ingest + cuboid-track helpers
│   ├── model/
│   │   ├── backbone/               # multi-view encoder, dense/PQ decoders, sky decoder
│   │   ├── blocks/                 # attention, embeddings, DPT, camera encoding
│   │   ├── kelvin.py               # dense full-model composition
│   │   ├── static_core.py          # eager PLY-reconstruction heads
│   │   ├── inference.py            # masking + primitive packaging
│   │   └── system.py               # predict-loop harness
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

## What's next?

The PLY you just wrote is usable directly as a static reconstruction.
If you want a high-fidelity, fully-trained scene, feed the PLY into
[NuRec](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html)
as initialization for per-scene refinement.

## License

This project is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt)
and individual file headers for details. Third-party attributions are
in [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@techreport{nvidia2026instantnurec,
  title       = {Instant NuRec: Feed-Forward 3D Gaussian Reconstruction
                 for Driving Scene Simulation},
  author      = {{NVIDIA}},
  institution = {NVIDIA},
  year        = {2026},
  url         = {https://arxiv.org/abs/2607.14203}
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
