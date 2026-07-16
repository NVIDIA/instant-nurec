<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# InstantNuRec: Feed-Forward 3D Gaussian Reconstruction from Driving Logs

[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt)
[![Model](https://img.shields.io/badge/HF-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/instant-nurec)
[![Data](https://img.shields.io/badge/NCore-0d9488?logo=database&logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)

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

## Model Architecture

The released model is implemented entirely in this repository. Its
pixel-aligned pipeline consists of:

1. a multi-view vision-transformer encoder with alternating local and
   global attention;
2. DPT heads for depth, RGB, normals, semantics, Gaussian parameters,
   and motion;
3. a cubemap sky decoder; and
4. per-camera affine color post-processing.

The reusable attention, embedding, DPT, and camera-encoding layers live
under [`instant_nurec/model/blocks`](instant_nurec/model/blocks), the
encoder/decoder definitions under
[`instant_nurec/model/backbone`](instant_nurec/model/backbone), and the
complete model composition in
[`instant_nurec/model/kelvin.py`](instant_nurec/model/kelvin.py).
Inference runs the PLY-relevant heads directly from source via
[`static_core.py`](instant_nurec/model/static_core.py); the downloaded
checkpoint contains weights only.

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

> **Note:** `pth/instant_nurec_pa_front_1.1.0.pth` is auto-downloaded into the Hugging Face
> hub cache on the first inference run.

However, you can also manually download the model into a directory of
your choice:

```bash
pip install huggingface_hub[cli]
hf auth login
hf download nvidia/instant-nurec --local-dir checkpoints
```

This places the following file in `checkpoints/`:

    checkpoints/
    └── pth/
        └── instant_nurec_pa_front_1.1.0.pth

Point the pipeline at this local copy by exporting:

```bash
export INSTANT_NUREC_FULL_PT="$(pwd)/checkpoints/pth/instant_nurec_pa_front_1.1.0.pth"
```

</details>

<details>
<summary><b>Inference</b></summary>

> **Note:** The pretrained weights `pth/instant_nurec_pa_front_1.1.0.pth` are fetched
> on first inference run from the Hugging Face
> repo `nvidia/instant-nurec` and cached locally; subsequent runs read
> it from the cache. Set `INSTANT_NUREC_FULL_PT` to a local path to
> override the auto-download.

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
    --ncore-path ./demo_clip/clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.json \
    --output-dir ./demo_output \
    --merge
```

Success looks like a single PLY at
`./demo_output/<run_id>/ply/pai_000da9de-.../pai_000da9de-....ply` —
~1.88 M Gaussians, kl-optimal voxelized from 2.87 M merged (3.18 M
pre-merge across 2 chunks) to land in `[0.9 * --n-gaussians,
--n-gaussians]` (default target 2 M). Omit `--merge` to write
per-chunk PLYs instead (voxelization is bundled with merge and
runs only when the flag is set).

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
| `--ncore-path` | (required) | A `.json` file (single sequence) or a `.lst` manifest (one JSON path per line). |
| `--output-dir` | (required) | Directory the pipeline writes PLYs into. |
| `--merge` | absent (false) | Boolean flag. When set, merges per-chunk primitives into a single frustum-ownership PLY per sequence (`<seq>.ply`) and runs kl-optimal voxelization (target count from `--n-gaussians`). Absent (default): per-chunk PLYs (`<seq>_chunk{N}.ply`), no voxelization. |
| `--n-gaussians` | `2000000` | Target number of static Gaussians after voxelization. Only consulted when `--merge` is set. The voxel size is searched iteratively via bracketed binary search to land the count in `[0.9 * target, target]`. |
| `--camera-id` | `camera_front_wide_120fov` | ncorev4 context-camera id used as model input. Exactly one camera is required. |
| `--max-chunks` | `8` | Maximum number of time-chunks processed per clip. One chunk spans up to 13.5 s, so the default covers 8 × 13.5 = 108 s. Longer clips are truncated and a `WARNING` is logged naming the dropped chunk count and the `--max-chunks` value needed to cover the full clip — bump to `ceil(clip_seconds / 13.5)` to silence it. |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |

#### Environment variables

| variable | purpose |
| --- | --- |
| `INSTANT_NUREC_FULL_PT` | Absolute path to a local `instant_nurec_pa_front_1.1.0.pth`. Takes priority over the auto-downloaded copy. |
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
│   │   ├── backbone/               # multi-view encoder, DPT decoder, sky decoder
│   │   ├── blocks/                 # attention, embeddings, DPT, camera encoding
│   │   ├── kelvin.py               # complete source model composition
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

## Support

For common errors and fixes (HF auth, driver / CUDA mismatch, OOM at
chunk-prep, `--max-chunks` truncation), see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Anything not listed there:
file a GitHub issue with the full traceback, `nvidia-smi`, and
`python --version`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt)
and individual file headers for details. Third-party attributions are
in [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{instantnurec2026,
  author       = {{NVIDIA}},
  title        = {Instant NuRec},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/NVIDIA/instant-nurec}}
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
