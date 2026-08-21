<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# InstantNuRec: Feed-Forward 3D Gaussian Reconstruction from Driving Logs

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://research.nvidia.com/labs/sil/projects/instant-nurec/) [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2607.14203) [![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt) [![Model](https://img.shields.io/badge/HF-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/instant-nurec) [![Data](https://img.shields.io/badge/NCore-0d9488?logo=database&logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)

NVIDIA InstantNuRec is a feed-forward neural reconstruction model for autonomous vehicle simulation that turns a multi-camera driving log into a fully simulatable 3D Gaussian scene in a single forward pass. It emits a Gaussian primitive per pixel covering geometry, appearance, and motion, renderable in real time, and its output can initialize downstream Omniverse NuRec training for higher fidelity.

## Announcements

- **July 2026:** The [InstantNuRec project page](https://research.nvidia.com/labs/sil/projects/instant-nurec/) and [paper](https://arxiv.org/pdf/2607.14203) are now available.

## Abstract

3D simulation platforms are critical for autonomous driving because they enable end-to-end policy evaluation, thereby reducing development costs and improving safety. In recent years, neural simulation has become predominant, with methods such as NuRec playing a central role; however, these methods remain relatively slow and typically require per-scene tuning. In this work, we present Instant NuRec, a feed-forward neural reconstruction model that turns a multi-view driving log into a fully simulatable 3D Gaussian Splatting (3DGS) world in a single forward pass. The model accepts multi-view input from a calibrated camera rig and emits a layered output consisting of static and dynamic 3DGS layers, a sky cubemap, and per-camera ISP corrections, while providing native support for non-pinhole camera models via 3DGUT. It reconstructs a 10–20-second multi-camera scene in roughly 1.5 seconds and achieves a PSNR on the Waymo Open Dataset that is 2.01 dB above the strongest evaluated baseline. Instant NuRec is deeply integrated into NuRec and is compatible with AlpaSim for closed-loop simulation.

> **Repository scope:** This standalone CLI exports the static scene Gaussians
> to PLY together with an observation-derived sky cubemap sidecar and optional
> reference render. The abstract above describes the complete research model,
> including layers that are not part of this static export path.

![InstantNuRec demo](docs/demo.gif)

## Pipeline Overview

This repo goes from ncorev4 ingest → frame batch prep → forward pass
→ 3D-Gaussian PLY and sky-cubemap export. The PLY output is usable
directly as a static reconstruction, and can also serve as initialization
for downstream NuRec training to reach higher fidelity.

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

NCore V4 Sequence ─► Frame Batching ─► Eager PyTorch Model ─► 3D Gaussians + Sky ─► Export Bundle

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
tree from `uv.lock` into `.venv/`. In this default installation, the only
CUDA dependency is whatever the pinned `torch` wheel ships with.

The optional calibrated sky-composited renders use `gsplat`, which is not
installed by `setup.sh`. Install the render extra before using
`--render-preview` or `--render-video`:

```bash
uv sync --extra render
```

The first calibrated render JIT-compiles the pinned public CUDA kernels for
the active PyTorch/CUDA/GPU target and can take several minutes; subsequent
runs reuse the cache.

The PLY, sky sidecar, and cubemap quick-look image do not require this
extra. Full-video export also requires `ffmpeg` with the `libx264` encoder on
`PATH`.

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

For the default `pa-front` command above, success looks like one merged export
bundle whose PLY is at
`./demo_output/<run_id>/ply/pai_000da9de-.../pai_000da9de-....ply` —
~1.88 M Gaussians, kl-optimal voxelized from 2.87 M merged (3.18 M
pre-merge across 2 chunks) to land in `[0.9 * --n-gaussians,
--n-gaussians]` (default target 2 M). Omit `--merge` to write
per-chunk bundles instead (voxelization is bundled with merge and
runs only when the flag is set).

To run another model on the same clip, select its profile, such as
`--model pa-multiview` or `--model pq-front`.

##### Sky outputs and rendered preview

Every exported PLY has two sky files with the same stem. With `--merge`,
the stem is `<sequence_id>`; without it, each stem is
`<sequence_id>_chunk<N>`.

| output | contents |
| --- | --- |
| `<stem>.ply` | Static 3D Gaussian scene. |
| `<stem>.sky.npz` | World-aligned cubemap, visibility mask, per-camera affine matrices, face order, and format metadata. Keep this machine-readable sidecar next to the PLY. |
| `<stem>.sky.png` | 3×2 cubemap-face layout for a quick visual check; this is an atlas, not a camera render. |
| `<stem>.render.png` | First context frame with the cubemap alpha-composited behind the Gaussians and the camera affine correction applied. Written only with `--render-preview`. |
| `<stem>.render.mp4` | Every original frame from the first context camera, rendered at the profile resolution with its calibrated projection, exposure start/end trajectory, rolling shutter, sky, and camera affine. Written only with `--merge --render-video`. |

The `<stem>.sky.npz` sidecar uses format version 1 and can be loaded with
`numpy.load(path, allow_pickle=False)`. It contains exactly these keys, where
`S` is the cubemap face size (448 for the current released profiles) and `C`
is the number of distinct context-camera sensors:

| key | shape | NumPy dtype | meaning |
| --- | --- | --- | --- |
| `format_version` | scalar `()` | `int32` | The integer `1`. |
| `sky_cubemap` | `(6, S, S, 3)` | `float16` | World-aligned canonical RGB cubemap. Values are intentionally not clamped before the camera affine transform. |
| `sky_cubemap_mask` | `(6, S, S, 1)` | `float16` | Feathered observation-coverage weights in `[0, 1]`. |
| `affine_matrix` | `(C, 3, 4)` | `float32` | Per-sensor RGB affine transforms `[A | b]`. |
| `affine_sensor_indices` | `(C,)` | `int64` | `unique_sensor_idx` corresponding to each affine row. |
| `affine_sensor_ids` | `(C,)` | Unicode | NCore sensor ID corresponding to each affine row. |
| `face_order` | `(6,)` | Unicode `<U6` | `("right", "left", "top", "bottom", "front", "back")`. |
| `face_axes` | `(6,)` | Unicode `<U2` | `("+X", "-X", "-Y", "+Y", "+Z", "-Z")`. |
| `uv_convention` | scalar `()` | Unicode | `"u_left_to_right_v_top_to_bottom"`. |
| `coordinate_frame` | scalar `()` | Unicode `<U11` | `"ncore_world"`. |
| `sky_source` | scalar `()` | Unicode | `"observed_rgb_semantics"`, `"observed_rgb_semantics_plus_fallback"`, or `"synthetic_fallback"`. |
| `sky_observed_fraction` | scalar `()` | `float32` | Mean feathered observation-mask coverage. |

Cubemap directions use the `ncore_world` `(x, y, z)` axes. For a nonzero
direction, let `a = max(|x|, |y|, |z|)`; the dominant signed axis selects a
face, and `(u, v)` are `grid_sample` coordinates in `[-1, 1]` (`+u` moves
right across an image row and `+v` moves down across image rows):

| index / face | dominant axis | `u` | `v` |
| --- | --- | --- | --- |
| `0` / `right` | `+X` | `-z/a` | `y/a` |
| `1` / `left` | `-X` | `z/a` | `y/a` |
| `2` / `top` | `-Y` | `x/a` | `z/a` |
| `3` / `bottom` | `+Y` | `x/a` | `-z/a` |
| `4` / `front` | `+Z` | `x/a` | `y/a` |
| `5` / `back` | `-Z` | `-x/a` | `y/a` |

The `<stem>.sky.png` quick-look lays these faces out as
`[[left, front, right], [back, bottom, top]]`.

Affine row `i` corresponds to `affine_sensor_indices[i]` and
`affine_sensor_ids[i]`. Their order comes from the context rig's ordered camera
calibrations, which follows the configured profile or repeated `--camera-id`
order; the numeric sensor index is an identifier, not an array index.
For a composited RGB color `c`, apply row `[A | b]` as
`clamp(A @ c + b, 0, 1)`, after sky alpha compositing.

To also produce a calibrated still preview and full source-trajectory video:

```bash
uv sync --extra render
python run_inference.py \
    --model pa-front \
    --ncore-path /path/to/sequence.json \
    --output-dir /tmp/out \
    --merge \
    --render-preview \
    --render-video
```

Standard 3DGS PLY has no field for an environment cubemap or per-camera
ISP correction. Consequently, SuperSplat and `ply_viewer` display the
Gaussian foreground from `<stem>.ply` but do not automatically show the
sky from `<stem>.sky.npz`; a sidecar-aware renderer must composite it.

Both render outputs currently support NCore F-theta cameras and use their
calibrated per-pixel world rays, including the original projection and
interpolated exposure start/end rolling-shutter poses. `--render-preview`
writes the first context frame associated with each exported PLY.
`--render-video` reopens the source sequence and streams every original frame
from the first configured context camera without loading the full ray sequence
into memory. It requires `--merge`, because a complete trajectory should be
rendered against one complete merged scene. The video path currently accepts
one resolved NCore sequence per invocation and fails with the required
`--max-chunks` value if the configured cap would truncate its reconstruction.
These are source-trajectory checks, not arbitrary novel-view renders. Other
camera models and F-theta cameras with an external windshield-distortion model
are rejected explicitly rather than silently approximated.

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

Output bundles are written under
`out_dir/<run_id>/ply/<sequence_id>/<stem>.*`; the optional
`<stem>.render.png` appears only with `--render-preview`, and
`<stem>.render.mp4` appears only with `--merge --render-video`.

#### CLI reference

| flag | default | purpose |
| --- | --- | --- |
| `--model` | `pa-front` | Input/checkpoint profile: `pa-front`, `pa-multiview`, or `pq-front`. |
| `--ncore-path` | (required) | A `.json` file (single sequence) or a `.lst` manifest (one JSON path per line). |
| `--output-dir` | (required) | Directory the pipeline writes PLY and sky-output bundles into. |
| `--merge` | absent (false) | Boolean flag. When set, merges per-chunk primitives into a single frustum-ownership PLY per sequence (`<seq>.ply`) and runs kl-optimal voxelization (target count from `--n-gaussians`). Absent (default): per-chunk PLYs (`<seq>_chunk{N}.ply`), no voxelization. |
| `--n-gaussians` | `2000000` | Target number of static Gaussians after voxelization. Only consulted when `--merge` is set. The voxel size is searched iteratively via bracketed binary search to land the count in `[0.9 * target, target]`. |
| `--camera-id` | profile-dependent | Override a context camera. Repeat once per camera in canonical order. `pa-front` requires 1; `pa-multiview` supports 1, 3, or 5; `pq-front` is fixed to `camera_front_wide_120fov`. |
| `--max-chunks` | `8` | Maximum number of time-chunks processed per clip. One chunk spans up to 13.5 s, so the default covers 108 s. Longer clips are truncated and a `WARNING` logs the required value. |
| `--render-preview` | absent (false) | Write `<stem>.render.png` for the first context frame of each exported PLY, using the calibrated NCore F-theta projection, exposure trajectory, and sky. Requires `uv sync --extra render`. |
| `--render-video` | absent (false) | Write `<stem>.render.mp4` from every original frame of the first context camera, using calibrated NCore F-theta rays, rolling-shutter poses, sky, and camera affine. Requires one resolved sequence, `--merge`, enough `--max-chunks` for full coverage, `uv sync --extra render`, and `ffmpeg` with `libx264`. |
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
│   │   ├── static_core.py          # eager static + observed-sky reconstruction heads
│   │   ├── inference.py            # masking + primitive packaging
│   │   └── system.py               # predict-loop harness
│   ├── predict/                    # predict loop + PLY/sky export + preview + merge
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

<details>
<summary><b>Camera Adaptation</b></summary>

The released model is trained on NVIDIA's Autonomous Vehicle (AV)
platform data. While the model generalizes reasonably to cameras with similar
field of view and mounting geometry, **rectifying your source camera to match
the training intrinsics yields noticeably better reconstruction quality**.

If your camera has different intrinsics, we recommend performing a one-time camera adaptation (intrinsic
rectification) before running inference. Invalid regions introduced by the
remapping should be inpainted rather than filled with black to avoid
training/inference artifacts.

For the complete rectification workflow, target intrinsics extraction, and
script templates, see
[`docs/camera_rectification.md`](docs/camera_rectification.md).

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
