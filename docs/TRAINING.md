# Training the Kelvin model family

This guide covers reproducible Kelvin training in this standalone repository,
including NVIDIA ClipGT and the Waymo Open Dataset. The parity target is the
Kelvin PA-front Bazel contract audited on 2026-08-13. The public reference label
`kelvin-pa-front-2026-08-13` identifies that behavior snapshot without
publishing private repository provenance; it is not a claim that distributed
runs will be bitwise identical.

## What can be trained here

| Kelvin variant | Decoder | Standalone inference | Standalone training | Notes |
| --- | --- | --- | --- | --- |
| PA front | dense DPT, one Gaussian per context pixel | yes | yes, two phases | Primary supported training path. |
| PA multiview | dense DPT | yes | fixed camera/frame sets | The Bazel-only random 1/3/5-camera and 12/16/18-frame curriculum is not yet ported. |
| Point-query `noroad` / `road` | sparse cross-attention | yes | no | `KelvinInstantNuRec` deliberately rejects the point-query decoder; use the internal Bazel recipes for training. |
| TokenGS | token decoder | no | no | Still an internal model-development path in the audited reference. |

The runnable examples therefore cover the dense PA-front model. They retain the
same model, losses, optimizer, scheduler, camera calibration, and two-phase
structure as the Bazel PA-front recipes. The final section lists the remaining
standalone data/orchestration differences explicitly.

## Bazel contract being reproduced

The reference is the two-phase dense DAv3 PA-front recipe: context
supervision first, then differentiable novel-view rendering from the phase-one
checkpoint.

The standalone trainer preserves these training-step rules:

1. Run model/loss batch-start hooks, including the affine-gradient gate.
2. Zero all optimizer gradients.
3. Reconstruct context primitives and differentiable supervision tensors.
4. Add novel-view rendering only after `enable_render_global_step`.
5. Mean-reduce the per-item total across the batch and call manual backward.
6. Skip the optimizer only when every parameter gradient is `None`.
7. Set epoch/local-step progress and step the progress-based scheduler once.

Forward and backward/step are each wrapped in a distributed exception
broadcast. If one DDP rank fails, every rank exits instead of leaving peers
blocked at the backward synchronization point.

The shared numerical contract is:

| Setting | Value |
| --- | --- |
| epochs per phase | 40 |
| precision | BF16 mixed precision |
| optimizer | fused Adam when NVIDIA Apex is installed |
| learning rate / epsilon / betas | `1e-4` / `1e-15` / `(0.9, 0.99)` |
| warmup | 100 steps, starting at factor `0.01` |
| cosine floor | factor `0.0333` at progress `1.0` |
| front context | 18 frames at a fixed 0.5 s gap, one `camera_front_wide_120fov`, 784x448, batch 2/GPU |
| front supervision | 6 frames/camera, 1296x720 |
| context render gate | `1_000_000` (rendering stays disabled) |
| render gate | `0` |
| render initialization | phase-one checkpoint; encoder frozen; GS head reinitialized |
| render DDP | find-unused-parameters enabled |

The context loss weights are sky cubemap L1 `1.0`, context RGB L1 `0.1`,
distance L1 `1.0` over 0.1-300 m with a 0.98 quantile reduction, multiscale
distance-gradient L1 `1.0` over 0.1-50 m at strides 1/2/4/8, semantics CE `0.01`,
normal cosine `0.2`, and velocity L1 `1.0`. The render phase changes sky,
context RGB, and distance-gradient regularization to `0.01`, and adds rendered
RGB MSE `1.0`, LPIPS `0.2`, inverse-depth MSE `0.1`, and background MSE `2.0`.
Synthetic rendered RGB has weight `0.25`. Only missing normal,
rendered-distance, and velocity supervision is allowed; all other configured
labels are required.

### Kelvin-family scope

PA-front is the complete two-phase reference implemented here. PA-varying uses
a random camera/frame curriculum that is not yet exposed by the standalone data
schema. Point-query is a one-stage frozen-encoder path with sparse
cross-attention; TokenGS remains a model-development path without a public
production data/schedule contract. Use the support table above when reporting
which variants were actually trained in this repository.

### Camera calibration and affine color transform

There are two separate transformations and both are used during training:

- Geometry uses the original NCore intrinsics, distortion coefficients,
  `T_sensor_rig`, rig trajectory, exposure start/end poses, and shutter model.
  OpenCV pinhole, OpenCV fisheye, and F-theta cameras are supported directly;
  the F-theta path also preserves the optional bivariate windshield model.
  Rendering uses calibrated world rays and the original rolling/global shutter
  trajectory rather than substituting a pinhole approximation.
- Appearance uses Kelvin's learned per-camera 3x4 RGB affine matrix `[A | b]`.
  It is applied after foreground/sky alpha composition as
  `clamp(A @ rgb + b, 0, 1)`. The affine head is zero-initialized, so the
  initial transform is identity. Its decoded output is detached until global
  step 1000 when `model.post_processing.optimization_start_global_step: 1000`.

The affine is an ISP/color correction. It does not replace or modify geometric
camera intrinsics or trajectories.

The differentiable renderer follows the Kelvin 3DGUT contract: `RGB-d` output,
`RendererConfig_ParallelBatch`, eval3d enabled, and the unscented transform
`alpha=1`, `beta=2`, `kappa=0`, image-margin factor `0.1`, with every sigma
point required to be valid. The returned distance is opacity-weighted and is
normalized exactly once in the inverse-distance loss. CUDA sky composition
uses nvdiffrast cube-boundary filtering so gradients remain continuous across
cube-face seams.

## Environment

Use Python 3.11 and an NVIDIA GPU for real training. From the repository root:

The training extra includes nvdiffrast under NVIDIA's Source Code License
(1-Way Commercial), whose non-NVIDIA use is limited to non-commercial research
or evaluation. Read the complete terms in `THIRD_PARTY_LICENSE.txt` before
installing, and obtain the required release/legal approval for redistribution
or commercial use.

```bash
uv sync --frozen --extra training
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
instant-nurec-train --help
```

The base environment pins `nvidia-ncore==18.7.0`; the training extra pins
PyTorch Lightning 2.6.5, TorchMetrics 1.7.0, LPIPS 0.1.4, and the calibrated
`gsplat` renderer. `optimizer.implementation: auto`
uses `apex.optimizers.FusedAdam` when available and otherwise logs a warning and
uses `torch.optim.Adam`. Set `apex-fused-adam` to fail rather than fall back when
exact optimizer-kernel parity matters.

The first phase-two render JIT-compiles the pinned CUDA kernels for the active
PyTorch/CUDA/GPU target. Instant NuRec limits the fallback build to calibrated
3DGUT and its RGB/RGB-d channel counts; explicit `BUILD_3DGUT` and
`NUM_CHANNELS` environment values still take precedence. Ensure a compatible
CUDA toolkit/compiler is visible, allow several minutes for the first step,
and keep the extension cache between workers on the same software image.

Download the exact public DAv3 Base initialization used by the pinned recipe.
The revision and SHA256 below make the input reproducible:

```bash
mkdir -p checkpoints/dav3
hf download depth-anything/DA3-BASE model.safetensors \
  --revision f4a6c9b3c95e41c82048423d3493a81ec3fa810e \
  --local-dir checkpoints/dav3
echo 'e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5  checkpoints/dav3/model.safetensors' \
  | sha256sum --check
```

Set `model.init_weights_paths.dav3` to that file in the context-phase
configuration. The standalone converter was validated against this exact
541,518,028-byte checkpoint.

## NCore V4 input contract

Each dataset split is a list of absolute NCore V4 sequence-metadata `.json`
files. Every JSON must resolve all of its component-store paths. A training
sample requires:

- dynamic `rig -> world` poses;
- the configured camera and its frames, calibration, and `T_sensor_rig`;
- intrinsics and masks component groups named `default`;
- a cuboids group named `default` (it may be empty, but the current loader
  expects the component to exist).

RGB and any named `ego` mask are always consumed. The configured production
profile strictly requires RGB, metric distance, semantic flags, and sky
supervision; a missing required signal raises an error. Only normal,
render-distance, and velocity supervision are allowed to be unavailable, and
each omitted loss is reported. A plain Waymo conversion has RGB, poses,
calibration, lidar, and cuboids (the v18.7 converter writes an empty
camera-mask group), but not the derived dense depth/semantic/normal auxiliary
labels. It therefore needs the explicit RGB-only profile below.

The loader discovers optional stores adjacent to each NCore data shard using
`<data-shard>.aux.<signal>.zarr[.itar]` (legacy `-annotations` archives are also
accepted). It recognizes the `semantic_segmentation`, `depth`, and `egomask`
groups. Keep these sidecars beside the base stores and enable them explicitly:

```yaml
aux_data:
  enabled: true
  enabled_context: true
  semantic_segmentation: true
  depth: true
  egomask: true
```

Depth is loaded as metric distance; context normals are derived from that
distance and calibrated rays. Semantic class names are mapped to Kelvin's sky,
road, vehicle, ego, and validity flags. Cuboids supply motion supervision.

### Deterministic train/validation manifests

Generate manifests only after conversion has completed. When the source does
not provide an official split, make a deterministic whole-sequence split.
NCore 18.7.0's Waymo converter does not emit manifests:

```bash
mkdir -p "$NCORE_ROOT/manifests"
find "$NCORE_ROOT/all" -type f -name '*.json' -print | sort \
  > "$NCORE_ROOT/manifests/all.lst"
awk 'NR % 10 == 0' "$NCORE_ROOT/manifests/all.lst" \
  > "$NCORE_ROOT/manifests/val.lst"
awk 'NR % 10 != 0' "$NCORE_ROOT/manifests/all.lst" \
  > "$NCORE_ROOT/manifests/train.lst"
```

For a small dataset, split by whole sequence, never by frame, to prevent
trajectory leakage. Inspect the result before launching:

```bash
wc -l "$NCORE_ROOT/manifests/"{all,train,val}.lst
comm -12 \
  <(sort "$NCORE_ROOT/manifests/train.lst") \
  <(sort "$NCORE_ROOT/manifests/val.lst")
```

`comm` must print nothing. Point each split at its manifest with
`dataset.{train,val}.ncore_json_list_path`. Entries may be absolute, or relative
to `ncore_json_base_path` when that optional base is set. Explicit
`ncore_json_paths` arrays remain available for short, programmatically generated
runs and take precedence when both forms are present. All paths in the
committed examples are placeholders and must be replaced with local absolute
paths.

### Exact PA-front mixture and sampling

The checked-in configs reproduce the pinned mixture weights:

| Phase/split | source A | source B |
| --- | ---: | ---: |
| context train | 0.1 | 1.0 |
| context val | 0.1 | 0.1 |
| render train | 0.1 | 0.1 |
| render val | 0.1 | 0.1 |

Every sequence yields ten indexed samples. Each item uses 18 uniformly spaced
context frames at a 500,000 us gap and six random supervision frames. Dataset
and mixture resampling use a NumPy generator seeded from the first eight bytes
of `SHA256("{rng_epoch}_{item_index}_{global_seed}")`, interpreted as one
big-endian integer. `global_seed` is the top-level `seed` in the resolved
training config; validation keeps
`rng_epoch=-1`. Dataloaders are recreated each epoch so worker processes see
the new training epoch.

For source B, supervision also samples the sequence-relative
`nurec/all.zarr.itar` camera at ratio 0.3. Those frames carry
`RayFlags.SYNTHETIC`, so rendered RGB receives weight 0.25. They use
`unique_sensor_idx: 0`, intentionally sharing the context camera's learned
affine matrix instead of creating another color transform.

## NVIDIA ClipGT

Point the example manifests at an authorized local NCore V4 ClipGT copy. A
single clip may be reused across train and validation for an optimizer smoke
test only; it is not a quality evaluation. Production training needs the
adjacent metric-depth, semantic, and ego-mask stores described above.

The data path was checked at two levels without publishing private locations or
identifiers:

- A small CPU fixture produced the expected context and supervision RGB
  tensors, tracks, and poses with stable internal checksums. This fixture is
  RGB-only.
- A production V3/V4 differential matched all 101 rig poses/timestamps, all six
  camera models, every full 4x4 `T_sensor_rig`, exposure timestamps, masks, and
  sampled raw pixels. The compared V4 copy was camera-only, so this validates
  trajectory/calibration conversion, not the full production loss profile.
- The 1280x720 to 504x280 preprocessing matched the pixel-center affine
  `[[0.39375, 0, -0.303125], [0, 0.39375, -2.303125], [0, 0, 1]]`: resize to
  504x284, then crop two rows at top and bottom. Both paths use the first
  context camera at end exposure as `T_world_ref` while retaining original
  start/end exposure timing and `T_sensor_rig` in the aligned scene frame.

The internally audited source-A manifests contained a small train/validation
overlap, so their validation metrics must be treated as potentially leaked
until the duplicate paths are assigned to one split. Source B was disjoint.
Always rerun the `comm` check after changing a manifest; do not publish private
manifest paths or identifiers in experiment reports.

## Waymo Open Dataset v1.4.3 to NCore V4

Waymo data remains subject to the [Waymo Open Dataset Terms](https://waymo.com/open/terms/).
Accept them and choose the Perception v1.4.3 release on the official
[download page](https://waymo.com/open/download/). The files are hosted in the
[v1.4.3 Cloud Storage bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3).

### 1. Download TFRecords

Install and authenticate the Google Cloud CLI, then list the training objects.
Download one or a few segments first; the full release is large:

```bash
export RAW_ROOT=/absolute/path/to/waymo-v1.4.3
mkdir -p "$RAW_ROOT/train-one"

gcloud storage ls \
  'gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord' \
  > "$RAW_ROOT/training-objects.txt"

# Pick a specific object from the list so the trial is reproducible.
WAYMO_OBJECT="$(sed -n '1p' "$RAW_ROOT/training-objects.txt")"
gcloud storage cp "$WAYMO_OBJECT" "$RAW_ROOT/train-one/"
```

Keep the original TFRecords immutable and record the object names used for each
split. Do not put segments from the same source sequence in both train and val.
After validating the one-segment path, the corresponding full downloads are:

```bash
mkdir -p "$RAW_ROOT/training" "$RAW_ROOT/validation"
gcloud storage cp \
  'gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord' \
  "$RAW_ROOT/training/"
gcloud storage cp \
  'gs://waymo_open_dataset_v_1_4_3/individual_files/validation/*.tfrecord' \
  "$RAW_ROOT/validation/"
```

These transfers are large. Preserve Waymo's official training/validation split
rather than repartitioning the combined objects.

### 2. Build the official NVIDIA converter

The supported conversion path is the
[NCore Waymo converter](https://nvidia.github.io/ncore/conversions/waymo/waymo.html),
not a hand-written TensorFlow parser. Pin NCore so its schema and command-line
contract cannot drift:

```bash
git clone --branch v18.7.0 --depth 1 https://github.com/NVIDIA/ncore.git ncore-18.7.0
cd ncore-18.7.0
bazel version
```

The converter implementation and flags are documented in the official pinned
[converter source](https://github.com/NVIDIA/ncore/blob/v18.7.0/tools/data_converter/waymo/converter.py)
and [README](https://github.com/NVIDIA/ncore/blob/v18.7.0/tools/data_converter/waymo/README.md).

### 3. Convert to split NCore V4 stores

```bash
export RAW_TRAIN_ONE="$RAW_ROOT/train-one"
export NCORE_ROOT=/absolute/path/to/waymo-ncore-v4
mkdir -p "$NCORE_ROOT/train"

bazel run //tools/data_converter/waymo -- \
  --root-dir "$RAW_TRAIN_ONE" \
  --output-dir "$NCORE_ROOT/train" \
  --camera-id camera_front_50fov \
  --lidar-id lidar_top \
  waymo-v4 --profile separate-sensors
```

If a local GPU build fails during conversion (the audited RTX 5090 run failed
with invalid PTX), force the same v18.7.0 converter onto CPU:

```bash
CUDA_VISIBLE_DEVICES='' bazel run //tools/data_converter/waymo -- \
  --root-dir "$RAW_TRAIN_ONE" \
  --output-dir "$NCORE_ROOT/train" \
  --camera-id camera_front_50fov \
  --lidar-id lidar_top \
  waymo-v4 --profile separate-sensors
```

The CPU path is slower but produces the same NCore layout. The conversion
smoke test for this guide used an already available Waymo v1.4.2 TFRecord; the
download instructions target v1.4.3, but that v1.4.3 download was not exercised
here.

NCore 18.7.0 has no `--duration-sec` option. To make a bounded conversion,
place only the selected TFRecords directly under `RAW_TRAIN_ONE`; its TFRecord
glob is non-recursive. The output uses the
OpenCV pinhole camera model `camera_front_50fov` and lidar `lidar_top`; the
standalone trainer supports that pinhole calibration directly.

Once the bounded conversion passes, run the same command with
`--root-dir "$RAW_ROOT/training" --output-dir "$NCORE_ROOT/train"`, and again
with `--root-dir "$RAW_ROOT/validation" --output-dir "$NCORE_ROOT/val"`.
Generate manifests without mixing the official splits:

```bash
mkdir -p "$NCORE_ROOT/manifests"
find "$NCORE_ROOT/train" -type f -name '*.json' -print | sort \
  > "$NCORE_ROOT/manifests/train.lst"
find "$NCORE_ROOT/val" -type f -name '*.json' -print | sort \
  > "$NCORE_ROOT/manifests/val.lst"
```

Replace the example mixture with a direct Waymo train/val dataset. In both
splits, select `EXTERNAL` cuboids because the v18.7.0 converter writes every
Waymo box with that source. `AUTOLABEL` and `GT_ANNOTATION` select no tracks.
The converter emits no Kelvin auxiliary sidecars, so disable them explicitly:

```yaml
dataset:
  train: &waymo
    ncore_json_list_path: /absolute/path/to/manifests/train.lst
    camera_subsampler: {frame_width: 784, frame_height: 448}
    context_camera_ids: [camera_front_50fov]
    supervision_camera_ids: [camera_front_50fov]
    frame_batch_sampler:
      name: uniform
      n_frames_per_sample: 18
      n_samples_per_sequence: 10
      frame_gap_timestamp_us: 500000
    supervision_frame_batch:
      n_frames_per_camera: 6
      prepend_timestamps_us: 100000
      append_timestamps_us: 100000
      sample_strategy: random
      camera_subsampler: {frame_width: 1296, frame_height: 720}
      include_context_frames: false
    cuboid_tracks_params:
      track_label_source: EXTERNAL
    aux_data:
      enabled: false
      enabled_context: false
      semantic_segmentation: false
      depth: false
      egomask: false
  val:
    <<: *waymo
    ncore_json_list_path: /absolute/path/to/manifests/val.lst
```

The converter stores the original pinhole intrinsics, rational radial,
tangential and thin-prism distortion, camera extrinsics, and poses. Do not
rewrite these values to imitate the ClipGT F-theta camera. This repository also
accepts lowercase `cyclist` as dynamic. The pinned Bazel class list recognized
only uppercase `CYCLIST`, while the official Waymo converter emits lowercase;
the added spelling is an intentional upstream-integration correctness fix.

The exercised converted sequence contained 18,410 EXTERNAL cuboid
observations across 210 tracks. A reduced dense-Kelvin 70x70, two-frame CUDA
smoke run completed one context optimization step, resumed to the next global
step, and initialized/completed one render optimization step with the RGB-only
profile. It retained the full
OpenCV pinhole distortion and rolling-shutter calibration; render training also
created nonzero optimizer state for the affine camera transform. Apex was not
installed, so this smoke used the documented PyTorch Adam fallback. It verifies
the conversion/loader/optimizer/checkpoint/render path, not production quality
or the unexercised v1.4.3 download.

## Configure a run

Start from:

- `configs/training/kelvin_pa_front_context.yaml`
- `configs/training/kelvin_pa_front_render.yaml`

Replace the manifest, output, and initialization path placeholders in both
files. For Waymo, replace the complete mixture with the direct dataset block
above and use the RGB-only losses below; do not retain the ClipGT external
`nurec/all.zarr.itar` supervision entry.
Validate the resolved config without allocating a GPU:

```bash
python - <<'PY'
from pathlib import Path
from instant_nurec.training.run import load_training_config

for name in (
    "configs/training/kelvin_pa_front_context.yaml",
    "configs/training/kelvin_pa_front_render.yaml",
):
    cfg = load_training_config(Path(name))
    print(name, cfg.phase, cfg.system.max_epochs, cfg.bazel_reference)
PY
```

### Phase 1: context supervision

Phase 1 trains the encoder, DPT context geometry/RGB/semantic/motion heads, and
sky decoder. Rendering is disabled by the one-million-step gate, so neither the
Gaussian head nor affine output is consumed; both are intentionally unused in
this phase, and phase 2 reinitializes the Gaussian head. Independently sampled
supervision frames are still used to build the observed sky-cubemap target.

```bash
instant-nurec-train \
  --config configs/training/kelvin_pa_front_context.yaml
```

For production parity, initialize the DAv3 encoder from the exact base
safetensors used by the Bazel recipe through `model.init_weights_paths.dav3`.
The loader converts the official DAv3 keys into Kelvin's encoder, DPT
reassembly, and depth head; it zero-initializes the context and Gaussian output
heads. Sky starts from its model initialization and the affine linear layer
starts at identity. Do not silently train a production run from random weights.
A random initialization is useful only for testing forward, backward,
optimizer, checkpoint, and resume mechanics.

### Phase 2: differentiable render supervision

Set `model.init_weights_paths.full` to phase 1's `last.ckpt`. Phase 2 resets the
Gaussian head, resets the affine linear layer to identity, freezes the encoder,
renders every supervision camera with its original calibration, and enables
the render losses from step zero.

```bash
instant-nurec-train \
  --config configs/training/kelvin_pa_front_render.yaml
```

Outputs are written under `<out_dir>/<run_id>/`:

```text
<run>/
├── resolved.yaml
├── checkpoints/
│   ├── last.ckpt
│   └── epoch=...-step=....ckpt
└── logs/version_0/metrics.csv
```

Every checkpoint stores `kelvin_training_contract` with the audited reference label,
phase, selected optimizer implementation, and world size.

### A bounded end-to-end smoke test

Before a long run, copy each production config and change only:

```yaml
system:
  max_epochs: 1
  train_batch_size: 1
  val_batch_size: 1
  train_num_workers: 0
  val_num_workers: 0
  devices: 1
  num_nodes: 1
  limit_train_batches: 1
  limit_val_batches: 1
  save_every_n_train_steps: 1
```

For a lower-memory plumbing test, also use four context frames and a 196x112
context crop plus one 196x112 supervision frame. That deliberately changes the
model/data contract and is not a quality or Bazel-parity run. A successful
smoke test must complete backward and optimizer step, write `last.ckpt`, then
resume that checkpoint for at least one more step.

## Distributed training

The official front recipes used four nodes; varying-camera PA used eight. The
batch sizes in the configs are per process/GPU. On one machine, set
`devices` to the GPU count and use `strategy:
ddp_find_unused_parameters_true` for both phases. Context leaves the Gaussian
head and affine output unused while rendering is gated off; render can also
have conditional branches. The launcher automatically upgrades `auto`, `ddp`,
or explicit find-unused-false to this safe strategy whenever the requested
world size is distributed. Lightning starts the local workers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  instant-nurec-train --config /path/to/four-gpu-config.yaml
```

For multi-node work, set the same `num_nodes`, `devices`, config, code commit,
and visible dataset paths on every node, then launch through the site's Slurm
or torchrun integration. Do not wrap a Lightning self-spawning `devices: 8`
run in a second eight-process launcher. Confirm from startup logs that world
size equals `num_nodes * devices`.

## Checkpoint initialization and resume

These are different operations:

- `model.init_weights_paths.dav3` initializes phase 1 from the official DAv3
  safetensors; `model.init_weights_paths.full` initializes a new run from a
  complete Kelvin/Lightning state dict. Use `full` for phase 2. Optimizer,
  scheduler, epoch, and global step start fresh.
- `resume_from_checkpoint` restores a Lightning training checkpoint, including
  optimizer, scheduler, epoch, and global step. Use it only to continue the
  same phase/config.

To resume, copy the original resolved YAML, add:

```yaml
resume_from_checkpoint: /absolute/path/to/run/checkpoints/last.ckpt
```

Keep `phase`, model topology, data ordering, world size, optimizer, and schedule
unchanged. Inspect the stored contract before resuming:

```bash
python - /path/to/last.ckpt <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(checkpoint["kelvin_training_contract"])
print("epoch", checkpoint["epoch"], "global_step", checkpoint["global_step"])
PY
```

## No-auxiliary versus production-auxiliary training

The production loss profile is strict: RGB, context distance, semantics, sky,
render RGB/LPIPS, and render-background inputs must be present when their
weights are nonzero. Only normal, render-distance, and velocity may be reported
as unavailable and omitted. This prevents an apparently successful RGB-only
run from silently claiming full supervision parity.

| Data | Active supervision | Intended use |
| --- | --- | --- |
| official Waymo-to-NCore conversion | RGB; calibration/trajectory; lidar/EXTERNAL cuboids | ingestion, camera-model, render, optimizer, checkpoint and resume validation with the explicit profile below |
| RGB-only ClipGT fixture | RGB; masks/cuboids present in the base stores | loader and data-integrity smoke tests only |
| production NCore plus adjacent derived aux | RGB, metric distance, semantic flags, derived normals, motion/cuboids | full Bazel loss profile and model-quality training |

For a Waymo plumbing run, use this context-phase loss block:

```yaml
loss:
  primitive_sky_cubemap: 0.0
  primitive_rgb: 0.1
  primitive_distance: 0.0
  primitive_distance_gradient: 0.0
  primitive_semantics: 0.0
  primitive_normal: 0.0
  primitive_velocity: 0.0
  rgb: 0.0
  lpips: 0.0
  distance: 0.0
  background: 0.0
```

In render phase keep context RGB plus image reconstruction, but leave all
derived-label losses disabled:

```yaml
loss:
  primitive_sky_cubemap: 0.0
  primitive_rgb: 0.01
  primitive_distance: 0.0
  primitive_distance_gradient: 0.0
  primitive_semantics: 0.0
  primitive_normal: 0.0
  primitive_velocity: 0.0
  rgb: 1.0
  lpips: 0.2
  distance: 0.0
  background: 0.0
```

Do not interpret an RGB-only run as a reproduction of the released model's
quality. In particular, lidar points alone are not automatically rasterized to
`CameraFrameLabels.metric_distance`, and Waymo's class labels are not
automatically converted into Kelvin per-pixel semantic flags. Those dense aux
labels must be generated and attached in the same coordinate frame and mask
conventions as the cameras.

## Validation checklist

Use this order; stop at the first failure:

1. Parse both YAML files and confirm the audited reference label.
2. Open every NCore JSON and all referenced stores.
3. Decode the configured camera, validate monotonic timestamps, finite
   intrinsics/extrinsics, and overlapping rig-pose coverage.
4. Materialize one batch and confirm context/supervision image counts and
   resolutions.
5. Run a CPU loss/optimizer unit test.
6. Run one true CUDA context optimization step and save a checkpoint.
7. Resume it and confirm the global step advances.
8. Initialize render phase from that checkpoint and run one calibrated CUDA
   render/backward step.
9. Run one short validation epoch and inspect `metrics.csv` for finite loss.
10. Only then remove batch limits and launch the 40-epoch phases.

Useful gates from the repository root are:

```bash
pytest -q
ruff check instant_nurec tests
```

The standalone validation loop currently reports aggregate loss components; it
does not yet reproduce the internal PSNR/depth metric dashboard. Compare model
quality only after evaluating the same checkpoint, cameras, frames, masks, and
metric implementation on both systems.

## Remaining differences from the Bazel system

No known unimplemented training-step, configured-loss, calibrated-renderer, or
mixture-sampling blocker remains for the stock dense DAv3 PA-front two-phase
recipe. Keep these bounded differences visible in experiment reports:

- Random varying-camera/frame curricula, point-query training, and TokenGS
  training are not exposed. Legacy TokenGS checkpoint-key conversion is also
  absent; it is not used by the DAv3 context-to-render path.
- Cuboid loading ranges tracks from context frames only and does not implement
  the negative full-clip sentinel. With the stock 1 s extrapolation and
  supervision window of +/-0.1 s, this does not truncate the PA-front recipe.
- Apex FusedAdam is optional, and the calibrated public gsplat/nvdiffrast pins
  are not the private dependency commits from the Bazel workspace. The PyTorch
  Adam fallback and public CUDA kernels are not bitwise substitutes.
- CUDA sky filtering follows the nvdiffrast cube contract; the CPU-only utility
  fallback samples faces independently and is not seam-equivalent. All-invalid
  RGB/background masks return zero/skip rather than the Bazel path's NaN.
- Lowercase Waymo `cyclist` is deliberately treated as dynamic, correcting the
  converter/class-list mismatch present across the pinned repositories.
- Remote-cache and media/dashboard integrations are not reproduced. The
  standalone loop reports loss components, not the full PSNR/depth dashboard.

The real-data verification completed bounded context, resume, and render
optimization steps; it did not run both 40-epoch phases to convergence, did not
download Waymo v1.4.3, and did not establish model-quality equivalence. Full
production claims require disjoint manifests, complete auxiliary labels, the
same initialization weights/hardware/optimizer, and end-to-end validation of
the resulting checkpoints.

Record the standalone Git commit, `resolved.yaml`, NCore converter version,
Waymo object list or ClipGT revision, GPU count/type, optimizer implementation,
and emitted checkpoint contract with every result.
