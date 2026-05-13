# Plan: Extract NRM Predict-Only into Instant-nurec (aka nrm or storm) Standalone Repo

## Context

Instant-nurec (aka nrm or storm) is a system within the NRE monorepo (`/storage/projects/nre`) that reconstructs 3D Gaussian Splatting scenes from camera data. The goal is to extract the **predict-only** workflow into a standalone, minimalistic repository at `/storage/projects/instant-nurec`, following the structure of [NVIDIA/asset-harvester](https://github.com/NVIDIA/asset-harvester).

**Why:** NRM lives inside a large monorepo with Bazel builds, training infrastructure, and many unrelated systems. A standalone predict-only package makes NRM inference accessible without the full NRE stack — zero NRE dependency.

**Pipeline:** ncore file (cameras + images) -> KelvinNRM model -> 3D Gaussians -> PLY export
**Two merging modes:** No merge (per-chunk PLY) or frustum-ownership merge (single PLY)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Keep pytorch_lightning? | **No** | For single-ncore predict, PL is unnecessary overhead. Replace with ~50-line pure torch predict loop. Checkpoint loading is just `torch.load()` + `model.load_state_dict()`. |
| Keep Hydra config? | **No** | Predict only loads a pre-resolved `parsed.yaml`. Replace Hydra with `OmegaConf.load()` + `NRMConfig.model_validate()`. Eliminates hydra-core, click, click-default-group. |
| Model scope | **Kelvin only** | Requirements specify Kelvin. Celsius has separate 53K-line model and 35K-line primitive. Can add later. |
| libs.sensors / libs.geometry | **Rewrite in pure torch** | Only 3 functions used in predict path. Rewrite `se3pose_from_matrix()`, `image_points_to_world_rays_shutter_pose()`, and `generate_spinning_lidar_rays()` as pure PyTorch. Zero NRE dependency. |
| libs.losses | **Remove entirely** | Not optional — completely absent from the codebase. Predict doesn't need losses at all. |
| lietorch | **Remove** | BSD 3-Clause (Apache 2.0 compatible), but not used anywhere in the predict path. The only usages are in `geometry.py:se3_matrix_to_se3()` and `types.py:TracksData` — neither called during predict. Eliminate the dependency entirely. |
| Video rendering | **Skip** | Not part of core "ncore -> gaussians" pipeline. Eliminates nre.render.*, nrend dependency. |
| USDZ export | **Skip** | Not part of core "ncore -> gaussians" pipeline. Eliminates pxr/USD dependency. |
| PLY export for Kelvin | **Enable** | `export_ply.py` already has `export_kelvin_ply()`. The original `on_predict_batch_end` gates it behind `isinstance(CelsiusNRMPrimitive)` — we remove that gate. |
| DAv3 weights | **In checkpoint** | All encoder weights (including DAv3) are saved inside `last.ckpt` during training. No external weight files needed for predict-only inference. |
| ncore | **ncorev4 from Physical AI huggingface** | Use the ncorev4 format from Physical AI huggingface only. |
| NRE dependency | **None** | Fully standalone. No `pip install nre`, no submodules, no shared packages. Everything needed is vendored or rewritten. |

---

## NRM Predict Workflow (simplified)

The core predict pipeline, stripped to essentials:

```
1. Load config (parsed.yaml) + checkpoint (last.ckpt)
2. Load ncorev4 dataset (Physical AI huggingface format) -> iterate frames
   - NCoreNRMIndexableDataset reads ncorev4 data
   - Build NRMDataBatch with context images + camera calibrations
   - Compute rendering data (camera rays via pure-torch projection)
3. For each batch: KelvinNRM.reconstruct()
   - Encode context images (DAv3 encoder)
   - Decode to gaussian layers (static + dynamic + sky)
   - Build KelvinNRMPrimitive
4. Remove non-finite gaussians (one-liner: mask on torch.isfinite(densities))
5. Optional: merge_primitives() across chunks
   - Transform to reference frame
   - Frustum-ownership overlap strategy
   - Concatenate layers + voxelize
6. export_kelvin_ply() -> .ply files (static layer only)
```

**Static vs dynamic gaussians:** KelvinNRM decodes three conceptual layers: a **static layer** (fixed-position 3D gaussians with positions, rotations, scales, rgb, densities, semantic_class), **dynamic layers** (one per actor/object, with piecewise-linear keyframed positions — shape/color attributes are time-invariant, only position varies), and a **sky cubemap** (not gaussians — 6-face cubemap for background). During rendering, dynamic layers are interpolated to the frame timestamp and concatenated with static for the full render pass. However, `export_kelvin_ply()` explicitly exports the **static layer only** — dynamic layers and sky cubemap are not written to PLY. This matches the existing NRE behavior. Dynamic gaussian export can be added as a future extension if needed (e.g., exporting per-timestamp snapshots or keyframe data).

**Gaussian cleanup:** Only non-finite gaussians (NaN/Inf densities) are removed — a one-liner `mask = torch.isfinite(densities)`. No need for a separate `preprocess_for_export()` routine with configurable thresholds. Road/sky masks are derived from per-gaussian `semantic_class` during PLY export if present.

**Key source files in NRE (for reference during vendoring):**
- Entry: `nre/nrm/run.py` (320 lines)
- System: `nre/nrm/systems/gaussians_nrm.py` (668 lines, predict_step at line 504)
- Model: `nre/nrm/models/kelvin_model.py` (325 lines)
- Kelvin backbone: `nre/nrm/models/kelvin_backbone/` (~1375 lines total)
- Transformer blocks: `nre/nrm/models/blocks/` (~2065 lines total)
- Primitives: `nre/nrm/primitives/kelvin_primitive.py` (594 lines)
- PLY export: `nre/nrm/predict/export_ply.py` (210 lines)
- Merging: `nre/nrm/predict/primitive_merge.py` (736 lines)
- Dataset: `nre/nrm/datasets/nrm_ncore.py` (1112 lines)
- Batch: `nre/utils/batch.py` (2114 lines)

---

## libs.sensors / libs.geometry Replacement Strategy

Only 3 compiled CUDA kernel functions are used in the predict path. All will be rewritten as pure PyTorch:

| Original Function | Location | What It Does | Replacement Strategy |
|-------------------|----------|-------------|---------------------|
| `se3pose_from_matrix()` | `libs.geometry.kernels.pose` | Converts 4x4 SE3 matrices to (translation, quaternion) pairs | Pure torch: `translation = matrix[:, :3, 3]`, rotation via `matrix_to_quaternion()` using standard algorithms |
| `image_points_to_world_rays_shutter_pose()` | `libs.sensors.kernels.cameras` | Projects image pixels to world rays with rolling shutter interpolation | Pure torch: implement pinhole/fisheye/f-theta unprojection + distortion models. Most complex of the three. |
| `generate_spinning_lidar_rays()` | `libs.sensors.kernels.lidars` | Generates world-space rays for spinning lidar with shutter interpolation | Pure torch: implement spinning lidar geometry. Only needed if lidar data present in ncore. |

Supporting types that need vendoring/rewriting:
- `CameraModelConverter` / `LidarModelConverter` — ncore sensor model to kernel-compatible projection objects
- `DynamicPose` — start/end poses for rolling shutter interpolation
- `CameraProjection` / `LidarProjection` — parameter containers for the kernel functions

The pure-torch replacements lose GPU kernel optimization but:
1. Predict runs once per ncore, not in a training loop — performance is not critical
2. Eliminates the entire Slang/CUDA build dependency chain
3. Makes the package pip-installable on any system with PyTorch

**TDD approach for sensor replacements:** Each function is developed test-first — write tests that call the original CUDA kernel on real ncore data, capture reference outputs, then implement the pure-torch version and validate against those references within floating-point tolerance. This ensures correctness during the rewrite. See Step 2 below.

---

## Proposed Directory Structure

> Note: This structure will likely evolve during implementation as we discover what can be simplified further.

```
instant-nurec/
+-- instant_nurec/                    # Main Python package
|   +-- __init__.py                   # Version
|   +-- predict.py                    # Pure torch predict loop (~50 lines)
|   +-- config/                       # Simplified configs
|   |   +-- __init__.py
|   |   +-- nrm.py                    # NRMConfig + load_config()
|   |   +-- models.py                 # KelvinModelConfig
|   |   +-- predict.py                # PredictConfig, merge, PLY export
|   |   +-- dataset.py                # Dataset configs
|   +-- models/                       # Vendored model code
|   |   +-- __init__.py
|   |   +-- kelvin_model.py           # KelvinNRM (reconstruct only)
|   |   +-- activations.py            # GaussianActivations
|   |   +-- kelvin_backbone/          # Encoder/decoder/sky
|   |   +-- blocks/                   # Transformer blocks
|   +-- primitives/                   # Gaussian primitive definitions
|   |   +-- __init__.py
|   |   +-- base.py                   # BaseNRMPrimitive
|   |   +-- kelvin_primitive.py       # KelvinNRMPrimitive
|   +-- datasets/                     # Data loading
|   |   +-- __init__.py
|   |   +-- nrm_ncore.py             # ncore dataset loading
|   |   +-- batch.py                  # NRMDataBatch
|   +-- export/                       # Export functionality
|   |   +-- __init__.py
|   |   +-- export_ply.py            # PLY export (Kelvin)
|   |   +-- primitive_merge.py       # Frustum ownership merging
|   +-- sensors/                      # Pure-torch sensor projection (replaces libs.sensors/geometry)
|   |   +-- __init__.py
|   |   +-- cameras.py               # image_points_to_world_rays (pure torch)
|   |   +-- lidars.py                # spinning_lidar_rays (pure torch)
|   |   +-- pose.py                  # se3pose_from_matrix (pure torch)
|   +-- utils/                        # Minimal utilities
|       +-- __init__.py
|       +-- types.py                  # RigTrajectories, FrameMeta, CameraCalibration
|       +-- geometry.py               # SE3/quaternion math (pure torch, no lietorch)
|       +-- gaussians_utils.py        # write_ply_3dgs, RGB2SH
+-- tests/                            # One test per function
|   +-- conftest.py
|   +-- test_load_config.py
|   +-- test_load_ncore.py
|   +-- test_build_batch.py
|   +-- test_reconstruct.py
|   +-- test_export_ply.py
|   +-- test_merge_primitives.py
|   +-- test_camera_projection.py     # Pure-torch camera rays vs reference
|   +-- test_e2e.py                   # Full pipeline: ncore -> PLY
+-- run_inference.py                  # CLI entry point (argparse)
+-- setup.sh                          # Environment setup
+-- pyproject.toml                    # Dependencies (no NRE)
+-- README.md
+-- CONTRIBUTING.md                  # Based on asset-harvester, uses ruff linter
+-- LICENSE.txt
+-- .gitignore
```

---

## Implementation Steps

### Step 1: Scaffold + Build System
Create directory structure, `pyproject.toml`, `setup.sh`, `.gitignore`, `LICENSE.txt`, `CONTRIBUTING.md` (based on [asset-harvester CONTRIBUTING.MD](https://github.com/NVIDIA/asset-harvester/blob/main/CONTRIBUTING.MD), using ruff as linter).

**pyproject.toml dependencies:**
```
torch, omegaconf, pydantic, numpy, einops, scipy,
safetensors, torchvision, imageio, tqdm, nvidia-ncore,
point_cloud_utils, dataclasses_json, huggingface_hub
```

Note: No pytorch_lightning, no lietorch, no libs.*, no hydra-core, no click. CLI uses **argparse** (standard library), matching [asset-harvester's approach](https://github.com/NVIDIA/asset-harvester/blob/main/run_inference.py).

### Step 2: Pure-Torch Sensor Projection (TDD: `test_camera_projection.py` first)
The most novel piece — rewriting the 3 CUDA kernel functions in pure PyTorch:
- `sensors/pose.py`: `se3pose_from_matrix()` — matrix decomposition to translation + quaternion
- `sensors/cameras.py`: `image_points_to_world_rays()` — pinhole/fisheye/f-theta unprojection with rolling shutter
- `sensors/lidars.py`: `spinning_lidar_rays()` — spinning lidar ray generation

Validate against reference outputs from the original CUDA kernels on real ncore data.

### Step 3: Vendor Config Layer (TDD: `test_load_config.py` first)
- Write `load_config(yaml_path) -> NRMConfig` using `OmegaConf.load()` + pydantic validation
- Vendor only predict-relevant config fields: `model`, `dataset`, `predict`, `mode`
- Remove all training fields: losses, logger, checkpoint, profiling, optimizer, scheduler

### Step 4: Vendor Data Loading (TDD: `test_load_ncore.py`, `test_build_batch.py` first)
- Vendor `nrm_ncore.py` — ncore dataset reading
- Vendor/simplify `batch.py` — NRMDataBatch construction, replace `libs.sensors` calls with `sensors/` module
- Remove train/val/test dataloaders, keep only predict iteration

### Step 5: Vendor Model (TDD: `test_reconstruct.py` first)
- Vendor `kelvin_model.py` — keep only `reconstruct()` and `__init__`
- Vendor `kelvin_backbone/` (encoder, decoder, sky)
- Vendor `blocks/` (transformer blocks)
- Strip all training methods (`prepare_supervision`, `prepare_context`, loss computation)
- Checkpoint loading: `torch.load(ckpt_path)` + `model.load_state_dict(state_dict)` — all weights including DAv3 are in the checkpoint

### Step 6: Vendor Primitives + Export (TDD: `test_export_ply.py`, `test_merge_primitives.py` first)
- Vendor `kelvin_primitive.py` and `base.py`
- Vendor `export_ply.py` — remove Celsius path, enable Kelvin PLY export
- Vendor `primitive_merge.py` — frustum ownership merging

### Step 7: Write Predict Loop + Entry Point (TDD: `test_e2e.py` first)
- Write `predict.py` — ~50 line pure torch predict loop:
  ```python
  def predict(config, checkpoint_path, ncore_path, output_dir):
      model = KelvinNRM(config.model)
      state_dict = torch.load(checkpoint_path)["state_dict"]
      model.load_state_dict(state_dict)
      model.eval().cuda()

      dataset = NCoreDataset(config.dataset, ncore_path)
      with torch.no_grad():
          for batch in dataset:
              primitive = model.reconstruct(batch.cuda())
              # optional: preprocess_for_export, merge
          export_kelvin_ply(config.predict.ply_export, primitive, output_path)
  ```
- Write `run_inference.py` (argparse CLI)
- Write `setup.sh` environment setup

### Step 8: Polish
- Run all tests, verify numerical equivalence with original NRE pipeline
- Remove dead code
- Write `README.md`, `CLAUDE.md`

---

## Import Rewriting Strategy

All vendored files get imports rewritten. The key difference from the original plan: **no NRE imports remain at all**.

| Original Import | New Import |
|----------------|------------|
| `nre.nrm.config.*` | `instant_nurec.config.*` |
| `nre.nrm.models.*` | `instant_nurec.models.*` |
| `nre.nrm.primitives.*` | `instant_nurec.primitives.*` |
| `nre.nrm.predict.*` | `instant_nurec.export.*` |
| `nre.nrm.datasets.*` | `instant_nurec.datasets.*` |
| `nre.utils.batch` | `instant_nurec.datasets.batch` |
| `nre.utils.types` | `instant_nurec.utils.types` |
| `nre.utils.geometry` | `instant_nurec.utils.geometry` |
| `nre.models.gaussians.utils` | `instant_nurec.utils.gaussians_utils` |
| `libs.geometry.kernels.pose` | `instant_nurec.sensors.pose` |
| `libs.sensors.kernels.cameras` | `instant_nurec.sensors.cameras` |
| `libs.sensors.kernels.lidars` | `instant_nurec.sensors.lidars` |
| `nre.utils.sensors.*` | `instant_nurec.sensors.*` |
| `nre.render.*` | **REMOVED** |
| `nre.viewer.*` | **REMOVED** |
| `nre.utils.callbacks` | **REMOVED** |
| `nre.utils.log` | **REMOVED** |
| `libs.losses.*` | **REMOVED** |
| `lietorch` | **REMOVED** |
| `pytorch_lightning` | **REMOVED** |
| `ncore.*` | **KEPT** (pip dependency) |

---

## Testing Strategy

One test per function, written **before** implementation (TDD):

| Test File | What It Validates |
|-----------|-------------------|
| `test_load_config.py` | `load_config(parsed.yaml)` returns valid NRMConfig |
| `test_load_ncore.py` | ncore dataset reads frames with images + cameras |
| `test_build_batch.py` | NRMDataBatch construction from ncore frames |
| `test_camera_projection.py` | Pure-torch `image_points_to_world_rays()` matches CUDA kernel reference |
| `test_reconstruct.py` | `KelvinNRM.reconstruct(batch)` returns primitive with positions/rotations/scales/densities/rgb |
| `test_export_ply.py` | `export_kelvin_ply()` writes valid PLY file (includes non-finite gaussian removal) |
| `test_merge_primitives.py` | Frustum ownership merge produces single primitive from multi-chunk input |
| `test_e2e.py` | Full ncore -> PLY matches reference output numerically |

**Test data access via Hugging Face mock:**

Both the model artifacts and the ncorev4 dataset will eventually live on Hugging Face. Tests
use `huggingface_hub` API to download both, with mocks that resolve to local test data during
development:

```python
# conftest.py — mock HF hub to resolve to local test data
@pytest.fixture(autouse=True)
def mock_hf_hub(monkeypatch):
    """Mock huggingface_hub downloads to return local test data paths."""
    local_model_dir = Path.home() / ".cache/nrm/pretrained_models/kcd-fixgrad-maxd250_04-06_162523"
    local_ncore_dir = Path("/storage/data/nre/baseline_dataset_6cam")

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        if filename == "parsed.yaml":
            return str(local_model_dir / "parsed.yaml")
        elif filename == "last.ckpt":
            return str(local_model_dir / "last.ckpt")
        raise FileNotFoundError(f"Mock: unknown file {filename}")

    def fake_snapshot_download(repo_id, **kwargs):
        # ncorev4 dataset repo resolves to local ncore data directory
        return str(local_ncore_dir)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
```

Production code uses `huggingface_hub.hf_hub_download()` for model artifacts (config, checkpoint)
and `huggingface_hub.snapshot_download()` for the ncorev4 dataset. When both are published to
Hugging Face, the mocks are removed and tests download real artifacts.

**Local test data paths** (used by the mocks):
- `parsed.yaml`: `${HOME}/.cache/nrm/pretrained_models/kcd-fixgrad-maxd250_04-06_162523/parsed.yaml`
- `last.ckpt`: `${HOME}/.cache/nrm/pretrained_models/kcd-fixgrad-maxd250_04-06_162523/last.ckpt`
- ncorev4 dataset: `/storage/data/nre/baseline_dataset_6cam/` + `debug.lst`

---

## Estimated Line Counts

> These estimates will evolve as implementation reveals further simplification opportunities.

| Category | Lines |
|----------|-------|
| New code (predict.py, run_inference.py, sensors/, setup.sh, pyproject.toml) | ~500 |
| Vendored model code (kelvin_model, backbone, blocks) | ~3,700 |
| Vendored primitives + export + merge | ~1,900 |
| Vendored dataset + batch (predict path only) | ~2,000 |
| Vendored config (predict fields only) | ~400 |
| Vendored utils (types, geometry, gaussians_utils) | ~1,500 |
| Tests | ~500 |
| **Total** | **~10,500** |

---

## Verification Plan

1. **Unit tests pass:** Each vendored module has at least one test per function
2. **Numerical equivalence:** Run predict on same ncore data with both original NRE pipeline and instant-nurec, compare PLY output within floating-point tolerance
3. **CLI works:** `python run_inference.py --config parsed.yaml --checkpoint last.ckpt --ncore-path /data/clip.json --output-dir /tmp/out`
4. **Both merge modes:** Test with `--merge-strategy none` (multiple PLY) and `--merge-strategy frustum_ownership` (single PLY)
5. **Zero NRE dependency:** `pip install .` works in a clean environment with no NRE on the system

---

## Resolved Questions

1. **cosmos_predict1:** NOT needed for Kelvin predict. It's only used in `tokenizer.py` (COSMOS-1 Video Tokenizer), which is completely isolated from the Kelvin model. Kelvin processes raw image inputs directly through its encoder/decoder — no video tokenizer involved. The BUILD.bazel dependency on `@cosmos_predict1` is overspecified (covers all models, not just Kelvin). Eliminated from dependencies.
2. **License:** Apache License 2.0 — same as [NVIDIA/asset-harvester](https://github.com/NVIDIA/asset-harvester).

## Open Questions

1. **Camera distortion models:** The pure-torch camera projection rewrite needs to support OpenCV Pinhole, OpenCV Fisheye, and F-Theta models with optional Bivariate Windshield distortion. Verify which subset the Kelvin predict path actually exercises.
