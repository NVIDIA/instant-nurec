# Plan 2: Finish the asset-harvester-style standalone (follow-up to plan.md)

> All conventions (parity-at-every-commit, NRE@`a54a6af` source-of-truth, autonomy, sandbox rules)
> from the prior plan still apply unless overridden below.

## Context

The previous plan (`plan.md`) targeted three crucial outcomes — torch-only CUDA, no bazel, asset-harvester-style flat layout — but they weren't actually delivered:

1. **slang/CUDA kernels still live.** Plan.md Phase 2 only swapped three named deps (`torch_scatter`, `nvdiffrast`, `gsplat`). The compiled libs/ kernels are still load-bearing in predict (concrete call sites + signatures collected in §"Kernel call sites" below):
   - `libs.geometry.kernels.pose.se3pose_from_matrix` (likely a duplicate of pure-torch `instant_nurec/_pkg/utils/geometry.py:se3_matrix_to_tquat`).
   - `libs.sensors.kernels.cameras.parameters.{CameraProjection, ExternalDistortion, NoExternalDistortion, OpenCVPinholeProjection, OpenCVFisheyeProjection, FThetaProjection, FThetaPolynomialType, ReferencePolynomial, ShutterType, BivariateWindshieldDistortion}` (all value containers, no math).
   - `libs.sensors.kernels.common.{Pose, DynamicPose}` (also value containers).
   - `libs.sensors.kernels.pose_calib.compute_poses_and_timestamps` (rolling-shutter pose interp; real math).
   - `libs.sensors.kernels.cameras.image_points_to_world_rays_shutter_pose` (camera model + shutter-aware ray gen; real math).
   - `libs.packed_ops.interface.packed_ops.{linstep_interleave, packed_searchsorted_indexed_vals}` (packed-array helpers; real math).
   - `libs.vren.interface.{ray_cuboidtracks_intersection, point_cuboidtracks_intersection_interpolate_pose, camera_rays_to_image_points}` (ray-cuboid intersection + projection; real math).

2. **Bazel still drives the build.** `MODULE.bazel`, `MODULE.bazel.lock`, `BUILD.bazel`, all `**/BUILD.bazel`, `bazel/`, `bazel-{bin,instant-nurec,out,testlogs}` symlinks, `.bazelrc{,.user}`, `.bazelversion`, `tools/`, `external/` symlink, `deps/` are still present. `pyproject.toml` exists but `pip install -e .` won't produce a runnable install because the libs/ kernels need bazel-built `.so` files.

3. **Layout doesn't match asset-harvester.** Current: `instant_nurec/{cli,config,_hf_mock}.py + _pkg/{nrm,utils,datasets,models,config}/`. Target (plan.md §8.2): flat `instant_nurec/{cli.py, model.py, primitives/, datasets/, predict/, utils/}`. The `_pkg/` namespace was Phase 1.4.6's stopgap and was supposed to be dissolved by Phase 3.8.2.

4. **Many tracked files don't belong** in an asset-harvester-style standalone. Comparing instant-nurec's ~50 top-level entries to asset-harvester's 15 (`asset_harvester/`, `benchmark/`, `data_samples/`, `docs/`, `scripts/`, `.gitignore`, `CONTRIBUTING.MD`, `LICENSE.txt`, `pyproject.toml`, `README.md`, `run_inference.py`, `run.sh`, `SECURITY.md`, `setup.sh`, `THIRD_PARTY_LICENSE.txt`), there are ~14 NRE-org files that should be removed and ~5 standard files that should be added.

This follow-up plan delivers all four outcomes, parity-gated at every step.

## Hard rules (carried over from plan.md, restated)

1. **Parity at every commit.** Run `benchmark/validate_parity.py` against `baselines/original_baseline` after each step; commit only on green. Phase 2-style CUDA→torch swaps may ratchet `tests/tolerance.json` upwards (per-property), never downwards. Re-document any tolerance bump in the commit body.
2. **NRE@`a54a6af` is source-of-truth.** Before each kernel replacement, read both the bazel-built kernel's slang/cu/cc source under `libs/.../*.slang|*.cu|*.cc` AND its NRE counterpart at the same path under `/storage/projects/nre/`. Self-invented impls only when NRE has no equivalent (commit message: `(self-invented: <reason>)`).
3. **TDD.** Write the equivalence test first, then the torch impl, then bazel run + parity. Full branch coverage on every new/modified function.
4. **One commit per substep.** Descriptive messages.
5. **Sandbox rule.** GPU-bound calls (bazel run, `python run_inference.py …` once Phase B lands) are unsandboxed; everything else is sandboxed.
6. **Communication.** Send a Slack update per commit, or hourly if idle (per plan.md "Communication" section).

---

## Phase A — Replace remaining slang/CUDA kernels with torch-native

Each substep: equivalence test → torch impl → equivalence test green → bazel run both modes → `validate_parity.py` → tolerance bump (if needed) → commit.

**Dependency analysis (revised after A.1 deferral):**

- **Independent** (can land in any order, no cross-coupling): A.4 (`packed_ops` in `tracks.py`), A.7 (`vren` in `tracks.py` + `cubemap.py`).
- **Tightly coupled** (must land in one bundle): A.2 (`cameras.parameters.*` dataclasses) + A.3 (`common.{Pose, DynamicPose}` dataclasses) + A.5 (`compute_poses_and_timestamps`) + A.6 (`image_points_to_world_rays_shutter_pose`). The bazel binding for A.6 does `isinstance(projection, OpenCVPinholeProjection)` etc. on the A.2 classes (`libs/sensors/kernels/cameras/bindings.py:71-90`), so swapping A.2/A.3 without A.5/A.6 breaks the kernel; conversely, A.5/A.6 need the new A.2/A.3 dataclasses to type-narrow the inputs in the torch impl.
- **Deferred** (until the coupled bundle lands): A.1 (`se3pose_from_matrix`). FP-precision drift between slang and torch on f32 GPU ops only matters while the still-slang ray-gen consumes the rotation; once A.6 is torch the whole pose → ray chain is FP-controlled in Python.

Ordering: A.4 → A.7 → bundle(A.2+A.3+A.5+A.6) → A.1 → A.8 → A.9.

### A.1 — `se3pose_from_matrix` → torch (DONE — `fc23075`)

- **Outcome:** f64-internal Shepperd's method in
  ``instant_nurec/_pkg/utils/geometry.py:se3pose_from_matrix``;
  ``batch.py`` switched to import the torch helper.
- Per-quaternion ULP drift (vs slang on GPU): 0-3 ULP, mostly 0-1.
- Bit-exact match impossible (different SASS sequences); per-vertex drift
  flips ~5-30 cull-boundary Gaussians. Resolved by adding a
  ``_vertex_count_delta=50`` band to ``benchmark/validate_parity.py``
  (per user direction "plan2 > CLAUDE.md").
- Tests: `tests/test_se3pose_torch.py` (12 branch-coverage tests, revived
  in `7a5d8dc`).

### A.2 + A.3 + A.6 bundle — `cameras.parameters` + `common.{Pose,DynamicPose}` + `image_points_to_world_rays_shutter_pose` → torch (DONE — `037ed34`)

- **Bundle rationale:** the slang ``image_points_to_world_rays_shutter_pose``
  binding does ``isinstance(projection, OpenCVPinholeProjection)`` etc.
  on the A.2 classes, so swapping any of A.2/A.3/A.6 alone breaks the
  binding. They had to land together.
- **A.2 + A.3:** ``instant_nurec/_pkg/utils/sensors/_kernel_types.py`` with
  ``OpenCVPinholeProjection`` / ``OpenCVFisheyeProjection`` / ``FThetaProjection``
  / ``BivariateWindshieldDistortion`` / ``Pose`` / ``DynamicPose`` /
  ``ShutterType`` / ``FThetaPolynomialType`` / ``ReferencePolynomial``.
  Same field names as the libs version (the converter still works);
  unpacked rather than packed-intrinsics-tensor (torch path reads fields
  directly).
- **A.6:** ``instant_nurec/_pkg/utils/sensors/_image_points_to_world_rays_torch.py``.
  FTheta + NoExternalDistortion only. Math taken from
  ncore@a54a6af's pure-python ``FThetaCameraModel.image_points_to_world_rays_shutter_pose``
  (ncore/impl/sensors/camera.py:1014-1112 + 1347-1377).
- **Drift on this bundle alone:** chunk0 +5, chunk1 -4 (better than A.1
  alone — torch rolling-shutter matches ncore's reference more closely
  than the slang kernel).

### A.4 — `linstep_interleave` + `packed_searchsorted_indexed_vals` → torch (DONE — `2b48686`)

- **Call sites:**
  - `instant_nurec/_pkg/utils/packed_ops.py:18` (import).
  - `instant_nurec/_pkg/datasets/tracks.py:259` (`linstep_interleave` for interleaved pack indexing in `Ops.subset_from_indices`).
  - `instant_nurec/_pkg/datasets/tracks.py:393` (`packed_searchsorted_indexed_vals` in `interpolate_tracks_poses`).
- **Replacement:**
  - `linstep_interleave(start, num_steps, step_size, return_idx)`: per-pack arange-style interleaved sequence. Pure torch via `torch.repeat_interleave` + `torch.arange` per pack, then offset by `start`.
  - `packed_searchsorted_indexed_vals(values, packinfo, queries, pack_idx)`: per-pack `torch.searchsorted` indexing using the packinfo `[start, count]` slabs.
- **Test:** `tests/test_packed_ops_torch.py` — empty/single/multi-pack inputs; contiguous and non-contiguous tensors; dtype variations (int32/int64/float32). Equivalence vs the bazel-built kernel within `1e-6` for floats / exact for ints.
- **Parity risk:** very low (pure index math).
- **Commits:** one per replacement.

### A.5 — `compute_poses_and_timestamps` → torch (DONE — `6b32da4`)

- **Outcome:** since the standalone always pins ``enable_calib=False`` and
  ``rect_points_lb=None``, the slang kernel reduces to per-sample
  ``T_in[frame_idx]`` + ``ts_in[frame_idx]`` indexing. The torch helper
  ``_compute_poses_and_timestamps_torch`` lives next to the call site in
  ``sensors.py``; no rolling-shutter spatial interpolation, no calibration
  delta, no SE3 round-trip.
- Drift attributable to A.5 alone: chunk1 -4 vs the kernel's -5, merge -30 vs -25.

### A.7 — `libs.vren.interface.*` → torch (DONE — `efa4cc9`)

- **Outcome:** three sub-kernels ported to pure torch:
  - ``ray_cuboidtracks_intersection`` + ``point_cuboidtracks_intersection_interpolate_pose``
    in ``instant_nurec/_pkg/datasets/_vren_torch.py``. Slab-method AABB
    intersection in cuboid local frame at the ray/point timestamp.
  - ``camera_rays_to_image_points`` (forward FTheta projection) added to
    ``_image_points_to_world_rays_torch.py``. Accepts ncore
    ``FThetaCameraModelParameters`` directly to keep the cubemap call site
    minimal.
- **Critical bug found during bring-up:** the slang ``ray_aabb_intersect``
  returns ``(-1, -1)`` on miss (``t1 > t2``), and the caller checks
  ``t1t2.y > 0``. My initial torch slab impl returned ``(t_near, t_far)``
  unfiltered; when ``t_far > 0`` but ``t_near > t_far`` (a true miss),
  this produced a false-positive hit. Downstream ``decoders.py:382``
  requires ``intersections_cnt == 1`` to mark a Gaussian as movable, so
  every false-positive multi-track hit dropped a movable Gaussian
  (-107k chunk0 regression). Fixed by clamping ``(t_near, t_far)`` to
  ``(-1, -1)`` on miss to match the kernel.

### A.8 — Drop libs/ subtree

After A.1–A.7, no remaining `from libs.X` imports anywhere. Verify with `grep -rn "from libs\." instant_nurec/ scripts/ tests/`.

- `git rm -rf libs/`.
- Re-pickle `kelvin_full.pt` because class shapes shifted (sensor model containers, ray-cuboid intersection result, …). Use the existing `INSTANT_NUREC_FULL_PT` save path.
- Full unit suite + bazel run + validate_parity sweep.
- **Commit:** `refactor(strip): drop libs/ now that all CUDA/slang kernels are torch-native (Phase A.8)`.

### A.9 — Re-strip pass (plan.md Phase 1 Step 4.0–4.8) against the post-A codebase

After each major phase, re-run the iterative aggressive strip from plan.md §"Step 4 — Iterative aggressive strip". This keeps the repo lean by deleting whatever just became dead. Sub-iterations are the same as plan.md (4.1 drop non-Kelvin → 4.2 drop training → 4.3 drop output-irrelevant → 4.4 drop yaml → 4.5 drop NRE/PL → 4.6 drop "nre" strings → 4.7 simplify pass → 4.8 convergence). Most of these have already been done; the re-strip looks for *new* dead code that the kernel replacements exposed:

- ncore.* call sites that only existed to feed the bazel kernels (e.g. `ncore.sensors.*` import paths now reachable only via dataclass conversion).
- BUILD-only deps and `# type: ignore` shims pointing at the deleted `libs.*`.
- pyproject deps that became unreachable.
- `tests/test_*.py` stubs for dropped kernels.

Iterate until convergence (4.8): try removing one more file/function/import; parity breaks → keep, parity holds → delete + commit. Stop when no more deletions are possible.

- **Commits:** one per deletion, `refactor(strip): drop <X> after Phase A torch-native swap (re-strip)`.

---

## Phase B — Drop bazel (DONE — `07c8b20`)

Phase A.8 removed ``libs/``. Phase B prep replaced the last two
non-pip-installable native deps with pure-torch shims:
* ``instant_nurec/utils/_se3_torch.py`` (``c144996``) — drop-in for
  ``lietorch.SE3`` / ``lietorch.SO3``.
* ``_RGBNormalize`` ``nn.Module`` + ``F.interpolate(antialias=True)``
  (``b33892f``) — replaces ``torchvision.transforms.Normalize`` and
  ``torchvision.transforms.functional.resize``.

Final removal landed in ``07c8b20`` after a fresh ``.venv-b`` venv was
brought up with ``pip install -e .`` + ``pip install torch
--index-url https://download.pytorch.org/whl/cu126``. ``python
run_inference.py`` runs end-to-end on GPU; both modes parity GREEN.

**Vertex-count drift via the public CUDA torch wheel:**
```
no_merge chunk0=1748398 (target 1748388, +10)
no_merge chunk1=1428224 (target 1428209, +15)
merge=2871691         (target 2871662, +29)
```
All within the 50-vertex tolerance band. Drift differs slightly from
the bazel-runtime numbers because public-wheel torch defaults
TF32/cuDNN differently than the bazel-internal build; the math is
identical.

**Bug fixed during bring-up:**
``utils/sensors/_image_points_to_world_rays_torch.py:_ncore_ftheta_to_projection_and_resolution``
is called from inside ``cubemap.unproject_to_sky_cubemap`` which is
``@torch.compile``-decorated. Numpy → torch conversions inside that
compiled frame tripped dynamo's ``___from_numpy`` guard machinery
(``DispatchKeySet`` mismatch). Fix: convert numpy arrays to plain
Python lists before ``torch.tensor(...)`` AND wrap the helper with
``@torch._dynamo.disable`` so dynamo never tries to trace the conversion.

### B.1 — Verify pip-install runs without bazel

Sandboxed dry-run:

```
python -m venv /tmp/pip_test_venv
source /tmp/pip_test_venv/bin/activate
pip install -e .
python -c "from instant_nurec.cli import main; print('ok')"
```

If the import fails, Phase A missed something — fix before proceeding.

### B.2 — Remove bazel infrastructure (one commit)

`git rm`:
- `MODULE.bazel`, `MODULE.bazel.lock`
- `BUILD.bazel` (root) and all `**/BUILD.bazel`
- `bazel/` (rules, conditions, version, typing subdirs)
- `bazel-bin`, `bazel-instant-nurec`, `bazel-out`, `bazel-testlogs` (symlinks)
- `.bazelrc`, `.bazelrc.user`, `.bazelversion`
- `tools/` (bazel-only)
- `external/` (symlink → `/storage/projects/nre/external`)
- `deps/` (vendored NRE third-party deps tree; bazel-only)

Update `instant_nurec_example_call.sh` to be the canonical reference invocation. (Note: `nre_example_call.sh` and `uv.lock` are dropped in Phase D.1.d, not here.)

- **Commit:** `chore(build): drop bazel; canonical invocation is python run_inference.py (Phase B.2)`.

### B.3 — Confirm runtime + parity

Run `python run_inference.py --merge none` and `--merge frustum-ownership` against the dataset; `validate_parity.py` for both. Green → done.

### B.4 — Re-strip pass (plan.md Phase 1 Step 4.0–4.8) against the post-B codebase

After bazel removal, re-run the strip iteration. Bazel removal usually exposes more dead paths (per plan.md §8.5 "bazel-removal usually exposes more dead paths e.g. BUILD.bazel-only deps"):

- Imports that only existed to satisfy bazel `py_library` declarations.
- `# type: ignore` comments that pointed at deleted bazel-built `.so` files.
- `pyproject.toml` deps that pyproject infers from but bazel was actually providing (e.g. transitive native deps).
- Empty `__init__.py` packages where the bazel target used to live.

Iterate to 4.8 convergence.

- **Commits:** one per deletion.

---

## Phase C — Flatten layout to asset-harvester shape (DONE)

Per plan.md §8.2. Basic flatten landed in `e177e4c`:

```
instant_nurec/_pkg/utils/         → instant_nurec/utils/
instant_nurec/_pkg/datasets/      → instant_nurec/datasets/
instant_nurec/_pkg/models/        → instant_nurec/models/
instant_nurec/_pkg/nrm/           → instant_nurec/nrm/
instant_nurec/_pkg/config/        → instant_nurec/config_schema/
                                    (renamed to avoid clash with config.py)
```

~70 .py files updated (`instant_nurec._pkg.X` → `instant_nurec.X`).
``//instant_nurec/_pkg/...`` BUILD labels collapsed to
``//instant_nurec/...``. Predict + tests + parity all GREEN.

The deferred nested merges (plan2 §C.1) landed across six commits
`4bd2c91` (config_schema), `22e8748` (primitives), `029878b`
(predict), `bd0c4c0` (datasets), `02bd658` (utils), `97d4ecb` (model).
Final layout matches the plan target:

```
instant_nurec/
  __init__.py
  cli.py
  config.py
  _hf_mock.py
  config_schema/   # was nrm/config/* + base_schema.py
  model/           # was nrm/systems/* + nrm/models/* (kelvin/blocks/backbone)
  primitives/      # was nrm/primitives/
  datasets/        # was top-level + nrm/datasets/ MERGED
  predict/         # was nrm/predict/* + nrm/run.py
  utils/           # was utils/ + models/ + nrm/utils/ MERGED
```

Re-pickled `kelvin_full.pt` end-to-end on the GPU; both modes parity
GREEN within the existing 50-vertex tolerance band:

```
no_merge chunk0=1748398 (+10), chunk1=1428224 (+15) — PASS
merge=2871691 (+29) — PASS
```

The HF mock wired in `aa707aa` is exercised by the second run: after
the first run writes `kelvin_full.pt` to `INSTANT_NUREC_FULL_PT`, the
mock seeds the canonical cache (`~/.cache/instant_nurec/kelvin_full.pt`)
and the second run loads through the HF surface.

### C.1 — Source moves

Target shape (matches asset-harvester's flat package style):

```
instant_nurec/
  __init__.py
  cli.py                 # current instant_nurec/cli.py
  config.py              # current instant_nurec/config.py (load_predict_config + _PREDICT_CONFIG)
  _hf_mock.py            # current instant_nurec/_hf_mock.py
  config_schema/         # was _pkg/nrm/config/* + _pkg/config/base_schema.py
  model.py               # was _pkg/nrm/models/kelvin_model.py + _pkg/nrm/systems/gaussians_nrm.py
  model/                 # if model.py grows — split into model/{__init__.py, blocks/*, backbone/*, post_processing.py, activations.py}
  primitives/            # was _pkg/nrm/primitives/*
  datasets/              # was _pkg/{nrm/datasets/*, datasets/*} merged
  predict/               # was _pkg/nrm/predict/* + _pkg/nrm/run.py
  utils/                 # was _pkg/{utils/*, models/*, nrm/utils/*} merged
```

Conflict: `instant_nurec/config.py` (loader) vs `instant_nurec/config_schema/` (pydantic schemas, was `_pkg/nrm/config/`). Resolution: keep `config.py` as the user-facing loader; nest the pydantic models under `config_schema/` (or `_config_schema/` if we want to mark them internal). asset-harvester uses a single flat `config.py`-style module — we deviate slightly here because the pydantic tree is genuinely 4-deep.

### C.2 — sed all imports

`instant_nurec._pkg.X.Y` → `instant_nurec.X.Y` (mapped per the moves above) across:
- All py files under `instant_nurec/` and `tests/`.
- String literals in `tests/` (`monkeypatch.setitem(sys.modules, "instant_nurec._pkg.X", …)`, `importlib.import_module("instant_nurec._pkg.X")`).
- BUILD.bazel labels — N/A; bazel is gone after Phase B.

### C.3 — Update tests

Test fixtures that drop cached modules need the new module paths. sed-friendly.

### C.4 — Re-pickle full model

Class qualnames change (e.g. `instant_nurec._pkg.nrm.systems.gaussians_nrm.GaussiansNRMSystem` → `instant_nurec.system.GaussiansNRMSystem`). The existing `kelvin_full.pt` becomes invalid. Re-run with `INSTANT_NUREC_FULL_PT` set to a fresh path; the construct-and-save fallback in `instant_nurec/utils/...:make()` (was `_pkg/nrm/systems/__init__.py`) writes the new pickle.

### C.5 — Full unit + parity sweep

Both modes; `validate_parity.py`; ruff clean.

- **Commits:** one per top-level move (utils, datasets, predict, primitives, model, config_schema), then a final `refactor(layout): flatten _pkg/ into asset-harvester-style top-level (Phase C)`.

### C.6 — Re-strip pass (plan.md Phase 1 Step 4.0–4.8) against the post-C codebase

Layout flatten typically exposes:

- Re-export `__init__.py` shims that only existed because of the `_pkg/` indirection.
- Now-trivial single-import modules that can be merged into a parent.
- Test fixtures that were keyed off the old import paths.
- Duplicated helpers between `_pkg/utils/` and `_pkg/nrm/utils/` that the merge in C.1 exposed.

Iterate to 4.8 convergence.

- **Commits:** one per deletion.

---

## Phase D — File hygiene (asset-harvester comparison)

asset-harvester's exact top-level (15 entries): `asset_harvester/`, `benchmark/`, `data_samples/`, `docs/`, `scripts/`, `.gitignore`, `CONTRIBUTING.MD`, `LICENSE.txt`, `pyproject.toml`, `README.md`, `run_inference.py`, `run.sh`, `SECURITY.md`, `setup.sh`, `THIRD_PARTY_LICENSE.txt`.

The end-state of instant-nurec must contain only files in this list (modulo `instant_nurec/` instead of `asset_harvester/`, plus `baselines/`, `tests/`, `instant_nurec_example_call.sh`, `CLAUDE.md`, `.git/`, `.gitignore`, `.gitattributes`, `.pre-commit-config.yaml` — these are the project-specific essentials we keep).

### D.1 — Drop NRE-org tooling

One commit per logical group; `git rm` each.

#### D.1.a — NRE LLM tooling (DONE — already gitignored, never tracked)
- `prompts.yaml` was added to `.gitignore` ahead of plan2 ("Prior-agent prompt history"); not in repo state.

#### D.1.b — NRE GitLab/code-review tooling (DONE — `d8577a7`)
- `.coderabbit.yaml` (NRE CodeRabbit config)
- `CODEOWNERS` (NRE GitLab codeowners)
- `sonar-project.properties` (NRE SonarQube)
- `.gitlab/` (subdirs `ci/` and `merge_request_templates/` — NRE GitLab artifacts)
- `.gitlab-ci.yml` (NRE GitLab CI; asset-harvester ships no committed CI — add a minimal CI later if needed)

#### D.1.c — NRE build/dev environment (DONE — `57d512c`)
- `.clang-format` (no C++ left after Phase A.8 / Phase B drops `libs/`)
- `.dockerignore` (NRE Docker workflow)
- `.gitmodules` (NRE submodules; in this checkout it appears as a container char-device pseudo-file rather than a tracked file — skipped)
- `NOTICES` (NRE-side legal notice; replaced by `THIRD_PARTY_LICENSE.txt` in D.4)

#### D.1.d — NRE-side reference + scratch (DONE — `bc07884`)
- `nre_example_call.sh` (NRE reference invocation; the standalone keeps `instant_nurec_example_call.sh`)
- `uv.lock` (uv-managed lockfile from the NRE port; pyproject + pip is the asset-harvester pattern). Also dropped the now-orphaned `[tool.uv]` block from `pyproject.toml`.

### D.2 — Drop tracked build artifacts (DONE — already covered)

None of `.coverage`, `.pytest_cache/`, `.ruff_cache/`, `.venv/` are tracked; all four are already in `.gitignore`. No-op.

### D.3 — Drop IDE/user state (DONE — already covered)

None of the listed paths are tracked. `.gitignore` already excludes `.idea/`, `.vscode/`, `.ripgreprc`, `.mcp.json`, `.bashrc`, `.bash_profile`, `.zshrc`, `.zprofile`, `.profile`, `.gitconfig`. No-op.

### D.4 — Add asset-harvester-style files (DONE — `d6c7977`)

Per user direction "use the same files as the asset harvester repo
has", the corp files were sourced verbatim from
https://github.com/NVIDIA/asset-harvester:
- `LICENSE.txt` — Apache License 2.0 (asset-harvester's). DONE.
- `THIRD_PARTY_LICENSE.txt` — third-party attributions, 745 lines. DONE.
- `SECURITY.md` — corp standard security-disclosure policy. DONE.
- `CONTRIBUTING.md` — short dev guide pointing at `setup.sh`, `pytest`, `ruff`, `validate_parity.py`. DONE — `dd8b5b3` (kept the project-specific NRM body rather than asset-harvester's generic one).
- `data_samples/` — placeholder directory + README. DONE (the actual ≤50 MB ncorev4 fixture is deferred until the corp publishes `nvidia/instant-nurec-kelvin`; the HF mock at `instant_nurec/_hf_mock.py:get_sample_data_path` is already wired up to look for `ncorev4_sample/` here).
- *(optional)* `benchmark/` — deliberately skipped. asset-harvester's benchmark is rendering-metrics-specific (DINOv3 / SAM 3D Body for Gaussian splat asset evaluation); the standalone's parity surface is already covered by `benchmark/derive_determinism_tolerance.py` + `benchmark/validate_parity.py`.

A follow-up audit (commit pending) aligned the SPDX file headers across all 110 .py/.sh files to match asset-harvester (Apache-2.0, copyright year 2026), and switched `pyproject.toml`'s `[project].license` to `Apache-2.0`.

### D.5 — Resolve `internal/` (DONE — `33a30e8`)

Plan.md §8.3 reserved `internal/` for migration scaffolding (NRE provenance markers, parity helpers — anything used to *reach* parity but not at runtime). asset-harvester has no `internal/`.
- Moved `internal/parity_proofs/` → `docs/internal/parity_proofs/`.
- `internal/` directory removed.
- Updated `instant_nurec/_pkg/nrm/utils/cubemap.py` comment reference to the new path.
- Verified `python run_inference.py` CLI imports still work; 759/759 tests pass.

### D.6 — Re-strip pass (plan.md Phase 1 Step 4.0–4.8) against the post-D codebase (PARTIAL — `3b0b484`)

After file hygiene, re-run the strip iteration one final time. Most of the deletions in D will have surfaced extra cleanups:

- Test fixtures referencing removed config files.
- README/docs sections referencing removed files.
- Stale `.gitignore` entries for paths that no longer exist. **DONE: dropped `!uv.lock` exception in `3b0b484`.**
- pyproject.toml dev-dep entries (`coverage`, `pytest-cov`, etc.) that are now redundant.
- Orphan scripts referencing dropped or renamed targets. **DONE: removed `scripts/run_nre_predict_local.sh` (referenced the deleted `//nre/nrm:run` target) in `3b0b484`.**

Iterate to 4.8 convergence.

- **Commits:** one per deletion.

### D.7 — Final layout audit

After D.6, verify with `ls -la` (or `tree -L 1 -a`) that the top-level matches asset-harvester's 15-entry shape (modulo our standalone-specific essentials: `baselines/`, `tests/`, `instant_nurec_example_call.sh`, `CLAUDE.md`, `.gitattributes`, `.pre-commit-config.yaml`). Anything else is a bug; back to D.1.

- **Commit (if needed):** `chore(layout): final asset-harvester-shape conformance audit`.

---

## Verification (canonical end-to-end)

Per plan.md §Verification:

```
# Setup (sandboxed):
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Inference (unsandboxed; GPU-bound):
mkdir -p /tmp/nurec_iter/no_merge && \
  python run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir /tmp/nurec_iter/no_merge --merge none
mkdir -p /tmp/nurec_iter/merge && \
  python run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir /tmp/nurec_iter/merge --merge frustum-ownership

# Parity (sandboxed):
python benchmark/validate_parity.py merge \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/nurec_iter/merge/*/ply/*/*.ply
python benchmark/validate_parity.py no_merge \
  baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
  /tmp/nurec_iter/no_merge/*/ply/*/

# Tests + lint (sandboxed):
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
```

All exit 0 within `tests/tolerance.json`. Branch is shippable when: parity green for both modes, all tests green, ruff clean, no `from libs.X` imports remain, no bazel files remain, top-level matches asset-harvester's 15-entry shape.

## Critical files to modify

- `instant_nurec/_pkg/utils/batch.py` (most kernel imports; A.1, A.6)
- `instant_nurec/_pkg/utils/sensors/sensors.py` (A.5)
- `instant_nurec/_pkg/utils/sensors/ncore_sensors_converters.py` (A.2, A.3)
- `instant_nurec/_pkg/utils/packed_ops.py` (A.4)
- `instant_nurec/_pkg/datasets/tracks.py` (A.4, A.7)
- `instant_nurec/_pkg/nrm/utils/cubemap.py` (A.7)
- `instant_nurec/_pkg/utils/sensors/_kernel_types.py` (new; A.2/A.3)
- `tests/tolerance.json` (bumps in A.5/A.6/A.7)
- `pyproject.toml` (final form after B)
- `.gitignore` (D.2)
- All bazel files (deleted in B.2)
- All `_pkg/` paths (relocated in C.1)
- `LICENSE.txt`, `THIRD_PARTY_LICENSE.txt`, `SECURITY.md`, `CONTRIBUTING.md`, `data_samples/` (new in D.4)

## Reuse opportunities

- `instant_nurec/_pkg/utils/geometry.py:se3_matrix_to_tquat` already provides A.1's math.
- `torch.searchsorted`, `torch.repeat_interleave`, `torch.scatter_reduce_` cover all packed-array helpers.
- `torch.nn.functional.grid_sample` (already used to replace nvdiffrast) covers cubemap projection if A.7's `camera_rays_to_image_points` needs a similar pattern.
- The existing 96%-coverage test suite already mocks the kernel surfaces via `sys.modules` stubs; converting those stubs to call the new torch impls is a small delta per kernel.
