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

1. **Parity at every commit.** Run `scripts/validate_parity.py` against `baselines/original_baseline` after each step; commit only on green. Phase 2-style CUDA→torch swaps may ratchet `tests/tolerance.json` upwards (per-property), never downwards. Re-document any tolerance bump in the commit body.
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

### A.1 — `se3pose_from_matrix` → reuse existing `se3_matrix_to_tquat` (DEFERRED)

- **Status:** deferred until after A.6 lands. A faithful slang-equivalent torch
  impl (strict-comparison Shepperd, slang's ``s = 2*sqrt(1+...)`` formula,
  ``normalizeSafe``) still produces ~1-ULP drift relative to the bazel-built
  kernel on GPU. The drift propagates through the still-slang
  ``image_points_to_world_rays_shutter_pose`` (A.6) into a non-deterministic
  Gaussian cull boundary, breaking the exact vertex-count contract
  (1748388 → 1748396 chunk0; 1428209 → 1428215 chunk1; 2871662 → 2871628 merge).
- **Call site:** `instant_nurec/_pkg/utils/batch.py:20` (import), `:773-774` (call returning `(translations, rotations)`).
- **Resume condition:** once A.5+A.6 land a torch ray-gen, the FP precision of the entire pose → ray chain is controlled in Python and the kernel can be replaced. The bit-identity-with-slang requirement disappears.
- **Replacement (deferred):** new function `instant_nurec/_pkg/utils/geometry.py:se3pose_from_matrix` mirroring `libs/geometry/kernels/quaternion.slang::fromMatrix` exactly (strict-cmp Shepperd + slang's `s = 2*sqrt(1+...)` formula + slang's `normalizeSafe`).
- **Test:** `tests/test_se3pose_torch.py` — equivalence on identity, translation-only, rotation-only, random SE(3) batches; check `(out_torch - out_kernel).abs().max() ≤ 1e-6`.
- **Parity risk:** ~1-ULP per-component until A.6 absorbs the drift.
- **Commit (deferred):** `feat(geometry): replace libs.geometry.kernels.pose.se3pose_from_matrix with torch helper (Phase A.1)`.

### A.2 — `libs.sensors.kernels.cameras.parameters.*` → in-tree dataclasses

- **Call site:** `instant_nurec/_pkg/utils/sensors/ncore_sensors_converters.py:22-32` (imports), `:79-96` (instantiation in `_convert_*` methods).
- **Replacement:** `instant_nurec/_pkg/utils/sensors/_kernel_types.py` (new file). Reproduce each as `@dataclass(slots=True, frozen=True)` containers (or pydantic if validation needed). All are pure value structs — no math.
- **Test:** `tests/test_kernel_types.py` — round-trip through `from_components` factory; field-name parity against the bazel-built names.
- **Parity risk:** none (no math).
- **Commit:** `feat(sensors): replace libs.sensors.kernels.cameras.parameters with in-tree dataclasses (Phase A.2)`.

### A.3 — `libs.sensors.kernels.common.{Pose, DynamicPose}` → in-tree dataclasses

- **Call site:** `ncore_sensors_converters.py:34`, used as `DynamicPose(start_pose=Pose(translation=..., rotation=...), end_pose=...)` in `batch.py:785-787`.
- **Replacement:** add `Pose`/`DynamicPose` dataclasses to `instant_nurec/_pkg/utils/sensors/_kernel_types.py`.
- **Test:** equivalence on construction + attribute access.
- **Parity risk:** none.
- **Commit:** `feat(sensors): replace libs.sensors.kernels.common.{Pose,DynamicPose} with in-tree dataclasses (Phase A.3)`.

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

### A.5 — `compute_poses_and_timestamps` → torch rolling-shutter interp

- **Call site:** `instant_nurec/_pkg/utils/sensors/sensors.py:20` (import), `:79-88` (call).
- **Reference:** read NRE@`a54a6af`/libs/sensors/kernels/pose_calib.slang for the math + the docstring on `sensors.py:67-73` ("rolling-shutter interpolation via the Slang kernel").
- **Replacement:** torch SE(3) linear pose interpolation between `T_sensor_world_startend[:, 0]` and `T_sensor_world_startend[:, 1]` per-frame, parameterized by the row-time normalized to `[0, 1]` from the rolling-shutter timing model. The standalone always pins `subsample_rect_points_lb=None`, `subsample_resolution=None`, `embed_weights=None`, `enable_calib=False`, so only the bare interp path needs implementing.
- **Test:** `tests/test_pose_calib_torch.py` — equivalence on synthetic SE(3) batches against the bazel kernel; check translation diff ≤ `1e-5`, rotation diff ≤ `1e-5` rad.
- **Parity risk:** medium. Per-property tolerance for `x,y,z,rot_*` may need to bump from current values; document the band measured in the equivalence test in the commit body.
- **Commit:** `feat(sensors): replace libs.sensors.kernels.pose_calib.compute_poses_and_timestamps with torch impl (Phase A.5)`.

### A.6 — `image_points_to_world_rays_shutter_pose` → torch

- **Call site:** `instant_nurec/_pkg/utils/batch.py:21` (import), `:779-792` (call).
- **Reference:** NRE@`a54a6af`/libs/sensors/kernels/cameras/*.slang for the per-camera-model projection + shutter-aware ray gen.
- **Replacement:** pure-torch path consuming the A.2/A.3 dataclasses. For each camera-model branch (OpenCVPinhole / OpenCVFisheye / FTheta), implement the inverse-projection (image-point → camera-frame ray) using torch tensor ops; combine with `DynamicPose`-interpolated rotation/translation (from A.5) to produce world-frame rays + timestamps. `image_points=None` is the only case the standalone hits — that's "all pixels" shorthand.
- **Test:** `tests/test_image_points_to_world_rays_shutter_pose_torch.py` — equivalence against bazel kernel for each camera model on a synthetic image; check ray-direction dot-product ≥ `1 - 1e-5`.
- **Parity risk:** medium-high. This is the most numerically sensitive kernel; tolerance bump likely.
- **Commit:** `feat(sensors): replace image_points_to_world_rays_shutter_pose with torch impl (Phase A.6)`.

### A.7 — `libs.vren.interface.*` → torch

- **Call sites:**
  - `instant_nurec/_pkg/datasets/tracks.py:20-21` (imports), `:322` (`ray_cuboidtracks_intersection`), `:363` (`point_cuboidtracks_intersection_interpolate_pose`).
  - `instant_nurec/_pkg/nrm/utils/cubemap.py:15` (import), `:66` (`camera_rays_to_image_points`).
- **Replacement:**
  - `ray_cuboidtracks_intersection(rays_o, rays_d, ts_us, packinfo, poses_data, ts_us, dims, max_n, max_per_ray, with_ts)`: slab-method AABB intersection per (ray × cuboid × keyframe) over the packed cuboid-tracks tensor. Pure-torch broadcasting + top-k. Memory-aware chunking for the (rays × cuboids) outer product when needed.
  - `point_cuboidtracks_intersection_interpolate_pose(points, ts_us, packinfo, poses_data, ts_us, dims, max_n)`: per-point inside-cuboid test + linear-interp pose at the point's timestamp. Pure-torch.
  - `camera_rays_to_image_points(rays_o, rays_d, projection)`: forward camera projection (inverse of A.6's per-pixel ray gen). Pure-torch.
- **Test:** `tests/test_vren_torch.py` — equivalence against bazel kernel on synthetic cuboid-track / ray batches; tolerate `1e-4` for projection, exact for intersection counts.
- **Parity risk:** highest of the kernels. Tolerance bump probable on multiple PLY properties.
- **Commits:** one per function (3 commits).

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

## Phase B — Drop bazel

Per plan.md §8.1 — but only safely doable AFTER Phase A because libs/ is gone.

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

## Phase C — Flatten layout to asset-harvester shape

Per plan.md §8.2.

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

### D.4 — Add asset-harvester-style files

asset-harvester has these; we don't:
- `LICENSE.txt` — NVIDIA proprietary text. Source from corp / asset-harvester's own copy. **Pending — needs corp source material.**
- `THIRD_PARTY_LICENSE.txt` — replaces the dropped `NOTICES`. Source from corp / asset-harvester. **Pending — needs corp source material.**
- `SECURITY.md` — corp standard security-disclosure policy. **Pending — needs corp source material.**
- `CONTRIBUTING.md` — short dev guide pointing at `setup.sh`, `pytest`, `ruff`, `validate_parity.py`. **DONE — `dd8b5b3`.**
- `data_samples/` — small ncorev4 fixture (1 sequence, ≤ 50 MB) to make the README quickstart self-contained. Source: extract from `/storage/data/nurec/ncorev4` using the same sample that produced the baselines. The HF mock (`instant_nurec/_hf_mock.py:get_sample_data_path`) already references this name. **Pending — needs user direction on size/extraction strategy.**
- *(optional)* `benchmark/` — relocate `scripts/derive_determinism_tolerance.py` here, plus a `benchmark/parity.py` that wraps `validate_parity.py` for repeatable benchmarking. Keep the underlying script in `scripts/` if simpler.

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
python scripts/validate_parity.py merge \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/nurec_iter/merge/*/ply/*/*.ply
python scripts/validate_parity.py no_merge \
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
