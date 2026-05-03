# Plan: Extract NRM Kelvin predict-mode into a standalone repo

## Communication
Send me a slack message with the latest status and updates every commit or every hour, what ever comes first.

## Context

`/storage/projects/instant-nurec` is currently empty (only `CLAUDE.md`, baselines, prior planning docs). The goal is to make a single entrypoint of `/storage/projects/nre` — **NRM Kelvin model in predict mode only** — fully standalone here, with no NRE dependency, no Bazel, no Hydra, no PyTorch Lightning, and no non-torch CUDA libraries (no `nvdiffrast`, `torch_scatter`, `gsplat`).

Reference invocation in NRE (target to match exactly, byte-by-byte modulo non-determinism): `nre_example_call.sh`. Reference invocation for the **final** standalone CLI (post-Phase-3): `instant_nurec_example_call.sh`.

The work proceeds in 4 phases under one autonomous branch, with PLY parity against the existing baselines maintained at every step.

## Resolved decisions (from user)

- **Branch:** `kelvin-standalone`, off `main`. Single branch for all phases. No reuse of `port/*`, `plan/*`, or `nurec/*`.
- **HF mock target (Phase 4):** `nvidia/instant-nurec-kelvin` (placeholder; corp will replace later).
- **Tooling:** loose inspiration from `NVIDIA/asset-harvester`. **No PyTorch Lightning, no Hydra, no Bazel** in the final state. Only CUDA dependency in the final state is via `torch` itself (Phase 2 enforces the CUDA-lib piece; Phase 3 enforces the Bazel piece).
- **Build-system timeline:**
  - Phases 0–2 keep Bazel as the build system. Reason (user): bazel-compiled slang/CUDA binaries are deterministic across runs, which keeps parity stable while we strip and refactor. The Phase 1 Step 3 CLI is therefore invoked as `bazel run //instant_nurec:run -- --merge … --ncore-path … --output-dir … --log-level …` (new argparse flags, but bazel launches it). Phase 3 swaps the build system, and only then does the canonical call become `python run_inference.py …` per `instant_nurec_example_call.sh`.
- **PLY output naming for the new CLI:** `merged.ply` (merge mode) and `chunk_0000.ply`, `chunk_0001.ply`, … (no-merge mode). The baselines use `pai_<UUID>.ply` and `pai_<UUID>_chunkN.ply` — that's fine; `validate_parity.py` takes explicit paths and does not care about names.
- **Determinism tolerance:** measured empirically from the 5 reruns in `baselines/more_baselines/run_{1..5}` and stored in `tests/tolerance.json`. Per-property max pairwise diff. Tolerance can only ratchet upwards (Phase 2 CUDA→torch swaps may bump it; Phase 1 strips must not).
- **Autonomy:** run end-to-end without permission prompts; resolve blockers using NRE@`a54a6af0a177beabd01fe37e398c45be165a270f` as the source of truth (the commit pinned by the user — `CLAUDE.md` no longer pins a branch). Per CLAUDE.md §4.1.1: GPU-bound calls are unsandboxed; everything else is sandboxed.

## Baselines (already exist; do not regenerate)

- `baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_000da9de-…/pai_000da9de-….ply` — 211 MB, 2,871,662 vertices.
- `baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_000da9de-…/pai_000da9de-…_chunk0.ply` — 129 MB, 1,748,388 vertices.
- `baselines/original_baseline/no_merge/.../pai_000da9de-…_chunk1.ply` — 105 MB, ~1.1M vertices.
- `baselines/original_baseline/{merge,no_merge}/*/config/parsed.yaml` — full Hydra-resolved config; this is the contract for predict-mode functionality.
- `baselines/original_baseline/log.txt` — reference log.
- `baselines/more_baselines/run_{1..5}/{merge,no_merge}/...` — five reruns of the same script for non-determinism measurement.

PLY schema (both modes): `x,y,z` (float), `nx,ny,nz` (float), `red,green,blue,alpha` (uchar), `rot_0..rot_3` (float), `scale_0..scale_2` (float), `opacity` (float), `f_dc_0..f_dc_2` (float), `road_mask` (uchar), `sky_mask` (float).

## Source of truth (NRE files for porting; commit `a54a6af0a177beabd01fe37e398c45be165a270f`)

- Entrypoint: `nre/nre/nrm/run.py:main()` → `launch_trainer_loop()` → `trainer.predict()`.
- Predict driver: `nre/nre/nrm/systems/gaussians_nrm.py:predict_step()` (lines 706–749) and `on_predict_batch_end()` (lines 751–873).
- PLY export: `nre/nre/nrm/predict/export_ply.py:PLYExportGaussians.export()`; writer in `nre/nre/models/gaussians/utils.py:write_ply_3dgs()`.
- Primitive merge (frustum-ownership): `nre/nre/nrm/predict/primitive_merge.py` (uses `torch_scatter.scatter_max/scatter_add`).
- Model: `nre/nre/nrm/models/kelvin_model.py:KelvinNRM`; backbone in `nre/nre/nrm/models/kelvin_backbone/`.
- Primitive: `nre/nre/nrm/primitives/kelvin_primitive.py` (uses `nvdiffrast.torch as dr` for the cubemap sky path).
- Dataset: `nre/nre/nrm/datasets/nrm_ncore.py:NCoreNRMDataset`; data module: `nre/nre/nrm/datasets/datamodule.py`.
- Configs: `nre/configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml`, `nre/configs/nrm/apps/options/_kelvin_predict.yaml`.
- Pretrained weights cache (NRE creates on first run): `${HOME}/.cache/nrm/pretrained_models/kelvin_pa_front/`.

When in doubt during porting, refer to NRE@`a54a6af`. Self-invented fixes only when NRE has no equivalent (e.g. lightning removal); commit message must include `(self-invented: <reason>)` per CLAUDE.md §4.1.2.2.

## Hard rules (apply to every step)

1. **Parity at every step.** After each commit, run `scripts/validate_parity.py` against `baselines/original_baseline` (and `more_baselines/run_*` for the determinism tolerance band). If parity breaks, fix before moving on. Use the iteration loop in `CLAUDE.md` §4.1/4.2.
2. **CLAUDE.md §0:** before any implementation/fix, check NRE@`a54a6af` first.
3. **CLAUDE.md §1–3:** TDD. Full branch coverage testing on every function, where possible. New code is added test-first.
4. **One commit per (sub)step**, descriptive messages.
5. **Sandbox rule:** only GPU-bound calls (the Phase 1+ end-to-end run via bazel, and post-Phase-3 `python run_inference.py …`) are unsandboxed. Every other command (pytest, ruff, git, validate_parity.py, file edits) is sandboxed.

---

## Phase 0 — Setup & parity tooling

### Step 1 — `scripts/validate_parity.py`

Critical file: `scripts/validate_parity.py` (new).

CLI:
- `validate_parity.py merge <baseline_ply> <proposed_ply>` — single-file compare.
- `validate_parity.py no_merge <baseline_dir> <proposed_dir>` — directory compare.

Implementation: pure `plyfile` for I/O, **`torch` for the per-property diff** (PLYs are 100–200 MB, 2–3M vertices each — torch is dramatically faster than numpy here, and a GPU is already required for the inference step that produces the inputs). Property tensors are loaded onto the same device the user's torch is configured for; falls back to CPU. Tolerances loaded from `tests/tolerance.json` (per-property; default `1e-3` if missing).

Checks (hard fail on any miss; exit 1 with details):
1. File count exact (no_merge mode).
2. Per-file vertex count exact.
3. Property names + dtypes exact.
4. Per-property `(a − b).abs().max()` ≤ tolerance for that property.
5. Exit 0 only if all pass.

Pair files between dirs by sorted filename (stable mapping under both `chunk0/1.ply` baseline names and `chunk_0000/0001.ply` post-Phase-3 names).

Tests in `tests/test_validate_parity.py` cover: identical files (pass), 1-LSB-perturbed property (fail at tolerance), missing file (fail at count), mismatched dtype (fail at schema), torch-CPU vs torch-CUDA path equivalence.

Commit: `feat(parity): add scripts/validate_parity.py with torch-backed diff`.

### Step 1.7 — Self-test

Run `validate_parity.py` on `original_baseline` vs itself (both modes). Must exit 0. Commit: `test(parity): self-compare green`.

### Step 1.8 — Determinism tolerance from 5 reruns

Run all C(5,2)=10 pairwise comparisons across `baselines/more_baselines/run_{1..5}` for both `merge/` and `no_merge/`. For each property, take `max(|a − b|)` across all pairs and all elements. Write `tests/tolerance.json` keyed by property name. This is the run-to-run noise floor of the original NRE pipeline; all subsequent steps must remain inside it (Phase 2 may ratchet upward, never downward).

Commit: `chore(parity): record per-property determinism tolerance from 5 baseline reruns`.

### Step 2 — Verbatim copy from NRE

Copy NRE@`a54a6af` into the repo. Strip `.git`, `bazel-out`, `bazel-bin`, `bazel-nre`, `bazel-testlogs`, `.test_cache`, `.venv`, `.mypy_cache`, `.ruff_cache`, `.cursor`, `tmp`. Keep `MODULE.bazel`, `BUILD.bazel`, `bazel/`, `tools/`, `pyproject.toml`, the entire `nre/` source tree, `configs/`, `libs/`, `internal/`, etc. (Bazel is intentionally retained until Phase 3.)

Substep 2.1: Reproduce `nre_example_call.sh` *from this directory* — same flags, same overrides, but bazel runs against the local copy instead of `cd /storage/projects/nre`. Generate fresh PLYs to a scratch `out_dir`.

Substep 2.2: Run `validate_parity.py` against `original_baseline` using the tolerance from §1.8. Must pass. If it doesn't, the copy is incomplete or bazel picked up an external dep — fix before proceeding.

Commit: `feat(copy): import NRE@a54a6af verbatim and verify parity from this directory`.

---

## Phase 1 — Build the minimal codebase (most important phase)

Throughout Phase 1, the invocation is **bazel-based**: `bazel run //instant_nurec:run -- --merge {none,frustum-ownership} --ncore-path … --output-dir … --log-level …`. The `bazel run` underpinning keeps slang/CUDA-compiled binaries deterministic; the Python entrypoint behind the bazel target uses the new argparse flags. Each step ends with parity green.

### Step 3 — Minimal CLI (bazel-launched)

Critical files (new):
- `instant_nurec/cli.py` — argparse entrypoint exposing `--ncore-path`, `--output-dir`, `--merge`, `--log-level`.
- `instant_nurec/BUILD.bazel` — `py_binary(name = "run", srcs = ["cli.py"], deps = [...])`.

Mapping:
- `--merge none` (default) → internal config sets `predict.primitive_merge.enabled=false`.
- `--merge frustum-ownership` → `predict.primitive_merge.enabled=true` + `overlap_strategy=frustum_ownership`.
- `--ncore-path` → maps onto `dataset.predict.ncore_json_base_path` and the default `debug.lst` lookup.
- `--output-dir` → maps to `out_dir`.
- `--log-level` → standard `logging.basicConfig(level=...)`.

Initial implementation: `cli.py` constructs the same `parsed.yaml`-equivalent config the NRE-copied code expects, then delegates to the NRE-copied predict path (which still imports `pytorch_lightning`, `hydra`, etc. — those go away in step 4). At this stage we are *only* introducing the new flag surface, not yet stripping.

Tests: `tests/test_cli.py` — argparse parsing for default and both `--merge` values, bad values rejected, log-level forwarded, mapping to internal config dict matches `parsed.yaml`.

Verify end-to-end: run both `--merge none` and `--merge frustum-ownership` via `bazel run //instant_nurec:run -- …`, then `validate_parity.py` green.

Commit: `feat(cli): add bazel-launched argparse entrypoint with --merge {none,frustum-ownership}`.

### Step 4 — Iterative aggressive strip

Reference: `parsed.yaml` from `baselines/original_baseline/{merge,no_merge}/*/config/`. Anything not transitively touched by either run is a strip candidate.

Sub-iterations (each its own commit; parity-checked after each):
- **4.0** Drop all files of all types that are not needed by nrm and not within the current path.
- **4.1** Drop everything not Kelvin: `nre/nrm/models/celsius*`, all non-Kelvin model code, non-Kelvin systems.
- **4.2** Drop training/val/test paths: `train_step`, `validation_step`, callbacks not used by predict, `Trainer.fit/validate/test`, optimizer/scheduler config, EMA, AMP, distributed strategies (except single-GPU passthrough). The `Trainer.predict` shim is the only PL surface kept here — fully removed in 4.5.
- **4.3** Drop output-irrelevant code: profiling (`torch.profiler`, perf hooks), W&B/TensorBoard, USDZ export, the entire video render path (`render_video.enabled=false` is hard-coded), `render_rig_trajectories_video`, anything whose deletion does not change the output PLY. Slang in `libs/slang_gaussians/` and `libs/losses/kernel/` is reachable only through training/loss paths (per exploration: `force_disable_cuda=True` in predict's loss base) — drop after their callers are gone.
- **4.4** Drop all YAML files. Encode the predict config as a Python `dataclass`/dict in `instant_nurec/config.py`, sourced 1:1 from `parsed.yaml`. After this, no `.yaml` files, no `hydra-core`/`omegaconf` imports.
- **4.5** Drop NRE as a dependency: rename the package (everything moves under `instant_nurec/`), fix imports, prune unreachable `libs/*` subtrees. Drop `pytorch_lightning` entirely — replace `Trainer.predict` with a plain loop over `predict_dataloader()` calling `model.predict_step()` then `on_predict_batch_end()` (commit message: `(self-invented: lightning removal — no equivalent in NRE)`).
- **4.6** Remove the strings `nre`/`NRE` from the codebase (modulo a single provenance line if useful).
- **4.7** `/simplify` pass: only ncorev4 ingest → batch prep → Kelvin predict → PLY export remain. Anything else → loop back to 4.1.
- **4.8** Convergence check: try removing one more file/function/import. Parity breaks → keep. Parity holds → delete and commit. Stop when no more deletions are possible.

After every sub-iteration: bazel-run both modes, then `validate_parity.py`. Commit only on green.

### Step 5 — Save/load entire model

Reference: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model

Run the stripped predict path once, then `torch.save(model, "kelvin_full.pt")`. Replace constructor-from-config with `model = torch.load("kelvin_full.pt", map_location=device, weights_only=False)`.

This unlocks deleting:
- Factory functions used only at construction (`make_encoder`, `make_decoder`, `make_sky`, `BaseGaussianRenderer.factory`, …).
- Config branches that exist only to drive construction.
- Class-level config objects no longer needed at runtime.

After 5, **return to 4** and strip what just became dead. Iterate 4↔5 until convergence (no further deletion holds parity).

`kelvin_full.pt` is *not* committed (large). Generated locally; plumbed through the HF mock in Phase 4 step 9.

Commits: `feat(model): save/load entire kelvin model`, then per-iteration `refactor(strip): remove <X> after full-model load`.

### Step 6 — Unit tests

Full branch coverage on every function still in the codebase, where possible (CLAUDE.md §1–2). Layout: `tests/` mirrors source tree. `pytest` only, no fancy framework.

Coverage strategy: every public function has at least one test; functions with branches have one test per branch; trivial wrappers ("call torch op X") have a single smoke test. `pytest --cov` reported in CI but the gate is "passes", not "100%".

Commit: `test: full-branch coverage suite for stripped codebase`.

---

## Phase 2 — Strip non-torch CUDA (`nvdiffrast`, `torch_scatter`, `gsplat`)

Bazel still drives the build through Phase 2. The strip here is at the Python-import level: we drop `import nvdiffrast`, `import torch_scatter`, `import gsplat` and replace each call with pure-torch equivalents.

### Step 7 — TDD CUDA-library replacement

| Dep | Used at | Predict-relevant? | Replacement |
|---|---|---|---|
| `torch_scatter.scatter_max`, `scatter_add` | `nrm/predict/primitive_merge.py` | Yes (frustum-ownership merge) | `torch.scatter_reduce_(reduce="amax"/"sum")` |
| `nvdiffrast.torch as dr` | `nrm/utils/cubemap.py`, `nrm/primitives/kelvin_primitive.py` | Likely yes (cubemap sky sample) | `torch.nn.functional.grid_sample` over cubemap face buffers |
| `gsplat` | `models/gaussians/renderers.py` (factory) | **Probably no** when `render_video.enabled=false` (no rasterization in PLY-only path); confirmed once 1.4.3 deletes the renderer call sites. | If unused: drop the dep. If still reachable: pure-torch rasterizer (with TDD equivalence test). |

Per replacement (one commit each):
- **7.1** Write equivalence tests *first* (TDD). Cover all branches and edge cases (empty inputs, single element, large batches, dtype mismatches, contiguous/non-contiguous tensors). Tests run both implementations on identical input and assert `(out_torch − out_cuda).abs().max() ≤ machine_eps_band`.
- **7.2** Implement the pure-torch version.
- **7.3** Run the equivalence tests until green.
- **7.4** Run `validate_parity.py` against `original_baseline`. If a property's diff exceeds the current tolerance, set `TOL_new[prop] = max(TOL_old[prop], max_observed_in_equivalence_test)` in `tests/tolerance.json`. Document every TOL bump in the commit body.
- **7.5** Run Phase 1.4 strip on whatever just became dead (the original CUDA imports, factory branches, BUILD deps).

Commits: `feat(merge): replace torch_scatter with torch.scatter_reduce_`, `feat(sky): replace nvdiffrast cubemap sample with grid_sample`, `chore(deps): drop gsplat (unused in predict)`.

### Step 7-bis — Re-iterate Phase 1 over the CUDA-stripped codebase, until convergence

After all CUDA-library deps are gone, re-run Phase 1.4 (strip) → 1.5 (full-model save/load — re-pickle since the model class changed) → 1.6 (test pass) **iteratively, not once**. Each loop: try to delete more, save/load again, re-run tests. Continue until a complete pass yields zero deletions and zero diffs from the previous pickle. Parity gated at every commit.

---

## Phase 3 — Build system + repo structure (asset-harvester loose inspiration)

Reference: https://github.com/NVIDIA/asset-harvester (loose template).

### Step 8 — Restructure & switch the call from `bazel run` to `python run_inference.py`

Substeps:
- **8.1** Drop bazel: remove `MODULE.bazel`, `BUILD.bazel` files, `bazel/`, `bazel-*/`, `tools/`. Replace with `pyproject.toml` (setuptools backend). Drop remaining non-essential deps: `hydra-core`, `omegaconf`, `nvidia-ncore-internal` (if a public-only path works; otherwise scope to the ncorev4 reader). Keep: `torch`, `numpy`, `plyfile`, `pillow`, `pyyaml` (only if needed for ncorev4 metadata), `tqdm`, `pydantic` (only if reachable).
- **8.2** Repo layout (target):
  ```
  instant_nurec/                  # main package
    __init__.py
    cli.py                        # argparse + dispatch (was bazel-py_binary)
    model.py                      # KelvinNRM, save/load
    primitives/                   # KelvinNRMPrimitive
    datasets/                     # ncorev4 reader, frame batcher
    predict/                      # predict loop, primitive merge, ply export
    utils/                        # batch, geometry, log
  scripts/
    validate_parity.py
  tests/
    tolerance.json
    test_*.py (mirroring instant_nurec/)
  data_samples/                   # tiny ncorev4 fixture
  internal/                       # see 8.3
  baselines/                      # existing, untouched
  run_inference.py                # thin wrapper around instant_nurec.cli
  setup.sh
  run.sh
  pyproject.toml
  README.md
  ```
- **8.3** Move migration scaffolding (NRE provenance markers, parity helpers, anything used to *reach* parity but not at runtime) into `internal/`. Hard rule: `rm -rf internal/` must not break `python run_inference.py …`. Add `tests/test_internal_optional.py` that imports the runtime package with `internal/` renamed away.
- **8.4** Switch the canonical invocation from `bazel run //instant_nurec:run -- …` to `python run_inference.py …` (matching `instant_nurec_example_call.sh`).
- **8.5** Re-run Phase 1.4 strip + parity end-to-end on the new layout — bazel-removal usually exposes more dead paths (e.g. `BUILD.bazel`-only deps).

Tooling (loose-inspiration tier from asset-harvester):
- `pyproject.toml` + setuptools.
- `ruff` for lint + format.
- Plain venv in `setup.sh` (no conda required, no CUDA-version pinning — torch wheels handle it).
- `run.sh` wraps `python run_inference.py` with arg validation (input path exists, output dir creatable).

Commits: `refactor(layout): migrate to instant_nurec/ package shape`, `chore(build): drop bazel; switch to pyproject.toml`, `chore(deps): trim to torch+numpy+plyfile+...`, `chore(internal): isolate migration scaffolding under internal/`, `feat(cli): switch canonical call to python run_inference.py`.

---

## Phase 4 — Mocks, README, polish

### Step 9 — HuggingFace mock

Stub `huggingface_hub.snapshot_download` (or whichever HF entrypoint we adopt) to resolve `nvidia/instant-nurec-kelvin` (placeholder) to:
- `kelvin_full.pt` → local cached path (`~/.cache/instant_nurec/kelvin_full.pt`), produced by Phase 1 step 5 / Phase 2 step 7-bis.
- `ncorev4_sample/` → local fixture path (or `--ncore-path` override).

Mock lives in `instant_nurec/_hf_mock.py`. Production code imports `huggingface_hub.snapshot_download` normally; the mock monkey-patches it for now. `INSTANT_NUREC_HF_MOCK=1` (default on) selects the mock. When the corp publishes the actual repo, flipping the env var off uses real HF.

Tests: `tests/test_hf_mock.py` — mock returns expected paths, missing-file errors clearly, env var toggle works.

Commit: `feat(hf): mock snapshot_download to resolve to local model+data`.

### Step 10 — README, setup.sh, run.sh

- `README.md`: Overview → Setup → Quickstart (the two example calls) → CLI reference → Troubleshooting → License. Self-contained.
- `setup.sh`: create venv, `pip install -e .`, optionally pre-fetch model via HF mock, print `source .venv/bin/activate` line.
- `run.sh`: validate `--ncore-path` exists and `--output-dir` is writable, then exec `python run_inference.py "$@"`.

Commit: `docs(readme): add README, setup.sh, run.sh`.

### Final — MR

- Push `kelvin-standalone` to `origin` (GitLab `nrs/instant-nurec`).
- Open MR via `glab` (or the `create-mr` skill) titled "Standalone NRM Kelvin predict mode". Body summarizes: what the code does, parity proof (paste `validate_parity.py` outputs against `tests/tolerance.json`), per-phase summary, list of removed deps, reproduction commands.

---

## Verification

End-to-end (GPU-bound; **unsandboxed** per CLAUDE.md §4.1.1):

Phase 1+2 form (bazel-launched):
```
bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec_iter/no_merge --merge none
bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec_iter/merge --merge frustum-ownership
```

Phase 3+ form (matches `instant_nurec_example_call.sh`):
```
mkdir -p /tmp/nurec_iter/no_merge && \
  python run_inference.py --ncore-path /storage/data/nurec/ncorev4 \
                          --output-dir /tmp/nurec_iter/no_merge --merge none

mkdir -p /tmp/nurec_iter/merge && \
  python run_inference.py --ncore-path /storage/data/nurec/ncorev4 \
                          --output-dir /tmp/nurec_iter/merge --merge frustum-ownership
```

Parity (sandboxed):
```
python scripts/validate_parity.py merge \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/nurec_iter/merge/*/ply/*/merged.ply

python scripts/validate_parity.py no_merge \
  baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
  /tmp/nurec_iter/no_merge/*/ply/*/
```

Both must exit 0 within `tests/tolerance.json`.

Tests (sandboxed):
```
pytest -q
ruff check .
```

Branch is shippable when: parity green for both modes, all tests green, ruff clean, MR opened.

## Critical files to create / heavily modify

- `scripts/validate_parity.py` (new; torch-backed)
- `tests/tolerance.json` (new; populated 1.8, may bump in 7.4)
- `instant_nurec/cli.py`, `instant_nurec/model.py`, `instant_nurec/predict/*.py`, `instant_nurec/datasets/*.py`, `instant_nurec/primitives/*.py`, `instant_nurec/utils/*.py` (new package; ported and stripped from NRE)
- `instant_nurec/BUILD.bazel` (Phase 1 only; deleted in Phase 3.1)
- `instant_nurec/_hf_mock.py` (new, Phase 4)
- `run_inference.py` (Phase 3.4)
- `pyproject.toml`, `setup.sh`, `run.sh`, `README.md` (new, Phase 3/4)
- `tests/test_*.py` (mirroring source tree)
- `internal/` (new, Phase 3.3 — migration scaffolding)

## Reuse opportunities (from exploration)

- `plyfile` library handles all PLY I/O — no custom parsing in `validate_parity.py`.
- NRE@`a54a6af` `nre/nrm/predict/export_ply.py:PLYExportGaussians.export()` is the exact PLY-writer shape to keep (ports cleanly after Phase 1.4.5).
- NRE `nre/nrm/predict/primitive_merge.py` has the frustum-ownership math; only the two `torch_scatter` calls need replacement in Phase 2.
- NRE `nre/nrm/datasets/nrm_ncore.py` is the ncorev4 reader; carry forward minus `nvidia-ncore-internal` paths if they aren't reachable in predict.
- `parsed.yaml` from baselines is the Hydra-resolved config that feeds Phase 1.4.4's Python-literal config.
