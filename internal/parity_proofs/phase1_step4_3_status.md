# Phase 1 Step 4.3 status — extended strip

Branch: `kelvin-standalone` (pushed to `origin/kelvin-standalone`).
Last commit at writing: `bde8f01 refactor(strip): drop apps/ + root run.py/setup.py + image rules (Phase 1 step 4.3)`.

## Progress since last status doc

Phase 1 step 4.3 expanded into 7+ atomic commits. Each kept parity
bit-identical (every property max-diff = 0.0 vs `baselines/original_baseline`).

| Commit | Strip target |
|---|---|
| `1fef75a` | USDZ + video-render branches in `on_predict_batch_end` |
| `7e58fed` | Drop `export_usdz.py`, dataverse / websocket / benchmark / sampler-test files |
| `d864a41` | Excise training/val/test methods from `gaussians_nrm.py` (~620 LOC) |
| `0f471f6` | Simplify `nre/nrm/run.py` to predict-only Trainer config (~200 LOC dropped) |
| `39ee90b` | Excise training/val/test methods from `BaseNRMSystem` (~210 LOC) |
| `34f2657` | Drop `nre/viewer/` + render import from gaussians_nrm |
| `f6b35ba` | Drop `nre/{benchmark,grpc,metrics,run,systems,render}` (~10K LOC, 118 files) |
| `38e84c9` | Drop `libs/{nrend,packed_ops,pytorch3d_knn,gaussian_mcmc,ray_utils}` keeps + visualdebugger stub (then restored 4 of 5 libs that the predict path actually imports) |
| `4ed3ad5` | Drop `internal/`, `nre/internal/`, image-build infra (~16K LOC) |
| `bde8f01` | Drop `apps/`, root `run.py`/`setup.py`, image rules from root BUILD.bazel |

## Rough scope

- **.py file count**: 1801 → 427 (-76%)
- **.py LOC**: ~440K → ~149K (-66%)

## What still needs Phase 1

- **4.4 — drop YAML/Hydra**. The Hydra config tree under `configs/` is intact;
  `nre/nrm/config/nrm.py` still resolves Hydra defaults. Replace with a
  Python literal sourced from `baselines/original_baseline/*/config/parsed.yaml`.
- **4.5 — drop NRE name + pytorch_lightning**. Replace `pl.Trainer.predict(...)`
  with a hand-written predict driver: load model from checkpoint, set device,
  iterate `predict_dataloader()`, call `model.predict_step()` then
  `on_predict_batch_end()`. Rename `nre.*` imports under `instant_nurec.*`.
- **4.6 / 4.7 / 4.8** — string scrub, `/simplify` pass, convergence check.

## What still needs Phase 2 / 3 / 4

Same as in `phase1_step4_status.md`: Phase 2 (CUDA-lib TDD replacement),
Phase 3 (drop bazel; switch to `python run_inference.py`), Phase 4 (HF mock,
README, setup.sh, run.sh, MR).

## Reproduction

GPU-bound (unsandboxed):

```
mkdir -p /tmp/nurec/{no_merge,merge}
bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec/no_merge --merge none
bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec/merge --merge frustum-ownership
```

Sandboxed parity:

```
.venv/bin/python scripts/validate_parity.py no_merge \
  baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
  /tmp/nurec/no_merge/*/ply/*/

.venv/bin/python scripts/validate_parity.py merge \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/nurec/merge/*/ply/*/pai_*.ply
```

Both must exit 0 with `tests/tolerance.json` (every property = 0.0).

## Notes for next agent

- The repo no longer has `nre/{benchmark,grpc,metrics,run,systems,render,viewer,visualdebugger}` (visualdebugger is a 30-LOC stub) or any of `internal/`, `nre/internal/`, `apps/`, `libs/{ray_utils,nerfacc,optixtracer}`.
- `libs/{nrend, packed_ops, pytorch3d_knn, gaussian_mcmc}` are kept because the predict path imports them transitively.
- `libs/losses/models/primitive_losses.py` has stubbed Celsius classes (a pre-4.5 hack).
- `nre/utils/callbacks.py` has stubbed BaseSystem / BaseSystemSO classes (the real nre/systems was deleted).
- `nre/utils/types.py` has an inline get_visualdebugger() stub.
- `nre/visualdebugger/` is a tiny shim (`__init__.py` returns a no-op debugger).
- All BUILD.bazel files under nre/, libs/, configs/, bazel/version/ are now free of //internal, //apps, //nre/{benchmark,grpc,...} references.
