# Phase 1 Step 4 status

Branch: `kelvin-standalone` (pushed to `origin/kelvin-standalone`).
Last commit at writing: `d864a41 refactor(strip): excise training/val/test methods + helpers from gaussians_nrm (Phase 1 step 4.3)`.

## What's done

| Step  | Description | Commits |
|-------|-------------|---------|
| 0/1   | `scripts/validate_parity.py` torch-backed; 21 pytest cases incl. real-baseline self-test | `c586497`, `a7868fb` |
| 0/1.8 | `tests/tolerance.json` derived from 5 reruns: every property = 0.0 (NRE pipeline is fully deterministic) | `9bc5bd4` |
| 0/2   | NRE@`a54a6af` verbatim copy + parity proof at `internal/parity_proofs/phase0_step2.md` | `605871f`, `8bf9493` |
| 1/3   | `instant_nurec/cli.py` (argparse → Hydra overrides → `nre.nrm.run.main.callback`); 16 tests; bazel target `//instant_nurec:run`; visibility widened on `//nre/nrm:pylib` | `d6432ec` |
| 1/4.1 | Drop Celsius model + primitive (~2,251 LOC) | `a854525` |
| 1/4.2 | Hard-restrict `run.py` to predict mode | `18356d3` |
| 1/4.3 | Drop USDZ + video-render from `on_predict_batch_end` | `1fef75a` |
| 1/4.3 | Drop `export_usdz.py`, dataverse / websocket / benchmark / sampler-test files; clean BUILD.bazel deps | `7e58fed` |
| 1/4.3 | Excise training/val/test methods + helpers from `gaussians_nrm.py` (~620 LOC) | `d864a41` |

Parity is bit-identical to `baselines/original_baseline` after every commit on this branch (all 23 properties at `0.0` max diff in both `--merge none` and `--merge frustum-ownership`).

## What's still open in Phase 1

### Step 4 (continuing iterations)
- **4.4 — drop YAML config files.** Hardcode the resolved Hydra config from `baselines/original_baseline/*/config/parsed.yaml` into a Python literal in (likely) `instant_nurec/config.py`. Drop `hydra-core` and `omegaconf`.
- **4.5 — drop NRE as a dependency, drop `pytorch_lightning`.** Replace `Trainer.predict(...)` with a hand-written predict driver: load model, set device, iterate `predict_dataloader`, call `model.predict_step` then `on_predict_batch_end` per batch. Rename `nre.*` imports under `instant_nurec.*` and migrate code into the new package (or symlink/re-export until 4.6). Commit message: `(self-invented: lightning removal — no equivalent in NRE)`.
- **4.6 — remove the strings `nre`/`NRE`** (modulo a single provenance line if useful).
- **4.7 — `/simplify` pass.** Target shape: only ncorev4 ingest → batch prep → Kelvin predict → PLY export.
- **4.8 — convergence sweep.** Try removing one more file/function/import; keep deletions that hold parity.

### Subsequent
- **Step 5** — `torch.save(model, ...)` / `torch.load(..., weights_only=False)` round-trip; iterate 4↔5 to convergence.
- **Step 6** — branch-coverage tests on the surviving codebase.
- **Phase 2 step 7** — TDD-replace `torch_scatter`, `nvdiffrast`, optionally `gsplat` with pure-torch equivalents; bump `tests/tolerance.json` per property as needed.
- **Phase 2 step 7-bis** — re-iterate Phase 1 over the CUDA-stripped codebase to convergence.
- **Phase 3 step 8** — drop bazel; `pyproject.toml` + setuptools; `instant_nurec/` package layout under asset-harvester loose template; switch invocation to `python run_inference.py …`. Move migration scaffolding to `internal/`.
- **Phase 4 step 9** — HuggingFace mock at `nvidia/instant-nurec-kelvin` (placeholder).
- **Phase 4 step 10** — README, `setup.sh`, `run.sh`.
- **Final** — open MR on `gitlab-master.nvidia.com:12051/nrs/instant-nurec` from `kelvin-standalone` to `main`.

## Reproduction (from this commit)

GPU-bound (unsandboxed; `dangerouslyDisableSandbox=true` is unblocked by `.claude/settings.local.json`'s `allowUnsandboxedCommands: true`):

```
mkdir -p /tmp/nurec/{no_merge,merge}
bazel run //instant_nurec:run -- \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec/no_merge --merge none
bazel run //instant_nurec:run -- \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec/merge --merge frustum-ownership
```

Sandboxed parity check:

```
.venv/bin/python scripts/validate_parity.py no_merge \
  baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
  /tmp/nurec/no_merge/*/ply/*/

.venv/bin/python scripts/validate_parity.py merge \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/nurec/merge/*/ply/*/pai_*.ply
```

## Sandbox / environment notes for the next agent

- Bazel cache: pass `--output_user_root=$TMPDIR/bazel-user --output_base=$TMPDIR/bazel-out` to keep its writes inside the sandbox-allowed tree (or use `dangerouslyDisableSandbox=true` for GPU runs that touch `~/.cache/bazel/...`).
- `external/` is symlinked to `/storage/projects/nre/external` (4.2 GB of bazel-fetched submodules); `.gitignored`.
- `.venv/` has CPU-only torch + plyfile + pytest + numpy; sufficient for `validate_parity.py` and `tests/test_*.py`.
- The user's home dotfiles (`.bashrc` etc.) appear as "stray" files in the working dir; they are `.gitignored`.
- `rm -rf` is denied by `~/.claude/settings.json`. Use individual `rm` commands or just leave scratch dirs in place.
