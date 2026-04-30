# Phase 1 step 3 parity proof

Commit: 8bf94935eb93ea901833af4350b861bc7356413e
Date: 2026-04-30T15:55:11Z

## New CLI invocation (bazel-launched)

```
bazel run //instant_nurec:run -- \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/nurec_step3/{no_merge,merge} \
    --merge {none,frustum-ownership} \
    --log-level INFO
```

## Parity (against baselines/original_baseline at tests/tolerance.json)

no_merge:
PASS: 2 PLY pair(s) match within tolerance.

merge:
PASS: pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.ply matches pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.ply within tolerance.

## Notes

- The CLI translates --merge {none,frustum-ownership} into the corresponding
  predict.primitive_merge.{enabled,overlap_strategy} Hydra overrides.
- All other Hydra overrides (config-name, ncore_json_*, lidar_id, render_video=false,
  +nrm/apps/options=_kelvin_predict) are baked in to match nre_example_call.sh.
- nre/nrm/BUILD.bazel pylib visibility was widened to //visibility:public so the new
  //instant_nurec:pylib can depend on it (self-invented: NRE never needed this).
