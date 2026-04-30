# Phase 0 step 2 parity proof

Commit: 605871f4e5c07a56726c42d28aaf000be3dc96de
Date: 2026-04-30T15:47:24Z

## Reproduction commands

```
bazel run //nre/nrm:run -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_base_path=/storage/data/nurec/ncorev4/ \
    dataset.predict.ncore_json_list_path=/storage/data/nurec/ncorev4/debug.lst \
    dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
    out_dir=/tmp/nurec_phase0_copy/{no_merge,merge} \
    predict.primitive_merge.enabled={false,unset} \
    predict.render_video.enabled=false
```

## Parity (against baselines/original_baseline at tests/tolerance.json with all properties=0.0)

no_merge:
PASS: 2 PLY pair(s) match within tolerance.

merge:
PASS: pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.ply matches pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.ply within tolerance.

## Output sizes (bytes / vertex count)

| File            | New run                  | Baseline                 |
|-----------------|--------------------------|--------------------------|
| no_merge chunk0 | 134,626,423 / 1,748,388  | 134,626,423 / 1,748,388  |
| no_merge chunk1 | 109,972,640 / 1,428,209  | 109,972,640 / 1,428,209  |
| merge           | 221,118,521 / 2,871,662  | 221,118,521 / 2,871,662  |

Byte-for-byte identical sizes; per-property `(a-b).abs().max() == 0.0` for all
23 properties on every PLY pair.
