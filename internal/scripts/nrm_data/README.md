# NRM Data Processing Scripts

## `nvs_ncore`

Function: Render an existing reconstructed `usdz` file, and put rendered novel views into the ncore bag (v3 format).

Example command:

```bash
bazel run //internal/scripts/nrm_data:nvs_ncore -- \
    --artifact-path $DATA_DIR/last.usdz \
    --output-path $OUT_DIR/${clip_id}.zarr.itar \
    --camera-rotation-offset camera_front_wide_120fov 0 40 0 True \
    --camera-translation-offset camera_front_wide_120fov 0 0 -8 \
    --camera-force-pinhole camera_front_wide_120fov True \
    --camera-rotation-offset camera_cross_left_120fov 0 30 0 True \
    --camera-translation-offset camera_cross_left_120fov 0 0 -8 \
    --camera-force-pinhole camera_cross_left_120fov True \
    --camera-rotation-offset camera_cross_right_120fov 0 30 0 True \
    --camera-translation-offset camera_cross_right_120fov 0 0 -8 \
    --camera-force-pinhole camera_cross_right_120fov True \
    --camera-rotation-offset camera_rear_left_70fov 0 25 0 True \
    --camera-translation-offset camera_rear_left_70fov 0 0 -12 \
    --camera-force-pinhole camera_rear_left_70fov True \
    --camera-rotation-offset camera_rear_right_70fov 0 25 0 True \
    --camera-translation-offset camera_rear_right_70fov 0 0 -12 \
    --camera-force-pinhole camera_rear_right_70fov True \
    --no-difix \
    --visualize
```
