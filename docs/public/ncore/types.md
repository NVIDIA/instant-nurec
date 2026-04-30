# NCore Data Types

Use these tables to understand the different data types in the NCore data format.

## Pose Data

| Category | Parameter                 | Type    | Shape   | Description                                  |
| -------- | ------------------------- | ------- | ------- | -------------------------------------------- |
| Poses    | T_rig_world_base          | float64 | [4,4]   | Base transformation matrix from rig to world |
| Poses    | T_rig_worlds              | float64 | [N,4,4] | All rig-to-world transformations             |
| Poses    | T_rig_world_timestamps_us | uint64  | [N]     | Timestamps for transformations               |
| Poses    | global_start_timestamp_us | int     |         | Start timestamp of global range              |
| Poses    | global_end_timestamp_us   | int     |         | End timestamp of global range                |
| Poses    | T_world_sensorRef         | float32 | [4,4]   | World to sensor reference transformation     |

## Camera

### Common Camera

| Category      | Parameter           | Type    | Shape  | Description                          |
| ------------- | ------------------- | ------- | ------ | ------------------------------------ |
| Common Camera | image               | bytes   | varies | JPEG/PNG encoded image data          |
| Common Camera | T_sensor_rig        | float32 | [4,4]  | Camera to rig transformation         |
| Common Camera | resolution          | uint32  | [2]    | Image width and height               |
| Common Camera | frame_timestamps_us | uint64  | [J]    | Frame timestamps                     |
| Common Camera | shutter_type        | str     | scalar | Camera shutter type (ROLLING/GLOBAL) |

### FTheta Camera

| Category      | Parameter               | Type    | Shape  | Description                                                       |
| ------------- | ----------------------- | ------- | ------ | ----------------------------------------------------------------- |
| FTheta Camera | principal_point         | float32 | [2]    | Principal point coordinates (u,v)                                 |
| FTheta Camera | reference_poly          | str     | scalar | Reference polynomial type (PIXELDIST_TO_ANGLE/ANGLE_TO_PIXELDIST) |
| FTheta Camera | pixeldist_to_angle_poly | float32 | [6]    | Backward distortion polynomial coefficients                       |
| FTheta Camera | angle_to_pixeldist_poly | float32 | [6]    | Forward distortion polynomial coefficients                        |
| FTheta Camera | max_angle               | float32 | scalar | Maximum ray angle with principal direction                        |
| FTheta Camera | linear_cde              | float32 | [3]    | Linear transform coefficients                                     |

### OpenCV Pinhole

| Category       | Parameter         | Type    | Shape | Description                                        |
| -------------- | ----------------- | ------- | ----- | -------------------------------------------------- |
| OpenCV Pinhole | principal_point   | float32 | [2]   | Principal point coordinates (u,v)                  |
| OpenCV Pinhole | focal_length      | float32 | [2]   | Focal lengths in u and v direction                 |
| OpenCV Pinhole | radial_coeffs     | float32 | [6]   | Radial distortion coefficients [k1,k2,k3,k4,k5,k6] |
| OpenCV Pinhole | tangential_coeffs | float32 | [2]   | Tangential distortion coefficients [p1,p2]         |
| OpenCV Pinhole | thin_prism_coeffs | float32 | [4]   | Thin prism distortion coefficients [s1,s2,s3,s4]   |

### OpenCV Fisheye

| Category       | Parameter       | Type    | Shape  | Description                                   |
| -------------- | --------------- | ------- | ------ | --------------------------------------------- |
| OpenCV Fisheye | principal_point | float32 | [2]    | Principal point coordinates (u,v)             |
| OpenCV Fisheye | focal_length    | float32 | [2]    | Focal lengths in u and v direction            |
| OpenCV Fisheye | radial_coeffs   | float32 | [4]    | Fisheye distortion coefficients [k1,k2,k3,k4] |
| OpenCV Fisheye | max_angle       | float32 | scalar | Maximum ray angle with principal direction    |

## Lidar

| Category | Parameter      | Type    | Shape  | Description                                |
| -------- | -------------- | ------- | ------ | ------------------------------------------ |
| Lidar    | xyz_s          | float32 | [N,3]  | Start point coordinates                    |
| Lidar    | xyz_e          | float32 | [N,3]  | End point coordinates (motion compensated) |
| Lidar    | intensity      | float32 | [N]    | Normalized intensity [0.0-1.0]             |
| Lidar    | timestamp_us   | uint64  | [N]    | Point timestamps                           |
| Lidar    | T_sensor_rig   | float32 | [4,4]  | Lidar to rig transformation                |
| Lidar    | frame_labels   | object  | varies | Frame-specific label data                  |
| Lidar    | model_elements | uint16  | [N,2]  | Row/column indices in lidar model          |

## Radar

| Category | Parameter           | Type    | Shape | Description                 |
| -------- | ------------------- | ------- | ----- | --------------------------- |
| Radar    | xyz_s               | float32 | [N,3] | Start point coordinates     |
| Radar    | xyz_e               | float32 | [N,3] | End point coordinates       |
| Radar    | T_sensor_rig        | float32 | [4,4] | Radar to rig transformation |
| Radar    | frame_timestamps_us | uint64  | [K]   | Frame timestamps            |

## Metadata

| Category | Parameter        | Type | Shape  | Description                |
| -------- | ---------------- | ---- | ------ | -------------------------- |
| Metadata | version          | str  | scalar | Dataset version            |
| Metadata | egomotion_type   | str  | scalar | Type of ego-motion used    |
| Metadata | calibration_type | str  | scalar | Type of sensor calibration |
| Metadata | sequence_id      | str  | scalar | Source dataset identifier  |
| Metadata | shard_id         | int  | scalar | Current shard index        |
| Metadata | shard_count      | int  | scalar | Total number of shards     |

## Camera Mask

| Category    | Parameter | Type | Shape | Description                                                         |
| ----------- | --------- | ---- | ----- | ------------------------------------------------------------------- |
| Camera Mask | image     |      |       | constant mask image, which currently only contains the ego car mask |

## Configuration Parameters

| Category | Parameter             | Type  | Shape | Description                              |
| -------- | --------------------- | ----- | ----- | ---------------------------------------- |
| Config   | seek_sec              | float |       | Seek time in seconds                     |
| Config   | duration_sec          | float |       | Duration time in seconds                 |
| Config   | seek_camera           | bool  |       | Whether to seek camera frames            |
| Config   | camera_use_nvimgcodec | bool  |       | Whether to use NVIDIA codec              |
| Config   | camera_quality        | int   |       | JPEG encoding quality                    |
| Config   | camera_gpu            | bool  |       | Whether to use GPU for camera processing |
| Config   | debug                 | bool  |       | Debug mode flag                          |

## Session/Clip Information

| Category | Parameter                  | Type | Shape | Description             |
| -------- | -------------------------- | ---- | ----- | ----------------------- |
| Session  | clip_id                    | str  |       | Unique clip identifier  |
| Session  | session_id                 | str  |       | Session identifier      |
| Session  | session_start_timestamp_us | int  |       | Session start timestamp |
| Session  | session_end_timestamp_us   | int  |       | Session end timestamp   |

## NVIDIA Specific

| Category        | Parameter                   | Type | Shape  | Description                 |
| --------------- | --------------------------- | ---- | ------ | --------------------------- |
| NVIDIA Specific | camera_quality              | int  | scalar | Camera encoding quality     |
| NVIDIA Specific | camera_use_nvimgcodec       | bool | scalar | Whether to use NVIDIA codec |
| NVIDIA          | lidar_column_spin_alignment | str  | scalar | Alignment metric type       |

## Rig and Calibration

| Category | Parameter                | Type            | Shape | Description                          |
| -------- | ------------------------ | --------------- | ----- | ------------------------------------ |
| Rig      | rig                      | dict            |       | Complete rig configuration from JSON |
| Rig      | dw_rig                   | object          |       | DriveWorks rig object                |
| Rig      | sensors_calibration_data | dict            |       | Sensor calibration data              |
| Rig      | constants                | NvidiaConstants |       | NVIDIA-specific constants            |
