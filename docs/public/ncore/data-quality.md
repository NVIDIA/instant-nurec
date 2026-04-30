# Ensure Data Quality

To generate the highest quality simulations using NVIDIA Neural Reconstruction, you must first ensure that the quality of your input data is high. Follow these best practices to get the best output using NuRec.

## Cameras

- **Camera Calibration Accuracy:**

  - Accurate camera model / distortions with the actual FOV matching the expected FOV
  - Provided per sensor calibration (both extrinsics and intrinsics) need to be estimated for the _actual_ sensors on the vehicle (no per-carline / nominal values that were not calibrated for the vehicle)
  - Accuracy Goals:
    - Extrinsics: \< 0.5deg relative orientation errors, \< 2cm relative translation errors, relative to vehicle coordinate system
    - Intrinsics: \< 1px reprojection errors across the whole image domain\]

- **Image Data Spatial / Temporal Resolution:**

  - _Original_ sensor resolution, _full_ temporal frame-rate of the sensor
  - No undistortion or rectification of the image frames, provide distorted images (faithful to the original sensor’s raw data), with the distortion modeled accurately by the associated intrinsic camera model

- **Image Masks:**

  - Require binary masks of all the non-ego-vehicle pixels for each sensor

- **Timestamps, rolling-shutter timing and direction:**

  - Provide highest resolution timestamps (millisecond) per camera frame (provide per frame timestamps explicitly, don’t assume implicitly synchronized sensors)
  - Accurate per-sensor _start-of-frame_ and _end-of-frame_ timestamps in milliseconds are strictly required (computed from per-imager rolling-shutter offset and delay timings)
  - Per-sensor _rolling-shutter direction_ (top-down, bottom-up, left-right, or right-left) is required

- **Accuracy of Camera Model and Parameters:**

  - Provide a _full_ formal mathematical model for the employed intrinsic camera models at hand (or full implementation, or reference standard models like OpenCV Pinhole / Fisheye where applicable). In particular, provide image-domain coordinate system convention related to how pixel indices are mapped to the continuous image-domain (center of first pixel coordinates)

  - If sensors are behind a windshield, also provide optical distortion model of the windshield as well as the associated model parameters that were used while calibrating the sensors

- **Camera Calibration Data Format:**

  - Ideally formatted in an easy-to-parse format (e.g. as JSON), not requiring pickling or any internal data-types

    - example JSON structure

    ```
    {
        "name": "camera_one"
        "intrinsics": {
            "camera_model": <string>,
            "fx": <float32>,
            "fy": <float32>,
            "cx": <float32>,
            "cy": <float32>,
            "k1": <float32>,
            "k2": <float32>,
            "k3": <float32>,
            "p1": <float32>,
            "p2": <float32>,
            "fov_degrees": <float32>,
            <other intrinsic parameters>...
        },
        "extrinsics": {
            "stm": <float32>
            <other extrinsic parameters>...
        }
    }

    ```

- **Consistent Coloration** ideally with as little glare/occlusion as possible

## Egomotion Trajectory

- **Resolution and Accuracy:**

  - Provide vehicle coordinate system poses relative to the scene / world at highest possible temporal resolution (millisecond) accuracy

  - Relative pose coordinates need to be as accurate at possible, accuracy goals: \< 0.5deg relative consecutive orientation errors, \< 2.5cm relative consecutive translation errors

- **Egomotion Temporal Range:**

  - Egomotion poses need to cover the full range of sensor frame timestamps (i.e., cover _both_ start and end frame timestamps), otherwise sensor frames will need to be dropped

## Object Bounding Boxes

- **Resolution and Accuracy:**

  - Provide timestamped object bounding box poses relative to the scene / world at highest possible temporal resolution (millisecond) accuracy
  - Timestamps need to represent the absolute time of observation of the object at the given pose without any offsets
  - Absolute object bounding box pose coordinates relative to the scene need to be as accurate at possible, accuracy goals: \< 1deg relative orientation errors, \< 5cm translation errors
  - Bounding box dimensions need to accurately reflect the static extent of the associated object, accuracy goal: \< 1deg orientation errors, \< 5cm dimension extent errors

- **Conventions:**

  - Object poses should represent the frame located at the center of the bounding aligned with the boxes principal directions, without any offsets
  - Timestamped bounding box tracks need to be provided for all visible and distinct objects in the scene, and be labeled with categorical labels (vehicle, pedestrian, sign, …) and unique object identifiers
