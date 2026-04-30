```{eval-rst}
.. include:: ../_includes/_global_substitutions.rst
```

# Convert Data to NCore Format

To use your real-world data in NuRec, you must first convert it to the NCore data format.

## Data Inputs

**Vehicle**

- Rig Data: Complete rig file as JSON. Sensor Calibration data
- Pose Data: Pose information with timestamps. Rig transformation matrix

**Camera**

- Image Data: JPG / PNG encoded Image Data
- Extrinsics: Camera to rig transformation
- Intrinsics: Camera lens parameters. Support for Ftheta, OpenCV Pinhole, OpenCV Fisheye
- Camera masks: JPG / PNG. Masks for car body for each camera

**Lidar (recommended)**

- Point Cloud: Point Cloud with intensity with timestamps for all frames
- Extrinsics: Lidar to rig transformation matrix

**General**

- Config: Configure HW GPU, NVDEC, and other general settings
- Session: Data for clip IDs, start/stop timestamps, alignment between camera / lidar
- Metadata: Relevant information about the data from the vehicle and the reconstruction

Follow the best practices to [Ensure Data Quality](data-quality) before you begin converting your data to NCore format.

## Create a Data Conversion Script

Use the [Waymo data conversion script](waymo-conversion-flow) as a reference implementation in building your own conversion script. Refer to the full [NCore Data Format definition](reference/conventions) and [NCore API](reference/apis/ncore) for assistance.
