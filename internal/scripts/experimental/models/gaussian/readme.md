# NVHuman Gaussian Animation Pipeline

Pipeline for animating 3D Gaussian Splatting models using NVHuman skeletal rigging and GenMO's HMR4D for pose extraction.

## Quick Start

Run the animation pipeline:

```bash
bazel run //internal/scripts/experimental/models/gaussian:animate_gaussians -- \
  --input-ply /path/to/gaussian.ply \
  --output-dir /path/to/output \
  --target-poses /path/to/poses.pt \
  --static-cam
```

> **Note:** Currently only `--static-cam` mode is supported. Dynamic camera support with DROID-SLAM requires additional external dependencies and will be added in a future MR.

## Directory Structure

```
gaussian/
├── animate_gaussians.py       # End-to-end animation pipeline (orchestrator)
├── genmo/                      # GenMO integration utilities
│   ├── genmo_init.py          # GenMO PROJ_ROOT initialization for Bazel
│   └── pose_extraction.py     # HMR4D pose extraction pipeline
└── nvhuman/                    # NVHuman Gaussian animation
    ├── gaussian_to_nvhuman.py     # Gaussian → NVHuman conversion
    ├── gaussian_nvhuman_layer.py  # Extended NVHuman with Gaussian support
    ├── animate_nvhuman.py     # Apply poses to rigged models
    └── tools/
        ├── render_ply.py      # PLY visualization
        ├── render_nvhuman.py  # NVHuman rendering
        └── utils.py           # Shared utilities
```

### Component Overview

- **`animate_gaussians.py`**: Top-level end-to-end pipeline that orchestrates all stages (PLY rendering, pose extraction, Gaussian→NVHuman conversion, animation).
- **`genmo/`**: Shared GenMO utilities for pose extraction and initialization. Can be reused by other projects that need HMR4D functionality.
- **`nvhuman/`**: NVHuman-specific code for binding Gaussians to skeletal rigs and animating them.

## Common Options

### Required

| Option         | Description                          |
| -------------- | ------------------------------------ |
| `--input-ply`  | Input Gaussian PLY file              |
| `--output-dir` | Output directory for all results     |
| `--static-cam` | Use static camera (required for now) |

### Optional

| Option           | Default | Description                                                                         |
| ---------------- | ------- | ----------------------------------------------------------------------------------- |
| `--target-poses` | None    | HMR4D pose sequence (.pt file). If not provided, extracts poses from rendered video |
| `--fps`          | 30      | Output video frame rate                                                             |
| `--output-size`  | 512     | Output image resolution                                                             |
| `--start-frame`  | 0       | Starting frame index for animation                                                  |
| `--end-frame`    | -1      | Ending frame index (-1 for all frames)                                              |
| `--frame-step`   | 1       | Frame step size for animation                                                       |
| `--elevation`    | 0       | Camera elevation angle (degrees)                                                    |
| `--distance`     | 1.5     | Camera distance from subject                                                        |
| `--fov`          | 70      | Field of view (degrees)                                                             |
| `--save-frames`  | True    | Save individual animation frames                                                    |
| `--save-video`   | True    | Save final animation video                                                          |

## Technical Details

### GenMO Integration

The pipeline uses GenMO's HMR4D model for human pose and shape estimation:

- **Pose Extraction**: Automatically extracts human poses from input videos using:

  - Bounding box tracking (YOLO)
  - VitPose for 2D keypoint detection
  - ViT features for appearance encoding
  - HMR4D model for 3D pose/shape estimation in NVHuman format

- **Initialization**: `genmo_init.py` ensures GenMO's project root is correctly configured for Bazel runfiles, allowing proper checkpoint and data file loading.
