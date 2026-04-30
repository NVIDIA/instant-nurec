# 3D Asset Harvesting Inference Pipeline

## Overview

This pipeline enables you to harvest 3D Gaussian assets (.ply) from NCore data.

## Prerequisites

### NGC API Key Setup

1. Get your Personal API key from [NGC website](https://org.ngc.nvidia.com/setup/api-keys)
2. Set it as an environment variable:
   ```bash
   export NGC_API_KEY=your_key_here
   ```

### Model Cache Setup

Create a directory to cache downloaded models:

```bash
mkdir -p ~/.cache/nre
```

## Running the Asset Harvester

The default config_path passed to the harvester is `configs/experimental/asset_harvesting/harvest.yaml`.

Notable config overrides:

- `ncore_parser.camera_ids`: List of camera IDs for view extraction.
  Default uses all available cameras:
  - "camera_cross_left_120fov"
  - "camera_cross_right_120fov"
  - "camera_front_wide_120fov"
  - "camera_rear_right_70fov"
  - "camera_rear_left_70fov"
- `tokengs_lifting.use_ttt`: Enable test-time training for higher quality (default: false)
- `tokengs_lifting.bbox_size`: Bounding box size for lifting (default: 1.0)

### Container Deployment

Sample command with config overrides (config overrides are optional)

```bash
docker run -it --rm \
    --gpus=all \
    -v /path/to/output:/output \
    -v /path/to/data:/data \
    -v ~/.cache/nre:/cache \
    -e NGC_API_KEY=${NGC_API_KEY} \
nvcr.io/nvidia/nre/nre-tools:latest \
    asset-harvester \
    --component-store="/data/component-store.zarr.itar" \
    --output-dir="/output" \
    --track-ids="track_id1,track_id2" \
    --cache-dir="/cache" \
    ncore_parser.camera_ids=["camera_front_wide_120fov"]
```

The container will automatically:

1. Download required model checkpoints from NGC if they don't exist in the cache directory
2. Use cached models if they exist, avoiding redundant downloads
3. Store outputs in the specified output directory

## Common Errors

### CUDA Device Not Available

**Error Message:**

```bash
  RuntimeError: Attempting to deserialize object on a CUDA device but
  torch.cuda.is_available() is False. If you are running on a CPU-only
  machine, please use torch.load with map_location=torch.device('cpu')
  to map your storages to the CPU.
```

**Solution:**
Add `--gpus=all` to the bazel run command when running the asset harvester image.
