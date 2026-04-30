# 3D Asset Harvesting Inference Pipeline

## Overview

This pipeline enables you to harvest 3D Gaussian assets (.ply) from NCore data.

## Prerequisites

### NGC Org

You will need to be a member of the nvidian/ct-toronto-ai (org/team) to download the required models.

You can request access via the slack channel #swngc-help

### NGC API Key Setup

1. Get your Personal API key for nvidian/ct-toronto-ai from [NGC website](https://org.ngc.nvidia.com/setup/api-keys)
2. Set it as an environment variable:
   ```bash
   export NGC_API_KEY=your_key_here
   ```

### Model Cache Setup

Create a directory to cache downloaded models:

```bash
mkdir -p ~/.cache/nre
```

### Sample Data

Download the sample dataset: [clipgt-1ea7dc88-88ed-4c91-81fe-b6eb489cfa71.zip](https://drive.google.com/file/d/1bhQVY3QVWy7HTwAN-P_C2WNEuPrMJBD5/view?usp=sharing)

## Building

### Build Options

```bash
# Build Python binary
bazel build //apps/asset_harvester:asset_harvester

# Build container image
bazel run //apps:load_nre_tools_image_oci --config=asset_harvester
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

> **NCore V3 → V4**: This pipeline requires NCore V4 component stores as input. For V3 data, pull and run the conversion tool first: `ngc registry resource download-version "nvidia/nre/nre-ncore:ncore_3to4"`

### Python Binary Execution (Recommended)

> **Note:** For Python binary execution, please refer to the [Prerequisites](../../../README.md#prerequisites) section in the main README. Users on older Ubuntu versions may encounter GLIBC issues.

Make sure the NGC_API_KEY environment variable is set before running the binary:

```bash
export NGC_API_KEY=your_key_here
```

```bash
bazel run //apps/asset_harvester:asset_harvester -- \
    --component-store="path/to/component-store.zarr.itar" \
    --output-dir="path/to/output" \
    --track-ids="track_id1,track_id2" \
    --cache-dir="~/.cache/nre" \
    ncore_parser.camera_ids=["camera_front_wide_120fov"]
```

Each run stores a `metadata.yaml` file. This contains the runtime config + output of the view extraction step. You can pass this file as input via `--metadata-file`, the ncore component store is not required if using this file.

```bash
bazel run //apps/asset_harvester:asset_harvester -- \
    --metadata-file="path/to/metadata.yaml" \
    --output-dir="path/to/output" \
    --track-ids="track_id1,track_id2" \
    --cache-dir="~/.cache/nre" \
    ncore_parser.camera_ids=["camera_front_wide_120fov"]
```

### Container Deployment

> **Note:** Make sure you've built the container image following the steps above

Sample command with config overrides (config overrides are optional)

```bash
docker run -it --rm \
    --gpus=all \
    -v /path/to/output:/output \
    -v /path/to/data:/data \
    -v ~/.cache/nre:/cache \
    -e NGC_API_KEY=${NGC_API_KEY} \
nvcr.io/nvidian/ct-toronto-ai/nre_tools:latest \
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

### Unit Test

You can run the AH Unit Tests via

```bash
bazel run //apps/asset_harvester:asset_harvest_test
```

To specify a custom cache directory (instead of the default temporary directory), use the `ASSET_HARVESTER_CACHE_DIR` environment variable:

```bash
bazel test //apps/asset_harvester:asset_harvest_test --test_env=ASSET_HARVESTER_CACHE_DIR=/path/to/your/cache/dir
```

This is useful for persisting cached models between test runs or using a shared cache location.

To specify a custom output directory (instead of the default temporary directory), use the `ASSET_HARVESTER_OUTPUT_DIR` environment variable. Since bazel tests run in a sandbox, you must also use `--sandbox_writable_path` to allow writes to the output directory:

```bash
bazel test //apps/asset_harvester:asset_harvest_test \
  --test_env=ASSET_HARVESTER_CACHE_DIR=/tmp/ah_cache \
  --test_env=ASSET_HARVESTER_OUTPUT_DIR=/tmp/ah_output \
  --sandbox_writable_path=/tmp/ah_output
```

Outputs will be written to `<output_dir>/3dgs/` subdirectory.

## Replacing Assets in NRE

This section describes how to replace assets in a trained NRE artifact using harvested 3D Gaussian assets.

### Step 1: Package Assets into a USDZ

Use the `export-external-assets` command to merge harvested assets into an artifact. You can use the asset harvester output directly.

**Using the CLI:**

```bash
bazel run //:run -- export-external-assets \
    --artifact-path /path/to/original.usdz \
    --external-assets-path /path/to/asset_harvester_output \
    --output-artifact-path /path/to/output_with_assets.usdz \
    --output-edit-file /path/to/edit-assets.json
```

**Using Docker:**

```bash
docker run -it --rm \
    --gpus=all \
    -v /path/to/artifact:/artifact \
    -v /path/to/asset_harvester_output:/external_assets \
    -v /path/to/output:/output \
    nvcr.io/nvidia/nre/nre:latest export-external-assets \
    --artifact-path /artifact/original.usdz \
    --external-assets-path /external_assets \
    --output-artifact-path /output/output_with_assets.usdz \
    --output-edit-file /output/edit-assets.json
```

This command will:

1. Create a new artifact with the external assets in the correct structure (`external_assets/{track_id}/gs.ply`)
2. Generate an `edit-assets.json` file with the valid track IDs for replacement

### Step 2: Render with Replaced Assets

Use the generated `edit-assets.json` file to render with replaced assets.

**Configure the edit file:**

The generated `edit-assets.json` contains a `replace` field that maps original track IDs to replacement track IDs. Modify it as needed:

```json
{
  "replace": [
    {
      "original_id": "20",
      "replacement_id": "20"
    },
    {
      "original_id": "8",
      "replacement_id": "8"
    }
  ]
}
```

**Start the gRPC server:**

Spin up the gRPC server with the output USDZ from `export-external-assets`:

```bash
bazel run //:run -- serve-grpc \
    --artifact-glob /path/to/output_with_assets.usdz \
    --port 8080 \
    --enable-editing-actors
```

**Render with asset replacement:**

Use `render-grpc` with the `--edit-assets` flag to render with the replaced assets:

```bash
bazel run //:run -- render-grpc \
    --edit-assets /path/to/edit-assets.json \
    # ... other render options
```

## Common Errors

### CUDA Device Not Available

**Error Message:**

```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

**Solution:**
Add `--gpus=all` to the bazel run command when running the asset harvester image.

### CUDA_ERROR_UNSUPPORTED_PTX_VERSION

**Error Message:**

```
CUDA_ERROR_UNSUPPORTED_PTX_VERSION
```

**Solution:**
Your device driver must be 550 or higher or make sure you are specifying bazel to build / run against cuda 12-4 libraries. You can do this by create a .bazelrc.user file in the root of the NRE repository with the following content:

```
build --repo_env=CUDA_PATH=/usr/local/cuda-12.4
run --repo_env=CUDA_PATH=/usr/local/cuda-12.4
```

## Technical Details

### Gaussian Visualization

We support a single Gaussian type: 3DGS
