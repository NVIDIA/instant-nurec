# Algorithm Probing and Test Data Generation

## Prober Integration:

The prober can be integrated throughout the NRE pipeline. For example, in `BilateralGridPerCamera` and `BilateralGridPerFrame` post-processing:

```python
# Example from nre/models/post_processing.py
def forward(self, results, rays_cam, rays_cam_meta, ...):
    # ... processing logic ...

    # Apply bilateral grid transformation
    coords_xy = rays_cam_meta.pixel_idxs / (rays_cam_meta.image_res - 1)
    transformed_rgb = self.bilateral_grid(rgb, coords_xy, unique_sensor_idx)
    results.rendered_cam.rgb = transformed_rgb

    # Prober integration - saves tensors for testing
    prober(
        global_step,
        "bilateral_grid_per_camera",
        rgb=rgb,
        coords_xy=coords_xy,
        unique_sensor_idx=rays_cam_meta.unique_sensor_idx,
        bilateral_grid=self.bilateral_grid.grid,
        output_rgb=results.rendered_cam.rgb,
    )
```

## Generating Test Data:

To probe an algorithm and generate test data for validation, you can use the dedicated `test_data:generate` command. This workflow involves running the NRE pipeline with prober enabled to capture tensor data at specific steps for later test consumption.

```
# Generate test data using the test_data:generate command
bazel run //internal/scripts/test_data:generate_test_data -- \
  --test-data-dir $(realpath test_data) \
  --dataset-path <path-to-dataset> \
  --config-name apps/prod/Hyperion-8.1/car2sim.yaml \
  --n-samples-per-epoch 100 \
  --every-n-steps 100
```

It will generate a tar.gz file in the test_data directory, you can upload it to the GitLab Package Registry if needed.

```
export GITLAB_TOKEN=$(awk '/machine.*gitlab/ {found=1} found && /login/ {login=$2} found && /password/ {print $2; exit}' ~/.netrc)
curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
     --upload-file ./test_data_prober_generated.tar.gz \
     "https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/test_data_prober_generated/0.1.3/test_data_prober_generated.tar.gz"
```

## Consuming Test Data in Tests:

Once test data is generated, you can extract it and consume it in your test suite using the prober's built-in loading utilities:

To use a custom test data directory, set the `NRE_PROBER_DIR` environment variable to the path of the test data directory.

```
tar xvzf test_data_prober_generated.tar.gz test_data
NRE_PROBER_DIR=$(realpath test_data) bazel run //nre/models:bilateral_grid_cuda_test
```

This will be useful if you add probes to the codebase and want to test the code with the new data before uploading it to the GitLab Package Registry.

````python

```python
import pytest
import torch
from nre.utils.prober import prober_test_decorator, TRUE_FALSE, ProberDataSet, ProberTestResult

# This decorator will ensure the test will be executed on each snapshot of the test data
@prober_test_decorator(
    snapshot_set_name="bilateral_grid_per_camera",
    test_args_combinations=TRUE_FALSE
)
def test_bilateral_grid_forward_backward(data: ProberDataSet, use_cuda: bool):
    """Test bilateral grid forward and backward passes."""

    # Extract tensors from the probed data
    rgb_input = data["rgb"]
    pixel_idxs = data["pixel_idxs"]
    image_res = data["image_res"]
    grid_idcs = data["unique_sensor_idx"]
    bilateral_grid = data["bilateral_grid"]

    # Your test logic here
    # ...

    return ProberTestResult(f"BilateralGrid {'CUDA' if use_cuda else 'PyTorch'} test passed")

````

This approach ensures that algorithm changes are validated against real world data.
