# Sensor-Specific Extra Signals Architecture

## Overview

The sensor-specific extra signals architecture enables different rendering behaviors for different sensor types (camera vs lidar) while maintaining efficiency and reducing memory overhead. This system replaces the previous universal extra signal approach with a more flexible, sensor-aware design.

## Key Benefits

- **Memory Efficiency**: Only allocate parameters for signals actually used by each sensor
- **Rendering Flexibility**: Different sensors can have completely different extra signal sets
- **Performance Optimization**: Avoid computing unnecessary signals for each sensor type
- **Extensibility**: Easy to add new sensor types and signal combinations

## Architecture Components

### 1. Signal Types

The system supports three types of extra signals:

#### Common Signals (`extra_signal`)

- **Purpose**: Signals used by both camera and lidar rendering
- **Examples**: Semantic labels, surface normals, depth
- **Memory**: Stored once, rendered for both sensor types
- **Configuration**: `sensor_type: "common"`

#### Camera-Specific Signals (`camera_extra_signal`)

- **Purpose**: Signals only relevant for camera/RGB rendering
- **Examples**: DINOv2 features, visual texture features, camera-specific semantics
- **Memory**: Only allocated for camera rendering pipeline
- **Configuration**: `sensor_type: "camera"`

#### Lidar-Specific Signals (`lidar_extra_signal`)

- **Purpose**: Signals only relevant for lidar/point cloud rendering
- **Examples**: Intensity, raydrop probability, reflectance
- **Memory**: Only allocated for lidar rendering pipeline
- **Configuration**: `sensor_type: "lidar"`

### 2. Packing Strategy

The system uses a two-level packing strategy:

#### Gaussian-Level Packing

Signals are packed into separate parameter tensors based on sensor type:

- `extra_signal`: [n_gaussians, common_signal_dims]
- `camera_extra_signal`: [n_gaussians, camera_signal_dims]
- `lidar_extra_signal`: [n_gaussians, lidar_signal_dims]

#### Ray-Level Packing

For rendering output, signals are repacked based on the target sensor:

- **Camera rendering**: common_signals + camera_signals
- **Lidar rendering**: common_signals + lidar_signals

## Configuration Guide

### Basic Configuration

```yaml
extra_signal:
  # Common signal available to all sensors
  semantic_logits:
    sensor_type: "common"
    n_signal_dim: 64

  # Camera-only signal
  dinov2_feats:
    sensor_type: "camera"
    n_signal_dim: 384

  # Lidar-only signal
  intensity:
    sensor_type: "lidar"
    n_signal_dim: 1
```

### Automatic Dimension Calculation

The system automatically calculates required dimensions:

```yaml
particle:
  # Computed automatically from extra_signal config
  extra_signal_dim: ${eval:'sum([v["n_signal_dim"] if v["sensor_type"] == "common" else 0 for k,v in ${..extra_signal}.items()])'}
  camera_extra_signal_dim: ${eval:'sum([v["n_signal_dim"] if v["sensor_type"] == "camera" else 0 for k,v in ${..extra_signal}.items()])'}
  lidar_extra_signal_dim: ${eval:'sum([v["n_signal_dim"] if v["sensor_type"] == "lidar" else 0 for k,v in ${..extra_signal}.items()])'}
```

### Optimizer Configuration

Each signal type can have independent learning rates:

```yaml
optimizers:
  - name: fused_adam
    params:
      extra_signal: # Common signals
        args:
          lr: 0.01
      camera_extra_signal: # Camera signals
        args:
          lr: 0.005
      lidar_extra_signal: # Lidar signals
        args:
          lr: 0.02
```

## Migration Guide

### From Old Universal System

**Old Configuration:**

```yaml
extra_signal:
  semantic_logits:
    n_signal_dim: 64
    # No sensor_type specified - used by all sensors
```

**New Configuration:**

```yaml
extra_signal:
  semantic_logits:
    sensor_type: "common" # Explicitly specify sensor compatibility
    n_signal_dim: 64
```

### Key Changes

1. **Required `sensor_type` Field**: All extra signals must now specify which sensors they target
2. **Separate Parameter Tensors**: Signals are now stored in sensor-specific parameter tensors
3. **Updated Dimension Calculations**: Config expressions now filter by sensor type
4. **New Optimizer Parameters**: Added `camera_extra_signal` and `lidar_extra_signal` optimizer configs

## Implementation Details

### Model Architecture

The `BaseGaussianModel` class now includes three extra signal parameter tensors:

```python
class BaseGaussianModel(BaseModel, ABC):
    # Common extra signals for both camera and lidar
    extra_signal: nn.Parameter | nn.UninitializedParameter

    # Camera-specific extra signals
    camera_extra_signal: nn.Parameter | nn.UninitializedParameter

    # Lidar-specific extra signals
    lidar_extra_signal: nn.Parameter | nn.UninitializedParameter
```

### Rendering Pipeline

The renderer automatically selects the appropriate signal packing based on sensor type:

```python
# Camera rendering
out_cam = self.gaussians_renderer.render(
    rendering_data=rendering_data_cam,
    gaussian_parameters=gaussian_parameters,
    n_active_features=self.get_n_active_features(),
    extra_ray_signal_infos=self.camera_extra_ray_signal_infos,  # Camera signals
    frame_meta=frame_meta_cam,
)

# Lidar rendering
out_lidar = self.gaussians_renderer.render(
    rendering_data=rendering_data_lidar,
    gaussian_parameters=gaussian_parameters,
    n_active_features=self.get_n_active_features(),
    extra_ray_signal_infos=self.lidar_extra_ray_signal_infos,   # Lidar signals
    frame_meta=frame_meta_lidar,
)
```

## Inference Defaults

When loading a trained artifact via `RenderableModel.load_from_artifact` which covers the `render` CLI, `serve-grpc`, the USDZ viewer, and the NRM NVS helpers, the gsplat camera rasterizer defaults to **CDIM=4 (RGB + depth)** or **CDIM=3 (RGB)** rather than the full CDIM=24 (RGB + 20 extra_signal + depth) the model was trained on. This is enforced by `_INFERENCE_DEFAULT_OVERRIDES` in `nre/render/render.py`, which prepends:

```
model.renderer.outputs.camera.enable_extended_features=False
```

to every call's `config_overrides`. Training is unaffected, `mode=train`/`trainval`/`val`/`test` runs through PyTorch Lightning against the in-memory model (not `load_from_artifact`), so full CDIM is preserved and `test.save_extra_signals=True` still emits `pred_semantic` during validation.

### Rationale

No inference entry point surfaces the 20 extra_signal channels through its output: the `render` CLI writes RGB JPG/PNG, gRPC streams RGB tensors, the viewer displays RGB only, NRM NVS saves RGB JPEGs. Dropping CDIM from 24 to 4 for the gsplat camera rasterizer is a measured +9% FPS at 1080p and +14% at 4K for artifacts trained with a semantic head. A runtime config flip, no retraining or artifact change required.

### Opting Back In (e.g. to render `pred_semantic`)

If your workflow needs the rasterized extra_signal tensors at inference (typically to consume `pred_semantic` from a trained semantic head) re-enable them by passing an explicit override. Later entries in `config_overrides` win the merge, so your override cleanly supersedes the default.

**Render CLI** (any positional arg after the recognized flags is forwarded as a Hydra override to `load_from_artifact`):

```bash
bazel run //:run -- render \
  --artifact-path /path/to/artifact.usdz \
  --output-dir /output \
  --camera-id camera_front_wide_120fov \
  model.renderer.outputs.camera.enable_extended_features=True
```

**Programmatic callers** (gRPC service, custom renderer scripts):

```python
model = RenderableModel.load_from_artifact(
    artifact,
    enable_nrend=False,
    config_overrides=("model.renderer.outputs.camera.enable_extended_features=True",),
)
```

Re-enabling the override puts the 20 extra_signal channels back into `GaussiansRenderReturn.extra_ray_signals` on every render call. Note that no inference CLI today writes those tensors to disk. The built-in `render` command only saves `color_image`/`distance_image`/`opacity_image`. To surface `pred_semantic` as an artifact output, a caller must read `extra_ray_signals` explicitly (as the Lightning validation path does via `test.save_extra_signals=True` in `nre/systems/gaussians.py`).

## Best Practices

### Signal Type Selection

- Use `"common"` for signals that benefit both sensor types (e.g., semantic labels)
- Use `"camera"` for visual/appearance features (e.g., DINOv2, texture features)
- Use `"lidar"` for physical properties (e.g., intensity, reflectance, raydrop)

### Performance Optimization

- Keep common signals minimal to avoid unnecessary computation
- Use sensor-specific signals for specialized features
- Consider memory usage when designing signal dimensions

### Debugging

- Check signal dimensions match expected values using the automatic calculation expressions
- Verify sensor_type values are correctly specified in configs
- Use logging to trace signal packing during initialization

## Example Configurations

### Semantic Segmentation Setup

```yaml
extra_signal:
  semantic_logits:
    sensor_type: "common" # Both sensors can benefit from semantics
    n_signal_dim: 20 # Number of classes
```

### Multi-Modal Setup

```yaml
extra_signal:
  # Shared semantic information
  semantic_logits:
    sensor_type: "common"
    n_signal_dim: 20

  # Camera-specific visual features
  dinov2_feats:
    sensor_type: "camera"
    n_signal_dim: 384

  # Lidar-specific physical properties
  intensity:
    sensor_type: "lidar"
    n_signal_dim: 1
  raydrop:
    sensor_type: "lidar"
    n_signal_dim: 1
```

This architecture provides a solid foundation for building sophisticated multi-sensor rendering systems while maintaining efficiency and extensibility.
