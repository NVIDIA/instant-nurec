# Sensors Library

GPU-accelerated camera and LiDAR projection with differentiable kernels.

## Architecture

The library is organized into two layers:

| Layer       | Location   | Purpose                                                                       |
| ----------- | ---------- | ----------------------------------------------------------------------------- |
| **Layer 0** | `kernels/` | GPU kernels (Slang) with Python bindings. Stateless projection operations.    |
| **Layer 2** | `models/`  | Stateful `nn.Module` wrappers with learnable parameters and frame structures. |

```
libs/sensors/
├── kernels/
│   ├── cameras/      # Camera projection kernels
│   ├── lidars/       # LiDAR projection kernels
│   ├── common/       # Pose, trajectory types
│   └── pose_calib/   # Pose calibration kernels
└── models/
    ├── cameras/      # CameraModel, ImageFrame
    ├── lidars/       # LidarModel, LidarFrame
    └── common/       # Base Frame, return types
```

---

## Coordinate Systems & Conventions

### Camera Coordinate System (OpenCV Convention)

All camera models follow the [OpenCV coordinate conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html):

![Camera Coordinate Convention](resources/camera_convention.jpg)

- **Origin**: Camera optical center
- **X-axis**: Points right
- **Y-axis**: Points down
- **Z-axis**: Points forward along the optical axis (into the scene)

**Image coordinates** `(u, v)`:

- `u`: Horizontal pixel coordinate (0 at top-left, increases rightward)
- `v`: Vertical pixel coordinate (0 at top-left, increases downward)
- **Principal point**: Where optical axis (Z) intersects the image plane

### LiDAR Coordinate System (Spherical Convention)

LiDAR uses a right-handed coordinate system with spherical angles:

```
           Z (up)
           ↑
           │   Y (left)
           │  ╱
           │ ╱
           │╱
           ●───────────→ X (forward, azimuth = 0)
          ╱
         ╱ elevation: angle from XY plane toward Z
        ╱  azimuth: angle in XY plane from X toward Y
```

- **X-axis**: Points forward (reference direction for azimuth = 0)
- **Y-axis**: Points left (positive azimuth direction)
- **Z-axis**: Points up (positive elevation direction)

**Spherical angles** `(elevation, azimuth)`:

- **Elevation**: Angle from XY plane (positive toward +Z, negative toward -Z)
- **Azimuth**: Angle in XY plane from +X toward +Y (counter-clockwise when viewed from above)

**Conversion formulas**:

```
Spherical → Cartesian:
    x = cos(elevation) × cos(azimuth)
    y = cos(elevation) × sin(azimuth)
    z = sin(elevation)

Cartesian → Spherical:
    elevation = atan2(z, √(x² + y²))
    azimuth   = atan2(y, x)
```

---

## Camera Models

### OpenCV Pinhole (`OpenCVPinholeCameraModel`)

OpenCV's Standard pinhole camera model with rational radial, tangential, and thin prism distortion coefficients.

**Projection** (given camera ray `(X, Y, Z)`):

```
1. Perspective projection:     x = X/Z,  y = Y/Z

2. Compute radial distance:    r² = x² + y²

3. Radial distortion factor:   k_radial = (1 + k₁r² + k₂r⁴ + k₃r⁶) / (1 + k₄r² + k₅r⁴ + k₆r⁶)

4. Tangential distortion:      δx_tan = 2p₁xy + p₂(r² + 2x²)
                               δy_tan = p₁(r² + 2y²) + 2p₂xy

5. Thin prism distortion:      δx_prism = s₁r² + s₂r⁴
                               δy_prism = s₃r² + s₄r⁴

6. Combine distortions:        x' = x × k_radial + δx_tan + δx_prism
                               y' = y × k_radial + δy_tan + δy_prism

7. Apply intrinsics:           u = fx × x' + cx
                               v = fy × y' + cy
```

**Parameters**:

- `focal_length`: `(fx, fy)` - Focal lengths in pixels
- `principal_point`: `(cx, cy)` - Principal point in pixels
- `radial_coeffs`: `[k1, k2, k3, k4, k5, k6]` - Radial distortion (rational model)
- `tangential_coeffs`: `[p1, p2]` - Tangential (decentering) distortion
- `thin_prism_coeffs`: `[s1, s2, s3, s4]` - Thin prism distortion

### OpenCV Fisheye (`OpenCVFisheyeCameraModel`)

OpenCV’s fisheye camera model with polynomial distortion for ultra-wide angle and fisheye lenses.

**Projection** (given camera ray `(X, Y, Z)`):

```
1. Perspective projection:     a = X/Z,  b = Y/Z
                               r = √(a² + b²)

2. Compute incident angle:     θ = atan(r)
   (angle from optical axis)

3. Apply distortion:           θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)

4. Distorted coordinates:      x' = (θ_d / r) × a
                               y' = (θ_d / r) × b
   (when r → 0: x' → a, y' → b)

5. Apply intrinsics:           u = fx × x' + cx
                               v = fy × y' + cy
```

**Parameters**:

- `focal_length`: `(fx, fy)` - Focal lengths in pixels
- `principal_point`: `(cx, cy)` - Principal point in pixels
- `forward_poly`: `[k1, k2, k3, k4]` - Distortion polynomial coefficients

### F-Theta (`FThetaCameraModel`)

NVIDIA’s FTheta camera model with polynomial distortion parameterization, suitable for both perspective and wide field-of-view cameras.

**Projection** (given camera ray `(X, Y, Z)`):

```
1. Normalize ray:              (x, y, z) = normalize(X, Y, Z)

2. Compute incident angle:     θ = acos(z)
   (angle from optical axis)

3. Compute radial distance:    r = fw_poly(θ)  [if FORWARD reference]
                               r = invert(bw_poly)(θ)  [if BACKWARD reference]

4. Compute 2D offset:          scale = r / √(x² + y²)
                               offset = (x × scale, y × scale)

5. Apply affine transform:     (x', y') = A × offset

6. Apply principal point:      u = x' + cx
                               v = y' + cy
```

**Polynomial types**:

- `FORWARD`: Reference polynomial maps angle → radius (`r = fw_poly(θ)`)
- `BACKWARD`: Reference polynomial maps radius → angle (`θ = bw_poly(r)`)

**Parameters**:

- `principal_point`: `(cx, cy)` - Principal point in pixels
- `fw_poly`: Forward polynomial coefficients (up to degree 10)
- `bw_poly`: Backward polynomial coefficients (up to degree 10)
- `A`: 2×2 affine transformation matrix (handles skew, aspect ratio)

---

## LiDAR Models

### Row-Offset Structured Spinning (`RowOffsetStructuredSpinningLidarModel`)

For spinning LiDARs with structured beam patterns (e.g., Hesai P128, Waymo, Pandar).

**Key concepts**:

- **Rows**: Discrete elevation angles (laser beams)
- **Columns**: Discrete azimuth angles (rotation positions)
- **Row offsets**: Per-row azimuth corrections (accounts for beam stagger)

**Parameters**:

- `n_rows`, `n_columns`: Sensor dimensions
- `row_elevations_rad`: Elevation angle per row
- `column_azimuths_rad`: Base azimuth angle per column
- `row_azimuth_offsets_rad`: Per-row azimuth offset
- `fov_horiz_*`, `fov_vert_*`: Field of view boundaries
- `spinning_frequency_hz`: Rotation frequency
- `spinning_direction`: `"cw"` (clockwise) or `"ccw"` (counter-clockwise)

---

## Shutter Types

Cameras support multiple shutter behaviors for rolling shutter modeling:

| Type                    | Description                              |
| ----------------------- | ---------------------------------------- |
| `GLOBAL`                | All pixels captured simultaneously       |
| `ROLLING_TOP_TO_BOTTOM` | Rows captured sequentially from top      |
| `ROLLING_BOTTOM_TO_TOP` | Rows captured sequentially from bottom   |
| `ROLLING_LEFT_TO_RIGHT` | Columns captured sequentially from left  |
| `ROLLING_RIGHT_TO_LEFT` | Columns captured sequentially from right |

---

## External Distortion

Optional 3D ray distortion applied _before_ camera projection (e.g., windshield refraction):

- `NoExternalDistortion`: Identity (no external distortion)
- `BivariateWindshieldDistortion`: Polynomial windshield distortion model

---

## Usage Examples

### Layer 0: Direct Kernel Access

```python
from libs.sensors.kernels import cameras, lidars

# Project camera rays to image points
image_points, valid = cameras.camera_rays_to_image_points(
    camera_rays, projection, external_distortion
)

# Convert LiDAR elements to world rays with rolling shutter
world_rays = lidars.elements_to_world_rays_shutter_pose(
    elements, projection, dynamic_pose
)
```

### Layer 2: High-Level Model API

```python
from libs.sensors.models import (
    OpenCVPinholeCameraModel,
    OpenCVPinholeProjection,
    NoExternalDistortion,
    ShutterType,
    Pose,
    DynamicPose,
    ImageFrame,
)

# Create camera projection
projection = OpenCVPinholeProjection.from_components(
    focal_length=torch.tensor([1000.0, 1000.0]),
    principal_point=torch.tensor([960.0, 540.0]),
    radial_coeffs=torch.zeros(6),
    tangential_coeffs=torch.zeros(2),
    thin_prism_coeffs=torch.zeros(4),
    resolution=torch.tensor([1920, 1080]),
)

# Create camera model
camera = OpenCVPinholeCameraModel(
    projection=projection,
    external_distortion=NoExternalDistortion(),
    resolution=(1920, 1080),
    shutter_type=ShutterType.GLOBAL,
)

# Project world points to image
pose = Pose(translation=translation, rotation=rotation)
dynamic_pose = DynamicPose.from_static_pose(pose)

result = camera.world_points_to_image_points_shutter_pose(
    world_points, dynamic_pose
)
# result.image_points: (N, 2) projected coordinates
# result.valid: (N,) validity mask
```

### LiDAR Example

```python
from libs.sensors.models import (
    RowOffsetStructuredSpinningLidarModel,
    RowOffsetStructuredSpinningLidarProjection,
    DynamicPose,
)

# Create LiDAR projection
projection = RowOffsetStructuredSpinningLidarProjection(
    n_rows=128,
    n_columns=2048,
    row_elevations_rad=row_elevations,
    column_azimuths_rad=column_azimuths,
    row_azimuth_offsets_rad=row_offsets,
    fov_horiz_start_rad=-torch.pi,
    fov_horiz_span_rad=2 * torch.pi,
    fov_vert_start_rad=0.26,
    fov_vert_span_rad=0.52,
    spinning_frequency_hz=10.0,
    spinning_direction="cw",
)

# Create LiDAR model
lidar = RowOffsetStructuredSpinningLidarModel(
    projection=projection,
    angles_to_columns_map_init=True,  # Build inverse map eagerly
)

# Generate world rays with rolling shutter compensation
result = lidar.elements_to_world_rays_shutter_pose(
    elements=elements,  # (N, 2) [row, col] indices
    dynamic_pose=dynamic_pose,
    return_timestamps=True,
)
# result.world_rays: (N, 6) [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
# result.timestamps_us: (N,) per-ray timestamps
```

---

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV Fisheye Model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
