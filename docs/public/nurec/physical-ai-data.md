# Render the Physical AI Dataset with NuRec

NVIDIA Neural Reconstruction (NuRec) builds photorealistic 3D renderings of real-world scenarios captured from sensor data. To expedite developers hoping to render scenes with NuRec, NVIDIA has provided the [Physical AI dataset on HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec). However, the key challenge when working with the Physical AI dataset is to properly handle the coordinate system transformations between different reference frames.

The reference frames and their corresponding coordinate systems are as follows:

- **ECEF**: Earth-Centered, Earth-Fixed coordinates using WGS84 datum

- **ENU**: East-North-Up local coordinates defined by OpenDRIVE georeference

- **OpenDRIVE Map Space**: Local ENU coordinate system with origin defined by proj string

- **NuRec Space**: Reconstruction coordinate system where frame 0 has identity transform

The NuRec coordinate system is fundamentally defined by the first frame having an identity transform, with `T_rig_ecef` representing the ECEF pose at frame 0 in the trajectories included in the Physical AI dataset. The complete transformation chain follows NuRec Space → ECEF → ENU (OpenDRIVE Map Space), where OpenDRIVE maps contain strings that define the local ENU origin. The critical transformation matrix `T_nurec_map = T_ecef_enu @ T_rig_ecef` aligns NuRec and map coordinate systems, enabling bidirectional conversion using `T_nurec_map` for NuRec→Map and its inverse for Map→NuRec. All transformations go through ECEF as the common reference frame, and you can use the NuRec gRPC API to render custom trajectories in NuRec coordinate space. NuRec space is defined relative to the first frame's ECEF position, making `T_rig_ecef` the bridge between the reconstruction coordinate system and global ECEF coordinates.

This guide provides the foundation for working with pre-reconstructed NVIDIA Physical AI datasets in custom simulation environments while maintaining proper coordinate alignment for photorealistic neural rendering.

## Load and Extract Data Assets

Before you can transform the coordinate systems and leverage the NuRec gRPC API to render them on a simulation platform, you must first load the assets and extract their data and references. Follow the steps in this section.

### Load USDZ File

The NuRec datasets are USDZ package files containing the following files and corresponding information:

- Neural reconstruction data

- **`data_info.json`:** Scenario metadata including time ranges and sequence info

- **`rig_trajectories.json`:** Camera calibrations, ego trajectory, and `T_world_base` transform

- **`sequence_tracks.json`:** Object track data with poses, timestamps, and labels

- **`map.xodr`:** OpenDRIVE map file defining the road network

In your script, load the files and extract their contents for later reference.

```python
import zipfile
import json


def load_nurec_scenario(usdz_path: str):
    """Load NuRec scenario data from USDZ file."""
    scenario_data = {}

    with zipfile.ZipFile(usdz_path, "r") as zip_ref:
        # Extract scenario metadata
        with zip_ref.open("data_info.json") as metadata_file:
            scenario_data["metadata"] = json.load(metadata_file)

        # Extract rig trajectories and camera calibrations
        with zip_ref.open("rig_trajectories.json") as rig_file:
            scenario_data["rig_trajectories"] = json.load(rig_file)

        # Extract object tracks data
        with zip_ref.open("sequence_tracks.json") as tracks_file:
            scenario_data["sequence_tracks"] = json.load(tracks_file)

        # Extract map data
        with zip_ref.open("map.xodr") as xodr_file:
            scenario_data["xodr_data"] = xodr_file.read().decode("utf-8")

    return scenario_data


# Load scenario
usdz_path = "path/to/your/scenario.usdz"
scenario = load_nurec_scenario(usdz_path)
```

### Unzip and Extract Map Data

The USDZ file contains an OpenDRIVE map that defines the road network. Set up your script to automatically extract the map data on loading
the scenario:

```python
import zipfile


def extract_map_from_usdz(usdz_path: str) -> str:
    """Extract OpenDRIVE map data from USDZ file."""

    with zipfile.ZipFile(usdz_path, "r") as zip_ref:
        with zip_ref.open("map.xodr") as xodr_file:
            xodr_data = xodr_file.read().decode("utf-8")
            return xodr_data


# Usage
xodr_data = extract_map_from_usdz(usdz_path)
```

### Load Trajectory Data

NuRec scenarios contain pre-recorded trajectories for all actors (vehicles, pedestrians) including the ego vehicle. Define these trajectories in your
script, as follows:

```python
import numpy as np


def parse_trajectory_data(scenario_data):
    """Parse trajectory data from loaded scenario."""
    metadata = scenario_data["metadata"]
    rig_trajectories = scenario_data["rig_trajectories"]
    sequence_tracks = scenario_data["sequence_tracks"]

    # Get time range
    start_time = metadata["pose-range"]["start-timestamp_us"]
    end_time = metadata["pose-range"]["end-timestamp_us"]
    duration_seconds = (end_time - start_time) / 1_000_000

    # Parse ego trajectory from rig_trajectories
    ego_poses = rig_trajectories["rig_trajectories"][0]["T_rig_worlds"]
    ego_timestamps = rig_trajectories["rig_trajectories"][0]["T_rig_world_timestamps_us"]

    # Parse object tracks from sequence_tracks
    tracks_data = sequence_tracks["dummy_chunk_id"]["tracks_data"]
    track_ids = tracks_data["tracks_id"]

    object_tracks = {}
    for i, track_id in enumerate(track_ids):
        track_poses = tracks_data["tracks_poses"][i]
        track_timestamps = tracks_data["tracks_timestamps_us"][i]
        track_label = tracks_data["tracks_label_class"][i]

        object_tracks[track_id] = {
            "poses": track_poses,
            "timestamps": track_timestamps,
            "label": track_label
        }

    return start_time, end_time, {
        "ego": {"poses": ego_poses, "timestamps": ego_timestamps},
        "objects": object_tracks
    }


# Usage
start_time, end_time, trajectories = parse_trajectory_data(scenario)
```

### Load Map and Extract Georeference

The OpenDRIVE map defines the local ENU coordinate system through its georeference, as follows:

```python
import xml.etree.ElementTree as ET


def parse_opendrive_georeference(xodr_data: str):
    """Extract georeference information from OpenDRIVE map."""
    tree = ET.fromstring(xodr_data)
    geo_reference = tree.find(".//geoReference")

    # Parse proj string
    # Example: "+proj=tmerc +lat_0=37.4419 +lon_0=-122.1430 +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
    proj_parts = geo_reference.text.split(" ")

    georeference = {}
    for part in proj_parts:
        if part.startswith("+lat_0="):
            georeference["latitude"] = float(part[7:])
        elif part.startswith("+lon_0="):
            georeference["longitude"] = float(part[7:])
        elif part.startswith("+alt_0="):
            georeference["altitude"] = float(part[7:])
        elif part.startswith("+proj="):
            georeference["projection"] = part[6:]

    georeference.setdefault("altitude", 0.0)

    return georeference


# Parse georeference from loaded map
georeference = parse_opendrive_georeference(scenario["xodr_data"])
```

## Transform the Coordinate Systems

Once you've loaded the assets and defined them in your script, you can transform the coordinates from NuRec Space to Earth-Centered, Earth-Fixed (ECEF) Space, to OpenDRIVE Map Space (East-North-Up, or ENU space).

**NuRec Space**: The reconstruction coordinate system where frame 0 (first frame) has an identity transformation matrix. This is defined as the inverse of the ECEF matrix at the first frame.

**T_rig_ecef**: The ECEF pose of the first frame. Converting from NuRec space to ECEF is achieved using the formula `ECEF_pose = NuRec_pose × T_rig_ecef`.

**OpenDRIVE ENU**: Local East-North-Up coordinate system with origin defined by the proj string in the map's georeference.

The transformation pipeline follows the sequence: **NuRec Space → ECEF → ENU (OpenDRIVE Map Space)**

![NuRec coordinate transformation chain](/images/nurec-coordinate-transformation.png)

```{tip}
NuRec space is defined such that frame 0 transforms to identity, making `T_rig_ecef` the ECEF pose of the recording start point.
```

### Add the Transform Calculation

The following snippet shows a detailed transform calculation from NuRec Space to ECEF Space to OpenDRIVE Map Space (ENU):

```python
import numpy as np
import math


def lat_lng_alt_to_ecef(lat_lng_alt: np.ndarray) -> np.ndarray:
    """Convert GPS coordinates to ECEF using WGS84 ellipsoid."""
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    flattening = 1.0 / 298.257223563
    b = a * (1.0 - flattening)  # Semi-minor axis

    phi = np.deg2rad(lat_lng_alt[:, 0])
    gamma = np.deg2rad(lat_lng_alt[:, 1])

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    e_square = (a * a - b * b) / (a * a)

    N = a / np.sqrt(1 - e_square * sin_phi * sin_phi)

    x = (N + lat_lng_alt[:, 2]) * cos_phi * cos_gamma
    y = (N + lat_lng_alt[:, 2]) * cos_phi * sin_gamma
    z = (N * (b * b) / (a * a) + lat_lng_alt[:, 2]) * sin_phi

    return np.column_stack([x, y, z])


def ecef_to_enu_transform(lat_lon_alt: np.ndarray) -> np.ndarray:
    """Compute transformation matrix from ECEF to local ENU frame."""
    # Convert reference point to ECEF
    ecef_origin = lat_lng_alt_to_ecef(lat_lon_alt)

    # Extract latitude and longitude in radians
    lat_rad = np.deg2rad(lat_lon_alt[0, 0])
    lon_rad = np.deg2rad(lat_lon_alt[0, 1])

    # ENU rotation matrix
    rotation_matrix = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)],
    ])

    # Build 4x4 transformation matrix
    t_ecef_enu = np.eye(4)
    t_ecef_enu[:3, :3] = rotation_matrix
    t_ecef_enu[:3, 3] = -rotation_matrix @ ecef_origin.T.flatten()

    return t_ecef_enu
def calculate_nurec_to_map_transform(scenario_data, georeference):
    """
    Calculate complete transformation from NuRec space to OpenDRIVE map space.

    Args:
        scenario_data: Loaded scenario containing T_rig_ecef
        georeference: Parsed georeference from OpenDRIVE map

    Returns:
        t_nurec_map: 4x4 transformation from NuRec to map coordinates
    """

    # Step 1: Get T_rig_ecef from rig trajectories
    # This is the ECEF pose at frame 0 (first frame) - T_world_base
    t_rig_ecef = np.array(scenario_data["rig_trajectories"]["T_world_base"])

    # Step 2: Create transformation from ECEF to OpenDRIVE ENU
    map_origin = np.array([[
        georeference["latitude"],
        georeference["longitude"],
        georeference["altitude"]
    ]])

    t_ecef_enu = ecef_to_enu_transform(map_origin)

    # Step 3: Combine transformations: NuRec → ECEF → ENU
    # NuRec space coordinates × T_rig_ecef = ECEF coordinates
    # ECEF coordinates × T_ecef_enu = ENU coordinates
    # Therefore: NuRec coordinates × T_rig_ecef × T_ecef_enu = ENU coordinates
    t_nurec_map = t_ecef_enu @ t_rig_ecef

    return t_nurec_map, t_rig_ecef, t_ecef_enu


# Calculate the complete transformation
t_nurec_map, t_rig_ecef, t_ecef_enu = calculate_nurec_to_map_transform(scenario, georeference)
```

### Simulation in Map Space

Once you've completed the transform calculation, you can run your own simulation using the map coordinate system:

```python
def create_custom_trajectory(start_time, end_time, dt=50000):
    """Create a custom trajectory in map ENU coordinates."""
    times = np.arange(start_time, end_time, dt)  # 50ms intervals
    trajectory = []

    for i, t in enumerate(times):
        # Example: Simple straight line motion
        x = i * 0.5  # Move 0.5m per step (10 m/s at 50ms intervals)
        y = 0.0
        z = 0.0

        # Example orientation (facing forward along x-axis)
        pose_matrix = np.eye(4)
        pose_matrix[0, 3] = x
        pose_matrix[1, 3] = y
        pose_matrix[2, 3] = z

        trajectory.append({
            'timestamp_us': t,
            'pose_matrix': pose_matrix,
            'position': [x, y, z],
            'orientation': [0, 0, 0, 1]  # quaternion [x, y, z, w]
        })

    return trajectory


def simulate_in_map_space(scenario_data, t_nurec_map):
    """Example of custom simulation in map coordinate space."""

    # Get scenario time bounds
    start_time = scenario_data["metadata"]["pose-range"]["start-timestamp_us"]
    end_time = scenario_data["metadata"]["pose-range"]["end-timestamp_us"]

    # Create custom trajectory in map coordinates
    map_trajectory = create_custom_trajectory(start_time, end_time)

    return map_trajectory


# Run custom simulation
custom_trajectory = simulate_in_map_space(scenario, t_nurec_map)
```

### Convert Coordinates Back to NuRec Space

When you have new poses from your simulation, convert them back to NuRec space for rendering:

```python
def map_pose_to_nurec(map_pose_matrix: np.ndarray, t_nurec_map: np.ndarray) -> np.ndarray:
    """Convert map ENU pose to NuRec coordinate system."""

    # To go from map coordinates back to NuRec coordinates, use inverse transform
    t_map_nurec = np.linalg.inv(t_nurec_map)

    # Apply coordinate transformation
    nurec_pose = t_map_nurec @ map_pose_matrix

    return nurec_pose


def convert_trajectory_to_nurec(map_trajectory, t_nurec_map):
    """Convert entire trajectory from map space to NuRec space."""

    t_map_nurec = np.linalg.inv(t_nurec_map)
    nurec_trajectory = []

    for point in map_trajectory:
        map_pose = point['pose_matrix']
        nurec_pose = t_map_nurec @ map_pose

        nurec_point = {
            'timestamp_us': point['timestamp_us'],
            'pose_matrix': nurec_pose,
            'nurec_position': nurec_pose[:3, 3],
            'original_map_position': point['position']
        }
        nurec_trajectory.append(nurec_point)

    return nurec_trajectory


# Convert custom trajectory to NuRec coordinates for rendering
nurec_trajectory = convert_trajectory_to_nurec(custom_trajectory, t_nurec_map)
```

## Render Using NuRec gRPC API

Finally, render sensor data using the NuRec gRPC service. For custom trajectories, create render requests for the NuRec service, as shown in the following snippet:

```python
def setup_camera_specifications(scenario_data):
    """Define camera specifications from scenario calibrations."""

    camera_specs = {}

    # Extract camera calibrations from rig_trajectories.json
    camera_calibrations = scenario_data["rig_trajectories"]["camera_calibrations"]

    for camera_id, calibration in camera_calibrations.items():
        logical_name = calibration["logical_sensor_name"]
        camera_model = calibration["camera_model"]
        params = camera_model["parameters"]

        camera_specs[logical_name] = {
            "type": camera_model["type"],
            "resolution_w": params["resolution"][0],
            "resolution_h": params["resolution"][1],
            "intrinsics": {
                "principal_point_x": params["principal_point"][0],
                "principal_point_y": params["principal_point"][1],
                "max_angle": params["max_angle"],
                "pixeldist_to_angle_poly": params["pixeldist_to_angle_poly"],
                "angle_to_pixeldist_poly": params["angle_to_pixeldist_poly"],
                "reference_poly": params["reference_poly"]
            },
            "T_sensor_rig": calibration["T_sensor_rig"]
        }

    return camera_specs


# Setup camera specifications for gRPC rendering
camera_specs = setup_camera_specifications(scenario)
```

### Programmatically Render with the gRPC API

Use the NuRec gRPC API for programmatic rendering:

```python
import grpc
from grpc_proto.sensorsim_pb2_grpc import SensorsimServiceStub
from grpc_proto.sensorsim_pb2 import RGBRenderRequest, ImageFormat, CameraSpec, PosePair
from grpc_proto.common_pb2 import Empty as EmptyRequest
import numpy as np


def render_with_grpc_api(usdz_path, nurec_trajectory, camera_specs, scenario_data, output_dir):
    """Render using gRPC API directly."""

    # Start NuRec render service
    with NuRecRenderService(usdz_path, port=2000) as render_service:

        # Setup gRPC connection
        channel = grpc.insecure_channel("localhost:2000")
        client = SensorsimServiceStub(channel)

        for i, point in enumerate(nurec_trajectory):
            timestamp = point["timestamp_us"]
            pose_matrix = point["pose_matrix"]

            # Get camera specification (use first available camera)
            camera_name = list(camera_specs.keys())[0]
            camera_spec = camera_specs[camera_name]

            # Create gRPC camera spec
            grpc_camera_spec = CameraSpec(
                logical_id=camera_name,
                resolution_h=camera_spec["resolution_h"],
                resolution_w=camera_spec["resolution_w"],
                ftheta_param=camera_spec["intrinsics"]
            )

            # Create pose pair for gRPC
            pose_pair = PosePair(
                start_pose=pose_matrix,
                end_pose=pose_matrix
            )

            # Create render request
            render_request = RGBRenderRequest(
                scene_id=scenario_data["metadata"]["sequence_id"],
                resolution_h=camera_spec["resolution_h"],
                resolution_w=camera_spec["resolution_w"],
                camera_intrinsics=grpc_camera_spec,
                frame_start_us=timestamp,
                frame_end_us=timestamp + 1,
                sensor_pose=pose_pair,
                dynamic_objects=[],
                image_format=ImageFormat.JPEG,
                image_quality=95
            )

            # Render frame via gRPC
            response = client.render_rgb(render_request)

            # Save image
            output_path = f"{output_dir}/frame_{i:06d}_{timestamp}.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.image_bytes)


# Render using gRPC API
render_with_grpc_api(usdz_path, nurec_trajectory, camera_specs, scenario, "output_images/")
```
