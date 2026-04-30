```{eval-rst}
.. include:: ../_includes/_global_substitutions.rst
```

# Use the NuRec gRPC API Server

To interface with driving simulators, or to generate arbitrary novel views, NuRec implements a gRPC server, whose purpose is to serve rendered images for given camera and actor posing.

There are three aspects to rendering with the grpc server:

- Downloading the gRPC protobuf package for the client
- Launching the server
- Querying it as the client

## Downloading the gRPC protobuf package

To download the gRPC protobuf package for the client, run the following command:

```bash

   ngc registry resource download-version "nvidia/nre/nre_grpc_protos:25.06"
```

**Note:** You must have the [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) installed to run the command.

## Launching the gRPC Server

The gRPC server can be launched via the docker image:

```bash
    docker run --shm-size=64g -it --rm --gpus all \
        --net=host \
        --privileged \
        --volume /path/to/output/folder:/workdir/output \
        nvcr.io/nvidia/nre/nre:latest \
        serve-grpc \
        --artifact-glob /workdir/output/<RUN-ID>/usd-out/last.usdz
```

### Command Line Parameters

#### Required Parameters

- `--artifact-glob`: Glob expression to find artifacts. Must end in .usdz to find relevant files.

#### Optional Parameters

- `--host`: GRPC server host (default: "localhost")
- `--port`: GRPC server port (default: 8080)
- `--health-port`: gRPC health check port (optional; if set, health is served on that port instead of the main gRPC port)
- `--test-scenes-are-valid/--no-test-scenes-are-valid`: Try to load each detected scene before coming online (default: False)
- `--renderer [default|gsplat|nrend]`: Renderer backend selection. `default` uses the artifact's trained renderer (PyTorch forward pass), `gsplat` forces the GSplatRenderer, `nrend` uses the fast NRendWrapper (direct C++/CUDA JIT). (default: "default")

  **Deprecated flags** (still accepted — redirected to `--renderer` equivalents with deprecation warnings;):

  - `--enable-nrend` → equivalent to `--renderer nrend`
  - `--use-gsplat` → equivalent to `--renderer gsplat`

- `--enable-difix`: Use Difix (Fixer) postprocessing (default: False)
- `--difix-url`: URL of Difix checkpoint
- `--difix-cache`: Full path to local Difix cache dir (default: "~/.cache/nre/difix")
- `--difix-model-filename`: Filename of Difix checkpoint (default: "difix.pt")
- `--difix-resolution`: Resolution for Difix processing (default: (544, 960))
- `--enable-timing`: Enable timing of the different parts of the rendering pipeline (default: False)
- `--ray-chunk-size`: Maximum number of rays processed in a single forward pass (default: 2^62)
- `--egocar-hood-dir`: Directory with egocar hood images (default: None)

### Data Structures

1. Pose and Transformation Types

```python

    Pose: grpc_types.Pose
        vec: Vec3  # 3D position vector
            x: float  # X coordinate
            y: float  # Y coordinate
            z: float  # Z coordinate
        quat: Quat  # Quaternion representing rotation
            x: float  # X component of quaternion
            y: float  # Y component of quaternion
            z: float  # Z component of quaternion
            w: float  # W component of quaternion

    PosePair: PosePair
        start_pose: Pose  # Initial pose of the sensor
        end_pose: Pose  # Final pose of the sensor (must be different from start_pose)
```

2. Camera Types

```python

    CameraSpec: CameraSpec
        logical_id: str  # Unique identifier for the camera
        trajectory_idx: int  # Index of the trajectory this camera belongs to
        resolution_w: int  # Width of the camera's resolution
        resolution_h: int  # Height of the camera's resolution
        shutter_type: enum  # Type of shutter mechanism
        camera_param: One of:
            ftheta_param: FthetaCameraParam  # F-theta camera parameters
                principal_point_x: float  # X coordinate of principal point
                principal_point_y: float  # Y coordinate of principal point
                reference_poly: enum  # Reference polynomial type
                pixeldist_to_angle_poly: List[float]  # Polynomial coefficients for pixel to angle conversion
                angle_to_pixeldist_poly: List[float]  # Polynomial coefficients for angle to pixel conversion
                max_angle: float  # Maximum angle supported by the camera
```

3. Dynamic Object Types

```python

    DynamicObject: DynamicObject
        track_id: str  # Unique identifier for the track
        pose_pair: PosePair  # Start and end poses of the object
```

### API Endpoints

1. Basic Connection

```python

    test_grpc_connection(host: str, port: int)
    # Returns: bool (True if connection successful)
```

2. Scene Management

```python

    get_available_scenes()
    # Returns: List[str] (scene_ids)
```

3. Camera Management

```python

    get_available_cameras(scene_id: str)
    # Returns: List[AvailableCamera]
```

4. Trajectory Management

```python

    get_available_trajectories(scene_id: str)
    # Returns: List[AvailableTrajectory]
```

5. Rendering

```python

    render_rgb(request: RGBRenderRequest)
    # Returns: RGBRenderReturn (image_bytes)

    render_lidar(request: LidarRenderRequest)
    # Returns: LidarRenderReturn (point_cloud)
```

**Note:**

- The `nre.grpc.protos` package contains all the protobuf messages for the gRPC server / client.

## Sending gRPC Render Requests

To send gRPC render requests, launch the docker container with the following command:

```bash

    docker run --shm-size=64g -it --rm --gpus all \
        --network host \
        --volume /path/to/output/folder:/workdir/output \
        nvcr.io/nvidia/nre/nre:latest \
        render-grpc \
        --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
        --output-dir /path/to/render/directory \
        --camera-id <CAMERA-ID>

```

### Command Line Parameters

#### Required Parameters

- `--artifact-path TEXT`: Path to the NuRec artifact `last.usdz`
- `--output-dir TEXT`: Path to the output rendered image

#### Optional Parameters

- `--host TEXT`: gRPC server host _(default: localhost)_
- `--port INTEGER`: Port to run the gRPC server on _(default: 8080)_
- `--height INTEGER`: Height of the image _(default: 300)_
- `--camera-id TEXT`: Camera ID _(default: camera_front_wide_120fov)_
- `--image-format [png|jpeg]`: Image format for the output, options are PNG or JPEG _(default: jpeg)_
- `--frame-step INTEGER`: Step size in frames _(default: 1)_
- `--disable-rolling-shutter`: Disable rolling shutter by applying the frame-end timestamps to full
  frames, which is useful for debugging
- `--demo-actor-transform`: Apply a precomputed transformation to actor poses
- `--shutdown-server-on-completion`: Shutdown the server on completion
- `--rig-name TEXT`: Rig name for the inpainted ego hood (e.g. hyperion8.0 or hyperion8.1). Set to None to disable inpainting.

## Example Usage

Here's a step-by-step guide to render scenes from USDZ artifacts using the NuRec gRPC API:

1. Connect to the Server:

```python

    import grpc.aio
    from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub

    channel = grpc.aio.insecure_channel("localhost:8080")
    client_service = SensorsimServiceStub(channel)
```

2. Discover Available Scenes:

```python

    from nre.grpc.protos.common_pb2 import Empty
    response = await client_service.get_available_scenes(Empty())
    scene_ids = response.scene_ids
```

3. Get Scene Information:

```python

    from nre.grpc.protos.sensorsim_pb2 import AvailableTrajectoriesRequest, AvailableCamerasRequest

    # Get trajectories
    trajectories = await client_service.get_available_trajectories(
        AvailableTrajectoriesRequest(scene_id="your_scene_id")
    )

    # Get cameras
    cameras = await client_service.get_available_cameras(
        AvailableCamerasRequest(scene_id="your_scene_id")
    )
```

4. Render an Image:

```python

    from nre.grpc.protos.sensorsim_pb2 import RGBRenderRequest, ImageFormat, PosePair
    from nre.grpc.serve import se3_to_grpc_pose

    request = RGBRenderRequest(
        scene_id="your_scene_id",
        resolution_h=300,
        resolution_w=400,
        camera_intrinsics=front_wide_camera.intrinsics,
        frame_start_us=middle_timestamp,
        frame_end_us=middle_timestamp + 1,
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(pose),
            end_pose=se3_to_grpc_pose(pose)
        ),
        image_format=ImageFormat.JPEG,
        image_quality=95
    )

    response = await client_service.render_rgb(request)

    # Save the image
    with open("output.jpg", "wb") as f:
        f.write(response.image_bytes)
```
