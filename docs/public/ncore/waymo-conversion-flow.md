# Sample: Convert Waymo to NCore

To download sample data from the Waymo Open Dataset, sign up for the [Waymo Open Dataset](https://waymo.com/open/terms) and then download the sample data. The file type of raw Waymo data is `.tfrecords`.

## Convert

The convert block contains the necessary steps to convert raw Waymo data into NCore format.

**from_config**

The first API implemented by the Waymo
Data Converter is `from_config`. In the diagram, the first step is passing the config to the **convert** block. The `from_config`
function determines how the `DataConverter` is created given the config. Refer to the following implementation for Waymo:

.. code-block:: python

    @staticmethod
    def from_config(config) -> DataConverter:
        return WaymoConverter(config)

Where the `__init__` function is defined as follows:

.. code-block:: python

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

Note that the API expects you to pass the config to the Abstract `DataConverter` class (provided by the NCore API) that the waymo converter implements. As such,
it is standard to create the custom DataConverter by simply invoking the custom DataConverter using the config.

**get_sequence_paths**

The next step in the conversion flow is to get sequence paths given the config. Specifically, we notice that `get_sequence_paths`
is invoked upon the config file. Refer to the following implementation for Waymo data:

.. code-block:: python

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:
        return [p for p in sorted(Path(config.root_dir).glob("*.tfrecord"))]

This if the first function that must be implemented. This function is invoked to convert the config provided into a list of paths to the
raw data files which, in the case of Waymo data, are `.tfrecord` files.

## Convert Sequence

The convert sequence block contains the primary logic for converting data to NCore. The function `convert_sequence` is the last required implementation for
the Abstract `DataConverter` class. It contains the logic for converting specific data provided by the sequence paths `get_sequence_paths`
(in this case, `.tfrecords`) into NCore data. Refer to the specification:

.. code-block:: python

    def convert_sequence(self, sequence_path: Path) -> None:

The `convert_sequence` function implementation for Waymo DataConverter can be split into 5 parts.

**Part 1: Getting Frame Data from Sequence Paths (.tfrecords)**

Since the goal is to convert all frames present within the sequences provided, the `waymo_open_dataset` library is utilized. Specifically, `dataset_pb2`
and `label_pb2` from the `waymo_open_dataset` library provide APIs for dealing with and extracting data from `.tfrecord` files.

To start off, the **Tensorflow** library is used to load the **.tfrecord** file as a **TfRecordDataset** as follows:

.. code-block:: python

    dataset = tf.data.TFRecordDataset(sequence_path, compression_type="")

Then all frames within the given sequence are obtained and stored in a list:

.. code-block:: python

    frames: list[dataset_pb2.Frame] = []
    sequence_name = ""
    for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if not frames:
            sequence_name = frame.context.name
        frames.append(frame)
        if frame.context.name != sequence_name:
            raise ValueError("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

This step also prepares the NCore writer which is provided by the NCore API used to write and store data obtained from the
sequence. A **ShardDataWriter** allows for the writing of specific scene data. In the case of the waymo data, refer to the following
**ShardDataWriter** definition:

.. code-block:: python

    self.data_writer = ShardDataWriter(
        self.output_dir / sequence_name,                                             # Output directory path
        sequence_name,                                                               # Container name
        self.get_active_camera_ids([camera for camera in self.CAMERA_MAP.values()]), # Camera IDs
        self.get_active_lidar_ids([lidar for lidar in self.LIDAR_MAP.values()]),     # LiDAR IDs
        self.get_active_radar_ids([]),                                               # Radar IDs
        "waymo-calibration",                                                         # Calibration Type
        "waymo-egomotion",                                                           # Egomotion Type
        sequence_name,                                                               # Sequence ID
        {},  # no generic sequence meta data                                         # Generic Metadata
        0,  # single shard                                                           # Shard ID
        1,                                                                           # Shard Count
        False,                                                                       # Store shard metadata
    )

**Note:** the **ShardDataWriter** supports multi-sharding which consists of creating multiple shards based on time-ranges. For example,
suppose you have a 60 second clip. Sharding based on a max shard length of 30 seconds will generate two shards for that clip. Single shards are
simpler to deal with as you do not have to manage shard boundaries but multi-sharding does have its merits (e.g. you have a large clip and want to split
the data in multiple files).

**Part 2: Decoding Poses**

Waymo data includes rig poses which can be stored as part of the NCore format:

.. code-block:: python

    def decode_poses(self, frames) -> None:

As part of the **self.decode_poses(frames)** function, each frame is accessed and the image corresponding to the waymo frame is used to obtain
the corresponding image's pose. The **T_rig_world** arrays and corresponding timestamps. Using these timestamps, pose points are extrapolated
at the boundaries using velocity information to allow interpolation at lidar timestamps:

.. code-block:: python

    T_rig_worlds_array = []
    T_rig_world_timestamps_us_array = []

    for i, frame in enumerate(frames):
        for image in frame.images:
            T_rig_worlds_array.append(
                np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4]))
            )
            T_rig_world_timestamps_us_array.append(
                int(image.pose_timestamp * 1e6)
            )  # Convert the poses to microseconds (rounding decimal)

            # Extrapolate pose points on the boundaries using velocity information to allow interpolation at lidar timestamps
            dt_us = 0.0
            if i == 0:
                # extrapolate exactly to first lidar start-of-spin time
                dt_us = frame.timestamp_micros - T_rig_world_timestamps_us_array[-1]
            if i == len(frames) - 1:
                # make sure to overshoot a little over last lidar end-of-spin time
                dt_us = int(1.25 * (frames[-1].timestamp_micros - frames[-2].timestamp_micros))

            if dt_us:
                T_rig_world = T_rig_worlds_array[-1]
                velocity_global = np.array(
                    [image.velocity.v_x, image.velocity.v_y, image.velocity.v_z], dtype=np.float32
                ).reshape(3, 1)
                omega_vehicle = np.array(
                    [image.velocity.w_x, image.velocity.w_y, image.velocity.w_z], dtype=np.float32
                ).reshape(3, 1)
                omega_world = np.matmul(T_rig_world[:3, :3], omega_vehicle)

                T_rig_worlds_array.append(
                    extrapolate_pose_based_on_velocity(T_rig_world, velocity_global, omega_world, dt_us / 1e6)
                )
                T_rig_world_timestamps_us_array.append(T_rig_world_timestamps_us_array[-1] + dt_us)

NCore API provides a **Pose** type which is used to store pose information. Given the rig poses and timestamps, **Pose(s)** are created
after transforming the timestamp and poses arrays to the common canonical format convention as follows. Note that the **Pose** type
requires 3 arguments.

    1. **T_rig_world_base**: Base rig-to-global-world SE3 transformation (float64, np.ndarray [4,4])
    2. **T_rig_worlds**: All rig-to-local-world SE3 transformations of the trajectory (float64, np.ndarray [N,4,4])
    3. **T_rig_world_timestamps_us**: All rig-to-local-world transformation timestamps of the trajectory (uint64, np.ndarray [N,])

For Waymo data, the base transformation is an identity matrix because the waymo data is already shifted. Refer to the following implementation:

.. code-block:: python

    # make unique + sort + stack all poses (common canonical format convention)
    T_rig_world_timestamps_us, unique_indices = np.unique(
        np.array(T_rig_world_timestamps_us_array, dtype=np.uint64), return_index=True
    )
    T_rig_worlds = np.stack(T_rig_worlds_array)[unique_indices]

    # Use identity base pose as waymo data is already shifted
    T_rig_world_base = np.eye(4, dtype="float64")

    self.poses = Poses(
        T_rig_world_base=T_rig_world_base,
        T_rig_worlds=T_rig_worlds,
        T_rig_world_timestamps_us=T_rig_world_timestamps_us,
    )

The **ShardDataWriter** that was created earlier is then used to store these poses:

.. code-block:: python

    self.data_writer.store_poses(self.poses)

Note that the pose data can also be stored in a **PoseInterpolator** utility provided by the NCore API as follows:

.. code-block:: python

    self.pose_interpolator = PoseInterpolator(self.poses.T_rig_worlds, self.poses.T_rig_world_timestamps_us)

The benefit of this is the rig transformation for a list of timestamps can easily be obtained using:

.. code-block:: python

    self.pose_interpolator.interpolate_to_timestamps(timestamps_us)

This is used when decoding lidars in the next section. When implementing a conversion of pose data, focus on obtaining the arguments required for the following:

1. `self.data_writer.store_poses(...)`

**Part 3: Decoding LiDARs**

Waymo data also provides LiDAR data from the single LiDAR sensor:

.. code-block:: python

    def decode_lidars(self, frames) -> None:

The NCore Writer provides the **store_lidar_frame** API to store LiDAR data:

.. code-block:: python

    def store_lidar_frame(
        self,
        lidar_id: str,                          # LiDAR ID
        continuous_frame_index: int,            # Frame index
        # The following point cloud points need to be motion-compensated and in end-of-spin frame
        xyz_s: np.ndarray,                      # Point Cloud points at the start of spin (motion-compensated)
        xyz_e: np.ndarray,                      # Point Cloud points at the end of spin (motion-compensated)
        intensity: np.ndarray,                  # LiDAR intensity
        timestamp_us: np.ndarray,               # Per-point timestamps
        model_element: Optional[np.ndarray],    # Per-point 2d element index into intrinsic lidar model, if available
        frame_labels: List[types.FrameLabel3],  # Frame label data
        T_rig_worlds: np.ndarray,               # Poses: rig-to-world SE3 transformations
        timestamps_us: np.ndarray,              # Timestamps
        generic_data: Dict[str, np.ndarray],    # Generic per-frame data (key-value pairs, *not* dimension / dtype validated) [NOT MANDATORY]
        generic_meta_data: Dict[str, JsonLike], # Generic metadata [NOT MANDATORY]
    ) -> None:

**Note: store_lidar_frame** should be called in a linear order for each frame (frame 1, frame 2, frame 3, ...). Otherwise, linear parsing
through the container will be inefficient. The frame in indicated by the **continuous_frame_index** argument.

The invocation for Waymo data is as follows:

.. code-block:: python

    self.data_writer.store_lidar_frame(
        lidar_ncore_id,
        continuous_frame_index,
        xyz_s,
        xyz_e,
        intensity,
        point_timestamps_us,
        None,
        frame_labels,
        T_rig_worlds,
        timestamps_us,
        {
            "dynamic_flag": dynamic_flag.astype(np.int8),  # N
            # primary ray data
            "elongation": elongation.reshape(-1).astype(np.float32),  # N
            "range_image_indices": range_image_indices.reshape((-1, 2)).astype(
                np.uint32
            ),  # N x 2 (indices into HxW source range image)
            # secondary ray data
            "primary_indices": primary_indices.reshape(-1).astype(
                np.uint32
            ),  # S (indices of the primary parent ray)
            "xyz_e_second": xyz_e_second.reshape((-1, 3)).astype(np.float32),  # S x 3
            "intensity_second": intensity_second.reshape(-1).astype(np.float32),  # S
            "elongation_second": elongation_second.reshape(-1).astype(np.float32),  # S
        }
        | ({"semantic_class": semantic_class} if semantic_class is not None else {}),  # N
        {},
    )

NCore also stores LiDAR metadata using the following function call:

.. code-block:: python

    def store_lidar_meta(
        self,
        lidar_id: str,                                                             # LiDAR ID
        frame_timestamps_us: np.ndarray,                                           # All frame timestamps
        T_sensor_rig: np.ndarray,                                                  # Extrinsic data: sensor-to-rig SE3 transformation
        lidar_model_parameters: Optional[types.ConcreteLidarModelParametersUnion], # Intrinsic lidar model parameters, if available
        generic_meta_data: Dict[str, JsonLike],                                    # Generic sensor meta-data (has to be json-serializable) [NOT MANDATORY]
    ) -> None:

The invocation for Waymo data is as follows:

.. code-block:: python

    self.data_writer.store_lidar_meta(
        lidar_ncore_id,
        np.array(frame_end_timestamps_us, dtype=np.uint64),
        T_sensor_rig,
        None,
        {
            "label-class-string-id-map": {
                label_string: label_id
                for label_id, label_string in self.LIDAR_LABEL_CLASS_ID_STRING_MAP.items()
            },
            "angles": {
                # angles associated with range-image "pixels"
                "inclinations_rad": inclinations_rad.reshape(-1)
                .astype(np.float32)
                .tolist(),  # H (one per range-image row)
                "azimuths_rad": azimuths_rad.reshape(-1)
                .astype(np.float32)
                .tolist(),  # W (one per range-image column)
            },
        },
    )

And lastly, the accumulated tracks (of the rig) in global time are stored in the NCore data using the **Tracks** type (provided by NCore API) as follows:

.. code-block:: python

    self.data_writer.store_tracks(Tracks(track_labels))

Refer to `waymo3.py` for a detailed implementation,
how variables are defined, and how the functions above were invoked. In particular, focus on the types provided by NCore API that are utilized as they often provide great convenience
for storing data. The following are example types utilized in the implementation:

    - **FrameLabel3**
    - **BBox3**
    - **LabelSource**
    - **Tracks**

When implementing a conversion of LiDAR data, focus on obtaining the arguments required for the following:

1. `self.data_writer.store_lidar_frame(...)`
2. `self.data_writer.store_lidar_meta(...)`
3. `self.data_writer.store_tracks(...)`

**Part 4: Decoding Cameras**

Waymo data also provides camera data including rgb and segmentation data:

.. code-block:: python

    def decode_cameras(self, frames) -> None:

The NCore Writer provides the **store_camera_frame** API to store camera data:

.. code-block:: python

    def store_camera_frame(
        self,
        camera_id: str,                         # Camera ID
        continuous_frame_index: int,            # Frame index
        image_file_binary_data: bytes,          # Raw binary frame image data
        image_file_format: str,                 # Frame image format (e.g. 'jpeg')
        T_rig_worlds: np.ndarray,               # Poses: rig-to-world SE3 transformations
        timestamps_us: np.ndarray,              # Timestamps
        generic_data: Dict[str, np.ndarray],    # Generic per-frame data (key-value pairs, *not* dimension / dtype validated) [NOT MANDATORY]
        generic_meta_data: Dict[str, JsonLike], # Generic metadata [NOT MANDATORY]
    ) -> None:

The invocation for Waymo data is as follows:

.. code-block:: python

    self.data_writer.store_camera_frame(
        camera_ncore_id,
        continuous_frame_index,
        image.image,
        "jpeg",
        T_rig_worlds,
        timestamps_us,
        generic_data,
        generic_meta_data,
    )

NCore also stores camera metadata using the following function call:

.. code-block:: python

    def store_camera_meta(
        self,
        camera_id: str,                                                    # Camera ID
        frame_timestamps_us: np.ndarray,                                   # All frame timestamps
        T_sensor_rig: np.ndarray,                                          # Extrinsic data: sensor-to-rig SE3 transformation
        camera_model_parameters: types.ConcreteCameraModelParametersUnion, # Intrinsic camera model parameters provided by NCore API
        mask_image: Optional[PILImage.Image],                              # Mask image that validates pixels from invalid ones (e.g. pixels that observe ego-vehicle or value 0)
        generic_meta_data: Dict[str, JsonLike],                            # Generic sensor meta-data (needs to be json-serializable)
    ) -> None:

Note that `camera_model_parameters` expects a type provided by NCore API (e.g. `ncore.data.OpenCVPinholeCameraModelParameters`). Refer
to the implementation for Waymo data:

.. code-block:: python

    self.data_writer.store_camera_meta(
        camera_ncore_id,
        np.array(frame_end_timestamps_us, dtype=np.uint64),
        T_sensor_rig,
        OpenCVPinholeCameraModelParameters(
            np.array([width, height], dtype=np.uint64),
            rolling_shutter_direction,
            np.array([c_u, c_v], dtype=np.float32),
            np.array([f_u, f_v], dtype=np.float32),
            np.array([k1, k2, k3, 0, 0, 0], dtype=np.float32),
            np.array([p1, p2], dtype=np.float32),
            np.array([0, 0, 0, 0], dtype=np.float32),
        ),
        None,
        {
            "label-class-string-id-map": {
                label_string: label_id
                for label_id, label_string in self.CAMERA_LABEL_CLASS_ID_STRING_MAP.items()
            }
        },
    )

Refer to `waymo3.py` for a detailed implementation.
Similarly, make notice of how types provided by NCore API are utilized for conversion and storage to NCore.

**Step 5: Finalize Writer**

The final step of the conversion process is to invoke **finalize()** function as part of the **ShardDataWriter** which stores the
shard in the output directory provided in the config:

.. code-block:: python

    self.data_writer.finalize()

As such, the resulting is the implementation of **convert_sequences** for Waymo data:

.. code-block:: python

    def convert_sequence(self, sequence_path: Path) -> None:
        dataset = tf.data.TFRecordDataset(sequence_path, compression_type="")

        frames: list[dataset_pb2.Frame] = []
        sequence_name = ""
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if not frames:
                sequence_name = frame.context.name
            frames.append(frame)
            if frame.context.name != sequence_name:
                raise ValueError("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

        self.data_writer = ShardDataWriter(
            self.output_dir / sequence_name,                                             # Output directory path
            sequence_name,                                                               # Container name
            self.get_active_camera_ids([camera for camera in self.CAMERA_MAP.values()]), # Camera IDs
            self.get_active_lidar_ids([lidar for lidar in self.LIDAR_MAP.values()]),     # LiDAR IDs
            self.get_active_radar_ids([]),                                               # Radar IDs
            "waymo-calibration",                                                         # Calibration Type
            "waymo-egomotion",                                                           # Egomotion Type
            sequence_name,                                                               # Sequence ID
            {},  # no generic sequence meta data                                         # Generic Metadata
            0,  # single shard                                                           # Shard ID
            1,                                                                           # Shard Count
            False,                                                                       # Store shard metadata
        )

        self.decode_poses(frames)

        self.decode_lidars(frames)

        self.decode_cameras(frames)

        self.data_writer.finalize()

## Summary

The conversion of Waymo data to NCore can be summarized as follows:

    1. Implement ``from_config`` which creates the custom **DataConverter** given the config
    2. Implement ``get_sequence_paths`` for Waymo data which consists of returning the list ``.tfrecord`` file Paths obtained from the directory provided in the config file
    3. Implement ``convert_sequences`` which is composed of 5 parts

        a. Get frame data from ``.tfrecord`` file using **TensorFlow**'s **TFRecordDataset** class and ``waymo_open_dataset``'s ``dataset_pb2`` API and create a NCore Writer (**ShardDataWriter**) instance
        b. Convert poses by decoding poses
        c. Convert LiDAR data by decoding LiDAR
        d. Convert camera data by decoding cameras
        e. Finalize **ShardDataWriter** to generate and store NCore shard

This approach also applies to other DataConverter implementations and the `convert_sequences` step can be adapted to based on the data provided by the original data source.
Refer to `waymo3.py` for the implementation file.
