.. Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

.. _data_conventions: 


Conventions and Specification
=============================

All data has to be stored following these NCore data specifications.
If modules use a different convention internally, the conversion should be done within the module,
and output data should be converted back to this convention.


Coordinate Systems and Transformations
--------------------------------------
There are several coordinate systems that are used in practice -
NCore uses the following conventions.


**Transformations between the Coordinate Systems**

All the transformations are stored in the form of 4x4 ``SE3`` matrices, where
the top left 3x3 elements represent the rotation matrix ``R`` and the
first three rows of the last column denote the translation ``t`` in
meters. They are stored using the convention ``T_a_b``, which denotes the
transformation matrix that transforms the points from the coordinate
system ``a`` to the coordinate system ``b``. For example a point
:math:`\mathbf{p}_a` in coordinate system ``a`` can be transformed to
point :math:`\mathbf{p}_b` as

.. math::
    \mathbf{p}_b = \mathbf{R}_a^b \mathbf{p}_a + \mathbf{t}_a^b


**Global Coordinate Frame**

.. figure:: ecef.png
   :figwidth: 40%
   :width: 50%

The position and orientation of the ego car, as well as other objects in
the scene, are expressed in the earth-centered-earth-fixed (ECEF)
coordinate system in the form of SE3 matrices. To avoid very large
coordinates, we set the first pose of the ego car in the sequence to an
identity matrix and express all other poses relative to it. This
reference pose is available as ``T_rig_world_base``.


**Rig Coordinate Frame**

.. figure:: rig.png
   :figwidth: 40%
   :width: 50%

The ``rig`` coordinate system is defined as a right-handed coordinate
system with the x-axis pointing to the front of the car, y is pointing
left, and z up. The origin of the coordinate system is located in the
middle of the rear axis on the nominal ground. All poses of the ego car
are always expressed as ``T_rig_world`` transformations with two
associated start and end timestamps (e.g., the start- and end-poses
corresponding to a lidar spin).


**Camera and Image Coordinate System**

.. figure:: camera.jpg
   :figwidth: 40%
   :width: 80%

Both extrinsic camera and intrinsic image coordinate systems are
right-handed coordinate systems. The axes of the extrinsic camera
coordinate system are defined such that the camera's principal axis is
along the +z axis, the x-axis points to the right, and the y-axis points
down. The principal point corresponds to the optical center of the
camera.

.. _image_coordinate_conventions:

The image coordinate system is defined such that the u-axis points to
the right and the v-axis down. The origin of the image coordinate system
is in the top left corner of the image, and the units are pixel s.
Continuous pixel coordinates start with ``[0.0, 0.0]`` at the top-left corner of
the top-left pixel in the image, i.e., both the u and v coordinates of
the first pixel span the range ``[0.0, 1.0]``.


.. _v3-data-format:

Shard Data Hierarchy (NCore V3 Data Format)
-------------------------------------------

The per-shard data converted using NCore V3 converters is represented as
*self-contained* shard archives [#f1]_.

Within each shard archive the data is represented in the following
dataset hierarchy [#f2]_:

.. code-block:: text

   /
   │
   ├── {generic_meta_data}...
   │
   ├── poses
   │   ├── T_rig_world_base (4, 4) float64
   │   ├── T_rig_world_timestamps_us (I,) uint64
   │   └── T_rig_worlds (I, 4, 4) float64
   │
   ├── {labels}
   │   └── track_labels () object
   │
   └── sensors
      ├── cameras
      │   ├── camera_front_wide_120fov
      │   ⁞   ├── frame_timestamps_us (J,) uint64
      │   ⁞   │
      │   ⁞   ├── {mask} () |Sx
      │   ⁞   │
      │   ⁞   ├── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   ├── 000000
      │   ⁞   │   ├── T_rig_worlds (2, 4, 4) float32
      │   ⁞   │   ├── timestamps_us (2,) uint64
      │   ⁞   │   │
      │   ⁞   │   ├── image () |Sx
      │   ⁞   │   │      
      │   ⁞   │   ├── {generic_data}...
      │   ⁞   │   └── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   └── 000001...
      │
      ├── lidars
      │   ├── lidar_gt_top_p128_v4p5
      │   ⁞   ├── frame_timestamps_us (K,) uint64
      │   ⁞   │
      │   ⁞   ├── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   ├── 000000
      │   ⁞   │   ├── T_rig_worlds (2, 4, 4) float32
      │   ⁞   │   ├── timestamps_us (2,) uint64
      │   ⁞   │   │
      │   ⁞   │   ├── frame_labels (M,) object
      │   ⁞   │   │
      │   ⁞   │   ├── dynamic_flag (N,) int8 [deprecated - see footnote 4]
      │   ⁞   │   ├── intensity (N,) float32
      │   ⁞   │   ├── timestamp_us (N,) uint64
      │   ⁞   │   ├── xyz_e (N, 3) float32
      │   ⁞   │   ├── xyz_s (N, 3) float32
      │   ⁞   │   │
      │   ⁞   │   ├── {generic_data}...
      │   ⁞   │   └── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   └── 000001...
      │
      └── {radars}
          ├── radar_corner_front_left
          ⁞   ├── frame_timestamps_us (K,) uint64
          ⁞   │
          ⁞   ├── {generic_meta_data}...
          ⁞   │
          ⁞   └── 000000...

For instance, the dataset representing the encoded image of the first
frame of the camera with ID ``camera_front_wide_120fov`` is referenced
by the path ``/sensors/cameras/camera_front_wide_120fov/000000/image``
within this dataset hierarchy.

Pose Data
---------
The timestamped trajectory of the rig frame is stored in the dataset
``/poses/T_rig_worlds`` as SE3 transformation matrices from the rig to
the local world coordinate system (float64, [n,4,4]) with associated
timestamps ``/poses/T_rig_world_timestamps_us`` (in microseconds
(uint64, [n])).

All of these poses are relative to the single base pose stored in the
dataset ``/poses/T_rig_world_base`` (float64, [4,4], see details above).

Sensor Data
-----------

**Images**

Camera-associated image data is saved either in binary `jpeg` (raw
sensor data) or in the `png` (masks / semantic segmentations) formats.
The concrete format of image data is available in the dataset
attributes, e.g.,
``/sensors/cameras/camera_front_wide_120fov/mask.attrs['format']`` is
the image format of the static mask associated with the camera
``camera_front_wide_120fov``.

**Point clouds**

Lidar and radar point clouds are saved as compressed and named per-point
datasets per frame.

Lidar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the *start* of the ray in the sensor's
  end-of-spin reference frame (float32, [n,3])
* ``xyz_e`` - Motion-compensated 3D coordinate of the *end* of the ray
  in the sensor's end-of-spin reference frame (float32, [n,3])
* ``intensity`` - measured intensity (float32, [n]): normalized to
  ``[0.0, 1.0]`` range
* ``dynamic_flag`` - dynamic flag (int8, [n]): ``-1`` if not evaluated,
  ``0`` for static points, and ``1`` for dynamic points. Note: points
  are classified as dynamic if the associated object-track is dynamic
  at *any* point in time of the sequence [deprecated [#f3]_]
* ``timestamp_us`` - point timestamp in microseconds (uint64, [n])

Radar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the start of the ray in the sensor's
  end-of-frame reference frame (float32, [n,3])
* ``xyz_e`` - 3D coordinate of the end of the ray in the sensor's
  end-of-frame reference frame (float32, [n,3])

Additional radar readings (like radial velocities / radar-cross-section
measurements) are exposed in dataset-specific generic per-frame data.

**Metadata**

Metadata is available per frame, but also for individual sensors and for
the general sequence.

Per-frame metadata contains the following datasets for all sensors and
all frames:

* ``timestamps_us`` - timestamps of the frame's start and end point in
  microseconds (uint64, [2,])
* ``T_rig_worlds`` - SE3 transformation matrices from the rig to the
  world coordinate system at the start and end timestamp of the frame
  (float32, [2,4,4])
* ``generic_data`` - a group with non-enforced dataset members
  (usually data-source dependent, encoding data-source specific
  per-frame data not common to all data-sources)

For individual sensor types, we also save sequence-wide sensor-specific
meta data either as attributes of the sensor-group or individual datasets:

*All Sensors*:

* ``T_sensor_rig``- SE3 transformation matrix from the sensor to the rig
  coordinate system (float32, [4,4])
* ``frame_timestamps_us`` - end-of-frame timestamps of all the sensor's
  frames in microseconds (uint64, [J,])
* ``generic_meta_data`` - a group with non-enforced members
  (usually data-source dependent, encoding data-source specific sensor
  meta data not common to all data-sources)

*Cameras*:

* ``camera_model_type`` - camera model type (str, one of [ftheta,
  opencv-pinhole, opencv-fisheye])

The field ``camera_model_parameters`` will unconditionally contain:

* ``resolution`` - width and height of the image in pixels (uint32,
  [2,])
* ``shutter_type`` - shutter type of the camera's imaging sensor (str, one of
  [ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT, ROLLING_BOTTOM_TO_TOP,
  ROLLING_RIGHT_TO_LEFT, GLOBAL])

If ``camera_model_type = 'ftheta'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the NVIDIA default convention for FTheta camera models
  in which the pixel indices represent the center of the pixel
  (not the top-left corners). NOTE: principal point coordinates
  will be adapted internally in camera model APIs to reflect
  the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])

* ``reference_poly`` - indicating which of the two polynomials is the
  *reference* polynomial - the other polynomial is only an approximation
  of the inverse of the reference polynomial (str, one of
  [PIXELDIST_TO_ANGLE, ANGLE_TO_PIXELDIST])
* ``pixeldist_to_angle_poly`` - coefficients of the backward distortion
  polynomial (conditionally approximate, depending on
  ``reference_poly``), mapping pixel-distances to angles [rad] (float32,
  [6,])
* ``angle_to_pixeldist_poly`` - coefficients of the forward distortion
  polynomial (conditionally approximate, depending on
  ``reference_poly``), mapping angles [rad] to pixel-distances (float32,
  [6,])
* ``max_angle`` - maximal extrinsic ray angle [rad] with the principal
  direction (float32)
* ``linear_cde`` - Coefficients of the constrained linear term
  :math:`\begin{bmatrix} c & d \\ e & 1 \end{bmatrix}` transforming between
  sensor coordinates (in mm) to image coordinates (in px) (float32, [3,])

If ``camera_model_type` = 'opencv-pinhole'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])
* ``focal_length`` - focal lengths in u and v direction, resp., mapping
  (distorted) normalized camera coordinates to image coordinates relative
  to the principal point (float32, [2,])
* ``radial_coeffs`` - radial distortion coefficients
  ``[k1,k2,k3,k4,k5,k6]`` parameterizing the rational radial distortion
  factor :math:`\frac{1 + k_1r^2 + k_2r^4 + k_3r^6}{1 + k_4r^2 + k_5r^4
  + k_6r^6}` for squared norms :math:`r^2` of normalized camera
  coordinates (float32, [6,])
* ``tangential_coeffs`` - tangential distortion coefficients ``[p1,p2]``
  parameterizing the tangential distortion components
  :math:`\begin{bmatrix} 2p_1x'y' + p_2 \left(r^2 + 2{x'}^2 \right) \\
  p_1 \left(r^2 + 2{y'}^2 \right) + 2p_2x'y' \end{bmatrix}` for
  normalized camera coordinates :math:`\begin{bmatrix} x' \\ y'
  \end{bmatrix}` (float32, [2,])
* ``thin_prism_coeffs`` - thins prism distortion coefficients
  ``[s1,s2,s3,s4]`` parameterizing the thin prism distortion components
  :math:`\begin{bmatrix} s_1r^2 + s_2r^4 \\ s_3r^2 + s_4r^4
  \end{bmatrix}` for squared norms :math:`r^2` of normalized camera
  coordinates (float32, [4,])

If ``camera_model_type` = 'opencv-fisheye'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])
* ``focal_length`` - focal lengths in u and v direction, resp., mapping
  (distorted) normalized camera coordinates to image coordinates relative
  to the principal point (float32, [2,])
* ``radial_coeffs`` - radial distortion coefficients representing
  OpenCV-style ``[k1,k2,k3,k4]`` parameters of the
  fisheye distortion polynomial :math:`\theta(1 + k_1\theta^2 +
  k_2\theta^4 + k_3\theta^6 + k_4\theta^8)` for extrinsic camera ray
  angles :math:`\theta` with the principal direction (float32, [4,])
* ``max_angle`` - maximal extrinsic ray angle [rad] with the principal
  direction (float32)

Finally, we also save general metadata related to the session (input
data and versioning):

* ``version`` - version of the dataset (str)
* ``egomotion_type`` - type of ego-motion that was used to generate the
  data (str)
* ``calibration_type`` - type of sensor calibration that was used to
  generate the data (str)
* ``camera_ids`` / ``lidar_ids`` / ``radar_ids`` - individual lists of
  sensor names processed available in the data ([str])
* ``sequence_id`` - the identifier of the source dataset, if available
  (str)
* ``shard_id`` - the index of the current shard within all shards
  generated for a given sequence
* ``shard_count`` - the total number of shards generated for a given
  sequence

Labels
------

**4D autolabels**

Annotation data is stored in a segmented format:

* All observations of each unique track instance is stored in a
  session-wide format in the ``/labels/track_labels`` dataset by
  *references* to the individual sensor frames that are observing the
  track's instance.
* Individual time-associated instances of each track observation (e.g.,
  3D bounding boxes in lidar frames or 2D bounding boxes in camera
  frames) are stored in the per-frame meta data of each sensor's frame.
  For instance, the dataset
  ``/sensors/lidars/lidar_gt_top_p128_v4p5/000000/frame_labels`` contains
  the observed track instances in the first frame of a given lidar.

.. rubric:: Footnotes

.. [#f1] NCore V3 data shards are represented by
         `zarr <https://zarr.readthedocs.io/en/stable/>`_
         groups within a custom ``.itar`` archive format that can be
         loaded most easily using the ``ShardDataLoader`` type, which
         also supports shard concatenation to reconstruct full source
         sequences.
.. [#f2] Curly brackets denote optional data.
.. [#f3] Deprecated property, might be available as generic frame
         data still
