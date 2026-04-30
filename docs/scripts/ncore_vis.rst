.. Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

.. _ncore_visualizer:

NCore Visualizer Tool
=====================

The NCore Visualizer tool located at `scripts/ncore_vis` is an interactive 3D tool to visualize ncore data. It utilizes the
viser web framework to render data in real-time. To run the visualizer, refer to the follow command::

    bazel run //internal/scripts/ncore_vis:ncore_vis \
        -- \
        --shard-file-pattern=<path-to-ncore-shard> \

This launches the web app and a HTTP url (and websocket) are provided in the command line. Open the link to use the visualizer

.. figure:: ncore_vis_sample.png


Usage
-----

This visualization tool takes in an NCore Shard and produces an interactive visualization with a control panel on the top right of the screen.
There are 3 tabs,

    1. Cameras (Provides visualization control over the camera sensors)
    2. Lidars (Provides visualization control over the lidar sensors)
    3. Scene (Provides visualization control over the entire scene)

**Camera Tab**

The Camera tab allows control of the camera sensors including position, frame, data type, etc. You can go to each camera to visualize
the data at a given frame.

Sample RGB Camera data

.. figure:: ncore_vis_rgb_camera_data.png

Sample Semantic Camera 

.. figure:: ncore_vis_semantic_camera_data.png

**Lidar Tab**

The Lidar tab allows control of data provided by the lidar which includes the point cloud frame, ghosting (capturing pcs of frames close to the current frame), 
bounding boxes, or even fusing multiple frames to create a fused point cloud and fused bounding boxes.

Sample Point Cloud

.. figure:: ncore_vis_pc.png

Sample Fused Point Cloud

.. figure:: ncore_vis_fused_pc.png

Sample Point Cloud with Intensity

.. figure:: ncore_vis_pc_with_intensity.png

Sample Bounding boxes

.. figure:: ncore_vis_bboxes.png

**Scene Tab**

The Scene tab provides synchronization tooling. Specifically, it allows the user to select a reference sensor and set the frames of
all other sensors that are closes to the selected frame.


