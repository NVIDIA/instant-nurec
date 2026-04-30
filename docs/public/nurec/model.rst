.. include:: ../_includes/_global_substitutions.rst

===========================
Use Neural Reconstruction
===========================

To use |nurec| to reconstruct real-world scenes as 3D simulations, follow the steps in this guide.

**Before you begin,** make sure you have completed all the steps in the `Set Up Your Environment <../setup>`_ guide and `prepared your data <basics/get-data>`_.


Download the |nurec| Container
------------------------------

To download the |nurec| container, run the following command:

.. code-block:: bash

   docker pull nvcr.io/nvidia/nre/nre:latest

Launch the Reconstruction Model Training
----------------------------------------

To begin training the reconstruction model in the Docker container, edit the following command to reflect the correct variables and then run it:

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
   -e NGC_API_KEY=${NGC_API_KEY} \
   --volume /path/to/dataset/top/directory:/workdir/dataset \
   --volume /path/to/output/directory:/workdir/output \
   nvcr.io/nvidia/nre/nre:latest \
   --config-name=configs/apps/AV/Waymo/3dgut_dynamic.yaml \
   mode=trainval \
   dataset.path=/workdir/dataset/<DATASET_NAME>.json \
   out_dir=/workdir/output \
   logger=tensorboard

**Notes:**

* Copy/move all the auxiliary ``.zarr`` files with ``.aux`` added to the filename (e.g. ``<DATASET_NAME>.aux.sseg.zarr`` and ``<DATASET_NAME>.aux.depth.zarr``) to the same directory as the ``.zarr.itar`` and ``.json`` file from step 1.
* Update the ``--volume /path/to/dataset/top/directory:/workdir/dataset`` flag to use the path to the folder containing the auxiliary ``.zarr`` files and ``.json`` file.

.. _multi-gpu-nurec:

Run NuRec on Multi-GPU Systems
******************************

To run |nurec| on multi-GPU systems, append the following flags to the launch command:

.. code-block:: bash

   trainer.world_size=<NUM_GPUS> trainer.num_nodes=<NUM_NODES>

**Notes:**

* ``<NUM_GPUS>`` is the number of GPUs to use for training.
* ``<NUM_NODES>`` is the number of nodes to use for training.
* You can assign four tasks per node, and 1 GPU per task, so, if you have 8 GPUs, you can use 2 nodes.
* By default, NuRec uses only the first visible GPUs. When you set ``trainer.world_size=0`` and ``trainer.num_nodes=0``, NuRec will try to
  utilize all the visible GPUs and compute nodes (operating as a SLURM environment). To limit the visible GPUs, use the
  `CUDA_VISIBLE_DEVICES <https://docs.nvidia.com/deploy/topics/topic_5_2_1.html>`_ environment variable and identify the GPUs by ID.
* If you specify the available GPUs, the ``trainer.world_size`` flag pulls the GPUs in the order they are specified in the
  ``CUDA_VISIBLE_DEVICES`` environment variable. For example, if you have 6 GPUs and you specify ``CUDA_VISIBLE_DEVICES=1,2,3,4,5,0`` and
  pass the ``trainer.world_size=4`` flag, NuRec uses the GPUs that correspond to the IDs ``1,2,3,4``.


Optional Configuration
**********************

* There are multiple modes available for the reconstruction: ``train``, ``val``, ``trainval``.

   * ``train``: Runs training steps for reconstruction
   * ``val``: Runs validation using a trained checkpoint
   * ``trainval``: Runs both training and validation steps

* The number of training epochs is set to 30 by default. Append the following option to override the epoch value:

   .. code-block:: bash

      trainer.max_epochs=N

* For reconstruction on a subset of cameras, append the following option at the end of the launch command:

   .. code-block:: bash

      dataset.camera_ids="['<ID1>','<ID2>','<ID3>']"

   **Note:** There should not be any space between the IDs and commas.

* Increase the debug log level by appending the following option at the end of the launch command:

   .. code-block:: bash

      +log_level=N

   Where N can be any of the following options:

   * For FATAL errors only, use ``0``
   * For ERROR messages and fatal errors, use ``1``
   * For WARNING (and errors), use ``2``
   * For INFO (and lower verbosity levels), use ``3``
   * For DEBUG (all logs), use ``4``

Details about the artifacts from the training
*********************************************

The training pipeline creates two types of artifacts:

* The full configuration used for training, named ``parsed.yaml``
* Checkpoints from the training epochs

The validation pipeline creates three types of artifacts:

* A ``metrics.yaml`` file containing per-frame metrics and other details
* A depth map per frame and its corresponding video file
* An opacity map per frame and its corresponding video file
* A segmentation map per frame and its corresponding video file

There is also a folder with log files from both the training and validation pipelines.

Reconstruct Novel Views via Validation Pipeline
------------------------------------------------

You can also generate novel views, using the X-, Y-, and Z-axis shifts, as shown in the following figure.

.. image:: /images/novel_view_synthesis.png

To run validation on the trained model on your docker container, use the following command:

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
     -e NGC_API_KEY=${NGC_API_KEY} \
     --volume /path/to/dataset:/workdir/dataset \
     --volume /path/to/output/:/workdir/output \
     nvcr.io/nvidia/nre/nre:latest \
     --config-name=/workdir/output/config/parsed.yaml \
     mode=val \
     resume=/workdir/output/checkpoints/last.ckpt \
     out_dir=/workdir/output \
     dataset.val_sensor_transl_delta_m="[0,2,0]"

**Note:**

* Here the ``/path/to/output`` is different from the training step. It should point to the internal top directory containing the artifacts from the training.
* This step may ask you to configure ``wandb``. You may select option (3) to skip the configuration step.

For novel view synthesis:

* You can use ``dataset.val_sensor_transl_delta_m="[x,y,z]"`` to provide shifts in translation. It accepts values in meters. There should not be any space between the values.
* You can use ``dataset.val_sensor_rot_delta_deg="[degree1, degree2, degree3]"`` to provide roll-pitch-yaw Euler angles relative to the car as shown in the picture above. There shouldn't be any space between the values.

Details about the artifacts from the training
*********************************************

The validation pipeline creates the following artifacts:

* Validation creates a file named ``metrics.yaml`` in the ``/path/to/output/val`` directory to store metrics.
* You can check the PSNR vs training view in ``metrics.yaml``. It should be under the ``test/psnr`` field in the YAML file.
* MP4 files to show the base reconstruction, depth map, opacity.
* Individual images to show the base reconstruction, depth map, opacity per frame.

Export Utilities
----------------

There are a few export utilities available in the |nurec| container that can be used to export the following from the trained model:

* PLY (Polygon File Format, ``.ply``) files
* Ego masks
* Sequence tracks of cuboid tracjectories 
* NCore tracks (per sensor per frame pose information) of ego vehicle

Generate PLY files
******************

Use the following command to generate PLY files from the trained reconstruction models.

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    --volume /path/to/dataset/<DATASET NAME>:/workdir/dataset \
    --volume /path/to/output:/workdir/output \
    nvcr.io/nvidia/nre/nre:latest \
    export-gaussian-plys \
    --config-name /workdir/output/<RUN ID>/config/parsed.yaml \
    --checkpoint-name last.ckpt

.. note::
   - <RUN ID> is a unique ID used as the subfolder's name inside the output directory.
   - The destination path for the dataset directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 
   - The destination path for the output directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 

Generate Ego Masks
******************

Use the following command to generate ego masks from the trained reconstruction models.

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    --volume /path/to/dataset/<DATASET NAME>:/workdir/dataset \
    --volume /path/to/output:/workdir/output \
    nvcr.io/nvidia/nre/nre:latest \
    export-ego-mask \
    --shard-file-pattern "/workdir/dataset/<DATASET NAME>.zarr.itar" \
    --output-dir "/workdir/output" \
    --camera-ids <ID1> \
    --camera-ids <ID2>

Generate Sequence Tracks
*************************

Use the following command to generate sequence tracks from the trained reconstruction models.

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    --volume /path/to/dataset/<DATASET NAME>:/workdir/dataset \
    --volume /path/to/output:/workdir/output \
    nvcr.io/nvidia/nre/nre:latest \
    export-sequence-tracks \
    --config-name "/workdir/output/<RUN ID>/config/parsed.yaml" \
    --checkpoint-name "/workdir/output/<RUN ID>/checkpoints/last.ckpt" \
    --format=json \
    --controllable-only \
    --output-dir "/workdir/output" \
    resume="/workdir/output/<RUN ID>/checkpoints/last.ckpt"

.. note::
   - <RUN ID> is a unique ID used as the subfolder's name inside the output directory.
   - The destination path for the dataset directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 
   - The destination path for the output directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 

Generate NCore Tracks
*********************

Use the following command to generateNCore tracks (containing per sensor per frame pose information) from theNCore dataset.

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    --volume /path/to/dataset/<DATASET NAME>:/workdir/dataset \
    --volume /path/to/output:/workdir/output \
    nvcr.io/nvidia/nre/nre:latest \
    export-ncore-tracks \
    --shard-file-pattern "/workdir/dataset/<DATASET NAME>.zarr.itar" \
    --model-tracks-json "/workdir/output/sequence_tracks.json" \
    --output-dir "/workdir/output" \
    --camera-id <ID1> \
    --camera-id <ID2> \
    --lidar-id <ID3>

.. note::
   - It requires the sequence tracks from the previous section and theNCore dataset as inputs.
   - The destination path for the dataset directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 
   - The destination path for the output directory must be the same destination path used in the training step because the export command reads the path of the artifacts from parsed.yaml. 

Help
----

You can run help command to get more information about training, validation, and the export utilities.

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    nvcr.io/nvidia/nre/nre:latest --help

.. code-block:: bash

   docker run --shm-size=64g -it --rm --gpus all \
    -e NGC_API_KEY=${NGC_API_KEY} \
    nvcr.io/nvidia/nre/nre:latest \
    <export-utility-name> \
    --help