.. include:: ../_includes/_global_substitutions.rst

============================
Refine Rendering with Fixer
============================

Fixer is a post-trained model trained to perform artifact removal in Neural Reconstruction tasks. The original model is derived from the Sana
model. The Fixer-Cosmos model leverages NVIDIA Cosmos and is available in three variants: base, light, and ultra-light.

Fixer is meant to assist developers of Autonomous Vehicles in their efforts to enhance and improve Neural Reconstruction pipelines.
The model takes an image as an input and outputs a fixed image.

**References**

* `Model Card++ <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer>`_ on NGC.
* `Sana GitHub Repo <https://github.com/NVlabs/Sana%20>`_
* `Sana project on GitHub <https://nvlabs.github.io/Sana/%20>`_
* Sana research paper: `SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers <https://arxiv.org/abs/2410.10629>`_.

Prerequisites
#############

Set Up Required Hardware
-------------------------

+--------------------+----------------+------------+
| Model Variant      | GPU Memory (GB)| # of GPUs  |
+====================+================+============+
|       FP32         | >= 24 GB       | 1          |
+--------------------+----------------+------------+
|       BF16         | >= 12 GB       | 1          |
+--------------------+----------------+------------+
| cosmos_base        | >= 3 GB        | 1          |
+--------------------+----------------+------------+
| cosmos_light       | >= 2 GB        | 1          |
+--------------------+----------------+------------+
| cosmos_ultra_light | >= 2 GB        | 1          |
+--------------------+----------------+------------+
| cosmos_3dgut       | >= 3 GB        | 1          |
+--------------------+----------------+------------+


Set Up Required Software
-------------------------

* `Docker <https://docs.docker.com/engine/install/>`_ - minimum version: 23.0.1 (requires `OS that supports Docker <https://docs.docker.com/engine/install/#supported-platforms>`_)
* `NVIDIA Drivers <https://www.nvidia.com/en-us/drivers/>`_ - minimum version: 535. You can use the NVIDIA driver release 535.86 (or later R535), or 545.23 (or later R545) which are supported on L40S DataCenter GPUs.
* `Install <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit%3E>`_ and `configure <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration>`_ the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html>`_ - minimum version: 1.13.5
* Verify your container runtime supports NVIDIA GPUs by running the following:

.. code-block:: bash

    docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

For more information on enumerating multi-GPU systems, please see the NVIDIA Container Toolkit's `GPU Enumeration Docs <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html?highlight=enumerate#gpu-enumeration>`_.

Create and Configure NGC Access
-------------------------------

You must have an authenticated NGC (NVIDIA GPU Cloud) account with access to the model checkpoint. Use the following procedure to log in to NGC, and set the NGC_API_KEY environment variable.

1. Create an account on `NGC <https://catalog.ngc.nvidia.com/>`_.
2. Generate an `API Key <https://org.ngc.nvidia.com/setup/api-key>`_. The following steps require your NGC API key.
3. Authenticate local Docker with NGC by running the following code. For more details, see the `NGC authentication documentation <https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html>`_.

.. code-block:: bash

    docker login nvcr.io
    Username: $oauthtoken
    Password: <NGC API key>

4. Set the NGC_API_KEY environment variable in your shell.

.. code-block:: bash

    export NGC_API_KEY=<NGC API key>

Use the Fixer Model
####################

Multiple versions of the model are available for download. Choose the model that best suits your needs.


.. list-table::
   :header-rows: 1

   * - Model Precision
     - Model Name
     - Model Tag
     - Model Size
   * - FP32
     - ``nurec-fixer``
     - ``0.1_fp32``
     - 17 GB
   * - BF16
     - ``nurec-fixer``
     - ``0.1_bf16``
     - 8.5 GB
   * - BF16
     - ``nurec-fixer``
     - ``cosmos_base``
     - 1.45 GB
   * - BF16
     - ``nurec-fixer``
     - ``cosmos_light``
     - 1.45 GB
   * - BF16
     - ``nurec-fixer``
     - ``cosmos_ultra_light``
     - 1.45 GB
   * - BF16
     - ``nurec-fixer``
     - ``cosmos_3dgut``
     - 1.45 GB

.. tip::

   - **For better performance on slower GPUs,** use the light and ultra-light models, ``cosmos_ultra_light``, ``cosmos_light``, and ``cosmos_based``.
   - **For better inference and 3DGUT-trained models,** use the ``cosmos_3dgut`` model.


The Fixer model takes the following inputs and produces the following outputs:

.. tab-set::

    .. tab-item:: Fixer
        :sync: original

        * **Input Format**: RGB
        * **Input Resolution**: 1024 x 576
        * **Input Precision**: FP32, BF16
        * **Output Format**: RGB
        * **Output Resolution**: 1024 x 576
        * **Output Precision**: FP32


    .. tab-item:: Fixer-Cosmos
        :sync: cosmos

        * **Input Format**: RGB
        * **Input Resolution**: 1024 x 576, 832 x 448, 704 x 384, 512 x 288
        * **Input Precision**: BF16
        * **Output Format**: RGB
        * **Output Resolution**: 1024 x 576, 832 x 448, 704 x 384, 512 x 288
        * **Output Precision**: BF16


Download and Run the Model
---------------------------

1. Download the model with the following command:

.. code-block:: bash

    ngc registry model download-version "nvidia/nre/nurec-fixer:<model_tag>" --org
    <org>

2. Download the docker container with the following command:

.. code-block:: bash

    docker pull nvcr.io/nvidia/pytorch:latest

3. To launch the docker container, update the ``/path/to/working/directory`` portion of the following command and then run it:

.. code-block:: bash    

    docker run --rm --runtime=nvidia --gpus all --shm-size 2g \
    -v /path/to/working/directory:/path/to/working/directory:rw \
    -it nvcr.io/nvidia/pytorch:latest


Check the Model Quality
-----------------------

Use the provided inference script, ``infer.py`` and corresponding model weights to run the model inference on a set of input images.

The inference script takes a directory of JPG and/or PNG images and a location of the model weight, and generates JPG and/or PNG images as output.

Use the following command to run the inference script inside the docker container:


.. tab-set::

    .. tab-item:: Fixer
        :sync: original

        .. code-block:: bash

            python3 infer.py --image_path /path/to/images/ --output_path /path/to/output/
            --model_path /path/to/cosmos_difix.pt


    .. tab-item:: Fixer-Cosmos
        :sync: cosmos

        **Before you begin,** rename the inference script to ``infer_cosmos.py``.

        .. code-block:: bash

            # Run the inference script for cosmos_base

            python3 infer_cosmos.py --model_path "cosmos_base.pt" \
            --image_path "input folder" \
            --output_path "output folder" \
            --res 1024

            # Run the inference script for cosmos_light

            python3 infer_cosmos.py --model_path "cosmos_light.pt" \
            --image_path "input folder" \
            --output_path "output folder" \
            --res 704

            # Run the inference script for cosmos_ultra_light

            python3 infer_cosmos.py --model_path "cosmos_ultra_light.pt" \
            --image_path "input folder" \
            --output_path "output folder" \
            --res 512

            # Run the inference script for cosmos_3dgut

            python3 infer_cosmos.py --model_path "cosmos_3dgut.pt" \
            --image_path "input folder" \
            --output_path "output folder" \
            --res 1024
