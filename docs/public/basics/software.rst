.. include:: ../_includes/_global_substitutions.rst

=================================
Software Setup and Prerequisites
=================================

Before you use NVIDIA Neural Reconstruction, set up the appropriate software and access you need on the `recommended hardware setup <../basics/hardware>`_.

Required software
------------------

- `Docker installation <https://docs.docker.com/engine/install/>`_ on a `supported platform <https://docs.docker.com/engine/install/#supported-platforms>`_ (min. version 23.0.1).
- NVIDIA Driver version 560.x.x or higher. `Search for the driver appropriate to your system <https://www.nvidia.com/en-us/drivers/>`_  and then follow the instructions to `install NVIDIA Drivers <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html>`_. 
- NVIDIA Container Toolkit (min. version 1.13.5) `installed and configured <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

Once you have the above prerequisites, follow these steps to set up your NGC authentication and confirm that your system is ready to continue.

1. Confirm that your container runtime supports `NVIDIA GPUs <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration>`_. Run the following command:

.. code-block:: bash

   docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi 

2. Set up your NGC account and get an API key.

   1. Go to `NVIDIA NGC <https://catalog.ngc.nvidia.com//>`_ and create an account.
   2. `Generate an API key <https://org.ngc.nvidia.com/setup/api-key>`_ (navigate to this page from the Setup page, under *Keys/Secrets*, following the **Generate API Key** link).

3. Use the NGC API key to authenticate your Docker container.

.. code-block:: bash

   docker login nvcr.io
   Username: $oauthtoken 
   Password: <NGC API key> 

4. Set the ``NGC_API_KEY`` environment variable for your shell.

.. code-block:: bash

   export NGC_API_KEY=<your-api-key>

5. Install and Configure NGC CLI

You must `download and install the NGC CLI <https://docs.ngc.nvidia.com/cli/cmd.html>`_ on the host system that you're using.

Once you have installed the NGC CLI, set the NGC configuration file:

.. code-block:: bash

   ngc config set

Enter information about the following properties when the system prompts you:

* **Enter API key:** [<VALID_APIKEY>, 'no-apikey']
* **Enter CLI output format type:** ['ascii', 'csv', 'json']
* **Enter org:** [<org1>, <org2>]
* **Enter team:** [<team1>, <team2>]
* **Enter ace:** ['no-ace']

When the process completes, the NGC config is saved and the following output is displayed on the terminal:

.. code-block:: bash

   Validating configuration...
   Successfully validated configuration.
   Saving configuration...
   Successfully saved NGC configuration to /path/to/home/.ngc/config

Once you've completed the prerequisite setup, move to the next steps to reconstruct your driving data:

1. `Prepare Your Data <../ncore/convert>`_
2. `Validate Your Data <../ncore/validate>`_
3. `Run the NuRec Model <../nurec/model>`_
