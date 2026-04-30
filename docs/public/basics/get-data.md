```{eval-rst}
.. include:: ../_includes/_global_substitutions.rst
```

# Prepare Data for Use with NVIDIA Neural Reconstruction

Before you can generate scenes for use with driving or robotics simulation platforms, you must first validate and convert your real-world driving or robotics data to the standardized NCore data format.

## About NCore

NCore, in the context of NuRec, refers to NVIDIA's standardized data format for autonomous vehicle and robotics datasets. It provides a unified schema for storing sensor data, annotations, and metadata required for neural reconstruction and simulation. The NCore format includes data types that define sensor rig configurations, camera, lidar, and radar setups in the NuRec scenarios, and metadata and session information. Converting data to the standardized NCore format supports consistent, quality reconstruction and the highest quality output through NuRec rendering.

For more information on NCore data types, see [NCore Data Schema](../ncore/types).

## Ensure Data Quality

To generate the highest quality simulations using NVIDIA Neural Reconstruction, you must first ensure that the quality of your input data is high. Follow these best practices to get the best output using NuRec.

- [Ensure Data Quality](../ncore/data-quality)

## Get NCore-standard Data

Select one of the following paths to generate, convert, or get data to use with NuRec:

1. **Get generated, neurally reconstructed scenes from the [NVIDIA Physical AI dataset on HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)** and then follow the instructions to
   [Render the Physical AI dataset with NuRec](../nurec/physical-ai-data).

2. **Convert Waymo data to NCore-formatted data.** Follow the steps in [Convert Waymo Data](../ncore/convert) to download Waymo sample data and convert it to NCore-standard format.

3. **Generate USDZ scenes from PLY data with [3DGRUT](../nurec/3dgrut).**

## Generate Auxiliary Data

Generate additional data required by the reconstruction engine. Follow the steps in [Generate Auxiliary Data](../ncore/nurec-aux-data).

## Validate Your Data Quality

Once your data is in the standard NCore format, use the Data Quality Toolkit to validate the data quality and ensure the best possible reconstruction
and rendering quality in your simulated scenes.
