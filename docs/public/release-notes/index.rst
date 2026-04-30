=============
Release Notes
=============

Current Release
================

Version 25.11 — December 4, 2025
---------------------------------

Major Features and Updates
**************************
- **Configurable Run Binaries:** Added configurable run binary and repository paths that are configurable
  rather than hardcoded, with enhanced robustness of script line replacements and informative warnings for
  missing or mismatched tasks.
- **Asset Harvesting Pipeline Harvested Asset Modification:** Improved Asset Harvesting pipeline and
  added support for the modification and inclusion ofharvested assets in reconstructions. See
  `Use Asset Harvester Output in Reconstructions <../nurec/use-ah-assets>`_ for more details.


Bug Fixes
**********

- **Bilateral Grid Inference:** Fixed RGB tensor contiguity issue to ensure proper functioning of bilateral grid
  operations by returning RGB as contiguous tensors
- **Asset Harvesting RGB2SH Calculation:** Removed incorrect rgb2sh conversion from particle radiance calculation
  during 3D reconstruction, correcting the radiance calculations in the Asset Harvesting workflow.
- **Render GRPC Asset Editing:** Fixed undefined ``remove`` set error when not using ``edit-assets.json`` in
  ``render-grpc``, preventing errors in gRPC rendering when asset editing is not used.


Past Releases
=============

.. dropdown:: Version 25.09 — October 15, 2025

    .. include:: _includes/25-09.md
        :start-line: 5
 
.. dropdown:: Version 25.08 — August 25, 2025

    .. include:: _includes/25-08.md
        :start-line: 3

.. dropdown:: Version 25.07 — July 28, 2025

    .. include:: _includes/25-07.md
        :start-line: 3

.. dropdown:: Version 25.06 — June 30, 2025

    .. include:: _includes/25-06.md
