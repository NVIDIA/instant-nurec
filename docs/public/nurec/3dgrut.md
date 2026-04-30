# Use the 3DGRUT Model to Generate Scenes

You can use software like NVIDIA’s Neural Reconstruction and the open-source 3DGRUT to produce gaussian-based reconstructions from image collections. After outputting to a custom USDZ-based format that uses an extension of the UsdVolVolume Schema, you can load the resulting reconstructions as a scene. You can then make adjustments to the scene, such as adding a `proxy` mesh to serve as the ground if there isn’t one already.

Follow the steps in this guide and on the 3DGRUT repository to train scenes using the 3DGRUT models.

1. Follow the instructions to clone and set up [3DGRUT](https://github.com/nv-tlabs/3dgrut/blob/main/README.md).

2. To generate the USDZ data output, run one of the following scripts from your local clone of the repo:
   - **For 3DGRUT data:**
   ```python
   python train.py --config-name apps/colmap_3dgut.yaml path=path/to/your/data out_dir=runs experiment_name=garden_3dgut dataset.downsample_factor=2 export_usdz.enabled=true
   ```
   - **For 3DGS PLY data:**
   ```python
   python ply_to_usd.py /path/to/your/model_3dgs.ply --output_file /path/to/output.usdz
   ```

**Note:** The data conversion script does not currently output any specific Gaussian USD standard, and instead leverages the `Volume` class from the `USDVol` namespace. It is still actively under development and subject to change. If you encounter errors on future attempts to run the script, check this doc for updates.

Once you've generated your USDZ scenes, load them in a compatible simulation platform.
