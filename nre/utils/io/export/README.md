# Export Commands

This directory contains export utility commands for the NuRec library.

## `export-parsed-config`

This command is used to more easily export a parsed config from the existing config files or an USDZ artifact file. This can then be compared with the parsed config from a different nre version to help check what changes were done between them, and then use this information to write the appropriate config upgrade function. The parsed configuration is printed to standard output (`stdout`) or written to a file.

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-parsed-config --config-name <config_name> [-o <output_file>] [--sort-keys/--no-sort-keys] [-- hydra_args...]
bazel run //:run -- export-parsed-config --input <artifact.usdz> [-o <output_file>] [--sort-keys/--no-sort-keys]
```

**Docker:**

```bash
docker run nre_run export-parsed-config --config-name <config_name> [-o <output_file>] [--sort-keys/--no-sort-keys] [-- hydra_args...]
docker run nre_run export-parsed-config --input <artifact.usdz> [-o <output_file>] [--sort-keys/--no-sort-keys]
```

**Arguments:**

- `-c, --config-name`: Path to the input config file that needs to be parsed. You can find the available configs in the `configs` directory. The `.yaml` extension is optional.
- `-i, --input`: Path to the input artifact file (`.usdz`).
- `-o, --output`: (Optional) Path to the output file. When specified, a confirmation message is printed to standard output upon completion. If not provided, the parsed config is printed to standard output.
- `--sort-keys/--no-sort-keys`: Whether or not to sort the dictionary before exporting. It is recommended to sort (default) to make it comparable with other (sorted) configs.
- `--upgrade`: Enable upgrade mode to upgrade the config to the current version before exporting.
- `hydra_args`: (Optional) Any additional hydra overrides. Only applicable when using `--config-name`.

### Examples

**Using the CLI command:**

To parse a config file and save the output to `parsed_config.yaml`:

```bash
bazel run //:run -- export-parsed-config --config-name configs/path/to/config_file --output parsed_config.yaml
```

To export the parsed config from an artifact and print it to the console:

```bash
bazel run //:run -- export-parsed-config --input /path/to/my_artifact.usdz
```

To parse a config with Hydra overrides and upgrade to current version:

```bash
bazel run //:run -- export-parsed-config --config-name configs/my_config.yaml --upgrade -- dataset.batch_size=32
```

**Using Docker:**

```bash
docker run nre_run export-parsed-config --config-name configs/my_config.yaml --output parsed_config.yaml
```

## `export-artifact-structure`

This command inspects the model checkpoint (`checkpoint.ckpt`) inside a `.usdz` artifact or a raw PyTorch checkpoint file and outputs the structure of all tensors as a hierarchical JSON object. For each tensor, the JSON output includes its shape, data type (`dtype`) and device.

This is useful for creating model upgrade functions, as it allows you to compare the checkpoint structure between two different NRE versions and identify what has changed (e.g., renamed layers, modified tensor shapes).

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-artifact-structure --input <input_file> [-o <output.json>]
```

**Docker:**

```bash
docker run nre_run export-artifact-structure --input <input_file> [-o <output.json>]
```

**Arguments:**

- `-i, --input`: (Required) Path to the input file. Can be a `.usdz` artifact or a PyTorch checkpoint file (`.ckpt`).
- `-o, --output`: (Optional) Path to the output JSON file. When specified, a confirmation message is printed to standard output upon completion. If not provided, the JSON is printed to standard output.

### Examples

**Using the CLI command:**

To inspect an artifact and print to console:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/my_artifact.usdz
```

To inspect a checkpoint and save the output to a file:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/checkpoint.ckpt --output structure.json
```

To inspect an artifact and save for comparison:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/my_artifact.usdz --output my_artifact_structure.json
```

**Using Docker:**

```bash
docker run nre_run export-artifact-structure --input /path/to/my_artifact.usdz --output structure.json
```

## `export-external-assets`

This command merges external assets from the asset harvester output into an existing NRE artifact (`.usdz` file) and generates an `edit-assets.json` configuration file for use with `render-grpc`.

The command validates that the track IDs in the external assets match the controllable tracks in the artifact, copies the PLY files with the correct directory structure, and creates a new artifact with the external assets included.

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-external-assets --artifact-path <artifact.usdz> --external-assets-path <asset_harvester_output> --output-artifact-path <output.usdz> --output-edit-file <edit-assets.json>
```

**Docker:**

```bash
docker run nre_run export-external-assets --artifact-path <artifact.usdz> --external-assets-path <asset_harvester_output> --output-artifact-path <output.usdz> --output-edit-file <edit-assets.json>
```

**Arguments:**

- `--artifact-path`: (Required) Path to the input `.usdz` artifact file.
- `--external-assets-path`: (Required) Path to the asset harvester output directory containing the external assets and `metadata.yaml`.
- `--output-artifact-path`: (Required) Path for the output `.usdz` artifact file with external assets included.
- `--output-edit-file`: (Required) Path for the `edit-assets.json` output file for use with `render-grpc`.

### Examples

**Using the CLI command:**

To merge external assets into an artifact:

```bash
bazel run //:run -- export-external-assets \
    --artifact-path /path/to/original.usdz \
    --external-assets-path /path/to/asset_harvester_output \
    --output-artifact-path /path/to/output_with_assets.usdz \
    --output-edit-file /path/to/edit-assets.json
```

**Using Docker:**

```bash
docker run nre_run export-external-assets \
    --artifact-path /path/to/original.usdz \
    --external-assets-path /path/to/asset_harvester_output \
    --output-artifact-path /path/to/output_with_assets.usdz \
    --output-edit-file /path/to/edit-assets.json
```

The command will:

1. Load track IDs from both the artifact and external assets
2. Validate that external asset track IDs match the artifact's controllable tracks
3. Create a new artifact with the external assets in the correct structure (`external_assets/{track_id}/gs.ply`)
4. Generate an `edit-assets.json` file with the "replace" field containing the valid track IDs

## `export-gaussian-usd-asset`

This command exports Gaussian splatting model checkpoints to USD format. It supports two USD schema types:

- **geompoints**: Uses `UsdGeomPoints` primitives with custom primvars for Gaussian attributes
- **lightfield**: Uses the UsdVol ParticleField schema (`ParticleField3DGaussianSplat` or `ParticleField` with surflet API). Requires **usd-core>=26.3** for the UsdVol schema API. Supports split half-precision (geometry vs features).

The export can produce individual USD files (`.usda`, `.usdc`, `.usd`) or a packaged USDZ archive containing all assets.

### Features

- **Schema Selection**: Choose between `geompoints` (legacy) or `lightfield` (modern) USD schemas
- **Half-Precision Export**: Reduce file size using float16 attributes (LightField schema only)
- **Rigid Track Animation**: Export tracked objects with animated transforms
- **Deformable Gaussians**: Export models with animated positions, rotations, and scales
- **Temporal Appearance**: Export animated spherical harmonics / albedo coefficients
- **Environment Maps**: Automatically exports `SkyEnvMapBackground` as USD DomeLight with HDR texture
- **Rig Trajectories**: Optionally include camera/rig trajectory data
- **Gaussian Subsampling**: Reduce exported Gaussian count by percentage

### Usage

**Bazel:**

```bash
bazel run //:run -- export-gaussian-usd-asset \
    --config-name <config_name> \
    [--output-dir <output_directory>] \
    [--usd-format <usda|usdc|usd|usdz>] \
    [--usd-schema <geompoints|lightfield>] \
    [--percentage-gaussians <0-100>] \
    [--half-precision] \
    [--force-sh-0] \
    [--apply-activation] \
    [--flip-axis <xyz>] \
    [--do-not-cast-shadows] \
    [--skip-gaussian-deformation] \
    [--resample-animation/--no-resample-animation] \
    [--export-rig-trajectories/--no-export-rig-trajectories] \
    [-- hydra_args...]
```

**Arguments:**

- `--config-name`: (Required) Hydra config to load - must contain a dataset specification and checkpoint path
- `--output-dir`: Output directory path (defaults to `<checkpoint_dir>/../usd_asset`)
- `--usd-format`: USD output format: `usda` (text), `usdc` (binary), `usd`, or `usdz` (packaged archive, default)
- `--usd-schema`: USD schema type: `geompoints` (default) or `lightfield`
- `--percentage-gaussians`: Percentage of Gaussians to export, 0-100 (default: 100)
- `--half-precision`: Use half-precision (float16) attributes for reduced file size (LightField schema only)
- `--force-sh-0`: Force SH degree to 0 (skip f_rest coefficients, export only DC color)
- `--apply-activation`: Export post-activation parameter values (applies sigmoid/exp/normalize)
- `--flip-axis`: Axes to flip in 'xyz' form (e.g., `--flip-axis y` to flip Y axis)
- `--do-not-cast-shadows`: Author `doNotCastShadows` primvar on Gaussian prims
- `--skip-gaussian-deformation`: Disable Gaussian deformation animation export
- `--resample-animation/--no-resample-animation`: Resample track timestamps to rig timestamps (default: enabled)
- `--export-rig-trajectories/--no-export-rig-trajectories`: Include rig trajectory data (default: enabled)

### Examples

**Export to USDZ with LightField schema:**

```bash
bazel run //:run -- export-gaussian-usd-asset \
    --config-name configs/my_scene.yaml \
    --usd-schema lightfield \
    --output-dir /output/path
```

# Other Export Commands

This directory also contains many other export commands for various data types:

- `depth.py` - Export depth maps
- `ego_mask.py` - Export ego masks
- `gaussian_plys.py` - Export Gaussian PLY files
- `gaussian_statistics.py` - Export Gaussian statistics
- `gaussian_usd_asset.py` - Export Gaussian models to USD format (see above)
- `ground_mesh.py` - Export ground meshes
- `mesh.py` - Export meshes
- `ncore_diagnostic.py` - Export NCore diagnostics
- `ncore_tracks.py` - Export NCore tracks
- `point_cloud.py` - Export point clouds
- `render.py` - Render images
- `render_grpc.py` - Render via gRPC
- `rig_trajectories.py` - Export rig trajectories
- `sequence_tracks.py` - Export sequence tracks
- `usdz_artifact.py` - Export USDZ artifacts

For usage details of these commands, run:

```bash
bazel run //:run -- <command-name> --help
```
