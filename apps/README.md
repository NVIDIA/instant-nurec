# NRE Toolkit

A collection of tools for NRE processing.

## Overview

- **ncore-aux-data**: Data preprocessing utility
- **asset-harvester**: Asset extraction tool
- **mask-annotator**: GUI tool for mask annotation

All tools are **enabled by default** in `.bazelrc`.

## Building

### Binary (Recommended)

```bash
# Default build (all tools, see .bazelrc)
bazel build //apps:nre_tools
```

### Container Images

```bash
# Standard container
bazel build //apps:nre_tools_image_oci

# Obfuscated container
bazel build //apps:obfuscated_nre_tools_image_oci
```

### Testing Obfuscated Builds

To verify the obfuscated binary works correctly:

```bash
bazel run //internal/scripts/pycena/runtime:pycena_nre_tools_test
```

## Tool Selection

All tools are enabled by default in `.bazelrc`:

```bash
build --//bazel/flags:ncore_aux_data=True
build --//bazel/flags:mask_annotator=True
build --//bazel/flags:asset_harvester=True
```

To disable specific tools, override with `=False`:

```bash
# Build with only asset_harvester
bazel build //apps:nre_tools \
    --//bazel/flags:mask_annotator=False \
    --//bazel/flags:ncore_aux_data=False

# Build without mask_annotator
bazel build //apps:nre_tools \
    --//bazel/flags:mask_annotator=False
```

**Flag Reference**

| Flag                              | Default | Description                  |
| --------------------------------- | ------- | ---------------------------- |
| `--//bazel/flags:asset_harvester` | `True`  | Include asset_harvester tool |
| `--//bazel/flags:mask_annotator`  | `True`  | Include mask_annotator tool  |
| `--//bazel/flags:ncore_aux_data`  | `True`  | Include ncore_aux_data tool  |

## Usage

### Ncore Aux Data

#### Binary Execution

```bash
# Show help
bazel run //apps:nre_tools -- ncore-aux-data --help

# Example usage
bazel run //apps:nre_tools -- ncore-aux-data \
    --shard-file-pattern=/path/to/input/zarr.itar \
    --output-dir=/path/to/output
```

#### Container Execution

```bash
bazel run //apps:load_nre_tools_image_oci
docker run -it --rm --gpus all nvcr.io/nvidian/ct-toronto-ai/nre_tools:latest \
    -v /path/to/zarr.itar:/path/to/zarr.itar \
    -- ncore-aux-data \
    --shard-file-pattern=/path/to/zarr.itar \
    --output-dir=/path/to/output
```

### Asset Harvester

#### Binary Execution

```bash
# Show help
bazel run //apps:nre_tools -- asset-harvester --help

# Example usage
bazel run //apps:nre_tools -- asset-harvester  \
    --component-store="path/to/component-store.zarr.itar" \
    --output-dir="path/to/output" \
    --track-ids="track_id1,track_id2" \
    --cache-dir="~/.cache/nre" \
    ncore_parser.camera_ids=["camera_front_wide_120fov"]
```

### Mask Annotator

#### Binary Execution

Make sure to export `DISPLAY=:0` before running:

```bash
bazel run //apps:nre_tools -- mask-annotator
```

#### Container Execution

```bash
bazel run //apps:load_nre_tools_image_oci
docker run -it --rm --gpus all nvcr.io/nvidian/ct-toronto-ai/nre_tools:latest \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -- mask-annotator
```

## Notes

### X11 Display Configuration

For mask-annotator GUI functionality in containers, ensure your `$DISPLAY` environment variable matches an entry in `~/.Xauthority`:

```bash
# Check available displays
xauth list

# Set DISPLAY if needed
export DISPLAY=:0
```

### GPU Support

Container runs should always include `--gpus=all` otherwise you may get some missing `.so` errors.

## Adding New Tools to the Toolkit

### 1. Tool Configuration

Start by adding your tool to the configuration file:

**File:** `apps/tools.yaml`

- Add the import path
- Specify the function to import
- Optionally provide an alias for the function

### 2. Bazel Configuration

Create the necessary Bazel configuration files:

- **Flag:** Add a flag at `//bazel/flags` for your tool
- **Condition:** Add a condition at `//bazel/conditions` for your tool
- **Condition:** Add combination conditions (e.g., `your_tool_and_asset_harvester`) for multi-tool builds
- **Default:** Add your tool to the defaults in `.bazelrc`:

```bash
build --//bazel/flags:your_tool=True
```

### 3. Unobfuscated Container Integration

#### Update Build Dependencies

**File:** `apps/BUILD.bazel`

Add your tool to the unified select in `NRE_TOOLS_DEPS`:

```python
select({
    "//bazel/conditions:all_tools": [..., "<tool_pylib_target>"],
    "//bazel/conditions:<tool>_and_<other>": [...],  # combination conditions
    "//bazel/conditions:<tool_condition_name>": ["<tool_pylib_target>"],
    "//conditions:default": [],
})
```

#### Update Entrypoint Generation

**File:** `//apps:gen_entrypoint`

Add your tool to the unified select:

```python
select({
    "//bazel/conditions:all_tools": ["ncore-aux-data", "mask-annotator", "asset-harvester", "<your-tool>"],
    "//bazel/conditions:<tool>_and_<other>": [...],  # combination conditions
    "//bazel/conditions:<tool_condition_name>": ["<tool name from scripts/tools.yaml>"],
    "//conditions:default": [],
})
```

### 4. Obfuscated Container Integration

#### Update Pycena Sources

**File:** `//internal/scripts/pycena/BUILD.bazel`

1. Define your tool's diff files list:

```python
PY_FILES_YOUR_TOOL_DIFF = [
    "//your/tool:py_files",
]
```

2. Add combination lists for multi-tool builds:

```python
PY_FILES_YOUR_TOOL_AND_ASSET_HARVESTER = PY_FILES_YOUR_TOOL_DIFF + PY_FILES_ASSET_HARVESTER_DIFF
```

3. Update `PY_FILES_ALL_TOOLS` to include your tool:

```python
PY_FILES_ALL_TOOLS = PY_FILES_NCORE_AUX_DATA_DIFF + PY_FILES_ASSET_HARVESTER_DIFF + PY_FILES_MASK_ANNOTATOR_DIFF + PY_FILES_YOUR_TOOL_DIFF
```

4. Add your tool to the unified select in `pycena_nre_tools_srcs`:

```python
select({
    "//bazel/conditions:all_tools": PY_FILES_ALL_TOOLS,
    "//bazel/conditions:<tool>_and_<other>": PY_FILES_<TOOL>_AND_<OTHER>,
    "//bazel/conditions:your_tool": PY_FILES_YOUR_TOOL_DIFF,
    "//conditions:default": [],
})
```

Be mindful of existing py_files - Bazel does not resolve duplicate imports.

#### Update Dependencies Configuration

**File:** `scripts/pycena/runtime/update_deps.py`

Add your new tool's flag to the `CONFIG` under `//apps:nre_tools`:

```python
"//apps:nre_tools": {
    "pip_pkgs_var": "NRE_TOOLS_PIP_PKGS",
    "flags": {
        "mask_annotator": {
            "cquery_arg": "--//bazel/flags:mask_annotator=True",
            "condition": "//bazel/conditions:mask_annotator",
        },
        "your_new_tool": {
            "cquery_arg": "--//bazel/flags:your_new_tool=True",
            "condition": "//bazel/conditions:your_new_tool",
        },
    },
},
```

#### Generate Dependencies

Run the dependency update script to automatically generate the tool's pip dependencies:

```bash
bazel run //internal/scripts/pycena/runtime:update_deps -- //apps:nre_tools
```

Verify that `NRE_TOOLS_PIP_PKGS` in `//internal/scripts/pycena/runtime/BUILD.bazel` now includes your tool's dependencies in the appropriate `select()` condition.

#### Build the Container

Build the obfuscated tools image (all tools included by default):

```bash
bazel build //apps:obfuscated_nre_tools_image_oci
```
