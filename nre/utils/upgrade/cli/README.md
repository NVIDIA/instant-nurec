# Upgrade Commands

This directory contains utility CLI commands for upgrading NRE artifacts and checkpoints to newer versions of the codebase.

## `upgrade-config`

This command upgrades **configuration files** to a specified version. It supports two input methods:

1. **Raw config file** (`--config-name`) - Upgrades a YAML config file with optional Hydra overrides
2. **USDZ artifact** (`--input`) - Extracts and upgrades the config from inside a USDZ artifact
   > **Note:** This command does not upgrade the checkpoints even when you provide a USDZ artifact.

### Usage

**Method 1: From config file (supports Hydra overrides)**

```bash
# Bazel
bazel run //:run -- upgrade-config \
  --config-name <path_to_config_file> \
  --output <path_to_upgraded_config_file> \
  [--target-version <version>] \
  [--sort-keys/--no-sort-keys] \
  [-- hydra_args...]

# Docker
docker run nre_run upgrade-config \
  --config-name <path_to_config_file> \
  --output <path_to_upgraded_config_file> \
  [--target-version <version>] \
  [--sort-keys/--no-sort-keys] \
  [-- hydra_args...]
```

**Method 2: From USDZ artifact (no Hydra overrides)**

```bash
# Bazel
bazel run //:run -- upgrade-config \
  --input <path_to_usdz> \
  --output <path_to_upgraded_config_file> \
  [--target-version <version>] \
  [--sort-keys/--no-sort-keys]

# Docker
docker run nre_run upgrade-config \
  --input <path_to_usdz> \
  --output <path_to_upgraded_config_file> \
  [--target-version <version>] \
  [--sort-keys/--no-sort-keys]
```

### Arguments

**Input (choose one):**

- `-c, --config-name`: Path to input config file (supports Hydra overrides; `.yaml` extension is optional)
- `-i, --input`: Path to input USDZ artifact file (pre-parsed config, no overrides)

**Output:**

- `-o, --output`: **Required.** Path where the upgraded config will be saved

**Options:**

- `-t, --target-version`: Target version in `major.minor[.patch]` format (defaults to current version). If only `major.minor` is provided (e.g., `1.2`), it will upgrade to the most recent patch version in that branch (e.g., `1.2.99999` internally).
- `--sort-keys/--no-sort-keys`: Sort keys in output for easier comparison with other sorted configs (default: sort)
- `hydra_args`: Additional Hydra overrides (only with `--config-name`)

> **Note:** Hydra arguments only work with `--config-name` because USDZ artifacts contain pre-parsed configs that can't be modified with Hydra overrides.

### Examples

**From config file:**

```bash
# To current version
bazel run //:run -- upgrade-config \
  --config-name configs/path/to/config_file \
  --output upgraded_config.yaml

# With Hydra overrides
bazel run //:run -- upgrade-config \
  --config-name configs/path/to/config_file \
  --output upgraded_config.yaml \
  -- dataset.batch_size=32 trainer.max_epochs=100

# To specific version
bazel run //:run -- upgrade-config \
  --config-name configs/path/to/config_file \
  --output upgraded_config.yaml \
  --target-version 1.2.3

# Docker example (with Hydra overrides and target version)
docker run nre_run upgrade-config \
  --config-name configs/path/to/config_file \
  --output upgraded_config.yaml \
  --target-version 1.2.3 \
  -- dataset.batch_size=32 trainer.max_epochs=100
```

**From USDZ artifact:**

```bash
# To current version
bazel run //:run -- upgrade-config \
  --input my_artifact.usdz \
  --output upgraded_config.yaml

# To specific version
bazel run //:run -- upgrade-config \
  --input my_artifact.usdz \
  --output upgraded_config.yaml \
  --target-version 1.2.3

# Docker example (to specific version with sorted keys)
docker run nre_run upgrade-config \
  --input my_artifact.usdz \
  --output upgraded_config.yaml \
  --target-version 1.2.3 \
  --sort-keys
```

## `upgrade-artifact`

This command upgrades complete USDZ artifact files to a specified version. It upgrades both the configuration and model checkpoint contained within the artifact, creating a new USDZ file.

### Usage

```bash
# Bazel
bazel run //:run -- upgrade-artifact \
  --input <artifact.usdz> \
  --output <upgraded_artifact.usdz> \
  [--target-version <version>] \
  [--debug]

# Docker
docker run nre_run upgrade-artifact \
  --input <artifact.usdz> \
  --output <upgraded_artifact.usdz> \
  [--target-version <version>] \
  [--debug]
```

### Arguments

**Required:**

- `-i, --input`: Path to input USDZ artifact file
- `-o, --output`: Path for the upgraded USDZ artifact file

**Options:**

- `-t, --target-version`: Target version in `major.minor[.patch]` format (defaults to current version). If only `major.minor` is provided (e.g., `1.2`), it will upgrade to the most recent patch version in that branch (e.g., `1.2.99999` internally).
- `--debug`: Enable detailed debug logging

### Examples

```bash
# Upgrade to current version
bazel run //:run -- upgrade-artifact \
  --input old_artifact.usdz \
  --output new_artifact.usdz

# Upgrade to specific version
bazel run //:run -- upgrade-artifact \
  --input old_artifact.usdz \
  --output new_artifact.usdz \
  --target-version 1.2.3

# With debug logging
bazel run //:run -- upgrade-artifact \
  --input old_artifact.usdz \
  --output new_artifact.usdz \
  --debug

# Docker example (to specific version with debug logging)
docker run nre_run upgrade-artifact \
  --input old_artifact.usdz \
  --output new_artifact.usdz \
  --target-version 1.2.3 \
  --debug
```

## When to Use Each Command

| Command            | Use Case                                              | Input                        | Output           |
| ------------------ | ----------------------------------------------------- | ---------------------------- | ---------------- |
| `upgrade-config`   | Upgrade config only, compare configs between versions | Config file or USDZ artifact | YAML config file |
| `upgrade-artifact` | Upgrade whole artifact                                | USDZ artifact                | USDZ artifact    |

## Related Documentation

- [Artifact Upgrade System](../README.md) - Detailed explanation of the upgrade system
- [Export Commands](../../io/export/README.md) - Tools for exporting and inspecting config files and checkpoints

## Implementation

The upgrade functionality is provided by:

- `nre.utils.upgrade.config` - Configuration upgrade functions
- `nre.utils.upgrade.model` - Model checkpoint upgrade functions
