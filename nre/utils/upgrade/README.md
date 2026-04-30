# Artifact upgrade system

This document explains how to upgrade NuRec artifacts (`.usdz` files) to be compatible with newer versions of the codebase. The upgrade system handles modifications to both the artifact's configuration (`parsed_config.yaml`) and its model state (`checkpoint.ckpt`).

## How upgrading works

The upgrade process is sequential and version-driven. When an artifact is loaded, its version is compared against the **target version** (usually the current codebase version). If the artifact's version is older, the system applies a series of registered upgrade functions in order, bringing the artifact step-by-step to the target version.

This process is managed by two main modules:

- `nre.utils.upgrade.config`: Handles upgrades for the `OmegaConf` configuration.
- `nre.utils.upgrade.model`: Handles upgrades for the model's state dictionary.

Each module maintains a registry of upgrade functions. Each upgrade function is tagged with a `from_version` and a `to_version`. These version intervals of different upgrade functions must be non-overlapping but gaps in between are allowed. The system iterates through the registry, applying each function gradually bridging the gap between the artifact's current version and the target version.

## What can be upgraded?

The system can upgrade two key components of an artifact:

1.  **Configuration (`parsed_config.yaml`)**: Any changes to the structure or values within the `OmegaConf` configuration file.
2.  **Model State (`checkpoint.ckpt`)**: Any changes to the keys, shapes, or values within the model's `state_dict`.

## When to add an upgrade function

An upgrade function is required whenever a **backward-incompatible change** is introduced to either the configuration or the model state. If a change would prevent an older artifact from loading correctly with the new codebase, an upgrade function must be created to bridge the gap.

Examples of changes that require an upgrade function:

- Renaming a key in the configuration.
- Changing the data type or structure of a config value.
- Moving a subtree in the config tree under a different (existing or new) parent node.
- Renaming a layer or parameter in the model's state dictionary.
- Modifying the shape or size of a model parameter.

## Guidelines for upgrades

When writing upgrade functions, especially for configuration changes, follow these guidelines to ensure smooth and predictable artifact upgrades.

### Guiding principle: Preserve original behavior

The primary goal of an upgrade is to make an old artifact compatible with new code while preserving its original rendered output as closely as possible. Avoid adding new features or layers unless they are strictly necessary for the artifact to function.

### Configuration upgrades (`config.py`)

We provide helper functions like `insert_config_key()`, `remove_config_key()`, `copy_config_key()`, `move_config_key()` to facilitate the following operations with minimal code. The helper functions take a dot notation config path (e.g. `"checkpoint.artifact.mesh.generic.frame_step"`), do the necessary checks, handle missing nodes in the paths and are permissive overall to make their usage as smooth as possible.

- **Adding new configs**:

  - If a new configuration key is introduced that does not have a default value in the code, you must add it in the upgrade function. Use the `insert_config_key()` helper to insert a key by also creating parent nodes if they do not exist yet.
  - Choose a default value that maintains the original behavior. For rendering-related parameters, select values that will make the visual output of the upgraded artifact match the output from the original version.

- **Moving configs**:

  - If a configuration key is moved (e.g., from `system.foo` to `model.foo`), your upgrade function should handle the migration of the value from the old path to the new one. Use the `move_config_key()` helper to move the key to any new path in the config tree by also creating the target parent nodes if they do not exist yet. Alternatively, you can also use `copy_config_key()` and `remove_config_key()`.

- **Removing old configs**:
  - You can safely remove configuration keys that are obsolete and no longer processed by the current NuRec version. This helps keep the upgraded configuration clean. Use the `remove_config_key()` helper.

### Model and loss upgrades (`model.py`)

- **New layers and losses**:
  - If a new version of NuRec adds new model layers or losses by default, **do not** add them in an upgrade function if the artifact can still render correctly without them.
  - Consequently, do not add the corresponding buffers or parameters for these optional new layers to the model's `state_dict`.
- The goal is to upgrade for compatibility, not to add new features to old artifacts.

## NuRec helper command for configuration upgrades

When creating a configuration upgrade function, it can be helpful to compare the fully parsed configuration from an old version of NuRec with the new one. This helps identify exactly what keys have been renamed, moved, or added. Both configs should be sorted by key, otherwise they may not be easy to compare.

A NuRec command is available to parse a configuration file and output the fully resolved output. You can use the `run.py` subcommand `export-parsed-config` for this purpose. See the [full command documentation](../io/export/README.md#export-parsed-config) for detailed usage instructions.

By running this command with two different NuRec versions (e.g., from two different branches) and diffing the output, you can easily spot the changes that your upgrade function needs to handle.

## NuRec helper command for configuration upgrades

The helper command [export-artifact-structure](../io/export/README.md#export-artifact-structure) can be used to compare the model structure from two different versions. This allows you to identify renamed layers, or parameters that have changed shape or datatype.

It is used to inspect the checkpoint within a `.usdz` artifact or a raw PyTorch checkpoint (`.ckpt`) and output the structure of its tensors as a JSON object. This includes the shape and data type for each tensor. By generating this structural JSON for an artifact from an old and a new NuRec version, you can diff the files to see exactly what your model upgrade function needs to adapt.

## How to add an upgrade function step-by-step

Follow these steps to add a new upgrade function.

### 1. Identify the change type

First, determine whether your change affects the **configuration** or the **model state**.

- For **configuration** changes, you will work in `nre/utils/upgrade/config.py`.
- For **model state** changes, you will work in `nre/utils/upgrade/model.py`.

### 2. Identify the from/to versions of the upgrade function to be created

Each upgrade function is characterized by two versions:

- `to_version`: A tuple of integers representing the version of the _merge commit_ of your change. This is the version the upgrade function will upgrade the config or model to.
- `from_version`: A tuple of integers representing the last _merge commit before_ your change. This is the expected version of the config or model your upgrade function expects as input.

> Our current versioning system uses a semantic versioning scheme `<major>.<minor>.<patch>-<hash>`, where `<major>.<minor>` originate from a `VERSION_FILE` checked into the repository, the `<patch>` number is calculated automatically as the number of _merge commits_ since the last change of `<major>.<minor>`, and `<hash>` is the short hash of the corresponding _merge commit_.

To determine `from_version` and `to_version` for your change, find your MR in Gitlab and find the branch of that MR, then search git history for the merge commits as follows, for example:

```bash
git log --oneline | grep <branch_name>   # -> <to_commit> = the MR's merge commit titled "Merge branch ..."
git checkout <to_commit>
bazel run //:run -- --version            # -> <to_version>
git log --oneline                        # -> <from_commit> = latest commit titled "Merge branch ..."
git checkout <from_commit>
bazel run //:run -- --version            # -> <from_version>
```

Verify that `to_version` is exactly one patch version after `from_version` to keep the scope of your upgrade function tight.

### 3. Visualize the config change

A possible way to obtain a configuration diff that your upgrade function will need to reproduce is the following:

1. Checkout `from_commit` and do a quick training with config `<config>` to generate a USDZ.
2. Checkout `to_commit` and export the parsed config `<config>` by using the [export-parsed-config](../io/export/README.md#export-parsed-config) command.
3. Stay on `to_commit`, run the [upgrade-artifact](scripts/README.md#upgrade-artifact) command to upgrade the USDZ and use the [export-parsed-config](../io/export/README.md#export-parsed-config) command to extract its `parsed_config.yaml`.
4. Compare the configs from Steps 2 and 3 e.g. in `meld` or VSCode, and find the relevant changes. Discard the changes due to some runtime information introduced by training into the config.

As you implement your upgrade function (next section), steps 3 and 4 can be repeated to make sure the differences vanish.

> We will soon eliminate the need to train (generate a USDZ) from this process with a small extension to [export-parsed-config](scripts/README.md#export-parsed-config) command.

### 4. Create and register a new upgrade function

In the appropriate file (`config.py` or `model.py`), define a new Python function that performs the necessary transformation.

- A **config upgrade** function needs to be decorated with `@register_config_upgrade` and accepts a `DictConfig` object to upgrade from. It should modify the config **in-place**.
- A **model upgrade** function needs to be decorated with `@register_model_upgrade`, and should accept two arguments: the `state_dict` (a mutable mapping) to upgrade from and the original (pre-upgrade) `DictConfig`. It should **return** the modified `state_dict`.
- Give the function a descriptive name that clearly indicates its purpose (e.g., `_rename_optimizer_to_scheduler`).
- The decorators bind `from_version` and `to_version` to the function.

> :warning: The upgrade function should **only** perform the transformation. It **should not** modify the version number. The main upgrade loop handles version bumping automatically.

> :warning: We target a decorator function to a specific change, i.e. we make `to_version` the _merge commit_ of your change and `from_version` the merge commit right before that tightly, otherwise other developers are forced to insert subsequent (and likely unrelated) follow-up config changes into the same upgrade function.

#### Example: Adding a config upgrade

Let's say you renamed the `system.optimizer` key to `system.scheduler` in the configuration. This change is introduced in version `(0, 3, 1)`. The previous version was `(0, 3, 0)`.

In `nre/utils/upgrade/config.py`:

```python
# ... existing code ...

@register_config_upgrade(from_version=(0, 3, 0), to_version=(0, 3, 1))
def _rename_optimizer_to_scheduler(cfg: DictConfig) -> None:
    """Renames the system.optimizer config key to system.scheduler."""
    if "optimizer" in cfg.system:
        cfg.system.scheduler = cfg.system.pop("optimizer")

# ... existing code ...
```

#### Example: Adding a model upgrade

Suppose a change in version `(0, 4, 0)` requires adding a new buffer `strategy.new_buffer` to the model's state dictionary if it doesn't exist. The previous version was `(0, 3, 15)`.

In `nre/utils/upgrade/model.py`:

```python
import torch
from collections.abc import MutableMapping
from omegaconf import DictConfig
# ... existing code ...

@register_model_upgrade(from_version=(0, 3, 15), to_version=(0, 4, 0))
def _add_new_buffer_to_strategy(
    model: MutableMapping[str, any], config_orig: DictConfig
) -> MutableMapping[str, any]:
    """Adds a new_buffer to the strategy if it's not present."""
    key = "strategy.new_buffer"
    if key not in model:
        model[key] = torch.zeros((1,), dtype=torch.float32)
    return model

# ... existing code ...
```

### 5. Verify

After adding your function, ensure that:

- The `from_version` matches the last version number before your backward-incopatible changes.
- The `to_version` is the version of the _merge commit_ of your backward-incopatible changes.
- Your upgrade function correctly modifies the artifact's data from the old format to the new one. You can use the process described in Section 3 to verify this.
