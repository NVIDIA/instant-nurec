# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from omegaconf import DictConfig

from nre.config.version import Version, get_version


# Use TYPE_CHECKING to avoid circular dependencies at runtime
if TYPE_CHECKING:
    from nre.datasets.summary import DataSourceSummary  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

UpgradeFunc = Callable[[MutableMapping[str, Any], DictConfig, Any], MutableMapping[str, Any]]

# Keep this local to preserve compatibility with older releases where
# libs/losses/kernel/constants.py does not exist (e.g. 25.12.147).
DEFAULT_SOURCE_CHROMS_VALUES: list[list[float]] = [
    [0.0, 0.0],  # pure blue
    [1.0, 0.0],  # pure red
    [0.0, 1.0],  # pure green
    [1.0 / 3.0, 1.0 / 3.0],  # neutral gray
]


@dataclass
class ModelUpgrade:
    """Struct to hold model upgrade functions and their metadata."""

    from_version: Version
    to_version: Version
    upgrade_fn: UpgradeFunc


MODEL_UPGRADE_REGISTRY: list[ModelUpgrade] = []


def register_model_upgrade(
    *, from_version: tuple[int, int, int], to_version: tuple[int, int, int]
) -> Callable[[UpgradeFunc], UpgradeFunc]:
    """Decorator to register a model upgrade function."""

    def decorator(func: UpgradeFunc) -> UpgradeFunc:
        from_version_obj = Version.from_components(*from_version)
        to_version_obj = Version.from_components(*to_version)
        if any(u.from_version == from_version_obj for u in MODEL_UPGRADE_REGISTRY):
            raise ValueError(f"An upgrade function for version {from_version_obj} is already registered.")

        MODEL_UPGRADE_REGISTRY.append(
            ModelUpgrade(from_version=from_version_obj, to_version=to_version_obj, upgrade_fn=func)
        )

        # Sort by from_version to ensure correct upgrade order
        MODEL_UPGRADE_REGISTRY.sort(key=lambda x: x.from_version)
        return func

    return decorator


def upgrade_model(
    model: MutableMapping[str, Any],
    config_orig: DictConfig,
    version_target: Version | None = None,
    datasource_summary: Any = None,
) -> MutableMapping[str, Any]:
    """
    Upgrade model to the latest format based on version information.
    Args:
        model: Model to upgrade (state_dict)
        config_orig: Original configuration
        version_target: Target version to upgrade to
        datasource_summary: Optional datasource summary for extracting temporal information (DataSourceSummary | None)
    Returns:
        Upgraded model
    """
    if version_target is None:
        version_target = get_version()
        if version_target is None:
            # In certain environments (e.g. CI test sandboxes), version information might not be
            # available. In such cases, the target is the latest compatible version that we can upgrade to.
            log.warning("Current version (upgrade target) not available, assuming latest version.")

    if "version" not in config_orig or config_orig.version is None:
        log.warning("No version info found in the artifact's config, assuming that no updates to model are needed")
        return model

    version_orig = Version.model_validate(config_orig.version)

    # Stop condition: current version reached the target.
    # If version_target is None, assume version_orig < version_target and continue.
    if version_target is not None and version_orig >= version_target:
        log.info(
            f"Model v${version_orig.semantic_string()} is already compatible with NuRec v{version_target.semantic_string()}, no upgrade needed"
        )
        return model

    if version_target is None:
        log.info(f"Upgrading config from {version_orig.semantic_string()} to latest compatible version...")
    else:
        log.info(f"Upgrading config from {version_orig.semantic_string()} to {version_target.semantic_string()}...")

    current_model = model
    version_cur = version_orig

    # Note: registry is sorted by from_version, so we can stop at the first upgrade that is greater than the target version
    for upgrade in MODEL_UPGRADE_REGISTRY:
        # Stop condition: current version reached the target.
        # If version_target is None, assume version_cur < version_target and continue.
        if version_target is not None and version_cur >= version_target:
            break

        # Version upgrade gaps are allowed. Detect and log them.
        if version_cur < upgrade.from_version:
            log.info(f" no changes from {version_cur.semantic_string()} to {upgrade.from_version.semantic_string()}")
            version_cur = upgrade.from_version  # close the gap, assume no changes

        # Both of the following conditions need to be met in order to apply the current upgrade function:
        # 1. the current version is older than the version the function upgrades to, and
        # 2. the version the function upgrades to does not go beyond the target version.
        # If the target version is not available, the second condition is disabled.
        if version_cur < upgrade.to_version and (version_target is None or upgrade.to_version <= version_target):
            log.info(f" converting from {version_cur.semantic_string()} to {upgrade.to_version.semantic_string()}")
            current_model = upgrade.upgrade_fn(current_model, config_orig, datasource_summary)
            version_cur = upgrade.to_version

    if version_target is not None and version_cur < version_target:
        log.info(
            f"Assuming no model changes from version {version_cur.semantic_string()} to {version_target.semantic_string()}"
        )

    return current_model


@register_model_upgrade(from_version=(0, 2, 577), to_version=(0, 2, 685))
def upgrade_model_2505_to_2506(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    # For each layer,
    for layer_name, layer in config_orig.model.layers.items():
        if "strategy" in layer and layer.strategy.name == "gsplat":
            # Patch missing GSplatStrategy buffers
            pos_key = f"gaussians_nodes.{layer_name}.positions"
            if pos_key in model:
                num_gaussians: int = model[pos_key].shape[0]
                for node in [
                    {"name": "densify_grad_norm_accum", "type": torch.float32},
                    {"name": "densify_grad_norm_denom", "type": torch.int32},
                ]:
                    key = f"gaussians_nodes.{layer_name}.strategy.{node['name']}"
                    if key not in model:
                        ndims = (num_gaussians, 1)
                        dtype = cast(torch.dtype, node["type"])
                        log.info(f"  adding zero-filled tensor {key} with shape {ndims} and dtype {dtype}")
                        model[key] = torch.zeros(ndims, dtype=dtype)
    return model


@register_model_upgrade(from_version=(0, 2, 685), to_version=(25, 7, 5))
def upgrade_model_2506_to_2507rc2(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    # For each layer,
    for layer_name, layer in config_orig.model.layers.items():
        if "strategy" in layer and layer.strategy.name == "gsplat":
            # Move densify_grad_norm_accum and densify_grad_norm_denom to gaussians_strategy
            for node_name in ["densify_grad_norm_accum", "densify_grad_norm_denom"]:
                key_from = f"gaussians_nodes.{layer_name}.strategy.{node_name}"
                if key_from in model:
                    key_to = f"gaussians_strategy.{node_name}.{layer_name}"
                    model[key_to] = model[key_from]
                    del model[key_from]
                else:
                    log.info(f"  skipping non-existent key {key_from}")
    return model


@register_model_upgrade(from_version=(25, 7, 5), to_version=(25, 8, 1))
def upgrade_model_2507_to_2508_add_temporal_appearance(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Add time_embed state for background and dynamic_rigids layers with temporal appearance (fourier_features_dim > 1).
    Extracts timestamp information from datasource_summary to populate:
    - For background layer: scene start/end timestamps
    - For dynamic_rigids layer: per-track timestamp ranges

    Upgrade from 25.07 to 25.08+ for temporal appearance V1 to V2.
    """

    # Lazy imports to avoid circular dependencies
    import torch_scatter

    from nre.models.composite import LayerTrackIds  # type: ignore[import-untyped]

    if datasource_summary is None:
        log.warning("No datasource_summary provided, skipping temporal appearance upgrade")
        return model

    log.info("Adding time_embed state for layers with temporal appearance")

    # Get rig trajectories for extracting timestamps
    rig_trajectories = datasource_summary.get_rig_trajectories()
    if rig_trajectories is None or len(rig_trajectories.rig_trajectories) == 0:
        log.warning("No rig_trajectories found in datasource_summary, skipping temporal appearance upgrade")
        return model

    # Extract scene-wide timestamp range
    scene_start_timestamp_us = int(rig_trajectories.rig_trajectories[0].T_rig_world_timestamps_us[0].item())
    scene_end_timestamp_us = int(rig_trajectories.rig_trajectories[-1].T_rig_world_timestamps_us[-1].item())

    for layer_name, layer_config in config_orig.model.layers.items():
        fourier_features_dim = layer_config.get("fourier_features_dim", 1)

        if fourier_features_dim <= 1:
            # No temporal appearance for this layer
            continue

        log.info(f"  Processing layer '{layer_name}' with fourier_features_dim={fourier_features_dim}")

        match layer_name:
            case "background":
                # Check if time_embed._extra_state already exists in checkpoint
                extra_state_key = "gaussians_nodes.background.time_embed._extra_state"
                if extra_state_key in model:
                    log.info("    time_embed._extra_state already exists for background layer, skipping")
                    continue

                # Background uses scene-wide timestamps
                log.info(f"    Adding time_embed._extra_state for background layer")
                model[extra_state_key] = {
                    "timestamps_us_min": scene_start_timestamp_us,
                    "timestamps_us_max": scene_end_timestamp_us,
                }

            case "dynamic_rigids":
                # Check if time_embed.timestamps_us_ranges already exists in checkpoint
                timestamps_us_ranges_key = "gaussians_nodes.dynamic_rigids.time_embed.timestamps_us_ranges"
                if timestamps_us_ranges_key in model:
                    log.info("    time_embed.timestamps_us_ranges already exists for dynamic_rigids layer, skipping")
                    continue

                # Dynamic rigids uses per-track timestamp ranges
                log.info("    Adding time_embed.timestamps_us_ranges for dynamic_rigids layer")

                # Get track information from datasource_summary
                cuboid_tracks_all = datasource_summary.get_cuboid_tracks(dynamic_only=False)
                if cuboid_tracks_all is None:
                    log.warning(
                        "    No cuboid_tracks found, skipping temporal appearance upgrade for dynamic_rigids layer"
                    )
                    continue

                # Apply the same track filtering logic used during model initialization
                obj_track_id = LayerTrackIds(config=layer_config.get("tracks", {}))
                obj_track_id.initialize_from_tracks(cuboid_tracks_all)
                cuboid_tracks_filtered = obj_track_id.get_layer_tracks(cuboid_tracks_all)

                log.info(
                    f"    Filtered to {len(cuboid_tracks_filtered.tracks_id)} tracks (from {len(cuboid_tracks_all.tracks_id)} total)"
                )

                # Extract per-track timestamp ranges
                tracks_indptr = cuboid_tracks_filtered.tracks_packinfo[:, 1].cumsum(0)
                tracks_indptr = torch.cat([torch.tensor([0], device=tracks_indptr.device), tracks_indptr])
                min_timestamps_us = torch_scatter.segment_min_csr(
                    cuboid_tracks_filtered.tracks_timestamps_us, tracks_indptr
                )[0]
                max_timestamps_us = torch_scatter.segment_max_csr(
                    cuboid_tracks_filtered.tracks_timestamps_us, tracks_indptr
                )[0]
                timestamps_us_ranges = torch.stack([min_timestamps_us, max_timestamps_us], dim=1)

                log.info(f"    Added timestamp ranges for {len(timestamps_us_ranges)} tracks")
                model[timestamps_us_ranges_key] = timestamps_us_ranges

            case _:
                # Unknown layer with temporal appearance - skip it
                log.warning(
                    f"    Layer '{layer_name}' has temporal appearance but is not recognized (not in 25.07), skipping"
                )

    return model


@register_model_upgrade(from_version=(25, 9, 78), to_version=(25, 9, 79))
def upgrade_model_remove_cached_view_geom(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Remove obsolete cached view geometry entries that were incorrectly persisted in 25.08
    checkpoints. These keys are transient caches and are not part of the learnable model state.

    This upgrade strips any keys rooted under:
      - _cached_camera_view_geometry
      - _cached_lidar_view_geometry
    """

    log.info("Removing obsolete cached view geometry entries that were incorrectly persisted in 25.08 checkpoints.")

    obsolete_prefixes = [
        "_cached_camera_view_geometry",
        "_cached_lidar_view_geometry",
    ]

    to_remove = [
        key
        for key in list(model.keys())
        if any(key == prefix or key.startswith(prefix + ".") for prefix in obsolete_prefixes)
    ]

    for key in to_remove:
        del model[key]

    if len(to_remove) > 0:
        log.info(f"  removed {len(to_remove)} obsolete cached view geometry keys")

    return model


@register_model_upgrade(from_version=(25, 10, 58), to_version=(25, 10, 59))
def upgrade_model_2510_58_to_2510_59_add_sensor_specific_extra_signal(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Add sensor-specific extra signal tensors (camera_extra_signal and lidar_extra_signal).
    These are initialized as empty tensors (0 dimensions) for backward compatibility.
    Also zeros out any existing extra_signal tensors to match the config upgrade.

    This prevents assert failures in ExtraSignal.from_packed_tensor where tensor shapes
    must match the dimensions specified in extra_signal_infos.

    Upgrade from 25.10.58 (1270e2e4) to 25.10.59 (c3b97fcd)
    """

    log.info("Adding sensor-specific extra signal tensors for camera and lidar.")

    # Process each layer
    for layer_name, layer in config_orig.model.layers.items():
        # Get the number of gaussians from positions tensor
        pos_key = f"gaussians_nodes.{layer_name}.positions"
        if pos_key not in model:
            log.warning(f"  skipping layer {layer_name}: positions tensor not found")
            continue

        num_gaussians = model[pos_key].shape[0]

        # IMPORTANT: For backward compatibility, create empty tensors (0 dimensions)
        # Old artifacts didn't have extra signal support, so we should start with empty tensors
        # The model initialization will handle any required resizing during runtime
        camera_dim = 0
        lidar_dim = 0

        # Add camera_extra_signal tensor
        camera_key = f"gaussians_nodes.{layer_name}.camera_extra_signal"
        if camera_key not in model:
            log.info(f"  adding {camera_key} with shape ({num_gaussians}, {camera_dim})")
            model[camera_key] = torch.zeros((num_gaussians, camera_dim), dtype=torch.float32)

        # Add lidar_extra_signal tensor
        lidar_key = f"gaussians_nodes.{layer_name}.lidar_extra_signal"
        if lidar_key not in model:
            log.info(f"  adding {lidar_key} with shape ({num_gaussians}, {lidar_dim})")
            model[lidar_key] = torch.zeros((num_gaussians, lidar_dim), dtype=torch.float32)

        # IMPORTANT: Also zero out any existing extra_signal tensors for backward compatibility
        # These might contain signals like semantic_logits with non-zero dimensions
        # We need to zero them out to match the 0 dimensions set by the config upgrade
        layer_prefix = f"gaussians_nodes.{layer_name}."
        for key in list(model.keys()):
            if key.startswith(layer_prefix) and "extra_signal" in key:
                # Skip the ones we just added
                if key.endswith(".camera_extra_signal") or key.endswith(".lidar_extra_signal"):
                    continue
                # Zero out existing extra_signal tensors
                tensor = model[key]
                if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                    new_shape = tensor.shape[:-1] + (0,)  # Keep all dims except last, set last to 0
                    log.info(f"  zeroing out {key}: {tensor.shape} -> {new_shape}")
                    model[key] = torch.zeros(new_shape, dtype=tensor.dtype)

    return model


@register_model_upgrade(from_version=(25, 12, 55), to_version=(25, 12, 56))
def upgrade_model_2512_55_to_2512_56_background_extra_state(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Migrate SkyEnvMapBackground.n_grad_updates from Buffer storage to _extra_state.

    Before 2aa69dd56, n_grad_updates was stored
    as torch.nn.Buffer(torch.tensor(1, dtype=torch.int32)), resulting in the state_dict
    key "background.n_grad_updates" (or similar paths for composite models).

    After NRE-2389, n_grad_updates is stored as a plain Python int with get_extra_state()
    and set_extra_state() methods, resulting in the state_dict key "background._extra_state"
    containing {"n_grad_updates": value}.

    This upgrade function handles the migration from the old Buffer format to the new
    _extra_state format for all background modules in the model.

    Upgrade from b8a4b4911 (25.12.55) to 2aa69dd56 (25.12.56)
    """
    log.info("Migrating background.n_grad_updates to background._extra_state format")

    # Find all keys that match the old format pattern
    # Handles both "background.n_grad_updates" and paths like "model.background.n_grad_updates"
    old_keys = [key for key in model.keys() if key.endswith(".n_grad_updates") and "background" in key]

    if not old_keys:
        log.info("  No old-format background.n_grad_updates keys found, skipping migration")
        return model

    migrated_count = 0
    for old_key in old_keys:
        # Extract the base path (e.g., "background" or "model.background")
        base_key = old_key.replace(".n_grad_updates", "")
        new_key = f"{base_key}._extra_state"

        # Skip if already migrated
        if new_key in model:
            log.info(f"  Skipping {old_key}: {new_key} already exists")
            continue

        # Extract value from old tensor buffer
        old_value_tensor = model[old_key]
        if isinstance(old_value_tensor, torch.Tensor):
            # Convert tensor to Python int
            n_grad_updates_value = int(old_value_tensor.item())
        else:
            # Fallback if somehow it's already an int
            n_grad_updates_value = int(old_value_tensor)

        # Create new _extra_state dict following the format expected by set_extra_state()
        model[new_key] = {"n_grad_updates": n_grad_updates_value}

        # Remove old key
        del model[old_key]

        log.info(f"  Migrated {old_key} (value={n_grad_updates_value}) -> {new_key}")
        migrated_count += 1

    log.info(f"  Successfully migrated {migrated_count} background module(s)")
    return model


@register_model_upgrade(from_version=(25, 12, 146), to_version=(25, 12, 147))
def upgrade_model_2512_146_to_2512_147_add_ppisp_default_source_chroms(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Add missing PPISP _default_source_chroms buffers introduced in 25.12.147.

    Older checkpoints (before 25.12.147) do not have these keys:
      - post_processings.<idx>.ppisp._default_source_chroms (PPISPSlang)
      - post_processings.<idx>.ppisp._color._default_source_chroms (PPISP)
    """
    log.info("Adding missing PPISP _default_source_chroms buffers")

    template = torch.tensor(DEFAULT_SOURCE_CHROMS_VALUES, dtype=torch.float32)

    for key in list(model.keys()):
        target_key: str | None = None
        if key.endswith(".ppisp.color_params"):
            target_key = key[: -len("color_params")] + "_default_source_chroms"
        elif key.endswith(".ppisp._color.color_params"):
            target_key = key[: -len("color_params")] + "_default_source_chroms"

        if target_key is None or target_key in model:
            continue

        ref_tensor = model[key]
        if not isinstance(ref_tensor, torch.Tensor):
            log.warning(f"  skipping {target_key}: reference key {key} is not a tensor")
            continue

        model[target_key] = template.to(device=ref_tensor.device, dtype=ref_tensor.dtype).clone()

        log.info(f"  adding {target_key} with shape {tuple(model[target_key].shape)}")

    return model


@register_model_upgrade(from_version=(26, 1, 56), to_version=(26, 1, 57))
def upgrade_model_2601_56_to_2601_57_remove_n_active_levels(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Remove obsolete n_active_levels key from feature_volume encoding.

    https://gitlab-master.nvidia.com/nrs/nre/-/merge_requests/2852
    """
    log.info("Removing obsolete n_active_levels keys from feature_volume encoding")

    # Match any layer that has this pattern
    suffix = ".deform_network.feature_volume.encoding.n_active_levels"

    keys_to_remove = [key for key in model.keys() if key.endswith(suffix)]

    for key in keys_to_remove:
        del model[key]
        log.info(f"  removed key: {key}")

    if keys_to_remove:
        log.info(f"  removed {len(keys_to_remove)} obsolete n_active_levels key(s)")

    return model


@register_model_upgrade(from_version=(26, 3, 24), to_version=(26, 3, 25))
def upgrade_model_2603_24_to_2603_25_add_mcmc_invisible_steps(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Add missing MCMC visibility-counter buffers introduced in 26.3.25.

    Older checkpoints do not have:
      gaussians_strategy.invisible_steps.<layer_name>
    """

    strategy_name = config_orig.get("model", {}).get("strategy", {}).get("name")
    if strategy_name != "mcmc":
        return model

    log.info("Adding missing gaussians_strategy.invisible_steps.* buffers for MCMC checkpoints")

    layers = config_orig.get("model", {}).get("layers", {})
    for layer_name in layers.keys():
        key = f"gaussians_strategy.invisible_steps.{layer_name}"
        if key in model:
            continue

        pos_key = f"gaussians_nodes.{layer_name}.positions"
        if pos_key not in model:
            log.warning(f"  skipping layer {layer_name}: positions tensor not found")
            continue

        positions = model[pos_key]
        if not isinstance(positions, torch.Tensor):
            log.warning(f"  skipping layer {layer_name}: positions entry is not a tensor")
            continue

        model[key] = torch.zeros((positions.shape[0],), dtype=torch.int32, device=positions.device)
        log.info(f"  adding {key} with shape ({positions.shape[0]},)")

    return model


@register_model_upgrade(from_version=(26, 4, 110), to_version=(26, 4, 111))
def upgrade_model_2604_110_to_2604_111_n_active_features_to_extra_state(
    model: MutableMapping[str, Any], config_orig: DictConfig, datasource_summary: Any = None
) -> MutableMapping[str, Any]:
    """
    Migrate SHGaussianModel.n_active_features from nn.Buffer to _extra_state.

    Before this change, n_active_features was stored as nn.Buffer(torch.tensor(...)),
    resulting in state_dict keys like "gaussians_nodes.<layer>.n_active_features".

    After this change, n_active_features is a plain int persisted via get_extra_state()
    and set_extra_state(), stored under "gaussians_nodes.<layer>._extra_state".

    NRE-3485: Remove CUDA sync caused by n_active_features.item() in render_lidar path.
    """
    log.info("Migrating n_active_features from nn.Buffer to _extra_state format")

    layers = config_orig.get("model", {}).get("layers", {})

    for layer_name in layers.keys():
        old_key = f"gaussians_nodes.{layer_name}.n_active_features"
        if old_key not in model:
            continue

        extra_state_key = f"gaussians_nodes.{layer_name}._extra_state"

        # Extract value from old tensor buffer
        old_value = model[old_key]
        if isinstance(old_value, torch.Tensor):
            n_active_features_value = int(old_value.item())
        else:
            n_active_features_value = int(old_value)

        # Merge into existing _extra_state if present, otherwise create new
        if extra_state_key in model:
            model[extra_state_key]["n_active_features"] = n_active_features_value
        else:
            model[extra_state_key] = {"n_active_features": n_active_features_value}

        del model[old_key]
        log.info(f"  Migrated {old_key} (value={n_active_features_value}) -> {extra_state_key}")

    return model
