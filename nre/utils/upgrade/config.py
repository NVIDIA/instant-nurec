# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from omegaconf import DictConfig, OmegaConf

from nre.config.version import Version, get_version
from nre.utils.upgrade.config_manip import (
    copy_config_key,
    exists_config_key,
    insert_config_key,
    move_config_key,
    remove_config_key,
)


log = logging.getLogger(__name__)

UpgradeFunc = Callable[[DictConfig], DictConfig]


@dataclass
class ConfigUpgrade:
    """Struct to hold config upgrade functions and their metadata."""

    from_version: Version
    to_version: Version
    upgrade_fn: UpgradeFunc


CONFIG_UPGRADE_REGISTRY: list[ConfigUpgrade] = []


def register_config_upgrade(
    *, from_version: tuple[int, int, int], to_version: tuple[int, int, int]
) -> Callable[[UpgradeFunc], UpgradeFunc]:
    """Decorator to register a config upgrade function."""

    def decorator(func: UpgradeFunc) -> UpgradeFunc:
        from_version_obj = Version.from_components(*from_version)
        to_version_obj = Version.from_components(*to_version)
        if any(u.from_version == from_version_obj for u in CONFIG_UPGRADE_REGISTRY):
            raise ValueError(f"An upgrade function for config version {from_version_obj} is already registered.")

        CONFIG_UPGRADE_REGISTRY.append(
            ConfigUpgrade(from_version=from_version_obj, to_version=to_version_obj, upgrade_fn=func)
        )
        CONFIG_UPGRADE_REGISTRY.sort(key=lambda x: x.from_version)
        return func

    return decorator


def upgrade_config(orig_cfg: DictConfig, version_target: Version | None = None) -> DictConfig:
    """
    Upgrade a configuration dictionary to the latest or specified target version.

    This function takes a configuration in the form of a DictConfig (typically loaded from YAML or OmegaConf),
    checks its version, and applies a series of upgrade transformations to bring it up to the specified target version.
    If no target version is provided, the current package version is used.

    Args:
        orig_cfg (DictConfig): The original configuration to upgrade.
        version_target (Version | None, optional): The target version to upgrade to. If None, uses the current version.

    Returns:
        DictConfig: The upgraded configuration dictionary.

    Raises:
        ValueError: If the original configuration version is too old to upgrade.

    Notes:
        - The function applies upgrades in sequence, starting from the original version up to the target version.
        - If the configuration is already at the target version, it is returned unchanged.
    """
    cfg = orig_cfg.copy()

    if version_target is None:
        version_target = get_version()
        if version_target is None:
            # In certain environments (e.g. CI test sandboxes), version information might not be
            # available. In such cases, the target is the latest compatible version that we can upgrade to.
            log.warning("Current version (upgrade target) not available, assuming latest version.")

    if "version" not in cfg or cfg.version is None:
        log.warning("Configuration contains no version info, assuming that no upgrade to the config is needed")
        return cfg

    # Do not upgrade NRM training config as it is not supported yet.
    if OmegaConf.select(cfg, "dataset.name", default=None) == "nrm":
        log.info("Skipping NRM training config upgrade as it is not supported yet.")
        return cfg

    original_version: DictConfig = cfg.version
    version_cur = Version.model_validate(original_version)
    if version_cur is None:
        raise ValueError("Could not parse version information from configuration.")
    # Stop condition: current version reached the target.
    # If version_target is None, assume version_cur < version_target and continue.
    if version_target is not None and version_cur >= version_target:
        log.info(
            f"Config v{version_cur.semantic_string()} is already compatible with NuRec v{version_target.semantic_string()}, no upgrade needed"
        )
        return cfg

    if version_target is None:
        log.info(f"Upgrading config from {version_cur.semantic_string()} to latest compatible version...")
    else:
        log.info(f"Upgrading config from {version_cur.semantic_string()} to {version_target.semantic_string()}...")

    # Note: registry is sorted by from_version, so we can stop at the first upgrade that is greater than the target version
    for upgrade in CONFIG_UPGRADE_REGISTRY:
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
            cfg = upgrade.upgrade_fn(cfg)
            version_cur = upgrade.to_version
            # Update the version in the config after applying the upgrade
            cfg.version.version_major = version_cur.version_major
            cfg.version.version_minor = version_cur.version_minor
            cfg.version.version_patch = version_cur.version_patch
            cfg.version.version_string = version_cur.semantic_string()
            cfg.version.git_commit_sha_short = "00000000"
            cfg.version.git_commit_date = datetime.now().isoformat()

    if version_target is not None:
        # Post-condition is that the config version corresponds to the current NuRec version:
        cfg.version.version_string = version_target.semantic_string()

        if version_cur < version_target:
            log.info(f" no changes from {version_cur.semantic_string()} to {version_target.semantic_string()} (target)")

    return cfg


@register_config_upgrade(from_version=(0, 2, 577), to_version=(0, 2, 685))
def upgrade_config_2505_to_2506(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Create trainer config once for reuse
    trainer_config = OmegaConf.create(OmegaConf.to_container(outcfg.trainer, resolve=True))

    # 1. Update trainer configuration
    trainer_config.relative_lr = True
    trainer_config.relative_schedule = True
    trainer_config.relative_num_workers = True
    trainer_config.annealing_update_every_n_steps = 1000

    # 2. Add datamodule prefetch section and trainer copy
    if "datamodule" in outcfg:
        outcfg.datamodule.trainer = trainer_config
        outcfg.datamodule.prefetch = OmegaConf.create({"enabled": False, "queue_size": 2})

    # 3. Add trainer config to each layer
    for layer_name, layer in outcfg.model.layers.items():
        layer.trainer = trainer_config

        # 3.1 Add trainer config to strategy
        if "strategy" in layer:
            layer.strategy.trainer = trainer_config

        # 3.2 Add trainer config to tracks_calib
        if "tracks_calib" in layer:
            layer.tracks_calib.trainer = trainer_config

        if "deform_network" in layer and "feature_volume" in layer.deform_network:
            layer.deform_network.feature_volume.trainer = trainer_config
            layer.deform_network.rotations_from_identity = True

    # 4. Add trainer config to post_processing
    if "post_processing" in outcfg.model:
        for item_name, item in outcfg.model.post_processing.items():
            item.trainer = trainer_config

    # 5. Add trainer config to system
    if "system" in outcfg:
        outcfg.system.trainer = trainer_config

    # 6. Add trainer config to model
    outcfg.model.trainer = trainer_config

    # 7. Add trainer config to loss functions that need it
    if "loss" in outcfg:
        if "semantic" in outcfg.loss:
            outcfg.loss.semantic.trainer = trainer_config
        if "node_semantic_gaussians" in outcfg.loss:
            outcfg.loss.node_semantic_gaussians.trainer = trainer_config

    # 8. Add missing model.renderer.profiling.frequency
    if "renderer" in outcfg.model:
        outcfg.model.renderer.profiling = OmegaConf.create({"frequency": 0})

    # 9. Add val_camera_ids: From 0.2.612 (f1ba3cb8) to 0.2.613 (d53fd44f)
    assert isinstance(outcfg, DictConfig)  # needed for mypy
    if not exists_config_key(outcfg, "dataset.val_camera_ids"):
        copy_config_key(outcfg, "dataset.camera_ids", "dataset.val_camera_ids", default_value=[])

    return cast(DictConfig, outcfg)


@register_config_upgrade(from_version=(0, 2, 685), to_version=(25, 7, 5))
def upgrade_config_2506_to_2507rc2(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # 1. Remove raydrop_mask from lidar_model
    if "lidar_model" in outcfg.dataset and "raydrop_mask" in outcfg.dataset.lidar_model:
        del outcfg.dataset.lidar_model["raydrop_mask"]

    # 2. Upgrade layers
    for layer_name, layer in outcfg.model.layers.items():
        if "particle" in layer:
            # 2.2. Remove particle.density_kernel_density_clamping
            if "density_kernel_density_clamping" in layer.particle:
                del layer.particle["density_kernel_density_clamping"]
        if "strategy" in layer:
            # 2.2. Move strategy from layers to model
            if not "strategy" in outcfg.model:
                outcfg.model.strategy = layer.strategy
            del layer["strategy"]

    # 3. Add metrics config to test config
    if "test" in outcfg.system:
        outcfg.system.test.metrics = OmegaConf.create(
            {
                "cpsnr": {
                    "enabled": True,
                    "classes": None,
                }
            }
        )

    # 5. Add max_number_of_images and remove_method to dataset.samplers.difix_batch_sampler
    # 0.2.688 (937a0ab4) - 0.2.689 (1e5600f6)
    if exists_config_key(outcfg, "dataset.samplers.difix_batch_sampler"):
        outcfg_difix_sampler = outcfg.dataset.samplers.difix_batch_sampler
        if not exists_config_key(outcfg_difix_sampler, "max_number_of_images"):
            insert_config_key(outcfg_difix_sampler, "max_number_of_images", 3000)
        if not exists_config_key(outcfg_difix_sampler, "remove_method"):
            insert_config_key(outcfg_difix_sampler, "remove_method", "random")

    return cast(DictConfig, outcfg)


@register_config_upgrade(from_version=(25, 7, 5), to_version=(25, 8, 1))
def upgrade_config_2507_to_2508(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Upgrade the temporal appearance setting
    for layer_name, layer in outcfg.model.layers.items():
        if layer_name == "background":
            insert_config_key(
                layer,
                "time_embed",
                OmegaConf.create({"name": "holistic-remap-time-input-embedding", "remap_min": 0, "remap_max": 1}),
            )
        if layer_name == "dynamic_rigids":
            insert_config_key(
                layer,
                "time_embed",
                OmegaConf.create({"name": "individual-remap-time-input-embedding", "remap_min": 0, "remap_max": 1}),
            )

    # copy val_lidar_ids from lidar_ids if it doesn't exist
    if not exists_config_key(outcfg, "dataset.val_lidar_ids"):
        copy_config_key(outcfg, "dataset.lidar_ids", "dataset.val_lidar_ids", default_value=[])

    return outcfg


@register_config_upgrade(from_version=(25, 9, 92), to_version=(25, 9, 93))
def upgrade_config_2509_92_to_93_mesh_config_changes(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    source = "checkpoint.artifact.mesh"
    generic = "checkpoint.artifact.mesh.generic"
    ground = "checkpoint.artifact.mesh.ground"

    # Default values below are taken from the from_version configs. They apply when the source keys are missing.

    # Move keys from checkpoint.artifact.mesh to checkpoint.artifact.mesh.generic.
    move_config_key(outcfg, f"{source}.enabled", f"{generic}.enabled", default_value=False)
    move_config_key(outcfg, f"{source}.formats", f"{generic}.formats", default_value=["ply", "usd"])
    move_config_key(outcfg, f"{source}.step_frame", f"{generic}.step_frame", default_value=1)
    move_config_key(outcfg, f"{source}.lidar_ids", f"{generic}.lidar_ids", default_value=None)
    move_config_key(outcfg, f"{source}.camera_ids", f"{generic}.camera_ids", default_value=None)
    move_config_key(outcfg, f"{source}.max_num_points", f"{generic}.max_num_points", default_value=None)
    move_config_key(outcfg, f"{source}.n_neighbors", f"{generic}.n_neighbors", default_value=200)
    move_config_key(outcfg, f"{source}.trim_distance", f"{generic}.trim_distance", default_value=0.225)
    move_config_key(outcfg, f"{source}.smooth", f"{generic}.smooth", default_value=False)
    move_config_key(
        outcfg, f"{source}.apply_road_segmentation", f"{generic}.apply_road_segmentation", default_value=False
    )
    move_config_key(
        outcfg, f"{source}.export_disjoint_meshes", f"{generic}.export_disjoint_meshes", default_value=False
    )
    move_config_key(outcfg, f"{source}.mesh_path", f"{generic}.mesh_path", default_value=None)

    # Remove checkpoint.artifact.mesh.non_dynamic_points_only
    remove_config_key(outcfg, f"{source}.non_dynamic_points_only")

    # Replicate some keys from checkpoint.artifact.mesh.generic in checkpoint.artifact.mesh.ground
    # so that the ground mesh can be controlled independently
    copy_config_key(outcfg, f"{generic}.formats", f"{ground}.formats", default_value=["ply", "usd"])
    copy_config_key(outcfg, f"{generic}.step_frame", f"{ground}.step_frame", default_value=1)
    copy_config_key(outcfg, f"{generic}.lidar_ids", f"{ground}.lidar_ids", default_value=None)
    copy_config_key(outcfg, f"{generic}.camera_ids", f"{ground}.camera_ids", default_value=None)

    return outcfg


@register_config_upgrade(from_version=(25, 9, 1), to_version=(25, 9, 2))
def upgrade_config_2509_01_to_02__every_n_epochs__to__every_n_train_steps(cfg: DictConfig) -> DictConfig:
    """Create `checkpoint.every_n_train_steps` if it doesn't exist.
    Upgrade from de961aa81326c8931dd4d21d53f046cdbc786584 (25.9.1) to b793db37600907866e32f69020734513a59810d7 (25.9.2)
    """

    # Create a copy of the input config to avoid modifying the original
    outcfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))

    # Upgrade from every_n_epochs to every_n_train_steps. We do not want to overwrite the existing
    # every_n_train_steps if somehow it is already present.
    if not exists_config_key(outcfg, "checkpoint.every_n_train_steps"):
        # Configure dataset.n_samples_per_epoch should always present, otherwise it's a bug for NRE
        n_samples_per_epoch = outcfg.dataset.n_samples_per_epoch
        every_n_epochs = OmegaConf.select(outcfg, "checkpoint.every_n_epochs", default=1)
        insert_config_key(outcfg, "checkpoint.every_n_train_steps", every_n_epochs * n_samples_per_epoch)

    # Safely remove the every_n_epochs key
    remove_config_key(outcfg, "checkpoint.every_n_epochs")

    # Also remove save_last from checkpoint config
    remove_config_key(outcfg, "checkpoint.save_last")

    return outcfg


@register_config_upgrade(from_version=(25, 9, 19), to_version=(25, 9, 20))
def upgrade_config_2509_19_to_2509_20_dynamic_tracks_classification(cfg: DictConfig) -> DictConfig:
    """Use speed-based classification for dynamic tracks.
    Upgrade from b77c81f07868b4ece3be3624811ebdc3917c8500 (25.9.19) to c7e4b6c58f46e323ba110ed54acac0ef4920631a (25.9.20)
    """

    # Create a copy of the input config to avoid modifying the original
    outcfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))

    # Disable displacement and distance based classification
    if not exists_config_key(outcfg, "dataset.cuboid_tracks_params.use_displacement_and_distance"):
        insert_config_key(outcfg, "dataset.cuboid_tracks_params.use_displacement_and_distance", False)

    return outcfg


# from 25.10.16 (d957afd1) to 25.10.17 (de4c2a11) loss fn changes
@register_config_upgrade(from_version=(25, 10, 16), to_version=(25, 10, 17))
def upgrade_config_2510_16_to_2510_17_loss_fn_changes(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Go through all loss functions and remove the prefix `mask_` from the loss fn name
    if "loss" in outcfg:
        for _loss_name, loss_config in outcfg.loss.items():
            # If loss_config.fn exists and starts with the prefix `mask_`, remove it
            if hasattr(loss_config, "fn") and isinstance(loss_config.fn, str) and loss_config.fn.startswith("mask_"):
                loss_config.fn = loss_config.fn[5:]  # Remove "mask_" prefix

    return outcfg


# from 25.8.17 (#82f7fcf0) to 25.8.18 (#00157d51)
@register_config_upgrade(from_version=(25, 8, 17), to_version=(25, 8, 18))
def upgrade_config_2508_17_to_2508_18_gaussians_system_test_lidar(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Create a LidarSystemTestConfig which contains LidarEvaluatorTestConfig
    copy_config_key(
        outcfg, "system.test.lidar.raydrop.threshold", "system.test.lidar.raydrop_threshold", default_value=0.5
    )
    min_m = OmegaConf.select(outcfg, "system.test.lidar.ROI.min_m", default=None)
    max_m = OmegaConf.select(outcfg, "system.test.lidar.ROI.max_m", default=None)
    insert_config_key(outcfg, "system.test.lidar.ROI", OmegaConf.create({"min_m": min_m, "max_m": max_m}))

    return outcfg


# from 25.9.79-04c6730c to 25.9.80-992d328c
@register_config_upgrade(from_version=(25, 9, 79), to_version=(25, 9, 80))
def upgrade_config_2509_79_to_2509_80(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    insert_config_key(outcfg, "model.calib.enabled", False)

    return outcfg


# from 25.9.82 to 25.9.83
@register_config_upgrade(from_version=(25, 9, 82), to_version=(25, 9, 83))
def upgrade_config_2509_82_to_2509_83(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # In 25.9, gaussian losses read optional per-layer lambdas from
    # loss.<loss_name>.layer_lambdas. Older artifacts may lack this key.
    # Insert an empty mapping so lookups fall back to 1.0 at runtime.
    for loss_name in ("gaussian_density", "gaussian_scale"):
        if exists_config_key(outcfg, f"loss.{loss_name}") and not exists_config_key(
            outcfg, f"loss.{loss_name}.layer_lambdas"
        ):
            insert_config_key(outcfg, f"loss.{loss_name}.layer_lambdas", {})

    return outcfg


# from 25.9.90 to 25.9.91-7bc73979
@register_config_upgrade(from_version=(25, 9, 90), to_version=(25, 9, 91))
def upgrade_config_2509_90_to_2509_91(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    insert_config_key(outcfg, "model.use_slang", True)
    for layer_name, layer in outcfg.model.layers.items():
        insert_config_key(layer, "use_slang", True)

    return outcfg


@register_config_upgrade(from_version=(25, 9, 142), to_version=(25, 9, 143))
def upgrade_config_2509_142_to_2509_143_calib_config_changes(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # separating calib.enabled to calib.lidar.enabled and calib.camera.enabled
    calib_enabled = OmegaConf.select(outcfg, "model.calib.enabled", default=False)
    if not exists_config_key(outcfg, "model.calib.camera.enabled"):
        insert_config_key(outcfg, "model.calib.camera.enabled", calib_enabled)
    if not exists_config_key(outcfg, "model.calib.lidar.enabled"):
        insert_config_key(outcfg, "model.calib.lidar.enabled", calib_enabled)
    remove_config_key(outcfg, "model.calib.enabled")

    return outcfg


@register_config_upgrade(from_version=(25, 10, 58), to_version=(25, 10, 59))
def upgrade_config_2510_58_to_2510_59_add_sensor_specific_extra_signal(cfg: DictConfig) -> DictConfig:
    """Add sensor-specific extra signal configuration fields.
    Add sensor_type and activation to individual extra_signal entries.
    Migrate particle.extra_signal_dim to sensor-specific dimensions.
    Add sensor-specific optimizer entries.

    For backward compatibility with old artifacts:
    - Sets all n_signal_dim values to 0 in existing extra_signal configs
    - Sets camera/lidar_extra_signal_dim to 0
    This ensures extra_signal_infos dimensions match the 0-dimension tensors created
    by the model upgrade, preventing assert failures in ExtraSignal.from_packed_tensor.

    Upgrade from 25.10.58 (1270e2e4) to 25.10.59 (c3b97fcd)
    """
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Process each layer
    for _layer_name, layer in outcfg.model.layers.items():
        # Add sensor_type and activation to individual extra_signal entries
        if "extra_signal" in layer and isinstance(layer.extra_signal, DictConfig):
            for _signal_name, signal_config in layer.extra_signal.items():
                if isinstance(signal_config, DictConfig):
                    # Add sensor_type and activation to each signal
                    insert_config_key(signal_config, "sensor_type", "camera")
                    insert_config_key(signal_config, "activation", "none")
                    # Set n_signal_dim to 0 for backward compatibility to match 0-dimension tensors
                    insert_config_key(signal_config, "n_signal_dim", 0)

        # Update particle section with sensor-specific dimensions
        if "particle" in layer:
            # IMPORTANT: Set extra_signal_dim to 0 for backward compatibility
            # This overrides any dynamic computation or existing value
            insert_config_key(layer, "particle.extra_signal_dim", 0)
            # Add sensor-specific dimensions
            insert_config_key(layer, "particle.camera_extra_signal_dim", 0)
            insert_config_key(layer, "particle.lidar_extra_signal_dim", 0)
            insert_config_key(layer, "particle.lidar_extra_signal_sph_degree", 0)

        # Add sensor-specific optimizer entries to the optimizer that has extra_signal params
        if "optimizers" in layer and isinstance(layer.optimizers, list):
            for optimizer in layer.optimizers:
                # Find the optimizer that has extra_signal in its params
                if "params" in optimizer and "extra_signal" in optimizer.params:
                    insert_config_key(optimizer.params, "camera_extra_signal", OmegaConf.create({"args": {"lr": 0.01}}))
                    insert_config_key(optimizer.params, "lidar_extra_signal", OmegaConf.create({"args": {"lr": 0.01}}))
                    break  # Only add to one optimizer

    return outcfg


@register_config_upgrade(from_version=(25, 10, 109), to_version=(25, 10, 110))
def upgrade_config_2510_109_to_2510_110_prepare_before_render(cfg: DictConfig) -> DictConfig:
    """Add model.renderer.prepare_before_render configuration key.
    Upgrade from 25.10.109 to 25.10.110
    """
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Add the missing prepare_before_render key to model.renderer
    if not exists_config_key(outcfg, "model.renderer.prepare_before_render"):
        insert_config_key(outcfg, "model.renderer.prepare_before_render", True)

    return outcfg


@register_config_upgrade(from_version=(25, 10, 121), to_version=(25, 10, 122))
def upgrade_config_2510_121_to_2510_122_calib_config_changes(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # seperate calib.enabled to calib.lidar.enabled and calib.camera.enabled
    remove_config_key(outcfg, "model.calib.enabled")
    insert_config_key(outcfg, "model.calib.lidar.enabled", False)
    insert_config_key(outcfg, "model.calib.camera.enabled", False)

    return outcfg


@register_config_upgrade(from_version=(25, 10, 171), to_version=(25, 10, 172))
def upgrade_config_2510_171_to_2510_172_remove_difix_batch_sampler(cfg: DictConfig) -> DictConfig:
    """Remove deprecated difix_batch_sampler from dataset.samplers.
    Upgrade from 25.10.171 to 25.10.172
    """

    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)

    # Remove difix_batch_sampler if it exists
    if exists_config_key(outcfg, "dataset.samplers.difix_batch_sampler"):
        remove_config_key(outcfg, "dataset.samplers.difix_batch_sampler")

    return outcfg


# from 25.10.188-6f256527 to 25.10.189-52c22548
@register_config_upgrade(from_version=(25, 10, 188), to_version=(25, 10, 189))
def upgrade_config_2510_188_to_2510_189_project_to_z_offset(cfg: DictConfig) -> DictConfig:
    """Add model.layers.road.initialization.project_to_z_offset configuration key.
    Upgrade from 25.10.188 to 25.10.189
    """
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Add the missing project_to_z_offset key to model.layers.road.initialization
    for layer_name, layer in outcfg.model.layers.items():
        if "initialization" in layer:
            if layer.initialization.name in (
                "lidar-rig-trajectory-road",
                "accumulated-point-cloud-road",
            ):
                if not exists_config_key(layer.initialization, "project_to_z_offset"):
                    insert_config_key(layer.initialization, "project_to_z_offset", True)

    return outcfg


@register_config_upgrade(from_version=(25, 10, 212), to_version=(25, 10, 213))
def upgrade_config_2510_212_to_2510_213_add_nearest_neighbor_track_for_lidar(cfg: DictConfig) -> DictConfig:
    """Add nearest_neighbor_track_for_lidar flag to model layers.
    Upgrade from 25.10.212 (3d721ead) to 25.10.213 (5ab07c75)
    """
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Add nearest_neighbor_track_for_lidar to each layer that doesn't have it
    for layer_name, layer in outcfg.model.layers.items():
        if not exists_config_key(layer, "nearest_neighbor_track_for_lidar"):
            insert_config_key(layer, "nearest_neighbor_track_for_lidar", False)

    return outcfg


@register_config_upgrade(from_version=(25, 10, 211), to_version=(25, 10, 212))
def upgrade_config_2510_211_to_2510_212_initialization_config_changes(cfg: DictConfig) -> DictConfig:
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Add the missing non_dynamic_points_only key
    for layer_name, layer in outcfg.model.layers.items():
        if "initialization" in layer:
            if layer.initialization.name in (
                "lidar-rig-trajectory",
                "lidar-rig-trajectory-road",
                "accumulated-point-cloud",
                "accumulated-point-cloud-road",
            ):
                if not exists_config_key(layer.initialization, "non_dynamic_points_only"):
                    insert_config_key(layer.initialization, "non_dynamic_points_only", True)

    return outcfg


@register_config_upgrade(from_version=(26, 2, 80), to_version=(26, 2, 81))
def upgrade_config_2602_80_to_2602_81_add_radiance_sph_O0(cfg: DictConfig) -> DictConfig:
    """Add particle.radiance_sph_O0=true parameter to SHGaussian layers.
    Upgrade from 26.2.80 (deeffb51) to 26.2.81 (cda9ec18)
    """
    # Create a copy of the input config to avoid modifying the original
    outcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(outcfg, DictConfig)  # needed for mypy

    # Add radiance_sph_O0=true if it doesn't exist
    for layer_name, layer in outcfg.model.layers.items():
        if layer.name == "sh-gaussians":
            if "particle" in layer:
                if not exists_config_key(layer.particle, "radiance_sph_O0"):
                    insert_config_key(layer.particle, "radiance_sph_O0", True)

    return outcfg
