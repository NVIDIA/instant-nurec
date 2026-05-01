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
import os
import re

from pathlib import Path
from typing import Any, Literal, Optional

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import model_validator

# Import from nre.config.parse to introduce OmegaConf resolvers
import nre.config.parse  # noqa: F401

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.logger import BatchMediaLoggerConfigMixin, LoggerConfigType
from nre.config.trainer import infer_slurm_environment
from nre.config.version import Version, get_version
from nre.nrm.config.dataset import NRMSplitsConfig
from nre.nrm.config.models import KelvinModelConfig
from nre.nrm.config.predict import PredictConfig
from nre.repo_root import __reporoot__
from nre.utils.model_registry import create_model_registry


SENTINEL = "<sentinel>"

current_version = get_version()

cmd_logger = logging.getLogger(__name__)


class BaseNRMSystemConfig(BatchMediaLoggerConfigMixin, BaseConfigSchema):
    """
    Currently we only have one NRM system, and this is the base class for it.
    In the future we might want to have different subclasses for this.

    Note that this is a merge of NRE's system config, trainer config, and datamodule config for the ease of use.
    """

    device_count: int = Field(
        default=0,
        description="Number of devices per node. If set to 0, will be inferred automatically.",
    )

    num_nodes: int = Field(
        default=0,
        description="Number of nodes to use for distributed training. If set to 0, will be inferred automatically.",
    )

    precision: int | str = Field(
        description="https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision",
    )

    predict_num_workers: int = Field(default=0, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=1, description="Batch size for the predict dataloader. Typically set to 1.")

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        slurm_environment = infer_slurm_environment()

        if self.device_count == 0:
            self.device_count = slurm_environment.num_tasks_per_node if slurm_environment else 1

        if self.num_nodes == 0:
            self.num_nodes = slurm_environment.num_nodes if slurm_environment else 1


class GaussiansNRMSystemConfig(BaseNRMSystemConfig):
    """
    System config for the Gaussians NRM system.
    """

    name: Literal["base-nrm-system"]


class _CheckpointConfig(BaseConfigSchema):
    save_top_k: int
    monitor: str | None
    mode: Literal["min", "max"]


class NRMConfig(BaseConfigSchema):
    """
    Top-level configuration for NRM training/validation/testing.
    """

    seed: int = Field(default=38, description="Random seed.")
    mode: Literal["train", "val", "test", "trainval", "predict"]

    resume: str | None
    resume_weights_only: bool
    call_train_from_scratch_hook_for_validation: bool = Field(
        default=False,
        description=(
            "When True, the model's on_train_from_scratch_start hook is also called when running val/test/predict "
            "without a checkpoint but with init weights. Set to False if the hook contains train-only logic that "
            "must not run during eval (e.g. writing training state)."
        ),
    )
    verbose: bool = Field(default=False, description="Verbose mode.")

    out_dir: str
    logger: LoggerConfigType = Field(discriminator="name")

    checkpoint: _CheckpointConfig

    system: GaussiansNRMSystemConfig = Field(discriminator="name")
    dataset: NRMSplitsConfig = Field(discriminator="name")

    model: KelvinModelConfig = Field(discriminator="name")

    # Predict configuration
    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Configuration for predict-time-only functionality such as primitive merging",
    )

    version: Version | None = Field(
        default=current_version,
        description="Not to be set by the user. Used to detect NRE version mismatch when loading old configs. Not available in sandboxed test executions",
    )

    save_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where images are saved during validation phase. If left unchanged, defaults to `out_dir/save`"
        ),
    )
    ckpt_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where model checkpoints are saved during training. If left "
            "unchanged, defaults to `out_dir/checkpoints`"
        ),
    )
    config_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where the config (with all the auto-generated and default "
            "fields) will be stored. If left unchanged, defaults to `out_dir/config`"
        ),
    )
    run_id: str = Field(
        default=SENTINEL,
        description=(
            "A unique identifier of the training run. If left unchanged, will be auto-generated. "
            "If resuming training from a checkpoint, the previous run_id will be restored."
        ),
    )

    class OverrideConfig(BaseConfigSchema):
        selector: list[str] = Field(
            description="Selector for the config to override. One can use regex to match the items."
        )
        config: Any = Field(description="The actual config to be merged with the selected configs.")

    shared_config: Optional[Any] = Field(
        default=None,
        description="Place to put common configs to be shared via YAML's `<<: *shared_config` syntax. "
        "This will not come to the final config, just for handy purposes",
    )
    override_config: dict[str, OverrideConfig] = Field(
        default_factory=dict,
        description="Override config for the entire config. Can be used in junction with shared_config to create various combinations.",
    )

    @model_validator(mode="before")
    @classmethod
    def merge_override_config(cls, objects: DictConfig) -> DictConfig:
        # Recursively print out keys from the config to perform matching
        def iter_key_paths(d: dict[str, Any], prefix: str = ""):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield path
                    yield from iter_key_paths(v, path)

        # Apply the overrides one by one
        if "override_config" in objects:
            for override_name, override_item in objects.override_config.items():
                override_model = cls.OverrideConfig.model_validate(override_item)

                matched_paths: list[str] = []
                for path in iter_key_paths(OmegaConf.to_container(objects)):  # type: ignore
                    for sel_regex in override_model.selector:
                        if re.fullmatch(sel_regex, path):
                            matched_paths.append(path)
                            break

                # Override the config paths
                for path in matched_paths:
                    cmd_logger.info(f"Overriding {path} with override key '{override_name}'.")
                    config_items = OmegaConf.select(objects, path)
                    assert isinstance(config_items, DictConfig)
                    with open_dict(config_items):
                        config_items.merge_with(override_model.config)
            OmegaConf.update(objects, "override_config", {}, merge=False)

        # Remove config overrides from the final config.
        if "shared_config" in objects:
            OmegaConf.update(objects, "shared_config", None)
        return objects

    def _setup_resume_path(self) -> None:
        if self.resume is None:
            return

        # When user use resume=auto, we will try to resume from the last checkpoint with the same run_id.
        # If the checkpoint does not exist, we will train from scratch.
        # This is useful for long-running training jobs on clusters where the job is constantly interrupted & restarted.
        if self.resume == "auto":
            is_auto_resume = True
            self.resume = "last"
        else:
            is_auto_resume = False

        # Add .ckpt if config.resume doesn't include it. Useful to run resume=last or resume=best
        if not self.resume.endswith(".ckpt"):
            self.resume += ".ckpt"
        if not os.path.exists(self.resume):
            if self.ckpt_dir == SENTINEL:
                if is_auto_resume:
                    # Automatically find out the checkpoint directory
                    self.ckpt_dir = os.path.join(self.out_dir, self.logger.run_id, "checkpoints")
                    cmd_logger.info(f"Automatically resuming from {self.ckpt_dir}")
                else:
                    raise FileNotFoundError("config.ckpt_dir is not provided, can't guess the resume path.")

            self.resume = os.path.join(self.ckpt_dir, self.resume)
            if not os.path.exists(self.resume):
                if is_auto_resume:
                    # If checkpoint not found, training from scratch!
                    self.resume = None
                    cmd_logger.info("Checkpoint not found, training from scratch.")
                else:
                    raise FileNotFoundError(f"Checkpoint {self.resume=} does not exist")

    def _setup_run_id(self) -> None:
        """propagate run_id set in self.logger to other fields"""
        run_id = self.logger.run_id
        self.run_id = run_id
        self.save_dir = os.path.join(self.out_dir, run_id, "save")
        self.ckpt_dir = os.path.join(self.out_dir, run_id, "checkpoints")
        self.config_dir = os.path.join(self.out_dir, run_id, "config")

    def model_post_init(self, __context) -> None:
        self._setup_resume_path()
        self._setup_run_id()  # modifies self.ckpt_dir so needs to be called AFTER _setup_resume_path


def parse_untyped_config(
    config_name: str, hydra_args: list[str], config_dir: str = "./configs"
) -> tuple[DictConfig, str]:
    """Utility to parse a hydra base config (with overwrites) without type checking"""

    config_name_path = Path(config_name)

    # Adding repo config path to hydra search path so we can refer to those in overrides
    # Example override: +nrm/dataset/concrete@dataset.predict=ncore_local_low_res
    search_path_config: str = f"hydra.searchpath=['{(__reporoot__) / Path(config_dir)}']"

    # Use hydra context for clean-up.
    # We always assume that the main config dir is the one containing config_name, and rest configs are found via search paths.
    with hydra.initialize_config_dir(config_dir=str(config_name_path.parent.resolve()), version_base=None):
        config = hydra.compose(config_name=str(config_name_path.name), overrides=hydra_args + [search_path_config])

    return config, search_path_config


def parse_pretrained_nrm_config(
    config_name: str, hydra_args: list[str], config_dir: str = "./configs"
) -> tuple[str, list[str]]:
    """Utility function to download a pretrained model checkpoint and associated config from registry, if the config is a pretrained NRM config.

    Returns *updated* config_name and hydra_args with `resume` overrides to the pretrained model"""

    # load basic config without overwrites
    config, _ = parse_untyped_config(config_name, [], config_dir=config_dir)

    # check if the basic config is a pretrained model config - if so, obtain the model checkpoint and config from the registry - patch up the config_name and hydra_args accordingly
    if (model_url := config.get("model_url", None)) is not None and (
        model_config := config.get("model_config", None)
    ) is not None:
        assert not any([param.startswith("resume=") for param in hydra_args]), (
            "resume cannot be used with pretrained model configs"
        )

        model_path = create_model_registry(model_url, Path(config["model_cache_dir"])).get_model()

        # Models stored in the registry often only have the state dict kept.
        hydra_args.insert(0, "++resume_weights_only=true")

        if model_config.startswith("http"):
            # if 'model_config' is a http URL, download the corresponding config file, otherwise assume string points to local config
            model_config = create_model_registry(model_config, Path(config["model_cache_dir"])).get_model()

        # Allow for overrides to the downloaded model config
        model_config_overrides: list[str] = config.get("model_config_overrides", [])
        hydra_args = model_config_overrides + hydra_args

        # use the downloaded checkpoint as the main entrypoint for the downstream config
        hydra_args.append(f"++resume={model_path}")  # points to the absolute path of the downloaded checkpoint
        config_name = (
            model_config  # points to the absolute path of the checkpoint-associated config to load, or a local config
        )

    return config_name, hydra_args


def parse_untyped_nrm_config(
    config_name: str,
    hydra_args: list[str],
    config_dir: str = "./configs",
) -> DictConfig:
    """
    Parse NRM config without type checking. This is made specifically for the NRM config where we want to
    refer to other NRM configs in the overrides, potentially relative to the nrm config directory.
    """

    # Load pretrained config/checkpoint if specified
    config_name, hydra_args = parse_pretrained_nrm_config(config_name, hydra_args, config_dir=config_dir)

    # Load the config
    config, _ = parse_untyped_config(config_name, hydra_args, config_dir=config_dir)

    OmegaConf.resolve(config)

    return config


def parse_typed_nrm_config(
    config_name: str,
    hydra_args: list[str] | tuple[str],
    config_dir: str = "./configs",
) -> NRMConfig:
    untyped_config = parse_untyped_nrm_config(
        config_name,
        # convert to list in case we have a tuple
        list(hydra_args),
        config_dir=config_dir,
    )
    typed_config = NRMConfig.model_validate(untyped_config, context={"config_name": config_name})
    return typed_config
