# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import copy
import json
import logging
import os
import warnings

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Literal, Mapping, Tuple, TypeVar, Union, cast

import numpy as np
import point_cloud_utils as pcu
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer, do_nothing_closure
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig as PL_OptimizerLRSchedulerConfig
from scipy.spatial.transform import Rotation
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

# DON'T REMOVE THIS IMPORT, Used to force pycena discovery of the losses functions
import libs.losses.models.registry  # noqa: F401

# Hack: Pure import datamodules and losses to ensure modules are registered for obfuscation, cf. pycena
# TODO: remove this once obfuscation supports __init__.py file parsing, see nischneider/rgb-error-debug
import nre.datamodules

from libs.losses.orchestration.config import LossAggregatorReturn, LossReturnType
from libs.losses.orchestration.loss_aggregator import LossAggregator
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.sensors import OpenCVPinholeCameraModel
from nre.config.base_schema import config_to_primitive
from nre.config.nre import NREConfig
from nre.config.systems import (
    BaseSystemConfig,
    GaussiansSystemConfig,
    NRendTestGaussiansSystemConfig,
)
from nre.datamodules.base import PrefetchDataLoader, SODataModule
from nre.datasets.base import RigTrajectoriesProvider
from nre.datasets.ncore import NCOREDataSource
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.difix.model import DifixModelFactory
from nre.models.background import SkyEnvMapBackground
from nre.models.base import BaseModel
from nre.models.composite import CompositeModel
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.gaussians_model import distributed_all_gather_gaussian_parameters
from nre.systems.utils import (
    system_config_compatibility_check,
    update_datamodule_epoch,
    update_module_step,
)
from nre.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    RenderingBatch,
    RenderingData,
)
from nre.utils.geometry import se3_matrix_to_tquat
from nre.utils.io.artifact_cache import ArtifactCache
from nre.utils.io.checkpoint import reduce_precision_to_fp16, upcast_fp16_to_fp32
from nre.utils.io.ground_mesh import get_nominal_ground_point_under_lidar, reconstruct_ground_mesh_from_points
from nre.utils.io.mesh import Mesh, get_usd_references, mesh_from_point_cloud, serialize_mesh, smooth_mesh
from nre.utils.io.metadata import get_metadata, serialize_metadata
from nre.utils.io.rig_trajectories import rig_trajectories_time_range, serialize_rig_trajectories
from nre.utils.io.sequence_tracks import serialize_sequence_tracks
from nre.utils.io.usd_nrend import serialize_nrend_usd
from nre.utils.metrics import MetricsCollector, create_metric_sample
from nre.utils.misc import (
    distributed_all_gather_nested,
    is_env_true,
    rank_zero_only,
    sync_objects,
    to_torch,
    unpack_optional,
)
from nre.utils.optim import configure_optimizers
from nre.utils.profiling import ScopedTimer
from nre.utils.trainer import BroadcastExceptions, TrainerConfig, adjust_step_for_world_size
from nre.utils.types import (
    Checkpoint,
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    NamedSerialized,
    PointCloud,
)
from nre.utils.visualize import log_image as log_image_from_utils_visualize


warnings.filterwarnings("ignore")
if is_env_true("CUDA_SYNC_DEBUG", False):
    warnings.filterwarnings("default", category=UserWarning, message=".*synchronizing CUDA operation.*")

logger = logging.getLogger(__name__)

R = TypeVar("R")


class DDPStrategySO(DDPStrategy):
    """Custom DDP strategy for Scene Optimization."""

    _sharded_params_and_buffers: list[str] | None = None
    _ddp_configured = False
    _verbose = True

    # NOTE: In this class, we are using a lot of internal APIs and implementations from pytorch-lightning, so we likely need to
    # revisit this implementation everytime we upgrade pytorch-lightning!!!
    import pytorch_lightning

    assert pytorch_lightning.__version__ == "2.4.0", (
        "DDPStrategySO is only validated with pytorch-lightning 2.4.0, please revisit its implementation when upgrading pytorch-lightning"
    )

    def setup(self, trainer: "pytorch_lightning.Trainer") -> None:
        """Setup DDP Strategy for Scene Optimization.
        Note that configure_ddp is only called for training, not for validation or testing. Therefore we need to overload the higher-level setup method.
        Reference:
        - https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/ddp.py#L280-L284
        - https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/ddp.py#L157
        """
        logger.info(f"Configuring DDP for {self.model.__class__.__name__}")
        assert isinstance(self.model, torch.nn.Module), "Module must have been initialized before calling configure_ddp"
        # Exclude all per-Gaussian parameters from the DDP, since their shapes vary during training,
        # which significantly slows down DDP. We will handle them manually by:
        #
        # Each rank will host a subset of Gaussians, which will be broadcasted to all ranks (all_gather)
        # every iteration before rendering happens.
        #
        # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/parallel/distributed.py#L2261
        #
        # Usage: https://github.com/pytorch/pytorch/blob/07df6ba7f5597488a93b3855d52d2ead55675125/torch/nn/parallel/distributed.py#L2285
        # - Need to call this function again on the top level model, otherwise the DDP will not ignore the parameters and buffers
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self.model, self.sharded_params_and_buffers)
        if self._sharded_params_and_buffers is not None:
            DDPStrategySO._pretty_log_keys(
                "DDP - excluding following parameters and buffers",
                self._sharded_params_and_buffers,
                print_func=logger.info,
            )
        self._ddp_configured = True
        # Call the original method to continue the setup process
        super().setup(trainer)

    @property
    def base_system(self) -> BaseModel:
        """Query the base system safely"""
        system = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        assert isinstance(system, BaseSystem), "Model must be an instance of BaseSystem"
        return system

    @property
    def sharded_params_and_buffers(self) -> list[str]:
        """Retrieve the sharded model parameters and buffers"""
        if self._sharded_params_and_buffers is None:
            self._sharded_params_and_buffers = self.base_system.configure_sharded_params_and_buffers()
        return self._sharded_params_and_buffers

    def sharded_optimizer_state_indices(
        self, optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]
    ) -> list[int]:
        """Retrieve the the sharded optimizer state indices of the given optimizer"""
        self.base_system.configure_sharded_params_and_buffers()  # Make sure sharded flags are available
        # At this stage, configure_ddp() should have been called, so the model is a DistributedDataParallel instance
        assert isinstance(self.model, DistributedDataParallel), "Model must be an instance of DistributedDataParallel"
        # We then sync the state of the optimizer for the parameters that are not sharded
        indices = []
        for group, group_dict in zip(optimizer.param_groups, state_dict["param_groups"]):
            for p, i in zip(group["params"], group_dict["params"]):
                if BaseModel.is_sharded(p):
                    indices.append(i)
        return indices

    def split_params_and_buffers(self, full_states: dict[str, Any]) -> dict[str, Any]:
        """Split the parameters and buffers"""
        world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # Make a deep copy of the full states to avoid modifying the original states
        sharded_states = copy.deepcopy(full_states)
        # Split the parameters and buffers
        for p in self.sharded_params_and_buffers:
            sharded_states[p] = sharded_states[p][world_rank::world_size]
        return sharded_states

    def split_optimizer_states(self, optimizer: torch.optim.Optimizer, full_states: dict[str, Any]) -> dict[str, Any]:
        """Split the optimizer states"""
        world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # Make a deep copy of the full states to avoid modifying the original states
        sharded_states = copy.deepcopy(full_states)
        # Split the optimizer states
        sharded_indices = self.sharded_optimizer_state_indices(optimizer, full_states)
        for i in sharded_indices:
            s = sharded_states["state"][i]
            for k, v in s.items():
                s[k] = v[world_rank::world_size]
        return sharded_states

    def lightning_module_state_dict(self) -> dict[str, Any]:
        """Returns model state.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L473-L476
        """
        assert self.lightning_module is not None
        state_dict = copy.deepcopy(self.lightning_module.state_dict())

        # Extract the sharded parameters and buffers. Because `sharded_params_and_buffers` contains the names of different gaussian nodes,
        # we need to perform the all-gather operation for each gaussian node separately due to their different gaussian counts.
        sharded_states: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for name in self.sharded_params_and_buffers:
            if name in state_dict:
                module_name = DDPStrategySO._get_module_name(name)
                sharded_states[module_name][name] = state_dict[name]

        # All-gather the sharded parameters and buffers
        for sharded_module_states in sharded_states.values():
            full_module_states = distributed_all_gather_gaussian_parameters(sharded_module_states, stable=True)
            state_dict.update(full_module_states)

        # Log the keys that are being all-gathered
        if sharded_states and DDPStrategySO._verbose:
            DDPStrategySO._pretty_log_keys(
                f"Gathering Module State for {self.lightning_module.__class__.__name__}", list(sharded_states.keys())
            )

        return state_dict

    def optimizer_state(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Return the optimizer state
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L173-L189
        """
        state_dict = copy.deepcopy(
            super().optimizer_state(optimizer)
        )  # We first get the regular state dict for the optimizer

        # We also need to access the optimizer directly
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        assert not hasattr(optimizer, "consolidate_state_dict"), "ZeroRedundancyOptimizer is not supported"

        # Collect the sharded states for this parameter group
        sharded_states: dict[str, Any] = {}
        for i in self.sharded_optimizer_state_indices(optimizer, state_dict):
            s = state_dict["state"][i]
            assert isinstance(s, dict), f"State must be a dict"
            for k, v in s.items():
                sharded_states[f"{i}/{k}"] = v

        # All-gather the sharded states (We know that different gaussian nodes have different optimizers, so we don't
        # need to worry about gaussian count mismatch).
        full_states = distributed_all_gather_gaussian_parameters(sharded_states, stable=True)
        for key, value in full_states.items():
            si, k = key.split("/")
            state_dict["state"][int(si)][k] = value  # type: ignore[index]

        # We log the keys that are being all-gathered
        if sharded_states and DDPStrategySO._verbose:
            DDPStrategySO._pretty_log_keys(
                f"Gathering Optimizer State for {self.lightning_module.__class__.__name__}", list(sharded_states.keys())
            )

        return state_dict

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        """The function is called by the trainer to actually load the model state_dict.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L369-L371
        """
        # Actually load the sharded states into the model (use assign=True to initialize Gaussians parameters)
        sharded_states = self.split_params_and_buffers(checkpoint["state_dict"])
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(sharded_states, strict=strict, assign=True)
        # Restore sharded flags are still valid after assignments
        self.base_system.configure_sharded_params_and_buffers()
        self._resumed_with_full_state_dict = True
        logger.info(f"Loaded model state dict for {self.lightning_module.__class__.__name__}")

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Load the optimizer state_dict.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L373-L377
        """
        from lightning_fabric.utilities.optimizer import _optimizer_to_device  # type: ignore[import-untyped]

        optimizer_states = checkpoint["optimizer_states"]
        for opt, opt_state in zip(self.optimizers, optimizer_states):
            opt.load_state_dict(self.split_optimizer_states(opt, opt_state))
            _optimizer_to_device(opt, self.root_device)
        logger.info(f"Loaded optimizer state dict for {self.lightning_module.__class__.__name__}")

    @staticmethod
    def _get_module_name(name: str) -> str:
        """Get the module name from the full name"""
        return ".".join(name.split(".")[:-1])

    @staticmethod
    def _pretty_log_keys(
        banner: str, keys: list[str], *, indent: int = 0, print_func: Callable[[str], None] = tqdm.write
    ) -> None:
        print_func(f"{' ' * indent}{banner}:")
        for k in keys[:-1]:
            print_func(f"{' ' * (indent + 4)}|─{k}")
        print_func(f"{' ' * (indent + 4)}└─{keys[-1]}")


class BaseSystem(LightningModule, ABC, Generic[LossReturnType]):
    model: BaseModel
    config: BaseSystemConfig
    device: torch.device

    def __init__(self, config: NREConfig) -> None:
        super().__init__()

        self.trainer_config: TrainerConfig = config.trainer

        # handle the training loop iteration ourselves (e.g., to support multiple optimizers)
        self.automatic_optimization = False

        self.save_hyperparameters(config_to_primitive(config.to_dictconfig()))

        # save what we need from the config
        self.mode = config.mode
        self.max_train_steps = config.trainer.max_epochs * adjust_step_for_world_size(
            config.trainer, config.dataset.n_samples_per_epoch
        )
        self.update_n_epochs = config.datamodule.update_n_epochs
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.resume = config.resume
        self.difix_config = config.difix

        self.config = config.system
        self.trainer_config = config.trainer
        self.logger_enabled = config.logger.enabled

        self.datamodule = nre.datamodules.make(config.datamodule.name, config)
        assert config.loss is not None
        self.loss = LossAggregator(config.loss, config.trainer)
        self.training_step_outputs: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        self.validation_step_outputs: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

        self.last_checkpoint_step = -1  # prevent re-serialization of already processed steps

        # Map config mode to metrics collector expected values
        if config.mode == "train":
            metrics_mode: Literal["train", "val", "grpc_server", "unknown"] = "train"
        elif config.mode == "val":
            metrics_mode = "val"
        else:
            metrics_mode = "unknown"

        self._validation_metrics_collector = MetricsCollector(
            train_config_name=config.train_config_name,
            mode=metrics_mode,
            run_id=config.run_id,
            version=config.version,
        )

        self.outer_loop_timer = ScopedTimer("BaseSystem/outer_loop")
        self.inner_loop_timer = ScopedTimer("BaseSystem/training_step")

    def configure_sharded_params_and_buffers(self) -> list[str]:
        return [f"model.{p}" for p in self.model.configure_sharded_params_and_buffers()]

    @abstractmethod
    def forward(self, batch: DataAndRenderingBatch) -> Any:
        """Forward function for the system."""
        ...

    def setup(self, stage: str) -> None:
        # The `setup` function is called by the `pl.Trainer` fit/validate/test functions. At this point,
        # the DDP environment is already initialized, unlike in the `__init__` function. Therefore,
        # we move the initialization of the model to the `setup` function.
        super().setup(stage)
        self.model.to(self.device)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfers a batch to the target device with non-blocking copy to avoid
        cudaStreamSynchronize being added under the hood.

        The `dataloader_idx` argument is kept to match the
        Lightning hook signature.
        Ref: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html#lightning.pytorch.core.hooks.DataHooks.transfer_batch_to_device
        """
        return batch.to(device, non_blocking=True)

    def training_step(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        """Evaluates single iteration of the external training loop (can be overwritten in derived classes to modify)"""
        self.outer_loop_timer.stop(force=True)
        # Set step BEFORE starting inner_loop_timer so scopedtimer_window range
        # properly encompasses training_step range (correct NVTX nesting)
        ScopedTimer.set_step(int(self.global_step))
        self.inner_loop_timer.start()

        # Due to a barrier inside manual_backward below,
        # we need to ensure that either all process or none enters it.
        # If an exception is raised, it'll be broadcasted to all processes
        # and training will abort instead of hang.
        with BroadcastExceptions(self.trainer):
            # 0. Construct list of optimizers (also for single optimizer case)
            pl_optimizers = self.optimizers(
                # PL optimizers will automatically handle precision and profiling
                use_pl_optimizer=True
            )
            pl_optimizers = pl_optimizers if isinstance(pl_optimizers, list) else [pl_optimizers]

            # 1. Invalidate gradients in all optimizers
            for pl_optimizer in pl_optimizers:
                # note: this assumes to set None (not zero-values) to '.grad' properties of parameter tensors, but we can't
                # request this explicitly here as the "default" 'set_to_none' keyword property is not supported by all
                # optimizers, e.g., 'apex.FusedAdam.zero_grad()' is doesn't respect a 'set_to_none' keyword parameter
                pl_optimizer.optimizer.zero_grad()

            # 2. Evaluate *system-level* training losses for given batch
            loss_return = self.training_losses(batch, batch_local_idx)
            self.on_training_losses_return(loss_return)

        # Broadcast exceptions raised during optimization.
        # Note: this is adding synchronization overhead, which might be significant if the
        #       workload is uneven across processes. Ideally one would only broadcast exceptions
        #       when the processes are idle, e.g., right after backward pass
        #       (where the gradients are synchronized).
        #       I couldn't implement this idea. We'd have to call self.manual_backward() with a possibly
        #       non-existing loss_return. I've tried to create dummy total_value Tensor:
        #           1. empty tensor: resulted in error (no gradient found in element 0)
        #           2. 0.0 value, with gradient: assertion error
        with BroadcastExceptions(self.trainer):
            # 3. Backprop gradients of *system-level* training losses to parameters
            # Note 1: there's a inter-process barrier here to gather the gradients of
            #         all subprocesses.  All processes or none must call this function
            #         or else training will hang indefinitely.
            # Note 2: we're assuming that no exceptions occur in the code
            #         leading to the barrier inside manual_backward. If they do, process will hang.
            #         I don't think this is likely to happen, though (famous last words).
            with ScopedTimer("BaseSystem/training_step/manual_backward"):
                self.manual_backward(loss_return.total_value)

            # 4. Step optimizers to update parameters
            # Manually maintain by ourselves the trainer's optimizer step progress, to temporarily fix https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
            self.trainer.profiler.start("optimizer_step")  # type: ignore[attr-defined]

            with ScopedTimer("BaseSystem/training_step/optimizer_step"):
                for pl_optimizer in pl_optimizers:
                    # skip step if none of the parameters has associated gradients after backpropagation (e.g., because all parameters are frozen)
                    # circumventing gradient scaler shortcomings not able to handle this setting
                    if all([p.grad is None for group in pl_optimizer.optimizer.param_groups for p in group["params"]]):
                        continue
                    if pl_optimizer._strategy is not None:
                        pl_optimizer._strategy.optimizer_step(pl_optimizer._optimizer, do_nothing_closure)

            self.trainer.profiler.stop("optimizer_step")  # type: ignore[attr-defined]
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

            # 5. Handle lr-schedulers
            for lr_scheduler_config in self.trainer.strategy.lr_scheduler_configs:
                # check if we need to step this lr-scheduler / exit early if not
                match interval := lr_scheduler_config.interval:
                    case "epoch":
                        if not (
                            self.trainer.is_last_batch
                            and (self.trainer.current_epoch + 1) % lr_scheduler_config.frequency == 0
                        ):
                            continue  # step not required
                    case "step":
                        if not ((batch_local_idx + 1) % lr_scheduler_config.frequency == 0):
                            continue  # step not required
                    case _:
                        raise RuntimeError(f"Invalid scheduler interval '{interval}'")

                # step lr-scheduler depending on type
                if not isinstance(
                    scheduler := lr_scheduler_config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # step a regular scheduler
                    scheduler.step()
                else:
                    # incorporate the monitored metric for ReduceLROnPlateau
                    assert (monitor := lr_scheduler_config.monitor) is not None, (
                        "Missing metric to monitor for ReduceLROnPlateau scheduler"
                    )

                    if (monitor_metric := self.trainer.callback_metrics.get(monitor)) is not None:
                        # step the scheduler with available metric
                        scheduler.step(monitor_metric)
                    else:
                        # monitored metric is not available - warn / error out depending on strict flag
                        if lr_scheduler_config.strict:
                            raise RuntimeError(f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler")
                        else:
                            logger.warning(
                                f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler - skipping scheduler update"
                            )

        self.inner_loop_timer.stop()
        self.outer_loop_timer.start()

    @abstractmethod
    def training_losses(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> LossReturnType:
        """Evaluate system-specific losses for a single training batch"""
        ...

    @property
    def is_trainer_attached(self) -> bool:
        """Check if pytorch_lightning.Trainer is hooked up (we're inside .fit loop)"""
        # There seems to be no official way to check if trainer is attached
        # https://github.com/Lightning-AI/pytorch-lightning/issues/17517#issuecomment-1532325528
        return self._trainer is not None

    def on_training_losses_return(self, loss_return: LossReturnType) -> None:
        """Callback for when training losses are returned."""
        pass

    def on_validation_epoch_end(self) -> None:
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """

    def on_test_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """

    def configure_optimizers(self) -> list[PL_OptimizerLRSchedulerConfig]:
        """
        Returns all system-associated as well as model-associated optimizers
        """
        system_optims = configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self.model)
        model_local_optims = self.model.configure_optimizers()
        return [cast(PL_OptimizerLRSchedulerConfig, x) for x in system_optims + model_local_optims]

    def on_validation_start(self) -> None:
        super().on_validation_start()

        val_path = os.path.join(self.out_dir, self.run_id, "val")
        if self.config.test.separate_val_dir_per_step:
            val_path = os.path.join(val_path, f"{self.global_step:06d}")
        # NOTE: Make sure all ranks have the same val_dir (should we?)
        self.val_dir = sync_objects(lambda: val_path)

        if self.difix_config.inference.enabled:
            self.difix_model = DifixModelFactory.get(
                self.difix_config.model_url,
                self.difix_config.cache_dir,
                self.difix_config.model_filename,
                self.difix_config.model_resolution,
            )

    def on_validation_end(self):
        super().on_validation_end()
        if self.difix_config.inference.enabled:
            DifixModelFactory.clear_cache()

    def on_test_end(self):
        super().on_test_end()

    def on_test_start(self) -> None:
        # NOTE: Make sure all ranks have the same test_dir (should we?)
        self.test_dir = sync_objects(lambda: os.path.join(self.out_dir, self.run_id, "test"))

        if self.difix_config.inference.enabled:
            logger.warning("Difix not supported in test mode. Continuing without Difix application.")

    def on_train_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        super().on_train_batch_start(batch, batch_local_idx)

        with ScopedTimer("BaseSystem/update_step_train_batch_start"):
            additional_parameters: dict[str, torch.Tensor] = update_module_step(
                self.model, self.current_epoch, self.global_step, self
            )
            additional_parameters |= update_module_step(self.loss, self.current_epoch, self.global_step, self)
            # We iterate over the additional parameters, if the same parameter group already exist we
            # append the parameter to the group, else we initialize a new parameter group.
            for key, value in additional_parameters.items():
                for param_group in self.trainer.optimizers[0].param_groups:
                    if param_group["name"] == key:
                        param_group["params"].append(value)
                        break
                else:
                    self.trainer.optimizers[0].add_param_group({"params": value, "name": key})

    def on_train_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_local_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_local_idx)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.outer_loop_timer.start()

    def on_train_epoch_end(self) -> None:
        self.outer_loop_timer.stop()

        timer = ScopedTimer("BaseSystem/on_train_epoch_end")
        timer.start()

        super().on_train_epoch_end()
        if self.update_n_epochs != 0 and (self.current_epoch + 1) % self.update_n_epochs == 0:
            update_datamodule_epoch(self.datamodule, self.current_epoch, self)

        timer.stop()

    def on_validation_batch_start(
        self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)
        torch.cuda.empty_cache()

    def on_test_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)

    def on_predict_batch_start(
        self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)

    def decode_value(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError("Scalar specification only supports list, got", type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
        return value

    def log_image(self, image: np.ndarray | torch.Tensor, caption: str, step: int) -> None:
        """
        Log an image to the logger.
        """
        log_image_from_utils_visualize(self.logger, image, caption, step)

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        The issue here is that the buffers can change in size so we cannot initialize them in advance and load them directly from the checkpoint
        """
        if self.is_trainer_attached:
            # Manually load all the loops as otherwise global_step and current epoch are not set correctly in the validation mode
            # see: https://github.com/Lightning-AI/lightning/issues/17127
            self.trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
            self.trainer.validate_loop.load_state_dict(checkpoint["loops"]["validate_loop"])
            self.trainer.test_loop.load_state_dict(checkpoint["loops"]["test_loop"])
            self.trainer.predict_loop.load_state_dict(checkpoint["loops"]["predict_loop"])

        # Upcast fp16 tensors back to fp32 so CUDA/Slang kernels receive the expected dtype.
        # This must happen before any loading path (DDP or non-DDP).
        checkpoint["state_dict"] = upcast_fp16_to_fp32(checkpoint["state_dict"])

        # Load the remainder of the state_dict. The reason here is that the buffers can change in size so we cannot initialize them in
        # advance and load them directly from the checkpoint.
        # NOTE: We are loading the full state_dict here. The DDPStrategySO will handle the sharding of the parameters.
        # Load the remainder of the state_dict
        if self.is_trainer_attached and isinstance(self.trainer.strategy, DDPStrategySO):
            pass
        else:
            self.load_state_dict(checkpoint["state_dict"], assign=True)

        if "trainer.world_size" in checkpoint:
            ckpt_world_size = checkpoint["trainer.world_size"]
        else:
            ckpt_world_size = 1

        # Check if the world size matches. This is a sanity check.
        if self.trainer_config.world_size != ckpt_world_size:
            logger.warning(
                f"Loading checkpoints produced by a different world size. "
                f"This might cause some minor discrepancies with the training. "
                f"World size mismatch: {self.trainer_config.world_size} != {ckpt_world_size}."
            )

        # Restore other useful information.
        self.last_checkpoint_step = checkpoint["system.checkpoint.last_checkpoint_step"]
        self.training_step_outputs = defaultdict(list, checkpoint["system.checkpoint.training_step_outputs"])

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        # First call for this step - do the aggregation across all ranks
        with ScopedTimer("BaseSystem/distributed_all_gather_nested"):
            aggregated_training_step_outputs = distributed_all_gather_nested(
                dict(self.training_step_outputs),
                only_on_rank_zero=False,  # We need it on all ranks for checkpoint
            )

        checkpoint["system.checkpoint.training_step_outputs"] = aggregated_training_step_outputs
        checkpoint["trainer.world_size"] = self.trainer_config.world_size

        self.last_checkpoint_step = self.trainer.global_step
        # TODO Check if needed once is_resumable feature is checked
        checkpoint["system.checkpoint.last_checkpoint_step"] = self.last_checkpoint_step

    def log(self, *args, **kwargs) -> None:
        """
        Log metrics if logging is enabled in the configuration.

        This method is a wrapper around the PyTorch Lightning's log method
        that respects the logger configuration setting.

        Args:
            *args: Arguments to pass to PyTorch Lightning's log method
            **kwargs: Keyword arguments to pass to PyTorch Lightning's log method
        """
        if self.logger_enabled:
            super().log(*args, **kwargs)
        else:
            pass

    def on_train_start(self) -> None:
        """Start of the training loop"""
        super().on_train_start()

    def on_train_end(self) -> None:
        """Clean up resources when training ends."""
        super().on_train_end()

        # Clean up train dataloader if exists
        try:
            train_loader = self.datamodule.train_dataloader()

            # PrefetchDataLoader runs a background thread in the main process that prefetches data
            # from worker processes and transfers it to the GPU. This thread needs to be explicitly
            # terminated to prevent resource leaks and ensure clean process shutdown.
            if isinstance(train_loader, PrefetchDataLoader):
                train_loader.shutdown()
        except (AttributeError, NotImplementedError, TypeError):
            # Safely handle cases where train_dataloader is not available
            pass


class BaseSystemSO(BaseSystem[LossAggregatorReturn]):
    model: GaussiansComposite  # to be initialized by subclasses
    datamodule: SODataModule
    config: GaussiansSystemConfig | NRendTestGaussiansSystemConfig

    def __init__(self, config: NREConfig) -> None:
        system_config_compatibility_check(config.to_dictconfig())

        super().__init__(config)
        assert isinstance(self.datamodule, SODataModule)
        self.train_num_rays = config.model.train_num_rays
        # For each artifact type that does not get trained, cache its data after the first export
        self.artifact_cache = ArtifactCache(
            artifact_config=config.checkpoint.artifact,
        )

        self._validation_metrics_collector = MetricsCollector(
            train_config_name=config.train_config_name,
            mode="val",
            run_id=self.run_id,
        )

        # Value of every_n_train_steps is adjusted in main.py for DDP to ensure that roughly the same number of checkpoints are saved
        self.checkpoint_every_n_train_steps = config.checkpoint.every_n_train_steps

        self.dataset_path = config.dataset.path if config.dataset.name == "ncore" else None

        # TODO: Clean this up:
        # These are needed by the difix sampler, sanity_dataset, serializer, and metadata
        self.cached_config = config

        # These are needed by the difix sampler
        self.train_batch_size = config.datamodule.train_batch_size
        self.dataset_name = config.dataset.name

    @abstractmethod
    def forward(self, batch: DataAndRenderingBatch) -> Any: ...

    def validation_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        pass

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._validation_metrics_collector.reset()

    def on_validation_end(self):
        super().on_validation_end()
        self._validation_metrics_collector.write_metrics_to_yaml(self.val_dir)

    def test_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        pass

    @torch.amp.autocast("cuda")
    def precompute_extra_signals(
        self,
        mode: Literal["val", "test"],
        forward_closure: Callable,
        map_location: torch.device,
        sample_every_n_frames: int,
        n_rays_per_frame: int,
    ) -> list[ExtraSignal]:
        """
        Precompute a subset of the extra signals to be used in, e.g., the visualization.
        This is usually called in on_validation_start/on_test_start.
        """

        class EveryNSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, N: int):
                """
                Args:
                    data_source (Dataset): The dataset to sample from.
                    N (int): The interval for sampling every N-th item.
                """
                self.data_source = data_source
                self.N = N

            def __iter__(self):
                # Create an iterator that yields every N-th index
                return iter(range(0, len(self.data_source), self.N))

            def __len__(self):
                # The number of samples that will be drawn
                return len(self.data_source) // self.N

        dataloader_sampler = EveryNSampler(self.datamodule.val_dataset, sample_every_n_frames)
        if mode == "val":
            dataloader = self.datamodule.val_dataloader(dataloader_sampler=dataloader_sampler)
        elif mode == "test":
            dataloader = self.datamodule.test_dataloader(dataloader_sampler=dataloader_sampler)
        else:
            raise ValueError(f"Invalid mode {mode}")

        extra_signals: list[ExtraSignal] = []
        for batch in tqdm(dataloader, desc="Precomputing extra signals"):
            if batch.rays_cam is None:
                continue

            batch_skip = batch.rays_cam.size(0) // n_rays_per_frame
            batch = batch.to(self.device)[::batch_skip]

            results = forward_closure(batch)
            rendered: GaussiansRenderReturn = unpack_optional(results.rendered_cam)
            assert isinstance(rendered, GaussiansRenderReturn)
            extra_signals.append(unpack_optional(rendered.extra_ray_signals).to(map_location))

        return extra_signals

    def on_train_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        super().on_train_batch_start(batch, batch_local_idx)

    def populate_artifact_cache(self, checkpoint: Checkpoint) -> None:
        """Populate artifact cache fields based on available and selected entries"""

        # Torch checkpoint
        if self.artifact_cache.artifact_config.checkpoint.enabled:
            self.serialize_artifact_checkpoint(checkpoint)

        # Parsed config
        if self.artifact_cache.artifact_config.parsed_config.enabled and self.artifact_cache.parsed_config is None:
            self.artifact_cache.parsed_config = NamedSerialized.from_config(self.cached_config.to_dictconfig())

        datasource = self.datamodule.get_datasource()

        # If available, compute global timestamp offset as the global start timestamp of _all_ rig trajectories
        # of the datasource (required for USD timestamp offsetting to maintain floating point precision).
        # Use zero timestamp offset if datasource is not providing rig-trajectories
        usd_timestamp_offset_us = (
            rig_trajectories_time_range(datasource.get_rig_trajectories()).start
            if isinstance(datasource, RigTrajectoriesProvider)
            else 0
        )

        # Datasource summary
        if self.artifact_cache.datasource_summary is None:
            self.artifact_cache.datasource_summary = NamedSerialized(
                filename="datasource_summary.json",
                serialized=DataSourceSummary.from_datasource(datasource).to_json(),
            )

        # Certain fields are only supported if we use the NCOREDataSource
        if isinstance(datasource, NCOREDataSource) and self.dataset_path:
            if os.path.isfile(self.dataset_path):
                # Dataset info
                if self.artifact_cache.data_info is None:
                    with open(self.dataset_path, "r") as f:
                        self.artifact_cache.data_info = NamedSerialized(
                            filename="data_info.json", serialized=json.dumps(json.load(f), indent=4)
                        )

                # Metadata:
                # Do not cache the metadata, since we export the PSNR, which changes during training.
                metadata = get_metadata(self.cached_config, Path(self.dataset_path), self.training_step_outputs)
                self.artifact_cache.metadata = serialize_metadata(metadata)

            # Mesh
            # Warn if mesh_path is provided but mesh export is disabled
            generic_mesh_config = self.artifact_cache.artifact_config.mesh.generic
            if generic_mesh_config.mesh_path is not None and not generic_mesh_config.enabled:
                logger.warning("Mesh path provided but mesh.generic.enabled is False")

            # Do not recompute the mesh if it is already cached, because meshing is expensive.
            # Side effect: the mesh will be computed with unoptimized poses even if the calibration module is enabled.
            if self.artifact_cache.meshes is None:
                if generic_mesh_config.enabled:
                    mesh_path = generic_mesh_config.mesh_path
                    if mesh_path is not None:
                        try:
                            verts, faces = pcu.load_mesh_vf(mesh_path)
                            mesh = Mesh(vertices=verts, faces=faces)
                        except Exception as e:
                            logger.warning(f"Failed to load mesh from {mesh_path}: {e}")
                            # Fall back to point cloud generation
                            mesh_path = None

                    if mesh_path is None:
                        point_cloud = PointCloud.collate_fn(
                            [
                                pc
                                for pc in datasource.get_point_clouds(
                                    device=torch.device("cpu"),
                                    lidar_ids=generic_mesh_config.lidar_ids,
                                    camera_ids=generic_mesh_config.camera_ids,
                                    valid_points_only=False,  # BUG:5195915 WAR
                                    non_dynamic_points_only=True,
                                    color_type=None,
                                    step_frame=generic_mesh_config.step_frame,
                                )
                            ],
                            device=torch.device("cpu"),
                        )
                        max_num_points = generic_mesh_config.max_num_points
                        if max_num_points is not None and max_num_points > 0 and point_cloud.n_points > max_num_points:
                            random_idxs = np.random.choice(point_cloud.n_points, max_num_points, replace=False)
                            point_cloud = point_cloud[torch.from_numpy(random_idxs)]

                        nre_to_world = datasource.world_to_nre.inverse()
                        mesh = mesh_from_point_cloud(
                            point_cloud,
                            n_neighbors=generic_mesh_config.n_neighbors,
                            trim_distance=generic_mesh_config.trim_distance,
                            apply_road_segmentation=generic_mesh_config.apply_road_segmentation,
                            source_to_target=nre_to_world,
                        )

                    serialized_mesh = serialize_mesh(
                        mesh,
                        export_disjoint_meshes=generic_mesh_config.export_disjoint_meshes,
                        filename="mesh",
                        formats=generic_mesh_config.formats,
                    )

                    if self.artifact_cache.meshes is None:
                        self.artifact_cache.meshes = serialized_mesh
                    else:
                        self.artifact_cache.meshes.extend(serialized_mesh)

                    # Proxy meshes are extracted before smoothed mesh is added -- smoothed mesh is never a proxy for the NeRF.
                    self.artifact_cache.proxy_meshes = get_usd_references(serialized_mesh)

                    if generic_mesh_config.smooth:
                        rig_trajectories = datasource.get_rig_trajectories()
                        smoothed_mesh = smooth_mesh(mesh, rig_trajectories)
                        self.artifact_cache.meshes.extend(
                            serialize_mesh(
                                smoothed_mesh,
                                export_disjoint_meshes=generic_mesh_config.export_disjoint_meshes,
                                filename="mesh_smoothed",
                                formats=generic_mesh_config.formats,
                            )
                        )

                ground_mesh_config = self.artifact_cache.artifact_config.mesh.ground

                if ground_mesh_config.enabled:
                    point_cloud_generator = datasource.get_point_clouds(
                        device=torch.device("cpu"),
                        lidar_ids=ground_mesh_config.lidar_ids,
                        camera_ids=ground_mesh_config.camera_ids,
                        valid_points_only=False,  # BUG:5195915 WAR
                        non_dynamic_points_only=True,
                        color_type="camera-rgb" if ground_mesh_config.colored else None,
                        step_frame=ground_mesh_config.step_frame,
                    )

                    nre_to_world = datasource.world_to_nre.inverse()

                    # Reconstruct the ground mesh.
                    ground_vertices, ground_triangles, _, _, _, _, _, ground_vertex_colors = (
                        reconstruct_ground_mesh_from_points(
                            point_cloud_generator=point_cloud_generator,
                            min_ray_length=ground_mesh_config.min_ray_length,
                            ground_control_point=get_nominal_ground_point_under_lidar(datasource),
                            ground_compat_max_distance=ground_mesh_config.ground_compat_max_distance,
                            ground_compat_max_angle_deg=ground_mesh_config.ground_compat_max_angle_deg,
                            num_plane_hypotheses=ground_mesh_config.num_plane_hypotheses,
                            plane_max_distance=ground_mesh_config.plane_max_distance,
                            plane_max_angle_deg=ground_mesh_config.plane_max_angle_deg,
                            plane_min_eval_points=ground_mesh_config.plane_min_eval_points,
                            plane_max_eval_points=ground_mesh_config.plane_max_eval_points,
                            plane_min_inliers=ground_mesh_config.plane_min_inliers,
                            enable_plane_extension=ground_mesh_config.enable_plane_extension,
                            voxel_size=ground_mesh_config.voxel_size,
                            min_points_per_voxel=ground_mesh_config.min_points_per_voxel,
                            smoothing_passes=ground_mesh_config.smoothing_passes,
                            points_to_world_transf=nre_to_world,
                            verbose=ground_mesh_config.verbose,
                        )
                    )

                    # Add the ground mesh to the artifact cache.
                    serialized_ground_mesh = serialize_mesh(
                        mesh=Mesh(vertices=ground_vertices, faces=ground_triangles, colors=ground_vertex_colors),
                        export_disjoint_meshes=False,
                        filename="mesh_ground",
                        formats=ground_mesh_config.formats,
                    )

                    if self.artifact_cache.meshes is None:
                        self.artifact_cache.meshes = serialized_ground_mesh
                    else:
                        self.artifact_cache.meshes.extend(serialized_ground_mesh)

            # Sequence tracks
            # These tracks can be optimized during training, so update them in the cache, otherwise the tracks saved to
            # USDZ will not be the fully optimized ones, potentially causing actor pose discrepancies when rendering.
            if self.artifact_cache.artifact_config.sequence_tracks.enabled:
                logger.debug("populate_artifact_cache() triggered caching of cuboid tracks")
                self.artifact_cache.sequence_tracks = []

                # export the raw datasource tracks
                cuboid_tracks = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True)
                usd_excluded_tracks = datasource.cuboidtracks_dynamic

                # if available merge the updated tracks from the model with the datasource tracks
                if isinstance(self.model, CompositeModel):
                    model_cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
                        self.model.get_updated_cuboid_tracks(), datasource.world_to_nre.inverse(), None
                    )
                    model_cuboid_tracks.tracks_data.tracks_flags |= TrackFlags.CONTROLLABLE

                    # Compare tracks using canonical IDs to avoid mismatches between raw IDs like
                    # "<id>@<source>" and cleaned IDs like "<id>".
                    clean_track_id = DataSourceSummary._clean_track_id_str
                    datasource_tracks_by_clean_id: dict[str, list[str]] = {}
                    for track_id in cuboid_tracks.tracks_id:
                        datasource_tracks_by_clean_id.setdefault(clean_track_id(track_id), []).append(track_id)

                    model_tracks_by_clean_id: dict[str, list[str]] = {}
                    for track_id in model_cuboid_tracks.tracks_id:
                        model_tracks_by_clean_id.setdefault(clean_track_id(track_id), []).append(track_id)

                    tracks_in_both_clean_ids = set(datasource_tracks_by_clean_id) & set(model_tracks_by_clean_id)
                    tracks_in_both = [
                        track_id
                        for clean_id in tracks_in_both_clean_ids
                        for track_id in model_tracks_by_clean_id[clean_id]
                    ]
                    tracks_datasource_only = [
                        track_id
                        for clean_id, track_ids in datasource_tracks_by_clean_id.items()
                        if clean_id not in tracks_in_both_clean_ids
                        for track_id in track_ids
                    ]

                    # exclude the tracks that are not in the model from USD export
                    usd_excluded_tracks = CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, tracks_datasource_only)

                    if self.artifact_cache.artifact_config.sequence_tracks.controllable_only:
                        cuboid_tracks = model_cuboid_tracks
                    else:
                        cuboid_tracks = CuboidTracks.Ops.concatenate(
                            [
                                CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, tracks_datasource_only),
                                CuboidTracks.Ops.subset_from_tracks_id(model_cuboid_tracks, tracks_in_both),
                            ]
                        )

                # Alpasim depends on the sequence tracks being keyed with `dummy_chunk_id`
                self.artifact_cache.sequence_tracks += serialize_sequence_tracks(
                    sequence_id="dummy_chunk_id",
                    cuboid_tracks=cuboid_tracks,
                    excluded_tracks=usd_excluded_tracks,
                    usd_timestamp_offset_us=usd_timestamp_offset_us,
                )

                self.artifact_cache.usd_sequence_tracks = get_usd_references(self.artifact_cache.sequence_tracks)

        # nrend checkpoint
        if self.artifact_cache.artifact_config.nrend.enabled:
            assert self.trainer_config.world_size == 1, (
                "Generating nrend artifacts is currently not supported for distributed training"
            )
            self.artifact_cache.nrend_checkpoint = serialize_nrend_usd(
                self.model,
                datasource.aabb,
                datasource.get_offset(),
                self.artifact_cache.proxy_meshes,
                self.artifact_cache.usd_sequence_tracks,
                self.artifact_cache.artifact_config.nrend.use_legacy_model_dict,
            )

        # Rig trajectories
        if (
            self.artifact_cache.artifact_config.rig_trajectories.enabled
            and self.artifact_cache.rig_trajectories is None
            and isinstance(datasource, RigTrajectoriesProvider)
        ):
            # Need to export both start and end frame timestamps to enable rendering input views with rolling shutter,
            # otherwise there will be a time sync misalignment between rendered and input views.
            rig_trajectories = datasource.get_rig_trajectories()
            self.artifact_cache.rig_trajectories = serialize_rig_trajectories(
                rig_trajectories,
                usd_timestamp_offset=usd_timestamp_offset_us,
                add_default_cameras=self.artifact_cache.artifact_config.rig_trajectories.add_default_cameras,
            )

    def serialize_artifact_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Serialize the checkpoint for the artifact. Subclasses must override to set load_in_place and apply transforms."""
        raise NotImplementedError("Subclasses must implement serialize_artifact_checkpoint")

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)

        # NOTE: At this point, checkpoints should be in synchronize across all ranks !!!
        if self.artifact_cache.artifact_config.enabled:
            self._populate_and_save_artifacts_rank_zero(checkpoint)

        # Apply optional transformations to the training .ckpt file.
        # This must happen after artifact population since the artifact path makes its own copy.
        checkpoint_config = self.cached_config.checkpoint
        if checkpoint_config.strip_optimizer:
            for key in ("loops", "callbacks", "optimizer_states", "lr_schedulers", "MixedPrecisionPlugin"):
                checkpoint.pop(key, None)
        if checkpoint_config.fp16:
            checkpoint["state_dict"] = reduce_precision_to_fp16(checkpoint["state_dict"])

    @rank_zero_only
    def _populate_and_save_artifacts_rank_zero(self, checkpoint: Checkpoint) -> None:
        os.makedirs(artifact_dir := os.path.join(self.out_dir, self.run_id, "artifacts"), exist_ok=True)

        # Populate artifact cache
        self.populate_artifact_cache(checkpoint)

        # Write artifact for this step
        if (
            self.checkpoint_every_n_train_steps is not None
            and self.trainer.global_step % self.checkpoint_every_n_train_steps == 0
        ):
            self.artifact_cache.write_to_usdz(
                file_path=Path(os.path.join(artifact_dir, f"{self.last_checkpoint_step:06d}.usdz")),
            )

        # cleanup files according to save_top_k / save_last configuration
        if self.trainer.global_step == self.max_train_steps:
            # Write artifact as last - only on rank 0
            self.artifact_cache.write_to_usdz(
                file_path=Path(os.path.join(artifact_dir, "last.usdz")),
            )

    def log_and_inpaint_env_map(self, env_map: SkyEnvMapBackground) -> None:
        """
        Logs and potentially inpaints the textures used by SkyEnvMapBackground to Wandb/Tensorboard.
        The inpainting is just used at inference time and should be restored by calling env_map.restore_original_textures()
        before resuming training.
        """

        image_group = "sky_textures-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
        textures = env_map.flattened_textures().clamp(0, 1)

        self.log_image((textures * 255).byte().cpu().numpy(), image_group, self.global_step)

        inpaint_result = env_map.maybe_inpaint()
        if inpaint_result is not None:
            inpainted_textures, inpaint_mask = inpaint_result

            mask_image_group = "sky_textures-mask-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
            inpainted_image_group = (
                "sky_textures-inpainted-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
            )

            self.log_image((inpaint_mask * 255).byte().cpu().numpy(), mask_image_group, self.global_step)
            self.log_image(
                (inpainted_textures * 255).byte().cpu().numpy(),
                inpainted_image_group,
                self.global_step,
            )

    def get_track_orbit_views(
        self,
        track: CuboidTracks,
        distance: float,
        width: int,
        height: int,
        n_frames: int,
    ) -> Generator[DataAndRenderingBatch, None, None]:
        """
        Generates views that orbit around a specified track.
        Args:
            track: singleton cuboid track
            distance: orbit distance from track
            width: render width
            height: render height
            n_frames: number of views to render
        Returns: Batches for the orbit views
        """
        assert track.n_tracks == 1, "expected single track id"
        # Derive a fov that roughly captures the object and some background
        fov = 3 * np.rad2deg(np.arctan2(0.5 * track.cuboids_dims.max().item(), distance))

        camera_model, camera_params, target_pixels = self._get_pinhole_camera(width, height, fov)
        timestamp_us = int(track.tracks_timestamps_us[track.tracks_poses.shape[0] // 2])

        data_batch = DataBatch(
            camera=DataBatch.Camera(
                meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)],
                labels=CameraFrameLabels(),  # empty labels
            ),
        )

        track_pose = track.tracks_poses[track.tracks_poses.shape[0] // 2].matrix()
        for degree in np.arange(0, 360, 360 / n_frames):
            c2w_object = torch.eye(4, device=self.device)
            rotation = Rotation.from_euler("xz", [-90, degree], degrees=True)
            c2w_object[:3, :3] = to_torch(rotation.as_matrix(), device=self.device)
            c2w_object[:3, 3] = to_torch(rotation.apply(np.array([0, 0, -distance])), device=self.device)

            c2w_nre = track_pose @ c2w_object

            # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
            timestamps_gpu = torch.tensor([timestamp_us, timestamp_us], dtype=torch.int64, device=self.device).reshape(
                1, 2
            )
            timestamps_cpu = torch.tensor([timestamp_us, timestamp_us], dtype=torch.int64, device="cpu").reshape(1, 2)
            rendering_batch = RenderingBatch(
                camera=RenderingData(
                    rays=camera_model.pixels_to_world_rays_static_pose(target_pixels, c2w_nre).world_rays.reshape(
                        1, height, width, 6
                    ),
                    sensor_model_parameters=[camera_params],
                    poses_tquat_startend=se3_matrix_to_tquat(c2w_nre).unsqueeze(0).expand(2, -1).reshape(1, 2, 7),
                    timestamps_startend_us=timestamps_gpu,  # on GPU
                    rays_timestamps_us=torch.full(
                        (target_pixels.shape[0],), timestamp_us, dtype=torch.int64, device=self.device
                    ).reshape(1, height, width, 1),
                    timestamps_startend_us_cpu=timestamps_cpu,  # on CPU
                )
            )
            yield DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)

    def _get_pinhole_camera_parameters(
        self, image_width: int, image_height: int, fov: float
    ) -> OpenCVPinholeCameraModelParameters:
        """
        Generates pinhole camera parameters with the desired width, height, and fov
        """
        image_dim = np.array([image_width, image_height], dtype=np.uint64)
        target_principal_point = image_dim.astype(np.float32) / 2
        target_focal_length = np.min(image_dim.astype(np.float32)) / (2.0 * np.tan(np.deg2rad(fov) * 0.5))
        camera_params = OpenCVPinholeCameraModelParameters(
            resolution=image_dim,
            shutter_type=ShutterType.GLOBAL,
            focal_length=np.array([target_focal_length, target_focal_length], dtype=np.float32),
            principal_point=target_principal_point,
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        return camera_params

    def _get_pinhole_camera(
        self, image_width: int, image_height: int, fov: float
    ) -> Tuple[OpenCVPinholeCameraModel, OpenCVPinholeCameraModelParameters, torch.Tensor]:
        """
        Generates a pinhole camera model with the desired width, height, and fov
        """
        camera_params = self._get_pinhole_camera_parameters(image_width, image_height, fov)
        camera_model = OpenCVPinholeCameraModel(camera_params, device=self.device)

        pixels_x = torch.arange(image_width, dtype=torch.int16, device=self.device)
        pixels_y = torch.arange(image_height, dtype=torch.int16, device=self.device)
        grid_x, grid_y = torch.meshgrid(pixels_x, pixels_y, indexing="xy")
        target_pixels = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        return camera_model, camera_params, target_pixels

    def apply_alpha(self, batch: DataAndRenderingBatch, out: GaussiansCompositeReturn) -> None:
        """Perform alpha-compositing if alpha labels are given - updates batch in-place!"""

        if (
            out.rendered_cam is None
            or out.rendered_cam.extra_ray_signals is None
            or out.rendered_cam.extra_ray_signals.rgb_background is None
            or batch.data.camera is None
            or batch.data.camera.labels.alpha is None
        ):
            return

        batch.data.camera.labels.rgb = unpack_optional(batch.data.camera.labels.rgb) * batch.data.camera.labels.alpha[
            ..., None
        ] + out.rendered_cam.extra_ray_signals.rgb_background * (1 - batch.data.camera.labels.alpha[..., None])

    def collect_metric(
        self,
        name: str,
        value: Any,
        is_lidar: bool = False,
        frame_meta: FrameMeta | None = None,
        timestamps_startend_us: torch.Tensor | None = None,
        sequence_id: Union[list[str], str, None] = None,
        reduce_fx: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Overrides the log method to retain some metrics for later saving
        """

        unique_sensor_id = None
        is_lidar = False

        if frame_meta is not None:
            unique_sensor_idx = frame_meta.unique_sensor_idx
            unique_sensor_id = (
                self.datamodule.get_datasource().get_camera_sensor_ids(unique_sensors=True)[unique_sensor_idx]
                if not is_lidar
                else self.datamodule.get_datasource().get_lidar_sensor_ids(unique_sensors=True)[unique_sensor_idx]
            )

        self._validation_metrics_collector.save_metric(
            create_metric_sample(name, value, frame_meta, timestamps_startend_us),
            unique_sensor_id=unique_sensor_id,
            sequence_id=sequence_id,
            reduce_fx=reduce_fx,
            is_lidar=is_lidar,
        )

        # We can't `self.log()` inside `on_validation_end`. HINT: Can still log directly to
        # the logger by using `self.logger.experiment`.
        try:
            self.log(name, value, prog_bar=False)
        except MisconfigurationException as e:
            logger.warning(f"Failed to log metric {name}: {e}")
        except Exception as e:
            raise e
