# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import math

from typing import Literal, Optional, Tuple

import torch

from torch import nn
from tqdm import tqdm

from libs.gaussian_mcmc.interface import gaussian_mcmc  # type: ignore
from libs.slang_gaussians.interface import mcmc as slang_mcmc  # type: ignore
from nre.config.model import MCMCStrategyConfig
from nre.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nre.models.gaussians.strategies.base import BaseGaussianStrategy
from nre.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nre.models.nn_extensions import TypedModuleDict
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.misc import _multinomial_sample, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.torch_compile import TorchCompile
from nre.utils.trainer import TrainerConfig, adjust_gaussian_count_for_world_size, adjust_step_for_world_size


log = logging.getLogger(__name__)


class MCMCStrategy(BaseGaussianStrategy):
    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.

    """

    config: MCMCStrategyConfig

    def __init__(
        self,
        config: MCMCStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        super().__init__(config, trainer_config, init_from_datasource, gaussians_nodes)

        self.config = config.model_copy(deep=True)
        # Guards against duplicate visibility counter updates for the same results object
        self._last_visibility_update_id: Optional[int] = None
        self.invisible_steps = nn.ParameterDict()
        # Initialize the invisible steps buffer for each layer to ensure model loading works correctly
        for layer_id in gaussians_nodes.keys():
            self.invisible_steps[layer_id] = nn.UninitializedParameter(requires_grad=False)

        self.config.relocate.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.start_iteration,
        )
        self.config.relocate.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.end_iteration,
        )
        self.config.relocate.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.frequency,
        )
        if self.config.relocate.max_invisible_steps is not None:
            self.config.relocate.max_invisible_steps = adjust_step_for_world_size(
                trainer_config, int(self.config.relocate.max_invisible_steps)
            )
        else:
            self.config.relocate.max_invisible_steps = None
        log.info(f"MCMCStrategy/relocate:")
        log.info(f"    |─start_iteration={self.config.relocate.start_iteration}")
        log.info(f"    |─end_iteration={self.config.relocate.end_iteration}")
        log.info(f"    |─frequency={self.config.relocate.frequency}")
        if self.config.relocate.max_invisible_steps is not None:
            log.info(f"    └─max_invisible_steps={self.config.relocate.max_invisible_steps}")
        else:
            log.info(f"    └─max_invisible_steps=disabled")

        self.config.add.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.add.start_iteration,
        )
        self.config.add.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.add.end_iteration,
        )
        self.config.add.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.add.frequency,
        )
        self._max_n_gaussians = self.config.add.max_n_gaussians
        self.config.add.max_n_gaussians = adjust_gaussian_count_for_world_size(
            trainer_config,
            self.config.add.max_n_gaussians,
        )
        log.info(f"MCMCStrategy/add:")
        log.info(f"    |─start_iteration={self.config.add.start_iteration}")
        log.info(f"    |─end_iteration={self.config.add.end_iteration}")
        log.info(f"    |─frequency={self.config.add.frequency}")
        log.info(f"    └─max_n_gaussians={self.config.add.max_n_gaussians}")

        self.config.perturb.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.start_iteration,
        )
        self.config.perturb.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.end_iteration,
        )
        self.config.perturb.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.frequency,
        )

        log.info(f"MCMCStrategy/perturb:")
        log.info(f"    |─start_iteration={self.config.perturb.start_iteration}")
        log.info(f"    |─end_iteration={self.config.perturb.end_iteration}")
        log.info(f"    |─frequency={self.config.perturb.frequency}")
        log.info(f"    └─noise_lr={self.config.perturb.noise_lr.default}")

        # Precompute the look up table for binomial coefficients (Eq 9 in the MCMC paper)
        self.binoms = nn.Buffer(
            torch.FloatTensor(
                [
                    [math.comb(n, k) if k <= n else 0 for k in range(self.config.binom_n_max)]
                    for n in range(self.config.binom_n_max)
                ],
            ),
            persistent=False,
        )

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            max_n_gaussians = torch.distributed.nn.functional.all_reduce(
                torch.tensor(self.config.add.max_n_gaussians, dtype=torch.int32, device="cuda"),
                op=torch.distributed.ReduceOp.SUM,
            ).item()
            assert max_n_gaussians == self._max_n_gaussians, (
                f"max_n_gaussians={max_n_gaussians} != {self._max_n_gaussians}"
            )

        # Store perturb_noise_lr on CPU to avoid cudaStreamSynchronize when calling .item()
        self.perturb_noise_lr = {
            layer_id: self.config.perturb.noise_lr.layers.get(layer_id, self.config.perturb.noise_lr.default)
            for layer_id in gaussians_nodes.keys()
        }

        if self._should_track_visibility():
            for layer_id, gaussian_model in gaussians_nodes.items():
                if isinstance(gaussian_model.positions, nn.UninitializedParameter):
                    continue
                num_gaussians = gaussian_model.get_num_gaussians()
                self._ensure_invisible_steps_buffer(layer_id, num_gaussians, gaussian_model.device)

    def _should_track_visibility(self) -> bool:
        return self.config.relocate.max_invisible_steps is not None and self.config.relocate.max_invisible_steps > 0

    def _ensure_invisible_steps_buffer(self, layer_id: str, num_gaussians: int, device: torch.device) -> None:
        if layer_id not in self.invisible_steps:
            self.invisible_steps[layer_id] = nn.UninitializedParameter(requires_grad=False)
        buffer = self.invisible_steps[layer_id]
        needs_init = isinstance(buffer, nn.UninitializedParameter) or buffer.numel() != num_gaussians
        if needs_init or buffer.device != device:
            self.invisible_steps[layer_id] = nn.Parameter(
                torch.zeros((num_gaussians,), dtype=torch.int32, device=device),
                requires_grad=False,
            )

    @torch.no_grad()
    def update_visibility_counters(
        self,
        *,
        results,
        visibility: torch.Tensor,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        if not self._should_track_visibility():
            return
        if visibility.numel() == 0:
            return
        results_id = id(results)
        if self._last_visibility_update_id == results_id:
            return
        self._last_visibility_update_id = results_id

        offset = 0
        for layer_id, gaussian_model in gaussians_nodes.items():
            if isinstance(gaussian_model.positions, nn.UninitializedParameter):
                continue
            count = gaussian_model.get_num_gaussians()
            if count == 0:
                continue
            layer_visibility = visibility[offset : offset + count]
            if layer_visibility.numel() != count:
                log.warning(
                    "MCMCStrategy/visibility_counters: visibility length mismatch for layer '%s' (%d != %d)",
                    layer_id,
                    layer_visibility.numel(),
                    count,
                )
                return
            self._ensure_invisible_steps_buffer(layer_id, count, gaussians_nodes[layer_id].device)
            counters = self.invisible_steps[layer_id]
            seen_mask = layer_visibility > 0
            counters.add_(1)
            counters.masked_fill_(seen_mask, 0)
            counters.clamp_(max=self.config.relocate.max_invisible_steps)
            offset += count
        if offset != visibility.numel():
            log.warning(
                "MCMCStrategy/visibility_counters: visibility length mismatch total (%d != %d)",
                offset,
                visibility.numel(),
            )

    def update_step_train_batch_end(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None:
        if self._check_step_condition(
            step=global_step,
            start=self.config.relocate.start_iteration,
            end=self.config.relocate.end_iteration,
            freq=self.config.relocate.frequency,
        ):
            # Relocates dead gaussians close to sampled alive Gaussians.
            # Motivation: Use the Gaussian budget in areas where it makes an impact.
            self.relocate_gaussians(gaussians_nodes)

        if self._check_step_condition(
            step=global_step,
            start=self.config.add.start_iteration,
            end=self.config.add.end_iteration,
            freq=self.config.add.frequency,
        ):
            # Adds new Gaussians until reaching the specified maximum number of Gaussians.
            self.add_new_gaussians(gaussians_nodes)

        if self._check_step_condition(
            step=global_step,
            start=self.config.perturb.start_iteration,
            end=self.config.perturb.end_iteration,
            freq=self.config.perturb.frequency,
        ):
            # Applies random perturbation to Gaussians per Gaussian layer.
            for layer_id, gaussian_model in gaussians_nodes.items():
                if layer_id in self.config.exclude_layer_ids:
                    continue
                with ScopedTimer(f"MCMCStrategy/perturb_gaussians.{layer_id}"):
                    self.perturb_gaussians(layer_id, gaussian_model, enable_torch_compile=True)

    @ScopedTimer("MCMCStrategy/relocate_gaussians")
    @torch.no_grad()
    def relocate_gaussians(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        # Get the per Gaussian densities and scales (after sigmoid)
        selected_gaussian_nodes = SelectedGaussianNodes(gaussians_nodes, self.config.exclude_layer_ids)
        densities = selected_gaussian_nodes.get_densities().view(-1)  # Ensure 1D
        # Find the dead indices
        dead_mask = (densities <= self.config.opacity_threshold).view(-1)  # Ensure 1D to prevent broadcasting
        if self._should_track_visibility() and densities.numel() > 0:
            invisible_steps_list: list[torch.Tensor] = []
            for layer_id in selected_gaussian_nodes.layer_ids:
                num_gaussians = selected_gaussian_nodes.layer_num_gaussians[layer_id]
                self._ensure_invisible_steps_buffer(
                    layer_id, num_gaussians, selected_gaussian_nodes.gaussians_nodes[layer_id].device
                )
                invisible_steps_list.append(self.invisible_steps[layer_id])
            if invisible_steps_list:
                invisible_steps_tensor = torch.cat(invisible_steps_list).view(
                    -1
                )  # Ensure 1D to prevent broadcasting issues
                if invisible_steps_tensor.numel() == densities.numel():
                    invisible_dead_mask = invisible_steps_tensor >= self.config.relocate.max_invisible_steps  # type: ignore
                    dead_mask = torch.logical_or(dead_mask, invisible_dead_mask)
                else:
                    log.warning(
                        "MCMCStrategy/relocate_gaussians: invisible steps size mismatch (%d != %d)",
                        invisible_steps_tensor.numel(),
                        densities.numel(),
                    )
        dead_idxs = torch.where(dead_mask)[0]
        alive_idxs = torch.where(~dead_mask)[0]
        n_dead_gaussians = len(dead_idxs)

        if n_dead_gaussians and len(alive_idxs) > 0:
            scales = selected_gaussian_nodes.get_scales()

            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(
                densities, scales, n_dead_gaussians, alive_idxs
            )

            # Save the optimizer states across all layers as we need them when relocating the dead gaussians
            optimizer_states = selected_gaussian_nodes.get_optimizer_states()

            for layer_id in selected_gaussian_nodes.layer_ids:
                layer_sampled_idxs, layer_new_densities, layer_new_scales, relocating_to_layer_idxs = (
                    selected_gaussian_nodes.get_layer_sampled_gaussians(
                        layer_id, sampled_idxs, new_densities, new_scales, self.config.opacity_threshold, dead_idxs
                    )
                )

                layer_keep_idxs = selected_gaussian_nodes.get_layer_keep_mask(layer_id, dead_idxs)

                def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                    # adjust the densities and scales for the sampled indices in the current layer
                    if name == "densities":
                        # Handle case where param has shape (N, 1) vs (N,) by matching target shape
                        target_shape = param[layer_sampled_idxs].shape
                        param[layer_sampled_idxs] = layer_new_densities.view(target_shape)
                    elif name == "scales":
                        param[layer_sampled_idxs] = layer_new_scales

                    # the current layer should now be the union of the kept and sampled indices
                    is_parameter = isinstance(param, torch.nn.Parameter)
                    param = torch.cat([param[layer_keep_idxs], param[layer_sampled_idxs]])
                    return torch.nn.Parameter(param) if is_parameter else param

                def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
                    # Set the optimizer values to zero for the sampled indices and use the saved optimizer states
                    # for the gaussians being relocated to the layer (sec. 3.4 of MCMC paper)
                    v[layer_sampled_idxs] = 0
                    relocating_to_layer_vals = optimizer_states[name][key][[unpack_optional(relocating_to_layer_idxs)]]
                    # Handle the case where some layers have parameters with shape [N, C, D] and others [N, D]
                    # (ie: when some layers use fourier_features_dim and others don't) by averaging over the channel dimension
                    if len(v.shape) == len(relocating_to_layer_vals.shape) - 1:
                        relocating_to_layer_vals = relocating_to_layer_vals.mean(dim=-2)
                    # Handle the case when some layers use different fourier_features_dims
                    # E.g., traffic_light layer has fourier_features_dim=5 with [N, 5, 3], while dynamic_rigids layer has fourier_features_dim=20 with [N, 20, 3],
                    # and if we're going to relocate a gaussian from dynamic_rigids to traffic_light, we will just sample the first 5 features from dynamic_rigids.
                    elif len(v.shape) == len(relocating_to_layer_vals.shape):
                        if len(v.shape) > 2 and v.shape[-2] != relocating_to_layer_vals.shape[-2]:
                            assert relocating_to_layer_vals.shape[-2] % v.shape[-2] == 0
                            relocating_to_layer_vals = relocating_to_layer_vals[..., : v.shape[-2], :]
                    return torch.cat([v[layer_keep_idxs], relocating_to_layer_vals])

                self._update_param_with_optimizer(gaussians_nodes[layer_id], update_param_fn, update_optimizer_fn)
                if self._should_track_visibility():
                    self._ensure_invisible_steps_buffer(
                        layer_id,
                        selected_gaussian_nodes.layer_num_gaussians[layer_id],
                        gaussians_nodes[layer_id].device,
                    )
                    counters = self.invisible_steps[layer_id]
                    if counters.numel() != layer_keep_idxs.numel():
                        counters = torch.zeros((layer_keep_idxs.numel(),), dtype=counters.dtype, device=counters.device)
                    new_counters = torch.cat(
                        [
                            counters[layer_keep_idxs].view(-1),  # Ensure 1D
                            torch.zeros((layer_sampled_idxs.shape[0],), dtype=counters.dtype, device=counters.device),
                        ]
                    )
                    self.invisible_steps[layer_id] = nn.Parameter(new_counters, requires_grad=False)

            if self.config.print_stats:
                tqdm.write(f"Relocated {n_dead_gaussians} ({n_dead_gaussians / len(densities) * 100:.2f}%) gaussians")
                for layer_id, layer_prev_num_gaussians in selected_gaussian_nodes.layer_num_gaussians.items():
                    tqdm.write(
                        f"{layer_id}: old count: {layer_prev_num_gaussians}, new count: {gaussians_nodes[layer_id].get_num_gaussians()} "
                        f"({(gaussians_nodes[layer_id].get_num_gaussians() - layer_prev_num_gaussians) / max(1, layer_prev_num_gaussians) * 100:.2f}%)"
                    )

    @ScopedTimer("MCMCStrategy/add_new_gaussians")
    @torch.no_grad()
    def add_new_gaussians(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        selected_gaussian_nodes = SelectedGaussianNodes(gaussians_nodes, self.config.exclude_layer_ids)

        # Get the current number of gaussians
        current_num_gaussians = sum(selected_gaussian_nodes.layer_num_gaussians.values())
        target_num_gaussians = min(self.config.add.max_n_gaussians, int(1.05 * current_num_gaussians))
        num_gaussians_to_add = max(0, target_num_gaussians - current_num_gaussians)
        if num_gaussians_to_add:
            densities = selected_gaussian_nodes.get_densities()
            scales = selected_gaussian_nodes.get_scales()

            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(densities, scales, num_gaussians_to_add)

            for layer_id in selected_gaussian_nodes.layer_ids:
                layer_sampled_idxs, layer_new_densities, layer_new_scales, _ = (
                    selected_gaussian_nodes.get_layer_sampled_gaussians(
                        layer_id, sampled_idxs, new_densities, new_scales, self.config.opacity_threshold, None
                    )
                )

                layer_num_gaussians_to_add = layer_sampled_idxs.shape[0]

                def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                    # adjust the densities and scales for the sampled indices in the current layer
                    if name == "densities":
                        # Handle case where param has shape (N, 1) vs (N,) by matching target shape
                        target_shape = param[layer_sampled_idxs].shape
                        param[layer_sampled_idxs] = layer_new_densities.view(target_shape)
                    elif name == "scales":
                        param[layer_sampled_idxs] = layer_new_scales

                    is_parameter = isinstance(param, torch.nn.Parameter)
                    param = torch.cat([param, param[layer_sampled_idxs]])
                    return torch.nn.Parameter(param) if is_parameter else param

                def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
                    v_new = torch.zeros((len(layer_sampled_idxs), *v.shape[1:]), device=v.device)
                    return torch.cat([v, v_new])

                self._update_param_with_optimizer(gaussians_nodes[layer_id], update_param_fn, update_optimizer_fn)
                if self._should_track_visibility() and layer_num_gaussians_to_add > 0:
                    self._ensure_invisible_steps_buffer(
                        layer_id,
                        selected_gaussian_nodes.layer_num_gaussians[layer_id],
                        gaussians_nodes[layer_id].device,
                    )
                    counters = self.invisible_steps[layer_id].view(-1)  # Ensure 1D
                    new_counters = torch.cat(
                        [
                            counters,
                            torch.zeros((layer_num_gaussians_to_add,), dtype=counters.dtype, device=counters.device),
                        ]
                    )
                    self.invisible_steps[layer_id] = nn.Parameter(new_counters, requires_grad=False)

                if self.config.print_stats:
                    tqdm.write(
                        f"{layer_id}: Added {layer_num_gaussians_to_add} "
                        f"({layer_num_gaussians_to_add / max(1, selected_gaussian_nodes.layer_num_gaussians[layer_id]) * 100:.2f}%) gaussians"
                    )

    @ScopedTimer("MCMCStrategy/perturb_gaussians")
    @torch.no_grad()
    def perturb_gaussians(
        self, layer_id: str, gaussian_model: BaseGaussianModel, enable_torch_compile: bool = False
    ) -> None:
        """
        Uses Slang implementation when available for improved performance by combining the following operations:
        - quat_scale_to_covariance
        - compute_perturb_gaussians (sigmoid, noise scaling, covariance multiplication)
        - add_noise to positions
        """
        quaternion_format: Literal["xyzw", "wxyz"] = "wxyz"

        current_lr = 0.0
        for opp in gaussian_model.optimizers:
            for param_group in opp["optimizer"].param_groups:
                if "name" in param_group and param_group["name"].split(".")[-1] == "positions":
                    current_lr = param_group["lr"]
                    break

        assert current_lr > 0.0, "Current learning rate is not set"
        assert self.perturb_noise_lr is not None, "Perturb noise learning rate is not set"

        # Compute effective learning rate (perturb_noise_lr is already a float, no .item() needed)
        effective_lr = current_lr * self.perturb_noise_lr[layer_id]

        # Use Slang implementation when available
        if slang_mcmc is not None:
            # Get raw (pre-activation) gaussian parameters for fused kernel
            # This avoids separate PyTorch calls for normalize, exp, sigmoid
            positions = gaussian_model.positions
            quats = gaussian_model.rotations
            scales = gaussian_model.scales
            densities = gaussian_model.densities.squeeze(-1)

            # Check if this is a RigidGaussianModel with cuboid bounds constraint
            use_cuboid_constraint = (not self.config.perturb.move_outside_of_cuboid) and isinstance(
                gaussian_model, RigidGaussianModel
            )

            if use_cuboid_constraint:
                assert isinstance(gaussian_model, RigidGaussianModel)
                cuboid_dims = gaussian_model.cuboid_tracks.cuboids_dims
                gaussian_cuboid_ids = gaussian_model.gaussian_cuboid_ids
                gaussian_cuboid_dims = cuboid_dims[gaussian_cuboid_ids].contiguous()

                slang_mcmc.fused_perturb_gaussians_rigid(
                    positions=positions,
                    quats=quats,
                    scales=scales,
                    densities=densities,
                    cuboid_dims=gaussian_cuboid_dims,
                    current_lr=effective_lr,
                    quaternion_format=quaternion_format,
                )
            else:
                slang_mcmc.fused_perturb_gaussians(
                    positions=positions,
                    quats=quats,
                    scales=scales,
                    densities=densities,
                    current_lr=effective_lr,
                    quaternion_format=quaternion_format,
                )
        else:
            # Fallback to original PyTorch implementation if slang is not available
            self._perturb_gaussians_pytorch(
                layer_id, gaussian_model, effective_lr, quaternion_format, enable_torch_compile
            )

    def _perturb_gaussians_pytorch(
        self,
        layer_id: str,
        gaussian_model: BaseGaussianModel,
        effective_lr: float,
        quaternion_format: Literal["xyzw", "wxyz"],
        enable_torch_compile: bool,
    ) -> None:
        """Fallback PyTorch implementation of perturb_gaussians."""
        current_lr_tensor = torch.tensor(effective_lr).to(gaussian_model.positions.device, non_blocking=True)

        with ScopedTimer("MCMCStrategy/perturb_gaussians/quat_scale_to_covariance"):
            quats = gaussian_model.get_rotations(quaternion_format=quaternion_format)
            scales = gaussian_model.get_scales()
            covariance = gaussian_mcmc.quat_scale_to_covariance(quats, scales, quaternion_format)

        noise = self._compute_perturb_gaussians_pytorch(
            gaussian_model, covariance, current_lr_tensor, enable_torch_compile
        )

        with ScopedTimer("MCMCStrategy/perturb_gaussians/add_noise"):
            gaussian_model.positions.add_(noise)

    @ScopedTimer("MCMCStrategy/compute_perturb_gaussians")
    @TorchCompile.conditional(fullgraph=True, dynamic=True)
    def _compute_perturb_gaussians_pytorch(
        self, gaussian_model: BaseGaussianModel, covariance: torch.Tensor, current_lr: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch implementation of compute_perturb_gaussians for fallback."""
        positions = gaussian_model.get_positions()
        densities = gaussian_model.get_densities()

        def op_sigmoid(x: torch.Tensor, k: int = 100, x0: float = 0.995) -> torch.Tensor:
            return 1 / (1 + torch.exp(-k * (x - x0)))

        # Current positional learning rate multiplied by the config parameter scale
        noise = torch.randn_like(positions) * (op_sigmoid(1 - densities)) * current_lr
        noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)

        # Avoid conflict with the out of bounds loss:
        # - Disallow moving inside the cuboid tracks to outside the cuboid tracks
        # - Disallow moving outside the cuboid tracks to further outside the cuboid tracks
        if (not self.config.perturb.move_outside_of_cuboid) and isinstance(gaussian_model, RigidGaussianModel):
            cuboid_dims = gaussian_model.cuboid_tracks.cuboids_dims
            gaussian_cuboid_ids = gaussian_model.gaussian_cuboid_ids
            gaussian_cuboid_dims = cuboid_dims[gaussian_cuboid_ids]

            bounds = gaussian_cuboid_dims / 2
            cur_pos_dist = positions.abs() - bounds
            pos_with_noise_dist = (positions + noise).abs() - bounds

            inside_to_outside_mask = torch.logical_and(
                cur_pos_dist < 0,
                pos_with_noise_dist >= 0,
            )
            outside_to_more_outside_mask = torch.logical_and(
                cur_pos_dist >= 0,
                pos_with_noise_dist > cur_pos_dist,
            )
            noise = torch.where(torch.logical_or(inside_to_outside_mask, outside_to_more_outside_mask), 0, noise)

        return noise

    @ScopedTimer("MCMCStrategy/sample_new_gaussians")
    def sample_new_gaussians(
        self,
        densities: torch.Tensor,
        scales: torch.Tensor,
        num_gaussians: int,
        valid_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if valid_indices is None:
            valid_indices = torch.arange(0, int(densities.shape[0]), device=densities.device, dtype=torch.int32)

        probabilities = densities[valid_indices].flatten()  # ensure its shape is [N,]

        # Sample the locations to which the dead Gaussians will be moved proportional to the opacity of the alive Gaussians
        with ScopedTimer("MCMCStrategy/sample_new_gaussians/multinomial_sample"):
            sampled_idxs = _multinomial_sample(probabilities, num_gaussians, replacement=True)
            sampled_idxs = valid_indices[sampled_idxs]

        with ScopedTimer("MCMCStrategy/sample_new_gaussians/compute_ratios"):
            ratios = (torch.bincount(sampled_idxs)[sampled_idxs] + 1).clamp_(min=1, max=self.config.binom_n_max).int()

        with ScopedTimer("MCMCStrategy/sample_new_gaussians/compute_relocation_tensor"):
            new_densities, new_scales = gaussian_mcmc.compute_relocation_tensor(
                densities[sampled_idxs].contiguous(),
                scales[sampled_idxs].contiguous(),
                ratios.contiguous(),
                self.binoms,
                self.config.binom_n_max,
                self.config.opacity_threshold,
            )

        return sampled_idxs, new_densities, new_scales


BaseGaussianStrategy.register_to_gaussian_strategy_factory("mcmc", MCMCStrategy)
