# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
from typing import Optional, Tuple

import torch

from nre.models.gaussians.gaussians_model import BaseGaussianModel
from nre.models.nn_extensions import TypedModuleDict


class SelectedGaussianNodes:
    def __init__(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], exclude_layer_ids: list[str]):
        self.gaussians_nodes = gaussians_nodes
        self.layer_ids = sorted(
            [
                layer_id
                for layer_id in gaussians_nodes.keys()
                if layer_id not in exclude_layer_ids and gaussians_nodes[layer_id].get_num_gaussians() > 0
            ]
        )

        layer_offset = 0
        self.layer_offsets: dict[str, int] = {}
        self.layer_num_gaussians: dict[str, int] = {}
        for layer_id in self.layer_ids:
            self.layer_offsets[layer_id] = layer_offset
            self.layer_num_gaussians[layer_id] = self.gaussians_nodes[layer_id].get_num_gaussians()
            layer_offset += self.layer_num_gaussians[layer_id]

    def get_densities(self) -> torch.Tensor:
        # Return empty tensor if no layers to concatenate
        if not self.layer_ids:
            return torch.empty(0)
        densities = torch.cat([self.gaussians_nodes[layer_id].get_densities() for layer_id in self.layer_ids])
        return densities

    def get_scales(self) -> torch.Tensor:
        # Return empty tensor if no layers to concatenate
        if not self.layer_ids:
            return torch.empty(0)
        scales = torch.cat([self.gaussians_nodes[layer_id].get_scales() for layer_id in self.layer_ids])
        return scales

    def get_layer_sampled_gaussians(
        self,
        layer_id: str,
        sampled_idxs: torch.Tensor,
        new_densities: torch.Tensor,
        new_scales: torch.Tensor,
        opacity_threshold: float,
        dead_idxs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Given a set of newly sampled gaussians across all layers, return the indices, densities, and scales of the gaussians that are being added to the specified layer.
        If dead (global) indices are provided, also return the (global) indices of the dead gaussians that are being relocated to the specified layer.
        Args:
            layer_id: The layer id to get the sampled gaussian parameters for
            sampled_idxs: The global indices of the newly sampled gaussians (across all layers)
            new_densities: The post-activation densities of the newly sampled gaussians across all layers
            new_scales: The post-activation scales of the newly sampled gaussians across all layers
            opacity_threshold: Minimum density threshold for newly sampled gaussians
            dead_idxs: The global indices of the dead gaussians (across all layers)
        Returns:
            layer_sampled_idxs: The layer-specific indices of the newly sampled gaussians in the specified layer
            layer_new_densities: The pre-activation densities of the newly sampled gaussians in the specified layer
            layer_new_scales: The pre-activation scales of the newly sampled gaussians in the specified layer
            relocating_to_layer_idxs: If dead_idxs is not None, the global indices of the dead gaussians that are being relocated to the specified layer (None otherwise)
        """

        layer_sampled_mask = torch.logical_and(
            sampled_idxs >= self.layer_offsets[layer_id],
            sampled_idxs < self.layer_offsets[layer_id] + self.layer_num_gaussians[layer_id],
        )

        layer_sampled_idxs = sampled_idxs[layer_sampled_mask] - self.layer_offsets[layer_id]

        epsilon = torch.finfo(torch.float32).eps
        layer_new_densities = self.gaussians_nodes[layer_id].density_activation_inv(
            torch.clamp(
                new_densities[layer_sampled_mask],
                max=1.0 - epsilon,
                min=opacity_threshold if opacity_threshold > 0 else epsilon,
            )
        )
        layer_new_scales = self.gaussians_nodes[layer_id].scale_activation_inv(new_scales[layer_sampled_mask])
        return (
            layer_sampled_idxs,
            layer_new_densities,
            layer_new_scales,
            dead_idxs[layer_sampled_mask] if dead_idxs is not None else None,
        )

    def get_layer_keep_mask(self, layer_id: str, dead_idxs: torch.Tensor) -> torch.Tensor:
        """
        Given a set of dead gaussians across all layers, return the mask of the gaussians that are being still alive in the specified layer.
        Args:
            layer_id: The layer id to get the keep mask for
            dead_idxs: The global indices of the dead gaussians (across all layers)
        Returns:
            layer_keep_idxs: Mask of the gaussians that are being still alive in the specified layer
        """

        layer_dead_mask = torch.logical_and(
            dead_idxs >= self.layer_offsets[layer_id],
            dead_idxs < self.layer_offsets[layer_id] + self.layer_num_gaussians[layer_id],
        )
        layer_keep_idxs = torch.ones(
            self.layer_num_gaussians[layer_id], device=layer_dead_mask.device, dtype=torch.bool
        )
        layer_keep_idxs[dead_idxs[layer_dead_mask] - self.layer_offsets[layer_id]] = False
        return layer_keep_idxs

    def get_optimizer_states(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get the (concatenated) optimizer states across all layers (used when relocating dead gaussians as in Sec. 3.4 of the MCMC paper)
        This method assumes that the selected gaussian nodes all have the same optimizable parameters (i.e. positions, densities, scales),
        which is typically the case. It also generally assumes that the optimizer shapes are identical across layers
        (with the exception of the first dimension), although we handle the case where some parameters have shape [N, C, D] and others [N, D]
        by expanding the latter to [N, C, D] (this is the case for feature_albedo when some layers use fourier_features_dim and others don't).
        Return:
            optimizer_states: The optimizer states (concatenated across all layers)
        """

        optimizer_states_list: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(dict)
        for layer_id in self.layer_ids:
            for optim_sched in self.gaussians_nodes[layer_id].optimizers:
                optimizer_state_dict = optim_sched["optimizer"].state_dict()
                for param_group in optimizer_state_dict["param_groups"]:
                    if "name" in param_group:
                        name = param_group["name"].split(".")[-1]
                        assert len(param_group["params"]) == 1, "Only one parameter group is supported"
                        # This can happen when doing lidar-only supervision (features_albedo and features_specular
                        # will not have optimizer states)
                        if param_group["params"][0] not in optimizer_state_dict["state"]:
                            continue

                        for key, value in optimizer_state_dict["state"][param_group["params"][0]].items():
                            if key in optimizer_states_list[name]:
                                optimizer_states_list[name][key].append(value)
                            else:
                                optimizer_states_list[name][key] = [value]

        optimizer_states: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for param_name, param_values in optimizer_states_list.items():
            for key, values in param_values.items():
                expanded_values = []
                reference_shape_size = len(max(values, key=lambda value: len(value.shape)).shape)
                reference_shape = max(
                    values, key=lambda value: value.shape[-2] if len(value.shape) == reference_shape_size else 0
                ).shape
                for value in values:
                    if len(value.shape) == len(reference_shape):
                        # Handle the case for feature_albedo when some layers use different fourier_features_dim.
                        # E.g., traffic_light layer has fourier_features_dim=5 with [N, 5, 3], while dynamic_rigids layer has fourier_features_dim=20 with [N, 20, 3],
                        # We need to expand the traffic_light layer to [N, 20, 3] by repeating the feature_albedo values 4 times for global densification.
                        # We expect the number of fourier_features dim to be divisible by the reference (maximum) shape.
                        if len(value.shape) > 2 and value.shape[-2] != reference_shape[-2]:
                            assert reference_shape[-2] % value.shape[-2] == 0, (
                                f"Reference shape must be divisible by value shape for {param_name}.{key}: {reference_shape} % {value.shape} != 0"
                            )
                            repeat_size = reference_shape[-2] // value.shape[-2]
                            expanded_values.append(
                                value.unsqueeze(-3)
                                .expand(*[-1] * (len(value.shape) - 2), repeat_size, -1, -1)
                                .reshape(*value.shape[0:-2], reference_shape[-2], value.shape[-1])
                            )
                        else:
                            expanded_values.append(value)
                    elif len(value.shape) == len(reference_shape) - 1:
                        expanded_values.append(
                            value.unsqueeze(-2).expand(*[-1] * (len(value.shape) - 1), reference_shape[-2], -1)
                        )
                    else:
                        raise ValueError(
                            f"Inconsistent shapes for {param_name}.{key}: {value.shape} != {reference_shape}"
                        )
                optimizer_states[param_name][key] = torch.cat(expanded_values)

        return optimizer_states
