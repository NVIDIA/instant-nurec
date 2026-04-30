# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
import dataclasses

from typing import Any, Callable, Generator, Optional

import torch

from pytorch_lightning.plugins.precision.amp import MixedPrecision
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler

from nre.config.model import ModelConfig
from nre.models.base import BaseModel
from nre.utils.misc import dataclass_items
from nre.utils.profiling import ScopedTimer


def update_module_step(m: BaseModel, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
    with ScopedTimer(f"{m.__class__.__name__}/update_step_train_batch_start"):
        additional_parameters = m.update_step_train_batch_start(epoch, global_step, system, **kwargs)
    return additional_parameters


def detached(value: Any):
    """
    Returns a detached version of possibly nested tensors and dataclasses.
    """
    if value is None:
        return None
    elif isinstance(value, (int, float)):
        return value
    elif dataclasses.is_dataclass(type(value)):
        return dataclasses.replace(value, **{k: detached(v) for k, v in dataclass_items(value)})
    elif isinstance(value, torch.Tensor):
        return value.detach()
    elif isinstance(value, list):
        return [detached(v) for v in value]
    elif isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    else:
        return value


def with_grad_enabled(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with torch.enable_grad(), torch.inference_mode(False):
            return func(*args, **kwargs)

    return wrapper


def model_config_compatibility_check(config: ModelConfig) -> None:
    """
    Performs the compatibility of the modules specified in the config. If the modules are not compatible an assert is triggered.
    """
    if hasattr(config, "extra_signal") and config.background is not None:
        assert config.background.name != "sky_mlp", "You might be looking for extra_signal_sky_mlp"


def concat_rays_timestamps(
    rays_cam_timestamps_us: Optional[torch.Tensor] = None,  # (N_rays_cam, )
    rays_lidar_timestamps_us: Optional[torch.Tensor] = None,  # (N_rays_lidar, )
) -> torch.Tensor | None:
    rays_timestamps_us = []
    if rays_cam_timestamps_us is not None:
        rays_timestamps_us.append(rays_cam_timestamps_us)
    if rays_lidar_timestamps_us is not None:
        rays_timestamps_us.append(rays_lidar_timestamps_us)
    return None if len(rays_timestamps_us) == 0 else torch.cat(rays_timestamps_us, dim=0)


def concat_rays(
    rays_cam: Optional[torch.Tensor] = None,
    rays_lidar: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rays_cam is not None:
        rays_o, rays_d = rays_cam[:, 0:3].contiguous(), rays_cam[:, 3:6].contiguous()  # both (N_rays, 3)
        if rays_lidar is not None:
            rays_o = torch.cat([rays_o, rays_lidar[:, 0:3]], dim=0).contiguous()
            rays_d = torch.cat([rays_d, rays_lidar[:, 3:6]], dim=0).contiguous()
    else:
        assert rays_lidar is not None, "At least one of [rays_cam, rays_lidar] should be given"
        rays_o, rays_d = rays_lidar[:, 0:3].contiguous(), rays_lidar[:, 3:6].contiguous()  # both (N_rays, 3)
    return rays_o, rays_d


def eval_tcnn_network(network, input_tensor: torch.Tensor) -> torch.Tensor:
    """Evaluates TCNN network in safe way, circumventing issues with empty tensor inputs
    (internal CUTLASS fails to evaluate these, e.g., in non-fused modes)"""
    if not input_tensor.numel():
        return torch.empty([*input_tensor.shape[:-1], network.n_output_dims], device=input_tensor.device)

    return network(input_tensor)


class BaseInvertibleActivation(nn.Module):
    inverse: bool

    def __init__(self, inverse: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            return self.inverse_function(x)
        else:
            return self.function(x)

    def function(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} function not defined for activation")

    def inverse_function(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} inverse function not defined for activation")


class SkipActivation(BaseInvertibleActivation):
    def function(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_function(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SigmoidActivation(BaseInvertibleActivation):
    def function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def inverse_function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1 - x))


class ExpActivation(BaseInvertibleActivation):
    def function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def inverse_function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


class NormalizeActivation(BaseInvertibleActivation):
    def function(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x)


class SaturateActivation(BaseInvertibleActivation):
    def function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0, max=1)


class SoftmaxChannel0Activation(BaseInvertibleActivation):
    """
    Applies softmax to the last dimension and returns the first channel.
    """

    def function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)[..., 0]


def get_activation(activation_name: str, inverse: bool = False) -> nn.Module:
    match activation_name.lower():
        case "relu":
            if inverse:
                raise NotImplementedError(f"inverse of {activation_name} currently not supported")
            else:
                return nn.ReLU(inplace=True)
        case "softplus":
            if inverse:
                raise NotImplementedError(f"inverse of {activation_name} currently not supported")
            else:
                return nn.Softplus(beta=10)
        case "sigmoid":
            return SigmoidActivation(inverse=inverse)
        case "exp":
            return ExpActivation(inverse=inverse)
        case "normalize":
            return NormalizeActivation(inverse=inverse)
        case "saturate":
            if inverse:
                raise NotImplementedError(f"inverse of {activation_name} currently not supported")
            else:
                return SaturateActivation()
        case "softmax-channel-0":
            return SoftmaxChannel0Activation(inverse=inverse)
        case "none":
            return SkipActivation(inverse=inverse)
        case "skip":
            return SkipActivation(inverse=inverse)
        case _:
            raise ValueError(f"activation function {activation_name} not supported.")


class MixedPrecisionNoCastPlugin(MixedPrecision):
    """
    This plugin uses the behavior of the original `MixedPrecision`, 
    but only uses the grad scaler and skip the global autocasting to half precision.
    Local half-precision casting is still possible when this plugin is enabled.
    
    Global autocasting to half is bad because:
    - We would never know which module breaks or misbehaves in half precision. \
        There will always be unknown unpredictable omission. \
    - When we add new modules, global autocasting make it difficult to debug, \
        while some existing modules must use grad scaler.
    So instead, we should disable autocast globally, and only enable it locally in known modules.
    
    Known issues on half precision (where you should not use half):
    - For rays, poses and coordinates, half precision introduce very large errors. We should at least use float.
    - Some of the pytorch module is not autocast-safe (e.g. bce, exp, ...)
    - ... TBA
    """

    def __init__(self, *args, **kwargs) -> None:
        # Use a smaller value of `init_scale` then the default `2**16==65536`
        kwargs["scaler"] = GradScaler(init_scale=128.0)
        super().__init__(*args, **kwargs)

    @contextlib.contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Skip the global autocast in the original plugin."""
        yield
