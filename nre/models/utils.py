# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from torch import nn

from nre.models.base import BaseModel
from nre.utils.profiling import ScopedTimer


def update_module_step(m: BaseModel, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
    with ScopedTimer(f"{m.__class__.__name__}/update_step_train_batch_start"):
        additional_parameters = m.update_step_train_batch_start(epoch, global_step, system, **kwargs)
    return additional_parameters



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


