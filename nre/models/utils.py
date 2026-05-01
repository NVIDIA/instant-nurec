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


def get_activation(activation_name: str, inverse: bool = False) -> nn.Module:
    """Predict-only standalone restricts activation_name to {"sigmoid", "exp"};
    NRE supported many more, but the ExportPLYConfig Literal locks the field to
    those two."""
    match activation_name.lower():
        case "sigmoid":
            return SigmoidActivation(inverse=inverse)
        case "exp":
            return ExpActivation(inverse=inverse)
        case _:
            raise ValueError(f"activation function {activation_name} not supported.")
