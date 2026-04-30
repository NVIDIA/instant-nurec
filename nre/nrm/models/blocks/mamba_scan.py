# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

from typing import Literal

import torch
import torch.nn as nn

from einops import rearrange

from nre.nrm.utils.optim import mark_parameter_no_weight_decay


# Mamba-SSM is found to be drastically increasing the bootstrap time of the nrm targets (specifically the venv creation stage).
# For local debugging, one might want to temporarily remove/comment out the "pip_requirement_internal" lines in the BUILD.bazel file.
# We hence add a try-except block here to avoid the import error.
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except ModuleNotFoundError:
    pass


class Mamba2SingleScan(nn.Module):
    """
    Linear-memory single-step uni-directional operation in the Mamba2 model.
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        conv_init: float | None,
        expand: int,
        headdim: int,
        ngroups: int,
        A_init_range: tuple[float, float],
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        dt_limit: tuple[float, float],
        activation: str,
        conv_bias: bool,
        # Fused kernel and sharding options
        chunk_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.activation = activation
        self.chunk_size = chunk_size

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)

        self._mark_parameters_no_weight_decay()
        # After loading state dict, the parameter customized attributes will be lost, so we mark them again.
        self.register_load_state_dict_post_hook(self._mark_parameters_no_weight_decay)

    def _mark_parameters_no_weight_decay(self, *args, **kwargs):
        mark_parameter_no_weight_decay(self.dt_bias)
        mark_parameter_no_weight_decay(self.A_log)
        mark_parameter_no_weight_decay(self.D)

    def forward(self, zxbcdt: torch.Tensor) -> torch.Tensor:
        """
        zxbcdt: (B, L, D)
        Returns: same shape as input
        """
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Fully fused path
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=self.chunk_size,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        return out


class Mamba2MultiScan(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        conv_init: float | None,
        expand: int,
        headdim: int,
        ngroups: int,
        A_init_range: tuple[float, float],
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        dt_limit: tuple[float, float],
        activation: str,
        bias: bool,
        conv_bias: bool,
        # Fused kernel and sharding options
        chunk_size: int,
        scan_type: Literal["single", "bi"],  # single, bi
        if_divide_out: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.scan_type = scan_type
        self.if_divide_out = if_divide_out

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        self.mamba_scans = nn.ModuleList()
        self.scan_num = 1 if scan_type == "single" else 2
        for _ in range(self.scan_num):
            self.mamba_scans.append(
                Mamba2SingleScan(
                    d_model,
                    d_state,
                    d_conv,
                    conv_init,
                    expand,
                    headdim,
                    ngroups,
                    A_init_range,
                    dt_min,
                    dt_max,
                    dt_init_floor,
                    dt_limit,
                    activation,
                    conv_bias,
                    chunk_size,
                )
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, D)
        Returns: same shape as input
        """

        xz = self.in_proj(hidden_states)  # (B, L, d_in_proj), [z,x,B,C,dt]

        xzs = [xz]
        if self.scan_type == "bi":
            xzs.append(xz.flip([1]))

        outs = []
        for i in range(self.scan_num):
            out = self.mamba_scans[i](xzs[i])
            if i == 0:
                outs.append(out)
            elif i == 1:
                outs.append(out.flip([1]))

        out = sum(outs)
        if self.if_divide_out:
            out = out / self.scan_num

        out = self.out_proj(out)

        return out


class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        conv_init: float | None = None,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        A_init_range: tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        learnable_init_states: bool = False,
        activation: str = "swish",
        bias: bool = False,
        conv_bias: bool = True,
        # Fused kernel and sharding options
        chunk_size: int = 256,
        scan_type: Literal["single", "bi"] = "bi",
        if_divide_out: bool = False,
        norm_cls: Literal["rms_norm", "layer_norm"] = "rms_norm",
    ):
        super().__init__()
        self.norm = RMSNorm(d_model) if norm_cls == "rms_norm" else nn.LayerNorm(d_model)
        self.mamba = Mamba2MultiScan(
            d_model,
            d_state,
            d_conv,
            conv_init,
            expand,
            headdim,
            ngroups,
            A_init_range,
            dt_min,
            dt_max,
            dt_init_floor,
            dt_limit,
            activation,
            bias,
            conv_bias,
            chunk_size,
            scan_type,
            if_divide_out,
        )
        self._mark_parameters_no_weight_decay()
        # After loading state dict, the parameter customized attributes will be lost, so we mark them again.
        self.register_load_state_dict_post_hook(self._mark_parameters_no_weight_decay)

    def _mark_parameters_no_weight_decay(self, *args, **kwargs):
        # Mark all bias parameters as no weight decay, recursively
        for param_name, param in self.named_parameters():
            if param_name.endswith(".bias"):
                mark_parameter_no_weight_decay(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) eg: (1, 3604, 1024)
        Returns: same shape as input
        """
        x = x + self.mamba(self.norm(x))
        return x
