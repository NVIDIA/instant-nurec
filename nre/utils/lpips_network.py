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

import torch
import torch.nn as nn
import torch.utils.checkpoint

from python.runfiles import runfiles
from torchvision import models


RUNFILES = runfiles.Create()


class VGG16Network(torch.nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super(VGG16Network, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> list[torch.Tensor]:
        h = self.slice1(X)
        h_relu1_2: torch.Tensor = h
        h = self.slice2(h)
        h_relu2_2: torch.Tensor = h
        h = self.slice3(h)
        h_relu3_3: torch.Tensor = h
        h = self.slice4(h)
        h_relu4_3: torch.Tensor = h
        h = self.slice5(h)
        h_relu5_3: torch.Tensor = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class ScalingLayer(nn.Module):
    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super(NetLinLayer, self).__init__()
        layers: list[nn.Module] = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class LPIPSNetwork(nn.Module):
    """
    Learned perceptual metric network with pretrained VGG-16 weights
    Reference: https://github.com/richzhang/PerceptualSimilarity/tree/master/models
    """

    def __init__(self, use_dropout: bool = True, chunk_size: int = 4):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG-16 features
        self.net = VGG16Network(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.chunk_size = chunk_size

        # Load pretrained weights
        assert RUNFILES is not None, "RUNFILES is not initialized"
        weight_path = RUNFILES.Rlocation("lpips_vgg_weights/vgg.pth")
        assert weight_path is not None, "Weight path is not initialized"
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    @staticmethod
    def _spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return x.mean([2, 3], keepdim=keepdim)

    def raw_forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the LPIPS feature distance between two images. Note that both input and target are of shape [N, 3, H, W]
        with RGB values normalized to [-1, 1]
        """
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = self._normalize_tensor(outs0[kk]), self._normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [self._spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

    @torch.compile(fullgraph=True)
    def compiled_raw_forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.raw_forward(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_chunks = math.ceil(input.shape[0] / self.chunk_size)
        loss_values: list[torch.Tensor] = []
        for i in range(n_chunks):
            input_chunk = input[i * self.chunk_size : (i + 1) * self.chunk_size]
            target_chunk = target[i * self.chunk_size : (i + 1) * self.chunk_size]
            # For small suffix chunks, use non-compiled version to avoid shape assertions during AOT-Autograd compilation
            # (where shape would mismatch due to different strides)
            fn = self.compiled_raw_forward if input_chunk.shape[0] == self.chunk_size else self.raw_forward
            val = torch.utils.checkpoint.checkpoint(fn, input_chunk, target_chunk, use_reentrant=False)
            loss_values.append(val)
        return torch.cat(loss_values, dim=0)
