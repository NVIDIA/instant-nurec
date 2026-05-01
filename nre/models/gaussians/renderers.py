# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Predict-only standalone keeps BaseGaussianRenderer as an empty stub. The
# full renderer implementations (libs/nrend bindings, gsplat, OptiX paths)
# only fire under primitive.forward()/render(), which predict never invokes;
# gaussians_renderer is set to None in KelvinNRM.__init__.

from __future__ import annotations

from abc import ABC


class BaseGaussianRenderer(ABC):
    """Stub: predict mode never instantiates a renderer."""
