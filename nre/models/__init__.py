# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone. The full nre/models/ tree
# previously re-exported background / calib / feature_volume /
# object_feature_volume / tracks_calib / composite / custom_modules /
# input_embedding / post_processing / etc. None of those are reachable
# from instant_nurec.cli; dropping them brings the codebase closer to the
# 15K-LOC target.

from nre.models.nrenderable import NRenderableModel


__all__ = [
    "NRenderableModel",
]
