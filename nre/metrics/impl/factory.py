# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable

from nre.metrics.impl.cpsnr import CPSNRMetric
from nre.metrics.impl.drift import FeatureDriftMetric
from nre.metrics.impl.fcs_adaptive import FCSAdaptiveMetric
from nre.metrics.impl.fid import FIDMetric
from nre.metrics.impl.higher_order_moments import HigherOrderMomentsMetric
from nre.metrics.impl.lidar_common import LIDARCommonMetrics
from nre.metrics.impl.lpips import LPIPSMetric
from nre.metrics.impl.ntd import NTDMetric
from nre.metrics.impl.object_level_perceptual import ObjectLevelPerceptualMetric
from nre.metrics.impl.object_level_semantic import ObjectLevelSemanticMetric
from nre.metrics.impl.perceptual import PerceptualMetric
from nre.metrics.impl.psnr import PSNRMetric
from nre.metrics.impl.ssim import SSIMMetric
from nre.metrics.impl.temporal_coherence import TemporalCoherenceMetric
from nre.metrics.metric import BaseMetric
from nre.metrics.types import MetricType


MetricFactory: dict[MetricType, Callable[..., BaseMetric]] = {
    MetricType.PSNR: PSNRMetric,
    MetricType.CPSNR: CPSNRMetric,
    MetricType.SSIM: SSIMMetric,
    MetricType.LPIPS: LPIPSMetric,
    MetricType.LIDAR_COMMON: LIDARCommonMetrics,
    MetricType.FEATURE_DRIFT: FeatureDriftMetric,
    MetricType.FID: FIDMetric,
    MetricType.PERCEPTUAL: PerceptualMetric,
    MetricType.TEMPORAL_COHERENCE: TemporalCoherenceMetric,
    MetricType.NTD: NTDMetric,
    MetricType.FCS_ADAPTIVE: FCSAdaptiveMetric,
    MetricType.D_SKEW: HigherOrderMomentsMetric,
    MetricType.D_KURT: HigherOrderMomentsMetric,
    MetricType.OBJECT_LEVEL_SEMANTIC: ObjectLevelSemanticMetric,
    MetricType.OBJECT_LEVEL_PERCEPTUAL: ObjectLevelPerceptualMetric,
}
