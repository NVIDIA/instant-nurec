# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import torch

from point_cloud_utils import chamfer_distance
from torch._prims_common import DeviceLikeType

from nre.config.systems import (
    GaussiansSystemConfig,
    NRendTestGaussiansSystemConfig,
)
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors
from nre.utils.batch import DataBatch, RenderingData
from nre.utils.misc import unpack_optional
from nre.utils.types import GaussiansRenderReturn, RayFlags


class LIDARCommonMetrics(BaseMetric):
    _NAME = MetricType.LIDAR_COMMON.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        compute_raydrop: bool = True,
        compute_depth: bool = True,
        compute_chamfer_distance: bool = True,
        compute_intensity: bool = True,
    ) -> None:
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for Common LIDAR metrics.")

        self.compute_raydrop = compute_raydrop
        self.compute_depth = compute_depth
        self.compute_chamfer_distance = compute_chamfer_distance
        self.compute_intensity = compute_intensity
        if device:
            self.to(device)

    def _compute(
        self,
        data_lidar: DataBatch.Lidar,
        rendering_lidar: RenderingData,
        rendered: GaussiansRenderReturn,
        did_return_pred: torch.Tensor | None,
        config: GaussiansSystemConfig | NRendTestGaussiansSystemConfig,
    ) -> MetricResult:
        values: dict[str, torch.Tensor] = {}

        # flags are shaped (b, h, w, 1)
        is_valid = data_lidar.labels.get_mask_flags_none(RayFlags.INVALID).squeeze(-1)
        # rendering_lidar has shape (b, h, w, 6)
        rays_lidar = rendering_lidar.rays[is_valid]
        rendered_valid = rendered[is_valid.reshape(-1)]

        dist_gt = unpack_optional(data_lidar.labels.distance)[is_valid].reshape(-1)
        dist_pred = unpack_optional(rendered_valid.distance)
        did_return_gt = data_lidar.labels.get_mask_flags_none(RayFlags.DROPPED)[is_valid].reshape(-1)
        if config.test.lidar.ROI.min_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt >= config.test.lidar.ROI.min_m)
        if config.test.lidar.ROI.max_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt <= config.test.lidar.ROI.max_m)
        if did_return_pred is None:
            if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.raydrop is not None:
                did_return_pred = (
                    rendered_valid.extra_ray_signals.raydrop.reshape(-1) < config.test.lidar.raydrop_threshold
                )
            else:
                did_return_pred = torch.ones_like(dist_pred, dtype=torch.bool)

        if self.compute_raydrop and did_return_pred is not None:
            raydrop_metrics = self._compute_raydrop(did_return_gt, did_return_pred)
            values.update(raydrop_metrics)
        else:
            self.compute_raydrop = False

        if self.compute_depth:
            depth_metrics = self._compute_depth(dist_gt[did_return_gt], dist_pred[did_return_gt])
            values.update(depth_metrics)

        if self.compute_chamfer_distance:
            xyz_end_pred = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_pred.unsqueeze(1)).clone()
            xyz_end_gt = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_gt.unsqueeze(1)).clone()
            if did_return_pred is not None:
                cd_metric = self._compute_chamfer_distance(xyz_end_gt[did_return_gt], xyz_end_pred[did_return_pred])
            else:  # when raydrop is disabled and GT rays are used in rendering
                cd_metric = self._compute_chamfer_distance(xyz_end_gt[did_return_gt], xyz_end_pred[did_return_gt])
            values.update(cd_metric)

        if self.compute_intensity:
            intensity_gt = (
                data_lidar.labels.intensity[is_valid].reshape(-1) if data_lidar.labels.intensity is not None else None
            )
            if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.intensity is not None:
                intensity_pred = rendered_valid.extra_ray_signals.intensity.squeeze()
                intensity_gt = unpack_optional(intensity_gt)
                intensity_metrics = self._compute_intensity(intensity_gt[did_return_gt], intensity_pred[did_return_gt])
                values.update(intensity_metrics)
            else:
                self.compute_intensity = False

        return MetricResult(values=values)

    def _compute_raydrop(
        self,
        did_return_gt: torch.Tensor,
        did_return_pred: torch.Tensor,
    ):
        # Validate shapes match
        if did_return_pred.shape != did_return_gt.shape:
            raise ValueError(
                f"Predicted and target shapes must match: {did_return_pred.shape} vs {did_return_gt.shape}"
            )
        raydrop_accuracy = (did_return_pred == did_return_gt).float().mean()

        raydrop_gt = ~did_return_gt
        raydrop_pred = ~did_return_pred
        raydrop_tp = torch.logical_and(raydrop_gt, raydrop_pred).sum()
        raydrop_fp = torch.logical_and(did_return_gt, raydrop_pred).sum()
        raydrop_fn = torch.logical_and(raydrop_gt, did_return_pred).sum()
        eps = 1e-8
        raydrop_precision = raydrop_tp / (raydrop_tp + raydrop_fp + eps)
        raydrop_recall = raydrop_tp / (raydrop_tp + raydrop_fn + eps)
        raydrop_IoU = raydrop_tp / (raydrop_tp + raydrop_fp + raydrop_fn + eps)
        raydrop_metrics = {
            "raydrop_accuracy": raydrop_accuracy,
            "raydrop_precision": raydrop_precision,
            "raydrop_recall": raydrop_recall,
            "raydrop_IoU": raydrop_IoU,
        }
        return raydrop_metrics

    def _compute_depth(
        self,
        dist_gt: torch.Tensor,
        dist_pred: torch.Tensor,
    ):
        # Validate shapes match
        if dist_gt.shape != dist_pred.shape:
            raise ValueError(f"Predicted and target shapes must match: {dist_pred.shape} vs {dist_gt.shape}")
        depth_difference = dist_pred - dist_gt
        depth_l2 = depth_difference.square()
        depth_median_l2 = depth_l2.median()
        eps = 1e-8
        depth_mean_rel_l2 = (depth_difference / (dist_gt + eps)).square().mean()
        depth_rmse = depth_l2.sqrt().mean()
        depth_mae = depth_difference.abs().mean()
        depth_medae = depth_difference.abs().median()
        depth_recall50 = torch.sum(depth_difference.abs() < 0.5) / len(depth_difference)
        depth_metrics = {
            "depth_median_l2": depth_median_l2,
            "depth_mean_rel_l2": depth_mean_rel_l2,
            "depth_rmse": depth_rmse,
            "depth_mae": depth_mae,
            "depth_medae": depth_medae,
            "depth_recall50": depth_recall50,
        }
        return depth_metrics

    def _compute_chamfer_distance(
        self,
        xyz_end_gt: torch.Tensor,
        xyz_end_pred: torch.Tensor,
    ):
        device = xyz_end_pred.device
        metric_cd = torch.FloatTensor(
            [chamfer_distance(xyz_end_pred.cpu().numpy(), xyz_end_gt.cpu().numpy())]
        ).squeeze()
        return {"chamfer_distance": metric_cd.to(device)}

    def _compute_intensity(
        self,
        intensity_gt: torch.Tensor,
        intensity_pred: torch.Tensor,
    ):
        # Validate shapes match
        if intensity_gt.shape != intensity_pred.shape:
            raise ValueError(f"Predicted and target shapes must match: {intensity_pred.shape} vs {intensity_gt.shape}")

        intensity_difference = intensity_pred - intensity_gt
        intensity_mae = intensity_difference.abs().mean()
        intensity_rmse = intensity_difference.square().mean().sqrt()
        intensity_metrics = {
            "intensity_mae": intensity_mae,
            "intensity_rmse": intensity_rmse,
        }
        return intensity_metrics

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                aggregate_values: dict[str, torch.Tensor] = {}

                if self.compute_raydrop:
                    aggregate_raydrop_accuracy = aggregate_tensors(
                        [v["raydrop_accuracy"] for v in self._values], method=method
                    )
                    aggregate_values["raydrop_accuracy"] = aggregate_raydrop_accuracy
                    aggregate_raydrop_precision = aggregate_tensors(
                        [v["raydrop_precision"] for v in self._values], method=method
                    )
                    aggregate_values["raydrop_precision"] = aggregate_raydrop_precision
                    aggregate_raydrop_recall = aggregate_tensors(
                        [v["raydrop_recall"] for v in self._values], method=method
                    )
                    aggregate_values["raydrop_recall"] = aggregate_raydrop_recall
                    aggregate_raydrop_IoU = aggregate_tensors([v["raydrop_IoU"] for v in self._values], method=method)
                    aggregate_values["raydrop_IoU"] = aggregate_raydrop_IoU

                if self.compute_depth:
                    aggregate_depth_median_l2 = aggregate_tensors(
                        [v["depth_median_l2"] for v in self._values], method=method
                    )
                    aggregate_values["depth_median_l2"] = aggregate_depth_median_l2
                    aggregate_depth_mean_rel_l2 = aggregate_tensors(
                        [v["depth_mean_rel_l2"] for v in self._values], method=method
                    )
                    aggregate_values["depth_mean_rel_l2"] = aggregate_depth_mean_rel_l2
                    aggregate_depth_rmse = aggregate_tensors([v["depth_rmse"] for v in self._values], method=method)
                    aggregate_values["depth_rmse"] = aggregate_depth_rmse
                    aggregate_depth_mae = aggregate_tensors([v["depth_mae"] for v in self._values], method=method)
                    aggregate_values["depth_mae"] = aggregate_depth_mae
                    aggregate_depth_medae = aggregate_tensors([v["depth_medae"] for v in self._values], method=method)
                    aggregate_values["depth_medae"] = aggregate_depth_medae
                    aggregate_depth_recall50 = aggregate_tensors(
                        [v["depth_recall50"] for v in self._values], method=method
                    )
                    aggregate_values["depth_recall50"] = aggregate_depth_recall50

                if self.compute_chamfer_distance:
                    aggregate_chamfer_distance = aggregate_tensors(
                        [v["chamfer_distance"] for v in self._values], method=method
                    )
                    aggregate_values["chamfer_distance"] = aggregate_chamfer_distance

                if self.compute_intensity:
                    aggregate_intensity_mae = aggregate_tensors(
                        [v["intensity_mae"] for v in self._values], method=method
                    )
                    aggregate_values["intensity_mae"] = aggregate_intensity_mae
                    aggregate_intensity_rmse = aggregate_tensors(
                        [v["intensity_rmse"] for v in self._values], method=method
                    )
                    aggregate_values["intensity_rmse"] = aggregate_intensity_rmse

                aggregated_metrics[method] = MetricResult(values=aggregate_values)
        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.LIDAR_COMMON

    def validate_inputs(
        self,
        data_lidar: DataBatch.Lidar | None,
        rendering_lidar: RenderingData | None,
        rendered: GaussiansRenderReturn | None,
        did_return_pred: torch.Tensor | None,
        config: GaussiansSystemConfig | NRendTestGaussiansSystemConfig | None,
    ):
        """
        Validate inputs for _compute. Raises ValueError or TypeError with useful messages on mismatch.
        """

        # Basic presence
        if data_lidar is None:
            raise ValueError("data_lidar is None")
        if rendering_lidar is None:
            raise ValueError("rendering_lidar is None")
        if rendered is None:
            raise ValueError("rendered is None")
        if config is None:
            raise ValueError("config is None")

        # Check data_lidar structure: expect labels with get_mask_flags_none and optional fields
        if not hasattr(data_lidar, "labels"):
            raise AttributeError("data_lidar must have attribute 'labels'")
        labels = data_lidar.labels
        if not hasattr(labels, "get_mask_flags_none"):
            raise AttributeError("data_lidar.labels must implement get_mask_flags_none(flag)")
        # distance and intensity may be optional; but if present, must be tensor-like
        if hasattr(labels, "distance") and labels.distance is not None:
            if not torch.is_tensor(unpack_optional(labels.distance)):
                raise TypeError("data_lidar.labels.distance must be a torch.Tensor or None")
        if hasattr(labels, "intensity") and labels.intensity is not None:
            if not torch.is_tensor(labels.intensity):
                raise TypeError("data_lidar.labels.intensity must be a torch.Tensor or None")

        # Check rendering_lidar.rays existence and shape
        if not hasattr(rendering_lidar, "rays"):
            raise AttributeError("rendering_lidar must have attribute 'rays'")
        rays = rendering_lidar.rays
        if not torch.is_tensor(rays):
            raise TypeError("rendering_lidar.rays must be a torch.Tensor")
        if rays.ndim != 4:
            raise ValueError(f"rendering_lidar.rays must be 4D tensor (b,h,w,6), got ndim={rays.ndim}")
        if rays.shape[-1] != 6:
            raise ValueError(f"rendering_lidar.rays last-dim must be 6 (origin 3 + dir 3), got {rays.shape[-1]}")

        b, h, w, _ = rays.shape
        total_rays = b * h * w

        # Check flags mask shape returned by get_mask_flags_none(RayFlags.INVALID)
        is_valid_mask = labels.get_mask_flags_none(RayFlags.INVALID)
        if not torch.is_tensor(is_valid_mask):
            raise TypeError("labels.get_mask_flags_none(...) must return a torch.Tensor")
        if is_valid_mask.ndim != 4:
            raise ValueError(f"ISVALID mask must be 4D (b,h,w,1) but got ndim={is_valid_mask.ndim}")
        if is_valid_mask.shape != (b, h, w, 1):
            raise ValueError(
                f"ISVALID mask spatial dims must match rendering_lidar.rays. "
                f"Expected (b,h,w,1), got {is_valid_mask.shape}"
            )

        if did_return_pred is not None:
            assert did_return_pred.shape[0] == torch.sum(is_valid_mask.reshape(-1)), (
                "did_return_pred need to have the same shape as is_valid_mask."
            )

        # Check rendered structure: often provided as a custom dataclass/tuple; we expect tensors sized total_rays
        # The code expects rendered.distance and possibly rendered.rgb and extra_ray_signals
        if not hasattr(rendered, "distance"):
            raise AttributeError("rendered must have attribute 'distance'")
        if not torch.is_tensor(rendered.distance):
            raise TypeError("rendered.distance must be a torch.Tensor")
        # rendered.distance should correspond to flattened rays shape
        if rendered.distance.numel() != total_rays:
            raise ValueError(
                f"rendered.distance.numel() ({rendered.distance.numel()}) != total rays ({total_rays}). "
                "Ensure 'rendered' corresponds to same (b,h,w) grid as rendering_lidar."
            )

        # extra_ray_signals optional structure
        if hasattr(rendered, "extra_ray_signals") and rendered.extra_ray_signals is not None:
            ers = rendered.extra_ray_signals
            # raydrop may be present
            if hasattr(ers, "raydrop") and ers.raydrop is not None:
                if not torch.is_tensor(ers.raydrop):
                    raise TypeError("rendered.extra_ray_signals.raydrop must be a torch.Tensor")
                # allow either (b*h*w,) or (b*h*w,1); check compatible numel
                if ers.raydrop.numel() != total_rays:
                    raise ValueError(
                        f"rendered.extra_ray_signals.raydrop must have same number of elements {ers.raydrop.numel()} as total rays {total_rays}"
                    )
            if hasattr(ers, "intensity") and ers.intensity is not None:
                if not torch.is_tensor(ers.intensity):
                    raise TypeError("rendered.extra_ray_signals.intensity must be a torch.Tensor")
                if ers.intensity.numel() != total_rays:
                    raise ValueError(
                        f"rendered.extra_ray_signals.intensity must have same number of elements {ers.intensity.numel()} as total rays {total_rays}"
                    )

        # Check labels.distance shape if present: expected to align with (b,h,w,1) or (b,h,w)
        if hasattr(labels, "distance") and labels.distance is not None:
            dist = unpack_optional(labels.distance)
            if not torch.is_tensor(dist):
                raise TypeError("unpack_optional(data_lidar.labels.distance) did not yield a tensor")
            if dist.numel() != total_rays:
                # Dist may be stored with trailing singleton channel; allow shapes like (b,h,w,1)
                if not (dist.ndim == 4 and dist.shape[0:3] == (b, h, w) and dist.shape[3] == 1):
                    raise ValueError(
                        "data_lidar.labels.distance must have same number of elements as total rays or shape (b,h,w,1)"
                    )

        # Check config path existence that _compute uses
        # We expect config.test.lidar.ROI.min_m/max_m and config.test.lidar.raydrop_threshold possibly present
        if not hasattr(config, "test"):
            raise AttributeError("config must have attribute 'test'")
        if not hasattr(config.test, "lidar"):
            raise AttributeError("config.test must have attribute 'lidar'")
        # ROI presence is optional, but if present should have min_m/max_m attributes
        if hasattr(config.test.lidar, "ROI") and config.test.lidar.ROI is not None:
            roi = config.test.lidar.ROI
            if not hasattr(roi, "min_m") or not hasattr(roi, "max_m"):
                raise AttributeError("config.test.lidar.ROI must have min_m and max_m attributes")
        if not hasattr(config.test.lidar, "raydrop_threshold"):
            # if raydrop_threshold missing it's okay only if you never compute raydrop; but warn/raise for safety
            raise AttributeError("config.test.lidar must have attribute 'raydrop_threshold'")

    def reset(self) -> None:
        """Reset the metric state."""
        pass
