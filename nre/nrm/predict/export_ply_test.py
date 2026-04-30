# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Any

import torch

from nre.nrm.config.predict import PrimitivePLYExportConfig
from nre.nrm.predict.export_ply import (
    export_kelvin_ply,
    get_gaussian_shape_pruning_mask,
)
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive, KelvinSemanticClass, KelvinStaticLayer


def _make_default_export_config(**overrides: Any) -> PrimitivePLYExportConfig:
    """Create a PrimitivePLYExportConfig with permissive defaults for testing."""
    defaults: dict[str, Any] = dict(
        enabled=True,
        density_activation="sigmoid",
        scale_activation="exp",
        color_mode="sh",
        apply_affine_mtx=False,
        falloff_sigma_timestamp_us=None,
        minimum_density=None,
        minimum_scale=None,
        minimum_surface_area=None,
        maximum_velocity=None,
        maximum_sky_mask=None,
    )
    defaults.update(overrides)
    return PrimitivePLYExportConfig(**defaults)


class TestGetGaussianShapePruningMask:
    """Tests for get_gaussian_shape_pruning_mask."""

    def test_no_filters(self):
        """With all thresholds None, all finite gaussians pass."""
        config = _make_default_export_config()
        densities = torch.tensor([[0.5], [0.8], [0.1]])
        scales = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.01, 0.01, 0.01]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask.shape == (3, 1)
        assert mask.all()

    def test_removes_non_finite_densities(self):
        """Non-finite (NaN, inf) densities are always filtered."""
        config = _make_default_export_config()
        densities = torch.tensor([[0.5], [float("nan")], [float("inf")]])
        scales = torch.tensor([[0.1, 0.1, 0.1]] * 3)
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [True, False, False]

    def test_minimum_density_filter(self):
        """Gaussians below minimum_density are filtered."""
        config = _make_default_export_config(minimum_density=0.5)
        densities = torch.tensor([[0.3], [0.5], [0.8]])
        scales = torch.tensor([[0.1, 0.1, 0.1]] * 3)
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [False, True, True]

    def test_minimum_scale_filter(self):
        """Gaussians with any scale dimension below threshold are filtered."""
        config = _make_default_export_config(minimum_scale=0.05)
        densities = torch.tensor([[1.0], [1.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.01, 0.1]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [True, False]

    def test_minimum_surface_area_filter(self):
        """Gaussians with surface area below threshold are filtered."""
        config = _make_default_export_config(minimum_surface_area=0.1)
        densities = torch.tensor([[1.0], [1.0]])
        # surface_area = x^2 + y^2 + z^2: [0.03, 0.12]
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [False, True]

    def test_combined_filters(self):
        """Multiple filters are applied together (AND logic)."""
        config = _make_default_export_config(minimum_density=0.5, minimum_scale=0.05)
        densities = torch.tensor([[0.8], [0.3], [0.8]])
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.01, 0.1, 0.1]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        # Only first passes both filters
        assert mask[:, 0].tolist() == [True, False, False]


class TestExportKelvinPly:
    """Tests for export_kelvin_ply."""

    def _make_kelvin_primitive(self, n_gaussians: int) -> KelvinNRMPrimitive:
        """Create a minimal KelvinNRMPrimitive on CPU for testing."""
        return KelvinNRMPrimitive.random(n_gaussians, use_2dgs=True, device=torch.device("cpu"))

    def _make_kelvin_primitive_with_semantics(self, n_gaussians: int) -> KelvinNRMPrimitive:
        """Create a KelvinNRMPrimitive with semantic_class populated."""
        prim = self._make_kelvin_primitive(n_gaussians)
        semantic_class = torch.zeros(n_gaussians, 1, dtype=torch.uint8)
        prim = KelvinNRMPrimitive(
            static_layer=KelvinStaticLayer(
                positions=prim.static_layer.positions,
                rotations=prim.static_layer.rotations,
                scales=prim.static_layer.scales,
                densities=prim.static_layer.densities,
                rgb=prim.static_layer.rgb,
                semantic_class=semantic_class,
            ),
            dynamic_layers=prim.dynamic_layers,
            sky_cubemap=prim.sky_cubemap,
            affine_matrix=prim.affine_matrix,
            use_2dgs=prim.use_2dgs,
            gaussians_renderer=None,
        )
        return prim

    def test_basic_export(self):
        """Export produces PLYExportGaussians with correct tensor shapes."""
        primitives = self._make_kelvin_primitive(100)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        # All gaussians should pass (no filters active)
        assert result.positions.shape == (100, 3)
        assert result.rotations.shape == (100, 4)
        assert result.scales.shape == (100, 3)
        assert result.densities.shape == (100, 1)
        assert result.rgb.shape == (100, 3)

    def test_masks_none_without_semantic_class(self):
        """Without semantic_class, road_mask and sky_mask are None."""
        primitives = self._make_kelvin_primitive(10)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.road_mask is None
        assert result.sky_mask is None

    def test_road_mask_from_semantic_class(self):
        """road_mask is derived from semantic_class == ROAD."""
        primitives = self._make_kelvin_primitive_with_semantics(5)
        primitives.static_layer.semantic_class[1] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[3] = KelvinSemanticClass.ROAD
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.road_mask is not None
        assert result.road_mask.shape == (5,)
        assert result.road_mask.dtype == torch.uint8
        assert result.road_mask.tolist() == [0, 1, 0, 1, 0]

    def test_sky_mask_from_semantic_class(self):
        """sky_mask is derived from semantic_class == SKY."""
        primitives = self._make_kelvin_primitive_with_semantics(4)
        primitives.static_layer.semantic_class[0] = KelvinSemanticClass.SKY
        primitives.static_layer.semantic_class[2] = KelvinSemanticClass.SKY
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.sky_mask is not None
        assert result.sky_mask.shape == (4,)
        assert result.sky_mask.dtype == torch.float32
        assert result.sky_mask.tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_masks_with_filtering(self):
        """Masks survive density filtering and have correct length."""
        primitives = self._make_kelvin_primitive_with_semantics(6)
        primitives.static_layer.semantic_class[0] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[2] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[4] = KelvinSemanticClass.SKY
        # Filter out half by density
        primitives.static_layer.densities[:3] = 0.9
        primitives.static_layer.densities[3:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 3
        assert result.road_mask is not None
        assert result.road_mask.shape == (3,)
        # Indices 0 (ROAD), 1 (OTHERS), 2 (ROAD) survive
        assert result.road_mask.tolist() == [1, 0, 1]

    def test_density_filtering(self):
        """Density filter reduces the number of exported gaussians."""
        primitives = self._make_kelvin_primitive(100)
        # Set half the densities below threshold
        primitives.static_layer.densities[:50] = 0.01
        primitives.static_layer.densities[50:] = 0.9
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 50

    def test_single_gaussian_squeeze(self):
        """Single-gaussian case doesn't produce scalar tensors (squeeze(-1) fix)."""
        primitives = self._make_kelvin_primitive(1)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.positions.ndim == 2
        assert result.positions.shape == (1, 3)
        assert result.densities.shape == (1, 1)

    def test_normals_propagated_to_export(self):
        """Predicted normals on the static layer survive filtering and are forwarded to the PLY export struct."""
        primitives = self._make_kelvin_primitive(6)
        normals = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
        primitives.static_layer.normals = normals
        primitives.static_layer.densities[:3] = 0.9
        primitives.static_layer.densities[3:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.normals is not None
        assert result.normals.shape == (3, 3)
        assert torch.allclose(result.normals, normals[:3])

    def test_normals_none_when_absent(self):
        """When normals are not present on the static layer, export carries None."""
        primitives = self._make_kelvin_primitive(4)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.normals is None

    def test_all_filtered_produces_empty(self):
        """When all gaussians are filtered, result has zero-length tensors."""
        primitives = self._make_kelvin_primitive(10)
        # Set all densities very low
        primitives.static_layer.densities[:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 0
