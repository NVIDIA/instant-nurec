# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Branch-coverage tests for KelvinPrimitiveMerge._maybe_voxelize_static_layer.

The full ``merge_primitives_and_batch`` flow needs a fully-populated batch
+ rig trajectories which is exercised end-to-end by integration runs and
is out of scope for the unit tests; what's testable here is the
voxelization hook itself, which is the only behavior added by this
commit.
"""

import logging

import pytest
import torch

from instant_nurec.config_schema.predict import PrimitiveMergeConfig
from instant_nurec.predict.primitive_merge import KelvinPrimitiveMerge
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinStaticLayer,
)


def _identity_quat_wxyz(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _make_static_layer(positions: torch.Tensor) -> KelvinStaticLayer:
    n = positions.shape[0]
    return KelvinStaticLayer(
        positions=positions,
        densities=torch.full((n, 1), 0.5),
        rotations=_identity_quat_wxyz(n),
        scales=torch.full((n, 3), 0.1),
        rgb=torch.zeros(n, 3),
    )


def _make_dynamic_layer(n: int = 2) -> KelvinDynamicLayer:
    return KelvinDynamicLayer(
        rotations=_identity_quat_wxyz(n),
        scales=torch.full((n, 3), 0.1),
        rgb=torch.zeros(n, 3),
        max_densities=torch.full((n, 1), 0.5),
        keyframe_positions=torch.zeros(n, 3, 3),
        keyframe_timestamps_us=torch.tensor([[0.0, 1.0, 2.0]]).expand(n, 3).contiguous(),
    )


def _make_primitive(positions: torch.Tensor) -> KelvinInstantNuRecPrimitive:
    return KelvinInstantNuRecPrimitive(
        static_layer=_make_static_layer(positions),
        dynamic_layers=[_make_dynamic_layer(n=2)],
        sky_cubemap=torch.zeros(6, 4, 4, 3),
        affine_matrix=torch.eye(3, 4).unsqueeze(0),
    )


class TestMaybeVoxelizeStaticLayer:
    """Branch coverage for the iterative voxelization search.

    Positions are picked so that ``voxelize()`` (which buckets via
    ``(positions / voxel_size).round().int()``) produces predictable
    counts at the voxel sizes the iteration visits.
    """

    def test_disabled_returns_primitive_unchanged(self, caplog):
        """enable_voxelization=False -> static_layer object is the same instance, no log."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(PrimitiveMergeConfig(enable_voxelization=False))
        original_layer = primitive.static_layer
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert out is primitive
        assert out.static_layer is original_layer
        assert not any("Voxelization" in r.message for r in caplog.records)

    def test_target_hits_at_initial_voxel_size(self, caplog):
        """Initial voxel_size already produces a count in [0.9 * target, target] -> no iteration."""
        # Two points at distinct buckets (size 0.1) -> 2 Gaussians; target=2 hits immediately.
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1, target_n_gaussians=2)
        )
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert len(out.static_layer) == 2
        msgs = [r.message for r in caplog.records]
        assert any("converged in 1 iter" in m for m in msgs)
        assert any("voxel_size=0.1" in m for m in msgs)

    def test_target_requires_doubling(self, caplog):
        """count > target -> voxel_size doubles until the count collapses into the band.

        4 corners of a 0.4-edge box give 4 distinct buckets at voxel_size 0.1,
        0.2, 0.4 (each point in its own cell) and collapse to a single bucket
        at voxel_size 0.8 (round(0.4 / 0.8) = round(0.5) = 0). With target=1,
        the iteration reaches voxel_size 0.8 in 4 attempts.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]]
        )
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1, target_n_gaussians=1)
        )
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert len(out.static_layer) == 1
        msgs = [r.message for r in caplog.records]
        assert any("converged in 4 iter" in m for m in msgs)
        assert any("voxel_size=0.8" in m for m in msgs)

    def test_target_requires_halving(self, caplog):
        """count < 0.9 * target -> voxel_size halves until the count climbs into the band.

        4 near-origin points (within 0.05) collapse to a single bucket at
        voxel_size 0.1 (round(0.05/0.1)=round(0.5)=0 for all three off-origin
        points) but split into 4 distinct buckets at voxel_size 0.05. target=4
        therefore takes one halving to land on (4 in [3.6, 4]).
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]
        )
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1, target_n_gaussians=4)
        )
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert len(out.static_layer) == 4
        msgs = [r.message for r in caplog.records]
        assert any("converged in 2 iter" in m for m in msgs)
        assert any("voxel_size=0.05" in m for m in msgs)

    def test_iteration_re_voxelizes_from_original_each_step(self):
        """Each iteration must consume the original static layer, not the
        result of the previous voxelization. Otherwise repeated halving
        would shrink a collapsed-to-one cloud forever (it never grows
        back) and the doubling case for target=1 below would return >1.
        """
        # Same data as test_target_requires_doubling: at voxel_size=0.4 the
        # 4 points are still 4 distinct buckets. If we mistakenly voxelized
        # from the previous (already 4-point) intermediate at each step,
        # iteration could never reduce below 4 because the intermediate
        # carries the same positions as the original. The distinguishing
        # check is that at voxel_size=0.8 the count drops to 1 (which only
        # happens when feeding the *original* positions).
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]]
        )
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1, target_n_gaussians=1)
        )
        out = merger._maybe_voxelize_static_layer(primitive)
        assert len(out.static_layer) == 1

    def test_binary_search_midpoint_hits_band_after_bracketing(self, monkeypatch, caplog):
        """When the count drops too sharply per doubling step, the bracketed
        midpoint between a too-many and a too-few voxel size hits the band
        even though pure doubling/halving would oscillate.

        Synthetic count function (mocked):
            voxel_size 0.1 -> 10   (above target 7)
            voxel_size 0.2 ->  5   (below 0.9 * 7 = 6.3)
            voxel_size 0.15 -> 7   (in band)

        With pure doubling/halving (the prior algorithm), this oscillates
        between 0.1 and 0.2. The bracketed binary-search step picks
        midpoint 0.15 on the third iteration and hits.
        """
        call_sizes: list[float] = []

        def mock_voxelize(self, voxel_size, confidence=None):  # noqa: ARG001
            call_sizes.append(voxel_size)
            # Plateaus by voxel_size band; the midpoint 0.15 lands in [0.125, 0.175].
            if voxel_size < 0.125:
                n = 10
            elif voxel_size < 0.175:
                n = 7
            else:
                n = 5
            return _make_static_layer(torch.zeros(n, 3))

        monkeypatch.setattr(KelvinStaticLayer, "voxelize", mock_voxelize)

        primitive = _make_primitive(torch.zeros(1, 3))
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(
                enable_voxelization=True, voxel_size=0.1, target_n_gaussians=7
            )
        )
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert call_sizes == pytest.approx([0.1, 0.2, 0.15])
        assert len(out.static_layer) == 7
        msgs = [r.message for r in caplog.records]
        assert any("converged in 3 iter" in m for m in msgs)
        assert any("voxel_size=0.15" in m for m in msgs)

    def test_bracket_collapse_terminates_with_warning(self, caplog):
        """When the band lies between two adjacent count plateaus, the
        bracket [v_low, v_high] shrinks below ``BRACKET_TOL`` without the
        count ever entering the band, and the loop exits with a WARNING.

        The 4-near-origin-points cloud has only two count plateaus on the
        relevant range: count=4 for voxel_size < 0.1 and count=1 for
        voxel_size >= 0.1. The band [1.8, 2] for target=2 sits between
        them and is unreachable.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]
        )
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(
                enable_voxelization=True,
                voxel_size=0.1,
                target_n_gaussians=2,
                max_voxelization_iterations=30,
            )
        )
        with caplog.at_level(logging.WARNING, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert len(out.static_layer) in (1, 4)
        msgs = [r.message for r in caplog.records]
        assert any("bracket collapsed" in m for m in msgs)

    def test_max_iter_exceeded_without_bracket(self, caplog):
        """When the upper bracket has not yet been established (all
        iterations so far saw count > target) and ``max_iter`` trips, the
        loop exits with the "did not converge" WARNING.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]]
        )
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(
            PrimitiveMergeConfig(
                enable_voxelization=True,
                voxel_size=0.1,
                target_n_gaussians=1,
                max_voxelization_iterations=3,
            )
        )
        with caplog.at_level(logging.WARNING, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        # Three doublings: 0.1 -> 0.2 -> 0.4 -> (about to try 0.8 but max_iter
        # already consumed). At v=0.4 each point is in its own cell -> 4.
        assert len(out.static_layer) == 4
        msgs = [r.message for r in caplog.records]
        assert any("did not converge after 3 iter" in m for m in msgs)
