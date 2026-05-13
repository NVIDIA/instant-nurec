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

"""Branch-coverage tests for instant_nurec.utils.gaussians.utils.

The module imports ``point_cloud_utils`` at top-level (compiled wheel
unavailable in the cpu test venv); we stub it via sys.modules. The
stub's TriangleMesh records what was written to it, so we can verify
write_ply_3dgs's mapping from torch tensors → mesh attributes without
actually writing a PLY.
"""

from __future__ import annotations

import sys
import types as _typesmod
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _FakeMesh:
    """Minimal stand-in for pcu.TriangleMesh — records what gets written."""

    def __init__(self):
        self.vertex_data = _typesmod.SimpleNamespace(custom_attributes={})
        self.saved_paths: list[str] = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


@pytest.fixture(autouse=True)
def _stub_pcu(monkeypatch: pytest.MonkeyPatch):
    fake_pcu = _typesmod.ModuleType("point_cloud_utils")
    fake_pcu.TriangleMesh = _FakeMesh  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "point_cloud_utils", fake_pcu)
    sys.modules.pop("instant_nurec.utils.gaussians.utils", None)


# ---------------------------------------------------------------------------
# RGB2SH
# ---------------------------------------------------------------------------


def test_rgb2sh_torch_at_half_yields_zero():
    """RGB=0.5 is the SH zero point: (0.5 - 0.5) / C0 = 0."""
    from instant_nurec.utils.gaussians.utils import RGB2SH

    out = RGB2SH(torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(out, torch.zeros(3))


def test_rgb2sh_numpy_path():
    from instant_nurec.utils.gaussians.utils import RGB2SH

    out = RGB2SH(np.array([0.5, 0.5, 0.5]))
    assert np.allclose(out, np.zeros(3))


def test_rgb2sh_torch_known_value():
    from instant_nurec.utils.gaussians.utils import C0, RGB2SH

    out = RGB2SH(torch.tensor([1.0]))
    expected = (1.0 - 0.5) / C0
    assert out.item() == pytest.approx(expected)


def test_rgb2sh_numpy_dtype_preserved():
    from instant_nurec.utils.gaussians.utils import RGB2SH

    rgb = np.array([0.7], dtype=np.float32)
    out = RGB2SH(rgb)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# write_ply_3dgs
# ---------------------------------------------------------------------------


def _minimal_kwargs(tmp_path):
    """Construct the smallest-valid set of kwargs for write_ply_3dgs."""
    n = 3
    return dict(
        path=tmp_path / "out.ply",
        positions=torch.randn(n, 3),
        rotations=torch.randn(n, 4),
        scales=torch.randn(n, 3),
        densities=torch.randn(n),
        features_albedo=torch.randn(n, 3),
    )


def test_write_ply_3dgs_minimal_required_attrs(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs

    write_ply_3dgs(**_minimal_kwargs(tmp_path))
    # Parent dir was created
    assert tmp_path.exists()


def test_write_ply_3dgs_writes_rotation_attrs(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs
    import instant_nurec.utils.gaussians.utils as mod

    seen_meshes: list[_FakeMesh] = []

    class _Capturing(_FakeMesh):
        def __init__(self):
            super().__init__()
            seen_meshes.append(self)

    mod.pcu.TriangleMesh = _Capturing  # type: ignore[attr-defined]
    write_ply_3dgs(**_minimal_kwargs(tmp_path))
    assert len(seen_meshes) == 1
    attrs = seen_meshes[0].vertex_data.custom_attributes
    # rot_0..rot_3, scale_0..scale_2, opacity, f_dc_0..f_dc_2
    for i in range(4):
        assert f"rot_{i}" in attrs
    for i in range(3):
        assert f"scale_{i}" in attrs
    assert "opacity" in attrs
    for i in range(3):
        assert f"f_dc_{i}" in attrs


def test_write_ply_3dgs_color_branch(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs
    import instant_nurec.utils.gaussians.utils as mod

    seen: list[_FakeMesh] = []

    class _Capturing(_FakeMesh):
        def __init__(self):
            super().__init__()
            seen.append(self)

    mod.pcu.TriangleMesh = _Capturing  # type: ignore[attr-defined]

    kwargs = _minimal_kwargs(tmp_path)
    kwargs["color"] = torch.full((3, 3), 0.42)
    write_ply_3dgs(**kwargs)
    # color was written
    assert hasattr(seen[0].vertex_data, "colors")
    assert seen[0].vertex_data.colors.shape == (3, 3)


def test_write_ply_3dgs_normals_branch(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs
    import instant_nurec.utils.gaussians.utils as mod

    seen: list[_FakeMesh] = []

    class _Capturing(_FakeMesh):
        def __init__(self):
            super().__init__()
            seen.append(self)

    mod.pcu.TriangleMesh = _Capturing  # type: ignore[attr-defined]

    kwargs = _minimal_kwargs(tmp_path)
    kwargs["normals"] = torch.randn(3, 3)
    write_ply_3dgs(**kwargs)
    assert hasattr(seen[0].vertex_data, "normals")


def test_write_ply_3dgs_normals_shape_mismatch_assertion(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs

    kwargs = _minimal_kwargs(tmp_path)
    # Wrong shape: positions is (3,3) but we hand normals (3,4)
    kwargs["normals"] = torch.randn(3, 4)
    with pytest.raises(AssertionError, match="normals must have the same shape"):
        write_ply_3dgs(**kwargs)


def test_write_ply_3dgs_custom_attributes(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs
    import instant_nurec.utils.gaussians.utils as mod

    seen: list[_FakeMesh] = []

    class _Capturing(_FakeMesh):
        def __init__(self):
            super().__init__()
            seen.append(self)

    mod.pcu.TriangleMesh = _Capturing  # type: ignore[attr-defined]

    kwargs = _minimal_kwargs(tmp_path)
    kwargs["custom_attributes"] = {
        "road_mask": torch.zeros(3, dtype=torch.uint8),
        "sky_mask": torch.ones(3, dtype=torch.float32),
    }
    write_ply_3dgs(**kwargs)
    attrs = seen[0].vertex_data.custom_attributes
    assert "road_mask" in attrs
    assert "sky_mask" in attrs


def test_write_ply_3dgs_creates_parent_dir(tmp_path):
    from instant_nurec.utils.gaussians.utils import write_ply_3dgs

    nested = tmp_path / "a" / "b" / "c"
    kwargs = _minimal_kwargs(nested)
    # Parent dir doesn't exist yet
    assert not nested.exists()
    write_ply_3dgs(**kwargs)
    assert nested.exists()
