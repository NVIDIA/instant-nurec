"""Branch-coverage tests for nre.utils.types.

The module imports ``lietorch`` and ``ncore.data`` at module load. Both
are unavailable in the cpu-only test venv, so we stub them via
``sys.modules`` for the duration of each test. The two pure-python /
pure-numpy types we exercise (``HalfClosedInterval`` and
``FrameConversion``) don't actually use the stubbed names at runtime.
"""

from __future__ import annotations

import sys
import types as _typesmod
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _stub_compiled_imports(monkeypatch: pytest.MonkeyPatch):
    """Provide minimal sys.modules stubs so ``import nre.utils.types``
    succeeds without lietorch/ncore."""
    fake_lt = _typesmod.ModuleType("lietorch")

    class _FakeSE3:
        pass

    fake_lt.SE3 = _FakeSE3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lietorch", fake_lt)

    ncore_mod = _typesmod.ModuleType("ncore")
    ncore_data_mod = _typesmod.ModuleType("ncore.data")
    ncore_data_mod.ConcreteCameraModelParametersUnion = type(  # type: ignore[attr-defined]
        "CCMP", (), {}
    )
    ncore_data_mod.ConcreteLidarModelParametersUnion = type(  # type: ignore[attr-defined]
        "CLMP", (), {}
    )
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.data", ncore_data_mod)

    # Force a fresh import (drop any prior cached version).
    sys.modules.pop("nre.utils.types", None)


# ---------------------------------------------------------------------------
# HalfClosedInterval
# ---------------------------------------------------------------------------


def test_halfclosed_post_init_accepts_valid_interval():
    from nre.utils.types import HalfClosedInterval

    h = HalfClosedInterval(0, 10)
    assert h.start == 0
    assert h.end == 10


def test_halfclosed_post_init_accepts_empty_interval():
    """start == end is valid (the interval is just empty)."""
    from nre.utils.types import HalfClosedInterval

    h = HalfClosedInterval(5, 5)
    assert h.start == 5 and h.end == 5


def test_halfclosed_post_init_rejects_inverted_interval():
    from nre.utils.types import HalfClosedInterval

    with pytest.raises(AssertionError):
        HalfClosedInterval(10, 5)


def test_halfclosed_intersection_overlapping():
    from nre.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 10)
    b = HalfClosedInterval(5, 15)
    out = a.intersection(b)
    assert out is not None
    assert out.start == 5
    assert out.end == 10


def test_halfclosed_intersection_subset():
    from nre.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 100)
    b = HalfClosedInterval(20, 30)
    out = a.intersection(b)
    assert out is not None
    assert out.start == 20 and out.end == 30


def test_halfclosed_intersection_disjoint_other_to_the_right():
    """First branch: other.start >= self.end."""
    from nre.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 10)
    b = HalfClosedInterval(10, 20)
    assert a.intersection(b) is None  # touching at the half-open end → empty


def test_halfclosed_intersection_disjoint_other_to_the_left():
    """Second branch: other.end <= self.start."""
    from nre.utils.types import HalfClosedInterval

    a = HalfClosedInterval(10, 20)
    b = HalfClosedInterval(0, 10)
    assert a.intersection(b) is None  # touching at the closed start → empty


# ---------------------------------------------------------------------------
# FrameConversion
# ---------------------------------------------------------------------------


def test_frameconversion_post_init_accepts_identity():
    from nre.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    assert fc.target_scale == 1.0
    assert fc.dtype == np.float64


def test_frameconversion_post_init_rejects_wrong_shape():
    from nre.utils.types import FrameConversion

    with pytest.raises(AssertionError):
        FrameConversion(matrix=np.eye(3, dtype=np.float64))


def test_frameconversion_post_init_rejects_non_floating_dtype():
    from nre.utils.types import FrameConversion

    with pytest.raises(TypeError, match="floating point"):
        FrameConversion(matrix=np.eye(4, dtype=np.int32))


def test_frameconversion_post_init_rejects_non_positive_scale_entry():
    from nre.utils.types import FrameConversion

    bad = np.eye(4, dtype=np.float64)
    bad[3, 3] = 0.0
    with pytest.raises(AssertionError):
        FrameConversion(matrix=bad)


def test_frameconversion_post_init_rejects_non_rotation_3x3():
    """The (3,3) block must be a rotation (det == 1)."""
    from nre.utils.types import FrameConversion

    bad = np.eye(4, dtype=np.float64)
    bad[0, 0] = 2.0  # det = 2
    with pytest.raises(AssertionError):
        FrameConversion(matrix=bad)


def test_frameconversion_target_scale_inverse_of_bottomright():
    """target_scale = 1 / matrix[3,3]."""
    from nre.utils.types import FrameConversion

    m = np.eye(4, dtype=np.float64)
    m[3, 3] = 0.5  # i.e. source -> target scale = 2.0
    fc = FrameConversion(matrix=m)
    assert fc.target_scale == 2.0


def test_frameconversion_get_transformation_matrices_identity():
    from nre.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float32))
    T, S = fc.get_transformation_matrices()
    assert T.shape == (4, 4) and S.shape == (4, 4)
    assert T.dtype == np.float32 and S.dtype == np.float32
    np.testing.assert_allclose(T, np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(S, np.eye(4, dtype=np.float32))


def test_frameconversion_get_transformation_matrices_with_scale():
    """target_scale=2 means T scales the rotation block by 2 and S has 0.5
    on the diagonal of the leading 3x3 (1/s with s=2)."""
    from nre.utils.types import FrameConversion

    m = np.eye(4, dtype=np.float64)
    m[3, 3] = 0.5  # target_scale = 2
    fc = FrameConversion(matrix=m)
    T, S = fc.get_transformation_matrices()
    # T = m * target_scale = m * 2 → leading 3x3 is 2*I, but bottom-right is 1
    # (since m[3,3] * 2 = 1).
    assert T[0, 0] == 2.0 and T[1, 1] == 2.0 and T[2, 2] == 2.0
    assert T[3, 3] == 1.0
    # S has 1/s = 0.5 on first 3 diagonal entries, 1.0 on the 4th.
    assert S[0, 0] == 0.5 and S[1, 1] == 0.5 and S[2, 2] == 0.5
    assert S[3, 3] == 1.0


def test_frameconversion_transform_poses_singular_input():
    """Input is a single (4,4) pose; output is also (4,4)."""
    from nre.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    p = np.eye(4, dtype=np.float64)
    out = fc.transform_poses(p)
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, p)


def test_frameconversion_transform_poses_batched_input():
    """Input is (N,4,4); output is (N,4,4)."""
    from nre.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    p = np.tile(np.eye(4, dtype=np.float64)[None], (3, 1, 1))
    out = fc.transform_poses(p)
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out, p)


def test_frameconversion_transform_poses_casts_to_declared_dtype():
    """If conversion dtype is float32 and input is float64, output is float32."""
    from nre.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float32))
    p = np.eye(4, dtype=np.float64)
    out = fc.transform_poses(p)
    assert out.dtype == np.float32
