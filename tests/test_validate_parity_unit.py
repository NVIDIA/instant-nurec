"""Direct-import unit tests for the helpers in scripts/validate_parity.py.

The existing tests/test_validate_parity.py exercises the script via
``subprocess`` (the actual user-facing surface). This file covers the
internal helpers directly so the coverage report reflects them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vp():
    """Direct import of scripts/validate_parity.py as a module so we can call
    its private helpers and verify them in coverage."""
    spec = importlib.util.spec_from_file_location(
        "validate_parity_module_under_test",
        str(REPO_ROOT / "scripts" / "validate_parity.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _load_tolerance
# ---------------------------------------------------------------------------


def test_load_tolerance_missing_file_returns_empty(vp, tmp_path):
    out = vp._load_tolerance(tmp_path / "nope.json")
    assert out == {}


def test_load_tolerance_existing_file_returns_dict(vp, tmp_path):
    p = tmp_path / "tol.json"
    p.write_text(json.dumps({"x": 1e-3, "y": 5e-2}))
    out = vp._load_tolerance(p)
    assert out == {"x": 1e-3, "y": 5e-2}


# ---------------------------------------------------------------------------
# _list_plys
# ---------------------------------------------------------------------------


def test_list_plys_returns_sorted_recursive(vp, tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    files = [tmp_path / "z.ply", tmp_path / "a" / "y.ply", tmp_path / "a" / "b" / "x.ply"]
    for f in files:
        f.write_bytes(b"")
    out = vp._list_plys(tmp_path)
    # Sorted lexicographically by full path
    assert out == sorted(files)


def test_list_plys_empty_dir_returns_empty(vp, tmp_path):
    assert vp._list_plys(tmp_path) == []


# ---------------------------------------------------------------------------
# _to_tensor
# ---------------------------------------------------------------------------


def test_to_tensor_float_dtype(vp):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = vp._to_tensor(arr, "f8")
    assert out.dtype == torch.float32  # float branch always casts to float32
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_uchar_dtype_promotes_to_int32(vp):
    arr = np.array([1, 2, 255], dtype=np.uint8)
    out = vp._to_tensor(arr, "u1")
    assert out.dtype == torch.int32


def test_to_tensor_other_int_dtype_promotes_to_int64(vp):
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = vp._to_tensor(arr, "i4")
    assert out.dtype == torch.int64


def test_to_tensor_unsupported_dtype_raises(vp):
    arr = np.array([1.0])
    with pytest.raises(ValueError, match="Unsupported"):
        vp._to_tensor(arr, "object")


# ---------------------------------------------------------------------------
# Helper to write minimal PLY files
# ---------------------------------------------------------------------------


def _write_ply(path: Path, n_vertex: int, *, perturb: float = 0.0) -> None:
    """Write a tiny PLY with x,y,z float + opacity float + uchar mask."""
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("road_mask", "u1"),
    ]
    arr = np.zeros(n_vertex, dtype=dtype)
    arr["x"] = np.linspace(0.0, 1.0, n_vertex) + perturb
    arr["y"] = np.linspace(2.0, 3.0, n_vertex) + perturb
    arr["z"] = np.linspace(4.0, 5.0, n_vertex)
    arr["opacity"] = 0.5
    arr["road_mask"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


# ---------------------------------------------------------------------------
# _read_ply
# ---------------------------------------------------------------------------


def test_read_ply_returns_count_and_props(vp, tmp_path):
    p = tmp_path / "a.ply"
    _write_ply(p, n_vertex=5)
    count, props = vp._read_ply(p)
    assert count == 5
    assert set(props.keys()) == {"x", "y", "z", "opacity", "road_mask"}
    # Each value is (dtype_str, tensor)
    for name, (dt, t) in props.items():
        assert isinstance(dt, str)
        assert t.shape == (5,)


def test_read_ply_no_vertex_element_raises(vp, tmp_path):
    """Forge a PLY with only a 'face' element — _read_ply should raise."""
    p = tmp_path / "noface.ply"
    arr = np.zeros(2, dtype=[("a", "i4"), ("b", "i4")])
    PlyData([PlyElement.describe(arr, "face")]).write(str(p))
    with pytest.raises(ValueError, match="no 'vertex' element"):
        vp._read_ply(p)


# ---------------------------------------------------------------------------
# _compare_pair
# ---------------------------------------------------------------------------


def test_compare_pair_identical_files_returns_no_errors(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=3)
    _write_ply(b, n_vertex=3)
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    assert errors == []


def test_compare_pair_vertex_count_within_tolerance_passes(vp, tmp_path):
    """Phase A torch swaps may drift vertex count by ≤ ``vertex_count_delta``.

    When counts differ but the absolute delta is within tolerance, the
    per-property comparison is skipped (no canonical 1:1 mapping for
    point clouds of different sizes) and the call returns no errors.
    """
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=10)
    _write_ply(b, n_vertex=12)
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=5)
    assert errors == []


def test_compare_pair_vertex_count_beyond_tolerance_fails(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=10)
    _write_ply(b, n_vertex=20)
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=5)
    assert len(errors) == 1
    assert "delta=10 > tolerance=5" in errors[0]


def test_compare_pair_vertex_count_mismatch_short_circuits(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=3)
    _write_ply(b, n_vertex=4)
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    assert len(errors) == 1
    assert "vertex count mismatch" in errors[0]


def test_compare_pair_property_name_mismatch_short_circuits(vp, tmp_path):
    """Write two PLYs with different property sets."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=2)
    # b has only x,y,z
    arr = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(b))
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    assert len(errors) == 1
    assert "property name mismatch" in errors[0]


def test_compare_pair_dtype_mismatch_records_error(vp, tmp_path):
    """Two PLYs with the same property name but different dtypes."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    arr_a = np.zeros(1, dtype=[("p", "f4")])
    arr_b = np.zeros(1, dtype=[("p", "f8")])
    PlyData([PlyElement.describe(arr_a, "vertex")]).write(str(a))
    PlyData([PlyElement.describe(arr_b, "vertex")]).write(str(b))
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    assert len(errors) == 1
    assert "dtype mismatch" in errors[0]


def test_compare_pair_diff_above_tol_records_error(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=3, perturb=0.0)
    _write_ply(b, n_vertex=3, perturb=0.5)  # x and y both shifted by 0.5
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    # 0.5 >> 1e-3 → diff exceeds tol on x and y
    assert len(errors) >= 1
    assert all("diff exceeds tolerance" in e for e in errors)


def test_compare_pair_diff_within_tol_passes(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=3, perturb=0.0)
    _write_ply(b, n_vertex=3, perturb=1e-5)
    errors = vp._compare_pair(a, b, tol={}, default_tol=1e-3, vertex_count_delta=0)
    assert errors == []


def test_compare_pair_per_property_tol_overrides_default(vp, tmp_path):
    """Properties listed in `tol` use that tolerance; others use default_tol."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=3, perturb=0.0)
    _write_ply(b, n_vertex=3, perturb=0.05)  # diff ~ 0.05
    # x has loose tol → passes; default_tol tight → other props (y) fail
    errors = vp._compare_pair(a, b, tol={"x": 1.0, "y": 1.0}, default_tol=1e-3, vertex_count_delta=0)
    # All float props that drift by 0.05 now have generous tols.
    # y is also perturbed via the helper, so it also passes; opacity/z don't drift.
    # z is unperturbed (only x and y get +perturb in our helper) — verify by asserting
    # any errors mention only the un-overridden numeric props.
    for e in errors:
        assert "x'" not in e
        assert "y'" not in e


# ---------------------------------------------------------------------------
# cmd_merge
# ---------------------------------------------------------------------------


def test_cmd_merge_baseline_missing_returns_1(vp, tmp_path, capsys):
    rc = vp.cmd_merge(
        baseline=tmp_path / "missing.ply",
        proposed=tmp_path / "exists.ply",
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "baseline file does not exist" in captured.err


def test_cmd_merge_proposed_missing_returns_1(vp, tmp_path, capsys):
    base = tmp_path / "a.ply"
    _write_ply(base, n_vertex=1)
    rc = vp.cmd_merge(
        baseline=base,
        proposed=tmp_path / "missing.ply",
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "proposed file does not exist" in captured.err


def test_cmd_merge_identical_files_return_0(vp, tmp_path, capsys):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=2)
    _write_ply(b, n_vertex=2)
    rc = vp.cmd_merge(
        baseline=a,
        proposed=b,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "PASS" in captured.out


def test_cmd_merge_diff_returns_1_with_fail_messages(vp, tmp_path, capsys):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=2)
    _write_ply(b, n_vertex=2, perturb=1.0)
    rc = vp.cmd_merge(
        baseline=a,
        proposed=b,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.err


# ---------------------------------------------------------------------------
# cmd_no_merge
# ---------------------------------------------------------------------------


def test_cmd_no_merge_baseline_dir_missing_returns_1(vp, tmp_path, capsys):
    rc = vp.cmd_no_merge(
        baseline_dir=tmp_path / "missing",
        proposed_dir=tmp_path,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "baseline dir does not exist" in captured.err


def test_cmd_no_merge_proposed_dir_missing_returns_1(vp, tmp_path, capsys):
    rc = vp.cmd_no_merge(
        baseline_dir=tmp_path,
        proposed_dir=tmp_path / "missing",
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "proposed dir does not exist" in captured.err


def test_cmd_no_merge_file_count_mismatch_returns_1(vp, tmp_path, capsys):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _write_ply(base / "0.ply", n_vertex=1)
    _write_ply(base / "1.ply", n_vertex=1)
    _write_ply(prop / "0.ply", n_vertex=1)
    rc = vp.cmd_no_merge(
        baseline_dir=base,
        proposed_dir=prop,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "file count mismatch" in captured.err


def test_cmd_no_merge_empty_dirs_return_1(vp, tmp_path, capsys):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    rc = vp.cmd_no_merge(
        baseline_dir=base,
        proposed_dir=prop,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "no PLY files" in captured.err


def test_cmd_no_merge_two_pairs_pass(vp, tmp_path, capsys):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _write_ply(base / "0.ply", n_vertex=1)
    _write_ply(base / "1.ply", n_vertex=1)
    _write_ply(prop / "0.ply", n_vertex=1)
    _write_ply(prop / "1.ply", n_vertex=1)
    rc = vp.cmd_no_merge(
        baseline_dir=base,
        proposed_dir=prop,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "PASS" in captured.out


def test_cmd_no_merge_per_pair_diff_records_error(vp, tmp_path, capsys):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _write_ply(base / "0.ply", n_vertex=2)
    _write_ply(prop / "0.ply", n_vertex=2, perturb=1.0)
    rc = vp.cmd_no_merge(
        baseline_dir=base,
        proposed_dir=prop,
        tol_path=tmp_path / "tol.json",
        default_tol=1e-3, vertex_count_delta=0,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.err


# ---------------------------------------------------------------------------
# main / argparse
# ---------------------------------------------------------------------------


def test_main_merge_subcommand(vp, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=1)
    _write_ply(b, n_vertex=1)
    rc = vp.main(["merge", str(a), str(b)])
    assert rc == 0


def test_main_no_merge_subcommand(vp, tmp_path):
    base = tmp_path / "base"
    prop = tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _write_ply(base / "0.ply", n_vertex=1)
    _write_ply(prop / "0.ply", n_vertex=1)
    rc = vp.main(["no_merge", str(base), str(prop)])
    assert rc == 0


def test_main_subcommand_required(vp):
    with pytest.raises(SystemExit) as excinfo:
        vp.main([])
    assert excinfo.value.code == 2  # argparse "missing required arg" exit


def test_main_unknown_subcommand_rejected(vp):
    with pytest.raises(SystemExit) as excinfo:
        vp.main(["bogus", "a", "b"])
    assert excinfo.value.code == 2


def test_main_with_explicit_tolerance_path(vp, tmp_path):
    """--tolerance-json overrides the default lookup path."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=1)
    _write_ply(b, n_vertex=1)
    tol = tmp_path / "tol.json"
    tol.write_text("{}")
    rc = vp.main(["--tolerance-json", str(tol), "merge", str(a), str(b)])
    assert rc == 0


def test_main_with_explicit_default_tolerance(vp, tmp_path):
    """--default-tolerance plumbs through to cmd_merge.

    We point --tolerance-json at a non-existent path so the JSON-derived
    per-property values can't override the default we're trying to test.
    """
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, n_vertex=1, perturb=0.0)
    _write_ply(b, n_vertex=1, perturb=0.4)  # 0.4 diff
    no_tol = tmp_path / "no_tol.json"  # does not exist → returns {}
    # Default tol 1e-3 → fail; loose default tol 1.0 → pass
    rc_tight = vp.main([
        "--tolerance-json", str(no_tol),
        "--default-tolerance", "1e-3",
        "merge", str(a), str(b),
    ])
    assert rc_tight == 1
    rc_loose = vp.main([
        "--tolerance-json", str(no_tol),
        "--default-tolerance", "1.0",
        "merge", str(a), str(b),
    ])
    assert rc_loose == 0


def test_default_tolerance_path_returns_repo_relative(vp):
    p = vp._default_tolerance_path()
    assert p.name == "tolerance.json"
    assert p.parent.name == "tests"
