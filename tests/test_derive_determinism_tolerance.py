"""Branch-coverage tests for scripts/derive_determinism_tolerance.py.

The full script main() walks ``baselines/more_baselines`` and writes
``tests/tolerance.json``; we don't reproduce that integration test here
(that would require shipping new baseline data). Instead we target the
three pure helpers — ``_read_props``, ``_glob_one``, ``_max_pair_diff`` —
because each owns one well-defined branch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.derive_determinism_tolerance import _glob_one, _max_pair_diff, _read_props


# ---------------------------------------------------------------------------
# helper to write tiny PLYs the helpers can chew on
# ---------------------------------------------------------------------------


def _write_ply(
    path: Path,
    *,
    floats: dict[str, np.ndarray] | None = None,
    uchars: dict[str, np.ndarray] | None = None,
) -> None:
    floats = floats or {}
    uchars = uchars or {}
    n = next(iter({**floats, **uchars}.values())).shape[0]
    dtype = []
    cols: list[np.ndarray] = []
    for name, arr in floats.items():
        dtype.append((name, "f4"))
        cols.append(arr.astype(np.float32))
    for name, arr in uchars.items():
        dtype.append((name, "u1"))
        cols.append(arr.astype(np.uint8))
    rec = np.empty(n, dtype=dtype)
    for (name, _), col in zip(dtype, cols):
        rec[name] = col
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(rec, "vertex")]).write(str(path))


# ---------------------------------------------------------------------------
# _read_props
# ---------------------------------------------------------------------------


def test_read_props_returns_float_dtype_as_float32_tensor(tmp_path: Path):
    p = tmp_path / "f.ply"
    _write_ply(p, floats={"x": np.array([1.0, 2.0, 3.0])})
    out = _read_props(p)
    assert "x" in out
    dtype, t = out["x"]
    assert dtype.startswith("f")
    assert t.dtype == torch.float32
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


def test_read_props_returns_uchar_dtype_as_int32_tensor(tmp_path: Path):
    """uchar (u1) is widened to int32 so per-property abs() doesn't overflow."""
    p = tmp_path / "u.ply"
    _write_ply(p, uchars={"red": np.array([0, 127, 255])})
    out = _read_props(p)
    dtype, t = out["red"]
    assert dtype == "u1"
    assert t.dtype == torch.int32
    assert torch.equal(t, torch.tensor([0, 127, 255], dtype=torch.int32))


def test_read_props_returns_multiple_properties_in_one_dict(tmp_path: Path):
    p = tmp_path / "mix.ply"
    _write_ply(p, floats={"x": np.array([1.0])}, uchars={"red": np.array([255])})
    out = _read_props(p)
    assert set(out.keys()) == {"x", "red"}


def test_read_props_widens_multibyte_int_dtypes_to_int64(tmp_path: Path):
    """A PLY with `u4` (unsigned int) or `i4` (signed int) properties must
    take the multi-byte int branch (line 47-48), widening to int64."""
    from plyfile import PlyData, PlyElement

    p = tmp_path / "ints.ply"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = np.empty(2, dtype=[("seq_id", "u4"), ("delta", "i4")])
    rec["seq_id"] = [10, 4_000_000_000 % (2**32)]  # large unsigned
    rec["delta"] = [-1, 2_000_000_000]
    PlyData([PlyElement.describe(rec, "vertex")]).write(str(p))

    out = _read_props(p)
    # val_dtype strings come straight from numpy: "u4"/"i4".
    assert out["seq_id"][0].startswith("u") and out["seq_id"][0] != "u1"
    assert out["seq_id"][1].dtype == torch.int64
    assert out["delta"][0].startswith("i")
    assert out["delta"][1].dtype == torch.int64
    # values round-trip through the int64 widen
    assert out["delta"][1].tolist() == [-1, 2_000_000_000]


def test_read_props_raises_on_unsupported_dtype(tmp_path: Path, monkeypatch):
    """Defensive ValueError branch (line 50): if the PLY has a property whose
    val_dtype isn't ``f*`` / ``u1`` / ``u*`` / ``i*``, _read_props must raise."""
    import scripts.derive_determinism_tolerance as ddt

    class _FakeProp:
        name = "weird"
        val_dtype = "z9"

    class _FakeVert:
        properties = (_FakeProp(),)

        def __getitem__(self, name):
            return np.array([0.0])

    class _FakePly:
        def __getitem__(self, name):
            return _FakeVert()

    monkeypatch.setattr(ddt.PlyData, "read", staticmethod(lambda _path: _FakePly()))
    with pytest.raises(ValueError, match="Unsupported dtype z9"):
        _read_props(tmp_path / "anything.ply")


# ---------------------------------------------------------------------------
# _glob_one
# ---------------------------------------------------------------------------


def _seed_run_dir(run_dir: Path, *, with_merge: bool, no_merge_chunks: int) -> None:
    """Build a minimal ``baselines/more_baselines/run_*/`` layout."""
    if with_merge:
        merge_dir = run_dir / "merge" / "uuidA" / "ply" / "scene"
        merge_dir.mkdir(parents=True)
        _write_ply(merge_dir / "scene.ply", floats={"x": np.array([0.0])})
    no_merge_dir = run_dir / "no_merge" / "uuidB" / "ply" / "scene"
    no_merge_dir.mkdir(parents=True)
    for i in range(no_merge_chunks):
        _write_ply(no_merge_dir / f"scene_chunk{i}.ply", floats={"x": np.array([float(i)])})


def test_glob_one_merge_returns_single_ply(tmp_path: Path):
    _seed_run_dir(tmp_path, with_merge=True, no_merge_chunks=0)
    out = _glob_one(tmp_path, "merge")
    assert out.name == "scene.ply"


def test_glob_one_merge_raises_when_zero_files(tmp_path: Path):
    _seed_run_dir(tmp_path, with_merge=False, no_merge_chunks=2)
    with pytest.raises(RuntimeError, match="expected 1 merge"):
        _glob_one(tmp_path, "merge")


def test_glob_one_merge_raises_when_multiple_files(tmp_path: Path):
    _seed_run_dir(tmp_path, with_merge=True, no_merge_chunks=0)
    extra = tmp_path / "merge" / "uuidA" / "ply" / "scene"
    _write_ply(extra / "scene2.ply", floats={"x": np.array([0.0])})
    with pytest.raises(RuntimeError, match="expected 1 merge"):
        _glob_one(tmp_path, "merge")


def test_glob_one_no_merge_returns_chunk_by_index(tmp_path: Path):
    _seed_run_dir(tmp_path, with_merge=False, no_merge_chunks=2)
    chunk0 = _glob_one(tmp_path, "no_merge", chunk_index=0)
    chunk1 = _glob_one(tmp_path, "no_merge", chunk_index=1)
    assert chunk0.name == "scene_chunk0.ply"
    assert chunk1.name == "scene_chunk1.ply"


def test_glob_one_no_merge_raises_on_chunk_index_out_of_range(tmp_path: Path):
    _seed_run_dir(tmp_path, with_merge=False, no_merge_chunks=1)
    with pytest.raises(RuntimeError, match="expected chunk index 5"):
        _glob_one(tmp_path, "no_merge", chunk_index=5)


def test_glob_one_no_merge_raises_when_chunk_index_none(tmp_path: Path):
    """chunk_index=None for no_merge mode should not silently match."""
    _seed_run_dir(tmp_path, with_merge=False, no_merge_chunks=2)
    with pytest.raises(RuntimeError):
        _glob_one(tmp_path, "no_merge", chunk_index=None)


def test_glob_one_unknown_mode_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        _glob_one(tmp_path, "neither")


# ---------------------------------------------------------------------------
# _max_pair_diff
# ---------------------------------------------------------------------------


def _props(values: dict[str, list[float]], *, dtype: str = "f4") -> dict[str, tuple[str, torch.Tensor]]:
    out: dict[str, tuple[str, torch.Tensor]] = {}
    for k, v in values.items():
        out[k] = (dtype, torch.tensor(v, dtype=torch.float32 if dtype.startswith("f") else torch.int32))
    return out


def test_max_pair_diff_identical_inputs_yield_zero():
    a = _props({"x": [1.0, 2.0]})
    b = _props({"x": [1.0, 2.0]})
    out = _max_pair_diff(a, b, prev={})
    assert out == {"x": 0.0}


def test_max_pair_diff_takes_running_max_against_prev():
    a = _props({"x": [1.0]})
    b = _props({"x": [4.0]})  # diff = 3
    out = _max_pair_diff(a, b, prev={"x": 1.0})  # prev tolerance was 1.0
    assert pytest.approx(out["x"], abs=1e-6) == 3.0


def test_max_pair_diff_keeps_prev_when_pair_diff_smaller():
    a = _props({"x": [1.0]})
    b = _props({"x": [1.5]})  # diff = 0.5
    out = _max_pair_diff(a, b, prev={"x": 5.0})  # prev was bigger
    assert out["x"] == 5.0


def test_max_pair_diff_creates_zero_entry_for_unseen_property():
    a = _props({"y": [1.0]})
    b = _props({"y": [1.0]})
    out = _max_pair_diff(a, b, prev={})
    assert "y" in out
    assert out["y"] == 0.0


def test_max_pair_diff_rejects_property_set_mismatch():
    a = _props({"x": [1.0]})
    b = _props({"y": [1.0]})
    with pytest.raises(RuntimeError, match="property sets differ"):
        _max_pair_diff(a, b, prev={})


def test_max_pair_diff_rejects_dtype_change():
    a = _props({"x": [1.0]}, dtype="f4")
    b = _props({"x": [1.0]}, dtype="i4")
    with pytest.raises(RuntimeError, match="dtype changed"):
        _max_pair_diff(a, b, prev={})


def test_max_pair_diff_rejects_shape_change():
    a = _props({"x": [1.0, 2.0]})
    b = _props({"x": [1.0]})
    with pytest.raises(RuntimeError, match="shape changed"):
        _max_pair_diff(a, b, prev={})


def test_max_pair_diff_does_not_mutate_prev():
    a = _props({"x": [1.0]})
    b = _props({"x": [4.0]})
    prev = {"x": 0.5}
    out = _max_pair_diff(a, b, prev=prev)
    # prev untouched; out is a fresh dict
    assert prev == {"x": 0.5}
    assert out is not prev


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _build_run_tree(tmp_path: Path, n_runs: int = 2, n_chunks: int = 1) -> Path:
    """Build a fake baselines/more_baselines tree under tmp_path."""
    base = tmp_path / "baselines" / "more_baselines"
    for i in range(n_runs):
        run_root = base / f"run_{i + 1}"
        # merge: 1 PLY
        merge_dir = run_root / "merge" / f"sess_{i}" / "ply" / f"pai_{i}"
        _write_ply(merge_dir / f"pai_{i}.ply", floats={"x": np.zeros(2)})
        # no_merge: n_chunks PLYs
        nm_dir = run_root / "no_merge" / f"sess_{i}" / "ply" / f"pai_{i}"
        for c in range(n_chunks):
            _write_ply(nm_dir / f"pai_{i}_chunk{c}.ply", floats={"x": np.zeros(2)})
    return tmp_path


def test_main_writes_tolerance_json(tmp_path, monkeypatch, capsys):
    """End-to-end: monkey-patch RUNS and OUT, run main(), verify tolerance.json
    is written with all expected property keys."""
    import scripts.derive_determinism_tolerance as ddt

    fake_repo = _build_run_tree(tmp_path, n_runs=3, n_chunks=2)
    runs = sorted((fake_repo / "baselines" / "more_baselines").glob("run_*/"))
    out = fake_repo / "tests" / "tolerance.json"
    monkeypatch.setattr(ddt, "RUNS", runs)
    monkeypatch.setattr(ddt, "OUT", out)
    monkeypatch.setattr(ddt, "REPO_ROOT", fake_repo)

    ddt.main()
    assert out.exists()
    data = json.loads(out.read_text())
    # Identical PLYs across runs → tolerance is 0 for x
    assert data == {"x": 0.0}


def test_main_with_drift_records_max_diff(tmp_path, monkeypatch):
    """Two runs with the same property but slightly different values — main()
    should record the larger diff into tolerance.json."""
    import scripts.derive_determinism_tolerance as ddt

    base = tmp_path / "baselines" / "more_baselines"
    # run_1: x = 0
    _write_ply(
        base / "run_1" / "merge" / "s" / "ply" / "p" / "p.ply", floats={"x": np.array([0.0, 0.0])}
    )
    _write_ply(
        base / "run_1" / "no_merge" / "s" / "ply" / "p" / "p_chunk0.ply",
        floats={"x": np.array([0.0])},
    )
    # run_2: x = 0.5
    _write_ply(
        base / "run_2" / "merge" / "s" / "ply" / "p" / "p.ply", floats={"x": np.array([0.0, 0.5])}
    )
    _write_ply(
        base / "run_2" / "no_merge" / "s" / "ply" / "p" / "p_chunk0.ply",
        floats={"x": np.array([0.5])},
    )

    runs = sorted(base.glob("run_*/"))
    out = tmp_path / "tests" / "tolerance.json"
    monkeypatch.setattr(ddt, "RUNS", runs)
    monkeypatch.setattr(ddt, "OUT", out)
    monkeypatch.setattr(ddt, "REPO_ROOT", tmp_path)
    ddt.main()
    data = json.loads(out.read_text())
    assert data["x"] == pytest.approx(0.5)


def test_main_no_merge_bumped_print_branch(tmp_path, monkeypatch, capsys):
    """The no_merge ``if bumped:`` branch (lines 133-137) only fires when a
    no_merge chunk-pair ratchets tolerance higher than the merge step did.
    Build identical merge PLYs (no merge ratchet) but differing no_merge
    chunks (forces a no_merge ratchet) and assert the print fires."""
    import scripts.derive_determinism_tolerance as ddt

    base = tmp_path / "baselines" / "more_baselines"

    # both runs: identical merge data → merge step does NOT ratchet on iter 1
    # because before == {} so 0.0 != prev.get(k, 0.0) is False ⇒ bumped is empty.
    for run_i in (1, 2):
        _write_ply(
            base / f"run_{run_i}" / "merge" / "s" / "ply" / "p" / "p.ply",
            floats={"x": np.array([0.0, 0.0])},
        )

    # run_1 no_merge x=0; run_2 no_merge x=0.7 → diff 0.7 > tolerance 0.0 ⇒ bumped fires
    _write_ply(
        base / "run_1" / "no_merge" / "s" / "ply" / "p" / "p_chunk0.ply",
        floats={"x": np.array([0.0])},
    )
    _write_ply(
        base / "run_2" / "no_merge" / "s" / "ply" / "p" / "p_chunk0.ply",
        floats={"x": np.array([0.7])},
    )

    runs = sorted(base.glob("run_*/"))
    monkeypatch.setattr(ddt, "RUNS", runs)
    monkeypatch.setattr(ddt, "OUT", tmp_path / "tol.json")
    monkeypatch.setattr(ddt, "REPO_ROOT", tmp_path)
    ddt.main()
    captured = capsys.readouterr().out
    # Bumped-print line for chunk0 / run_1 vs run_2 must appear.
    assert "chunk0 run_1 vs run_2: bumped" in captured
    assert "x:" in captured


def test_main_rejects_fewer_than_2_runs(tmp_path, monkeypatch):
    import scripts.derive_determinism_tolerance as ddt

    base = tmp_path / "baselines" / "more_baselines"
    _write_ply(
        base / "run_1" / "merge" / "s" / "ply" / "p" / "p.ply", floats={"x": np.zeros(1)}
    )
    runs = sorted(base.glob("run_*/"))  # only 1 run
    monkeypatch.setattr(ddt, "RUNS", runs)
    monkeypatch.setattr(ddt, "OUT", tmp_path / "tol.json")
    monkeypatch.setattr(ddt, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="need at least 2 runs"):
        ddt.main()
