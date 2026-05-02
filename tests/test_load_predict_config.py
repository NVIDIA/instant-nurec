"""Tests for instant_nurec.config.load_predict_config.

The loader is a thin shim that:

  1. holds the static portion of NRMConfig inline as a Python dict,
  2. resolves the pretrained checkpoint via the NGC registry,
  3. patches the CLI-derived fields (out_dir, ncore paths, merge toggle),
  4. validates the dict against ``NRMConfig``.

Steps (1), (3), (4) are pure / cheap — we exercise them directly. Step (2)
talks to NGC + filesystem, so we monkeypatch ``_resolve_pretrained_checkpoint``
to return a path to an empty ``.ckpt`` file we create under ``tmp_path``;
``NRMConfig.model_post_init`` only checks that the path exists, not that the
file is a valid torch checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


pytest.importorskip("yaml")
pytest.importorskip("pydantic")


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """Create an empty ``.ckpt`` file the ``NRMConfig`` post-init can stat."""
    ckpt = tmp_path / "kelvin_pa.ckpt"
    ckpt.write_bytes(b"")
    return ckpt


@pytest.fixture
def fake_ncore_root(tmp_path: Path) -> Path:
    """Create an ncore root with a debug.lst (the loader stats both)."""
    ncore = tmp_path / "ncore"
    ncore.mkdir()
    (ncore / "debug.lst").write_text("seq_a/meta.json\n")
    return ncore


@pytest.fixture(autouse=True)
def _patch_checkpoint_resolver(
    monkeypatch: pytest.MonkeyPatch, fake_checkpoint: Path
) -> Path:
    """Replace the registry round-trip with a stub that returns the tmp ckpt."""
    from instant_nurec import config as config_mod

    monkeypatch.setattr(
        config_mod, "_resolve_pretrained_checkpoint", lambda: str(fake_checkpoint)
    )
    return fake_checkpoint


def test_merge_enabled_propagates_to_predict_config(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=True,
    )
    assert cfg.predict.primitive_merge.enabled is True


def test_merge_disabled_propagates_to_predict_config(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg.predict.primitive_merge.enabled is False


def test_ncore_path_maps_to_base_and_list_paths(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg.dataset.predict is not None
    assert cfg.dataset.predict.ncore_json_base_path == str(fake_ncore_root)
    assert cfg.dataset.predict.ncore_json_list_path == str(
        fake_ncore_root / "debug.lst"
    )


def test_output_dir_maps_to_out_dir(fake_ncore_root: Path, tmp_path: Path):
    from instant_nurec.config import load_predict_config

    out = tmp_path / "out"
    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=out,
        merge_enabled=False,
    )
    assert cfg.out_dir == str(out)


def test_resume_path_is_pretrained_checkpoint_path(
    fake_ncore_root: Path, fake_checkpoint: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg.resume == str(fake_checkpoint)


def test_config_dir_is_auto_derived_from_out_dir_and_run_id(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    out = tmp_path / "out"
    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=out,
        merge_enabled=False,
    )
    assert cfg.config_dir == str(out / cfg.run_id / "config")


def test_run_ids_are_unique_per_invocation(fake_ncore_root: Path, tmp_path: Path):
    """Each load_predict_config() must mint a fresh run_id (shortuuid default_factory)."""
    from instant_nurec.config import load_predict_config

    cfg1 = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    cfg2 = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg1.run_id != cfg2.run_id


def test_missing_pretrained_checkpoint_raises_file_not_found(
    monkeypatch: pytest.MonkeyPatch, fake_ncore_root: Path, tmp_path: Path
):
    """If the resolver returns a path that does not exist, NRMConfig's
    post_init must raise FileNotFoundError (not silently accept the bogus path)."""
    from instant_nurec import config as config_mod

    monkeypatch.setattr(
        config_mod, "_resolve_pretrained_checkpoint", lambda: "/nope/missing.ckpt"
    )

    with pytest.raises(FileNotFoundError):
        config_mod.load_predict_config(
            ncore_path=fake_ncore_root,
            output_dir=tmp_path / "out",
            merge_enabled=False,
        )


def test_resolve_pretrained_checkpoint_invokes_create_model_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    """Cover the implementation of ``_resolve_pretrained_checkpoint`` itself by
    monkey-patching ``create_model_registry`` (the network-bound dep) to a
    captured-args stub. Verifies the function plumbs the configured
    URL+cache path through and returns whatever ``.get_model()`` yields.

    The autouse fixture above replaces ``_resolve_pretrained_checkpoint`` to
    avoid network during other tests; we reload the module to restore the
    original implementation, then rely on monkeypatch teardown to undo our
    own patches.
    """
    import importlib
    import instant_nurec.config as config_mod

    importlib.reload(config_mod)

    captured = {}

    class _FakeRegistry:
        def get_model(self):
            return "/cached/path/to/kelvin.ckpt"

    def _fake_create_model_registry(url, cache_dir):
        captured["url"] = url
        captured["cache_dir"] = cache_dir
        return _FakeRegistry()

    monkeypatch.setattr(config_mod, "create_model_registry", _fake_create_model_registry)

    out = config_mod._resolve_pretrained_checkpoint()
    assert out == "/cached/path/to/kelvin.ckpt"
    assert captured["url"] == config_mod._PRETRAINED_MODEL_URL
    assert captured["cache_dir"] == config_mod._PRETRAINED_CACHE_DIR
