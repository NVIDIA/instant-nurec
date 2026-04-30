#!/usr/bin/env python
# mask_annotator_test.py - Combined non-GUI tests for Mask Annotator

from pathlib import Path

import pytest

from click.testing import CliRunner

from apps.avmask_annotator.mask_annotator import (
    check_and_download_checkpoints,
    download_file,
    run_mask_annotator,
)


# CLI tests
def test_help_option():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--help"])
    assert result.exit_code == 0
    assert "Options" in result.output


def test_version_option():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--version"])
    assert result.exit_code == 0
    assert "Mask Annotator v" or "Mask Annotator version" in result.output


# Utils tests
class DummyResponse:
    def __init__(self, content, headers):
        self._content = content
        self.headers = headers

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def test_download_file_success(tmp_path, monkeypatch):
    data = b"hello"
    headers = {"content-length": str(len(data))}
    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.requests.get",
        lambda url, stream=True: DummyResponse(data, headers),
    )
    dest = tmp_path / "out.bin"
    assert download_file("http://example.com", str(dest))
    assert dest.read_bytes() == data


def test_download_file_incomplete(tmp_path, monkeypatch):
    data = b"abc"
    headers = {"content-length": "5"}
    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.requests.get",
        lambda url, stream=True: DummyResponse(data, headers),
    )
    dest = tmp_path / "out.bin"
    assert not download_file("http://example.com", str(dest))


def test_download_file_exception(tmp_path, monkeypatch):
    def raise_error(url, stream=True):
        raise RuntimeError("network error")

    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.requests.get",
        raise_error,
    )
    dest = tmp_path / "out.bin"
    assert not download_file("http://example.com", str(dest))


# Test SAM2 checkpoint download logic
def test_check_and_download_checkpoints_existing(monkeypatch, capsys):
    # Simulate checkpoint already exists: download_file should not be called
    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.os.path.exists",
        lambda path: True,
    )
    called = False
    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.download_file",
        lambda url, dest: (_ for _ in ()).throw(AssertionError("download_file was called")),
    )
    # Should not raise
    check_and_download_checkpoints()


def test_check_and_download_checkpoints_missing(monkeypatch):
    # Simulate missing checkpoint and successful download
    def fake_exists(path):
        # checkpoint file endswith .pt
        return not path.endswith("sam2_hiera_large.pt")

    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.os.path.exists",
        fake_exists,
    )
    calls = []
    monkeypatch.setattr(
        "apps.avmask_annotator.mask_annotator.download_file",
        lambda url, dest: calls.append((url, dest)) or True,
    )
    check_and_download_checkpoints()
    assert calls, "download_file was not called when checkpoint missing"
    url, dest = calls[0]
    assert url.startswith("https://") and dest.endswith("sam2_hiera_large.pt")


# CLI option parsing errors
def test_invalid_window_size():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--window-size", "bad_format"])
    assert result.exit_code != 0
    assert "Invalid window size format" in result.output


def test_invalid_window_pos():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--window-pos", "bad_format"])
    assert result.exit_code != 0
    assert "Invalid window position format" in result.output
