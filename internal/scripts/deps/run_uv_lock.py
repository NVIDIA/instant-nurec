# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# Generates uv.lock (TOML format) at repo root for Black Duck / security scans.
# Uses the uv binary from Bazel (multitool). Run: bazel run //:generate_lock
# Python version is taken from the active interpreter (same as Docker/NRE when run via Bazel).

import os
import re
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

from python.runfiles import runfiles  # type: ignore[import-not-found]


def _get_nre_version(workspace: Path) -> str:
    """Return NRE version from Bazel version_string.sh, minus any '+dirty' suffix."""
    version_script = workspace / "bazel" / "version" / "version_string.sh"
    if not version_script.is_file():
        return "0.1.0"
    try:
        result = subprocess.run(
            [str(version_script)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return "0.1.0"
        # Remove everything after the first '-' (including the dash), as commit hash is considered invalid
        version = (result.stdout or "").strip().removesuffix("+dirty")
        if "-" in version:
            version = version.split("-", 1)[0]
        return version if version else "0.1.0"
    except (subprocess.TimeoutExpired, OSError):
        return "0.1.0"


def _find_uv(r) -> Path | None:
    """Resolve uv executable from runfiles (from @multitool//tools/uv)."""
    candidates = [
        "multitool/tools/uv/uv",
        "multitool/tools/uv",
        "rules_uv~0.88.0/tools/uv/uv",
        "_main/../multitool/tools/uv/uv",
    ]
    for runfiles_path in candidates:
        loc = r.Rlocation(runfiles_path)
        if loc:
            p = Path(loc)
            if p.is_file() and os.access(p, os.X_OK):
                return p
            if p.is_dir() and (p / "uv").exists() and os.access(p / "uv", os.X_OK):
                return p / "uv"
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        rdir = Path(runfiles_dir)
        for candidate in ["multitool/tools/uv/uv", "multitool/tools/uv"]:
            p = rdir / candidate.replace("/", os.sep)
            if p.is_file() and os.access(p, os.X_OK):
                return p
            if p.is_dir():
                uv_exe = p / "uv"
                if uv_exe.exists() and os.access(uv_exe, os.X_OK):
                    return uv_exe
        for d in ["multitool", "rules_uv~0.88.0", "rules_uv"]:
            tools_uv = rdir / d / "tools" / "uv"
            if tools_uv.is_dir():
                for exe in [tools_uv / "uv", tools_uv]:
                    if exe.is_file() and os.access(exe, os.X_OK):
                        return exe
    return None


def _parse_requirements_in(
    path: Path,
    deps_dir: Path,
    index_urls: list[str],
    deps: list[str],
    seen_includes: set[str],
) -> None:
    """Parse a .in file and append index URLs and dependency specs."""
    text = path.read_text()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("-r\t"):
            include = line[2:].strip().split("#")[0].strip()
            if include in seen_includes:
                continue
            seen_includes.add(include)
            included = deps_dir / include
            if not included.exists():
                included = path.parent / include
            if included.exists():
                _parse_requirements_in(included, deps_dir, index_urls, deps, seen_includes)
            continue
        if line.startswith("--extra-index-url"):
            url = (
                line.split("=", 1)[1].strip()
                if "=" in line
                else (line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else "")
            )
            if url and url not in index_urls:
                index_urls.append(url)
            continue
        if line.startswith("--index-url"):
            url = (
                line.split("=", 1)[1].strip()
                if "=" in line
                else (line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else "")
            )
            if url and url not in index_urls:
                index_urls.append(url)
            continue
        spec = line.split("#")[0].strip()
        spec = re.sub(r"\s*--hash=sha256:[a-fA-F0-9]+\s*", " ", spec).strip()
        if spec:
            deps.append(spec)


def main() -> int:
    r = runfiles.Create()
    uv_path = _find_uv(r)
    if uv_path is None:
        print("Error: Could not find the 'uv' binary in the runfiles or PATH.", file=sys.stderr)
        return 1

    workspace = Path(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()))
    deps_dir = workspace / "deps" / "python"
    req_in = deps_dir / "requirements_3_11_x86_64.in"
    if not req_in.exists():
        print(f"requirements_3_11_x86_64.in not found at {req_in}.", file=sys.stderr)
        return 1

    # Use the active interpreter's version (when run via bazel run, this is the toolchain Python).
    vi = sys.version_info
    requires_python = f"=={vi.major}.{vi.minor}.*"

    index_urls: list[str] = []
    deps: list[str] = []
    _parse_requirements_in(
        req_in,
        deps_dir,
        index_urls,
        deps,
        set(),
    )

    nre_version = _get_nre_version(workspace)

    with tempfile.TemporaryDirectory(prefix="uv-lock-project-") as tmpdir:
        project_dir = Path(tmpdir)

        index_entries = []
        for i, url in enumerate(index_urls):
            name = f"extra-{i}"
            auth = '\nauthenticate = "always"' if "gitlab" in url.lower() else ""
            index_entries.append(f'[[tool.uv.index]]\nname = "{name}"\nurl = "{url}"{auth}')
        index_toml = "\n".join(index_entries) if index_entries else ""

        deps_toml = ",\n    ".join(f'"{d}"' for d in deps)
        toml_content = f'''# Generated for uv lock. Do not edit.
[project]
name = "nre"
version = "{nre_version}"
requires-python = "{requires_python}"
dependencies = [
    {deps_toml}
]

[tool.uv]
managed = true
index-strategy = "unsafe-best-match"
{index_toml}
'''
        (project_dir / "pyproject.toml").write_text(toml_content, encoding="utf-8")

        uv_cmd = [
            str(uv_path),
            "lock",
            "--project",
            str(project_dir),
            "--python",
            sys.executable,
            "--index-strategy",
            "unsafe-best-match",
        ]
        for url in index_urls:
            uv_cmd.extend(["--extra-index-url", url])

        env = dict(os.environ)
        env["UV_INDEX_STRATEGY"] = "unsafe-best-match"

        print("uv command:", " ".join(uv_cmd), flush=True)

        result = subprocess.run(uv_cmd, cwd=workspace, env=env)
        if result.returncode != 0:
            return result.returncode

        lock_in_dir = project_dir / "uv.lock"
        lock_at_root = workspace / "uv.lock"
        if lock_in_dir.exists():
            shutil.move(str(lock_in_dir), str(lock_at_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
