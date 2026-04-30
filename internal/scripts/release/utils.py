# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Git utilities for release scripts.

Provides common Git operations and Bazel sandbox handling for release management scripts.
"""

import json
import logging
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import git


# Configure logging
logger = logging.getLogger(__name__)


class GitError(Exception):
    """Exception for Git-related errors."""

    pass


class VersionError(Exception):
    """Exception for version-related errors."""

    pass


class ValidationError(Exception):
    """Exception for validation errors."""

    pass


@dataclass(frozen=True)
class SpdxPackage:
    """Minimal SPDX package record used for SBOM comparisons."""

    name: str
    version: str
    purl: Optional[str]
    download_location: Optional[str]


@dataclass(frozen=True)
class PurlInfo:
    """Parsed PURL components."""

    purl_type: str
    name: str
    version: str
    namespace: Optional[str]
    qualifiers: Dict[str, str]
    name_part: str


def validate_version_format(version: str) -> None:
    """Validate version string format: YY.MM (e.g., 25.09, 25.12)"""
    pattern = r"^[0-9]{2}\.(0[1-9]|1[0-2])$"
    if not re.match(pattern, version):
        raise VersionError(f"Invalid version format '{version}'\nExpected format: YY.MM (e.g., 25.09, 25.12)")


def validate_release_branch_format(branch_name: str) -> None:
    """Validate release branch format: release/YY.MM (e.g., release/25.09, release/25.12)"""
    pattern = r"^release/[0-9]{2}\.(0[1-9]|1[0-2])$"
    if not re.match(pattern, branch_name):
        raise VersionError(
            f"Invalid release branch format '{branch_name}'\n"
            f"Expected format: release/YY.MM (e.g., release/25.09, release/25.12)"
        )


def validate_image_tag_format(image_tag: str) -> None:
    """Validate source tag format: YY.M.PATCH-sha or YY.MM.PATCH-sha (e.g., 25.10.5-abc12345)"""
    pattern = r"^[0-9]{2}\.[0-9]{1,2}\.[0-9]+-[a-f0-9]+(\+dirty)?$"
    if not re.match(pattern, image_tag):
        raise ValidationError(
            f"Invalid source tag format\n"
            f"Expected format: YY.M.PATCH-sha or YY.MM.PATCH-sha (e.g., 25.10.5-abc12345)\n"
            f"Provided: {image_tag}"
        )


def version_update_commit_message(from_version: str, to_version: str) -> str:
    """Generate a version update commit message."""
    validate_version_format(from_version)
    validate_version_format(to_version)
    return f"Change: Update NuRec version {from_version} -> {to_version}"


def from_major_minor_to_version(major: int, minor: int) -> str:
    """Convert major and minor version numbers to version string format."""
    return f"{major:02d}.{minor:02d}"


def from_version_to_major_minor(version: str) -> Tuple[int, int]:
    """Convert version string to major and minor version numbers."""
    validate_version_format(version)
    parts = version.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version format: {version}")
    return int(parts[0]), int(parts[1])


def from_image_tag_to_version(image_tag: str) -> str:
    """Extract YY.MM from source tag (always returns zero-padded month format): YY.MM.PATCH-sha or YY.MM.PATCH-sha+dirty to YY.MM"""
    validate_image_tag_format(image_tag)
    match = re.match(r"^([0-9]{2})\.([0-9]{1,2})\.[0-9]+-.*$", image_tag)
    if not match:
        raise ValidationError(f"Could not extract version from source tag: {image_tag}")

    major = int(match.group(1))
    minor = int(match.group(2))

    return from_major_minor_to_version(major, minor)


def create_rc_image_tag(version: str, rc_number: int) -> str:
    """Create RC image tag from version and RC number: YY.MM-rcN (e.g., 25.10-rc1)"""
    validate_version_format(version)
    return f"{version}-rc{rc_number}"


def version_increment(current_version: str) -> str:
    """
    Calculate next version (increment by 1 month).

    Returns:
        formatted_version
    """
    current_major, current_minor = from_version_to_major_minor(current_version)
    new_minor = current_minor + 1
    new_major = current_major

    # Handle year rollover (minor > 12 -> reset to 01, increment major)
    if new_minor > 12:
        new_minor = 1
        new_major += 1

    return from_major_minor_to_version(new_major, new_minor)


def validate_version_sequence(from_version: str, to_version: str) -> None:
    """Validate that version sequence is correct (increment by 1)."""
    # Calculate expected next version
    expected_version = version_increment(from_version)
    if to_version != expected_version:
        raise VersionError(
            f"New version ({to_version}) should be the current version incremented by one ({expected_version})"
        )


def get_current_version(version_file_path: str = "bazel/version/VERSION_FILE") -> str:
    """
    Read current version from VERSION_FILE.

    Args:
        version_file_path: Path to the VERSION_FILE

    Returns:
        formatted_version
    """
    version_file = Path(version_file_path)

    if not version_file.exists():
        raise VersionError(f"VERSION_FILE not found at {version_file}")

    content = version_file.read_text()

    # Extract VERSION_MAJOR and VERSION_MINOR
    major_match = re.search(r"VERSION_MAJOR=(\d+)", content)
    minor_match = re.search(r"VERSION_MINOR=(\d+)", content)

    if not major_match or not minor_match:
        raise VersionError("Could not find VERSION_MAJOR or VERSION_MINOR in VERSION_FILE")

    major = int(major_match.group(1))
    minor = int(minor_match.group(1))
    formatted_version = from_major_minor_to_version(major, minor)

    return formatted_version


def get_gitlab_token(
    gitlab_url: str = "https://gitlab-master.nvidia.com",
    netrc_path: str = "~/.netrc",
) -> str:
    """Get GitLab token from a .netrc entry for the provided GitLab URL."""
    netrc_file = Path(netrc_path).expanduser()
    if not netrc_file.exists():
        raise GitError(f"GitLab token not found. Missing netrc file: {netrc_file}")

    host = re.sub(r"^https?://", "", gitlab_url).rstrip("/")
    content = netrc_file.read_text()
    pattern = rf"machine {re.escape(host)}\s+login oauth2\s+password ([^\s]+)"
    match = re.search(pattern, content, flags=re.MULTILINE)
    if not match:
        raise GitError(f"GitLab token not found in {netrc_file} for host: {host}")

    return match.group(1)


def get_gitlab_project_info(repo: git.Repo) -> tuple[str, str]:
    """Return (gitlab_base_url, project_path) from origin URL."""
    origin_url = repo.remote("origin").url
    if origin_url.startswith("http://") or origin_url.startswith("https://"):
        parsed = urllib.parse.urlparse(origin_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        project_path = parsed.path.lstrip("/")
    elif origin_url.startswith("ssh://"):
        parsed = urllib.parse.urlparse(origin_url)
        base_url = f"https://{parsed.hostname}"
        project_path = parsed.path.lstrip("/")
    else:
        match = re.match(r"git@([^:]+):(.+)$", origin_url)
        if not match:
            raise GitError(f"Unsupported origin URL format: {origin_url}")
        base_url = f"https://{match.group(1)}"
        project_path = match.group(2)

    if project_path.endswith(".git"):
        project_path = project_path[:-4]

    return base_url, project_path


def create_gitlab_branch(
    gitlab_base_url: str,
    project_path: str,
    token: str,
    branch_name: str,
    ref: str,
) -> None:
    """Create a GitLab branch via API."""
    project_id = urllib.parse.quote(project_path, safe="")
    api_url = f"{gitlab_base_url}/api/v4/projects/{project_id}/repository/branches"
    payload = urllib.parse.urlencode({"branch": branch_name, "ref": ref}).encode("utf-8")

    request = urllib.request.Request(api_url, data=payload, method="POST")
    request.add_header("PRIVATE-TOKEN", token)
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request) as response:
            response.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        if e.code == 400 and "already exists" in error_body.lower():
            logger.info("Release branch already exists on remote; skipping creation.")
            return
        raise GitError(f"Failed to create release branch via GitLab API (HTTP {e.code}). {error_body}") from e
    except urllib.error.URLError as e:
        raise GitError(f"Failed to create release branch via GitLab API: {e}") from e


def get_netrc_credentials(host: str, netrc_path: str = "~/.netrc") -> Tuple[str, str]:
    """Get login/password from a .netrc entry for the provided host."""
    netrc_file = Path(netrc_path).expanduser()
    if not netrc_file.exists():
        raise ValidationError(f"Netrc file not found: {netrc_file}")

    content = netrc_file.read_text()
    pattern = rf"machine {re.escape(host)}\s+login ([^\s]+)\s+password ([^\s]+)"
    match = re.search(pattern, content, flags=re.MULTILINE)
    if not match:
        raise ValidationError(f"Netrc entry not found for host: {host}")

    return match.group(1), match.group(2)


def get_registry_host(image_ref: str) -> Optional[str]:
    """Extract registry host from image reference if present."""
    if "/" not in image_ref:
        return None
    first = image_ref.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return None


_SYFT_IMAGE_DEFAULT = "anchore/syft:latest"


def run_syft_spdx(
    image_ref: str,
    output_path: Path,
    dry_run: bool = False,
    registry_host: Optional[str] = None,
    netrc_path: str = "~/.netrc",
    syft_image: str = _SYFT_IMAGE_DEFAULT,
) -> Path:
    """Generate SPDX SBOM by running the anchore/syft Docker container."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = str(output_path.parent)
    output_filename = output_path.name

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-v",
        f"{output_dir}:/output",
    ]

    if registry_host:
        username, password = get_netrc_credentials(registry_host, netrc_path=netrc_path)
        command.extend(
            [
                "-e",
                f"SYFT_REGISTRY_AUTH_AUTHORITY={registry_host}",
                "-e",
                f"SYFT_REGISTRY_AUTH_USERNAME={username}",
                "-e",
                f"SYFT_REGISTRY_AUTH_PASSWORD={password}",
            ]
        )

    command.extend(
        [
            syft_image,
            image_ref,
            "-o",
            f"spdx-json=/output/{output_filename}",
        ]
    )

    if dry_run:
        safe_cmd = list(command)
        for i, arg in enumerate(safe_cmd):
            if arg.startswith("SYFT_REGISTRY_AUTH_PASSWORD="):
                safe_cmd[i] = "SYFT_REGISTRY_AUTH_PASSWORD=***"
        logger.info(f"[DRY-RUN] Would pull image: {image_ref}")
        logger.info(f"[DRY-RUN] Would run: {' '.join(safe_cmd)}")
        return output_path

    logger.info(f"Generating SPDX SBOM for {image_ref} -> {output_path}")

    if not shutil.which("docker"):
        raise ValidationError(
            "Docker not found in PATH. Only Docker is required (Syft runs in a container); "
            "Syft does not need to be installed on the host."
        )
    logger.info(f"Pre-pulling image {image_ref} (uses host Docker credentials).")
    pull_result = subprocess.run(
        ["docker", "pull", image_ref],
        capture_output=True,
        text=True,
    )
    if pull_result.returncode != 0:
        err = (pull_result.stderr or pull_result.stdout or "").strip() or str(pull_result)
        raise ValidationError(
            f"Failed to pull image {image_ref}. Ensure you are logged in to the registry "
            f"(e.g. docker login gitlab-master.nvidia.com:5005). {err}"
        )
    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode != 0:
        err = (run_result.stderr or run_result.stdout or "").strip() or str(run_result)
        raise ValidationError(f"Syft failed for {image_ref}: {err}")

    return output_path


def read_spdx_packages(spdx_path: Path) -> List[SpdxPackage]:
    """Load SPDX JSON and return minimal package records."""
    spdx_path = spdx_path.expanduser().resolve()
    if not spdx_path.exists():
        raise ValidationError(f"SPDX file not found: {spdx_path}")

    data = json.loads(spdx_path.read_text())
    packages = []
    for pkg in data.get("packages", []):
        name = pkg.get("name") or ""
        version = pkg.get("versionInfo") or ""
        download_location = pkg.get("downloadLocation") or None
        purl = None
        for ref in pkg.get("externalRefs", []) or []:
            if ref.get("referenceType") == "purl" and ref.get("referenceLocator"):
                purl = ref.get("referenceLocator")
                break
        packages.append(
            SpdxPackage(
                name=name,
                version=version,
                purl=purl,
                download_location=download_location,
            )
        )
    return packages


def diff_packages(base_packages: Iterable[SpdxPackage], release_packages: Iterable[SpdxPackage]) -> List[SpdxPackage]:
    """Return packages present in release but not base."""
    base_keys: Set[Tuple[str, str, str]] = {(pkg.name.lower(), pkg.version, pkg.purl or "") for pkg in base_packages}
    diff = []
    for pkg in release_packages:
        key = (pkg.name.lower(), pkg.version, pkg.purl or "")
        if key not in base_keys:
            diff.append(pkg)
    return diff


def _normalize_download_location(download_location: Optional[str]) -> Optional[str]:
    if not download_location:
        return None
    if download_location.upper() in {"NOASSERTION", "NONE"}:
        return None
    if download_location.startswith("http://") or download_location.startswith("https://"):
        return download_location
    return None


def _parse_purl(purl: Optional[str]) -> Optional[PurlInfo]:
    if not purl or not purl.startswith("pkg:"):
        return None
    raw = purl[4:]
    purl_type, _, rest = raw.partition("/")
    qualifiers: Dict[str, str] = {}
    if "?" in rest:
        rest, qual_str = rest.split("?", 1)
        qualifiers = {k: v[0] for k, v in urllib.parse.parse_qs(qual_str).items()}
    if "@" in rest:
        name_part, version = rest.split("@", 1)
    else:
        name_part, version = rest, ""

    namespace = None
    name = name_part
    if purl_type == "deb" and "/" in name_part:
        namespace, name = name_part.split("/", 1)

    return PurlInfo(
        purl_type=purl_type,
        name=name,
        version=version,
        namespace=namespace,
        qualifiers=qualifiers,
        name_part=name_part,
    )


def _fetch_json(url: str) -> Dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        # On failure, return an empty dict so the script can continue gracefully
        return {}


def _pypi_sdist_url(package_name: str, version: str) -> Optional[str]:
    data = _fetch_json(f"https://pypi.org/pypi/{package_name}/{version}/json")
    releases_value = data.get("releases", {})
    if isinstance(releases_value, dict):
        releases = releases_value.get(version, [])
    else:
        releases = []
    if isinstance(releases, list):
        for entry in releases:
            if isinstance(entry, dict) and entry.get("packagetype") == "sdist" and entry.get("url"):
                return entry["url"]
    return None


def _deb_source_urls(package_name: str, version: str) -> List[str]:
    if not package_name:
        return []
    first_letter = package_name[0]
    return [
        f"https://launchpad.net/ubuntu/+archive/primary/+files/{package_name}_{version}.orig.tar.gz",
        f"https://launchpad.net/ubuntu/+archive/primary/+files/{package_name}_{version}.orig.tar.xz",
        f"https://snapshot.ubuntu.com/ubuntu/pool/main/{first_letter}/{package_name}/{package_name}_{version}.orig.tar.gz",
        f"https://snapshot.ubuntu.com/ubuntu/pool/main/{first_letter}/{package_name}/{package_name}_{version}.orig.tar.xz",
    ]


def _golang_proxy_url(module_path: str, version: str) -> str:
    quoted = urllib.parse.quote(module_path, safe="/")
    return f"https://proxy.golang.org/{quoted}/@v/{version}.zip"


def resolve_source_urls(package: SpdxPackage) -> List[str]:
    """Resolve candidate source URLs for a package."""
    download_location = _normalize_download_location(package.download_location)
    if download_location:
        return [download_location]

    purl_info = _parse_purl(package.purl)
    if not purl_info:
        return []

    purl_type = purl_info.purl_type
    name = purl_info.name
    version = purl_info.version
    name_part = purl_info.name_part

    if purl_type == "pypi" and name and version:
        url = _pypi_sdist_url(name, version)
        return [url] if url else []
    if purl_type == "deb" and name and version:
        return _deb_source_urls(name, version)
    if purl_type == "golang" and name_part and version:
        return [_golang_proxy_url(name_part, version)]

    return []
