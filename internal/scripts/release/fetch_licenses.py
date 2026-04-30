#!/usr/bin/env python3.11

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

"""
Script to fetch and concatenate license files from packages listed in a CSV.

Reads a CSV file with package information, downloads source archives,
extracts them, finds LICENSE files, and concatenates them into a single file.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request
from urllib.error import URLError


# Color constants
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"
NC = "\033[0m"  # No Color

# License categories
CATEGORY_OPEN_SOURCE = "open_source"
CATEGORY_PROPRIETARY = "proprietary"
CATEGORY_UNKNOWN = "unknown"

# Commercial use permissions
COMMERCIAL_YES = True  # Allows commercial use
COMMERCIAL_NO = False  # Does not allow commercial use
COMMERCIAL_UNKNOWN = None  # Unknown or needs review

# Consolidated license info: license_name -> (category, allows_commercial_use, url_or_none)
# This is the single source of truth for license categorization and URLs
LICENSE_INFO: Dict[str, Tuple[str, Optional[bool], Optional[str]]] = {
    # Open source licenses with downloadable templates - all allow commercial use
    "Apache": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/Apache-2.0.txt",
    ),
    "MIT": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt",
    ),
    "BSD": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/BSD-3-Clause.txt",
    ),
    "GPL": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,  # Allows commercial use but requires source disclosure
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/GPL-3.0-only.txt",
    ),
    "LGPL": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,  # Allows commercial use with linking exception
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/LGPL-3.0-only.txt",
    ),
    "AGPL": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,  # Allows commercial use but requires source for network use
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/AGPL-3.0-only.txt",
    ),
    "MPL": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MPL-2.0.txt",
    ),
    "ISC": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/ISC.txt",
    ),
    "Python": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/Python-2.0.txt",
    ),
    "PSF": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/Python-2.0.txt",
    ),
    "Zlib": (
        CATEGORY_OPEN_SOURCE,
        COMMERCIAL_YES,
        "https://raw.githubusercontent.com/spdx/license-list-data/main/text/Zlib.txt",
    ),
    # Open source licenses without standard templates (use OTHER file or package-specific)
    "Expat": (CATEGORY_OPEN_SOURCE, COMMERCIAL_YES, None),  # Expat is MIT-style
    "Dual": (CATEGORY_OPEN_SOURCE, COMMERCIAL_YES, None),  # Dual licenses typically allow commercial
    "Attribution": (CATEGORY_OPEN_SOURCE, COMMERCIAL_YES, None),  # Attribution licenses allow commercial
    "Sushi": (CATEGORY_OPEN_SOURCE, COMMERCIAL_YES, None),  # Custom open source
    # Proprietary licenses
    "NVIDIA": (CATEGORY_PROPRIETARY, COMMERCIAL_YES, None),  # NVIDIA EULA allows commercial use with restrictions
    # Unknown/undetected
    "LicenseRef": (CATEGORY_UNKNOWN, COMMERCIAL_UNKNOWN, None),  # Needs manual review
    "UNKNOWN": (CATEGORY_UNKNOWN, COMMERCIAL_UNKNOWN, None),
    "OTHER": (CATEGORY_UNKNOWN, COMMERCIAL_UNKNOWN, None),  # Has custom file, needs review
}

# Cache for generic license texts to avoid redundant downloads
_license_text_cache: Dict[str, str] = {}

# License overrides will be loaded from external TOML file
LICENSE_OVERRIDES: Dict[str, str] = {}

# Import shared HTTP utilities for safe response handling (CVE-2025-13836)
from internal.scripts.stream_read import stream_read


def load_license_overrides() -> Dict[str, str]:
    """
    Load license overrides from the TOML file.

    The overrides file has two sections:
    - user_overrides: Manually curated, take priority
    - auto_overrides: Auto-generated

    Returns:
        Dictionary mapping package names to license types (merged, user takes priority)
    """
    if tomllib is None:
        warning("Neither tomllib (Python 3.11+) nor tomli is available. License overrides will not be loaded.")
        warning("Install tomli with: pip install tomli")
        return {}

    # Try multiple locations for the overrides file
    possible_paths = [
        Path(__file__).parent / "license_overrides.toml",
    ]

    # When run via Bazel, also check workspace root
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        possible_paths.insert(0, Path(workspace_dir) / "internal/scripts/release/license_overrides.toml")

    for path in possible_paths:
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)

                # Handle sectioned format (user_overrides and auto_overrides)
                user_overrides = data.get("user_overrides", {})
                auto_overrides = data.get("auto_overrides", {})

                if user_overrides or auto_overrides:
                    # Merge with user taking priority
                    merged = {**auto_overrides, **user_overrides}
                    return merged

                # Legacy flat format - return all string values
                return {k: v for k, v in data.items() if isinstance(v, str)}
            except Exception as e:
                warning(f"Failed to load license overrides from {path}: {e}")

    return {}


def error(msg: str) -> None:
    """Print error message in red to stderr."""
    print(f"{RED}{msg}{NC}", file=sys.stderr)


def warning(msg: str) -> None:
    """Print warning message in yellow to stderr."""
    print(f"{YELLOW}{msg}{NC}", file=sys.stderr)


def is_valid_url(url: str) -> bool:
    """
    Validate that a URL has a supported scheme (http or https).

    Args:
        url: URL string to validate

    Returns:
        True if URL has http or https scheme, False otherwise
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def parse_csv(csv_path: str) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
    """
    Parse CSV file and extract package names, versions, licenses, and URLs.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (packages, versions, licenses, urls, stats)
        where stats is a dict with keys: 'from_syft', 'from_overrides', 'still_unknown'
    """
    packages = []
    versions = []
    licenses = []
    urls = []

    # Statistics
    from_csv = 0
    from_overrides = 0
    still_unknown = 0
    license_counts: Dict[str, int] = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Skip first three lines (headers)
            if i <= 2:
                continue

            if len(row) < 8:
                continue

            name = row[0]
            packages.append(name)

            # Column 1: version
            versions.append(row[1])

            # Column 2: license name (strip leading space, get first word)
            license_value = row[2].lstrip(" ")
            license_value = license_value.split()[0] if license_value.split() else ""

            if license_value and license_value.upper() not in ["UNKNOWN", "NOASSERTION"]:
                from_csv += 1
            elif name in LICENSE_OVERRIDES:
                license_value = LICENSE_OVERRIDES[name]
                from_overrides += 1
            else:
                still_unknown += 1

            licenses.append(license_value)

            # Count license types
            license_key = license_value if license_value else "UNKNOWN"
            license_counts[license_key] = license_counts.get(license_key, 0) + 1

            # Column 7: download URL
            urls.append(row[7])

    stats = {
        "from_syft": from_csv,  # Using same key for consistency (CSV is similar source)
        "from_overrides": from_overrides,
        "still_unknown": still_unknown,
        "license_counts": license_counts,
    }

    return packages, versions, licenses, urls, stats


def parse_spdx_json(json_path: str) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
    """
    Parse SPDX JSON file and extract package names, versions, licenses, and URLs.

    For SPDX files, URLs are typically git repos, so we mark them as empty
    to skip downloading and use generic license texts instead.

    When licenseDeclared is NOASSERTION, uses the LICENSE_OVERRIDES from the TOML file.

    Args:
        json_path: Path to the SPDX JSON file

    Returns:
        Tuple of (packages, versions, licenses, urls, stats)
        where stats is a dict with keys: 'from_syft', 'from_overrides', 'still_unknown'
    """
    packages = []
    versions = []
    licenses = []
    urls = []

    with open(json_path, "r") as f:
        data = json.load(f)

    # Statistics
    from_syft = 0  # Licenses detected by Syft (in SPDX file)
    from_overrides = 0  # Licenses from our TOML overrides
    still_unknown = 0  # Packages with no license info
    license_counts: Dict[str, int] = {}  # Count of each license type

    for package in data.get("packages", []):
        name = package.get("name", "unknown")
        version = package.get("versionInfo", "unknown")
        license_declared = package.get("licenseDeclared", "NOASSERTION")

        packages.append(name)
        versions.append(version)

        # Try normalized license first
        license_id = normalize_license_name(license_declared)

        if license_id != "UNKNOWN":
            # License was in the SPDX file (detected by Syft)
            from_syft += 1
        elif name in LICENSE_OVERRIDES:
            # Use override from TOML file
            license_id = LICENSE_OVERRIDES[name]
            from_overrides += 1
        else:
            # No license info available
            still_unknown += 1

        licenses.append(license_id)
        urls.append("")

        # Count license types
        license_counts[license_id] = license_counts.get(license_id, 0) + 1

    stats = {
        "from_syft": from_syft,
        "from_overrides": from_overrides,
        "still_unknown": still_unknown,
        "license_counts": license_counts,
    }

    return packages, versions, licenses, urls, stats


def normalize_license_name(license_str: str) -> str:
    """
    Normalize SPDX license string to a simple license name.
    """
    if not license_str or license_str in ["NOASSERTION", "NONE", ""]:
        return "UNKNOWN"

    # For LicenseRef-*, try to extract license info from the reference name
    # e.g., "LicenseRef-Apache-License--Version-2.0" -> check for "Apache"
    license_to_check = license_str
    if license_str.startswith("LicenseRef-"):
        # Remove the prefix and use the rest for pattern matching
        license_to_check = license_str[11:]  # Remove "LicenseRef-"

    # Handle common patterns
    license_upper = license_to_check.upper()

    if "MIT" in license_upper:
        return "MIT"
    elif "APACHE" in license_upper:
        return "Apache"
    elif "LGPL" in license_upper:
        return "LGPL"
    elif "AGPL" in license_upper:
        return "AGPL"
    elif "GPL" in license_upper:
        return "GPL"
    elif "BSD" in license_upper:
        return "BSD"
    elif "MPL" in license_upper or "MOZILLA" in license_upper:
        return "MPL"
    elif "ISC" in license_upper:
        return "ISC"
    elif "PYTHON" in license_upper:
        return "Python"
    elif "NVIDIA" in license_upper:
        return "NVIDIA"
    elif license_str.startswith("LicenseRef-"):
        # LicenseRef that didn't match any known pattern - treat as unknown
        return "UNKNOWN"
    else:
        # Return first word/token as fallback
        parts = license_str.replace("-", " ").split()
        if parts:
            return parts[0]
        return "UNKNOWN"


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file from URL to output_path.

    Args:
        url: URL to download from
        output_path: Path to save the file

    Returns:
        True if successful, False otherwise
    """
    # Validate URL format first
    if not is_valid_url(url):
        return False

    # Try using curl first (more reliable for GitLab and other tricky sites)
    try:
        result = subprocess.run(
            ["curl", "-L", "--fail", "-o", str(output_path), url],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            # Verify we didn't get HTML
            try:
                with open(output_path, "rb") as f:
                    first_bytes = f.read(200)
                    if first_bytes.startswith(b"<!") or b"<html" in first_bytes.lower():
                        error(f"Downloaded HTML instead of file from: {url}")
                        error(f"  -> This usually means the URL is incorrect or requires authentication")
                        output_path.unlink()  # Remove bad file
                        return False
            except Exception:
                pass  # If we can't read it, assume it's OK
            return True
        else:
            error(f"Failed to download: {url}")
            if result.stderr:
                error(f"  -> {result.stderr.strip()}")
            return False

    except FileNotFoundError:
        # curl not available, fall back to urllib
        try:
            if not is_valid_url(url):
                error(f"Unsupported URL scheme in: {url}")
                return False

            req = request.Request(
                url,
                headers={
                    "User-Agent": "curl/7.68.0",  # Mimic curl
                    "Accept": "*/*",
                },
            )
            with request.urlopen(req, timeout=300) as response:
                content = stream_read(response)

                # Check if we got HTML instead of a binary file
                if content.startswith(b"<!") or b"<html" in content[:200].lower():
                    error(f"Downloaded HTML instead of file from: {url}")
                    error(f"  -> This usually means the URL is incorrect or requires authentication")
                    return False

                with open(output_path, "wb") as out_file:
                    out_file.write(content)
            return True
        except (URLError, Exception) as e:
            error(f"Failed to download: {url}\n{e}")
            return False

    except Exception as e:
        error(f"Failed to download: {url}\n{e}")
        return False


def extract_archive(file_path: Path, output_dir: Path) -> bool:
    """
    Extract archive file to output directory.

    Args:
        file_path: Path to archive file
        output_dir: Directory to extract to

    Returns:
        True if extraction succeeded, False otherwise
    """
    file_str = str(file_path)

    try:
        if file_path.suffix == ".whl" or file_path.suffix == ".zip":
            print(f"Extracting wheel/zip: {file_path.name}")
            extract_dir = output_dir / f"{file_path.stem}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        elif file_path.suffix == ".deb":
            print(f"Extracting deb: {file_path.name}")
            # Validate it's actually a .deb file
            if not file_path.name.endswith(".deb"):
                error(f"File doesn't appear to be a .deb package: {file_path.name}")
                return False

            extract_dir = output_dir / f"{file_path.stem}_deb"
            extract_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["dpkg-deb", "-x", str(file_path), str(extract_dir)], capture_output=True, text=True
            )
            if result.returncode != 0:
                error(f"Failed to extract {file_path.name}: {result.stderr}")
                return False

        elif file_str.endswith(".tar.gz") or file_str.endswith(".tgz"):
            print(f"Extracting tar.gz/tgz: {file_path.name}")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(output_dir)

        elif file_str.endswith(".tar.bz2"):
            print(f"Extracting tar.bz2: {file_path.name}")
            with tarfile.open(file_path, "r:bz2") as tar:
                tar.extractall(output_dir)

        elif file_str.endswith(".tar.xz"):
            print(f"Extracting tar.xz: {file_path.name}")
            with tarfile.open(file_path, "r:xz") as tar:
                tar.extractall(output_dir)

        elif file_path.suffix == ".gz" and not file_str.endswith(".tar.gz"):
            print(f"Extracting gz: {file_path.name}")
            import gzip

            output_file = output_dir / file_path.stem
            with gzip.open(file_path, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif file_path.suffix == ".bz2" and not file_str.endswith(".tar.bz2"):
            print(f"Extracting bz2: {file_path.name}")
            import bz2

            output_file = output_dir / file_path.stem
            with bz2.open(file_path, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif file_path.suffix == ".xz" and not file_str.endswith(".tar.xz"):
            print(f"Extracting xz: {file_path.name}")
            import lzma

            output_file = output_dir / file_path.stem
            with lzma.open(file_path, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        else:
            error(f"Error: Unsupported file type: {file_path}")
            return False

        return True

    except (tarfile.ReadError, zipfile.BadZipFile, OSError, Exception) as e:
        error(f"Failed to extract {file_path.name}: {e}")
        # Check if file looks like HTML (common when download fails)
        try:
            with open(file_path, "rb") as f:
                first_bytes = f.read(100)
                if first_bytes.startswith(b"<!") or b"<html" in first_bytes.lower():
                    error(f"  -> File appears to be HTML (download may have failed)")
        except:
            pass
        return False


def find_license_file(directory: Path) -> Optional[Path]:
    """
    Find LICENSE, COPYRIGHT, or COPYING file in directory.

    Args:
        directory: Directory to search

    Returns:
        Path to license file if found, None otherwise
    """
    for base_name in ["LICENSE", "COPYRIGHT", "COPYING"]:
        # Case-insensitive search for files starting with base name
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.name.upper().startswith(base_name):
                return file_path
    return None


def get_generic_license_url(license_name: str) -> Optional[str]:
    """
    Get URL for generic license text.

    Args:
        license_name: Name of the license (e.g., 'Apache', 'MIT')

    Returns:
        URL to license text, or None if not found
    """
    # Use consolidated LICENSE_INFO
    info = LICENSE_INFO.get(license_name)
    if info:
        return info[2]  # Return URL (third element of tuple: category, commercial, url)
    return None


def find_other_license_file(package_name: str, version: str, licenses_dir: Path) -> Optional[Path]:
    """
    Find an "other" license file saved by update_license_overrides.py.

    These are license files that GitHub couldn't automatically identify.

    Args:
        package_name: Name of the package
        version: Version of the package
        licenses_dir: Directory where license files are stored

    Returns:
        Path to license file if found, None otherwise
    """
    package_name_safe = package_name.replace("/", "-").replace(" ", "_")
    version_safe = version.replace("/", "-").replace(" ", "_")
    license_filename = f"{package_name_safe}-{version_safe}-LICENSE-OTHER.txt"
    license_path = licenses_dir / license_filename

    if license_path.is_file():
        return license_path

    return None


def download_generic_license_text(license_name: str) -> Optional[str]:
    """
    Download and cache generic license text.

    Uses a cache to avoid redundant downloads of the same license type.
    Supports both URLs and local files (prefixed with "local:").

    Args:
        license_name: Name of the license (e.g., 'Apache', 'MIT')

    Returns:
        License text as string, or None if download failed
    """
    # Check cache first
    if license_name in _license_text_cache:
        return _license_text_cache[license_name]

    license_url = get_generic_license_url(license_name)
    if not license_url:
        return None

    # Handle local files (prefixed with "local:")
    if license_url.startswith("local:"):
        local_filename = license_url[6:]  # Remove "local:" prefix
        # Look in the licenses/ directory relative to this script
        script_dir = Path(__file__).parent
        workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if workspace_dir:
            licenses_dir = Path(workspace_dir) / "internal/scripts/release/licenses"
        else:
            licenses_dir = script_dir / "licenses"

        local_path = licenses_dir / local_filename
        if local_path.is_file():
            try:
                text = local_path.read_text()
                _license_text_cache[license_name] = text
                return text
            except Exception as e:
                warning(f"Failed to read local license file {local_path}: {e}")
                return None
        else:
            warning(f"Local license file not found: {local_path}")
            return None

    if not is_valid_url(license_url):
        warning(f"Unsupported URL scheme in: {license_url}")
        return None

    try:
        with request.urlopen(license_url, timeout=30) as response:
            text = stream_read(response).decode("utf-8")
            _license_text_cache[license_name] = text
            return text
    except Exception as e:
        warning(f"Failed to download generic license for {license_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and concatenate license files from packages listed in a CSV or SPDX JSON file."
    )
    parser.add_argument("input_file", help="Path to the CSV or SPDX JSON file with package information")
    parser.add_argument("output_dir", nargs="?", help="Output directory (optional, defaults to temp dir)")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if some downloads or extractions fail",
    )

    args = parser.parse_args()

    # Resolve input file path
    # When run via Bazel, resolve relative paths relative to BUILD_WORKSPACE_DIRECTORY
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if workspace_dir:
            input_path = Path(workspace_dir) / input_path

    # Validate input file
    if not input_path.is_file():
        error(f"File not found: {input_path}")
        # Also show what we tried if running via Bazel
        workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if workspace_dir:
            error(f"Note: When using bazel run, relative paths are resolved from workspace root: {workspace_dir}")
        sys.exit(1)

    # Load license overrides from TOML file
    global LICENSE_OVERRIDES
    LICENSE_OVERRIDES = load_license_overrides()
    print(f"Loaded {len(LICENSE_OVERRIDES)} license overrides")

    # Set up output directory
    cleanup_output_dir = False
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Resolve relative paths relative to workspace root when running via Bazel
        if not output_dir.is_absolute():
            workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
            if workspace_dir:
                output_dir = Path(workspace_dir) / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp())
        cleanup_output_dir = True

    try:
        # Detect file type and parse
        file_ext = input_path.suffix.lower()

        if file_ext == ".json":
            print(f"Detected SPDX JSON format")
            packages, versions, licenses, urls, license_stats = parse_spdx_json(str(input_path))
        elif file_ext == ".csv":
            print(f"Detected CSV format")
            packages, versions, licenses, urls, license_stats = parse_csv(str(input_path))
        else:
            error(f"Unsupported file format: {file_ext}")
            error("Supported formats: .csv, .json (SPDX)")
            sys.exit(1)

        print(f"Found {len(packages)} packages to process")
        print()

        # Download files (skip if no valid URLs)
        has_downloads = any(url for url in urls)
        if has_downloads:
            print("Downloading source archives...")
        else:
            print("No download URLs available (using license declarations only)")

        def download_single_file(url: str) -> Optional[str]:
            """Download a single file, return URL if failed."""
            if not url:
                return None

            filename = os.path.basename(url)
            output_path = output_dir / filename

            if output_path.exists():
                # Check if existing file is valid (not HTML error page)
                try:
                    with open(output_path, "rb") as f:
                        first_bytes = f.read(100)
                        if first_bytes.startswith(b"<!") or b"<html" in first_bytes.lower():
                            output_path.unlink()  # Delete corrupt file
                            if not download_file(url, output_path):
                                return url
                        else:
                            return None  # Already exists and valid
                except Exception:
                    if not download_file(url, output_path):
                        return url
            else:
                if not download_file(url, output_path):
                    return url

            return None

        # Download in parallel using thread pool
        failed_downloads = []
        urls_to_download = [url for url in urls if url]

        if urls_to_download:
            max_workers = min(32, len(urls_to_download))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_single_file, url): url for url in urls_to_download}

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 10 == 0 or completed == len(urls_to_download):
                        print(f"  Progress: {completed}/{len(urls_to_download)} files", end="\r")

                    failed_url = future.result()
                    if failed_url:
                        failed_downloads.append(failed_url)
                        if not args.continue_on_error:
                            executor.shutdown(wait=False, cancel_futures=True)
                            sys.exit(2)

                print()  # New line after progress

        if failed_downloads:
            warning(f"\n{len(failed_downloads)} download(s) failed:")
            for url in failed_downloads:
                warning(f"  - {url}")
            warning("")

        # Extract all archives in parallel (if any were downloaded)
        archive_files = [f for f in output_dir.iterdir() if f.is_file()]
        if archive_files:
            print()
            print("Extracting archives...")

            def extract_single_archive(file_path: Path) -> Optional[str]:
                """Extract a single archive, return filename if failed."""
                if extract_archive(file_path, output_dir):
                    return None
                return file_path.name

            failed_extractions = []
            max_workers = min(os.cpu_count() or 4, len(archive_files))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_single_archive, fp): fp for fp in archive_files}

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 5 == 0 or completed == len(archive_files):
                        print(f"  Progress: {completed}/{len(archive_files)} archives", end="\r")

                    failed_file = future.result()
                    if failed_file:
                        failed_extractions.append(failed_file)

                print()  # New line after progress

            if failed_extractions:
                warning(f"\nFailed to extract {len(failed_extractions)} file(s):")
                for filename in failed_extractions:
                    warning(f"  - {filename}")
                warning("")

        print()
        print("Finding licenses in extracted sources...")

        # Find license files in parallel
        def find_license_for_dir(index_and_dir):
            """Find license file in directory."""
            index, dir_path = index_and_dir
            if not dir_path.is_dir():
                return index, None, None

            license_file = find_license_file(dir_path)
            return index, dir_path, license_file

        license_paths = []
        missing_licenses = []
        missing_licenses_indices = []

        dir_paths = sorted(output_dir.iterdir())
        indexed_dirs = [(i, d) for i, d in enumerate(dir_paths) if d.is_dir()]

        if indexed_dirs:
            max_workers = min(os.cpu_count() or 4, len(indexed_dirs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(find_license_for_dir, item): item for item in indexed_dirs}

                results = []
                for future in as_completed(futures):
                    results.append(future.result())

                # Sort by index to maintain order
                results.sort(key=lambda x: x[0])

                for index, dir_path, license_file in results:
                    if license_file:
                        rel_path = license_file.relative_to(output_dir)
                        print(f"Found license: {rel_path}")
                        license_paths.append(license_file)
                    else:
                        missing_licenses.append(dir_path)
                        missing_licenses_indices.append(index)

        # For SPDX files where nothing was downloaded, all packages are "missing"
        if not has_downloads:
            print("No source archives downloaded - will use generic license texts for all packages")
            missing_licenses = [Path(f"{pkg}-{ver}") for pkg, ver in zip(packages, versions)]
            missing_licenses_indices = list(range(len(packages)))

        # Warn about missing licenses (different message for SPDX vs CSV)
        if missing_licenses and has_downloads:
            warning("")
            for idx, missing_dir in enumerate(missing_licenses):
                license_idx = missing_licenses_indices[idx]
                if license_idx < len(licenses):
                    warning(f"Warning: No LICENSE file found in {missing_dir.name}")
                    warning(f"  Declared license: {licenses[license_idx]}")
                    warning("")
            warning("Missing licenses will be substituted with generic license texts.")
            warning("")
        elif missing_licenses and not has_downloads:
            print()
            print(f"Using declared licenses from SPDX for {len(missing_licenses)} packages")

            # Show unique license types that will be downloaded
            unique_licenses = set()
            for idx in missing_licenses_indices:
                if idx < len(licenses):
                    lic = licenses[idx]
                    if get_generic_license_url(lic):
                        unique_licenses.add(lic)

            if unique_licenses:
                print(
                    f"Will download {len(unique_licenses)} unique license types: {', '.join(sorted(unique_licenses))}"
                )
            print()

        # Determine output directory for license files
        # When run via Bazel, use BUILD_WORKSPACE_DIRECTORY to get the actual workspace root
        workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if workspace_dir:
            licenses_dir = Path(workspace_dir) / "internal/scripts/release/licenses"
            concat_file = Path(workspace_dir) / "all_licenses_concatenated.txt"
            abridged_file = Path(workspace_dir) / "licenses_summary.txt"
        else:
            # Not run via bazel, use script directory
            script_dir = Path(__file__).parent
            licenses_dir = script_dir / "licenses"
            concat_file = script_dir / "all_licenses_concatenated.txt"
            abridged_file = script_dir / "licenses_summary.txt"

        # Create licenses directory
        licenses_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f"Saving individual license files to: {licenses_dir}")

        # Copy individual license files with meaningful names
        saved_license_paths = []
        for idx, license_path in enumerate(license_paths):
            if idx < len(packages):
                package_name = packages[idx].replace("/", "-").replace(" ", "_")
                version = versions[idx].replace("/", "-").replace(" ", "_") if idx < len(versions) else "unknown"
                # Get the original license filename extension
                license_ext = license_path.suffix if license_path.suffix else ".txt"
                saved_name = f"{package_name}-{version}-LICENSE{license_ext}"
                saved_path = licenses_dir / saved_name

                try:
                    shutil.copy2(license_path, saved_path)
                    saved_license_paths.append(saved_path)
                    print(f"  Saved: {saved_name}")
                except Exception as e:
                    warning(f"  Failed to save {saved_name}: {e}")
                    saved_license_paths.append(license_path)  # Use original if copy fails

        print()
        print("Creating concatenated license file...")

        with open(concat_file, "w") as out_file:
            # Write found licenses
            for idx, saved_license_path in enumerate(saved_license_paths):
                if idx < len(packages):
                    out_file.write(f"Package/Component: {packages[idx]}\n")
                if idx < len(versions):
                    out_file.write(f"Version: {versions[idx]}\n")
                out_file.write("License Text:\n")

                with open(saved_license_path, "r", errors="replace") as lic_file:
                    out_file.write(lic_file.read())

                out_file.write("\n")

            # Write missing licenses with generic substitutions
            missing_other_licenses = []  # Track OTHER licenses that couldn't be found
            for idx, missing_dir in enumerate(missing_licenses):
                license_idx = missing_licenses_indices[idx]

                if license_idx >= len(packages):
                    continue

                package = packages[license_idx]
                version = versions[license_idx] if license_idx < len(versions) else "unknown"
                declared_license = licenses[license_idx] if license_idx < len(licenses) else "unknown"

                # Save generic license to individual file
                package_name = package.replace("/", "-").replace(" ", "_")
                version_str = version.replace("/", "-").replace(" ", "_")
                saved_name = f"{package_name}-{version_str}-LICENSE-{declared_license}.txt"
                saved_path = licenses_dir / saved_name

                out_file.write(f"Package/Component: {package}\n")
                out_file.write(f"Version: {version}\n")

                license_text = None

                # For OTHER licenses, look for pre-saved license file
                if declared_license == "OTHER":
                    other_license_path = find_other_license_file(package, version, licenses_dir)
                    if other_license_path:
                        try:
                            with open(other_license_path, "r", errors="replace") as f:
                                license_text = f.read()
                            print(f"  Found other license: {other_license_path.name}")
                        except Exception as e:
                            warning(f"Failed to read license for {package}: {e}")

                # For standard licenses, download generic template
                if license_text is None and declared_license != "OTHER":
                    license_text = download_generic_license_text(declared_license)

                if license_text:
                    if declared_license == "OTHER":
                        out_file.write(f"License Text (other):\n")
                    else:
                        out_file.write(f"License Text (substituted - {declared_license}):\n")

                    # Save to individual file (skip for OTHER, already saved)
                    if declared_license != "OTHER":
                        with open(saved_path, "w") as lic_file:
                            lic_file.write(license_text)
                        print(f"  Saved (substituted): {saved_name}")

                    # Write to concatenated file
                    out_file.write(license_text)
                else:
                    out_file.write(f"License (no text available):\n")
                    out_file.write(f"Declared License: {declared_license}\n")
                    if declared_license == "OTHER":
                        missing_other_licenses.append(package)
                    else:
                        warning(f"No generic license template available for: {declared_license} (package: {package})")

                out_file.write("\n")
                out_file.write("=" * 80 + "\n")
                out_file.write("\n")

        # Create abridged summary file with package names and license types
        print()
        print("Creating abridged license summary file...")

        with open(abridged_file, "w") as summary_file:
            summary_file.write("License Summary\n")
            summary_file.write("=" * 60 + "\n")
            summary_file.write("\n")

            for idx in range(len(packages)):
                package = packages[idx]
                version = versions[idx] if idx < len(versions) else "unknown"
                license_name = licenses[idx] if idx < len(licenses) else "unknown"

                summary_file.write(f"{package} ({version}): {license_name}\n")

            summary_file.write("\n")
            summary_file.write("=" * 60 + "\n")
            summary_file.write(f"Total packages: {len(packages)}\n")

        print(f"  Saved: {abridged_file}")

        # Count packages with valid vs missing licenses
        packages_with_licenses = 0
        packages_without_licenses = 0

        for idx in missing_licenses_indices:
            if idx < len(licenses):
                license_type = licenses[idx]
                if get_generic_license_url(license_type):
                    packages_with_licenses += 1
                else:
                    packages_without_licenses += 1

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total packages processed: {len(packages)}")

        print()
        print("License sources:")
        print(f"  From SBOM (Syft):    {license_stats['from_syft']}")
        print(f"  From overrides:      {license_stats['from_overrides']}")
        print(f"  Still unknown:       {license_stats['still_unknown']}")

        print()
        print("License types:")
        license_counts = license_stats.get("license_counts", {})
        for license_type, count in sorted(license_counts.items(), key=lambda x: -x[1]):
            print(f"  {license_type}: {count}")

        # Categorize licenses using consolidated LICENSE_INFO
        open_source_count = 0
        proprietary_count = 0
        unknown_count = 0
        commercial_yes_count = 0
        commercial_no_count = 0
        commercial_unknown_count = 0

        for license_type, count in license_counts.items():
            info = LICENSE_INFO.get(license_type)
            if info:
                category = info[0]
                commercial = info[1]
                if category == CATEGORY_OPEN_SOURCE:
                    open_source_count += count
                elif category == CATEGORY_PROPRIETARY:
                    proprietary_count += count
                else:
                    unknown_count += count
                # Track commercial use permissions
                if commercial is True:
                    commercial_yes_count += count
                elif commercial is False:
                    commercial_no_count += count
                else:
                    commercial_unknown_count += count
            else:
                # Unrecognized license type - treat as unknown
                unknown_count += count
                commercial_unknown_count += count

        print()
        print("License categories:")
        print(f"  {GREEN}Open source:  {open_source_count}{NC}")
        print(f"  {YELLOW}Proprietary:  {proprietary_count}{NC}")
        print(f"  {RED}Unknown:      {unknown_count}{NC}")
        print()
        print("Commercial use:")
        print(f"  {GREEN}Allowed:      {commercial_yes_count}{NC}")
        print(f"  {RED}Not allowed:  {commercial_no_count}{NC}")
        print(f"  {YELLOW}Needs review: {commercial_unknown_count}{NC}")

        print()
        print(f"Licenses successfully retrieved: {packages_with_licenses}")
        print(f"Unique license types downloaded: {len(_license_text_cache)}")

        if packages_without_licenses > 0:
            print(f"{RED}\nPackages with missing/unknown licenses: {packages_without_licenses}")
            print(
                f"These had NOASSERTION, LicenseRef, or unsupported license types. They should be clarified in the license_overrides.toml file.{NC}"
            )

        print()
        print(f"Individual license files saved to: {licenses_dir}")
        print(f"All licenses concatenated in: {concat_file}")
        print(f"License summary (abridged): {abridged_file}")
        print("=" * 80)

        # Suggest running update_license_overrides if OTHER licenses are missing
        if missing_other_licenses:
            print()
            print(
                f"{YELLOW}WARNING: {len(missing_other_licenses)} package(s) have 'OTHER' license type but missing license files in cache:{NC}"
            )
            for pkg in missing_other_licenses[:5]:  # Show first 5
                print(f"  - {pkg}")
            if len(missing_other_licenses) > 5:
                print(f"  ... and {len(missing_other_licenses) - 5} more")
            print()
            print("To fetch these license files, run:")
            print(f"  bazel run //internal/scripts/release:update-license-overrides -- {input_path}")

    finally:
        # Clean up temp directory if needed
        if cleanup_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
