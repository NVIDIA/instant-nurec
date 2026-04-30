#!/usr/bin/env python3.11

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

"""
Script to automatically update license_overrides.toml by querying external sources.

Queries:
- GitHub API for Go modules and GitHub-hosted packages
- Ubuntu/Debian package repositories for system packages
- PyPI for Python packages

It also saves licenses for packages that are detected by the SBOM generator but are not in the license_overrides.toml.

Usage:
    python update_license_overrides.py <spdx_file.json>
    python update_license_overrides.py <spdx_file.json> --dry-run
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import tomllib

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError


# Color constants
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"
NC = "\033[0m"  # No Color

# Rate limiting settings
GITHUB_REQUESTS_PER_HOUR = 60  # Unauthenticated limit
GITHUB_REQUEST_DELAY = 3600 / GITHUB_REQUESTS_PER_HOUR  # ~60 seconds between requests

# Parallelization settings
MAX_WORKERS_PYPI = 20  # PyPI is fast and doesn't rate limit aggressively
MAX_WORKERS_UBUNTU = 10  # Ubuntu changelogs can be slower
MAX_WORKERS_GITHUB = 5  # GitHub rate limits, so keep this lower

# Track API usage (thread-safe)
_github_requests_made = 0
_github_requests_lock = threading.Lock()  # Protects _github_requests_made
_github_rate_limited = threading.Event()  # Thread-safe flag for rate limiting
_github_has_token = False
_github_user_limit: Optional[int] = None
_github_auth_failed = threading.Event()  # Thread-safe flag for auth failures
_non_interactive = False  # Skip prompts and auto-continue

# Import shared HTTP utilities for safe response handling (CVE-2025-13836)
from internal.scripts.stream_read import stream_read


def _handle_rate_limit_exceeded(has_token: bool = False, user_set_limit: Optional[int] = None) -> None:
    """
    Handle GitHub API rate limit by prompting user for action.

    Args:
        has_token: Whether a GitHub token was provided
        user_set_limit: The --max-github-requests value if set by user
    """
    print()
    print("=" * 60)

    # If user set a limit and we hit it, that's expected behavior
    with _github_requests_lock:
        requests_made = _github_requests_made
    if user_set_limit is not None and requests_made >= user_set_limit:
        print(f"{YELLOW}GITHUB REQUEST LIMIT REACHED{NC}")
        print("=" * 60)
        print()
        print(f"Made {requests_made} requests (limit: {user_set_limit}).")
        print()
        print("Continuing with remaining packages using existing overrides...")
        _github_rate_limited.set()
        return

    # Actual rate limit from GitHub API
    print(f"{RED}GITHUB API RATE LIMIT EXCEEDED{NC}")
    print("=" * 60)
    print()
    print(f"{RED}Made {requests_made} requests before hitting the limit.{NC}")
    print()

    # Non-interactive mode: auto-continue
    if _non_interactive:
        print("Non-interactive mode: continuing without GitHub lookups...")
        _github_rate_limited.set()
        return

    if has_token:
        print("Even with a token, the rate limit (5000 req/hour) was exceeded.")
        print("This may happen with very large SBOMs.")
        print()
        print("Options:")
        print("  [w] Wait 1 hour and resume (rate limit resets hourly)")
        print("  [c] Continue without GitHub lookups (use existing overrides)")
        print("  [a] Abort and exit")
        print()

        while True:
            try:
                response = input("Choose [w/c/a]: ").strip().lower()
                if response == "w":
                    print("\nWaiting for rate limit to reset (1 hour)...")
                    print("Press Ctrl+C to abort.")
                    import time

                    time.sleep(3600)
                    print("Resuming...")
                    return  # Don't set rate_limited, allow retries
                elif response == "c":
                    print("\nContinuing without GitHub lookups...")
                    _github_rate_limited.set()
                    return
                elif response == "a":
                    print("\nAborting.")
                    sys.exit(1)
                else:
                    print("Invalid choice. Please enter 'w', 'c', or 'a'.")
            except (EOFError, KeyboardInterrupt):
                print("\nAborting.")
                sys.exit(1)
    else:
        print("Without a token, GitHub API is limited to 60 requests/hour.")
        print()
        print("Options:")
        print("  [c] Continue without GitHub lookups (use existing overrides)")
        print("  [a] Abort and exit")
        print()
        print("To avoid this, set a GitHub token (5000 requests/hour):")
        print("  export GITHUB_TOKEN=your_token")
        print()

        while True:
            try:
                response = input("Choose [c/a]: ").strip().lower()
                if response == "c":
                    print("\nContinuing without GitHub lookups...")
                    _github_rate_limited.set()
                    return
                elif response == "a":
                    print("\nAborting.")
                    sys.exit(1)
                else:
                    print("Invalid choice. Please enter 'c' or 'a'.")
            except (EOFError, KeyboardInterrupt):
                print("\nAborting.")
                sys.exit(1)


def _handle_invalid_token() -> None:
    """Handle GitHub API 401 error (invalid or expired token)."""
    print()
    print("=" * 60)
    print(f"{RED}GITHUB API AUTHENTICATION FAILED{NC}")
    print("=" * 60)
    print()
    print(f"{RED}The provided GitHub token is invalid or expired (HTTP 401).{NC}")
    print()
    print("To fix this:")
    print("  1. Generate a new token at https://github.com/settings/tokens")
    print("  2. Set it with:")
    print("       export GITHUB_TOKEN=your_new_token")
    print("     or pass it as a flag:")
    print("       --github-token=your_new_token")
    print()

    # Non-interactive mode: auto-continue
    if _non_interactive:
        print("Non-interactive mode: continuing without GitHub lookups...")
        _github_rate_limited.set()
        return

    try:
        response = input("Continue without GitHub lookups? [y/N]: ").strip().lower()
        if response == "y":
            print("\nContinuing without GitHub lookups...")
            _github_rate_limited.set()
            return
        else:
            print("\nAborting.")
            sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print("\nAborting.")
        sys.exit(1)


def normalize_license_name(license_str: str) -> str:
    """Normalize license string to a simple license name."""
    if not license_str or license_str.upper() in ["NOASSERTION", "NONE", "UNKNOWN", ""]:
        return ""

    license_upper = license_str.upper()

    # For long license texts, only check the first line/100 chars for the license type
    # This avoids false positives like "perMITted" in BSD license text
    first_part = license_upper[:100].split("\n")[0]

    # Check first part for specific license indicators (order matters - more specific first)
    if "BSD" in first_part:
        return "BSD"
    elif "APACHE" in first_part:
        return "Apache"
    elif "LGPL" in first_part:
        return "LGPL"
    elif "AGPL" in first_part:
        return "AGPL"
    elif "GPL" in first_part:
        return "GPL"
    elif re.search(r"\bMIT\b", first_part):  # Word boundary to avoid "perMITted"
        return "MIT"
    elif "MPL" in first_part or "MOZILLA" in first_part:
        return "MPL"
    elif "ISC" in first_part:
        return "ISC"
    elif "PYTHON" in first_part or "PSF" in first_part:
        return "Python"
    elif "ZLIB" in first_part:
        return "Zlib"
    elif "UNLICENSE" in first_part:
        return "Unlicense"
    elif "CC0" in first_part:
        return "CC0"
    elif "WTFPL" in first_part:
        return "WTFPL"

    # Fallback: check full string with word boundaries for MIT
    if re.search(r"\bMIT\b", license_upper):
        return "MIT"
    elif "APACHE" in license_upper:
        return "Apache"
    elif "BSD" in license_upper:
        return "BSD"
    elif "LGPL" in license_upper:
        return "LGPL"
    elif "GPL" in license_upper:
        return "GPL"
    else:
        # Return first meaningful word
        parts = license_str.replace("-", " ").split()
        if parts:
            return parts[0]
        return ""


def lookup_github_license(package_name: str, token: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Look up license from GitHub API for a github.com package.

    Returns:
        Tuple of (license_type, license_content).
        - If recognized: (normalized_license_name, None)
        - If unrecognized but file exists: ("OTHER", raw_license_content)
        - If not found: (None, None)
    """
    global _github_requests_made

    if _github_rate_limited.is_set():
        return None, None

    # Extract owner/repo from github.com/owner/repo[/subpath]
    parts = package_name.replace("github.com/", "").split("/")
    if len(parts) < 2:
        return None, None

    owner = parts[0]
    repo = parts[1]

    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/license"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "license-updater/1.0",
        }
        if token:
            headers["Authorization"] = f"token {token}"

        req = request.Request(url, headers=headers)

        # Add delay to avoid rate limiting (only for unauthenticated requests)
        if not token:
            time.sleep(0.1)  # Small delay to be nice to the API

        with request.urlopen(req, timeout=15) as response:
            with _github_requests_lock:
                _github_requests_made += 1
            data = json.loads(stream_read(response).decode("utf-8"))
            license_info = data.get("license", {})
            spdx_id = license_info.get("spdx_id", "")

            if spdx_id and spdx_id != "NOASSERTION":
                # Recognized license
                return normalize_license_name(spdx_id), None
            else:
                # License file exists but GitHub couldn't identify the type
                # Fetch the raw content - try download_url first, then content field
                download_url = data.get("download_url")
                content_b64 = data.get("content")
                encoding = data.get("encoding", "")

                # Try direct download first
                if download_url:
                    try:
                        raw_req = request.Request(
                            download_url,
                            headers={"User-Agent": "license-updater/1.0"},
                        )
                        with request.urlopen(raw_req, timeout=15) as raw_response:
                            content = stream_read(raw_response).decode("utf-8", errors="replace")
                            return "OTHER", content
                    except Exception as e:
                        print(f"  Failed to download raw license for {package_name}: {e}")

                # Fallback: try base64-encoded content from API response
                if content_b64 and encoding == "base64":
                    import base64

                    try:
                        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                        return "OTHER", content
                    except Exception as e:
                        print(f"  Failed to decode license content for {package_name}: {e}")

                # License exists but we couldn't get content
                return "OTHER", None

    except HTTPError as e:
        if e.code == 401:
            # Signal auth failure - will be handled in main thread
            _github_auth_failed.set()
            return "AUTH_FAILED", None
        elif e.code == 403:
            # Rate limit - signal for main thread to handle
            return "RATE_LIMITED", None
        elif e.code == 404:
            pass  # Repo doesn't exist or has no license
        else:
            print(f"  GitHub API error for {package_name}: {e.code}")
    except Exception as e:
        print(f"  Error looking up {package_name}: {e}")

    return None, None


def lookup_ubuntu_license(package_name: str, version: str = "") -> Tuple[Optional[str], Optional[str]]:
    """
    Look up license from Ubuntu/Debian package repositories.

    Returns:
        Tuple of (license_type, license_content).
        - If recognized: (normalized_license_name, None)
        - If found but unrecognized: ("OTHER", raw_content)
        - If not found: (None, None)
    """
    # Extract base package name (remove version info)
    base_name = package_name.split()[0] if " " in package_name else package_name

    # Try to derive source package name from binary package name
    # Common patterns: libfoo1 -> libfoo, libfoo-dev -> libfoo
    source_candidates = [base_name]
    if re.match(r"^lib.*\d+$", base_name):
        # libfoo1, libfoo2 -> libfoo
        source_candidates.append(re.sub(r"\d+$", "", base_name))
    if base_name.endswith("-dev"):
        source_candidates.append(base_name[:-4])
    if "-" in base_name:
        # libfoo-bar1 -> libfoo-bar, libfoo
        parts = base_name.rsplit("-", 1)
        if parts[0] not in source_candidates:
            source_candidates.append(parts[0])

    # Clean version string for URL (remove epoch like "2:")
    clean_version = re.sub(r"^\d+:", "", version) if version else ""

    # Try different URL patterns
    urls_to_try = []

    # Try binary package URL first
    urls_to_try.append(f"https://changelogs.ubuntu.com/changelogs/binary/{base_name[0]}/{base_name}/noble/copyright")

    # Try source package URLs with version
    if clean_version:
        for src in source_candidates:
            if src and len(src) > 0:
                # Debian/Ubuntu pool uses special prefix for lib* packages:
                # libfoo -> libf/libfoo, libx11 -> libx/libx11
                # Regular packages use first letter: systemd -> s/systemd
                if src.startswith("lib") and len(src) > 3:
                    pool_prefix = src[:4]  # e.g., "libx" for libx11
                else:
                    pool_prefix = src[0]
                # Ubuntu changelogs
                urls_to_try.append(
                    f"https://changelogs.ubuntu.com/changelogs/pool/main/{pool_prefix}/{src}/{src}_{clean_version}/copyright"
                )

    for url in urls_to_try:
        try:
            req = request.Request(url, headers={"User-Agent": "license-updater/1.0"})

            with request.urlopen(req, timeout=10) as response:
                content = stream_read(response).decode("utf-8", errors="replace")

                # Parse common license patterns in Debian copyright files
                license_patterns = [
                    (r"License:\s*(Apache[- ]2\.0|Apache)", "Apache"),
                    (r"License:\s*(MIT)", "MIT"),
                    (r"License:\s*(BSD[- ]?[23]?[- ]?[Cc]lause)", "BSD"),
                    (r"License:\s*(GPL[- ]?[23]?)", "GPL"),
                    (r"License:\s*(LGPL[- ]?[23]?)", "LGPL"),
                    (r"License:\s*(MPL[- ]?[12]\.?[0]?)", "MPL"),
                    (r"License:\s*(ISC)", "ISC"),
                    (r"License:\s*(Zlib)", "Zlib"),
                    (r"License:\s*(PSF|Python)", "Python"),
                ]

                for pattern, license_name in license_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return license_name, None

                # If we got content but couldn't identify the license, return as OTHER
                if len(content) > 100:  # Sanity check that we got real content
                    return "OTHER", content

        except HTTPError:
            continue  # Try next URL
        except Exception:
            continue

    return None, None


def lookup_pypi_license(package_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Look up license from PyPI JSON API.

    Returns:
        Tuple of (license_type, github_fallback_url).
        - If recognized: (normalized_license_name, None)
        - If not found but has GitHub URL: (None, github_url) - caller should queue GitHub lookup
        - If not found: (None, None)
    """
    # Normalize package name for PyPI
    pypi_name = package_name.replace("_", "-").lower()
    github_url = None

    try:
        url = f"https://pypi.org/pypi/{pypi_name}/json"
        req = request.Request(url, headers={"User-Agent": "license-updater/1.0"})

        with request.urlopen(req, timeout=10) as response:
            data = json.loads(stream_read(response).decode("utf-8"))
            info = data.get("info", {})

            # Try license field first
            license_str = info.get("license", "")
            if license_str and license_str.upper() not in ["UNKNOWN", "", "NOASSERTION", "NONE"]:
                normalized = normalize_license_name(license_str)
                if normalized:
                    return normalized, None

            # Try classifiers
            classifiers = info.get("classifiers", [])
            for classifier in classifiers:
                if classifier.startswith("License :: OSI Approved ::"):
                    license_part = classifier.replace("License :: OSI Approved :: ", "")
                    normalized = normalize_license_name(license_part)
                    if normalized:
                        return normalized, None

            # Extract GitHub URL for fallback (to be queued by caller)
            project_urls = info.get("project_urls", {}) or {}
            home_page = info.get("home_page", "") or ""

            # Check various URL fields for GitHub
            for url_name, url_value in project_urls.items():
                if url_value and "github.com" in url_value:
                    github_url = url_value
                    break

            if not github_url and home_page and "github.com" in home_page:
                github_url = home_page

    except HTTPError:
        pass  # Package not found on PyPI
    except Exception:
        pass

    # Return GitHub URL for caller to queue if we found one
    if github_url:
        # Extract and normalize GitHub URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)", github_url)
        if match:
            owner, repo = match.groups()
            repo = repo.rstrip(".git")  # Remove .git suffix if present
            return None, f"github.com/{owner}/{repo}"

    return None, None


def parse_spdx_json(json_path: str) -> List[Tuple[str, str, str]]:
    """Parse SPDX JSON file and extract packages with unknown/unrecognized licenses."""
    packages = []

    with open(json_path, "r") as f:
        data = json.load(f)

    for package in data.get("packages", []):
        name = package.get("name", "unknown")
        version = package.get("versionInfo", "unknown")
        license_declared = package.get("licenseDeclared", "NOASSERTION")

        # Include packages with unknown/missing licenses OR LicenseRef-* (unrecognized by Syft)
        if license_declared in ["NOASSERTION", "NONE", ""] or license_declared.startswith("LicenseRef-"):
            packages.append((name, version, license_declared))

    return packages


def load_existing_overrides(toml_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load existing license overrides from TOML file.

    Returns:
        Tuple of (user_overrides, auto_overrides)
    """
    if not toml_path.is_file():
        return {}, {}

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    user_overrides = data.get("user_overrides", {})
    auto_overrides = data.get("auto_overrides", {})

    # Handle legacy flat format - treat all as auto
    if not user_overrides and not auto_overrides:
        return {}, {k: v for k, v in data.items() if isinstance(v, str)}

    return user_overrides, auto_overrides


def save_overrides(
    toml_path: Path,
    user_overrides: Dict[str, str],
    auto_overrides: Dict[str, str],
) -> None:
    """Save license overrides to TOML file with separate sections."""
    with open(toml_path, "w") as f:
        f.write("# License overrides for packages where the SBOM generator couldn't detect the license.\n")
        f.write("#\n")
        f.write("# This file has two sections:\n")
        f.write("# 1. [user_overrides] - Manually curated entries that take priority.\n")
        f.write("#    The update script will NEVER modify these, but will report if detected licenses differ.\n")
        f.write("# 2. [auto_overrides] - Auto-generated by update_license_overrides.py.\n")
        f.write("#    These are overwritten on each run.\n")
        f.write("#\n")
        f.write('# Format: "package-name" = "license-type"\n')
        f.write("\n")

        # Write user overrides (preserved exactly)
        f.write("[user_overrides]\n")
        f.write("# Add manual overrides here - these take priority and won't be auto-modified\n")
        if user_overrides:
            f.write("\n")
            for name, license_type in sorted(user_overrides.items()):
                f.write(f'"{name}" = "{license_type}"\n')
        f.write("\n")

        # Write auto overrides (grouped by category)
        f.write("[auto_overrides]\n")
        f.write("# Auto-generated section - DO NOT EDIT MANUALLY\n")

        if auto_overrides:
            go_modules = {}
            go_stdlib = {}
            google_packages = {}
            python_packages = {}
            ubuntu_packages = {}

            for name, license_type in sorted(auto_overrides.items()):
                if name.startswith("github.com/"):
                    go_modules[name] = license_type
                elif name.startswith("golang.org/x/"):
                    go_stdlib[name] = license_type
                elif name.startswith("google.golang.org/"):
                    google_packages[name] = license_type
                elif "/" not in name and not name.startswith("lib"):
                    python_packages[name] = license_type
                else:
                    ubuntu_packages[name] = license_type

            def write_group(comment: str, packages: Dict[str, str]) -> None:
                if packages:
                    f.write(f"\n# {comment}\n")
                    for pkg_name, lic_type in sorted(packages.items()):
                        f.write(f'"{pkg_name}" = "{lic_type}"\n')

            write_group("Go modules", go_modules)
            write_group("Go standard library extensions", go_stdlib)
            write_group("Google packages", google_packages)
            write_group("Python packages", python_packages)
            write_group("Ubuntu/system packages", ubuntu_packages)


def main():
    parser = argparse.ArgumentParser(description="Update license_overrides.toml by querying external sources.")
    parser.add_argument("spdx_file", help="Path to SPDX JSON file with package information")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing to file",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub API token for higher rate limits (5000 req/hour vs 60)",
    )
    parser.add_argument(
        "--max-github-requests",
        type=int,
        default=None,
        help="Maximum GitHub API requests to make (default: unlimited)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive mode: auto-continue on rate limits/auth failures without prompting",
    )

    args = parser.parse_args()

    # Also check environment variable for GitHub token
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")

    # Set global config for rate limit handling
    global _github_has_token, _github_user_limit, _non_interactive
    _github_has_token = github_token is not None
    _github_user_limit = args.max_github_requests
    _non_interactive = args.yes

    # Warn if no token found
    if not github_token:
        print(f"{YELLOW}WARNING: No GITHUB_TOKEN found.{NC}")
        print("Without a token, GitHub API is limited to 60 requests/hour.")
        print("With a token, you get 5000 requests/hour.")
        print()
        print("To set a token:")
        print("  export GITHUB_TOKEN=your_token")
        print("  # or pass --github-token=your_token")
        print()
        try:
            response = input("Continue without token? [y/N]: ").strip().lower()
            if response != "y":
                print("Aborted.")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        print()

    # Resolve paths
    # When run via Bazel, use BUILD_WORKSPACE_DIRECTORY for output paths
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        script_dir = Path(workspace_dir) / "internal/scripts/release"
    else:
        script_dir = Path(__file__).parent

    spdx_path = Path(args.spdx_file)
    if not spdx_path.is_absolute():
        if workspace_dir:
            spdx_path = Path(workspace_dir) / spdx_path
        else:
            spdx_path = Path(".") / spdx_path

    if not spdx_path.is_file():
        print(f"{RED}Error: SPDX file not found: {spdx_path}{NC}")
        sys.exit(1)

    # Find the overrides file
    toml_path = script_dir / "license_overrides.toml"

    # Load existing overrides (separate user and auto sections)
    user_overrides, auto_overrides = load_existing_overrides(toml_path)
    all_existing = {**auto_overrides, **user_overrides}  # User takes priority
    print(f"Loaded {len(user_overrides)} user overrides and {len(auto_overrides)} auto overrides from {toml_path}")

    # Parse SPDX file for packages with unknown licenses
    unknown_packages = parse_spdx_json(str(spdx_path))
    print(f"Found {len(unknown_packages)} packages with unknown/unrecognized licenses")

    # Look up ALL packages to verify/update overrides
    packages_to_lookup = [(name, version) for name, version, _ in unknown_packages]
    print(f"Will look up {len(packages_to_lookup)} packages (including existing overrides for verification)")

    if not packages_to_lookup:
        print("No packages to look up!")
        return

    # Look up licenses
    new_auto_overrides = {}  # Will replace auto_overrides section
    user_inconsistencies = {}  # Track where detected license differs from user override
    corrections = {}  # Track where our lookup differs from auto override
    new_finds = {}  # Track newly found licenses
    not_found = []  # Track packages where lookup failed
    custom_licenses = {}  # Track custom license content: name -> (version, content)
    max_github = args.max_github_requests  # None means unlimited

    # Determine licenses directory
    licenses_dir = script_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)

    # Categorize packages by lookup source for parallel processing
    github_packages = []
    golang_packages = []
    google_packages = []
    pypi_packages = []
    ubuntu_packages = []

    for name, version in packages_to_lookup:
        if name.startswith("github.com/"):
            github_packages.append((name, version))
        elif name.startswith("golang.org/x/"):
            golang_packages.append((name, version))
        elif name.startswith("google.golang.org/"):
            google_packages.append((name, version))
        elif "/" not in name and not name.startswith("lib"):
            # Check if version looks like a deb package (contains ubuntu, build, dfsg, etc.)
            # Also check for common system package name patterns
            deb_version_markers = ["ubuntu", "build", "dfsg", "deb"]
            deb_name_suffixes = ["-utils", "-common", "-data", "-dev", "-bin", "-base", "-core", "-tools"]
            is_deb_version = any(marker in version.lower() for marker in deb_version_markers)
            is_deb_name = any(name.endswith(suffix) for suffix in deb_name_suffixes)
            if is_deb_version or is_deb_name:
                ubuntu_packages.append((name, version))
            else:
                pypi_packages.append((name, version))
        elif name.startswith("lib") or any(c.isdigit() for c in name.split("-")[0] if name.split("-")):
            ubuntu_packages.append((name, version))

    # Results storage: name -> (license_type, license_content, source)
    lookup_results: Dict[str, Tuple[Optional[str], Optional[str], str]] = {}
    progress_lock = threading.Lock()
    completed = [0]  # Use list to allow mutation in nested function

    def update_progress(batch_name: str, batch_total: int):
        with progress_lock:
            completed[0] += 1
            current = completed[0]
        if current % 20 == 0 or current == batch_total:
            print(f"    {batch_name}: {current}/{batch_total}", end="\r", flush=True)

    # Parallel PyPI lookups (collect GitHub fallback URLs for later)
    pypi_github_fallbacks = []  # (original_name, version, github_package_name)
    if pypi_packages:
        print(f"  Looking up {len(pypi_packages)} PyPI packages...", flush=True)
        completed[0] = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_PYPI) as executor:
            futures = {executor.submit(lookup_pypi_license, name): (name, version) for name, version in pypi_packages}
            for future in as_completed(futures):
                name, version = futures[future]
                try:
                    license_type, github_fallback = future.result()
                    if license_type:
                        lookup_results[name] = (license_type, None, "PyPI")
                    elif github_fallback:
                        # Queue for GitHub lookup
                        pypi_github_fallbacks.append((name, version, github_fallback))
                    else:
                        lookup_results[name] = (None, None, "PyPI")
                except Exception:
                    lookup_results[name] = (None, None, "PyPI")
                update_progress("PyPI", len(pypi_packages))
        print()  # Newline after progress
        if pypi_github_fallbacks:
            print(f"    {len(pypi_github_fallbacks)} packages need GitHub fallback lookup")

    # Parallel Ubuntu lookups
    if ubuntu_packages:
        print(f"  Looking up {len(ubuntu_packages)} Ubuntu packages...", flush=True)
        completed[0] = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_UBUNTU) as executor:
            futures = {
                executor.submit(lookup_ubuntu_license, name, version): (name, version)
                for name, version in ubuntu_packages
            }
            for future in as_completed(futures):
                name, version = futures[future]
                try:
                    license_type, license_content = future.result()
                    lookup_results[name] = (license_type, license_content, "Ubuntu")
                except Exception:
                    lookup_results[name] = (None, None, "Ubuntu")
                update_progress("Ubuntu", len(ubuntu_packages))
        print()  # Newline after progress

    # GitHub lookups with controlled concurrency (rate limited)
    all_github_packages = []
    for name, version in github_packages:
        all_github_packages.append((name, version, name))
    for name, version in golang_packages:
        github_name = "github.com/golang/" + name.split("/")[-1]
        all_github_packages.append((name, version, github_name))
    for name, version in google_packages:
        pkg_name = name.replace("google.golang.org/", "")
        # Try googleapis first, then google
        all_github_packages.append((name, version, f"github.com/googleapis/{pkg_name}"))
    # Add PyPI packages that need GitHub fallback
    for name, version, github_name in pypi_github_fallbacks:
        all_github_packages.append((name, version, github_name))

    if all_github_packages:
        # Apply max_github limit
        if max_github is not None:
            all_github_packages = all_github_packages[:max_github]
        print(f"  Looking up {len(all_github_packages)} GitHub packages...", flush=True)
        completed[0] = 0
        github_total = len(all_github_packages)

        # Reset auth failure flag
        _github_auth_failed.clear()

        github_count = 0
        auth_failed = False
        rate_limited = False

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_GITHUB) as executor:
            futures = {
                executor.submit(lookup_github_license, github_name, github_token): (name, version, github_name)
                for name, version, github_name in all_github_packages
            }
            for future in as_completed(futures):
                # Check if auth failed in another thread - cancel remaining
                if _github_auth_failed.is_set() and not auth_failed:
                    auth_failed = True
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()

                name, version, github_name = futures[future]
                github_count += 1
                try:
                    license_type, license_content = future.result()

                    # Handle special return values
                    if license_type == "AUTH_FAILED":
                        auth_failed = True
                        lookup_results[name] = (None, None, "GitHub")
                        continue
                    elif license_type == "RATE_LIMITED":
                        rate_limited = True
                        lookup_results[name] = (None, None, "GitHub")
                        continue

                    # For google.golang.org packages, try fallback if first attempt failed
                    if license_type is None and name.startswith("google.golang.org/") and not auth_failed:
                        pkg_name = name.replace("google.golang.org/", "")
                        if "googleapis" in github_name:
                            # Try google org instead
                            fallback_name = f"github.com/google/{pkg_name}"
                            license_type, license_content = lookup_github_license(fallback_name, github_token)
                            github_count += 1
                            if license_type in ("AUTH_FAILED", "RATE_LIMITED"):
                                license_type = None
                    lookup_results[name] = (license_type, license_content, "GitHub")
                except Exception:
                    lookup_results[name] = (None, None, "GitHub")
                update_progress("GitHub", github_total)
        print()  # Newline after progress

        # Handle auth/rate limit failures from main thread
        if auth_failed:
            print()
            _handle_invalid_token()
        elif rate_limited:
            print(f"\n  GitHub API rate limit exceeded after {github_count} requests")
            _handle_rate_limit_exceeded(_github_has_token, _github_user_limit)

    # Process all results
    for name, version in packages_to_lookup:
        license_type, license_content, source = lookup_results.get(name, (None, None, "Unknown"))

        # Save "other" license content to file if available
        if license_type == "OTHER":
            if license_content:
                custom_licenses[name] = (version, license_content)
                # Save to licenses directory
                package_name_safe = name.replace("/", "-").replace(" ", "_")
                version_safe = version.replace("/", "-").replace(" ", "_")
                license_filename = f"{package_name_safe}-{version_safe}-LICENSE-OTHER.txt"
                license_path = licenses_dir / license_filename
                try:
                    with open(license_path, "w") as f:
                        f.write(license_content)
                except Exception as e:
                    print(f"\n  {RED}Failed to save license for {name}: {e}{NC}")
            else:
                print(f"\n  {YELLOW}WARNING: {name} has OTHER license but content could not be fetched{NC}")

        # Check result - handle user vs auto overrides separately
        user_license = user_overrides.get(name)
        auto_license = auto_overrides.get(name)

        if license_type:
            # Check for inconsistency with user override (report but don't change)
            if user_license and user_license != license_type:
                user_inconsistencies[name] = (user_license, license_type, source)

            # Only add to auto overrides if NOT in user overrides
            if name not in user_overrides:
                new_auto_overrides[name] = license_type

                if auto_license:
                    if auto_license != license_type:
                        # Found a discrepancy in auto overrides
                        corrections[name] = (auto_license, license_type, source)
                else:
                    # New find
                    new_finds[name] = (license_type, source)
        else:
            # Could not detect license
            not_found.append(name)
            # Keep existing auto override if we have one (user overrides are preserved anyway)
            if auto_license and name not in user_overrides:
                new_auto_overrides[name] = auto_license

                # If existing override is "OTHER", check if license file exists
                if auto_license == "OTHER":
                    package_name_safe = name.replace("/", "-").replace(" ", "_")
                    version_safe = version.replace("/", "-").replace(" ", "_")
                    license_filename = f"{package_name_safe}-{version_safe}-LICENSE-OTHER.txt"
                    license_path = licenses_dir / license_filename
                    if not license_path.is_file():
                        print(f"\n  {YELLOW}WARNING: {name} has OTHER override but license file is missing{NC}")
                        print(f"    Expected: {license_path}")
                        print(f"    Re-run with a GitHub token to fetch it")

    github_count = len([p for p in all_github_packages]) if all_github_packages else 0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GitHub API requests made: {github_count}")
    print(f"User overrides (preserved): {len(user_overrides)}")
    print(f"Auto overrides: {len(new_auto_overrides)}")
    print(f"{GREEN}New licenses found: {len(new_finds)}{NC}")
    print(f"{GREEN}Other licenses saved: {len(custom_licenses)}{NC}")
    print(f"Auto corrections: {len(corrections)}")
    print(f"{YELLOW}Could not detect (keeping existing if any): {len(not_found)}{NC}")

    # Report user override inconsistencies (important!)
    # Filter out LicenseRef detections - those are not reliable
    real_inconsistencies = {
        name: (user_val, detected, source)
        for name, (user_val, detected, source) in user_inconsistencies.items()
        if detected != "LicenseRef"
    }

    if corrections:
        print(f"\n{YELLOW}AUTO CORRECTIONS (auto override updated):{NC}")
        for name, (old, new, source) in sorted(corrections.items()):
            print(f"  {name}: {old} -> {new} (via {source})")

    if new_finds:
        print(f"\n{GREEN}NEW LICENSES FOUND:{NC}")
        for name, (license_type, source) in sorted(new_finds.items()):
            print(f"  {name}: {license_type} (via {source})")

    if not_found and len(not_found) <= 50:
        print(f"\n{YELLOW}COULD NOT DETECT (may need manual lookup):{NC}")
        for name in sorted(not_found)[:50]:
            existing = all_existing.get(name)
            if existing:
                source = "user" if name in user_overrides else "auto"
                print(f"  {name}: keeping existing '{existing}' ({source})")
            else:
                print(f"  {name}: no override")

    if new_auto_overrides or user_overrides:
        if not args.dry_run:
            save_overrides(toml_path, user_overrides, new_auto_overrides)
            print(f"\n{GREEN}Updated {toml_path}{NC}")
            print(f"  User overrides: {len(user_overrides)} (preserved)")
            print(f"  Auto overrides: {len(new_auto_overrides)}")
            if custom_licenses:
                print(f"Other license files saved to: {licenses_dir}/")
        else:
            print("\nDry run - no changes made")
    else:
        print("\nNo licenses could be verified")

    if real_inconsistencies:
        print(f"\n{YELLOW}USER OVERRIDE INCONSISTENCIES:{NC}")
        print(f"  The following user overrides differ from detected licenses.")
        print(f"  User overrides are preserved, but you may want to review these:")
        for name, (user_val, detected, source) in sorted(real_inconsistencies.items()):
            print(f"  {name}: user='{user_val}' vs detected='{detected}' (via {source})")


if __name__ == "__main__":
    main()
