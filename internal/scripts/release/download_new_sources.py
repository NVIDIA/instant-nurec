#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import csv
import json
import logging
import sys
import urllib.parse

from pathlib import Path
from typing import Dict, List, Optional

import click

from internal.scripts.release.utils import (
    SpdxPackage,
    ValidationError,
    diff_packages,
    get_registry_host,
    read_spdx_packages,
    resolve_source_urls,
    run_syft_spdx,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _default_sbom_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "base_sbom.spdx.json", output_dir / "release_sbom.spdx.json"


def _download_package_sources(
    package: SpdxPackage,
    output_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> Dict[str, str]:
    urls = resolve_source_urls(package)
    package_name = package.name or "unknown"
    package_version = package.version or "unknown"
    package_dir = output_dir / _safe_name(f"{package_name}-{package_version}")
    package_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "name": package_name,
        "version": package_version,
        "purl": package.purl or "",
        "download_url": "",
        "output_path": "",
        "status": "",
        "error": "",
    }

    if not urls:
        result["status"] = "failed"
        result["error"] = "No source URL resolved"
        if not continue_on_error:
            raise ValidationError(f"No source URL resolved for {package_name}@{package_version}")
        return result

    if dry_run:
        result["status"] = "dry-run"
        result["download_url"] = urls[0]
        result["output_path"] = str(package_dir)
        return result

    from internal.scripts.release.fetch_licenses import download_file

    for url in urls:
        filename = Path(urllib.parse.urlparse(url).path).name
        if not filename:
            filename = _safe_name(f"{package_name}-{package_version}.tar.gz")
        output_path = package_dir / filename
        if download_file(url, output_path):
            result["status"] = "downloaded"
            result["download_url"] = url
            result["output_path"] = str(output_path)
            return result

    result["status"] = "failed"
    result["download_url"] = urls[0]
    result["error"] = "All download attempts failed"
    if not continue_on_error:
        raise ValidationError(f"Failed to download sources for {package_name}@{package_version}")
    return result


@click.command(context_settings={"show_default": True})
@click.option("--base-image", help="Base Docker image reference")
@click.option("--release-image", help="Release Docker image reference")
@click.option("--base-sbom", type=click.Path(path_type=Path), help="Base SPDX SBOM path")
@click.option("--release-sbom", type=click.Path(path_type=Path), help="Release SPDX SBOM path")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True, help="Output directory for sources")
@click.option("--generate-sbom", is_flag=True, help="Generate SBOMs even if paths are provided")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing downloads")
@click.option("--continue-on-error", is_flag=True, help="Continue when a package download fails")
@click.option(
    "--use-netrc",
    is_flag=True,
    help="Use .netrc credentials for image registries",
)
@click.option(
    "--netrc-path",
    default="~/.netrc",
    show_default=True,
    help="Path to .netrc with registry credentials",
)
@click.option(
    "--registry-host",
    help="Override registry host used for .netrc lookup",
)
def download_new_sources(
    base_image: Optional[str],
    release_image: Optional[str],
    base_sbom: Optional[Path],
    release_sbom: Optional[Path],
    output_dir: Path,
    generate_sbom: bool,
    dry_run: bool,
    continue_on_error: bool,
    use_netrc: bool,
    netrc_path: str,
    registry_host: Optional[str],
) -> int:
    """Download sources for packages present in release but not base."""
    try:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if (base_image and not release_image) or (release_image and not base_image):
            raise ValidationError("Both --base-image and --release-image must be provided together.")

        sboms_generated = False

        if generate_sbom:
            if not base_image or not release_image:
                raise ValidationError("--generate-sbom requires both --base-image and --release-image.")
            if not base_sbom or not release_sbom:
                base_sbom, release_sbom = _default_sbom_paths(output_dir)
            base_registry = registry_host
            release_registry = registry_host
            if use_netrc and not registry_host:
                base_registry = get_registry_host(base_image)
                release_registry = get_registry_host(release_image)
                if not base_registry or not release_registry:
                    raise ValidationError("Unable to infer registry host from image reference; use --registry-host.")
            run_syft_spdx(
                base_image,
                base_sbom,
                dry_run=dry_run,
                registry_host=base_registry if use_netrc else None,
                netrc_path=netrc_path,
            )
            run_syft_spdx(
                release_image,
                release_sbom,
                dry_run=dry_run,
                registry_host=release_registry if use_netrc else None,
                netrc_path=netrc_path,
            )
            sboms_generated = True
        else:
            if not base_sbom or not release_sbom:
                if base_image and release_image:
                    base_sbom, release_sbom = _default_sbom_paths(output_dir)
                    base_registry = registry_host
                    release_registry = registry_host
                    if use_netrc and not registry_host:
                        base_registry = get_registry_host(base_image)
                        release_registry = get_registry_host(release_image)
                        if not base_registry or not release_registry:
                            raise ValidationError(
                                "Unable to infer registry host from image reference; use --registry-host."
                            )
                    run_syft_spdx(
                        base_image,
                        base_sbom,
                        dry_run=dry_run,
                        registry_host=base_registry if use_netrc else None,
                        netrc_path=netrc_path,
                    )
                    run_syft_spdx(
                        release_image,
                        release_sbom,
                        dry_run=dry_run,
                        registry_host=release_registry if use_netrc else None,
                        netrc_path=netrc_path,
                    )
                    sboms_generated = True
                else:
                    raise ValidationError("Provide SBOM paths or image references to generate them.")

        if dry_run and sboms_generated:
            logger.info("DRY RUN: skipping SBOM diff because SBOMs are being generated")
            new_packages: List[SpdxPackage] = []
        else:
            if dry_run:
                logger.info("🔍 DRY RUN: Using SBOMs for diff.")
            base_packages = read_spdx_packages(base_sbom)
            release_packages = read_spdx_packages(release_sbom)
            new_packages = diff_packages(base_packages, release_packages)

        logger.info(f"Found {len(new_packages)} packages to download.")

        results: List[Dict[str, str]] = []
        for package in new_packages:
            result = _download_package_sources(package, output_dir, dry_run, continue_on_error)
            results.append(result)

        summary_csv = output_dir / "download_summary.csv"
        summary_json = output_dir / "download_summary.json"

        with summary_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["name", "version", "purl", "download_url", "output_path", "status", "error"],
            )
            writer.writeheader()
            writer.writerows(results)

        summary_json.write_text(json.dumps(results, indent=2))

        logger.info(f"Summary written to: {summary_csv} and {summary_json}")
        return 0
    except ValidationError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(download_new_sources())
