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

import logging
import sys

from pathlib import Path

import click

from internal.scripts.release.utils import ValidationError, get_registry_host, run_syft_spdx


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@click.command(context_settings={"show_default": True})
@click.option("--image", required=True, help="Docker image reference to scan")
@click.option("--out", "output_path", required=True, help="Output SPDX JSON path")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option(
    "--use-netrc",
    is_flag=True,
    help="Use .netrc credentials for the image registry",
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
def generate_sbom(
    image: str,
    output_path: str,
    dry_run: bool,
    use_netrc: bool,
    netrc_path: str,
    registry_host: str | None,
) -> int:
    """Generate an SPDX SBOM for a Docker image using syft."""
    try:
        out_path = Path(output_path)
        registry = None
        if use_netrc:
            registry = registry_host or get_registry_host(image)
            if not registry:
                raise ValidationError("Unable to infer registry host from image reference; use --registry-host.")
        run_syft_spdx(image, out_path, dry_run=dry_run, registry_host=registry, netrc_path=netrc_path)
        if dry_run:
            logger.info("🔍 DRY RUN: SBOM generation skipped.")
        else:
            logger.info(f"✅ SPDX SBOM generated: {out_path}")
        return 0
    except ValidationError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(generate_sbom())
