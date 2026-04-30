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
import subprocess
import sys

from typing import List

import click

from internal.scripts.release.utils import (
    ValidationError,
    create_rc_image_tag,
    from_image_tag_to_version,
    validate_image_tag_format,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DockerError(Exception):
    """Exception for Docker-related errors."""

    pass


def run_docker_command(args: List[str], check: bool = True, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run a docker command and return the result."""
    cmd = ["docker", *args]

    if dry_run:
        logger.info(f"    {' '.join(cmd)}")
        # Return a mock successful result for dry run
        return subprocess.CompletedProcess(cmd, 0, "", "")

    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        raise DockerError(f"Docker command failed: {' '.join(cmd)}\nError: {e.stderr}") from e


def check_image_exists(image: str, dry_run: bool = False) -> bool:
    """Check if a Docker image exists in the registry."""
    if dry_run:
        logger.info("  Would check if image exists:")
        logger.info(f'    docker manifest inspect "{image}" >/dev/null 2>&1')
        logger.info("  (In dry-run mode, assuming image doesn't exist)")
        return False  # In dry-run mode, assume image doesn't exist to show the flow

    # Use docker manifest inspect to check if image exists without pulling it
    result = run_docker_command(["manifest", "inspect", image], check=False)
    return result.returncode == 0


def confirm_image_processing() -> bool:
    """Ask user for confirmation to process an image."""
    print()
    try:
        response = input("  Process this image? [y/N]: ").strip().lower()
        return response in ["y", "yes"]
    except KeyboardInterrupt:
        print("\n❌ Aborted")
        return False


def pull_tag_push_image(source_image: str, target_image: str, dry_run: bool = False) -> None:
    """Process a single image: pull, tag, and push."""
    # Pull source image
    if dry_run:
        logger.info("  Would pull source image:")
        run_docker_command(["pull", source_image], dry_run=True)
    else:
        logger.info("  Pulling source image...")
        run_docker_command(["pull", source_image])

    # Tag with new name
    if dry_run:
        logger.info("  Would tag image:")
        run_docker_command(["tag", source_image, target_image], dry_run=True)
    else:
        logger.info("  Tagging image...")
        run_docker_command(["tag", source_image, target_image])

    # Push to target registry
    if dry_run:
        logger.info("  Would push to target registry:")
        run_docker_command(["push", target_image], dry_run=True)
        logger.info("  🔍 Would be retagged and pushed!")
    else:
        logger.info("  Pushing to target registry...")
        run_docker_command(["push", target_image])
        logger.info("  ✅ Successfully retagged and pushed!")


@click.command(context_settings={"show_default": True})
@click.option("-t", "--source-tag", required=True, help="Source image tag (required, e.g., 25.10.5-abc12345)")
@click.option("-n", "--rc-number", type=int, default=1, help="Release candidate number (default: 1)")
@click.option("-y", "--yes", is_flag=True, help="Auto-confirm all pushes without prompting")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing docker commands")
def main(source_tag, rc_number, yes, dry_run):
    """
    Retag and push NRE stage images for release candidate.

    Checks if target images already exist and fails if they do (prevents overwriting).
    """

    try:
        if dry_run:
            logger.info(f"🔍 DRY RUN: Showing what would be done for release candidate RC{rc_number}...")
        else:
            logger.info(f"Retagging NRE images for release candidate RC{rc_number}...")

        # Validate source tag format
        validate_image_tag_format(source_tag)

        # Extract version from source tag
        current_version = from_image_tag_to_version(source_tag)
        logger.info(f"Extracted version: {current_version}")
        logger.info(f"Source image tag: {source_tag}")

        # Define source and target registries/repositories
        source_registry = "nvcr.io/nvidian/ct-toronto-ai"
        target_registry = "nvcr.io/nvstaging/nre"

        # Define image mappings: source_suffix -> target_name
        image_mappings = {
            "nre_run": "nre-enterprise",
            "nre_tools": "nre-tools-enterprise",
            "nre_obfuscated_run": "nre",
            "nre_obfuscated_tools": "nre-tools",
        }

        # RC tag format
        rc_tag = create_rc_image_tag(current_version, rc_number)
        logger.info(f"Creating release candidate: {rc_tag}")
        print()

        # Arrays to track processed images
        source_images = []
        target_images = []
        skipped_images = []

        # Process each image
        for source_suffix, target_name in image_mappings.items():
            source_image = f"{source_registry}/{source_suffix}:{source_tag}"
            target_image = f"{target_registry}/{target_name}:{rc_tag}"

            logger.info(f"Processing: {source_suffix} -> {target_name}")
            logger.info(f"  Source: {source_image}")
            logger.info(f"  Target: {target_image}")

            # Check if target image already exists
            if check_image_exists(target_image, dry_run):
                logger.error(f"  ❌ Error: Target image already exists: {target_image}")
                logger.error("  Release candidate images should not be overwritten!")
                logger.error("  Use a different RC number or delete the existing image first.")
                return 1
            else:
                if not dry_run:
                    logger.info("  ✅ Target image does not exist, safe to proceed")

            # Ask for confirmation per push unless auto-confirm is enabled
            if not yes:
                if not confirm_image_processing():
                    logger.info("  ⏭️  Skipped")
                    print()
                    skipped_images.append(source_image)
                    continue

            # Process the image
            pull_tag_push_image(source_image, target_image, dry_run)
            print()

            # Add to tracking lists
            source_images.append(source_image)
            target_images.append(target_image)

        # Final summary
        if dry_run:
            logger.info("🔍 DRY RUN: All operations completed (no actual changes made)!")
        else:
            logger.info("🎉 All images successfully retagged and pushed!")

        print()
        logger.info("Release candidate images:")
        for image in target_images:
            logger.info(f"  {image}")

        if skipped_images:
            print()
            logger.info("Skipped images:")
            for image in skipped_images:
                logger.info(f"  {image}")

        print()
        logger.info("Summary:")
        logger.info(f"  Release candidate: {rc_tag}")
        logger.info(f"  Source tag: {source_tag}")
        logger.info(f"  Images processed: {len(source_images)}")
        logger.info(f"  Images skipped: {len(skipped_images)}")

        if target_images:
            print()
            logger.info(
                "Please paste the following in the NSpect Artifacts section to register the newly pushed images:"
            )
            # Create comma-separated list of target images
            images_list = ",".join(target_images)
            logger.info(images_list)

        return 0

    except (DockerError, ValidationError) as e:
        logger.exception(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
