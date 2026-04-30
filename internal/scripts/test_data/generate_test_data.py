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
Script to generate test data by training with prober enabled.
"""

import argparse
import gc
import io
import logging
import shlex
import sys
import tarfile

from pathlib import Path

import torch

from click.testing import CliRunner

from nre.run.main import main


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_default_dataset_path() -> Path:
    """Get the path to the default CLIPGT test dataset."""
    from python.runfiles import runfiles

    RUNFILES = runfiles.Create()

    path = Path(
        RUNFILES.Rlocation("test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json"),
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Default test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


def create_test_data_archive(test_data_dir: Path, output_dir: Path | None = None) -> Path:
    """
    Create an archive containing .pth files from the first subdirectory of each subdirectory in test_data.

    Args:
        test_data_dir: Directory containing test data subdirectories
        output_dir: Directory to save the archive (defaults to test_data_dir parent)

    Returns:
        Path to the created archive file
    """
    if output_dir is None:
        output_dir = test_data_dir.parent

    # Create timestamp for archive name
    archive_name = f"test_data_prober_generated.tar.gz"
    archive_path = output_dir / archive_name

    logger.info(f"Creating archive: {archive_path}")

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            # Find all subdirectories in test_data_dir
            subdirs = [d for d in test_data_dir.iterdir() if d.is_dir()]

            # Create BUILD.bazel content
            build_content = []
            build_content.append("# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.")
            build_content.append('load("@rules_python//python:defs.bzl", "py_library")')
            build_content.append("")
            # Include a root-level anchor file to enable Rlocation of a file (not a directory)
            # from Bazel runfiles. The prober will Rlocation this file and use its parent directory.
            build_content.append('TEST_DATA_ANCHOR = ["test_data/.anchor"]')

            for subdir in subdirs:
                # Find the latest subdirectory within each subdir
                subdirs_in_subdir = sorted(
                    [d for d in subdir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                )

                if len(subdirs_in_subdir) == 0:
                    continue

                last_subdir = subdirs_in_subdir[-1]  # Get the last subdirectory
                logger.info(f"Processing {subdir.name}/{last_subdir.name}")

                # Find all .pth files in the first subdirectory
                pth_files = list(last_subdir.glob("*.pth"))

                # Add each .pth file to the archive
                for pth_file in pth_files:
                    # Create archive path: subdir_name/default/filename (renaming latest subdir to "default")
                    archive_name_in_tar = f"test_data/{subdir.name}/default/{pth_file.name}"
                    tar.add(pth_file, arcname=archive_name_in_tar)
                    logger.debug(f"Added to archive: {archive_name_in_tar}")

                # Add filegroup for this subdirectory (only for actual first-level subdirectories)
                # Skip creating filegroups for step directories (step=0, step=1, etc.)
                if not subdir.name.startswith("step="):
                    build_content.append("filegroup(")
                    build_content.append(f'    name = "{subdir.name}",')
                    build_content.append(f'    srcs = TEST_DATA_ANCHOR + glob(["test_data/{subdir.name}/**/*.pth"]),')
                    build_content.append('    visibility = ["//visibility:public"],')
                    build_content.append(")")
                    build_content.append("")

            # Add an anchor file at the root of test_data to be present in runfiles
            anchor_bytes = b"test_data\n"
            anchor_info = tarfile.TarInfo("test_data/.anchor")
            anchor_info.size = len(anchor_bytes)
            tar.addfile(anchor_info, io.BytesIO(anchor_bytes))
            logger.info("Added test_data/.anchor to archive")

            # Add BUILD.bazel to the archive
            build_content_str = "\n".join(build_content)
            build_info = tarfile.TarInfo("BUILD.bazel")
            build_info.size = len(build_content_str.encode("utf-8"))
            tar.addfile(build_info, io.BytesIO(build_content_str.encode("utf-8")))
            logger.info("Added BUILD.bazel to archive")

        logger.info(f"Archive created successfully: {archive_path}")

    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        # Remove partial archive if it exists
        if archive_path.exists():
            archive_path.unlink()
        raise

    return archive_path


def run_training(
    dataset_path: Path | None,
    output_dir: Path,
    test_data_dir: Path,
    n_samples_per_epoch: int,
    every_n_steps: int,
    config_name: str,
    camera_ids: list[str] | None,
    lidar_id: str | None,
    batch_limit: int,
    additional_args: list[str] | None = None,
) -> None:
    """
    Run training with prober enabled.

    Args:
        dataset_path: Path to the dataset JSON file
        output_dir: Directory to save training outputs
        test_data_dir: Directory where prober will save test data
        n_samples_per_epoch: Number of samples per epoch
        every_n_steps: Number of steps between test data saves
        config_name: Name of the config to use
        camera_ids: List of camera IDs to use
        lidar_id: LiDAR ID to use
        batch_limit: Limit on batch size for testing
        additional_args: Additional arguments to pass to training command
    """
    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    dataset_path_str = '"{0}"'.format(str(dataset_path))
    test_data_dir_str = '"{0}"'.format(str(test_data_dir))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Test data directory: {test_data_dir}")
    logger.info(f"Samples per epoch: {n_samples_per_epoch}")
    logger.info(f"Every n steps: {every_n_steps}")

    # Prepare training arguments
    training_args = [
        f"--config-name={config_name}",
        f"dataset.path={dataset_path_str}",
        f"dataset.n_samples_per_epoch={n_samples_per_epoch}",
        "mode=train",
        f"out_dir={output_dir}",
        "logger=dummy",
        # Enable prober for test data generation
        "prober.enabled=true",
        f"prober.test_data_dir={test_data_dir_str}",
        f"prober.every_n_steps={every_n_steps}",  # Save more frequently for testing
        f"prober.batch_limit={batch_limit}",  # Limit batch size for testing
        f"checkpoint.every_n_train_steps={n_samples_per_epoch}",
    ]

    # Add additional arguments if provided
    if additional_args:
        training_args.extend(additional_args)

    if camera_ids:
        training_args.append(f"dataset.camera_ids=[{','.join(camera_ids)}]")
    if lidar_id:
        training_args.append(f"dataset.lidar_ids=[{lidar_id}]")

    logger.info(f"Run with args: {training_args}")

    # Run the training
    CliRunner().invoke(
        main,
        training_args,
        catch_exceptions=False,
    )


def main_cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate test data by running training with prober enabled")
    parser.add_argument("--test-data-dir", type=Path, required=True, help="Directory where prober will save test data")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to dataset JSON file (defaults to CLIPGT test dataset)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/prober_output"), help="Directory to save training outputs"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="apps/prod/Hyperion-8.1/car2sim.yaml",
        help="Config name (default: apps/prod/Hyperion-8.1/car2sim.yaml)",
    )
    parser.add_argument(
        "--camera-ids",
        type=list[str],
        default=None,
        help="Comma-separated list of camera IDs to use (default: camera_front_wide_120fov)",
    )
    parser.add_argument("--lidar-id", type=str, default=None, help="LiDAR ID to use (default: lidar_gt_top_p128)")
    parser.add_argument(
        "--every-n-steps", type=int, default=100, help="Number of steps between test data saves (default: 100)"
    )
    parser.add_argument(
        "--n-samples-per-epoch", type=int, default=100, help="Number of samples per epoch (default: 100)"
    )
    parser.add_argument(
        "--batch-limit", type=int, default=0, help="Number of samples per batch (default: 0 to disable)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Generic passthrough for additional training arguments
    parser.add_argument(
        "--additional-args",
        dest="additional_args",
        nargs="*",
        help="Additional arguments to pass to the training command (e.g., 'trainer.world_size=4' 'dataset.duration_sec=1')",
    )

    # New combination args for running multiple configurations
    parser.add_argument(
        "--combination-args",
        dest="combination_args",
        nargs="*",
        help="Multiple sets of additional arguments to run different configurations (e.g., 'arg1=val1 arg2=val2' 'arg3=val3 arg4=val4')",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Get dataset path
        if args.dataset_path:
            dataset_path = args.dataset_path
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        else:
            dataset_path = get_default_dataset_path()
            if args.camera_ids is None:
                args.camera_ids = ["camera_front_wide_120fov"]
            if args.lidar_id is None:
                args.lidar_id = "lidar_gt_top_p128"
            # Initialize additional_args if it's None
            if args.additional_args is None:
                args.additional_args = []
            args.additional_args.append("dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=1")
            logger.info(f"Using default CLIPGT dataset: {dataset_path}")

        # Run training
        if not args.combination_args:
            args.combination_args = [""]

        # Run multiple combinations
        logger.info(f"Running {len(args.combination_args)} combinations...")

        for i, combination in enumerate(args.combination_args):
            # Parse combination args (space-separated)
            combination_args_list = shlex.split(combination)

            logger.info(f"Running combination {i + 1}/{len(args.combination_args)}")
            logger.info(f"Args: {combination_args_list}")

            run_training(
                dataset_path=dataset_path,
                output_dir=args.output_dir,
                test_data_dir=args.test_data_dir,
                n_samples_per_epoch=args.n_samples_per_epoch,
                every_n_steps=args.every_n_steps,
                config_name=args.config_name,
                camera_ids=args.camera_ids,
                lidar_id=args.lidar_id,
                batch_limit=args.batch_limit,
                additional_args=combination_args_list + args.additional_args,
            )

            # Collect garbage and empty cache after each training
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Test data generation completed successfully!")
        logger.info(f"Training outputs: {args.output_dir}")
        logger.info(f"Prober test data: {args.test_data_dir}")

        # Create archive of test data
        try:
            archive_path = create_test_data_archive(args.test_data_dir)
            logger.info(f"Test data archive created: {archive_path}")
        except Exception as e:
            logger.warning(f"Failed to create test data archive: {e}")
            # Don't fail the entire process if archive creation fails

    except Exception as e:
        logger.error(f"Test data generation failed: {e}")
        raise


if __name__ == "__main__":
    main_cli()
