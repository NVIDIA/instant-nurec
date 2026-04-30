# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, cast

from .artifacts import Artifacts
from .datasets import Dataset
from .train_val_configs import TrainValConfig


# Constants
script_paths = {
    "nre_image_trainval": "internal/sqa/scripts/nre_image_trainval.sh",
    "nre_image_render": "internal/sqa/scripts/nre_image_render.sh",
    "nre_eval_rendering_metrics": "internal/sqa/scripts/nre_eval_rendering_metrics.sh",
    "grpc_api_test": "internal/sqa/scripts/grpc_api_test.sh",
    "resolve_edit_assets_scenario": "internal/sqa/scripts/resolve_edit_assets_scenario.py",
    "nre_tools": "internal/sqa/scripts/nre_tools.sh",
    "nre_asset_harvest": "internal/sqa/scripts/asset_harvest.sh",
    "run_example": "internal/sqa/scripts/run_examples.sh",
}

# Bazel executable mappings for different test types and obfuscation settings
bazel_executables = {
    # Main NRE run executable
    "run": {
        "obfuscated": {
            "target": "//internal/scripts/pycena/runtime:pycena_run",
            "path": "_main/internal/scripts/pycena/runtime/pycena_run",
        },
        "non_obfuscated": {
            "target": "//:run",
            "path": "_main/run",
        },
    },
    # NRE tools executable
    "nre_tools": {
        "obfuscated": {
            "target": "//internal/scripts/pycena/runtime:pycena_nre_tools",
            "path": "_main/internal/scripts/pycena/runtime/pycena_nre_tools",
        },
        "non_obfuscated": {
            "target": "//apps:nre_tools",
            "path": "_main/apps/nre_tools",
        },
    },
    # Asset harvester executable
    "asset_harvester": {
        "obfuscated": {
            "target": "",  # TODO No obfuscated target supported yet for asset harvester
            "path": "",
        },
        "non_obfuscated": {
            "target": "//apps/asset_harvester:asset_harvester",
            "path": "_main/apps/asset_harvester/asset_harvester",
        },
    },
}

# Asset harvest track IDs - manually extracted via bazel run //internal/scripts/ncore_vis:ncore_vis
asset_harvest_track_ids = {
    "H81_Panda128_0d59b8c8_4cam_1lidar_v4": [39, 38, 42, 46, 29],
    "H81_Panda128_6b0e750d_4cam_1lidar_v4": [39, 44, 1, 3, 52],
}

# Relative path under NuRec's output directory for the final USDZ artifact produced in training
usdz_artifact_relative_path = "artifacts/last.usdz"
edit_assets_relative_path = "edit_assets/edit_assets.json"


@dataclass
class Command:
    script: str
    args: list[str]
    bazel_executable: dict[str, str] = field(default_factory=lambda: {"target": "", "path": ""})


@dataclass
class CommandGroup:
    """
    A group of commands that are run in parallel, starting with the first command.
    """

    commands: list[Command] = field(default_factory=list)
    wait_before_commands_s: list[float] = field(default_factory=list)
    run_in_background: list[bool] = field(default_factory=list)


@dataclass
class CommandConfig:
    test_type: str
    mode: str
    obfuscation: str
    dataset: Dataset | None
    nre_output_dir: str
    camera_ids: list[str]
    lidar_ids: list[str]
    artifacts: dict[str, Artifacts]
    grpc_port: int  # Port for gRPC server, unique per test to allow parallel execution
    train_val_config: TrainValConfig | None = None
    egocar_hood_dir: str = ""
    artifact_source: str = ""
    test_control_actor: str = ""
    edit_assets_scenario: str = ""
    use_gsplat: str = ""
    timings_filename: str = "timings.txt"
    script_filename: str = ""  # Script filename for test types that require a script file input

    def __get_artifacts_base_path(self) -> str:
        """Get the base path for artifacts based on artifact_source.

        Returns:
            Base path for artifacts (either NRE_OUTPUT_DIR for train_val or artifacts local path)
        """
        if self.artifact_source == "train_val":
            return "$(NRE_OUTPUT_DIR)"
        else:
            return str(self.artifacts[self.artifact_source].local_path)

    def generate_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate commands for this test configuration."""
        if self.test_type == "nre_image_trainval":
            return self.__generate_nre_image_trainval_commands()
        elif self.test_type == "nre_render_grpc":
            return self.__generate_nre_render_grpc_commands()
        elif self.test_type == "nre_image_render":
            return self.__generate_nre_image_render_commands()
        elif self.test_type == "train_val_grpc_composite":
            return self.__generate_train_val_grpc_composite_commands()
        elif self.test_type == "grpc_api":
            return self.__generate_grpc_api_full_commands()
        elif self.test_type == "nre_tools":
            return self.__generate_nre_tools_command()
        elif self.test_type == "asset_harvest":
            return self.__generate_asset_harvest_command()
        elif self.test_type == "run_example":
            return self.__generate_run_example_command()
        else:
            raise ValueError(f"No command generator found for test type: {self.test_type}")

    def __get_bazel_executable(self) -> dict[str, str]:
        """Get the appropriate Bazel target and executable path for this test configuration.

        Returns:
            dict[str, str]: {"target": bazel_target, "path": runfiles_path_for_rlocation}
        """
        # Determine the executable based on test type
        if self.test_type == "nre_tools":
            executable_name = "nre_tools"
        elif self.test_type == "asset_harvest":
            executable_name = "asset_harvester"
        else:
            executable_name = "run"

        # Get the appropriate executable based on obfuscation setting
        obfuscation_key = "obfuscated" if self.obfuscation == "yes" else "non_obfuscated"

        executable = bazel_executables[executable_name][obfuscation_key]

        # In case of lite test, validate that the executable path is not empty
        if self.mode == "lite" and (not executable.get("target") or not executable.get("path")):
            raise ValueError(
                f"Bazel executable not available for {executable_name} with obfuscation={self.obfuscation}. "
                f"This configuration is not yet supported for lite tests."
            )

        return executable

    def __build_trainval_command(self, for_render_test: bool = False) -> Command:
        """Build nre_image_trainval command.

        Args:
            for_render_test: If True, adds additional parameters optimized for generating artifacts used in rendering tests
        """
        # Dataset is validated to be non-None in test_cases.py for this test type
        dataset = cast(Dataset, self.dataset)

        args = [
            "--config-path",
            str(self.train_val_config.config) if self.train_val_config is not None else "",
            "--dataset-path",
            str(dataset.get_json_file()),
            "--output-dir",
            self.nre_output_dir,
            "--filename",
            self.timings_filename,
        ]

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        # Use all available GPUs for these tests. This gives us multi-GPU testing when running on multi-GPU machines,
        # and single-GPU testing when running on single-GPU machines.
        #
        # For the training step of lite render tests, we retain single-GPU execution: it has less overhead on these
        # short tests and lets us smoke check single-GPU in CI even though we run on a pool of machines with 2 GPUs.
        if not (self.mode == "lite" and for_render_test):
            # TODO Temporarily limited to non-obfuscated in lite tests due to bug 5849152
            if self.mode != "lite" or self.obfuscation == "no":
                args.extend(["--world-size", "0"])

        # Collect train and validation parameters
        train_params = []
        val_params = []

        # Add render test specific parameters
        if for_render_test:
            val_params.extend(
                [
                    "system.test.save_inputs=true",
                    "system.test.save_videos=false",
                ]
            )

            # These parameters conflict with lite mode, so we only add them for full
            if self.mode == "full":
                train_params.append("dataset.n_samples_per_epoch=5000")
                val_params.append("dataset.val_camera_frame_step=99")

        # Add lite test specific parameters
        if self.mode == "lite":
            train_params.extend(
                [
                    "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",
                    "dataset.n_train_sequential_image_subsample=2",
                ]
            )
            # Do a very quick training run for render tests, but perform more iterations on trainval tests to cover
            # functionality that is not enabled from the very start, ex. deformable gaussians.
            if for_render_test:
                train_params.append("dataset.n_samples_per_epoch=50")
            else:
                train_params.append("dataset.n_samples_per_epoch=1500")
            val_params.extend(
                [
                    "dataset.n_val_image_subsample=2",
                    "dataset.val_camera_frame_step=3",
                    "system.test.save_videos=false",
                ]
            )

        # Apply collected parameters (without duplicates)
        train_params_unique = list(dict.fromkeys(train_params))
        for param in train_params_unique:
            args.extend(["--train-append", param])

        val_params_unique = list(dict.fromkeys(val_params))
        for param in val_params_unique:
            args.extend(["--val-append", param])

        args.append("$(EXTRA_PARAMS)")  # Will be substituted with actual extra parameters

        return Command(
            script=script_paths["nre_image_trainval"],
            args=args,
            bazel_executable=self.__get_bazel_executable(),
        )

    def __generate_nre_image_trainval_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate nre_image_trainval test sequence."""
        commands: list[Command | CommandGroup] = []

        # Add the trainval command
        commands.append(self.__build_trainval_command())

        # Add eval-rendering-metrics comparing validation predictions against ground truth
        commands.append(
            self.__build_eval_rendering_metrics_command(
                eval_images_dir="$(NRE_OUTPUT_DIR)/val/pred_rgb",
                reference_dir="$(NRE_OUTPUT_DIR)/val/input_rgb",
            )
        )

        return commands

    def __generate_grpc_api_preprocess_command(self) -> Command:
        # Dataset is validated to be non-None in test_cases.py for this test type
        dataset = cast(Dataset, self.dataset)
        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        args = [
            "preprocess",
            "--artifact-path",
            f"{artifacts_base}/{usdz_artifact_relative_path}",
            "--output-dir",
            f"{self.nre_output_dir}/preprocess",
            "--dataset-path",
            str(dataset.get_zarr_itar_file()),
            "--camera-ids",
            ",".join(self.camera_ids),
            "--lidar-id",
            ",".join(self.lidar_ids),
            "$(EXTRA_PARAMS)",  # Will be substituted with actual extra parameters
        ]

        if self.mode == "lite":
            # Request frame 10 for ego mask export, the dataset does not contain
            # enough frames for the SW's default index of 50.
            args.extend(["--camera-frame-idx", "10"])

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        return Command(
            script=script_paths["grpc_api_test"],
            args=args,
            bazel_executable=self.__get_bazel_executable(),
        )

    def __generate_grpc_api_run_server_and_test_shim_commands(self) -> CommandGroup:
        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        # Dataset is validated to be non-None in test_cases.py for this test type
        dataset = cast(Dataset, self.dataset)

        bazel_executable = self.__get_bazel_executable()
        grpc_api_commands = [
            Command(
                script=script_paths["grpc_api_test"],
                args=[
                    "run-server",
                    "--artifact-path",
                    f"{artifacts_base}/{usdz_artifact_relative_path}",
                    "--egocar-hood-dir",
                    f"{self.nre_output_dir}/preprocess/ego-hoods",
                    "--port",
                    str(self.grpc_port),
                    "$(EXTRA_PARAMS)",  # Will be substituted with actual extra parameters
                ],
                bazel_executable=bazel_executable,
            ),
            Command(
                script=script_paths["grpc_api_test"],
                args=[
                    "test-shim",
                    "--dataset-path",
                    str(dataset.get_zarr_itar_file()),
                    "--nre-output-dir",
                    self.nre_output_dir,
                    "--camera-ids",
                    ",".join(self.camera_ids),
                    "--lidar-id",
                    ",".join(self.lidar_ids),
                    "--filename",
                    self.timings_filename,
                    "--port",
                    str(self.grpc_port),
                    "$(EXTRA_PARAMS)",  # Will be substituted with actual extra parameters
                ],
                bazel_executable=bazel_executable,
            ),
        ]
        return CommandGroup(
            commands=grpc_api_commands,
            wait_before_commands_s=[0, 10],
            run_in_background=[True, False],
        )

    def __generate_grpc_api_run_server_and_render_grpc_commands(self) -> CommandGroup:
        bazel_executable = self.__get_bazel_executable()

        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        # Build run-server command args
        run_server_args = [
            "run-server",
            "--artifact-path",
            f"{artifacts_base}/{usdz_artifact_relative_path}",
            "--egocar-hood-dir",
            f"{self.nre_output_dir}/preprocess/ego-hoods",
            "--port",
            str(self.grpc_port),
        ]

        if self.obfuscation == "no":
            run_server_args.append("--no-obfuscated")

        if self.test_control_actor == "yes":
            run_server_args.append("--enable-editing-actors")

        if self.edit_assets_scenario:
            run_server_args.append("--enable-editing-actors")
            run_server_args.extend(
                [
                    "--edit-assets",
                    f"{self.nre_output_dir}/{edit_assets_relative_path}",
                ]
            )

        if self.use_gsplat == "yes":
            run_server_args.append("--use-gsplat")

        run_server_args.append("$(EXTRA_PARAMS)")

        # Build render-grpc command args
        render_grpc_args = [
            "render-grpc",
            "--artifact-path",
            f"{artifacts_base}/{usdz_artifact_relative_path}",
            # No camera ID subfolder created by this command, contrary to nre_image_render.sh
            "--output-dir",
            f"{self.nre_output_dir}/render/{self.camera_ids[0]}",
            # Only 1 camera ID supported, validated through an integrity test case
            "--camera-id",
            self.camera_ids[0],
            "--port",
            str(self.grpc_port),
        ]

        # Add mode/dataset-specific parameters
        if self.mode == "lite":
            render_grpc_args.extend(
                [
                    "--frame-height",
                    "135",
                    "--frame-step",
                    "3",
                ]
            )
        else:
            render_grpc_args.extend(
                [
                    "--frame-height",
                    "540",
                    # Archived artifacts used a frame step of 3 in validation. Rendering must use a
                    # multiple of that value to achieve matching frames.
                    "--frame-step",
                    "99",
                ]
            )

        if self.obfuscation == "no":
            render_grpc_args.append("--no-obfuscated")

        if self.test_control_actor == "yes":
            render_grpc_args.append("--test-actor-control")

        if self.edit_assets_scenario:
            render_grpc_args.extend(
                [
                    "--enable-editing-actors",
                    "--edit-assets",
                    f"{self.nre_output_dir}/{edit_assets_relative_path}",
                ]
            )

        render_grpc_args.append("$(EXTRA_PARAMS)")

        grpc_api_commands = [
            Command(
                script=script_paths["grpc_api_test"],
                args=run_server_args,
                bazel_executable=bazel_executable,
            ),
            Command(
                script=script_paths["grpc_api_test"],
                args=render_grpc_args,
                bazel_executable=bazel_executable,
            ),
        ]
        return CommandGroup(
            commands=grpc_api_commands,
            wait_before_commands_s=[0, 20],
            run_in_background=[True, False],
        )

    def __generate_grpc_api_full_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate complete GRPC API test sequence (preprocess + server + test-shim)."""
        return [
            self.__generate_grpc_api_preprocess_command(),
            self.__generate_grpc_api_run_server_and_test_shim_commands(),
        ]

    def __generate_nre_render_grpc_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate nre_render_grpc sequence (train_val with render parameters + grpc tests)."""
        commands: list[Command | CommandGroup] = []

        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        # If artifact_source is "train_val", include the train_val command first
        if self.artifact_source == "train_val":
            commands.append(self.__build_trainval_command(for_render_test=True))

        # Run preprocess command
        # This is not strictly necessary for the render test, but validates some additional
        # commands exposed by the run image.
        commands.append(self.__generate_grpc_api_preprocess_command())

        if self.edit_assets_scenario:
            commands.append(self.__resolve_edit_assets_scenario_command())

        # Start GRPC server and render through it
        commands.append(self.__generate_grpc_api_run_server_and_render_grpc_commands())

        # Add eval-rendering-metrics command comparing rendered images against validation output
        commands.append(
            self.__build_eval_rendering_metrics_command(
                eval_images_dir=f"{self.nre_output_dir}/render",
                reference_dir=f"{artifacts_base}/val/pred_rgb",
                egocar_hood_dir=f"{self.nre_output_dir}/preprocess/ego-hoods",
            )
        )

        return commands

    def __resolve_edit_assets_scenario_command(self) -> Command:
        dataset = cast(Dataset, self.dataset)
        return Command(
            script=script_paths["resolve_edit_assets_scenario"],
            args=[
                "--scenario-id",
                self.edit_assets_scenario,
                "--dataset-name",
                dataset.name,
                "--output-edit-file",
                f"{self.nre_output_dir}/{edit_assets_relative_path}",
            ],
            bazel_executable=self.__get_bazel_executable(),
        )

    def __generate_nre_image_render_training_views_command(self) -> Command:
        """Generate nre_image_render.sh render-training-views command."""

        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        args = [
            "render-training-views",
            "--artifact-path",
            f"{artifacts_base}/{usdz_artifact_relative_path}",
            "--output-dir",
            f"{self.nre_output_dir}/render",
            # Only 1 camera ID supported, validated through an integrity test case
            "--camera-id",
            self.camera_ids[0],
        ]

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        # Add mode/dataset-specific parameters
        if self.mode == "lite":
            args.extend(
                [
                    "--image-scale",
                    "0.5",
                    "--frame-step",
                    "3",
                ]
            )
        else:
            # Archived artifacts used a frame step of 3 in validation. Rendering must use a
            # multiple of that value to achieve matching frames.
            args.extend(["--frame-step", "99"])

        args.append("$(EXTRA_PARAMS)")  # Will be substituted with actual extra parameters

        return Command(
            script=script_paths["nre_image_render"],
            args=args,
            bazel_executable=self.__get_bazel_executable(),
        )

    def __build_eval_rendering_metrics_command(
        self,
        eval_images_dir: str,
        reference_dir: str,
        egocar_hood_dir: str | None = None,
    ) -> Command:
        """Build nre_eval_rendering_metrics.sh eval command.

        Args:
            eval_images_dir: Path to directory containing images to evaluate
            reference_dir: Path to reference images directory
            egocar_hood_dir: Optional path to ego-hood images (will be exported from dataset if not passed)
        """
        args = [
            "eval",
            "--reference-dir",
            reference_dir,
            "--eval-images-dir",
            eval_images_dir,
            "--output-dir",
            f"{self.nre_output_dir}/eval",
            "--gif-tool",
            "$(GIF_TOOL)",
        ]
        if egocar_hood_dir:
            args.extend(["--egocar-hood-dir", egocar_hood_dir])
        else:
            # Dataset is validated to be non-None in test_cases.py for this test type
            dataset = cast(Dataset, self.dataset)
            args.extend(["--shard-file-pattern", str(dataset.get_zarr_itar_file())])

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        args.append("$(EXTRA_PARAMS)")  # Will be substituted with actual extra parameters

        return Command(
            script=script_paths["nre_eval_rendering_metrics"],
            args=args,
            bazel_executable=self.__get_bazel_executable(),
        )

    def __generate_nre_image_render_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate nre_image_render sequence."""
        commands: list[Command | CommandGroup] = []

        # Get base path for artifacts
        artifacts_base = self.__get_artifacts_base_path()

        # If artifact_source is "train_val", include the train_val command first
        if self.artifact_source == "train_val":
            commands.append(self.__build_trainval_command(for_render_test=True))

        # Add the render-training-views command
        commands.append(self.__generate_nre_image_render_training_views_command())

        # Add eval-rendering-metrics command comparing rendered images against validation output
        commands.append(
            self.__build_eval_rendering_metrics_command(
                eval_images_dir=f"{self.nre_output_dir}/render",
                reference_dir=f"{artifacts_base}/val/pred_rgb",
            )
        )

        return commands

    def __generate_train_val_grpc_composite_commands(self) -> Sequence[Command | CommandGroup]:
        """Generate composite test that runs nre_image_trainval followed by full grpc test."""
        nre_command = self.__build_trainval_command()
        grpc_commands = self.__generate_grpc_api_full_commands()
        return [nre_command, *grpc_commands]

    def __generate_nre_tools_command(self) -> Sequence[Command]:
        # Dataset is validated to be non-None in test_cases.py for this test type
        dataset = cast(Dataset, self.dataset)

        args = [
            "--dataset-path",
            str(dataset.get_zarr_itar_file()),
            "--output-dir",
            self.nre_output_dir,
            "--camera-ids",
            ",".join(self.camera_ids),
            "--filename",
            self.timings_filename,
        ]

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        args.append("$(EXTRA_PARAMS)")  # Will be substituted with actual extra parameters

        return [
            Command(
                script=script_paths["nre_tools"],
                args=args,
                bazel_executable=self.__get_bazel_executable(),
            )
        ]

    def __generate_asset_harvest_command(self) -> Sequence[Command]:
        """Generate asset harvest command."""
        # Dataset is validated to be non-None in test_cases.py for this test type
        dataset = cast(Dataset, self.dataset)

        # Asset harvest needs special configuration with track IDs
        dataset_name = dataset.name
        track_ids = asset_harvest_track_ids.get(dataset_name, [])
        if not track_ids:
            raise ValueError(f"No track IDs defined for asset harvest dataset: {dataset_name}")

        args = [
            "--component-store",
            str(dataset.get_json_file()),
            "--output-dir",
            self.nre_output_dir,
            "--track-ids",
            ",".join(str(track_id) for track_id in track_ids),
        ]

        # Add --no-obfuscated parameter if test is not obfuscated
        if self.obfuscation == "no":
            args.append("--no-obfuscated")

        # Add cache directory
        cache_dir = "~/.cache/nre"
        args.extend(["--cache-dir", cache_dir])

        args.append("$(EXTRA_PARAMS)")  # Will be substituted with actual extra parameters

        return [
            Command(
                script=script_paths["nre_asset_harvest"],
                args=args,
                bazel_executable=self.__get_bazel_executable(),
            )
        ]

    def __generate_run_example_command(self) -> Sequence[Command]:
        """Generate command to run an example script via run_examples.sh --run-script."""
        # The example script filename is passed via the script_filename field (add .py extension)
        script_name = self.script_filename + ".py"
        script_path = Path("docs/architecture/examples") / script_name

        args = [
            "--run-script",
            str(script_path),
            "$(EXTRA_PARAMS)",  # Will be substituted with actual extra parameters
        ]

        return [
            Command(
                script=script_paths["run_example"],
                args=args,
                bazel_executable=self.__get_bazel_executable(),
            )
        ]
