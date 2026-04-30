# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .artifacts import Artifacts, ArtifactsConfig, sqa_test_artifacts
from .commands import (
    Command,
    CommandConfig,
    CommandGroup,
)
from .datasets import Dataset, DatasetConfig, sqa_test_datasets
from .train_val_configs import TrainValConfig, sqa_test_configs


@dataclass
class TestCaseConfig:
    results_base: str
    dataset_config: DatasetConfig
    artifacts_config: ArtifactsConfig
    force_obfuscation: Optional[str]
    grpc_port_base: int  # Base port for gRPC servers, final port = base + test_id


@dataclass
class TestCase:
    name: str
    mode: str
    obfuscation: str
    dataset: Dataset | None
    results_dir: str
    train_val_config: TrainValConfig | None = None
    artifact_source: str = ""
    test_control_actor: str = ""
    edit_assets_scenario: str = ""
    use_gsplat: str = ""
    commands: Sequence[CommandGroup | Command] = field(default_factory=list)
    ci_runtime_limits: dict[str, int] = field(default_factory=dict)  # step_name -> limit_seconds
    eval_psnr_thresholds: dict[str, float] = field(default_factory=dict)  # camera_id -> min_psnr
    parallel_execution: bool = True
    description: str = ""
    owner: str = ""
    manual_validation: str = ""


def parse_yaml_test_entry(entry: str) -> tuple[str, dict[str, str]]:
    """Parse a YAML test entry in the format: test_type--param1-value1--param2-value2...

    Returns:
        tuple: (test_type, parameters_dict)
    """
    # Extract test type (everything before the first '--')
    if "--" in entry:
        test_type = entry.split("--")[0].strip()
    else:
        test_type = entry.strip()

    # Extract all parameters with double dashes and single dash separator
    param_pattern = r"--([^-]+)-([^-]+)(?=--|$)"
    params = {}
    for match in re.finditer(param_pattern, entry):
        key = match.group(1).strip()
        value = match.group(2).strip()
        params[key] = value

    return test_type, params


def create_test_case_from_params(
    test_id: int,
    test_name: str,
    test_type: str,
    params: dict[str, str],
    test_case_config: TestCaseConfig,
    datasets: dict[str, Dataset],
    artifacts: dict[str, Artifacts],
    configs: dict[str, TrainValConfig],
    ci_runtime_limits: dict[str, int] | None = None,
    eval_psnr_thresholds: dict[str, float] | None = None,
    parallel_execution: bool = True,
    description: str = "",
    owner: str = "",
    manual_validation: str = "",
) -> TestCase:
    """Create a TestCase from test type and parameters."""
    # Parse mode parameter (required)
    mode = params.get("mode")
    if not mode:
        raise ValueError("Mode parameter is required")
    if mode not in ["lite", "full"]:
        raise ValueError(f"Invalid mode value: {mode}. Must be 'lite' or 'full'")

    # Parse obfuscation parameter (required)
    obfuscation = params.get("obfuscation")
    if not obfuscation:
        raise ValueError("Obfuscation parameter is required")
    if obfuscation not in ["yes", "no"]:
        raise ValueError(f"Invalid obfuscation value: {obfuscation}. Must be 'yes' or 'no'")
    if obfuscation != "no" and test_type == "run_example":
        raise ValueError(f"run_example test type must use obfuscation='no', got '{obfuscation}'")

    # Apply force obfuscation if specified in config and update test name accordingly
    final_test_name = test_name
    if test_case_config.force_obfuscation is not None:
        original_obfuscation = obfuscation
        obfuscation = test_case_config.force_obfuscation
        # Update test name to reflect the forced obfuscation value
        final_test_name = test_name.replace(f"--obfuscation-{original_obfuscation}", f"--obfuscation-{obfuscation}")

    # Parse artifact_source parameter first (to determine if config is needed)
    artifact_source = params.get("artifact_source", "")

    if artifact_source:
        # Validate that artifact_source is only used with compatible test types
        if test_type not in ["nre_image_render", "nre_render_grpc"]:
            raise ValueError(
                f"artifact_source parameter is not supported for test type: {test_type}. Only supported for: nre_image_render, nre_render_grpc"
            )
    uses_prebuilt_artifacts = artifact_source and artifact_source != "train_val"

    # Parse dataset parameter
    dataset_name = params.get("dataset")
    if test_type not in ["run_example"]:
        if not dataset_name:
            raise ValueError("Dataset parameter is required")
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset = datasets[dataset_name]
    else:
        dataset = None

    # Parse config parameter
    config_name = params.get("config")

    if test_type in ["nre_tools", "asset_harvest", "run_example"]:
        # These test types never use config
        if config_name:
            raise ValueError(f"Config not allowed for {test_type} tests")
        config_name = None
    elif uses_prebuilt_artifacts:
        # Pre-built artifacts: config must not be provided (train_val will be skipped)
        if config_name:
            raise ValueError(
                f"Config parameter must not be provided when using pre-built artifacts (artifact_source={artifact_source})"
            )
        config_name = None
    else:
        # Other test types: config is required
        if not config_name:
            raise ValueError(f"Config parameter is required for {test_type} tests")

    # Parse script_filename parameter (required for some test types and not allowed for others)
    script_filename = params.get("script_filename", "")

    if test_type in ["run_example"]:
        if not script_filename:
            raise ValueError(f"script_filename parameter is required for {test_type} test type")
    else:
        if script_filename:
            raise ValueError(f"script_filename parameter not allowed for {test_type} test type")

    # Parse test_control_actor parameter (optional)
    test_control_actor = params.get("test_control_actor", "")

    if test_control_actor:
        # Validate that test_control_actor is only used with compatible test types
        if test_type not in ["nre_render_grpc"]:
            raise ValueError(
                f"test_control_actor parameter is not supported for test type: {test_type}. Only supported for: nre_render_grpc"
            )

    edit_assets_scenario = params.get("edit_assets_scenario", "")

    if edit_assets_scenario:
        if test_type not in ["nre_render_grpc"]:
            raise ValueError(
                f"edit_assets_scenario parameter is not supported for test type: {test_type}. Only supported for: nre_render_grpc"
            )

    # Parse use_gsplat parameter (optional)
    use_gsplat = params.get("use_gsplat", "")

    if use_gsplat:
        # Validate that use_gsplat is only used with compatible test types
        if test_type not in ["nre_render_grpc"]:
            raise ValueError(
                f"use_gsplat parameter is not supported for test type: {test_type}. Only supported for: nre_render_grpc"
            )

    # Get train_val_config if specified
    train_val_config = None
    if config_name:
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}")
        train_val_config = configs[config_name]

    # Compute test-specific results directory path using the final test name
    results_dir = str(Path(test_case_config.results_base) / final_test_name)

    # Validate artifact_source if provided
    if artifact_source and artifact_source != "train_val":
        if artifact_source not in artifacts:
            raise ValueError(f"Unknown artifact_source: {artifact_source}")

    # Get sensor IDs from dataset
    camera_ids = dataset.sensors.camera_ids if dataset else []
    lidar_ids = dataset.sensors.lidar_id if dataset else []

    # Compute unique gRPC port based on test ID to allow parallel test execution
    grpc_port = test_case_config.grpc_port_base + test_id

    # Create command config
    command_config = CommandConfig(
        test_type=test_type,
        mode=mode,
        obfuscation=obfuscation,
        dataset=dataset,
        nre_output_dir=results_dir,
        camera_ids=camera_ids,
        lidar_ids=lidar_ids,
        artifacts=artifacts,
        train_val_config=train_val_config,
        egocar_hood_dir="",  # TODO: add egocar hood dir if needed
        artifact_source=artifact_source,
        test_control_actor=test_control_actor,
        edit_assets_scenario=edit_assets_scenario,
        use_gsplat=use_gsplat,
        script_filename=script_filename,
        grpc_port=grpc_port,
    )

    # Generate commands
    commands = command_config.generate_commands()

    return TestCase(
        name=final_test_name,
        mode=mode,
        obfuscation=obfuscation,
        dataset=dataset,
        train_val_config=train_val_config,
        artifact_source=artifact_source,
        test_control_actor=test_control_actor,
        edit_assets_scenario=edit_assets_scenario,
        use_gsplat=use_gsplat,
        commands=commands,
        results_dir=results_dir,
        ci_runtime_limits=ci_runtime_limits or {},
        eval_psnr_thresholds=eval_psnr_thresholds or {},
        parallel_execution=parallel_execution,
        description=description,
        owner=owner,
        manual_validation=manual_validation,
    )


def parse_duration_string(duration_str: str) -> int:
    """Parse a duration string into seconds.

    Supported formats:
        - "180s" -> 180 seconds
        - "3m" -> 180 seconds (3 minutes)
        - "1h" -> 3600 seconds (1 hour)
        - "2m30s" -> 150 seconds
        - "1h5m" -> 3900 seconds
        - "1h5m30s" -> 3930 seconds

    Args:
        duration_str: Duration string to parse

    Returns:
        Duration in seconds as integer

    Raises:
        ValueError: If the duration string format is invalid
    """
    duration_str = duration_str.strip()

    # Parse duration components using regex
    pattern = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
    match = pattern.match(duration_str)

    if not match or not any(match.groups()):
        raise ValueError(
            f"Invalid duration format: '{duration_str}'. Expected formats: '180s', '3m', '1h', '2m30s', '1h5m30s'"
        )

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0

    return hours * 3600 + minutes * 60 + seconds


def load_test_plan_yaml(
    yaml_file_path: Path,
) -> list[tuple[str, dict[str, int], dict[str, float], bool, str, str, str]]:
    """Load test plan from YAML file and return list of (name, ci_runtime_limits, eval_psnr_thresholds, parallel_execution, description, owner, manual_validation) tuples.

    Each entry is a mapping with required 'name' and optional fields:
        - name: test_name--param1-value1--param2-value2
          ci_runtime_limit_train: "3m"
          ci_runtime_limit_val: "2m"
          eval_psnr_threshold_camera_front_wide_120fov: 30.0
          parallel_execution: false

    Returns:
        List of tuples: (test_name, ci_runtime_limits_dict, eval_psnr_thresholds_dict, parallel_execution, description, owner, manual_validation)
        where ci_runtime_limits_dict maps step names to limit in seconds.
        and eval_psnr_thresholds_dict maps camera IDs to minimum PSNR values.
    """
    if not yaml_file_path.exists():
        raise FileNotFoundError(f"Test plan YAML file not found: {yaml_file_path}")

    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    if "test_plan" not in data:
        raise ValueError("YAML file must contain a 'test_plan' key")

    test_entries = data["test_plan"]
    if not isinstance(test_entries, list):
        raise TypeError("'test_plan' must be a list")

    parsed_entries = []
    for entry in test_entries:
        if not isinstance(entry, dict):
            raise TypeError(f"Test entry must be a mapping with 'name' key, got: {type(entry)}")

        if "name" not in entry:
            raise ValueError(f"Test entry must have a 'name' key: {entry}")

        ci_runtime_limits = {}
        eval_psnr_thresholds = {}

        for key, value in entry.items():
            # Parse ci_runtime_limit_* fields
            if key.startswith("ci_runtime_limit_"):
                step_name = key.replace("ci_runtime_limit_", "")
                if not isinstance(value, str):
                    raise ValueError(f"Runtime limit must be a duration string (e.g., '3m', '180s'): {key}={value}")
                ci_runtime_limits[step_name] = parse_duration_string(value)
            # Parse eval_psnr_threshold_* fields
            elif key.startswith("eval_psnr_threshold_"):
                camera_id = key.replace("eval_psnr_threshold_", "")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"PSNR threshold must be a number: {key}={value}")
                eval_psnr_thresholds[camera_id] = float(value)

        parallel_execution = entry.get("parallel_execution", True)
        description = entry.get("description", "")
        owner = entry.get("owner", "")
        manual_validation = entry.get("manual_validation", "")

        parsed_entries.append(
            (
                entry["name"],
                ci_runtime_limits,
                eval_psnr_thresholds,
                parallel_execution,
                description,
                owner,
                manual_validation,
            )
        )

    return parsed_entries


def generate_test_cases(config: TestCaseConfig) -> list[TestCase]:
    """Generate test cases by loading and parsing the test_plan.yml file."""
    # Get the path to the test_plan.yml file (same directory as this file)
    current_dir = Path(__file__).parent
    yaml_file_path = current_dir / "test_plan.yml"

    # Load test entries from YAML
    test_entries = load_test_plan_yaml(yaml_file_path)

    # Get available datasets, artifacts and configs
    datasets = sqa_test_datasets(config.dataset_config)
    artifacts = sqa_test_artifacts(config.artifacts_config)
    configs = sqa_test_configs

    test_cases = []

    for test_id, (
        name,
        ci_runtime_limits,
        eval_psnr_thresholds,
        parallel_execution,
        description,
        owner,
        manual_validation,
    ) in enumerate(test_entries):
        # Parse the YAML entry format
        test_type, params = parse_yaml_test_entry(name)

        # Create test case directly from params
        test_case = create_test_case_from_params(
            test_id,
            name,
            test_type,
            params,
            config,
            datasets,
            artifacts,
            configs,
            ci_runtime_limits,
            eval_psnr_thresholds,
            parallel_execution,
            description,
            owner,
            manual_validation,
        )
        test_cases.append(test_case)

    # Deduplicate test cases by final name if force obfuscation is applied
    if config.force_obfuscation is not None:
        # Create a mapping from names to test cases, then extract unique values
        unique_dict = {tc.name: tc for tc in test_cases}
        test_cases = list(unique_dict.values())

    return test_cases
