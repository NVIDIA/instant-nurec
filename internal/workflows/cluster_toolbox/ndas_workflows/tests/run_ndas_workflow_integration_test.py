# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Integration tests for run_ndas_workflow.py

These tests validate that workflow specs are generated correctly for each config
flavor and pass maglev validation. Config files are automatically discovered from
cluster_configs/ndas_workflows/, so new configs are tested without code changes.
Tests run in dry-run mode and do not submit actual workflows.

Requirements:
- MAGLEV_API_KEY should be set for authentication.

Usage:
    bazel test //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test

    # Dump generated specs to a directory for diffing before/after refactors:
    NDAS_WORKFLOW_TEST_OUTPUT_DIR=$PWD/before bazel run //...tests:run_ndas_workflow_integration_test
    NDAS_WORKFLOW_TEST_OUTPUT_DIR=$PWD/after bazel run //...tests:run_ndas_workflow_integration_test
    diff -r before after
"""

import os
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import omegaconf
import yaml  # type: ignore[import-untyped]

from internal.workflows.cluster_toolbox.maglev_toolbox import MaglevToolbox
from internal.workflows.cluster_toolbox.ndas_workflows.run_ndas_workflow import (
    build_template_from_config,
    configure_worker_pools_and_tasks,
    fill_template,
)


def _get_output_dir() -> Path | None:
    """Get output directory from NDAS_WORKFLOW_TEST_OUTPUT_DIR environment variable.

    Specs are only dumped if explicitly requested via this env var.
    This prevents polluting directories during regular test runs.
    """
    output_dir = os.environ.get("NDAS_WORKFLOW_TEST_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir)
    return None


# Parse output directory once at module load
_OUTPUT_DIR = _get_output_dir()
if _OUTPUT_DIR:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDumping generated specs to: {_OUTPUT_DIR.absolute()}")


def save_spec_to_output_dir(config_name: str, spec: dict) -> None:
    """Save a generated spec to the output directory if configured.

    Args:
        config_name: Name of the config file (e.g., "ndas_workflow_base.yaml")
        spec: The generated workflow spec dictionary.
    """
    if _OUTPUT_DIR is None:
        return

    output_file = _OUTPUT_DIR / f"generated_{config_name}"
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=True, default_flow_style=False)
    print(f"  Saved: {output_file}")


# Configs to exclude from testing (e.g., base configs that aren't standalone)
EXCLUDED_CONFIGS = [
    "ndas_workflow_base.yaml",  # Base config, not a standalone workflow
    "workerpool.yaml",  # Worker-pool infra catalog, included by base via defaults
]


def _discover_configs() -> list[str]:
    """Auto-discover all config files in cluster_configs/ndas_workflows/.

    Under bazel, __file__ resolves to the runfiles tree, so relative paths work correctly.
    """
    tests_dir = Path(__file__).parent
    config_dir = tests_dir.parent.parent / "cluster_configs" / "ndas_workflows"

    configs = []
    for f in sorted(config_dir.glob("*.yaml")):
        if f.name not in EXCLUDED_CONFIGS:
            configs.append(f.name)
    return configs


NDAS_WORKFLOW_CONFIGS = _discover_configs()

# Docker images to use for testing (these don't need to exist for validation)
TEST_DOCKER_IMAGE = "nvcr.io/nvidian/ct-toronto-ai/nre-run-dev:test"
TEST_TOOLS_IMAGE = "nvcr.io/nvidian/ct-toronto-ai/nre_tools:test"
SJC_CLUSTER = "nv-sjc-maglev-001"
RNO_RESOURCE_SHARE = "av-car2sim-cicd-rno"


def _make_catalog() -> dict:
    return {
        "sjc": {
            "field": "cluster",
            "name": SJC_CLUSTER,
            "ppp": "av_sim_car2sim-cicd",
            "gpus": {"a40": "ovxa40.48gb.perf", "l4": "mgxl4.24gb.perf"},
        },
        "rno": {
            "field": "resource_share",
            "name": RNO_RESOURCE_SHARE,
            "ppp": "av_sim_car2sim-cicd",
            "gpus": {"a100_80gb": "dgxa100.80gb.perf", "ovx": "constellation-v2-ovx"},
        },
    }


def _make_minimal_spec() -> dict:
    return {
        "tasks": [
            {"name": "render_task_a", "workerPool": "render-gpu-pool"},
            {"name": "render_task_b", "workerPool": "render-gpu-pool"},
            {"name": "cpu_analyzer", "workerPool": "cpu-pool"},
        ],
        "workerPools": [
            {
                "name": "render-gpu-pool",
                "gpu": "1",
                "cluster": "stale-cluster",
                "resourceShare": "stale-share",
                "nodeConstraints": {"required": {"nodeType": "stale-node-type"}},
            },
            {"name": "cpu-pool", "gpu": "0", "cluster": "stale-cluster", "resourceShare": "stale-share"},
        ],
        "cluster": "stale-cluster",
        "resourceShare": "stale-share",
    }


def _make_config(**overrides) -> omegaconf.dictconfig.DictConfig:
    base = {
        "compute_targets": _make_catalog(),
        "default_compute_target": "sjc",
        "default_gpu": "a40",
        "tasks": {},
    }
    base.update(overrides)
    return omegaconf.OmegaConf.create(base)


class TestWorkerPoolsAndTasks(unittest.TestCase):
    """Unit tests for the task-first resolver."""

    def test_default_scheme_routes_pools_to_default_compute_target(self):
        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), _make_config())

        self.assertEqual(spec["cluster"], SJC_CLUSTER)
        self.assertNotIn("resourceShare", spec)
        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool"]["cluster"], SJC_CLUSTER)
        self.assertEqual(worker_pools["render-gpu-pool"]["nodeConstraints"]["required"]["nodeType"], "ovxa40.48gb.perf")
        self.assertEqual(worker_pools["cpu-pool"]["cluster"], SJC_CLUSTER)
        self.assertNotIn("resourceShare", worker_pools["render-gpu-pool"])

    def test_task_override_moves_pool_to_resource_share(self):
        config = _make_config(
            tasks={
                "render_task_a": {"compute_target": "rno", "gpu": "a100_80gb"},
                "render_task_b": {"compute_target": "rno", "gpu": "a100_80gb"},
            }
        )

        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool"]["resourceShare"], RNO_RESOURCE_SHARE)
        self.assertNotIn("cluster", worker_pools["render-gpu-pool"])
        self.assertEqual(
            worker_pools["render-gpu-pool"]["nodeConstraints"]["required"]["nodeType"], "dgxa100.80gb.perf"
        )

    def test_pool_splits_when_tasks_request_different_gpus(self):
        config = _make_config(tasks={"render_task_b": {"gpu": "l4"}})

        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertIn("render-gpu-pool-sjc-a40", worker_pools)
        self.assertIn("render-gpu-pool-sjc-l4", worker_pools)
        tasks = {t["name"]: t for t in spec["tasks"]}
        self.assertEqual(tasks["render_task_a"]["workerPool"], "render-gpu-pool-sjc-a40")
        self.assertEqual(tasks["render_task_b"]["workerPool"], "render-gpu-pool-sjc-l4")

    def test_pool_splits_when_tasks_request_different_compute_targets(self):
        config = _make_config(
            tasks={
                "render_task_b": {"compute_target": "rno", "gpu": "a100_80gb"},
            }
        )

        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool-sjc-a40"]["cluster"], SJC_CLUSTER)
        self.assertEqual(worker_pools["render-gpu-pool-rno-a100_80gb"]["resourceShare"], RNO_RESOURCE_SHARE)

    def test_unknown_compute_target_raises(self):
        config = _make_config(tasks={"render_task_a": {"compute_target": "atlantis"}})

        with self.assertRaisesRegex(ValueError, "Unknown compute target 'atlantis'"):
            configure_worker_pools_and_tasks(_make_minimal_spec(), config)

    def test_gpu_not_offered_by_compute_target_raises(self):
        config = _make_config(tasks={"render_task_a": {"gpu": "a100_80gb"}})

        with self.assertRaisesRegex(ValueError, "GPU 'a100_80gb' not offered by compute target"):
            configure_worker_pools_and_tasks(_make_minimal_spec(), config)

    def test_source_workers_count_preserved_when_no_task_pins(self):
        spec = _make_minimal_spec()
        spec["workerPools"][0]["workers"] = "32"
        config = _make_config()

        spec = configure_worker_pools_and_tasks(spec, config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool"]["workers"], "32")

    def test_per_task_workers_takes_max_within_pool(self):
        spec = _make_minimal_spec()
        spec["workerPools"][0]["workers"] = "1"
        config = _make_config(
            tasks={
                "render_task_a": {"workers": 200},
                "render_task_b": {"workers": 50},
            }
        )

        spec = configure_worker_pools_and_tasks(spec, config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool"]["workers"], 200)

    def test_split_pools_size_independently_when_tasks_pin_workers(self):
        spec = _make_minimal_spec()
        spec["workerPools"][0]["workers"] = "1"
        config = _make_config(
            tasks={
                "render_task_a": {"workers": 200},
                "render_task_b": {"gpu": "l4", "workers": 10},
            }
        )

        spec = configure_worker_pools_and_tasks(spec, config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool-sjc-a40"]["workers"], 200)
        self.assertEqual(worker_pools["render-gpu-pool-sjc-l4"]["workers"], 10)

    def test_workers_never_decreases_existing_count(self):
        spec = _make_minimal_spec()
        spec["workerPools"][0]["workers"] = "500"
        config = _make_config(tasks={"render_task_a": {"workers": 100}})

        spec = configure_worker_pools_and_tasks(spec, config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["render-gpu-pool"]["workers"], "500")

    def test_cpu_pool_skips_gpu_resolution(self):
        config = _make_config(tasks={"cpu_analyzer": {"compute_target": "rno"}})

        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), config)

        worker_pools = {wp["name"]: wp for wp in spec["workerPools"]}
        self.assertEqual(worker_pools["cpu-pool"]["resourceShare"], RNO_RESOURCE_SHARE)
        self.assertNotIn("nodeConstraints", worker_pools["cpu-pool"])

    def test_ppp_comes_from_default_compute_target_when_unset(self):
        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), _make_config())
        self.assertEqual(spec["ppp"], "av_sim_car2sim-cicd")

    def test_workflow_ppp_overrides_catalog_ppp(self):
        config = _make_config(ppp="some-other-ppp")
        spec = configure_worker_pools_and_tasks(_make_minimal_spec(), config)
        self.assertEqual(spec["ppp"], "some-other-ppp")

    def test_missing_catalog_raises(self):
        config = omegaconf.OmegaConf.create({"default_compute_target": "sjc"})

        with self.assertRaisesRegex(ValueError, "'compute_targets' catalog is required"):
            configure_worker_pools_and_tasks(_make_minimal_spec(), config)


def generate_workflow_spec(config_name: str) -> dict:
    """Generate a workflow spec for the given config.

    Args:
        config_name: Name of the config file (e.g., "ndas_workflow_base.yaml")

    Returns:
        The generated workflow spec as a dictionary.
    """
    hydra_args = [
        f"docker.image={TEST_DOCKER_IMAGE}",
        f"docker.tools_image={TEST_TOOLS_IMAGE}",
    ]

    # Mock _check_image_exists since we don't need real docker images for validation
    # Mock task_expired to skip expiration checks (we're testing workflow structure, not runtime state)
    with (
        patch.object(MaglevToolbox, "_check_image_exists"),
        patch.object(MaglevToolbox, "task_expired", return_value=False),
    ):
        maglev_toolbox = MaglevToolbox(
            config_name=f"ndas_workflows/{config_name}",
            hydra_args=hydra_args,
        )

        spec = build_template_from_config(maglev_toolbox.config, maglev_toolbox)
        spec = fill_template(spec, maglev_toolbox.config)

    return spec


def validate_workflow_spec(spec: dict, local: bool = True) -> tuple[bool, str]:
    """Validate a workflow spec using maglev CLI.

    Args:
        spec: The workflow spec dictionary to validate.
        local: If True, use --local flag to skip remote image validation.

    Returns:
        A tuple of (success: bool, output: str).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="test_workflow_",
        suffix=".yaml",
        delete=False,
    ) as tmp_file:
        yaml.safe_dump(spec, tmp_file, sort_keys=False)
        tmp_path = tmp_file.name

    try:
        cmd = ["maglev", "workflows2", "validate", "-f", tmp_path]
        if local:
            cmd.append("--local")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    finally:
        os.unlink(tmp_path)


class TestNdasWorkflowIntegration(unittest.TestCase):
    """Integration tests for NDAS workflow generation and validation."""

    @classmethod
    def setUpClass(cls):
        """Verify maglev CLI is available before running tests."""
        try:
            result = subprocess.run(
                ["maglev", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise unittest.SkipTest("Maglev CLI not available or not authenticated")
        except FileNotFoundError:
            raise unittest.SkipTest("Maglev CLI not installed")
        except subprocess.TimeoutExpired:
            raise unittest.SkipTest("Maglev CLI timed out")


def _make_config_test(config_name: str):
    """Factory function to create a test method for a specific config."""

    def _config_test_impl(self):
        """Test that workflow spec is generated and validates successfully."""
        # Generate the workflow spec
        spec = generate_workflow_spec(config_name)

        # Save to output directory if configured (for before/after diffing)
        save_spec_to_output_dir(config_name, spec)

        # Basic sanity checks on the generated spec
        self.assertIn("tasks", spec, "Spec should contain tasks")
        self.assertIn("workerPools", spec, "Spec should contain workerPools")
        self.assertGreater(len(spec["tasks"]), 0, "Spec should have at least one task")
        self.assertGreater(len(spec["workerPools"]), 0, "Spec should have at least one worker pool")

        # Validate with maglev CLI
        success, output = validate_workflow_spec(spec)
        self.assertTrue(
            success,
            f"Maglev validation failed for {config_name}:\n{output}",
        )

    # Set a descriptive name for the test method
    _config_test_impl.__name__ = f"test_config_{config_name.replace('.yaml', '').replace('-', '_')}"
    _config_test_impl.__doc__ = f"Test workflow generation and validation for {config_name}"

    return _config_test_impl


# Dynamically add test methods for each config
for _config in NDAS_WORKFLOW_CONFIGS:
    _test_fn = _make_config_test(_config)
    setattr(TestNdasWorkflowIntegration, _test_fn.__name__, _test_fn)

# Clean up loop variables to avoid pytest discovering them.
globals().pop("_config", None)
globals().pop("_test_fn", None)


if __name__ == "__main__":
    unittest.main()
