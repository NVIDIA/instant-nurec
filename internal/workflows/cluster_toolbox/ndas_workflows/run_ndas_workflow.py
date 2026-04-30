# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import json
import logging
import os
import re
import subprocess
import tempfile

from itertools import combinations
from typing import Optional, Union, cast

import click
import omegaconf
import yaml  # type: ignore[import-untyped]

from internal.workflows.cluster_toolbox.maglev_toolbox import MaglevToolbox


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(os.path.basename(__file__))


"""
cut_spec(spec: dict, required_tasks: list[str], source_wf: str) -> dict

cut_spec cuts the workflow such that:
- Every required task is recomputed (included in wf)
- A minimal set of intermediate tasks are also recomputed.

For instance, if we have a workflow with
 A -> B -> C -> D -> E
And we require B and D, then cut_spec will return a workflow with B, C, and D.
- A is cached (referenced externally)
- B is run on cached outputs of A
- C is run on recomputed outputs of B
- D is run on recomputed outputs of C
- E is completely skipped
"""


def cut_spec(spec: dict, required_tasks: list[str], source_wf: str) -> dict:
    def get_internal_dependencies(task: dict) -> list[str]:
        deps = []
        for input in task.get("inputs", []):
            if "task" in input and "workflow" not in input:
                deps.append(input["task"])
        return deps

    def cut_dag(spec: dict, required_tasks: list[str]) -> dict:
        """
        Helper function for cut_spec that does actual DAG cutting.
        """
        # algo:
        # find upstream tasks (tasks that don't depend on any other required tasks)
        # for each remaining required task, find dependent tasks that depend on any upstream task, include in cut

        if len(required_tasks) == 0:
            return spec
        # required_tasks is a list of substrings, this is fully qualified.
        required_tasks_full_name = set(
            [
                task["name"]
                for task in spec["tasks"]
                if any([required_task in task["name"] for required_task in required_tasks])
            ]
        )
        task_adjacencies = {}
        for task in spec["tasks"]:
            task_adjacencies[task["name"]] = get_internal_dependencies(task)

        # search to find all dependencies for each task O(n^2)
        tasks_to_full_dependencies: dict[str, set[str]] = {}
        for task in task_adjacencies:
            tasks_to_full_dependencies[task] = set()
            visited = set()
            queue = [t for t in task_adjacencies[task]]
            while queue:
                current_task = queue.pop(0)
                if current_task in visited:
                    continue
                visited.add(current_task)
                tasks_to_full_dependencies[task].add(current_task)
                queue.extend(task_adjacencies[current_task])
        # tasks_to_full_dependencies = {k: set(v) for k, v in tasks_to_full_dependencies.items()}

        upstream_tasks = []
        for task_name in required_tasks_full_name:
            if len(required_tasks_full_name.intersection(tasks_to_full_dependencies[task_name])) == 0:
                upstream_tasks.append(task_name)

        # exclude tasks that are beyond the cut
        task_candidates = set.union(*[tasks_to_full_dependencies[task] for task in required_tasks_full_name])
        task_candidates = set.union(task_candidates, required_tasks_full_name)

        cut_tasks = [
            task_name
            for task_name in task_candidates
            if task_name in upstream_tasks
            or len(tasks_to_full_dependencies[task_name].intersection(upstream_tasks)) > 0
        ]

        spec["tasks"] = [task for task in spec["tasks"] if task["name"] in cut_tasks]
        return spec

    def replace_inputs(spec: dict, source_wf: str) -> dict:
        """
        Helper function for cut_spec to replace inputs with external inputs.
        """
        cut_tasks = set([task["name"] for task in spec["tasks"]])
        wf, run = source_wf.split("/")

        for task in spec["tasks"]:
            task_inputs = task.get("inputs", [])
            for task_input in task_inputs:
                if "task" in task_input and "workflow" not in task_input and task_input["task"] not in cut_tasks:
                    # this is an external input, replace with the source workflow
                    task_input["workflow"] = wf
                    task_input["run"] = run
        return spec

    # cut_spec implementation
    if len(required_tasks) == 0:
        return spec
    spec = cut_dag(spec, required_tasks)
    replace_inputs(spec, source_wf)
    return spec


"""
join_specs(specs: list[dict]) -> dict

join_specs joins multiple specs into a single spec, resolving worker pool and task name conflicts.
"""


def join_specs(specs: list[dict]) -> dict:
    assert len(specs) > 0, "No specs to join"

    # first union and renumber worker pools
    worker_pool_names = [set([wp["name"] for wp in spec["workerPools"]]) for spec in specs]
    if len(worker_pool_names) > 1:
        pairwise_worker_pool_collisions = [
            set.intersection(names1, names2) for names1, names2 in combinations(worker_pool_names, 2)
        ]
        if len(pairwise_worker_pool_collisions) > 0:
            log.debug(f"Pairwise worker pool collisions: {pairwise_worker_pool_collisions}")
        worker_pool_collisions = set.union(*pairwise_worker_pool_collisions)
    else:
        worker_pool_collisions = set()
    log.debug(f"Worker pool names: {worker_pool_names}")
    log.debug(f"Worker pool collisions: {worker_pool_collisions}")
    unified_worker_pools = []
    for spec_idx, spec in enumerate(specs):
        for wp in spec["workerPools"]:
            if wp["name"] in worker_pool_collisions or wp["name"].startswith("wf"):
                wp["name"] = f"wf{spec_idx}-{wp['name']}"
            unified_worker_pools.append(wp)

    # union and renumber tasks
    task_names = [set([t["name"] for t in spec["tasks"]]) for spec in specs]
    if len(task_names) > 1:
        pairwise_task_collisions = [set.intersection(names1, names2) for names1, names2 in combinations(task_names, 2)]
        if len(pairwise_task_collisions) > 0:
            log.debug(f"Pairwise task collisions: {pairwise_task_collisions}")
        task_collisions = set.union(*pairwise_task_collisions)
    else:
        task_collisions = set()

    unified_tasks = []
    used_worker_pools = set()
    for spec_idx, spec in enumerate(specs):
        for task in spec["tasks"]:
            # rename collisions
            if task["name"] in task_collisions:
                task["name"] = f"wf{spec_idx}_{task['name']}"
            # require inputs
            for input in task.get("inputs", []):
                if "task" in input and "workflow" not in input and input["task"] in task_collisions:
                    # i.e. if the task input is internal, replace collision names
                    input["task"] = f"wf{spec_idx}_{input['task']}"
            # rename worker pool collisions
            if "workerPool" in task and (
                task["workerPool"] in worker_pool_collisions or task["workerPool"].startswith("wf")
            ):
                task["workerPool"] = f"wf{spec_idx}-{task['workerPool']}"

            if "workerPool" in task:
                used_worker_pools.add(task["workerPool"])

            unified_tasks.append(task)

    overall_template = specs[0]
    overall_template["tasks"] = unified_tasks
    overall_template["workerPools"] = unified_worker_pools

    # Remove worker pools that are not used.
    final_worker_pools = [wp for wp in overall_template["workerPools"] if wp["name"] in used_worker_pools]
    overall_template["workerPools"] = final_worker_pools

    return overall_template


def _resolve_compute_target(catalog: omegaconf.dictconfig.DictConfig, key: str) -> omegaconf.dictconfig.DictConfig:
    compute_target = catalog.get(key)
    if compute_target is None:
        raise ValueError(f"Unknown compute target '{key}'. Catalog keys: {list(catalog.keys())}")
    return compute_target


def _resolve_node_type(compute_target: omegaconf.dictconfig.DictConfig, gpu: str) -> str:
    gpus = compute_target.get("gpus") or {}
    if gpu not in gpus:
        raise ValueError(
            f"GPU '{gpu}' not offered by compute target '{compute_target.get('name')}'. Available: {list(gpus.keys())}"
        )
    return str(gpus[gpu])


def _resolve_task_overrides(
    config: omegaconf.dictconfig.DictConfig,
    canonical_name: str,
) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """Look up (compute_target, gpu, workers) override for a task by exact-match on its canonical name."""
    tasks_config = config.get("tasks") or {}
    entry = tasks_config.get(canonical_name)
    if entry is None:
        return None, None, None
    compute_target = entry.get("compute_target")
    gpu = entry.get("gpu")
    workers = entry.get("workers")
    return (
        str(compute_target) if compute_target is not None else None,
        str(gpu) if gpu is not None else None,
        int(workers) if workers is not None else None,
    )


def _is_gpu_pool(wp: dict) -> bool:
    return "nodeConstraints" in wp and "gpu" in wp and wp.get("gpu") != "0"


def configure_worker_pools_and_tasks(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    """Resolve placement, GPU node type, and worker count per task.

    Tasks are grouped by (source_pool, compute_target, gpu). If one source pool maps to multiple groups, derived pools
    are named `<source>-<compute_target>-<gpu>` or `<source>-<compute_target>-cpu`.

    Config schema:
        compute_targets:                          # see workerpool.yaml
          <key>: { field, name, ppp, gpus: { ... } }
        default_compute_target: <catalog key>
        default_gpu: <generic name>
        ppp: <maglev ppp>                         # optional override; defaults to default_compute_target's ppp
        tasks:
          <canonical_task_name>: { compute_target: <key>, gpu: <generic name>, workers: <int> }

    Task keys are exact canonical task names after workflow/numeric prefixes are stripped
    (e.g. `18-reconstruction` -> `reconstruction`). Task-level `workers` only bumps the source
    spec's existing worker count; it never lowers it.
    """
    assert isinstance(config, omegaconf.dictconfig.DictConfig)

    catalog = config.get("compute_targets")
    if catalog is None:
        raise ValueError("'compute_targets' catalog is required (see workerpool.yaml)")
    default_compute_target = config.get("default_compute_target")
    if default_compute_target is None:
        raise ValueError("'default_compute_target' is required")
    default_compute_target_key = str(default_compute_target)
    default_compute_target_def = _resolve_compute_target(catalog, default_compute_target_key)
    default_gpu = config.get("default_gpu")
    default_gpu_key = str(default_gpu) if default_gpu is not None else None

    spec.pop("cluster", None)
    spec.pop("resourceShare", None)
    default_field = str(default_compute_target_def.get("field"))
    default_target_name = str(default_compute_target_def.get("name"))
    if default_field == "cluster":
        spec["cluster"] = default_target_name
    elif default_field == "resource_share":
        spec["resourceShare"] = default_target_name
    else:
        raise ValueError(
            f"compute_targets.{default_compute_target_key}.field must be 'cluster' or 'resource_share', got '{default_field}'"
        )
    ppp = config.get("ppp")
    if ppp is None:
        ppp = default_compute_target_def.get("ppp")
    if ppp is not None:
        spec["ppp"] = str(ppp)

    source_pools = {wp["name"]: wp for wp in spec.get("workerPools", [])}
    if not source_pools:
        return spec

    # Resolve (compute_target, gpu, workers) per task, then group by (source_pool, compute_target, gpu).
    task_workers: dict[str, Optional[int]] = {}
    groups: dict[tuple[str, str, Optional[str]], list[dict]] = {}
    for task in spec.get("tasks", []):
        wp_name = task.get("workerPool")
        if not wp_name or wp_name not in source_pools:
            continue
        wp = source_pools[wp_name]
        canonical = _get_task_name_without_prefix(task.get("name", ""))
        override_compute_target, override_gpu, override_workers = _resolve_task_overrides(config, canonical)
        compute_target_key = override_compute_target or default_compute_target_key
        gpu: Optional[str] = None
        if _is_gpu_pool(wp):
            gpu = override_gpu or default_gpu_key
        task_workers[task["name"]] = override_workers
        groups.setdefault((wp_name, compute_target_key, gpu), []).append(task)

    groups_per_source: dict[str, int] = {}
    for src, _c, _g in groups.keys():
        groups_per_source[src] = groups_per_source.get(src, 0) + 1

    new_pools: list[dict] = []
    task_renames: dict[str, str] = {}

    for (src, compute_target_key, gpu), tasks_in_group in groups.items():
        if groups_per_source[src] == 1:
            derived = src
        else:
            suffix = gpu if gpu is not None else "cpu"
            derived = f"{src}-{compute_target_key}-{suffix}"

        wp = copy.deepcopy(source_pools[src])
        wp["name"] = derived
        wp.pop("cluster", None)
        wp.pop("resourceShare", None)

        compute_target = _resolve_compute_target(catalog, compute_target_key)
        field = str(compute_target.get("field"))
        target_name = str(compute_target.get("name"))
        if field == "cluster":
            wp["cluster"] = target_name
        elif field == "resource_share":
            wp["resourceShare"] = target_name
        else:
            raise ValueError(
                f"compute_targets.{compute_target_key}.field must be 'cluster' or 'resource_share', got '{field}'"
            )

        if gpu is not None:
            node_type = _resolve_node_type(compute_target, gpu)
            node_constraints = wp.setdefault("nodeConstraints", {})
            required = node_constraints.setdefault("required", {})
            required["nodeType"] = node_type

        # Pool sizing only bumps; it never decreases the source spec's existing worker count.
        group_workers = [workers for task in tasks_in_group if (workers := task_workers.get(task["name"])) is not None]
        target_workers = max(group_workers) if group_workers else None
        if target_workers is not None:
            current = wp.get("workers", 0)
            try:
                current_int = int(current)
            except (ValueError, TypeError):
                log.warning(f"Worker pool '{derived}' has non-numeric workers={current!r}, skipping bump")
                current_int = None
            if current_int is not None and current_int < target_workers:
                log.info(f"Bumping worker count for '{derived}' from {current_int} to {target_workers}")
                wp["workers"] = target_workers

        new_pools.append(wp)
        for task in tasks_in_group:
            task_renames[task["name"]] = derived

    spec["workerPools"] = new_pools
    for task in spec.get("tasks", []):
        renamed = task_renames.get(task.get("name", ""))
        if renamed is not None:
            task["workerPool"] = renamed

    return spec


def _get_base_workflow_components(
    config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig, target_key: str
) -> dict[str, omegaconf.dictconfig.DictConfig]:
    """Auto-detect base workflow component sections from config that contain a target config key.

    A base workflow component is a top-level config key whose value is a DictConfig
    containing both 'source_wf' and the specified target_key.

    Returns:
        Mapping of section_name -> section_config for matching sections.
    """
    if not isinstance(config, omegaconf.dictconfig.DictConfig):
        return {}
    sections: dict[str, omegaconf.dictconfig.DictConfig] = {}
    for key in config:
        key_str = str(key)
        section = getattr(config, key_str, None)
        if not isinstance(section, omegaconf.dictconfig.DictConfig):
            continue
        if getattr(section, "source_wf", None) is None:
            continue
        if getattr(section, target_key, None) is None:
            continue
        sections[key_str] = section
    return sections


def configure_task_retention_policies(
    spec: dict,
    config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig,
    component_task_names: dict[str, set[str]] | None = None,
) -> dict:
    """Configure retention policies for task outputs.

    Clears any existing retention policies and sets them based on config.
    Task-specific overrides from workflow sections take precedence over component defaults,
    which take precedence over the global default.

    Config structure:
        global:
            retention:
                default: "3d"  # Default retention for all task outputs

        car2sim:
            retention:
                default: "14d"  # Default retention for all car2sim tasks
                reconstruction: "45d"
                render_nre: "45d"

        nre_eval:
            retention:
                some_task: "30d"

    Args:
        component_task_names: Mapping of component name -> set of task names from that component's
            source workflow. Used to apply per-component default retention.
    """
    global_config = getattr(config, "global", None)
    if global_config is None:
        return spec

    global_retention = getattr(global_config, "retention", None)
    if global_retention is None:
        return spec

    default_retention = getattr(global_retention, "default", None)
    if default_retention is None:
        return spec

    # Set workflow-level retention policy
    spec["retentionPolicy"] = {"default": default_retention}

    # Collect task-specific overrides and per-component defaults from workflow components
    task_overrides: dict[str, str] = {}
    component_defaults: dict[str, str] = {}
    for section_name, section in _get_base_workflow_components(config, "retention").items():
        for task_name, retention in section.retention.items():
            if task_name == "default":
                component_defaults[section_name] = retention
            else:
                task_overrides[task_name] = retention

    if component_task_names is None:
        component_task_names = {}

    for task in spec.get("tasks", []):
        task_name = task.get("name", "")

        # Priority 1: task-specific override (substring match)
        retention_period = None
        for override_task_name, override_retention in task_overrides.items():
            if override_task_name in task_name:
                retention_period = override_retention
                log.info(f"Using override retention '{override_retention}' for task '{task_name}'")
                break

        # Priority 2: per-component default (if task belongs to that component)
        if retention_period is None:
            for component_name, component_default in component_defaults.items():
                if any(comp_task in task_name for comp_task in component_task_names.get(component_name, set())):
                    retention_period = component_default
                    log.info(f"Using {component_name} default retention '{component_default}' for task '{task_name}'")
                    break

        # Priority 3: global default
        if retention_period is None:
            retention_period = default_retention

        # Update existing retentionPolicy if present
        updated = False
        for output in task.get("outputs", []):
            if isinstance(output, dict) and "retentionPolicy" in output:
                output["retentionPolicy"]["default"] = retention_period
                updated = True
                break

        # If this task has a non-default retention but no existing retentionPolicy, add one
        if not updated and retention_period != default_retention:
            if not task.get("outputs"):
                task["outputs"] = [{"retentionPolicy": {"default": retention_period}}]
            else:
                task["outputs"].append({"retentionPolicy": {"default": retention_period}})

    return spec


def warn_for_missing_secrets(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig, secret_names: list[str]
) -> dict:
    secrets_set = set(secret_names)

    for task in spec["tasks"]:
        if "secrets" in task:
            filtered_secrets = []
            for secret in task["secrets"]:
                if secret["name"] in secrets_set:
                    filtered_secrets.append(secret)
                else:
                    log.warning(
                        f"Task {task['name']} has a secret that you do not have: {secret['name']}, omitting secret from spec"
                    )
            task["secrets"] = filtered_secrets

        # prevent saving artifacts in s3
        if "outputs" in task and config.car2sim.enabled and not config.car2sim.save_artifacts_in_s3:
            task["outputs"] = [
                output
                for output in task["outputs"]
                if not any(["storageSecret" in output_dict_key for output_dict_key in output])
            ]

    return spec


"""
fail_for_expired_inputs(spec: dict, maglev_toolbox: MaglevToolbox) -> None

fail_for_expired_inputs fails if any input tasks have expired. Uses maglev retention policy to check.
"""


def fail_for_expired_inputs(spec: dict, maglev_toolbox: MaglevToolbox) -> None:
    expired_tasks = []
    for task in spec["tasks"]:
        for input in task.get("inputs", []):
            if "task" in input and "workflow" in input and "run" in input:
                if maglev_toolbox.task_expired(input["workflow"], input["run"], input["task"]):
                    expired_tasks.append(f"{input['workflow']}/{input['run']}/{input['task']}")
    if len(expired_tasks) > 0:
        log.error(f"Could not launch wf, these input tasks have expired: {expired_tasks}")
        raise Exception(f"Could not launch wf, these input tasks have expired: {expired_tasks}")


"""
sanitize_spec(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict

sanitize_spec removes metadata from the spec.
"""


def sanitize_spec(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    spec.pop("userId", None)
    spec.pop("userEmail", None)
    spec.pop("createdAt", None)
    spec.pop("updatedTime", None)
    spec.pop("name", None)
    spec.pop("schedule", None)
    try:
        spec["tags"].pop("cost-center", None)
    except:
        log.warning("sanitize_spec: tags not found in spec, skipping cost-center removal")
    return spec


"""
build_template_from_config(config: dict, maglev_toolbox: MaglevToolbox) -> dict

build_template_from_config builds a template from our hydra config.
Templates can then be filled with a hydra config as well, but the template will use mostly original values from source maglev wf runs.

Mental model:
    We have 3 types of base workflows: car2sim, nre_eval, amo_cle.
    We then follow these steps:
    - Gather spec for each involved workflow directly from maglev
    - Cut each workflow spec, if necessary (cut_spec).
    - Join all workflows into a single workflow (join_specs).
    - Clean up some loose ends (secrets, worker pools) (sanitize_spec, remove_av5_things)
"""


def build_template_from_config(
    config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig, maglev_toolbox: MaglevToolbox
) -> dict:
    # Specs to be joined.
    specs = list()
    # Track which task names belong to which component (for per-component retention defaults)
    component_task_names: dict[str, set[str]] = {}

    # Car2Sim is the base of every workflow.
    if config.car2sim.enabled:
        car2sim_spec = maglev_toolbox.get_wf_spec(config.car2sim.source_wf)
        # Cut only parts that we need.
        if not config.car2sim.full_pipeline.enabled:
            cut = config.car2sim.cut
            car2sim_spec = cut_spec(car2sim_spec, cut, config.car2sim.source_wf)
        component_task_names["car2sim"] = {t["name"] for t in car2sim_spec.get("tasks", [])}
        specs.append(car2sim_spec)

    if config.nre_eval.enabled:
        # nre eval workflow construction here.
        nre_eval_spec = maglev_toolbox.get_wf_spec(config.nre_eval.source_wf)
        cut = config.nre_eval.cut
        nre_eval_spec = cut_spec(nre_eval_spec, cut, config.nre_eval.source_wf)
        component_task_names["nre_eval"] = {t["name"] for t in nre_eval_spec.get("tasks", [])}
        specs.append(nre_eval_spec)

    if config.nre_eval_regression.enabled:
        nre_eval_regression_spec = maglev_toolbox.get_wf_spec(config.nre_eval_regression.source_wf)
        cut = config.nre_eval_regression.cut
        nre_eval_regression_spec = cut_spec(nre_eval_regression_spec, cut, config.nre_eval_regression.source_wf)
        specs.append(nre_eval_regression_spec)

    if config.amo_cle.enabled:
        # amo-cle uses a shim workflow to join roadcast and our recomputed usdz artifacts into one output
        if config.amo_cle.shim_source_wf not in [None, ""]:
            shim_spec = maglev_toolbox.get_wf_spec(config.amo_cle.shim_source_wf)
            specs.append(shim_spec)
        # amo cle workflow construction here.
        amo_cle_spec = maglev_toolbox.get_wf_spec(config.amo_cle.source_wf)
        cut = ["base-pacsim-replay"] if config.amo_cle.cut in [None, []] else config.amo_cle.cut
        amo_cle_spec = cut_spec(amo_cle_spec, cut, config.amo_cle.source_wf)
        specs.append(amo_cle_spec)

    if config.nurec_cicd_amo_cle.enabled:
        nurec_cicd_amo_cle_spec = maglev_toolbox.get_wf_spec(config.nurec_cicd_amo_cle.source_wf)
        specs.append(nurec_cicd_amo_cle_spec)

    if hasattr(config, "failure_reason") and config.failure_reason.enabled:
        # Only add failure_reason workflow if base-pacsim-replay task will be present
        # Check if AMO CLE is enabled (which provides base-pacsim-replay)
        if config.amo_cle.enabled:
            # failure_reason workflow construction here.
            failure_reason_spec = maglev_toolbox.get_wf_spec(config.failure_reason.source_wf)
            cut = config.failure_reason.cut
            failure_reason_spec = cut_spec(failure_reason_spec, cut, config.failure_reason.source_wf)
            specs.append(failure_reason_spec)
        else:
            log.info("Skipping failure_reason workflow: base-pacsim-replay task not available (AMO CLE disabled)")

    # Generic workflow components
    generic_wf_components = config.get("generic_wf_components", {}) or {}  # type: ignore[arg-type]
    for component_name, component in generic_wf_components.items():
        if not getattr(component, "enabled", True):
            log.info(f"Skipping disabled generic workflow component: {component_name}")
            continue
        source_wf = getattr(component, "source_wf", None)
        if not source_wf:
            raise ValueError(f"Missing source_wf for generic workflow component: {component_name}")
        log.info(f"Adding generic workflow component: {component_name}")
        component_spec = maglev_toolbox.get_wf_spec(source_wf)
        cut = getattr(component, "cut", None)
        if cut:
            component_spec = cut_spec(component_spec, list(cut), component.source_wf)
        specs.append(component_spec)

    # Join all specs.
    joined_spec = join_specs(specs)

    for connection_name, connection in (config.switchboard.delete_connections or {}).items():
        try:
            assert "target_task" in connection, "Switchboard connection format is missing target_task."
            assert "remove_input_name" in connection, "Switchboard connection format is missing remove_input_name."
            pop_external_input(
                joined_spec, target_task_name=connection.target_task, remove_input_name=connection.remove_input_name
            )
        except AssertionError as e:
            if "force_enable" in connection and connection.force_enable:
                log.warning(f"Mandatory switchboard delete connection {connection_name} malformed: {e}")
                raise e
            else:
                log.warning(f"Non-mandatory switchboard delete connection {connection_name} malformed: {e}, skipping.")

    for connection_name, connection in (config.switchboard.connections or {}).items():
        # detect type of connection:
        try:
            if "with_external_wf" in connection:
                assert "target_task" in connection, "Switchboard external connection format is missing target_task."
                assert "replace_input_task" in connection, (
                    "Switchboard external connection format is missing replace_input_task."
                )
                replace_external_input(
                    joined_spec,
                    target_task_name=connection.target_task,
                    replace_input_task=connection.replace_input_task,
                    with_external_wf=connection.with_external_wf,
                    maglev_toolbox=maglev_toolbox,
                    external_task_override=connection.get("with_external_input", None),
                )
            elif "with_internal_input" in connection:
                assert "target_task" in connection, "Switchboard connection format is missing target_task."
                assert "replace_input_task" in connection, (
                    "Switchboard connection format is missing replace_input_task."
                )
                fallback_external_wf = (
                    None if "fallback_external_wf" not in connection else connection.fallback_external_wf
                )
                exact_match = connection.get("exact_match", False)
                convert_external_input_to_internal(
                    joined_spec,
                    target_task_name=connection.target_task,
                    replace_input_task=connection.replace_input_task,
                    with_internal_input=connection.with_internal_input,
                    fallback_external_wf=fallback_external_wf,
                    maglev_toolbox=maglev_toolbox,
                    exact_match=exact_match,
                )
            elif "add_external_inputs" in connection:
                assert "target_task" in connection, "Switchboard connection format is missing target_task."
                add_external_inputs_to_task(
                    joined_spec,
                    target_task_name=connection.target_task,
                    add_external_inputs=connection.add_external_inputs,
                )
            else:
                raise ValueError(
                    f"Connection does not have fields with_external_input or with_internal_input: {connection}"
                )
        except AssertionError as e:
            if "force_enable" in connection and connection.force_enable:
                log.warning(f"Mandatory switchboard connection with name {connection_name} malformed: {e}")
                raise e
            else:
                log.warning(
                    f"Non-mandatory switchboard connection with name {connection_name} malformed: {e}, skipping."
                )

    # Finally, some global configuration. Worker pool placement and GPU resolution run
    # later in run_ndas_workflow, after add_publish_dc_task, so the publish-dc pool
    # is also routed through the catalog.
    joined_spec = auto_fix_cuda_compat_in_script(joined_spec, config)
    joined_spec = sanitize_spec(joined_spec, config)
    joined_spec = configure_task_retention_policies(joined_spec, config, component_task_names)
    joined_spec = warn_for_missing_secrets(joined_spec, config, maglev_toolbox.get_available_secret_names())
    joined_spec = fix_hardcoded_run_ids(joined_spec)
    joined_spec = check_and_replace_regression_wf_selectors(joined_spec, config)
    fail_for_expired_inputs(joined_spec, maglev_toolbox)
    return joined_spec


def fill_clipgt_id_query(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    """Dispatch clipgt ID filling based on which task is present in the spec.

    Prefers session-list (lidar) over input-parser (lidarfree) when both are present.
    """
    task_names = {task["name"] for task in spec.get("tasks", [])}
    has_session_list = any("session-list" in name for name in task_names)
    has_input_parser = any("input-parser" in name for name in task_names)

    if has_session_list:
        spec = _fill_session_list_task(spec, config)
    elif has_input_parser:
        spec = _fill_input_parser_task(spec, config)

    return spec


def _fill_session_list_task(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    """Fill clipgt_ids in the session-list task for lidar workflows.

    The session-list task uses a SQL VALUES(...) pattern in its args.
    """
    if not config.car2sim.enabled or config.car2sim.full_pipeline.clipgt_ids in [None, "", []]:
        return spec

    if isinstance(config.car2sim.full_pipeline.clipgt_ids, str):
        clipgt_query_string = ", ".join([f"'{val}'" for val in config.car2sim.full_pipeline.clipgt_ids.split(",")])
    elif isinstance(config.car2sim.full_pipeline.clipgt_ids, (list, omegaconf.listconfig.ListConfig)):
        clipgt_query_string = ", ".join([f"'{val}'" for val in config.car2sim.full_pipeline.clipgt_ids])
    else:
        raise ValueError(f"Invalid type for clipgt_ids: {type(config.car2sim.full_pipeline.clipgt_ids)}")

    target_task_name = "session-list"
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
    target_task = target_task_candidates[0]
    pattern = r"\(VALUES\s*(.*?)\)"

    arg_idxes = [i for i, arg in enumerate(target_task["args"]) if re.search(pattern, arg)]
    assert len(arg_idxes) == 1, (
        f"Target task has more than one arg that matches clipid query pattern: {target_task_name}"
    )
    arg_idx = arg_idxes[0]
    replacement = f"(VALUES {clipgt_query_string})"
    target_task["args"][arg_idx] = re.sub(
        pattern, replacement, target_task["args"][arg_idx], flags=re.IGNORECASE | re.DOTALL
    )
    return spec


def _fill_input_parser_task(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    """Fill clipgt_ids in the input-parser task for lidarfree workflows.

    The input-parser task has hardcoded clipgt UUIDs in an echo statement.
    """
    if not config.car2sim.enabled:
        return spec

    sauron_clipgt_ids = getattr(config.car2sim, "sauron_clipgt_ids", None)
    if sauron_clipgt_ids is None:
        return spec

    # Normalize to list
    if isinstance(sauron_clipgt_ids, str):
        clipgt_list = [uid.strip() for uid in sauron_clipgt_ids.split(",") if uid.strip()]
    elif isinstance(sauron_clipgt_ids, (list, omegaconf.listconfig.ListConfig)):
        clipgt_list = list(sauron_clipgt_ids)
    else:
        raise ValueError(f"Invalid type for sauron_clipgt_ids: {type(sauron_clipgt_ids)}")

    if not clipgt_list:
        return spec

    target_task_name = "input-parser"
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    if len(target_task_candidates) == 0:
        log.warning(f"Could not fill lidarfree clipgt IDs, {target_task_name} not found in spec")
        return spec
    assert len(target_task_candidates) == 1, f"Multiple tasks found matching {target_task_name}"
    target_task = target_task_candidates[0]

    # Build replacement string in the same format: 'uuid1', 'uuid2', ...
    new_ids_str = ", ".join([f"'{uid}'" for uid in clipgt_list])

    # Match the echo line with hardcoded UUIDs in the input-parser script.
    # The line looks like: echo "'uuid1', 'uuid2', ..." | tr -d "'" | ...
    # We match from the echo up to the pipe to avoid over-capturing.
    uuid_echo_pattern = r"""echo\s+["'][^"'|]*['"](?=\s*\|)"""

    def _replace_in_script(script: str) -> tuple[str, bool]:
        match = re.search(uuid_echo_pattern, script)
        if match:
            log.info(f"Replacing lidarfree clipgt IDs in {target_task_name} with {len(clipgt_list)} IDs")
            return re.sub(uuid_echo_pattern, f'echo "{new_ids_str}"', script), True
        log.warning(f"Could not find clipgt UUID echo pattern in {target_task_name} script")
        return script, False

    if "script" in target_task:
        target_task["script"], _ = _replace_in_script(target_task["script"])
    elif "args" in target_task and len(target_task["args"]) > 0:
        target_task["args"][-1], _ = _replace_in_script(target_task["args"][-1])

    return spec


# -----------------------------------------------------------------------------
# Helper for filling RUN_BINARY continuation blocks in task scripts
# -----------------------------------------------------------------------------
# Tasks store bash scripts either as the last element of "args" or in "script".
# The script contains a line ending with "$RUN_BINARY \\" followed by continuation
# lines (each ending with "\\"). This helper replaces that continuation block
# with config-driven argument lines (used by car2sim reconstruction and NRM prediction).
# -----------------------------------------------------------------------------


def _fill_run_binary_continuation_in_task(
    target_task: dict,
    task_name: str,
    args_raw: list[str] | str,
    context_name: str,
) -> None:
    """Replace the RUN_BINARY continuation block in a task's script with the given args.

    Reads the task's script from task["args"][-1] or task["script"], finds the
    line ending with "$RUN_BINARY \\" and the following continuation block (lines
    ending with "\\"), replaces that block with args_raw formatted as bash
    continuation lines, and writes the result back. Mutates target_task in place.

    args_raw: list or str (legacy) of argument strings; will be formatted with
      indent and trailing " \\" on all but the last line.
    context_name: used in error messages (e.g. "reconstruction", "prediction").
    """
    assert isinstance(args_raw, (list, str)), f"args_raw must be list or str, got {type(args_raw)}"
    if isinstance(args_raw, str):
        args_raw = [args_raw]

    # Get script content and where it lives
    if "args" in target_task:
        script_lines = target_task["args"][-1].split("\n")
        task_field = "args"
    elif "script" in target_task:
        script_lines = target_task["script"].split("\n")
        task_field = "script"
    else:
        raise ValueError(f"Target task {task_name} has neither 'args' nor 'script' field")

    # Format args as bash continuation lines
    new_lines = []
    for idx, arg in enumerate(args_raw):
        new_lines.append(f"  {arg} \\" if idx < len(args_raw) - 1 else f"  {arg}")

    # Find and replace the continuation block
    start_idx = None
    for idx, line in enumerate(script_lines):
        if line.strip().endswith("$RUN_BINARY \\"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(
            f"Could not find start index for {context_name} args (line ending with '$RUN_BINARY \\') in script:\n"
            f"{script_lines}"
        )
    end_idx = None
    for idx in range(start_idx, len(script_lines)):
        if not script_lines[idx].strip().endswith("\\"):
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError(
            f"Could not find end index for {context_name} args (first line after $RUN_BINARY not ending with '\\') "
            f"in script:\n{script_lines}"
        )
    script_lines[start_idx + 1 : end_idx + 1] = new_lines

    # Write back
    if task_field == "args":
        target_task["args"][-1] = "\n".join(script_lines)
    else:
        target_task["script"] = "\n".join(script_lines)


def fill_sauron_model(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    """Override sauron model volume and/or checkpoint path in sauron-detection-tracking task.

    When ``checkpoint_path`` is provided, the hardcoded checkpoint path in the
    script is rewritten to use ``find`` with the given filename as the pattern,
    making it resilient to volume directory layout changes.

    Supports two script formats:
    - Original: hardcoded ``{{input}}/checkpoints/foo.ckpt`` path (transformed
      into a find block on first run).
    - Find-based: existing ``find {{input}}/checkpoints -name "..."`` block
      (checkpoint name is updated in-place).

    No-op when sauron_model is null or sauron_detection is not configured.

    Config (under generic_wf_components.sauron_detection):
        # Compact string: "volume:name:version" or "volume:name:version:checkpoint.ckpt"
        sauron_model: null
    """
    generic_wf_components = config.get("generic_wf_components", {}) or {}  # type: ignore[arg-type]
    sauron_detection = generic_wf_components.get("sauron_detection", None)
    if sauron_detection is None or not getattr(sauron_detection, "enabled", True):
        return spec

    sauron_model = getattr(sauron_detection, "sauron_model", None)
    if sauron_model is None:
        return spec

    # Parse compact string format: "volume:name:version[:checkpoint_path]"
    if not isinstance(sauron_model, str):
        log.warning(f"sauron_model must be a string, got {type(sauron_model)}")
        return spec

    parts = sauron_model.split(":")
    if len(parts) < 3 or parts[0] != "volume":
        raise ValueError(
            f"sauron_model must be 'volume:name:version' or 'volume:name:version:checkpoint.ckpt', got: {sauron_model}"
        )

    volume_name = parts[1]
    volume_version = parts[2]
    checkpoint_path = parts[3] if len(parts) >= 4 else None

    target_task_name = "sauron-detection-tracking"
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    if len(target_task_candidates) == 0:
        log.warning(f"Could not fill sauron model overrides, {target_task_name} not found in spec")
        return spec
    assert len(target_task_candidates) == 1, (
        f"Multiple tasks found matching {target_task_name}: {[t['name'] for t in target_task_candidates]}"
    )
    target_task = target_task_candidates[0]

    # Override volume name/version in inputs
    if volume_name is not None or volume_version is not None:
        for task_input in target_task.get("inputs", []):
            if "volume" in task_input:
                if volume_name is not None:
                    log.info(f"Overriding sauron model volume name to: {volume_name}")
                    task_input["volume"]["name"] = volume_name
                if volume_version is not None:
                    log.info(f"Overriding sauron model volume version to: {volume_version}")
                    task_input["volume"]["version"] = volume_version
                break

    # Rewrite checkpoint path to use `find` when checkpoint_path override is provided.
    if checkpoint_path is not None:

        def _get_script(task: dict) -> tuple[str | None, str]:
            if "script" in task:
                return task["script"], "script"
            if "args" in task and len(task["args"]) > 0:
                return task["args"][-1], "args"
            return None, ""

        script_content, script_field = _get_script(target_task)
        if script_content is None:
            log.warning(f"Could not find script in {target_task_name}")
            return spec

        ckpt_name = checkpoint_path

        # Check if the script already uses the find-based checkpoint format
        find_ckpt_pattern = r'find \{\{input\}\}/checkpoints -name "([^"]+)"'
        find_match = re.search(find_ckpt_pattern, script_content)

        if find_match:
            # Already has find-based format; replace the checkpoint name in-place
            old_ckpt_name = find_match.group(1)
            log.info(f"Updating existing sauron checkpoint from '{old_ckpt_name}' to '{ckpt_name}'")
            script_content = script_content.replace(old_ckpt_name, ckpt_name)
        else:
            # Original format: hardcoded checkpoint path (e.g. {{input}}/checkpoints/foo.ckpt)
            ckpt_pattern = r"(\{\{input\}\}/checkpoints/)\S+\.ckpt"
            match = re.search(ckpt_pattern, script_content)
            if not match:
                log.warning(f"Could not find checkpoint pattern in {target_task_name} {script_field}")
                return spec

            log.info(f"Using user-provided sauron checkpoint pattern: {ckpt_name}")

            find_replacement = (
                f'SAURON_CKPT=$(find {{{{input}}}}/checkpoints -name "{ckpt_name}" -type f | head -1)\n'
                f'    if [ -z "$SAURON_CKPT" ]; then\n'
                f"      echo \"ERROR: Could not find checkpoint '{ckpt_name}' in {{{{input}}}}/checkpoints\"\n"
                f"      find {{{{input}}}}/checkpoints -type f | head -20\n"
                f"      exit 1\n"
                f"    fi\n"
                f'    echo "Found sauron checkpoint: $SAURON_CKPT"'
            )

            # Replace the hardcoded path with $SAURON_CKPT and insert the find block before
            # the detection step
            detection_marker = "# Step 1: Detection"
            if detection_marker in script_content:
                script_content = script_content.replace(
                    detection_marker,
                    f"# Step 1: Find checkpoint\n    {find_replacement}\n\n    # Step 2: Detection",
                )
                script_content = re.sub(ckpt_pattern, "$SAURON_CKPT", script_content)
                # Re-number step 2 tracking -> step 3
                script_content = script_content.replace("# Step 2: Tracking", "# Step 3: Tracking")
            else:
                raise RuntimeError(
                    f"Could not find detection step marker '{detection_marker}' in {target_task_name} script. "
                    f"Cannot safely inject checkpoint find logic."
                )

        if script_field == "script":
            target_task["script"] = script_content
        else:
            target_task["args"][-1] = script_content

    return spec


"""
fill_car2sim_reconstruction_args(spec: dict, config: dict) -> dict

Fills the reconstruction task's bash script by replacing the argument block that
follows the "$RUN_BINARY \\" line with config.car2sim.reconstruction_args.
Only runs when config.car2sim.enabled and reconstruction_args is set.
"""


def fill_car2sim_reconstruction_args(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    if not config.car2sim.enabled or config.car2sim.reconstruction_args is None:
        return spec

    target_task_name = "reconstruction"
    if hasattr(config.car2sim, "nurec_task_name") and config.car2sim.nurec_task_name:
        target_task_name = config.car2sim.nurec_task_name

    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
    target_task = target_task_candidates[0]

    reconstruction_args = omegaconf.OmegaConf.to_container(config.car2sim.reconstruction_args)
    assert isinstance(reconstruction_args, (list, str)), (
        f"Invalid type for reconstruction_args: {type(config.car2sim.reconstruction_args)}, expected list or str"
    )
    _fill_run_binary_continuation_in_task(target_task, target_task_name, reconstruction_args, "reconstruction")
    return spec


"""
fill_nrm_prediction_args(spec: dict, config: dict) -> dict

Fills the nrm-prediction task's bash script by replacing the argument block that
follows the "$RUN_BINARY \\" line with config.nrm.prediction_args.
Only runs when config.nrm.enabled and prediction_args is set.
If the nrm-prediction task is not found in the spec, logs a warning and returns spec unchanged.
"""


def fill_nrm_prediction_args(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    if not hasattr(config, "nrm") or not config.nrm.enabled or config.nrm.prediction_args is None:
        return spec

    target_task_name = "nrm-prediction"
    if hasattr(config.nrm, "nrm_task_name") and config.nrm.nrm_task_name:
        target_task_name = config.nrm.nrm_task_name

    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    if len(target_task_candidates) == 0:
        log.warning(f"Could not fill NRM prediction args, {target_task_name} not found in spec")
        return spec
    assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
    target_task = target_task_candidates[0]

    prediction_args = omegaconf.OmegaConf.to_container(config.nrm.prediction_args)
    assert isinstance(prediction_args, (list, str)), (
        f"Invalid type for prediction_args: {type(config.nrm.prediction_args)}, expected list or str"
    )
    _fill_run_binary_continuation_in_task(target_task, target_task_name, prediction_args, "prediction")
    return spec


"""
fill_training_config(spec: dict, config: dict) -> dict

fill_training_config fills the training config input for the reconstruction task.
Also fills in reconstruction args, if any.
Also sets gpu world size.
(e.g. alpasim_3dgut.yaml, etc)
"""


# TODO @josliu: This function should be phased out once we move lidarfree to use fill_car2sim_reconstruction_args instead.
# workerpool configuration will be done in configure_worker_pools_and_tasks, and training_config and world_size will be added to
# reconstruction_args in fill_car2sim_reconstruction_args.
def fill_training_config(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    if config.car2sim.enabled and config.car2sim.training_config is not None:
        target_task_name = "reconstruction"
        if hasattr(config.car2sim, "nurec_task_name") and config.car2sim.nurec_task_name:
            target_task_name = config.car2sim.nurec_task_name

        target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
        assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
        target_task = target_task_candidates[0]

        patterns = [r"--config-name=([^\s'\"=]+)\.ya?ml", r"--config-name=\$CONFIG_NAME"]

        # Determine whether task uses 'args' or 'script' and detect pattern
        if "args" in target_task:
            task_field = "args"
            for pattern in patterns:
                arg_idxes = [i for i, arg in enumerate(target_task["args"]) if re.search(pattern, arg)]
                if arg_idxes:
                    assert len(arg_idxes) == 1, (
                        f"Target task {target_task_name} has {len(arg_idxes)} matches for pattern, expected exactly 1"
                    )
                    arg_idx = arg_idxes[0]
                    task_content = target_task["args"][arg_idx]
                    break
            else:
                raise ValueError(f"Target task {target_task_name} args do not contain --config-name pattern")
        elif "script" in target_task:
            task_field = "script"
            task_content = str(target_task["script"])
            assert task_content, f"Target task {target_task_name} script is empty"
            for pattern in patterns:
                if re.search(pattern, task_content):
                    break
            else:
                raise ValueError(f"Target task {target_task_name} script does not contain --config-name pattern")
            arg_idx = None
        else:
            raise ValueError(f"Target task {target_task_name} has neither 'args' nor 'script' field")

        additional_args = ""
        if config.car2sim.reconstruction_args in [None, ""]:
            # attempt to retrieve from env var. Alternative to bypass bash escaping.
            additional_args = os.environ.get("RECONSTRUCTION_ARGS", "")
        # Include warning for when training_config is set and reconstruction_args are both set.
        if isinstance(config.car2sim.reconstruction_args, omegaconf.listconfig.ListConfig):
            if not any(["--config-name=" in arg for arg in config.car2sim.reconstruction_args]):
                log.warning(
                    "WARNING: training_config is set, but reconstruction_args does not contain --config-name pattern. training_config will not have any effect."
                )
            else:
                log.warning(
                    "WARNING: training_config is set, and reconstruction_args also contains a --config-name pattern. training_config will take precedence over reconstruction_args."
                )

        replacement = f"--config-name={config.car2sim.training_config}"
        if additional_args:
            replacement = f"{replacement} {additional_args}"
        if config.car2sim.gpu_world_size != 1:
            replacement = f"{replacement} trainer.world_size={config.car2sim.gpu_world_size}"

        updated_content = re.sub(pattern, replacement, task_content, flags=re.DOTALL)
        if task_field == "args":
            target_task["args"][arg_idx] = updated_content
        else:
            target_task["script"] = updated_content

        # set gpu world size
        try:
            target_workerpool_name = target_task["workerPool"]
            target_workerpool_candidates = [wp for wp in spec["workerPools"] if wp["name"] == target_workerpool_name]
            assert len(target_workerpool_candidates) == 1, (
                f"Target workerpool not found in spec: {target_workerpool_name}"
            )
            target_workerpool = target_workerpool_candidates[0]
            target_workerpool["gpu"] = str(config.car2sim.gpu_world_size)
        except Exception as e:
            log.warning(f"Could not set gpu world size: {e}")
            if config.car2sim.gpu_world_size != 1:
                raise e

    return spec


"""
fill_metadata(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict

fill_metadata fills the metadata input for the metadata task.
"""


def fill_metadata(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    if os.environ.get("GITLAB_CI", None) is not None:
        if "tags" not in spec:
            spec["tags"] = {}
        spec["tags"]["nre_gitlab_triggered_by_user"] = os.environ.get("GITLAB_USER_LOGIN", None)
        spec["tags"]["nre_gitlab_job_name"] = os.environ.get("CI_JOB_NAME", None)
        spec["tags"]["nre_gitlab_job_id"] = os.environ.get("CI_JOB_ID", None)
        if "CI_COMMIT_BRANCH" in os.environ:
            spec["tags"]["nre_gitlab_branch_name"] = os.environ.get("CI_COMMIT_BRANCH", None)
        if "CI_COMMIT_REF_NAME" in os.environ:
            spec["tags"]["nre_gitlab_commit_ref_name"] = os.environ.get("CI_COMMIT_REF_NAME", None)
        if "CI_COMMIT_SHA" in os.environ:
            spec["tags"]["nre_gitlab_commit_sha"] = os.environ.get("CI_COMMIT_SHA", None)

    # Discover cross-project metadata passed via NUREC_WF_METADATA_* env vars.
    # Each var is a JSON object keyed by source project (e.g. NUREC_WF_METADATA_SAURON).
    for key, value in os.environ.items():
        if key.startswith("NUREC_WF_METADATA_"):
            source = key.removeprefix("NUREC_WF_METADATA_").lower()
            try:
                metadata = json.loads(value)
                if "tags" not in spec:
                    spec["tags"] = {}
                for field, field_value in metadata.items():
                    if field_value:
                        spec["tags"][f"{source}_gitlab_{field}"] = field_value
            except json.JSONDecodeError:
                log.warning(f"fill_metadata: failed to parse {key} as JSON, skipping")

    return spec


"""
fix_hardcoded_run_ids(spec: dict) -> dict

fix_hardcoded_run_ids replaces hardcoded run IDs with latest-success to prevent expiration issues.
"""


def fix_hardcoded_run_ids(spec: dict) -> dict:
    """Replace hardcoded run IDs with latest-success to prevent expiration issues."""
    hardcoded_run_ids = [
        "2025.09.07-0000-1c1vbrnllx7se",  # Known expired run ID
    ]

    for task in spec["tasks"]:
        for input in task.get("inputs", []):
            if "workflow" in input and "run" in input and input["run"] in hardcoded_run_ids:
                old_run_id = input["run"]
                input["run"] = "latest-success"
                log.info(f"Fixed hardcoded run ID {old_run_id} to latest-success in task {task.get('name', 'unknown')}")

    return spec


"""
check_and_replace_regression_wf_selectors(spec: dict, config: dict) -> dict

check_and_replace_regression_wf_selectors fixes selectors for regression workflows.
"""


def check_and_replace_regression_wf_selectors(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    if not config.nre_eval_regression.enabled:
        return spec

    for task in spec["tasks"]:
        if "merge_synthetic_data_into_clip" in task["name"]:
            if "selector" in task and task["selector"] == r"(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})/":
                task["selector"] = r"(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})"
    return spec


"""
fill_template(spec: dict, config: dict) -> dict

fill_template fills a template with a hydra config.
"""


def fill_template(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    def replace_image(spec_item: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig):
        """
        spec_item only needs to contain an "image" key.
        """
        if "image" not in spec_item:
            raise Exception(f"spec_item {spec_item} does not contain an image key")

        if (
            any([substring in spec_item["image"] for substring in config.replace_nre_images_that_contain_substrings])
            and config.docker.image is not None
        ):
            spec_item["image"] = config.docker.image
        elif (
            any([substring in spec_item["image"] for substring in config.replace_tools_images_that_contain_substrings])
            and config.docker.tools_image is not None
        ):
            spec_item["image"] = config.docker.tools_image
        elif (
            any(
                [
                    substring in spec_item["image"]
                    for substring in config.replace_clipgt_ncore_images_that_contain_substrings
                ]
            )
            and config.docker.clipgt_ncore_image is not None
        ):
            spec_item["image"] = config.docker.clipgt_ncore_image
        elif (
            any([substring in spec_item["image"] for substring in config.replace_nrm_images_that_contain_substrings])
            and config.docker.nrm_image is not None
        ):
            spec_item["image"] = config.docker.nrm_image
        elif (
            any([substring in spec_item["image"] for substring in config.replace_sauron_images_that_contain_substrings])
            and config.docker.sauron_image is not None
        ):
            spec_item["image"] = config.docker.sauron_image
        else:
            # Generic image replacement rules from generic_wf_components
            generic_components = config.get("generic_wf_components", {}) or {}  # type: ignore[arg-type]
            for component_name, component in generic_components.items():
                if not getattr(component, "enabled", True):
                    continue
                for rule in getattr(component, "image_replacement_rules", []) or []:
                    target_image = rule.get("image", None)
                    if target_image is None:
                        continue
                    match_substrings = rule.get("match_substrings", [])
                    if any(substring in spec_item["image"] for substring in match_substrings):
                        spec_item["image"] = target_image
                        return

    def replace_hanging_nre_version_tags(
        spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
    ):
        if config.docker.image is None:
            return spec
        nre_version_tag = config.docker.image.split(":")[-1]
        assert nre_version_tag != ""

        placeholder_tag = ""
        # Version pattern: major.minor.patch-8hexchars with optional suffix (e.g., -lidarfree)
        version_pattern = r"\d+\.\d+\.\d+-[a-fA-F0-9]{8}(?:-[a-zA-Z0-9_]+)?"
        attempted_search_locations = [
            ("swiftstack-mount-nre", rf"SEGMENT_ID/nre/({version_pattern})/"),
            ("swiftstack-uploader-nre", rf"MODULE_NAME/({version_pattern})"),
            ("usdz-porting", rf"NRE_VERSION=({version_pattern})"),
            ("video-generation", rf"ego-trajectory-videos/({version_pattern})"),
            ("coarse-validation", rf"nre_version=({version_pattern})"),
        ]

        for attempted_search_location in attempted_search_locations:
            for task in spec["tasks"]:
                if attempted_search_location[0] not in task["name"]:
                    continue
                # target args[-1] which contains our bash script
                if "args" in task:
                    script = task["args"][-1]
                    if not script:
                        continue
                    # replace placeholder_tag with nre_version_tag
                    match = re.search(attempted_search_location[1], script)
                    if match is not None:
                        placeholder_tag = match.group(1)
                        log.info(f"found placeholder tag {placeholder_tag} for {attempted_search_location}")
                        break  # Found placeholder, stop searching
                    else:
                        log.warning(f"Could not find placeholder tag for {attempted_search_location[0]}")
                        continue  # Continue searching other task types

        log.info(f"Replacing placeholder tag {placeholder_tag} with {nre_version_tag}")

        if placeholder_tag == "":
            log.warning("Could not find placeholder tag for swiftstack-mount-nre")
            return spec

        for task in spec["tasks"]:
            replacement = nre_version_tag
            if config.swiftstack_version_suffix is not None and any(
                [target_task in task["name"] for target_task in config.swiftstack_version_suffix_tasks]
            ):
                log.info(
                    f"Replacing placeholder tag {placeholder_tag} with {nre_version_tag}-{config.swiftstack_version_suffix}"
                )
                replacement = f"{nre_version_tag}-{config.swiftstack_version_suffix}"

            # target args[-1] which contains our bash script
            if "args" in task and len(task["args"]) > 0:
                script = task["args"][-1]
                if not script:
                    continue
                # replace placeholder_tag with nre_version_tag
                if placeholder_tag in script:
                    script = re.sub(re.escape(str(placeholder_tag)), replacement, script)
                    task["args"][-1] = script

        # Set DOCKER_IMAGE_TAG in trigger-gitlab-pipeline task to match the configured docker image tag
        # This ensures the triggered GitLab pipeline uses the correct image tag (including any suffix like -dev)
        try_replace_line_in_task_script(
            spec,
            "trigger-gitlab-pipeline",
            r"DOCKER_IMAGE_TAG=.*",
            f"DOCKER_IMAGE_TAG={nre_version_tag}",
        )

        return spec

    # Fill in the image for NRE-RUN tasks.
    for task in spec["tasks"]:
        if "image" not in task:
            for container in task.get("containers", []):
                replace_image(container, config)
        else:
            replace_image(task, config)

    spec = fill_clipgt_id_query(spec, config)
    spec = fill_sauron_model(spec, config)
    spec = fill_car2sim_reconstruction_args(spec, config)
    spec = fill_nrm_prediction_args(spec, config)
    spec = fill_training_config(spec, config)
    spec = fill_amo_cle_renderer_args(spec, config)
    spec = fill_run_binaries(spec, config)
    spec = apply_string_replacements(spec, config)
    spec = add_timing_log_collection(spec)
    spec = fill_metadata(spec, config)
    spec = replace_hanging_nre_version_tags(spec, config)
    return spec


"""
fill_amo_cle_renderer_args(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict

fill_amo_cle_renderer_args fills the nre renderer args for the amo-cle workflow.
"""


def fill_amo_cle_renderer_args(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    if config.amo_cle.enabled and config.amo_cle.renderer_args is not None:
        target_task_name = "base-pacsim-replay"
        target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
        if len(target_task_candidates) == 0:
            log.warning(f"Could not fill in amo cle renderer args, {target_task_name} not found in spec")
            return spec
        assert len(target_task_candidates) <= 1, (
            f"Too many tasks found in spec: {[task['name'] for task in target_task_candidates]}"
        )
        target_task = target_task_candidates[0]

        # find nre container:
        target_container_name = "nre"
        target_container_candidates = [
            container for container in target_task.get("containers", []) if target_container_name in container["name"]
        ]
        assert len(target_container_candidates) == 1, f"Target container not found in spec: {target_container_name}"
        target_container = target_container_candidates[0]

        # find renderer args arg. replace enable-nrend if exists, otherwise insert after serve-grpc
        script = target_container["script"]
        if "--enable-nrend" in script:
            script = re.sub(r"--enable-nrend", f"{config.amo_cle.renderer_args}", script)
        else:
            script = re.sub(r"serve-grpc", f"serve-grpc {config.amo_cle.renderer_args}", script)
        target_container["script"] = script
    return spec


"""
try_replace_line_in_task_script(spec: dict, task_name: str, line: str, replacement: str) -> dict

try_replace_line_in_task_script tries to replace a line in a task script.
E.g. RUN_BINARY=.../some_arbitrary_string -> RUN_BINARY=.../replacement_string
If the task is not found, it logs a warning and returns the spec unchanged.
If the task has no script or args key, it logs a warning and returns the spec unchanged.
If the task has more than one task with the same name, it logs a warning and returns the spec unchanged.
If the line is not found, it logs a warning and returns the spec unchanged.
If the replacement is not found, it logs a warning and returns the spec unchanged.
If the replacement is not found, it logs a warning and returns the spec unchanged.
"""


def apply_string_replacements(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    """Apply generic regex string replacements to task scripts.

    Reads ``string_replacements`` from both ``generic_wf_components.<component>``
    and ``car2sim``.

    Config:
        string_replacements:
          <name>:
            task_name: <task-name-substring>
            pattern: <regex-pattern>
            replacement: <replacement-string>
    """

    def _apply(string_replacements, source_name: str) -> None:
        for rule_name, rule in string_replacements.items():
            task_name = rule.get("task_name", None)
            pattern = rule.get("pattern", None)
            replacement = rule.get("replacement", None)
            if not all([task_name, pattern, replacement is not None]):
                log.warning(f"Incomplete string_replacement rule '{rule_name}' in {source_name}, skipping")
                continue
            log.info(f"Applying string replacement '{rule_name}' to task '{task_name}': {pattern} -> {replacement}")
            try_replace_line_in_task_script(spec, task_name, pattern, replacement)

    # generic_wf_components
    generic_wf_components = config.get("generic_wf_components", {}) or {}  # type: ignore[arg-type]
    for component_name, component in generic_wf_components.items():
        if not getattr(component, "enabled", True):
            continue
        string_replacements = getattr(component, "string_replacements", None)
        if string_replacements:
            _apply(string_replacements, f"generic_wf_components.{component_name}")

    # car2sim
    car2sim = config.get("car2sim", None)  # type: ignore[arg-type]
    if car2sim is not None and getattr(car2sim, "enabled", True):
        string_replacements = getattr(car2sim, "string_replacements", None)
        if string_replacements:
            _apply(string_replacements, "car2sim")

    return spec


def try_replace_line_in_task_script(spec: dict, task_name: str, line: str, replacement: str) -> dict:
    target_task_candidates = [task for task in spec["tasks"] if task["name"].endswith(task_name)]
    if len(target_task_candidates) == 0:
        log.warning(f"Could not replace line in task {task_name}, task not found in spec")
        return spec

    if len(target_task_candidates) > 1:
        log.warning(
            f"Too many tasks found in spec: {[task['name'] for task in target_task_candidates]}, selecting first task: {target_task_candidates[0]['name']}"
        )
    target_task = target_task_candidates[0]

    if "script" in target_task:
        target_task["script"] = re.sub(line, replacement, target_task["script"])
    elif "args" in target_task:
        target_task["args"] = [re.sub(line, replacement, arg) for arg in target_task["args"]]
    elif "containers" in target_task:  # special case for multi container tasks
        for container in target_task["containers"]:
            if "script" in container:
                container["script"] = re.sub(line, replacement, container["script"])
            else:
                log.warning(f"Container {container['name']} in task {task_name} does not have a script, skipping")
    else:
        log.warning(f"Task {task_name} does not have a script or args key, skipping")
    return spec


def fill_run_binaries(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    if config.reconstruction_run_binary_tasks is None or config.nre_tools_run_binary_tasks is None:
        log.warning(
            "reconstruction_run_binary_tasks or nre_tools_run_binary_tasks is not set in config, skipping fill_run_binaries"
        )
        return spec

    for task in config.reconstruction_run_binary_tasks:
        try_replace_line_in_task_script(spec, task, r"RUN_BINARY=.*", f"RUN_BINARY={config.reconstruction_run_binary}")
        try_replace_line_in_task_script(spec, task, r"REPO_PATH=.*", f"REPO_PATH={config.reconstruction_repo_path}")

    for task in config.nre_tools_run_binary_tasks:
        try_replace_line_in_task_script(spec, task, r"RUN_BINARY=.*", f"RUN_BINARY={config.nre_tools_run_binary}")
        try_replace_line_in_task_script(spec, task, r"REPO_PATH=.*", f"REPO_PATH={config.nre_tools_repo_path}")
    return spec


"""
auto_fix_cuda_compat_in_script(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict

auto_fix_cuda_compat_in_script fixes cuda compat in the script.
"""


def fix_cuda_library_versions(script: str) -> str:
    """
    Helper function to fix CUDA library version references in scripts.
    Replaces versioned .so files with their generic counterparts.

    Args:
        script: The script content to fix

    Returns:
        The script with fixed CUDA library references
    """
    # Define the library replacements as a list of (pattern, replacement) tuples
    cuda_lib_replacements = [
        (r"libcuda\.so\.\d+\.\d+\.\d+", "libcuda.so"),
        (r"libnvidia-nvvm\.so\.\d+\.\d+\.\d+", "libnvidia-nvvm.so"),
        (r"libnvidia-ptxjitcompiler\.so\.\d+\.\d+\.\d+", "libnvidia-ptxjitcompiler.so.1"),
    ]

    # Apply all replacements
    for pattern, replacement in cuda_lib_replacements:
        script = re.sub(pattern, replacement, script)

    return script


def auto_fix_cuda_compat_in_script(
    spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
) -> dict:
    if not config.auto_fix_cuda_compat:
        return spec

    for task in spec["tasks"]:
        try:
            # Fix scripts in containers
            if "containers" in task:
                for container in task["containers"]:
                    if "script" in container:
                        container["script"] = fix_cuda_library_versions(container["script"])

            # Fix script at task level
            if "script" in task:
                task["script"] = fix_cuda_library_versions(task["script"])

        except Exception as e:
            log.warning(f"Error fixing cuda compat in script for task {task['name']}: {e}")
    return spec


"""
add_timing_log_collection(spec: dict) -> dict

add_timing_log_collection adds timing log collection to the reconstruction task if it exists.
"""


def add_timing_log_collection(spec: dict) -> dict:
    target_task_name = "reconstruction"
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]

    if len(target_task_candidates) == 0:
        log.warning(f"Could not add timing log collection, {target_task_name} not found in spec")
        return spec

    target_task = target_task_candidates[0]

    if "args" in target_task and len(target_task["args"]) > 3 and isinstance(target_task["args"][3], str):
        original_script = target_task["args"][3]  # The script is in args[3]
    else:
        log.warning(f"Could not add timing log collection, task args structure unexpected: {target_task['name']}")
        return spec

    # Add timing log collection commands at the end
    timing_log_commands = """

TIMING_LOG_PATH=$(find "$WORK_RECONSTRUCTION_DIR" -type f -name "timing.log")
if [ -n "$TIMING_LOG_PATH" ] && [ -f "$TIMING_LOG_PATH" ]; then
  mv "$TIMING_LOG_PATH" "$OUTPUT_USD_DIR"
fi"""

    modified_script = original_script + timing_log_commands
    target_task["args"][3] = modified_script

    log.info(f"Added timing log collection to task: {target_task['name']}")

    return spec


"""
convert_external_input_to_internal(spec: dict,
    target_task_name: str,
    replace_input_task: list[str],
    with_internal_input: str,
    fallback_external_wf: Optional[str] = None,
    exact_match: bool = False,
) -> dict

convert_external_input_to_internal redirects external inputs to internal inputs.
In the case where one workflow originally depended on another (i.e. external input):
After joining, we can use this function to redirect inputs as they are now in the same workflow (i.e. internal).

Example:
convert_external_input_to_internal(spec,
    target_task_name="create_clips_parallel_structure",
    internal_task_names=["reconstruction"])

After you've joined two worfklows into one, you probably want to direct the outputs of one WF to the inputs of another.
Use this function to redirect inputs. Task names are substrings due to task numbering.
Set exact_match=True to match task names exactly (useful when multiple tasks share similar substrings).
"""


def _get_task_name_without_prefix(full_name: str) -> str:
    """Extract task name without prefixes added by join_specs or numeric step prefixes.

    Handles:
    - 'wf0_12.rr_entrypoint_synth' -> 'rr_entrypoint_synth' (join_specs prefix + numeric with dot)
    - '12.rr_entrypoint_synth' -> 'rr_entrypoint_synth' (numeric prefix with dot)
    - '15-clipgt-ncore' -> 'clipgt-ncore' (numeric prefix with hyphen, Maglev format)
    - 'rr_entrypoint_synth' -> 'rr_entrypoint_synth' (no prefix)
    - 'clipgt-ncore' -> 'clipgt-ncore' (no numeric prefix, hyphen is part of name)
    """
    name = full_name

    # First, strip optional 'wf{idx}_' prefix added by join_specs for collision handling
    if name.startswith("wf") and "_" in name:
        # Check if it matches pattern 'wf{digit(s)}_'
        underscore_idx = name.index("_")
        potential_prefix = name[2:underscore_idx]  # Extract part between 'wf' and '_'
        if potential_prefix.isdigit():
            name = name[underscore_idx + 1 :]  # Strip 'wf{idx}_' prefix

    # Strip numeric step prefix with dot (e.g., '12.' from '12.task_name')
    if "." in name:
        parts = name.split(".", 1)
        if parts[0].isdigit():
            return parts[1]

    # Strip numeric step prefix with hyphen (e.g., '15-' from '15-clipgt-ncore')
    if "-" in name:
        parts = name.split("-", 1)
        if parts[0].isdigit():
            return parts[1]

    return name


def _find_task_candidates(tasks: list[dict], task_name: str, exact_match: bool) -> list[dict]:
    """Helper to find task candidates based on exact or substring matching.

    When exact_match is True, matches task name without the numeric prefix
    (e.g., 'rr_entrypoint_synth' matches '12.rr_entrypoint_synth').
    When exact_match is False, uses substring matching.
    """
    if exact_match:
        return [task for task in tasks if _get_task_name_without_prefix(task["name"]) == task_name]
    return [task for task in tasks if task_name in task["name"]]


def _format_task_not_found_error(task_name: str, candidates: list[dict], task_type: str, exact_match: bool) -> str:
    """Helper to format error message when task is not found or multiple matches exist."""
    if len(candidates) == 0:
        return f"{task_type} task not found in spec: {task_name}"
    else:
        candidate_names = [task["name"] for task in candidates]
        return (
            f"Multiple tasks found matching '{task_name}': {candidate_names}. "
            f"Consider setting 'exact_match: true' in the switchboard connection config "
            f"and using the full task name."
        )


def convert_external_input_to_internal(
    spec: dict,
    target_task_name: str,
    replace_input_task: list[str],
    with_internal_input: str,
    fallback_external_wf: Optional[str] = None,
    maglev_toolbox=None,
    exact_match: bool = False,
) -> dict:
    assert len(replace_input_task) > 0, "replace_input_task must be non-empty"

    internal_task_candidates = _find_task_candidates(spec["tasks"], with_internal_input, exact_match)
    target_task_candidates = _find_task_candidates(spec["tasks"], target_task_name, exact_match)

    if len(internal_task_candidates) == 0 and fallback_external_wf is not None:
        replace_external_input(spec, target_task_name, replace_input_task, fallback_external_wf, maglev_toolbox)
        return spec

    assert len(internal_task_candidates) == 1, _format_task_not_found_error(
        with_internal_input, internal_task_candidates, "Internal", exact_match
    )
    assert len(target_task_candidates) == 1, _format_task_not_found_error(
        target_task_name, target_task_candidates, "Target", exact_match
    )

    internal_task = internal_task_candidates[0]
    target_task = target_task_candidates[0]

    if "inputs" not in target_task:
        target_task["inputs"] = []

    for target_input in target_task["inputs"]:
        if "task" in target_input and any(
            [replace_input_task_name in target_input["task"] for replace_input_task_name in replace_input_task]
        ):
            if "workflow" not in target_input:
                log.warning(
                    f"Input for target task '{target_task_name}' referencing '{target_input['task']}' "
                    f"is already internal (no 'workflow' key). Skipping rewire."
                )
                continue
            target_input["task"] = internal_task["name"]
            target_input.pop("workflow")
            target_input.pop("run")

    return spec


"""
add_external_input_to_task(spec: dict, target_task_name: str, add_external_input: dict) -> dict

add_external_input_to_task adds an arbitrary external input to a task. The add_external_input field
is converted to a raw dict and used directly in the maglev spec.

Example:
switchboard:
  connections:
    nrm_volume_to_car2sim:
      force_enable: false
      target_task: reconstruction
      add_external_input:
      - url: swift://pdx.s8k.io/team-ncore/nrm-init/05a61f9db
        storageSecret: team-ncore-s3
        selector: ().*
        cross: all
        prefix: temp_test
"""


def add_external_inputs_to_task(spec: dict, target_task_name: str, add_external_inputs: dict) -> dict:
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    assert len(target_task_candidates) == 1, (
        f"Target task not found in spec, or too many tasks found: {target_task_name}, {target_task_candidates}"
    )
    target_task = target_task_candidates[0]

    converted = omegaconf.OmegaConf.to_container(add_external_inputs, resolve=True)
    assert type(converted) == list, (
        f"Expected add_external_inputs to be list in connection for {target_task_name}, got {type(converted)} instead"
    )

    if "inputs" not in target_task:
        target_task["inputs"] = converted
    else:
        target_task["inputs"] += converted

    return spec


"""
pop_external_input(spec: dict, target_task_name: str, remove_input_name: str) -> dict

pop_external_input removes external inputs from a task that contain task substring remove_input_name.
"""


def pop_external_input(
    spec: dict,
    target_task_name: str,
    remove_input_name: str,
) -> dict:
    target_task_candidates = [task for task in spec["tasks"] if target_task_name in task["name"]]
    assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
    target_task = target_task_candidates[0]

    filtered_inputs = [
        task_input
        for task_input in target_task["inputs"]
        if "task" not in task_input or remove_input_name not in task_input["task"]
    ]
    target_task["inputs"] = filtered_inputs
    return spec


"""
replace_external_input(spec: dict,
    target_task_name: str,
    replace_input_task: str,
    with_external_wf: str,
) -> dict

replace_external_input redirects external inputs to internal inputs.
external_input_name is a workflow run with format "workflow/run".
"""


def replace_external_input(
    spec: dict,
    target_task_name: str,
    replace_input_task: list[str],
    with_external_wf: str,
    maglev_toolbox=None,
    external_task_override: Optional[Union[str, list[str]]] = None,
) -> dict:
    external_input = with_external_wf.split("/")
    assert len(external_input) == 2, f"External input must be in the format workflow/run: {with_external_wf}"

    target_task_candidates = _find_task_candidates(spec["tasks"], target_task_name, exact_match=False)
    assert len(target_task_candidates) == 1, f"Target task not found in spec: {target_task_name}"
    target_task = target_task_candidates[0]

    # Get the external workflow spec to check what tasks are actually available
    external_tasks: list[dict] = []
    if maglev_toolbox is not None:
        external_spec = maglev_toolbox.get_wf_spec(with_external_wf)
        external_tasks = external_spec["tasks"]

    for target_input in target_task["inputs"]:
        if "task" in target_input and any(
            [replace_input_task_name in target_input["task"] for replace_input_task_name in replace_input_task]
        ):
            target_input["workflow"], target_input["run"] = external_input

            # If explicit override is provided, use it
            if external_task_override:
                # Support list of task names - try each until one matches in external workflow
                if isinstance(external_task_override, (list, omegaconf.ListConfig)):
                    candidates = list(external_task_override)
                    matched_task_name = None
                    for candidate in candidates:
                        matches = _find_task_candidates(external_tasks, candidate, exact_match=False)
                        if matches:
                            matched_task_name = matches[0]["name"]
                            log.info(f"Using external task {matched_task_name} (matched from candidates: {candidates})")
                            break
                    if matched_task_name:
                        target_input["task"] = matched_task_name
                    else:
                        # No candidate matched - use first candidate literal, validation will catch errors
                        log.warning(
                            f"None of {candidates} found in external workflow {with_external_wf}, "
                            f"using first candidate: {candidates[0]}"
                        )
                        target_input["task"] = candidates[0]
                else:
                    target_input["task"] = external_task_override
            # Otherwise, check if the referenced task exists in the external workflow
            elif external_tasks:  # Only check if we have external tasks
                referenced_task = target_input["task"]
                external_task_names = [task["name"] for task in external_tasks]
                if referenced_task not in external_task_names:
                    # Use substring matching to find similar task (handles numeric prefixes like "15-clipgt-ncore")
                    # Strip numeric prefix if present for better matching
                    task_base = _get_task_name_without_prefix(referenced_task)
                    matches = _find_task_candidates(external_tasks, task_base, exact_match=False)
                    if matches:
                        matched_task_name = matches[0]["name"]
                        log.info(
                            f"Adjusting task reference from {referenced_task} to {matched_task_name} "
                            f"in external workflow {with_external_wf}"
                        )
                        target_input["task"] = matched_task_name
                    else:
                        log.warning(
                            f"Task {referenced_task} not found in external workflow {with_external_wf}, "
                            "keeping original reference"
                        )
    return spec


def publish_dataconditions_at_launch(
    config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig,
    maglev_toolbox: MaglevToolbox,
    dry_run: bool = False,
) -> None:
    """Publish NUREC_REQUEST and NUREC_GENERATION=START dataconditions for all clip IDs at launch time.

    Uses `maglev dataconditions event create` CLI which supports arbitrary DC names
    (unlike clip_dc_publisher which only allows a hardcoded set).
    """
    dc_config = getattr(config, "publish_dataconditions_at_launch", None)
    if dc_config is None or not getattr(dc_config, "enabled", False):
        return

    if not config.car2sim.enabled or config.car2sim.full_pipeline.clipgt_ids in [None, "", []]:
        log.warning("publish_dataconditions_at_launch enabled but no clipgt_ids found, skipping")
        return

    clipgt_ids = list(config.car2sim.full_pipeline.clipgt_ids)
    nre_version = config.docker.image.split(":")[-1].split("@")[0]

    nurec_request_description = json.dumps(
        {
            "priority": str(getattr(dc_config, "priority", "P1")),
            "request_version": nre_version,
            "flavor": str(getattr(dc_config, "flavor", "default")),
            "requestor": str(
                getattr(dc_config, "requestor", f"NUREC request created by {config.workflow_name_prefix} CICD")
            ),
            "use_case": str(getattr(dc_config, "use_case", "other")),
        }
    )

    nurec_generation_description = json.dumps(
        {
            "requestor": f"NUREC generation started by {config.workflow_name_prefix} CICD",
            "nre_version": nre_version,
        }
    )

    log.info(f"Publishing dataconditions for {len(clipgt_ids)} clips at launch time")

    for clip_id in clipgt_ids:
        for dc_name, dc_state, dc_description in [
            ("NUREC_REQUEST", "SUCCESS", nurec_request_description),
            ("NUREC_GENERATION", "START", nurec_generation_description),
        ]:
            cmd = [
                maglev_toolbox.cli,
                "dataconditions",
                "event",
                "create",
                "--type",
                "CLIP",
                "--id",
                clip_id,
                "--name",
                dc_name,
                "--state",
                dc_state,
                "--description",
                dc_description,
            ]
            log.info(f"Publishing {dc_name}={dc_state} for {clip_id}")
            if not dry_run:
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    log.warning(f"Failed to publish {dc_name}={dc_state} for {clip_id}: {e.stderr}")


# Default image for the publish-dc task. This image contains pycsft/clip/clip_dc_publisher.
PUBLISH_DC_IMAGE = (
    "nvcr.io/nv-maglev/dlav/bazel.pycsft.workflows.nre.nre_image:"
    "3b33bffa5a16@sha256:3b33bffa5a16f4e2eee7c24c8dc97bc21085c983f8099389828a1aa4d204a0bc"
)

PUBLISH_DC_SCRIPT = """\
ALL_CLIPS_FILE={{input}}/clip_ids/all_clip_ids.txt
SUCCESS_CLIPS_FILE={{output}}/success_clip_ids.txt
FAIL_CLIPS_FILE={{output}}/fail_clip_ids.txt
while read CLIP_ID; do
  if [ -f {{input}}/clipgt-$CLIP_ID/usd_out/last.usdz ]; then
    echo $CLIP_ID >> $SUCCESS_CLIPS_FILE
  else
    echo $CLIP_ID >> $FAIL_CLIPS_FILE
  fi
done < $ALL_CLIPS_FILE

if [ -f $SUCCESS_CLIPS_FILE ]; then
  pycsft/clip/clip_dc_publisher \\
    --clip-ids $SUCCESS_CLIPS_FILE \\
    --task-name {{task_name}} \\
    --data-condition-name NUREC_GENERATION \\
    --data-condition-state SUCCESS
fi

if [ -f $FAIL_CLIPS_FILE ]; then
  pycsft/clip/clip_dc_publisher \\
    --clip-ids $FAIL_CLIPS_FILE \\
    --task-name {{task_name}} \\
    --data-condition-name NUREC_GENERATION \\
    --data-condition-state FAIL
fi"""


def add_publish_dc_task(spec: dict, config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig) -> dict:
    """Add a publish-dc task to the workflow spec that publishes NUREC_GENERATION=SUCCESS/FAIL.

    Works with both lidar and lidarfree workflows. Gates on reconstruction (lidar) or
    nre-out (lidarfree). Clip IDs come from session-list (required).
    """
    dc_config = getattr(config, "publish_dataconditions_at_launch", None)
    if dc_config is None or not getattr(dc_config, "enabled", False):
        return spec

    task_names = [task["name"] for task in spec["tasks"]]

    def find_task(substring: str) -> Optional[str]:
        return next((name for name in task_names if substring in name), None)

    # session-list is required for clip IDs
    session_list_task = find_task("session-list")
    if session_list_task is None:
        raise ValueError("add_publish_dc_task: session-list task not found in spec but is required")

    # Gate on reconstruction (lidar) or nre-out (lidarfree)
    reconstruction_task = find_task("reconstruction")
    nre_out_task = find_task("nre-out")
    gate_task = reconstruction_task or nre_out_task

    if gate_task is None:
        raise ValueError("add_publish_dc_task: no suitable gate task found (looked for reconstruction, nre-out)")

    # Build inputs
    publish_dc_inputs: list[dict] = [
        {
            "task": session_list_task,
            "selector": "clip_ids/all_clip_ids.txt",
        },
        {
            "task": gate_task,
            "selector": r"clipgt-(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})/.+",
        },
    ]

    # Determine task name with proper numbering
    max_prefix = -1
    for name in task_names:
        match = re.match(r"^(\d+)-", name)
        if match:
            max_prefix = max(max_prefix, int(match.group(1)))
    publish_dc_task_name = f"{max_prefix + 1}-publish-dc" if max_prefix >= 0 else "publish-dc"

    image = str(getattr(dc_config, "image", PUBLISH_DC_IMAGE))
    if image in [None, "null", ""]:
        image = PUBLISH_DC_IMAGE

    publish_dc_task = {
        "name": publish_dc_task_name,
        "inputs": publish_dc_inputs,
        "image": image,
        "command": "/bin/bash",
        "args": ["-euxo", "pipefail", "-c", PUBLISH_DC_SCRIPT],
        "group": "all",
        "fail": "never",
        "cache": "disable",
        "jobRetries": 2,
        "jobTimeout": "8h",
        "workerPool": "publish-dc-workerpool",
        "outputs": [{"retentionPolicy": {"default": "5d"}}],
        "failAction": "stop-branch",
    }

    spec["tasks"].append(publish_dc_task)

    # Add the worker pool if it doesn't exist
    existing_wp_names = {wp["name"] for wp in spec.get("workerPools", [])}
    if "publish-dc-workerpool" not in existing_wp_names:
        spec["workerPools"].append(
            {
                "name": "publish-dc-workerpool",
                "workers": 1,
                "cpu": "4",
                "gpu": "0",
                "mem": "32Gi",
                "disk": "36Gi",
            }
        )

    log.info(f"Added publish-dc task '{publish_dc_task_name}' depending on [{session_list_task}, {gate_task}]")

    return spec


@click.command(
    "run_ndas_workflow",
    help="""Run NDAS workflow for reconstruction and rendering.

    Optional environment variables:

    - RECONSTRUCTION_ARGS: Alternative to hydra arg car2sim.reconstruction_args, for NuRec args that are hard to pass in due to bash escaping.

    - CI_PIPELINE_ID: Used for workflow naming in CI environments

    - USER: Used for backup naming when CI_PIPELINE_ID is not available
    """,
)
@click.option("--dry-run", is_flag=True, help="Flag to perform a dry run", default=False)
@click.option("--docker-image", type=str, default=None, help="NuRec docker image for reconstruction and rendering.")
@click.option("--tools-docker-image", type=str, default=None, help="Tools docker image containing nre-aux-data.")
@click.option("--nrm-docker-image", type=str, default=None, help="NRM docker image for NRM prediction tasks.")
@click.option(
    "--sauron-docker-image", type=str, default=None, help="Sauron docker image for sauron detection/tracking tasks."
)
@click.option(
    "--ndas-workflow-config",
    type=str,
    help="Config file path relative to internal/workflows/cluster_toolbox/cluster_configs/ndas_workflow",
    default="ndas_workflow_base.yaml",
)
@click.option("--name", type=str, help="Name of the workflow", default=None)
@click.argument("hydra-args", nargs=-1)
def run_ndas_workflow(
    dry_run: bool = False,
    docker_image: Optional[str] = None,
    tools_docker_image: Optional[str] = None,
    nrm_docker_image: Optional[str] = None,
    sauron_docker_image: Optional[str] = None,
    ndas_workflow_config: str = "ndas_workflow_base.yaml",
    hydra_args: list[str] = [],
    name: Optional[str] = None,
) -> None:
    """Entrypoint to run NDAS CICD"""

    # We use MaglevToolbox because it contains a lot of useful maglev cli functionality,
    # but we do not use its other features like "wandb sweep", which leads us to call
    # internal function "_submit_workflow". It would be cleaner in the long run to split this
    # functionality out into a new utility class.
    hydra_args = list(hydra_args)
    if docker_image is not None:
        hydra_args.append(f"docker.image={docker_image}")
    if tools_docker_image is not None:
        hydra_args.append(f"docker.tools_image={tools_docker_image}")
    if nrm_docker_image is not None:
        hydra_args.append(f"docker.nrm_image={nrm_docker_image}")
    if sauron_docker_image is not None:
        hydra_args.append(f"docker.sauron_image={sauron_docker_image}")

    maglev_toolbox = MaglevToolbox(config_name=f"ndas_workflows/{ndas_workflow_config}", hydra_args=hydra_args)

    # Check if NRE docker images are required (disabled by generic_wf_base.yaml)
    if getattr(maglev_toolbox.config, "require_nre_docker_images", True):
        if docker_image is None:
            raise click.UsageError("--docker-image is required for this workflow config.")
        if tools_docker_image is None:
            raise click.UsageError("--tools-docker-image is required for this workflow config.")

    spec = build_template_from_config(maglev_toolbox.config, maglev_toolbox)
    spec = fill_template(spec, maglev_toolbox.config)
    spec = add_publish_dc_task(spec, maglev_toolbox.config)
    spec = configure_worker_pools_and_tasks(spec, maglev_toolbox.config)

    # Publish NUREC_REQUEST and NUREC_GENERATION=START at launch time (before workflow submission)
    publish_dataconditions_at_launch(maglev_toolbox.config, maglev_toolbox, dry_run=dry_run)

    # Need to do this for mypy since config is a DictConfig | ListConfig
    workflow_name_prefix = "nurec-cicd-reconstruct"
    if isinstance(maglev_toolbox.config, omegaconf.dictconfig.DictConfig):
        workflow_name_prefix = maglev_toolbox.config.get("workflow_name_prefix", "nurec-cicd-reconstruct")

    backup_name = f"local-run-{os.environ.get('USER')}"
    if name is None:
        name = f"{workflow_name_prefix}-{os.environ.get('CI_PIPELINE_ID', backup_name)}"

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", prefix="maglev_workflow_", suffix=".yaml", delete=False
    ) as tmp_file:
        # Save modified Maglev workflow configuration into a temporary file for the maglev CLI command.
        log.info(f"Saving {tmp_file.name}")
        yaml.safe_dump(spec, tmp_file, sort_keys=False)
        maglev_toolbox._submit_workflow(
            name=name,
            spec_path=tmp_file.name,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    run_ndas_workflow()
