#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import argparse
import json

from copy import deepcopy
from pathlib import Path
from typing import Any


ASSET_REFERENCE_PREFIX = "asset://"
CATALOG_REPO_PATH = Path("internal/sqa/edit-assets/catalog.json")
SCENARIOS_REPO_DIR = Path("internal/sqa/edit-assets/scenarios")
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve a checked-in edit-assets scenario into runtime JSON.")
    parser.add_argument("--scenario-id", required=True, help="Scenario identifier, matching scenarios/<id>.json")
    parser.add_argument("--dataset-name", required=True, help="Dataset selected by the SQA test case")
    parser.add_argument(
        "--output-edit-file", required=True, help="Output path for the resolved runtime edit_assets.json"
    )
    return parser.parse_args()


def resolve_runfiles_or_repo_path(repo_relative_path: Path) -> Path:
    direct_candidates = [
        Path.cwd() / repo_relative_path,
        SCRIPT_REPO_ROOT / repo_relative_path,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    try:
        from python.runfiles import runfiles
    except ImportError:
        runfiles = None

    if runfiles is not None:
        rf = runfiles.Create()
        if rf is not None:
            for candidate in (repo_relative_path.as_posix(), f"_main/{repo_relative_path.as_posix()}"):
                resolved = rf.Rlocation(candidate)
                if resolved and Path(resolved).exists():
                    return Path(resolved).resolve()

    raise FileNotFoundError(f"Could not resolve repo path: {repo_relative_path}")


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return json.load(file)


def resolve_catalog_asset_path(asset_key: str, catalog: dict[str, Any]) -> str:
    if asset_key not in catalog:
        raise KeyError(f"Unknown asset key '{asset_key}' in edit-assets catalog")

    catalog_entry = catalog[asset_key]
    if isinstance(catalog_entry, str):
        catalog_path = Path(catalog_entry)
    elif isinstance(catalog_entry, dict) and "path" in catalog_entry:
        catalog_path = Path(catalog_entry["path"])
    else:
        raise ValueError(f"Catalog entry for asset '{asset_key}' must be a path string or object with 'path'")

    return str(resolve_runfiles_or_repo_path(catalog_path))


def resolve_asset_reference(value: str, catalog: dict[str, Any]) -> str:
    if not value.startswith(ASSET_REFERENCE_PREFIX):
        return value

    asset_key = value.removeprefix(ASSET_REFERENCE_PREFIX)
    return resolve_catalog_asset_path(asset_key, catalog)


def resolve_scenario_assets(scenario: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    resolved_scenario = deepcopy(scenario)

    for replace_spec in resolved_scenario.get("replace", []):
        replacement_id = replace_spec.get("replacement_id")
        if isinstance(replacement_id, str):
            replace_spec["replacement_id"] = resolve_asset_reference(replacement_id, catalog)

    insert_section = resolved_scenario.get("insert", {})
    asset_ids = insert_section.get("asset_ids", [])
    insert_section["asset_ids"] = [
        resolve_asset_reference(asset_id, catalog) if isinstance(asset_id, str) else asset_id for asset_id in asset_ids
    ]

    metadata = resolved_scenario.setdefault("metadata", {})
    metadata["resolved_asset_catalog"] = str(resolve_runfiles_or_repo_path(CATALOG_REPO_PATH))

    return resolved_scenario


def validate_scenario_metadata(scenario: dict[str, Any], dataset_name: str) -> None:
    metadata = scenario.get("metadata", {})
    scenario_dataset = metadata.get("dataset")
    if scenario_dataset != dataset_name:
        raise ValueError(f"Dataset mismatch: expected '{dataset_name}', found '{scenario_dataset}'")


def main() -> None:
    args = parse_args()

    scenario_path = resolve_runfiles_or_repo_path(SCENARIOS_REPO_DIR / f"{args.scenario_id}.json")
    catalog_path = resolve_runfiles_or_repo_path(CATALOG_REPO_PATH)

    scenario = load_json_file(scenario_path)
    catalog = load_json_file(catalog_path)

    validate_scenario_metadata(scenario, args.dataset_name)

    resolved_scenario = resolve_scenario_assets(scenario, catalog)
    resolved_scenario.setdefault("metadata", {})["resolved_from_scenario"] = str(scenario_path)

    output_path = Path(args.output_edit_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(resolved_scenario, file, indent=2)
        file.write("\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
