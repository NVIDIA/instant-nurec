# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import logging
import os

from pathlib import Path
from typing import Any, Optional

import click
import yaml


logger = logging.getLogger(__name__)

from nre.config.asset_harvest import AssetHarvestingConfig
from nre.config.parse import parse_untyped_config
from nre.utils.io.asset_harvester_tools.plan_tracks import generate_harvest_plan


class NoAliasDumper(yaml.SafeDumper):
    # Prevent anchors/aliases in YAML output
    def ignore_aliases(self, data):
        return True


class FlowSeq(list):
    """Marker type: dump this sequence in flow (inline) style."""

    pass


# Register representer inline to avoid name-lookup issues under obfuscation
NoAliasDumper.add_representer(
    FlowSeq,
    lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True),
)


def load_metadata(metadata_path: str) -> dict[str, Any]:
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)
    return meta if isinstance(meta, dict) else {}


def extract_track_ids(meta: dict[str, Any]) -> list[str]:
    track_ids: set[str] = set()

    def add_tid(val: Any) -> None:
        if val is None:
            return
        track_ids.add(str(val))

    # Handle tracks/assets keys
    for key_name in ("tracks", "assets"):
        value = meta.get(key_name)
        if isinstance(value, dict):
            # Accept either a single-object form: {"track_id": ... , ...}
            # or a mapping form: {"<track_id>": {...}, ...}
            if "track_id" in value:
                add_tid(value.get("track_id"))
            else:
                for k, v in value.items():
                    if isinstance(v, dict):
                        # Prefer nested track_id when present
                        nested_tid = v.get("track_id")
                        if nested_tid is not None:
                            add_tid(nested_tid)
    return sorted(track_ids)


def find_plys_from_metadata(metadata_path: str, harvested_assets_path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    meta = load_metadata(metadata_path)
    track_ids = extract_track_ids(meta)

    root = Path(harvested_assets_path)

    # Accept several layouts per track id
    def candidates_for(tid: str) -> list[Path]:
        # prefer per-track dir first, then scene-aware, then flat
        cands: list[Path] = []
        cands.append(root / tid / f"{tid}.ply")
        cands.append(root / tid / "gs.ply")
        for p in root.iterdir():
            if p.is_dir():
                cands.append(p / tid / f"{tid}.ply")
                cands.append(p / tid / "gs.ply")
        cands.append(root / f"{tid}.ply")
        return cands

    # Case 1: metadata contains explicit track_ids
    if track_ids:
        for tid in track_ids:
            for cand in candidates_for(tid):
                if cand.is_file():
                    mapping[tid] = str(cand)
                    break
        return mapping

    # Case 2: fallback scan when metadata lacks track_ids
    # Look for any *.ply under harvested_assets and infer track_id
    for ply in root.rglob("*.ply"):
        # try filename-based tid
        name = ply.stem
        parent = ply.parent.name
        maybe_tid: str | None = None
        if name.isdigit():
            maybe_tid = name
        elif parent.isdigit():
            maybe_tid = parent
        if maybe_tid is not None:
            mapping[maybe_tid] = str(ply)

    return mapping


def build_overlay_dual(
    harvested_track_to_ply: dict[str, str],
    fallback_track_to_ply: dict[str, str],
) -> dict[str, Any]:
    exclude_ids = sorted(set(harvested_track_to_ply.keys()) | set(fallback_track_to_ply.keys()))
    return {
        "model": {
            "layers": {
                "loaded_ply_assets": {"initialization": {"track_ids": harvested_track_to_ply}},
                "loaded_ply_assets_no_optim": {"initialization": {"track_ids": fallback_track_to_ply}},
                "dynamic_rigids": {"tracks": {"is_dynamic": True, "exclude_ids": exclude_ids}},
            }
        }
    }


def set_nested(d: dict[str, Any], path: list[str], value: Any) -> None:
    cur: dict[str, Any] = d
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def apply_template_dual(
    template: dict[str, Any],
    harvested_track_to_ply: dict[str, str],
    fallback_track_to_ply: dict[str, str],
) -> dict[str, Any]:
    # Plain dict edit (comments from template not preserved by dumper);
    # header comments will be manually prepended when writing.
    harvested_ids_sorted = sorted(harvested_track_to_ply.keys())
    fallback_ids_sorted = sorted(fallback_track_to_ply.keys())
    exclude_list = list(sorted(set(harvested_ids_sorted) | set(fallback_ids_sorted)))
    out_plain = dict(template)

    # Ensure base structure exists to allow edits
    if not isinstance(out_plain.get("model"), dict):
        out_plain["model"] = {}
    model = out_plain["model"]
    if not isinstance(model.get("layers"), dict):
        model["layers"] = {}
    layers = model["layers"]

    # Primary layer: only include when harvested mapping is non-empty;
    # otherwise, remove it if present in the template
    if harvested_track_to_ply:
        set_nested(
            out_plain, ["model", "layers", "loaded_ply_assets", "initialization", "track_ids"], harvested_track_to_ply
        )
        set_nested(out_plain, ["model", "layers", "loaded_ply_assets", "tracks", "ids"], list(harvested_ids_sorted))
    else:
        if "loaded_ply_assets" in layers:
            try:
                del layers["loaded_ply_assets"]
            except Exception:
                pass

    # Secondary layer: only include when fallback mapping is non-empty;
    # otherwise, remove it if present in the template
    if fallback_track_to_ply:
        set_nested(
            out_plain,
            ["model", "layers", "loaded_ply_assets_no_optim", "initialization", "track_ids"],
            fallback_track_to_ply,
        )
        set_nested(
            out_plain, ["model", "layers", "loaded_ply_assets_no_optim", "tracks", "ids"], list(fallback_ids_sorted)
        )
    else:
        if "loaded_ply_assets_no_optim" in layers:
            try:
                del layers["loaded_ply_assets_no_optim"]
            except Exception:
                pass

    # Always set exclude list for dynamic_rigids if present or needed
    set_nested(out_plain, ["model", "layers", "dynamic_rigids", "tracks", "exclude_ids"], exclude_list)

    # Remove skipped layers from model.strategy.exclude_layer_ids
    try:
        layers_missing: list[str] = []
        if not harvested_track_to_ply:
            layers_missing.append("loaded_ply_assets")
        if not fallback_track_to_ply:
            layers_missing.append("loaded_ply_assets_no_optim")
        if layers_missing:
            model_node = out_plain.get("model")
            if isinstance(model_node, dict):
                strategy = model_node.get("strategy")
                if isinstance(strategy, dict):
                    ex_layers = strategy.get("exclude_layer_ids")
                    if isinstance(ex_layers, list):
                        strategy["exclude_layer_ids"] = [x for x in ex_layers if x not in layers_missing]
    except Exception:
        pass

    # Preserve inline list style only if using our NoAliasDumper via FlowSeq
    try:
        defaults = out_plain.get("defaults")
        if isinstance(defaults, list):
            for i, entry in enumerate(defaults):
                if isinstance(entry, dict):
                    for k, v in list(entry.items()):
                        if isinstance(v, list):
                            defaults[i][k] = FlowSeq(v)
    except Exception:
        pass

    return out_plain


def _extract_header_comment_block(path: str) -> str:
    """Extract leading comment/blank lines from a YAML template to preserve header text.

    Returns the header text including trailing newline if present, else empty string.
    """
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        header_lines: list[str] = []
        seen_content = False
        for line in lines:
            if line.lstrip().startswith("#") or line.strip() == "":
                if not seen_content:
                    header_lines.append(line)
                else:
                    break
            else:
                seen_content = True
                break
        return "".join(header_lines)
    except Exception:
        return ""


def _load_plan_track_classes(plan_json_path: str | None) -> dict[str, str]:
    """Return mapping track_id -> label_class from plan JSON if available."""
    mapping: dict[str, str] = {}
    if not plan_json_path:
        logger.info("No --plan-json provided; skipping plan-based class mapping.")
        return mapping
    p = Path(plan_json_path)
    if not p.is_file():
        logger.warning(f"Plan JSON not found at: {plan_json_path}")
        return mapping
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to read/parse plan JSON at {plan_json_path}: {e}")
        return mapping

    # Accept one or multiple scene entries; aggregate all tracks
    scenes = data if isinstance(data, list) else []
    if not scenes:
        logger.warning(f"Plan JSON structure unexpected (not a list): {type(data)}")
        return mapping
    total_tracks = 0
    for scene in scenes:
        tracks = scene.get("tracks", []) if isinstance(scene, dict) else []
        for t in tracks:
            if not isinstance(t, dict):
                continue
            tid_val = t.get("track_id")
            lc = t.get("label_class")
            if tid_val is None or not isinstance(lc, str):
                continue
            tid = str(tid_val)
            mapping[tid] = lc
            total_tracks += 1
    if not mapping:
        logger.warning(f"No track classes found in plan JSON at {plan_json_path} (total_tracks_scanned={total_tracks})")
    else:
        logger.info(f"Loaded {len(mapping)} track->class entries from plan JSON.")
    return mapping


def _apply_asset_bank_overrides(
    *,
    class_to_exemplar: dict[str, str],
    plan_track_to_class: dict[str, str],
    asset_bank_dir: str,
) -> dict[str, str]:
    """Override class_exemplars with absolute paths from an asset bank.

    - Matches by filename stem (class name), case-insensitive
    - Overrides existing exemplar entries when present
    - Adds entries only for classes that appear in the plan if not present already
    - Always writes absolute paths
    """
    base_mapping = dict(class_to_exemplar)
    asset_bank_path = Path(asset_bank_dir)
    plys = list(asset_bank_path.rglob("*.ply"))
    if not plys:
        logger.warning(f"No .ply files found under asset bank: {asset_bank_dir}")
        return base_mapping

    # Upgrade exemplar entries by resolving their relative path filenames under asset_bank_dir
    overrides_applied = 0
    for key, value in list(base_mapping.items()):
        p = Path(value)
        if p.is_absolute():
            continue
        filename = p.name
        candidates = list(asset_bank_path.rglob(filename))
        if not candidates:
            continue
        if len(candidates) == 1:
            base_mapping[key] = str(candidates[0].resolve())
            overrides_applied += 1
            continue
        # Prefer parent folder name equals stem to avoid collisions (e.g., '4/4.ply')
        stem = p.stem
        preferred = [c for c in candidates if c.parent.name == stem]
        chosen = preferred[0] if preferred else candidates[0]
        base_mapping[key] = str(chosen.resolve())
        overrides_applied += 1

    return base_mapping


def _load_class_exemplars(class_exemplars_yaml: str | None) -> dict[str, str]:
    """Return mapping label_class -> exemplar PLY path as provided.

    Notes:
    - Absolute paths are kept unchanged.
    - Relative paths are kept unchanged (workspace-relative is recommended
      so Bazel runfiles can locate them at runtime in the trainer).
    """
    if not class_exemplars_yaml:
        return {}
    path = Path(class_exemplars_yaml)
    try:
        with open(path, "r") as f:
            obj = yaml.safe_load(f)
        if not isinstance(obj, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str):
                # Keep absolute as-is; keep relative as provided (e.g., workspace-relative)
                out[k] = v
        return out
    except Exception:
        return {}


@click.command("generate-asset-harvester-training-yaml")
@click.option("--metadata", required=True, help="Path to harvested_assets/metadata.yaml")
@click.option("--harvested-assets", required=True, help="Path to harvested_assets directory")
@click.option(
    "--template-yaml", default=None, help="Optional YAML template to shape the overlay. Placeholders will be filled."
)
@click.option("--output-yaml", default=None, help="Where to write the training overlay YAML")
@click.option(
    "--plan-json",
    default=None,
    help="Optional harvest plan JSON to obtain label_class per track for fallback assignment",
)
@click.option(
    "--class-exemplars", default=None, help="Optional YAML mapping label_class -> exemplar PLY path for fallback assets"
)
@click.option(
    "--asset-bank",
    default=None,
    help="Optional directory of exemplar PLYs to override class_exemplars entries. Filenames should match class names.",
)
@click.option(
    "--radius-m",
    type=float,
    default=None,
    help="If provided and --plan-json is absent, generate a plan by selecting tracks within this radius (meters)",
)
@click.option("--dataset-path", default=None, help="Absolute path to dataset .zarr.itar when generating plan")
@click.option("--camera-ids", default=None, help="Comma-separated camera ids (required for plan generation)")
@click.option(
    "--train-camera-ids", default=None, help="Comma-separated train camera ids (required for plan generation)"
)
@click.option("--lidar-ids", default=None, help="Comma-separated lidar ids (required for plan generation)")
@click.option(
    "--harvest-config",
    default="configs/experimental/asset_harvesting/harvest.yaml",
    help="Path to harvest config YAML to get harvestable class list",
)
@click.option(
    "--dataset-config",
    default="configs/dataset/ncore.yaml",
    help="Path to dataset config YAML for NCore dataset configuration",
)
@click.option(
    "--min-visible-frames", type=int, default=5, help="Min visible frames per track (optional, for plan generation)"
)
@click.option(
    "--label-source",
    default="scene:obstacles:autolabels:v2",
    help="Label source to annotate in plan (optional, for plan generation)",
)
def generate_asset_harvester_training_yaml_cli(
    metadata: str,
    harvested_assets: str,
    template_yaml: Optional[str] = None,
    output_yaml: Optional[str] = None,
    plan_json: Optional[str] = None,
    class_exemplars: Optional[str] = None,
    asset_bank: Optional[str] = None,
    radius_m: Optional[float] = None,
    dataset_path: Optional[str] = None,
    camera_ids: Optional[str] = None,
    train_camera_ids: Optional[str] = None,
    lidar_ids: Optional[str] = None,
    harvest_config: str = "configs/experimental/asset_harvesting/harvest.yaml",
    dataset_config: str = "configs/dataset/ncore.yaml",
    min_visible_frames: int = 5,
    label_source: str = "scene:obstacles:autolabels:v2",
) -> None:
    """Generate a minimal training overlay YAML from harvester metadata and a base training config."""

    # Parse harvest config to get harvestable classes for consistency
    try:
        # Ensure Hydra is not already initialized before parsing configs
        try:
            from hydra.core.global_hydra import GlobalHydra  # type: ignore

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
        except Exception:
            pass
        harvest_hydra_overrides = [
            "+logger.save_dir=.",
            "+save_dir=.",
            "+ckpt_dir=.",
            "+config_dir=.",
        ]
        harvest_untyped = parse_untyped_config(config_name=harvest_config, hydra_args=harvest_hydra_overrides)
        harvest_config_obj = AssetHarvestingConfig.model_validate(harvest_untyped)
        harvestable_classes = None
        logger.info("No class filtering applied (crop_labels removed in GA migration)")
    except Exception as e:
        logger.warning(f"Could not load harvest config {harvest_config}: {e}. Proceeding without class filtering.")
        harvestable_classes = None

    # Optionally generate a plan when requested and not provided
    plan_json_path: Optional[str] = plan_json
    if plan_json_path:
        logger.info(f"Using provided plan JSON: {plan_json_path}")
    if (not plan_json_path or not os.path.exists(plan_json_path)) and radius_m is not None:
        logger.info(f"{plan_json} is not provided. Generating plan with radius {radius_m}")
        if not dataset_path:
            raise SystemExit("--dataset-path is required when using --radius-m without --plan-json")
        if not camera_ids or not train_camera_ids or not lidar_ids:
            raise SystemExit(
                "--camera-ids, --train-camera-ids, and --lidar-ids are required when using --radius-m without --plan-json"
            )
        # Parse required lists
        camera_ids_list = [c.strip() for c in camera_ids.split(",") if c.strip()]
        train_camera_ids_list = [c.strip() for c in train_camera_ids.split(",") if c.strip()]
        lidar_ids_list = [x.strip() for x in lidar_ids.split(",") if x.strip()]

        # Ensure Hydra is not already initialized before generating the plan
        try:
            from hydra.core.global_hydra import GlobalHydra  # type: ignore

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
        except Exception:
            pass

        plan, out_path = generate_harvest_plan(
            dataset_path=dataset_path,
            camera_ids=camera_ids_list,
            train_camera_ids=train_camera_ids_list,
            lidar_ids=lidar_ids_list,
            radius_m=float(radius_m),
            min_visible_frames=int(min_visible_frames),
            label_source=str(label_source),
            harvest_config_name=harvest_config,
            dataset_config_name=dataset_config,
            output_dir=None,
            write_output=True,
        )
        plan_json_path = out_path
        logger.info(f"Regenerated plan: {plan}")

    harvested_track_to_ply = find_plys_from_metadata(metadata, harvested_assets)
    plan_track_to_class = _load_plan_track_classes(plan_json_path)
    # If still empty here, proceed without fallback mapping and log for visibility.
    if not plan_track_to_class:
        logger.warning("No plan-based class mapping found; proceeding without fallback exemplar assets.")
    # Keep exemplar paths as provided (prefer workspace-relative for Bazel runfiles)
    class_to_exemplar = _load_class_exemplars(class_exemplars)
    logger.info(f"Class to exemplar from class_exemplars: {class_to_exemplar}")

    # Optional: override exemplar paths from an asset bank with absolute paths
    if asset_bank:
        try:
            class_to_exemplar = _apply_asset_bank_overrides(
                class_to_exemplar=class_to_exemplar,
                plan_track_to_class=plan_track_to_class,
                asset_bank_dir=asset_bank,
            )
        except Exception:
            logger.exception("Failed to apply asset_bank overrides; continuing with existing exemplars")

    logger.info(f"Class to exemplar: {class_to_exemplar}")

    # Determine fallback mapping with case-insensitive class matching
    harvested_track_to_ply_in_plan: dict[str, str] = {}
    fallback_track_to_ply: dict[str, str] = {}
    for tid, label_class in plan_track_to_class.items():
        if tid in harvested_track_to_ply:
            harvested_track_to_ply_in_plan[tid] = harvested_track_to_ply[tid]
        else:
            # Case-insensitive lookup for exemplar path
            exemplar_path = None
            for cls, path in class_to_exemplar.items():
                if cls.lower() == label_class.lower():
                    exemplar_path = path
                    break
            if exemplar_path:
                fallback_track_to_ply[tid] = exemplar_path
            elif harvestable_classes and label_class.lower() not in harvestable_classes:
                logger.debug(f"Track {tid} with class '{label_class}' is not in harvestable classes, skipping fallback")

    logger.info(f"Harvested track to PLY in plan: {harvested_track_to_ply_in_plan}")
    logger.info(f"Fallback track to PLY: {fallback_track_to_ply}")

    # If no plan provided, default to only harvested
    if not harvested_track_to_ply_in_plan and not fallback_track_to_ply:
        raise SystemExit("No PLYs found. Provide plan/class-exemplars for fallback or ensure harvested assets exist.")

    if template_yaml:
        # Fallback to new location if a common old relative path is provided
        if not os.path.isfile(template_yaml) and template_yaml.startswith("scripts/asset_harvester/"):
            alt = template_yaml.replace("scripts/asset_harvester/", "nre/utils/io/asset_harvester_tools/")
            if os.path.isfile(alt):
                template_yaml = alt
        with open(template_yaml, "r") as f:
            template_obj = yaml.safe_load(f)
            if not isinstance(template_obj, dict):
                raise SystemExit("--template-yaml must be a YAML mapping at the top level.")
        overlay = apply_template_dual(template_obj, harvested_track_to_ply_in_plan, fallback_track_to_ply)
    else:
        overlay = build_overlay_dual(harvested_track_to_ply_in_plan, fallback_track_to_ply)

    out_path = output_yaml
    if out_path is None:
        base_dir = os.path.dirname(metadata)
        out_path = os.path.join(base_dir, "training_assets_overlay.yaml")

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    # Prepare header for training overlay
    header_text = ""
    if template_yaml:
        header_text = _extract_header_comment_block(template_yaml)

    with open(out_path, "w") as f:
        if header_text:
            if not header_text.endswith("\n"):
                header_text += "\n"
            f.write(header_text)
        yaml.dump(overlay, f, sort_keys=False, Dumper=NoAliasDumper, width=4096)

    print(out_path)


if __name__ == "__main__":
    generate_asset_harvester_training_yaml_cli()
