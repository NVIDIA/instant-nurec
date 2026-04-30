# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gc
import logging
import os
import random
import sys

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import imageio
import numpy as np
import torch
import torchvision.transforms as T
import yaml

from asset_harvester.multiview_diffusion.data.inference_utils import build_eval_cams  # pycena: skip
from asset_harvester.multiview_diffusion.data.nre_preproc import MVData as DiffusionMVData  # pycena: skip
from asset_harvester.multiview_diffusion.data.nre_preproc import preproc  # pycena: skip
from asset_harvester.multiview_diffusion.pipelines import SparseViewDiTPipeline  # pycena: skip
from asset_harvester.multiview_diffusion.utils.model_builder import get_models  # pycena: skip
from asset_harvester.ncore_parser.__main__ import resolve_component_store_paths  # pycena: skip
from asset_harvester.ncore_parser.parser import NCoreParser  # pycena: skip
from asset_harvester.ncore_parser.schemas import NCoreParserConfig  # pycena: skip
from asset_harvester.tokengs.lifting_inference import TokengsLiftingRunner  # pycena: skip
from asset_harvester.tokengs.utils.metrics import MetricsCalculator  # pycena: skip
from asset_harvester.utils.io import save_input_views, save_mvd_outputs  # pycena: skip
from asset_harvester.utils.mvd_farthest_pose import farthest_point_sampling  # pycena: skip
from diffusers.schedulers import DPMSolverMultistepScheduler
from omegaconf import OmegaConf
from PIL import Image

from apps.asset_harvester.asset_harvest_metadata import AssetMetadataManager
from apps.asset_harvester.flip_ply import build_rotation_matrix, rotate_ply
from apps.asset_harvester.utils import strip_track_id_suffix
from nre.config.asset_harvest import Asset, AssetHarvestingConfig, AssetHarvestingMetadata, AssetMetrics
from nre.config.parse import parse_untyped_config
from nre.config.scopedtimer import ScopedTimerConfig, VerbosityLevel
from nre.utils.model_registry import ModelRegistryError, create_model_registry, log_and_raise
from nre.utils.profiling import ScopedTimer


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_asset_harvesting_config(
    config_name: str, hydra_args: list[str], config_dir: str = "."
) -> Union[AssetHarvestingConfig]:
    """
    Parses and validates configuration for asset harvesting pipeline.

    Parameters:
    - config_name: Name of the configuration file.
    - hydra_args: List of Hydra override arguments.
    - config_dir: Directory containing config file.

    Returns:
    - Validated AssetHarvestingConfig object.
    """
    # Clear GlobalHydra if already initialized to prevent conflicts (this is a bugfix)
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    logger.info(f"Parsing config: {config_name}")

    # Parse and validate as AssetHarvestingConfig
    untyped_config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args, config_dir=config_dir)
    typed_config = AssetHarvestingConfig.model_validate(untyped_config, context={"config_name": config_name})

    return typed_config


def to_diffusion_mvdata(parser_mvdata) -> DiffusionMVData:
    """Bridge parser-side MVData to diffusion-side MVData."""
    if not parser_mvdata.masks_instance:
        raise ValueError(f"Track {parser_mvdata.obj_id}: masks_instance is required for diffusion preproc")

    return DiffusionMVData(
        clip_id=parser_mvdata.clip_id,
        obj_id=parser_mvdata.obj_id,
        frames=[np.asarray(frame) for frame in parser_mvdata.frames],
        cam_poses=np.asarray(parser_mvdata.cam_poses, dtype=np.float64),
        dists=np.asarray(parser_mvdata.dists, dtype=np.float64),
        fov=np.asarray(parser_mvdata.fov, dtype=np.float64),
        npct=parser_mvdata.npct,
        lwh=np.asarray(parser_mvdata.lwh, dtype=np.float64) if parser_mvdata.lwh is not None else None,
        masks=np.asarray(parser_mvdata.masks_instance),
        auto_label=None,
        sensor_id=list(parser_mvdata.sensor_id),
        caption=list(parser_mvdata.caption),
    )


@click.command("asset_harvest")
@click.option(
    "--component-store",
    type=str,
    required=False,
    help="Path to V4 component store(s). Accepts comma-separated paths, .zarr.itar globs, or a clip .json manifest.",
)
@click.option(
    "--metadata-file",
    type=click.Path(exists=True),
    required=False,
    help="Path to existing metadata YAML file to load instead of extracting from shard",
)
@click.option(
    "--track-ids",
    type=str,
    required=False,
    help="Harvest specific tracks if provided; otherwise try to do it for all tracks. Use ',' to split multiple track ids",
)
@click.option("--output-dir", type=str, required=True, help="Path to output directory")
@click.option(
    "--config-name",
    type=str,
    required=False,
    default="configs/experimental/asset_harvesting/harvest.yaml",
    help="Optional: Hydra config to load for advanced settings (not needed if using --metadata-file)",
)
@click.option(
    "--cache-dir",
    type=str,
    help="Directory for downloaded model files",
    default=os.path.expanduser("~/.cache/nre/"),
)
@click.option(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Random seed for reproducibility (Python, NumPy, and PyTorch if available)",
)
@click.option(
    "--enable-timing",
    is_flag=True,
    default=False,
    help="Enable timing measurements for pipeline stages",
)
@click.option(
    "--timing-logfile",
    type=str,
    required=False,
    default=None,
    help="Path to save timing logs (if not provided, logs to console)",
)
@click.option(
    "--timing-synchronize",
    is_flag=True,
    default=False,
    help="Synchronize GPU before timing measurements for accurate wall-clock time (adds minor overhead)",
)
@click.argument("hydra-args", nargs=-1)
def asset_harvest(
    component_store: Optional[str],
    metadata_file: Optional[str],
    track_ids: Optional[str],
    output_dir: str,
    config_name: Optional[str] = None,
    cache_dir: str = os.path.expanduser("~/.cache/nre/"),
    seed: Optional[int] = 42,
    enable_timing: bool = False,
    timing_logfile: Optional[str] = None,
    timing_synchronize: bool = False,
    hydra_args: Optional[list[str]] = None,
) -> None:
    """Asset harvesting pipeline.

    Either provide --component-store to extract from source data,
    or provide --metadata-file to load from existing metadata.
    """

    # Set default values for Optional parameters
    if hydra_args is None:
        hydra_args = []

    # Set RNG seeds for reproducibility if requested
    if seed is not None:
        logger.info(f"Seeding RNGs with seed={seed}")
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        # Suggest deterministic cuBLAS workspace (has effect on some GEMMs)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            random.seed(seed)
        except Exception:
            pass
        try:
            np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
        except Exception:
            logger.debug("PyTorch not available; skipping torch seeding")

    # Validate that either metadata file OR component store is provided
    if metadata_file is None and component_store is None:
        logger.error("Either --metadata-file OR --component-store must be provided")
        raise click.UsageError("Either --metadata-file OR --component-store must be provided")

    if metadata_file is not None and component_store is not None:
        logger.error("Both --metadata-file and --component-store provided. Use only one.")
        raise click.UsageError("Both --metadata-file and --component-store provided. Use only one.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Configure ScopedTimer for pipeline timing
    if enable_timing:
        timer_config = ScopedTimerConfig(
            enabled=True,
            verbosity=VerbosityLevel.BASIC,
            synchronize=timing_synchronize,
        )
        ScopedTimer.set_global_config(timer_config)

        if timing_logfile:
            timing_path = os.path.abspath(timing_logfile)
            ScopedTimer.set_global_logfile(timing_path)
            logger.info(f"Writing timing results to: {timing_path}")
        else:
            ScopedTimer.set_global_print_func(logger.info)

    metadata_manager = AssetMetadataManager(save_dir=Path(output_dir))

    # Process track_ids if provided
    track_id_list: Optional[List[str]] = None
    if track_ids is not None:
        track_id_list = track_ids.split(",")
        logger.info(f"Processing specific track IDs: {track_id_list}")

    # Extract from component store (default path)
    if metadata_file is None:
        logger.info(f"Extracting from component store: {component_store}")

        # Parse config if provided
        if config_name is None:
            logger.error("config_name is required when using --component-store")
            sys.exit(1)

        try:
            config = parse_asset_harvesting_config(
                config_name,
                hydra_args,
            )
        except Exception as e:
            logger.error(f"Failed to parse config: {e}")
            sys.exit(1)
    else:
        # Load from metadata file
        logger.info(f"Loading from metadata file: {metadata_file}")

        # Detect old config schema structurally before parsing
        raw = yaml.safe_load(Path(metadata_file).read_text())
        config_keys = set((raw.get("config") or {}).keys())
        if {"views_extractor", "gaussian_lift"} & config_keys:
            raise click.UsageError(
                "Metadata file uses old config schema (pre-GA). "
                "Re-run asset harvesting with --component-store instead of --metadata-file. "
                "Old metadata files are not compatible with the GA asset-harvester pipeline."
            )

        metadata: AssetHarvestingMetadata = metadata_manager.load_from_file(metadata_file, hydra_args)

        # Get AssetHarvestingConfig from loaded metadata
        config = metadata_manager.get_runtime_config()

        # Check if ViewExtractor assets exist
        if metadata.assets is None:
            logger.error("No assets found in metadata")
            raise ValueError(
                "No assets found in metadata, please run Asset Harvester w/ '--component-store /path/to/component-store' "
            )

    # Download models and set checkpoint paths
    with ScopedTimer("AssetHarvester/download_models"):
        try:
            logger.info("Setting up model checkpoints")
            # Set checkpoint paths using filenames from URLs
            av_multiview_ckpt = os.path.join(cache_dir, os.path.basename(config.urls.mv_diffusion_ckpt))
            tokengs_ckpt = os.path.join(cache_dir, os.path.basename(config.urls.tokengs_ckpt))
            segmentation_ckpt = os.path.join(cache_dir, os.path.basename(config.urls.segmentation_ckpt))

            # Download each model if it doesn't exist
            models_to_download = [
                ("mv_diffusion", config.urls.mv_diffusion_ckpt),
                ("tokengs", config.urls.tokengs_ckpt),
                ("segmentation", config.urls.segmentation_ckpt),
            ]

            for model_name, url in models_to_download:
                model_path = os.path.join(cache_dir, os.path.basename(url))
                if not os.path.exists(model_path):
                    # Check ngc api key only when you want to actually download a model
                    ngc_api_key = os.environ.get("NGC_API_KEY")
                    if not ngc_api_key:
                        raise Exception("NGC_API_KEY environment variable is not set")
                    try:
                        cache_dir_path = Path(cache_dir)
                        cache_dir_path.mkdir(parents=True, exist_ok=True)

                        logger.info("Initializing model registry for URL: %s", url)
                        registry = create_model_registry(
                            model_url=url,
                            model_cache_dir=cache_dir_path,
                            api_key=ngc_api_key,
                        )

                        _ = registry.get_model()

                    except ModelRegistryError as e:
                        log_and_raise(ModelRegistryError, f"Failed to download model: {str(e)}")
                    except Exception as e:
                        log_and_raise(Exception, f"Unexpected error: {str(e)}")
                else:
                    logger.info(f"Model {model_name} already exists at {model_path}")
        except Exception as e:
            raise Exception(f"Failed to download models: {str(e)}")

    # Stage 1: Views extraction via NCoreParser
    with ScopedTimer("NCoreParser/__init__"):
        parser_config = NCoreParserConfig(
            target_resolution=config.ncore_parser.target_resolution,
            num_lidar_ref_frames=config.ncore_parser.num_lidar_ref_frames,
            cam_pose_flip=config.ncore_parser.cam_pose_flip,
            max_threads=config.ncore_parser.max_threads,
            occ_rate_threshold=config.ncore_parser.occ_rate_threshold,
            crop_min_area_ratio=config.ncore_parser.crop_min_area_ratio,
            mask_exceed_threshold=config.ncore_parser.mask_exceed_threshold,
            min_instance_pixels=config.ncore_parser.min_instance_pixels,
            mask_overlap_threshold=config.ncore_parser.mask_overlap_threshold,
            camera_ids=config.ncore_parser.camera_ids,
            segmentation_ckpt=segmentation_ckpt,
        )
        parser = NCoreParser(config=parser_config)

    with ScopedTimer("NCoreParser/extract"):
        try:
            if metadata_file is None:
                data_paths = resolve_component_store_paths(component_store)
                # camera_ids passed here because NCoreParser.extract() forwards it to
                # ncore_extractor.extract() independently of self.config
                mvdata_objects: Dict[str, Any] = parser.extract(
                    src_data_paths=data_paths,
                    target_root_path=output_dir,
                    target_track_ids=track_id_list,
                    camera_ids=parser_config.camera_ids,
                )
            else:
                mvdata_objects = metadata_manager.get_mvdata(Path(metadata_file), track_id_list)
                logger.info(f"Converted {len(mvdata_objects)} tracks to MVData format")
        finally:
            parser.cleanup()

    # clipgt_id - derived from ncore shard
    clip_id = next(iter(mvdata_objects.values())).clip_id if mvdata_objects else "unknown"
    # Convert MVData objects to Asset format
    assets_metadata: Dict[str, Asset] = metadata_manager.convert_mvdata_to_assets_metadata(mvdata_objects)

    # Stage 2: Multiview diffusion via SparseViewDiTPipeline
    with ScopedTimer("SparseViewDiTPipeline/__init__"):
        logger.info("Loading multiview diffusion models...")
        # GA wheel's get_models() expects safetensors format; convert .pth if needed
        if av_multiview_ckpt.endswith(".pth"):
            safetensors_path = av_multiview_ckpt.replace(".pth", ".safetensors")
            if not os.path.exists(safetensors_path):
                logger.info(f"Converting {av_multiview_ckpt} to safetensors format...")
                from safetensors.torch import save_file as _save_safetensors

                sd = torch.load(av_multiview_ckpt, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                _save_safetensors(sd, safetensors_path)
            av_multiview_ckpt = safetensors_path
        vae, cradio_model, cradio_processor, transformer = get_models(
            av_multiview_ckpt, device="cuda", dtype=torch.bfloat16
        )
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="flow_prediction",
            flow_shift=1.0,
            use_flow_sigmas=True,
        )
        pipeline = SparseViewDiTPipeline(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            scheduler=scheduler,
            transformer=transformer,
            image_encoder=cradio_model,
            image_processor=cradio_processor,
        ).to(torch.bfloat16)
        logger.info("Multiview diffusion model loaded successfully!")

    transform = T.Compose([T.Resize(512), T.ToTensor(), T.Normalize([0.5], [0.5])])
    inference_preproc = partial(
        preproc,
        image_transform=transform,
        resolution=512,
        conditioning_mode="n",
        eval_mode=True,
        eval_cam_sampler=build_eval_cams,
    )

    fov_list: List[float] = []
    dist_list: List[float] = []
    lwh_np_list: List[np.ndarray] = []
    mvimages_dict: Dict[str, List[np.ndarray]] = {}
    data_dicts: Dict[str, Any] = {}

    with ScopedTimer("SparseViewDiTPipeline/generate"):
        for track_id, parser_mvdata in mvdata_objects.items():
            diffusion_mvdata = to_diffusion_mvdata(parser_mvdata)
            poses = diffusion_mvdata.cam_poses
            idx = farthest_point_sampling(poses, num_samples=min(4, len(poses)), dist_threshold=0.1)
            shuffle_inds = [int(i) for i in idx]
            data_dict = inference_preproc(diffusion_mvdata, shuffle_inds=shuffle_inds)
            data_dicts[track_id] = data_dict

            with torch.no_grad():
                output = pipeline(
                    data_dict=data_dict, num_inference_steps=30, guidance_scale=2.0, flow_shift=1.0, output_type="pil"
                )

            track_output_dir = os.path.join(output_dir, track_id)
            fov_val = (
                float(data_dict.fovs[0].item()) if hasattr(data_dict.fovs[0], "item") else float(data_dict.fovs[0])
            )
            dist_val = (
                float(data_dict.dists[0].item()) if hasattr(data_dict.dists[0], "item") else float(data_dict.dists[0])
            )
            lwh_val = diffusion_mvdata.lwh

            # Save conditioning input views from parser MVData
            cond_pil = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in parser_mvdata.frames]
            mask_pil = (
                [Image.fromarray((np.asarray(m) * 255).astype(np.uint8)) for m in parser_mvdata.masks_instance]
                if parser_mvdata.masks_instance
                else []
            )
            if cond_pil and mask_pil:
                save_input_views(
                    cond_images=cond_pil,
                    mask_images=mask_pil,
                    output_dir=track_output_dir,
                )

            # Save multiview diffusion outputs (also returns images_np for lifting)
            images_np = save_mvd_outputs(
                images=output["images"],
                fov=fov_val,
                dist=dist_val,
                lwh=lwh_val,
                output_dir=track_output_dir,
            )

            fov_list.append(fov_val)
            dist_list.append(dist_val)
            if lwh_val is None:
                logger.warning(f"Track {track_id}: lwh is None, falling back to [1,1,1]")
            lwh_np_list.append(lwh_val if lwh_val is not None else np.array([1.0, 1.0, 1.0]))
            mvimages_dict[track_id] = images_np

    # Offload diffusion pipeline before lifting to free VRAM
    pipeline.to("cpu")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3: Gaussian lifting via TokengsLiftingRunner
    with ScopedTimer("TokengsLiftingRunner/__init__"):
        lifting_runner = TokengsLiftingRunner(
            ckpt_path=tokengs_ckpt,
            bbox_size=config.tokengs_lifting.bbox_size,
            dtype=torch.bfloat16,
            render_img_size=512,
        )

    metrics_calc = MetricsCalculator(device="cuda")

    with ScopedTimer("TokengsLiftingRunner/lift"):
        for idx, track_id in enumerate(mvimages_dict.keys()):
            fov = fov_list[idx]
            dist = dist_list[idx]
            lwh = lwh_np_list[idx]
            images_np = mvimages_dict[track_id]

            with torch.no_grad():
                gaussians = lifting_runner.run_lifting(images_np, fov, dist, lwh)

            track_output_dir = os.path.join(output_dir, track_id)
            ply_path = os.path.join(track_output_dir, "gaussians.ply")
            os.makedirs(track_output_dir, exist_ok=True)

            # Save PLY first; rotate it in place per-class; then render mp4 from the
            # rotated PLY so 3d_lifted.mp4 matches the on-disk gaussians.ply orientation.
            lifting_runner.save_ply(gaussians, ply_path)

            label_class = mvdata_objects[track_id].npct
            rot_spec = config.output.rotation.by_class.get(label_class)
            if rot_spec is not None and rot_spec.degrees != 0.0:
                ply_path_obj = Path(ply_path)
                transform = build_rotation_matrix(rot_spec.axis, rot_spec.degrees, "cpu")
                rotate_ply(ply_path_obj, ply_path_obj, transform, "cpu", force=True)
                gaussians = (
                    lifting_runner.model.gs.load_ply(str(ply_path_obj))
                    .unsqueeze(0)
                    .to(next(lifting_runner.model.parameters()).device)
                )

            rendered = lifting_runner.render_orbit_views(gaussians, fov, dist, lwh)
            rendered_np = [im.permute(1, 2, 0).numpy() for im in rendered]
            imageio.v2.mimwrite(
                os.path.join(track_output_dir, "3d_lifted.mp4"),
                rendered_np,
                fps=5,
                macro_block_size=1,
            )

            # Compute metrics at conditioning viewpoints
            asset_metrics = None
            try:
                pred, gt = lifting_runner.render_at_cond_views(gaussians, data_dicts[track_id])
                if pred is not None and gt is not None:
                    psnr = metrics_calc.calculate_psnr(pred.cuda(), gt.cuda(), reduction="none")
                    ssim = metrics_calc.calculate_ssim(pred.cuda(), gt.cuda(), reduction="none")
                    asset_metrics = {
                        "psnr_mean": float(psnr.mean()),
                        "psnr_std": float(psnr.std()),
                        "ssim_mean": float(ssim.mean()),
                        "ssim_std": float(ssim.std()),
                    }
            except Exception as e:
                logger.warning(f"Track {track_id}: metrics computation failed: {e}")

            # Update metadata assets with ply file paths and metrics
            clean_track_id = strip_track_id_suffix(track_id)
            if clean_track_id in assets_metadata and Path(ply_path).exists():
                asset = assets_metadata[clean_track_id]
                asset.ply_file = str(Path(ply_path).relative_to(output_dir))

                if asset_metrics is not None:
                    asset.metrics = AssetMetrics(
                        psnr_mean=asset_metrics["psnr_mean"],
                        psnr_std=asset_metrics["psnr_std"],
                        ssim_mean=asset_metrics["ssim_mean"],
                        ssim_std=asset_metrics["ssim_std"],
                    )
                else:
                    asset.metrics = None

                logger.info(f"PLY file saved for track {track_id}: {asset.ply_file}")
            else:
                logger.warning(f"PLY file not found for track {track_id} at path: {ply_path}")

    # Offload lifting runner after all tracks are processed
    lifting_runner.model.cpu()
    del lifting_runner
    gc.collect()
    torch.cuda.empty_cache()

    metadata = AssetHarvestingMetadata(clip_id=clip_id, config=config, assets=assets_metadata)
    metadata_manager.set_metadata(metadata)
    metadata_manager.save()

    # Print timing summary if timing is enabled
    if enable_timing:
        ScopedTimer.print_summary()

    logger.info(f"Asset harvesting is done, output saved to {output_dir}!")


if __name__ == "__main__":
    try:
        asset_harvest()
    except Exception as e:
        logger.error(f"Asset harvester failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
