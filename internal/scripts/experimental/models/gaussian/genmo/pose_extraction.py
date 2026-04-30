#!/usr/bin/env python3
"""Pose extraction utilities using GenMO's HMR4D pipeline."""

import functools
import logging

from pathlib import Path

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)


# Path constants (relative to hmr4d.PROJ_ROOT)
CHECKPOINT_RELPATH = "inputs/mocap_mixed_v1/genmo/genmo_lg_nvhuman_v4+v9/version_0/checkpoints/last.ckpt"

PREPROCESS_FILES = {
    "bbx": "bbx.pt",
    "vitpose": "vitpose.pt",
    "vit_features": "vit_features.pt",
    "slam": "slam.npy",
    "cam_int": "cam_int.npy",
}

# Initialize GenMO PROJ_ROOT before importing GenMO modules
import hmr4d
import hydra

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.genmo.genmo_demo import GENMO_demo
from hmr4d.utils.geo.hmr_cam import convert_K_to_K4, estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, read_video_np, save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hydra import compose, initialize_config_module

from internal.scripts.experimental.models.gaussian.genmo import genmo_init  # noqa: F401


def run_preprocess(
    video_path: Path,
    output_dir: Path,
    static_cam: bool = True,
    verbose: bool = False,
    orig_fps: float = 30.0,
) -> dict:
    """Run preprocessing pipeline to extract features from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save preprocessing outputs
        static_cam: If True, assumes static camera. If False, runs DROID-SLAM for camera motion
        verbose: If True, saves debug visualizations
        orig_fps: Original video FPS for debug visualizations

    Returns:
        Dictionary containing paths to preprocessed data files
    """
    logger.info("[Preprocess] Start!")

    preprocess_dir = output_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    length, width, height = get_video_lwh(str(video_path))

    paths = {key: preprocess_dir / filename for key, filename in PREPROCESS_FILES.items()}

    # 1. Get bounding box tracking
    if not paths["bbx"].exists():
        logger.info("Extracting bounding boxes...")
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(str(video_path)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths["bbx"])
        del tracker
    else:
        bbx_xys = torch.load(paths["bbx"])["bbx_xys"]
        logger.info(f"[Preprocess] bbx (xyxy, xys) from {paths['bbx']}")

    if verbose:
        video = read_video_np(str(video_path))
        bbx_xyxy = torch.load(paths["bbx"])["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, str(preprocess_dir / "bbx_overlay.mp4"), fps=orig_fps)

    # 2. Get VitPose (17 joints)
    if not paths["vitpose"].exists():
        logger.info("Extracting VitPose...")
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(str(video_path), bbx_xys)
        torch.save(vitpose, paths["vitpose"])
        del vitpose_extractor
    else:
        vitpose = torch.load(paths["vitpose"])
        logger.info(f"[Preprocess] vitpose from {paths['vitpose']}")

    if verbose:
        video = read_video_np(str(video_path))
        if isinstance(vitpose, tuple):
            vitpose_vis = vitpose[0]
        else:
            vitpose_vis = vitpose
        video_overlay = draw_coco17_skeleton_batch(video, vitpose_vis, 0.5)
        save_video(video_overlay, str(preprocess_dir / "vitpose_overlay.mp4"), fps=orig_fps)

    # 3. Get DROID-SLAM results (if not static camera)
    if not static_cam:
        if not Path(paths["slam"]).exists():
            # TODO: Integrate with DROID-SLAM
            logger.info("Running DROID-SLAM for camera motion estimation...")
        else:
            logger.info(f"[Preprocess] slam results from {paths['slam']}")
    else:
        # Static camera - just save camera intrinsics
        K_fullimg = estimate_K(width, height)
        cam_int = [K_fullimg[0, 0], K_fullimg[1, 1], K_fullimg[0, 2], K_fullimg[1, 2]]
        np.save(paths["cam_int"], cam_int)

    # 4. Extract ViT features
    if not paths["vit_features"].exists():
        logger.info("Extracting ViT features...")
        extractor = Extractor()
        vit_features = extractor.extract_video_features(str(video_path), bbx_xys)
        torch.save(vit_features, paths["vit_features"])
        del extractor
    else:
        logger.info(f"[Preprocess] vit_features from {paths['vit_features']}")

    logger.info("[Preprocess] Complete!")
    return paths


def load_data_dict(video_path: Path, paths: dict, static_cam: bool = True) -> dict:
    """Load preprocessed data and prepare input dictionary for HMR4D model.

    Args:
        video_path: Path to input video file
        paths: Dictionary of paths to preprocessed data files
        static_cam: If True, uses static camera parameters. If False, loads SLAM results

    Returns:
        Dictionary containing all inputs for HMR4D model
    """
    length, width, height = get_video_lwh(str(video_path))
    video_name = Path(video_path).stem

    # Load camera parameters
    if static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
        mean_scale = torch.tensor(1.0)
        scales = torch.ones(length)
        T_w2c = torch.eye(4)[None].repeat(length, 1, 1)
        T_w2c[:, :3, :3] = R_w2c
        T_w2c[:, :3, 3] = t_w2c
    else:
        raise NotImplementedError("DROID-SLAM is not supported yet.")

    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    vitpose = torch.load(paths["vitpose"])
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
        assert vitpose.ndim == 3
        vitpose = vitpose[:, 5:]

    # Note: cfg parameters will be filled in by the calling function
    data = {
        "meta": [{"vid": video_name}],
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths["bbx"])["bbx_xys"],
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "f_imgseq": torch.load(paths["vit_features"]),
        "scales": scales,
        "mean_scale": mean_scale,
        "T_w2c": T_w2c,
        "has_text": torch.tensor([False]),
        "mask": {
            "has_img_mask": torch.ones(length).bool(),
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }
    return data


def extract_poses_genmo(
    video_path: Path,
    output_dir: Path,
    static_cam: bool = True,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Extract poses from video using GenMO's HMR4D pipeline.

    This function processes a video to extract human pose sequences using the GenMO HMR4D model.
    It handles the full pipeline including:
    - Bounding box detection and tracking
    - VitPose extraction for 17 joints
    - ViT feature extraction
    - Camera motion estimation (if not static_cam)
    - HMR4D model inference

    Args:
        video_path: Path to input video file
        output_dir: Directory to save preprocessing outputs and results
        static_cam: If True, assumes static camera. If False, runs DROID-SLAM for camera motion
        checkpoint_path: Optional path to model checkpoint. If None, uses default path.
        verbose: If True, saves debug visualizations

    Returns:
        Path to the saved HMR4D results (.pt file)

    Raises:
        FileNotFoundError: If GenMO project root or required files cannot be found
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    hmr4d_results_path = output_dir / video_name / "hmr4d_results.pt"

    # Get original FPS for visualization
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Run preprocessing
    paths = run_preprocess(video_path, output_dir, static_cam=static_cam, verbose=verbose, orig_fps=orig_fps)

    # Load data
    data = load_data_dict(video_path, paths, static_cam=static_cam)

    # Run HMR4D inference
    if not hmr4d_results_path.exists():
        logger.info("[HMR4D] Running inference...")
        hmr4d_results_path.parent.mkdir(parents=True, exist_ok=True)

        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            cfg = compose(config_name="demo_genmo_nvhuman", overrides=[f"static_cam={static_cam}"])

        model: GENMO_demo = hydra.utils.instantiate(cfg.model, _recursive_=False)

        # Add config-dependent fields to data
        length = data["length"].item()
        data["text_embed"] = torch.zeros(1, cfg.network.model_cfg.denoiser.encoded_text_dim)
        data["music_embed"] = torch.zeros(length, cfg.pipeline.args.encoded_music_dim)

        # Use provided checkpoint path or default
        if checkpoint_path is None:
            checkpoint_path = hmr4d.PROJ_ROOT / CHECKPOINT_RELPATH

        if checkpoint_path.exists():
            logger.info(f"Loading model checkpoint from: {checkpoint_path}")
            model.load_pretrained_model(str(checkpoint_path))
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Using default initialization.")

        model = model.eval().cuda()
        pred = model.predict(data, static_cam=static_cam)
        pred = detach_to_cpu(pred)
        torch.save(pred, hmr4d_results_path)
        logger.info(f"HMR4D results saved to: {hmr4d_results_path}")
    else:
        logger.info(f"Loading cached HMR4D results from: {hmr4d_results_path}")

    return hmr4d_results_path
