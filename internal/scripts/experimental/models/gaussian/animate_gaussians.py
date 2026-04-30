#!/usr/bin/env python3
"""Animate Gaussian Splat to NVHuman Animation Pipeline."""

import glob
import logging

from pathlib import Path

import click
import hmr4d
import imageio
import torch

# Initialize GenMO PROJ_ROOT before importing any GenMO/hmr4d modules
from internal.scripts.experimental.models.gaussian.genmo import genmo_init  # noqa: F401
from internal.scripts.experimental.models.gaussian.genmo.pose_extraction import extract_poses_genmo
from internal.scripts.experimental.models.gaussian.nvhuman.animate_nvhuman import animate_nvhuman_core
from internal.scripts.experimental.models.gaussian.nvhuman.gaussian_to_nvhuman import GaussianToNVHumanConverter
from nre.utils.gaussian_render import RenderConfig, render_ply_orbit


logger = logging.getLogger(__name__)


def create_video_from_frames(render_dir: Path, video_path: Path, fps: int):
    """Create video from PNG frames using imageio."""
    frame_files = sorted(glob.glob(str(render_dir / "*.png")))
    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {render_dir}")

    frames = [imageio.v2.imread(f) for f in frame_files]
    imageio.v2.mimwrite(str(video_path), frames, fps=fps)


def convert_gaussian_to_nvhuman(
    input_ply: Path,
    output_dir: Path,
    ply_name: str,
    reference_pose_file: Path,
    nvhuman_path: Path,
):
    """Convert Gaussian PLY to NVHuman format."""
    converter = GaussianToNVHumanConverter()
    reference_pose_params = converter.extract_pose_from_file(reference_pose_file, frame_idx=0)
    gaussian_ply = converter.load_gaussian_ply(input_ply)
    nvhuman_data = converter.create_nvhuman_model(gaussian_ply, output_dir, ply_name, reference_pose_params)
    converter.save_nvhuman_model(nvhuman_data, nvhuman_path)


@click.command()
@click.option("--input-ply", type=click.Path(exists=True), required=True, help="Path to input Gaussian PLY file")
@click.option("--output-dir", type=click.Path(), required=True, help="Output directory for all results")
@click.option(
    "--target-poses",
    type=click.Path(exists=True),
    default=None,
    help="Path to target pose sequence (.pt file) for animation. If not provided, uses extracted poses from rendered video.",
)
@click.option("--fps", type=int, default=30, help="Frames per second for output video")
@click.option("--start-frame", type=int, default=0, help="Starting frame index for animation")
@click.option("--end-frame", type=int, default=-1, help="Ending frame index (-1 for all frames)")
@click.option("--frame-step", type=int, default=1, help="Frame step size for animation")
@click.option("--save-frames/--no-save-frames", default=True, help="Save individual animation frames")
@click.option("--save-video/--no-save-video", default=True, help="Save final animation video")
@click.option("--output-size", type=int, default=512, help="Output image resolution")
@click.option("--elevation", type=float, default=0, help="Camera elevation angle in degrees")
@click.option("--distance", type=float, default=1.5, help="Camera distance from subject")
@click.option("--fov", type=float, default=70, help="Field of view in degrees")
@click.option(
    "--static-cam",
    is_flag=True,
    default=False,
    help="If set, assumes static camera. By default uses DROID-SLAM for camera motion estimation",
)
def main(
    input_ply,
    output_dir,
    target_poses,
    fps,
    start_frame,
    end_frame,
    frame_step,
    save_frames,
    save_video,
    output_size,
    elevation,
    distance,
    fov,
    static_cam,
):
    """Animate Gaussian Splat to NVHuman"""

    input_ply = Path(input_ply).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_name = input_ply.stem
    render_dir = output_dir / "render"
    video_path = render_dir / f"{ply_name}.mp4"
    pose_output_dir = output_dir / "poses"
    nvhuman_path = output_dir / f"{ply_name}_nvhuman.npz"
    animation_dir = output_dir / "animation"

    render_dir.mkdir(exist_ok=True)

    render_config = RenderConfig(output_size=output_size, elevation=elevation, dist=distance, fov=fov, fps=fps)

    # GenMO project root is already initialized by pose_extraction module
    logger.info(f"GenMO project root: {hmr4d.PROJ_ROOT}")

    logger.info("Gaussian Animation Pipeline")
    logger.info(f"Input: {input_ply.name}")
    logger.info(f"Output: {output_dir}")

    logger.info("Stage 1: Render PLY")

    render_ply_orbit(
        ply_path=input_ply, output_dir=render_dir, format="image+video", compatible=False, opt=render_config
    )
    create_video_from_frames(render_dir, video_path, fps)

    logger.info("Stage 2: Extract poses")
    pose_output_dir.mkdir(exist_ok=True)
    reference_pose_file = extract_poses_genmo(video_path, pose_output_dir, static_cam=static_cam)

    logger.info("Stage 3: Convert to NVHuman")
    convert_gaussian_to_nvhuman(input_ply, output_dir, ply_name, reference_pose_file, nvhuman_path)

    logger.info("Stage 4: Animate")
    animation_dir.mkdir(exist_ok=True)

    animation_pose_file = Path(target_poses) if target_poses else reference_pose_file

    animate_nvhuman_core(
        nvhuman_path=nvhuman_path,
        poses_path=animation_pose_file,
        output_dir=animation_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        fps=fps,
        save_frames=save_frames,
        save_video=save_video,
        output_size=output_size,
        elevation=elevation,
        distance=distance,
        fov=fov,
    )

    logger.info(f"✓ Done! Output: {animation_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
