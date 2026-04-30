import logging
import os
import sys
import traceback

from pathlib import Path
from typing import Optional

import click
import imageio
import torch

from tqdm import tqdm


logger = logging.getLogger(__name__)

from internal.scripts.experimental.models.gaussian.nvhuman.gaussian_nvhuman_layer import GaussianNVHumanLayer
from nre.utils.gaussian_render import RenderConfig, render_gaussians_single_view


def animate_nvhuman_core(
    nvhuman_path: Path,
    poses_path: Path,
    output_dir: Path,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_step: int = 1,
    fps: int = 30,
    save_frames: bool = True,
    save_video: bool = True,
    output_size: int = 512,
    elevation: float = 0,
    distance: float = 1.5,
    fov: float = 70,
):
    """Core animation logic without CLI decoration."""

    logger.info("Starting NVHuman animation...")
    logger.info(f"NVHuman model: {nvhuman_path}")
    logger.info(f"Poses file: {poses_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frame range: {start_frame} to {end_frame} (step: {frame_step})")

    opt = RenderConfig(output_size=output_size, elevation=elevation, dist=distance, fov=fov, fps=fps)

    logger.debug(f"Camera settings: elevation={opt.elevation}°, azimuth=0°, distance={opt.dist}")

    logger.info("Loading NVHuman model...")
    nvhuman = GaussianNVHumanLayer(str(nvhuman_path))
    nvhuman = nvhuman.cuda()

    logger.info(f"Loaded NVHuman model: {nvhuman.v_template.shape[0]} vertices, {nvhuman.num_joints} joints")
    if nvhuman.has_gaussians:
        logger.debug(f"Gaussian count: {nvhuman.num_gaussians}")

    logger.info("Loading pose sequence...")
    logger.debug(f"Loading pose sequence from: {poses_path}")
    if not Path(poses_path).exists():
        raise FileNotFoundError(f"Pose sequence file not found: {poses_path}")

    pose_data = torch.load(poses_path, map_location="cpu")
    if "smpl_params_incam" in pose_data:
        pose_sequence = pose_data["smpl_params_incam"]
        for key, value in pose_sequence.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: {value.shape}")
    else:
        raise ValueError("Could not find 'smpl_params_incam' in pose sequence file")

    # Get sequence length
    total_frames = 0
    for key, value in pose_sequence.items():
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            total_frames = value.shape[0]
            break

    if end_frame == -1:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)

    frame_indices = list(range(start_frame, end_frame, frame_step))

    logger.info("Animation info:")
    logger.info(f"  Total frames in sequence: {total_frames}")
    logger.info(f"  Frame range: {start_frame} to {end_frame} (step: {frame_step})")
    logger.info(f"  Output frames: {len(frame_indices)}")
    logger.info(f"  Video duration: {len(frame_indices) / fps:.2f}s at {fps} FPS")
    if len(frame_indices) > 0:
        logger.debug(f"  Frame indices: {frame_indices[0]} to {frame_indices[-1]}")

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = output_dir / "frames" if save_frames else None
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    device = torch.device("cuda")
    rendered_frames = []

    logger.info(f"Rendering {len(frame_indices)} frames...")

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Rendering frames")):
        # Extract frame pose parameters
        frame_pose_params = {}
        for key, value in pose_sequence.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1 and value.shape[0] > frame_idx:
                    frame_pose_params[key] = value[frame_idx : frame_idx + 1].to(device)
                elif value.dim() > 1:
                    frame_pose_params[key] = value[-1:].to(device)
                else:
                    frame_pose_params[key] = value.to(device)
            else:
                frame_pose_params[key] = value

        # Zero out transl - keep character at origin where camera is pointing
        # (Gaussians were bound at origin, so keep poses centered there too)
        if "transl" in frame_pose_params:
            frame_pose_params["transl"] = torch.zeros_like(frame_pose_params["transl"])

        with torch.no_grad():
            output = nvhuman(**frame_pose_params)

            if "gaussian_params" in output:
                gaussian_params = output["gaussian_params"]

                # Debug info for first frame
                if i == 0:
                    positions = gaussian_params["positions"][0]
                    logger.debug(
                        f"Gaussian positions: min={positions.min().item():.3f}, max={positions.max().item():.3f}, mean={positions.mean().item():.3f}"
                    )
                    logger.debug(
                        f"Position center: [{positions.mean(0)[0].item():.3f}, {positions.mean(0)[1].item():.3f}, {positions.mean(0)[2].item():.3f}]"
                    )

                # Use config camera settings (azimuth=0 for front view)
                frame_image = render_gaussians_single_view(gaussian_params, opt, opt.elevation, 0, opt.dist)

                if save_frames:
                    frame_filename = frames_dir / f"frame_{i:06d}.png"
                    imageio.imwrite(frame_filename, frame_image)

                rendered_frames.append(frame_image)
            else:
                logger.warning(f"No Gaussian parameters found for frame {frame_idx}")

    if save_video and len(rendered_frames) > 0:
        logger.info(f"Creating animation video from {len(rendered_frames)} frames...")
        video_path = output_dir / f"nvhuman_animation_{fps}fps.mp4"

        try:
            with imageio.get_writer(str(video_path), fps=fps, quality=8, macro_block_size=1) as writer:
                for frame in rendered_frames:
                    writer.append_data(frame)
            logger.info(f"Saved animation video: {video_path}")
        except (IOError, OSError) as e:
            logger.error(f"Error creating video: {e}")
            traceback.print_exc()
            try:
                imageio.mimsave(str(video_path), rendered_frames, fps=fps)
                logger.info(f"Saved animation video (fallback): {video_path}")
            except (IOError, OSError) as e2:
                logger.error(f"Fallback video creation failed: {e2}")
                traceback.print_exc()
    elif save_video:
        logger.warning("No frames to create video from")

    logger.info("Animation completed!")
    logger.info(f"Output directory: {output_dir}")
    if save_frames and frames_dir:
        logger.info(f"Individual frames: {frames_dir}")
    if save_video and len(rendered_frames) > 0 and "video_path" in locals():
        logger.info(f"Animation video: {video_path}")


@click.command()
@click.option("--nvhuman-path", type=click.Path(exists=True), help="Path to NVHuman .npz file")
@click.option("--poses-path", type=click.Path(exists=True), help="Path to pose sequence file (.pt)")
@click.option("--output-dir", type=click.Path(), help="Output directory for animation frames/video")
@click.option("--start-frame", type=int, default=0, help="Starting frame index")
@click.option("--end-frame", type=int, default=-1, help="Ending frame index (-1 for all frames)")
@click.option("--frame-step", type=int, default=1, help="Frame step size")
@click.option("--fps", type=int, default=30, help="Frames per second for output video")
@click.option("--save-frames/--no-save-frames", default=True, help="Save individual frames")
@click.option("--save-video/--no-save-video", default=True, help="Save output video")
@click.option("--output-size", type=int, default=512, help="Output image resolution")
@click.option("--elevation", type=float, default=0, help="Camera elevation angle in degrees")
@click.option("--distance", type=float, default=1.5, help="Camera distance from subject")
@click.option("--fov", type=float, default=70, help="Field of view in degrees")
def animate_nvhuman(
    nvhuman_path: Optional[str],
    poses_path: Optional[str],
    output_dir: Optional[str],
    start_frame: int,
    end_frame: int,
    frame_step: int,
    fps: int,
    save_frames: bool,
    save_video: bool,
    output_size: int,
    elevation: float,
    distance: float,
    fov: float,
):
    """CLI wrapper for animate_nvhuman_core."""
    if not nvhuman_path or not poses_path or not output_dir:
        logger.error("Error: --nvhuman-path, --poses-path, and --output-dir are required")
        return

    animate_nvhuman_core(
        nvhuman_path=Path(nvhuman_path),
        poses_path=Path(poses_path),
        output_dir=Path(output_dir),
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    animate_nvhuman()
