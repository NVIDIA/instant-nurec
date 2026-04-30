#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import subprocess

from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def cmd_images_to_mp4(
    frames_dir: Path,
    pattern: str,
    out_mp4: Path,
    framerate: int,
    profile: str,
    pix_fmt: str,
    bitrate: str,
    preset: str,
) -> None:
    ffmpeg = get_ffmpeg_exe()
    input_pattern = str(frames_dir / pattern)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(framerate),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-profile:v",
            profile,
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            bitrate,
            "-preset",
            preset,
            str(out_mp4),
        ]
    )


def cmd_grid3(
    top_mp4: Path,
    left_mp4: Path,
    right_mp4: Path,
    out_mp4: Path,
) -> None:
    ffmpeg = get_ffmpeg_exe()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Compose 2x2 layout: top centered on 2*W canvas; bottom is hstack(left, right); then vstack(top, bottom)
    filter_complex = (
        "[0:v]scale=iw:ih,pad=2*iw:ih:iw/2:0:black[top]; "
        "[1:v][2:v]hstack=inputs=2[bottom]; "
        "[top][bottom]vstack=inputs=2[v]"
    )
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-i",
            str(top_mp4),
            "-i",
            str(left_mp4),
            "-i",
            str(right_mp4),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            str(out_mp4),
        ]
    )


def cmd_grid4(
    top_left_mp4: Path,
    top_right_mp4: Path,
    bottom_left_mp4: Path,
    bottom_right_mp4: Path,
    out_mp4: Path,
) -> None:
    ffmpeg = get_ffmpeg_exe()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # 2x2 grid using stack filters: hstack top row (0,1), hstack bottom row (2,3), then vstack rows
    filter_complex = (
        "[0:v][1:v]hstack=inputs=2[top]; [2:v][3:v]hstack=inputs=2[bottom]; [top][bottom]vstack=inputs=2[v]"
    )
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-i",
            str(top_left_mp4),
            "-i",
            str(top_right_mp4),
            "-i",
            str(bottom_left_mp4),
            "-i",
            str(bottom_right_mp4),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            str(out_mp4),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FFmpeg wrapper using imageio-ffmpeg (no system ffmpeg required)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("images-to-mp4", help="Encode frames into MP4")
    p_img.add_argument("--frames-dir", required=True, type=Path, help="Directory containing frames")
    p_img.add_argument("--pattern", default="%06d.jpg", help="ffmpeg input pattern (default: %06d.jpg)")
    p_img.add_argument("--out", required=True, type=Path, help="Output MP4 path")
    p_img.add_argument("--framerate", type=int, default=30)
    p_img.add_argument("--profile", default="high")
    p_img.add_argument("--pix-fmt", default="yuv420p")
    p_img.add_argument("--bitrate", default="8462k")
    p_img.add_argument("--preset", default="medium")

    p_grid = sub.add_parser("grid3", help="Compose 3 MP4s into a 2x2 grid (top centered)")
    p_grid.add_argument("--top", required=True, type=Path)
    p_grid.add_argument("--left", required=True, type=Path)
    p_grid.add_argument("--right", required=True, type=Path)
    p_grid.add_argument("--out", required=True, type=Path)

    p_grid4 = sub.add_parser("grid4", help="Compose 4 MP4s into a 2x2 grid")
    p_grid4.add_argument("--top-left", required=True, type=Path)
    p_grid4.add_argument("--top-right", required=True, type=Path)
    p_grid4.add_argument("--bottom-left", required=True, type=Path)
    p_grid4.add_argument("--bottom-right", required=True, type=Path)
    p_grid4.add_argument("--out", required=True, type=Path)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "images-to-mp4":
        cmd_images_to_mp4(
            frames_dir=args.frames_dir,
            pattern=args.pattern,
            out_mp4=args.out,
            framerate=args.framerate,
            profile=args.profile,
            pix_fmt=args.pix_fmt,
            bitrate=args.bitrate,
            preset=args.preset,
        )
    elif args.cmd == "grid3":
        cmd_grid3(top_mp4=args.top, left_mp4=args.left, right_mp4=args.right, out_mp4=args.out)
    elif args.cmd == "grid4":
        cmd_grid4(
            top_left_mp4=args.top_left,
            top_right_mp4=args.top_right,
            bottom_left_mp4=args.bottom_left,
            bottom_right_mp4=args.bottom_right,
            out_mp4=args.out,
        )


if __name__ == "__main__":
    main()
