# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path
from typing import Optional, cast

import click
import numpy as np
import torch

from tqdm import tqdm

import nre.systems

from ncore.data import ConcreteCameraModelParametersUnion
from ncore.sensors import CameraModel
from nre.config.parse import parse_typed_config
from nre.utils.batch import DataAndRenderingBatch, generate_grid_2d_indices


@click.command("export-depth")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.option(
    "--checkpoint-name",
    type=str,
    help="Checkpoint file name to load",
    default="last.ckpt",
    required=False,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=False,
)
@click.option(
    "--depth-type",
    type=click.Choice(["euclidean", "z-depth"], case_sensitive=False),
    help="Whether to export euclidean or z-depth",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
@torch.inference_mode()
def export_depth(
    config_name: str,
    checkpoint_name: str,
    output_dir: Optional[str],
    depth_type: str,
    hydra_args: list[str],
):
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    # Load last checkpoint.
    checkpoint_path: Path = Path(config.ckpt_dir) / checkpoint_name
    config.mode = "val"
    if config.resume is None:
        config.resume = str(checkpoint_path)

    system = nre.systems.make(config.system.name, config, load_from_checkpoint=str(checkpoint_path))
    assert isinstance(system, nre.systems.GaussiansSystem), (
        f"Only GaussiansSystem is supported for depth export, got {type(system)}"
    )

    # If no separate out directory is specified, re-use the checkpoints directory.
    if output_dir is None:
        output_path = Path(config.ckpt_dir).parent / "depth"
    else:
        output_path = Path(config.out_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(tqdm(system.datamodule.train_dataloader_sequential())):
        assert isinstance(batch, DataAndRenderingBatch)
        batch = batch.to(system.device)
        rendered = system(batch).rendered_cam
        # batch.rendering should be inplace updated by the system
        assert batch.rendering is not None, "batch.rendering should be inplace updated by the system"
        assert batch.rendering.camera is not None, "batch.rendering.camera is expect to exist"

        assert batch.rendering.camera.b == 1, "Assuming batch size 1 here"
        height, width = batch.rendering.camera.h, batch.rendering.camera.w
        parameters_cam = batch.rendering.camera.sensor_model_parameters[0]
        parameters_cam = cast(ConcreteCameraModelParametersUnion, parameters_cam)

        distance = rendered.distance.view(height, width)
        match depth_type:
            case "euclidean":
                depth = distance
            case "z-depth":
                camera_model = CameraModel.from_parameters(
                    parameters_cam, device=str(distance.device), dtype=distance.dtype
                )
                pixel_indices = generate_grid_2d_indices(resolution=(width, height), order="xy", device=distance.device)
                camera_rays = camera_model.pixels_to_camera_rays(pixel_indices)
                depth = distance * camera_rays[..., 2].view(height, width)
            case _:
                raise RuntimeError(f"Unrecognized depth type: {depth_type}")

        np.savez_compressed(output_path / f"depth_{i:06d}.npy", depth=depth.cpu().numpy())
