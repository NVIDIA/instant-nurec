# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import ncore.data
import ncore.impl.common.transformations
import ncore_internal.data.v3

from nre.utils import types


@dataclass
class BBoxRect:
    points: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float64))

    def set_low_x(self, low_x):
        self.points[0, 0] = low_x
        self.points[1, 0] = low_x

    def set_low_y(self, low_y):
        self.points[0, 1] = low_y
        self.points[3, 1] = low_y

    def set_high_x(self, high_x):
        self.points[2, 0] = high_x
        self.points[3, 0] = high_x

    def set_high_y(self, high_y):
        self.points[1, 1] = high_y
        self.points[2, 1] = high_y

    def from_points(self, points):
        min_x = np.min(points[:, 0], axis=0)
        min_y = np.min(points[:, 1], axis=0)
        max_x = np.max(points[:, 0], axis=0)
        max_y = np.max(points[:, 1], axis=0)

        self.points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]], dtype=np.float64)

    def inside_mask(self, points):
        valid = points[:, 0] >= self.points[0, 0]
        valid &= points[:, 1] >= self.points[0, 1]
        valid &= points[:, 0] <= self.points[2, 0]
        valid &= points[:, 1] <= self.points[1, 1]

        return valid


@dataclass
class State:
    @dataclass
    class SequenceData:
        sequence_meta: dict
        sequence_meta_file: str
        positions: np.ndarray
        position_timestamps_us: np.ndarray

        reference_sensor: Optional[ncore_internal.data.v3.Sensor]

    sequence_datas: dict[str, SequenceData] = field(default_factory=dict)

    chunks: list[types.SequenceChunk] = field(default_factory=list)

    bbox_rect = BBoxRect()

    output_file: str = ""

    def process(self):
        self.ui_bbox()

        psim.Separator()  # type: ignore

        psim.TextUnformatted(f"nChunks: {len(self.chunks)}")  # type: ignore

        if len(self.chunks):
            chunk_times_sec = [chunk.time_length_sec() for chunk in self.chunks]

            psim.TextUnformatted(f"Chunk times: min {min(chunk_times_sec)}s / max {max(chunk_times_sec)}")  # type: ignore

        if psim.Button(f"Store to {self.output_file}"):  # type: ignore
            self.store_chunks()

    def ui_bbox(self):
        recompute_chunks = False

        changed, ui_v = psim.SliderFloat("bbox_low_x", self.bbox_rect.points[0, 0], v_min=-10000.0, v_max=10000.0)  # type: ignore
        if changed:
            self.bbox_rect.set_low_x(ui_v)
            recompute_chunks = True

        changed, ui_v = psim.SliderFloat("bbox_low_y", self.bbox_rect.points[0, 1], v_min=-10000.0, v_max=10000.0)  # type: ignore
        if changed:
            self.bbox_rect.set_low_y(ui_v)
            recompute_chunks = True

        changed, ui_v = psim.SliderFloat("bbox_high_x", self.bbox_rect.points[2, 0], v_min=-10000.0, v_max=10000.0)  # type: ignore
        if changed:
            self.bbox_rect.set_high_x(ui_v)
            recompute_chunks = True

        changed, ui_v = psim.SliderFloat("bbox_high_y", self.bbox_rect.points[2, 1], v_min=-10000.0, v_max=10000.0)  # type: ignore
        if changed:
            self.bbox_rect.set_high_y(ui_v)
            recompute_chunks = True

        if recompute_chunks:
            ps.get_curve_network("bbox_rect").update_node_positions(self.bbox_rect.points)
            self.update_chunks()

    def store_chunks(self):
        out = {"sequences": {}}

        for sequence_id, sequence_data in self.sequence_datas.items():
            out["sequences"][sequence_id] = {"sequence-meta-file": sequence_data.sequence_meta_file, "chunks": []}

            if reference_sensor := sequence_data.reference_sensor:
                out["sequences"][sequence_id]["reference-sensor-id"] = reference_sensor.get_sensor_id()

        for chunk in self.chunks:
            chunk_data = {
                "start-timestamp_us": int(chunk.time_range_us.start),
                "end-timestamp_us": int(chunk.time_range_us.end),
            }

            if reference_sensor := self.sequence_datas[chunk.sequence_id].reference_sensor:
                chunk_data |= {
                    "reference-sensor-start-frame": reference_sensor.get_closest_frame_index(
                        chunk_data["start-timestamp_us"]
                    ),
                    "reference-sensor-end-frame": reference_sensor.get_closest_frame_index(
                        chunk_data["end-timestamp_us"]
                    ),
                }

            out["sequences"][chunk.sequence_id]["chunks"].append(chunk_data)

        with open(self.output_file, "w") as f:
            json.dump(out, f, indent=4)

        logging.info(f"Stored {len(self.chunks)} chunks to {self.output_file}")

    def update_chunks(self):
        # Remove all current chunks from rendering
        for chunk in self.chunks:
            chunk_name = f"{chunk}"
            if ps.has_curve_network(chunk_name):
                ps.remove_curve_network(chunk_name)

        self.chunks.clear()

        # Compute chunks
        for sequence_id, sequence_data in self.sequence_datas.items():
            # true for all points inside the bbox
            inside_mask = self.bbox_rect.inside_mask(sequence_data.positions)

            # the indices of all inside points
            inside_indices = np.arange(len(sequence_data.positions))[inside_mask]

            # extract *consecutive* sub-ranges of inside points
            consecutive_inside_indices = np.split(inside_indices, np.where(np.diff(inside_indices) != 1)[0] + 1)

            for consecutive_inside in consecutive_inside_indices:
                if len(consecutive_inside) < 2:
                    continue

                chunk = types.SequenceChunk(
                    sequence_id,
                    types.HalfClosedInterval(
                        sequence_data.position_timestamps_us[consecutive_inside[0]].item(),
                        sequence_data.position_timestamps_us[consecutive_inside[-1]].item(),
                    ),
                )
                ps.register_curve_network(
                    f"{chunk}", sequence_data.positions[consecutive_inside], "line", enabled=True, radius=0.0015
                )

                self.chunks += [chunk]


@click.command()
@click.option(
    "--sequence-meta",
    "sequence_meta_files",
    type=str,
    multiple=True,
    help="NCORE sequence meta file to load",
    required=True,
)
@click.option(
    "--output-file", type=str, help="Path to the output file of selected sequence chunks (json)", required=True
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option(
    "--reference-sensor-id",
    default=None,
    type=str,
    help="If provided, a reference sensor to also output frame-bounds for",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging outputs")
def ncore_sequence_chunks(
    sequence_meta_files: list[str],
    output_file: str,
    open_consolidated: bool,
    reference_sensor_id: Optional[str],
    verbose: bool,
):
    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    state = State()
    state.output_file = output_file

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_user_callback(lambda: state.process())

    ## load NCORE sequence metas

    # transformation from base world back to common world (to keep coordinates small / centered) - common world is initialized to the world frame of the first sequence
    T_world_base_world_common: Optional[np.ndarray] = None

    for i, sequence_meta_file in enumerate(sequence_meta_files):
        sequence_path = Path(sequence_meta_file).parent
        with open(sequence_meta_file, "r") as fp:
            sequence_meta = json.load(fp)

        sequence_id = sequence_meta["sequence_id"]

        # collect + load all shards
        shard_paths: list[str] = [str(sequence_path / shard["path"]) for shard in sequence_meta["shards"]]

        loader = ncore_internal.data.v3.ShardDataLoader(shard_paths, open_consolidated=open_consolidated)
        poses = loader.get_poses()

        if T_world_base_world_common is None:
            # initialize common world first sequence's world frame
            T_world_base_world_common = ncore.impl.common.transformations.se3_inverse(poses.T_rig_world_base)

        # load world positions
        world_local_positions = poses.T_rig_worlds[:, :3, 3:].squeeze()

        # transform to common world positions
        world_common_positions = ncore.impl.common.transformations.transform_point_cloud(
            world_local_positions, T_world_base_world_common @ poses.T_rig_world_base
        )

        # load reference sensor, if specified
        reference_sensor = None
        if reference_sensor_id:
            reference_sensor = loader.get_sensor(reference_sensor_id)

        state.sequence_datas[sequence_id] = State.SequenceData(
            sequence_meta, sequence_meta_file, world_common_positions, poses.T_rig_world_timestamps_us, reference_sensor
        )

        curve = ps.register_curve_network(
            sequence_meta["sequence_id"],
            world_common_positions,
            "line",
            enabled=True,
            color=((0.25 * i) % 1.0,) * 3,
            radius=0.001,
        )
        curve.add_scalar_quantity("timestamp_us", poses.T_rig_world_timestamps_us)
        curve.add_scalar_quantity(
            "sequence_time_sec", (poses.T_rig_world_timestamps_us - poses.T_rig_world_timestamps_us[0]) / 1e6
        )

        if i == 0:
            # initialize bbox from first session
            state.bbox_rect.from_points(world_common_positions)
            ps.register_curve_network("bbox_rect", state.bbox_rect.points, "loop", enabled=True, radius=0.001)

    state.update_chunks()

    ps.show()


if __name__ == "__main__":
    ncore_sequence_chunks(show_default=True)
