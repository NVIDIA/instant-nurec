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

import asyncio
import logging
import os
import time

from typing import Generator, List

import click
import grpc
import imageio
import numpy as np

from nre.artifact import Artifact
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.grpc.protos.common_pb2 import Empty as EmptyRequest
from nre.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    CameraSpec,
    ImageFormat,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
from nre.grpc.serve import grpc_pose_to_tquat
from nre.render.novel_trajectory.controllers import (
    ActorController,
    EgoController,
    ReplayActorController,
    ReplayEgoController,
    SpinningActorController,
    SpiralEgoController,
    StopActorController,
    StopEgoController,
    Timestamp,
)
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.io.export.render_grpc import generate_request, save_request
from nre.utils.misc import to_numpy
from nre.utils.types import RigTrajectories


logger = logging.getLogger(__file__)


class Segment:
    """A segment of the novel trajectory with decoupled ego and actor controls"""

    def __init__(
        self,
        camera: CameraSpec,
        scene_id: str,
        height: int,
        ego_controller: EgoController,
        actor_controller: ActorController,
        *args,
        **kwargs,
    ) -> None:
        self.camera = camera
        self.scene_id = scene_id
        self.height = height
        self.ego_controller = ego_controller
        self.actor_controller = actor_controller

    def timestamp_generator(self) -> Generator[tuple[Timestamp, Timestamp], None, None]:
        """Generate timestamps for the segment (uses ego controller if available)"""
        yield from zip(self.ego_controller.timestamp_generator(), self.actor_controller.timestamp_generator())

    def generate_requests(self) -> Generator[RGBRenderRequest, None, None]:
        """Generate requests for the segment with decoupled ego and actor controls"""
        camera_to_rig = to_numpy(
            tquat_to_se3_matrix(grpc_pose_to_tquat(self.camera.rig_to_camera))
        )  # NB this is a known naming bug in grpc, rig_to_camera should be camera_to_rig

        # Generate timestamps from ego controller, with actors potentially on different timeline
        for frame_index, (ego_timestamp_us, actor_timestamp_us) in enumerate(self.timestamp_generator()):
            # Get ego pose
            rig_to_world = self.ego_controller.rig_pose_interpolator(ego_timestamp_us)
            assert rig_to_world.shape == (2, 4, 4)

            # Apply the transformation offset to the camera in the rig frame.
            rig_offset_se3 = self.ego_controller.rig_offset_se3(frame_index, ego_timestamp_us)
            frame_start_camera_to_world_se3 = rig_to_world[0] @ rig_offset_se3 @ camera_to_rig
            frame_end_camera_to_world_se3 = rig_to_world[1] @ rig_offset_se3 @ camera_to_rig

            # Get actor timestamp (could be different from ego timestamp)
            dynamic_actors = self.actor_controller.dynamic_actors_interpolator(frame_index, actor_timestamp_us)

            yield generate_request(
                self.scene_id,
                self.camera.intrinsics,
                frame_start_camera_to_world_se3,
                frame_end_camera_to_world_se3,
                ego_timestamp_us[0],
                ego_timestamp_us[1],
                self.height,
                dynamic_actors,
                ImageFormat.JPEG,
                None,
            )


class NovelTrajectoryRenderer:
    """
    A class to simplify the process of rendering novel trajectories.

    The key idea behind the design is that the renderer is a state machine that can be controlled by the user.
    The user can add segments to the renderer, and the renderer will render the segments in the order they were added.
    Each segment includes ego and actor controllers to create more complex trajectories.

    There are a few built-in segments that can be added to the renderer:
        - replay: Replay the trajectory.
        - spinning_actors: Spin the actors.
        - spiral: Spiral the trajectory.
        - stop_ego: Stop the ego.
        - stop_actors: Stop the actors.

    ie. renderer.replay(duration_s=2.0).stop_actors(2.0).spiral(duration_s=2.0).stop_ego(duration_s=2.0).replay()

    corresponds to the following trajectory:
        - replay for 2 seconds
        - stop the actors for 2 seconds
        - spiral for 2 seconds
        - stop the ego for 2 seconds
        - replay for the rest of the trajectory

    to add a custom segment, the user can create a Segment object and add it to the renderer.
    ie.     renderer.add_segment(
        Segment(
            renderer.camera,
            renderer.scene_id,
            renderer.height,
            StopEgoController(
                renderer.trajectory, renderer.unique_camera_id, duration_s, renderer._current_ego_timestamp_us
            ),
            StopActorController(
                renderer.cuboid_tracks,
                duration_s,
                renderer._current_actor_timestamp_us,
                renderer.trajectory.cameras_frame_timestamps_us[renderer.unique_camera_id].cpu().numpy()
            ),
        )
    )

    finally, the user can render the trajectory by calling renderer.render().

    See the code below in render_novel_trajectory for an example.

    Args:
        artifact_path: The path to the artifact.
        output_dir: The directory to save the rendered images.
        host: The host of the server.
    """

    def __init__(
        self,
        artifact_path: str,
        output_dir: str,
        host: str,
        port: int,
        height: int,
        camera_id: str,
        shutdown_server_on_completion: bool,
        store_video: bool,
    ) -> None:
        self.store_video = store_video

        artifacts_list = Artifact.discover_from_glob(artifact_path)
        assert len(artifacts_list) == 1
        artifact = artifacts_list[0]

        # Get test trajectory and poses
        self.scene_id = artifact.scene_id
        rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

        self.unique_camera_id = camera_id + f"@{self.scene_id}"

        # Assume we only have a single trajectory
        assert len(rig_trajectories.rig_trajectories) == 1, (
            f"_render_grpc: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
        )
        self.trajectory = rig_trajectories.rig_trajectories[0]

        sequence_tracks = {k: CuboidTracks.from_dict(v) for k, v in artifact.sequence_tracks.items()}
        assert isinstance(sequence_tracks, dict) and len(sequence_tracks.values()) == 1
        cuboid_tracks = next(iter(sequence_tracks.values()))  # Assumes only one sequence
        self.cuboid_tracks = CuboidTracks.Ops.subset_from_mask(
            cuboid_tracks, cuboid_tracks.get_mask_flags_all(TrackFlags.CONTROLLABLE)
        )

        self.output_dir = output_dir
        self.host = host
        self.port = port
        self.height = height
        self.camera_id = camera_id
        self.shutdown_server_on_completion = shutdown_server_on_completion

        self._current_ego_timestamp_us = (
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy()[0, 0]
        )
        self._current_actor_timestamp_us = (
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy()[0, 0]
        )
        self.segments: List[Segment] = []

        self.initialize_camera()

    def add_segment(self, segment: Segment) -> None:
        self.segments.append(segment)
        self._current_ego_timestamp_us = segment.ego_controller.last_timestamp_us()
        self._current_actor_timestamp_us = segment.actor_controller.last_timestamp_us()

    async def _initialize_camera(self) -> None:
        async with grpc.aio.insecure_channel(f"{self.host}:{self.port}") as channel:
            client_service = SensorsimServiceStub(channel)
            available_cameras: AvailableCamerasReturn = await client_service.get_available_cameras(
                AvailableCamerasRequest(scene_id=self.scene_id)
            )
            assert self.camera_id in [
                available_camera.logical_id for available_camera in available_cameras.available_cameras
            ]
            for available_camera in available_cameras.available_cameras:
                if available_camera.logical_id == self.camera_id:
                    self.camera = available_camera

    def initialize_camera(self) -> None:
        asyncio.run(self._initialize_camera())

    async def _render(self) -> None:
        async with grpc.aio.insecure_channel(f"{self.host}:{self.port}") as channel:
            client_service = SensorsimServiceStub(channel)

            requests: list[RGBRenderRequest] = []
            for segment in self.segments:
                for request in segment.generate_requests():
                    requests.append(request)

            responses: list[RGBRenderReturn] = []
            warmup_request, *requests = requests
            responses.append(await client_service.render_rgb(warmup_request))

            # time the batch execution time
            tic = time.perf_counter()
            responses.extend(await asyncio.gather(*[client_service.render_rgb(request) for request in requests]))
            toc = time.perf_counter()
            elapsed_s = toc - tic
            n_requests = len(requests)
            s_per_request = elapsed_s / n_requests
            logger.info(f"Served {n_requests=} in {elapsed_s=}, {s_per_request=}.")

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)

                if not self.store_video:
                    await asyncio.gather(
                        *[
                            # Frame index in the filename is contiguous to match the output file names of the validation step.
                            save_request(f"{self.output_dir}/{contiguous_frame_idx:06d}.jpeg", response)
                            for contiguous_frame_idx, response in enumerate(responses)
                        ]
                    )
                else:
                    video_path = os.path.join(self.output_dir, f"render_novel_trajectory_{self.scene_id}.mp4")
                    # Convert bytes responses to images asynchronously
                    raw_images = await asyncio.gather(
                        *[asyncio.to_thread(imageio.imread, response.image_bytes) for response in responses]
                    )
                    # Convert images to RGB format for video creation
                    rgb_images = []
                    for img in raw_images:
                        if len(img.shape) == 2:  # Grayscale
                            img = np.stack([img, img, img], axis=-1)
                        elif len(img.shape) == 3 and img.shape[-1] == 1:  # Single channel
                            img = np.concatenate([img, img, img], axis=-1)
                        elif len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                            img = img[:, :, :3]  # Remove alpha
                        rgb_images.append(img)
                    imageio.mimsave(video_path, rgb_images, fps=30)
                    logger.info(f"Stored video to {video_path}")

            if self.shutdown_server_on_completion:
                await client_service.shut_down(EmptyRequest())

    def render(self) -> None:
        asyncio.run(self._render())

    def replay(self, duration_s: float | None = None) -> NovelTrajectoryRenderer:
        ego_controller = ReplayEgoController(
            self.trajectory, self.unique_camera_id, duration_s, self._current_ego_timestamp_us
        )
        actor_controller = ReplayActorController(
            self.cuboid_tracks,
            duration_s,
            self._current_actor_timestamp_us,
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy(),
        )
        self.add_segment(
            Segment(
                self.camera,
                self.scene_id,
                self.height,
                ego_controller,
                actor_controller,
            )
        )
        return self

    def spiral(
        self,
        duration_s: float,
        spiral_num_spirals: int = 4,
        spiral_radius: float = 0.5,
        spiral_x_scale: float = 0.0,
        spiral_y_scale: float = 0.3,
        spiral_z_scale: float = 0.3,
        framerate: float = 30.0,
        stop_on_start_pose: bool = True,
    ) -> NovelTrajectoryRenderer:
        ego_controller = SpiralEgoController(
            self.trajectory,
            self.unique_camera_id,
            duration_s,
            self._current_ego_timestamp_us,
            spiral_num_spirals,
            spiral_radius,
            spiral_x_scale,
            spiral_y_scale,
            spiral_z_scale,
            framerate,
            stop_on_start_pose,
        )
        actor_controller = ReplayActorController(
            self.cuboid_tracks,
            duration_s,
            self._current_actor_timestamp_us,
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy(),
        )
        self.add_segment(
            Segment(
                self.camera,
                self.scene_id,
                self.height,
                ego_controller,
                actor_controller,
            )
        )
        return self

    def stop_ego(self, duration_s: float) -> NovelTrajectoryRenderer:
        ego_controller = StopEgoController(
            self.trajectory, self.unique_camera_id, duration_s, self._current_ego_timestamp_us
        )
        actor_controller = ReplayActorController(
            self.cuboid_tracks,
            duration_s,
            self._current_actor_timestamp_us,
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy(),
        )
        self.add_segment(
            Segment(
                self.camera,
                self.scene_id,
                self.height,
                ego_controller,
                actor_controller,
            )
        )
        return self

    def stop_actors(self, duration_s: float, framerate: float = 30.0) -> NovelTrajectoryRenderer:
        """Stop actors while continuing ego motion"""
        ego_controller = ReplayEgoController(
            self.trajectory, self.unique_camera_id, duration_s, self._current_ego_timestamp_us
        )
        actor_controller = StopActorController(
            self.cuboid_tracks,
            duration_s,
            self._current_actor_timestamp_us,
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy(),
            framerate,
        )

        self.add_segment(
            Segment(
                self.camera,
                self.scene_id,
                self.height,
                ego_controller,
                actor_controller,
            )
        )
        return self

    def spinning_actors(
        self,
        duration_s: float,
        framerate: float = 30.0,
        n_spins: int = 2,
    ) -> NovelTrajectoryRenderer:
        actor_controller = SpinningActorController(
            self.cuboid_tracks,
            duration_s,
            self._current_actor_timestamp_us,
            self.trajectory.cameras_frame_timestamps_us[self.unique_camera_id].cpu().numpy(),
            framerate,
            n_spins,
        )
        ego_controller = ReplayEgoController(
            self.trajectory, self.unique_camera_id, duration_s, self._current_ego_timestamp_us
        )
        self.add_segment(
            Segment(
                self.camera,
                self.scene_id,
                self.height,
                ego_controller,
                actor_controller,
            )
        )
        return self


@click.command("render-novel-trajectory")
@click.option("--artifact-path", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--host", type=str, required=True, default="localhost")
@click.option("--port", type=int, required=True, default=8080)
@click.option("--height", type=int, required=True, default=300)
@click.option("--camera-id", type=str, required=True, default="camera_front_wide_120fov")
@click.option("--shutdown-server-on-completion", type=bool, required=True, default=False)
@click.option(
    "--store-video/--no-store-video",
    is_flag=True,
    help="Store the video of the rendered images",
    default=True,
)
def render_novel_trajectory(
    artifact_path: str,
    output_dir: str,
    host: str,
    port: int,
    height: int,
    camera_id: str,
    shutdown_server_on_completion: bool,
    store_video: bool,
) -> None:
    # initialize the renderer
    renderer = NovelTrajectoryRenderer(
        artifact_path, output_dir, host, port, height, camera_id, shutdown_server_on_completion, store_video
    )
    # add some default segments
    renderer.replay(duration_s=2.0).stop_actors(2.0).spinning_actors(2.0).spiral(duration_s=2.0).stop_ego(
        duration_s=2.0
    )

    # add a custom segment
    duration_s = 1.0
    renderer.add_segment(
        Segment(
            renderer.camera,
            renderer.scene_id,
            renderer.height,
            StopEgoController(
                renderer.trajectory, renderer.unique_camera_id, duration_s, renderer._current_ego_timestamp_us
            ),
            StopActorController(
                renderer.cuboid_tracks,
                duration_s,
                renderer._current_actor_timestamp_us,
                renderer.trajectory.cameras_frame_timestamps_us[renderer.unique_camera_id].cpu().numpy(),
            ),
        )
    )

    # replay until the end
    renderer.replay()

    # render
    renderer.render()


if __name__ == "__main__":
    render_novel_trajectory()
