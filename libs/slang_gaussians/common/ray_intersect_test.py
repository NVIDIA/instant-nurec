# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import random
import tempfile
import time
import unittest

from collections import namedtuple

import slangtorch
import torch

from libs.slang_utils.utils import (
    add_ninja_to_path,
    enable_torch_nvtx,
    profile,
)


random.seed(123)

device = torch.device("cuda")


KERNEL_PRELUDE = """
#include "ray_intersect.slang"
"""

KERNEL_TEMPLATE = """
[CUDAKernel]
[AutoPyBindCUDA]
void {kernel_name}(
    TensorView<int64_t> output_tracks_timestamps_sum_and_count,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    no_diff int64_t frame_timestamp,
) {{
    typedef ray_intersect::{intersection_strategy_type}IntersectionStrategy IntersectionStrategy;
    typedef ray_intersect::timestamps_estimation::{intersection_handler_type}IntersectionHandler IntersectionHandler;
    ray_intersect::intersect_rays_with_timed_cuboids<IntersectionStrategy, IntersectionHandler>(
        rays,
        rays_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        cuboids_dimensions,
        IntersectionHandler(output_tracks_timestamps_sum_and_count, frame_timestamp),
    );
}}
"""


# This class performs a couple of sanity checks on the ray intersection kernel.
class TestSlangRayIntersect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kernels_arrays_to_generate = [
            ("test_quat_slerp_matrix_atomic_add", "AtomicAdd", "QuatSlerpMatrix"),
            ("test_quat_slerp_matrix_wave_reduce", "WaveReduce", "QuatSlerpMatrix"),
            ("test_quat_slerp_matrix_shared_accumulators", "SharedAccumulators", "QuatSlerpMatrix"),
            (
                "test_quat_slerp_matrix_wave_reduce_shared_accumulators",
                "WaveReduceSharedAccumulators",
                "QuatSlerpMatrix",
            ),
            ("test_lie_atomic_add", "AtomicAdd", "Lie"),
            ("test_lie_wave_reduce", "WaveReduce", "Lie"),
            ("test_lie_shared_accumulators", "SharedAccumulators", "Lie"),
            ("test_lie_wave_reduce_shared_accumulators", "WaveReduceSharedAccumulators", "Lie"),
        ]

        code = KERNEL_PRELUDE
        for kernel_name, intersection_handler_type, intersection_strategy_type in kernels_arrays_to_generate:
            code += KERNEL_TEMPLATE.format(
                kernel_name=kernel_name,
                intersection_handler_type=intersection_handler_type,
                intersection_strategy_type=intersection_strategy_type,
            )
        with tempfile.NamedTemporaryFile("w", suffix=".slang", delete=False) as tf:
            slang_file_path = tf.name
            tf.write(code)
        try:
            add_ninja_to_path()
            # Include paths for specific dependencies
            from python.runfiles import runfiles

            r = runfiles.Create()

            current_dir = os.path.dirname(__file__)
            # Point to libs/ directory to resolve full paths like slang_utils/extensions.slang
            libs_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            include_paths = [current_dir, libs_dir]
            slang_module = slangtorch.loadModule(slang_file_path, verbose=False, includePaths=include_paths)
        finally:
            os.unlink(slang_file_path)

        cls.slang_module = slang_module
        cls.kernels = {
            kernel_name: getattr(slang_module, kernel_name) for kernel_name, _, _ in kernels_arrays_to_generate
        }

    def test_basic_ray_intersection(self):
        TestRays = namedtuple("TestRays", ["rays", "expected_sum", "expected_count"])

        tests = [
            TestRays(rays=[(123, 0)], expected_sum=123, expected_count=1),
            TestRays(rays=[(123, 0.49)], expected_sum=123, expected_count=1),
            TestRays(rays=[(123, 0.51)], expected_sum=0, expected_count=0),
            TestRays(rays=[(123, -0.49)], expected_sum=123, expected_count=1),
            TestRays(rays=[(123, -0.51)], expected_sum=0, expected_count=0),
            TestRays(rays=[(1123, 0)], expected_sum=0, expected_count=0),
            TestRays(rays=[(123, 0), (456, 0.49), (789, -0.51)], expected_sum=579, expected_count=2),
        ]

        NB_TRACKS = 10
        for kernel_name, kernel in self.kernels.items():
            # Move the ray origin.
            for test in tests:
                # Ray origin is (x, 0, 0) and direction is (0, 0, 1)
                rays = torch.tensor([(x, 0, 0, 0, 0, 1) for _, x in test.rays], dtype=torch.float32, device=device)
                rays_timestamps = torch.tensor(
                    [timestamp for timestamp, _ in test.rays], dtype=torch.int64, device=device
                )
                # Cube centered at (0,0,2) with no rotation, size 1.0)
                tracks_poses = torch.tensor([(0, 0, 2, 0, 0, 0, 1)] * 2 * NB_TRACKS, dtype=torch.float32, device=device)
                tracks_timestamps = torch.tensor([0, 1000] * NB_TRACKS, dtype=torch.int64, device=device)
                tracks_packinfo = torch.tensor([(0, 2)] * NB_TRACKS, dtype=torch.int32, device=device)
                cuboids_dimensions = torch.tensor([[1.0, 1.0, 1.0]] * NB_TRACKS, dtype=torch.float32, device=device)

                output_tracks_timestamps_sum_and_count = torch.zeros((NB_TRACKS, 2), dtype=torch.int64, device=device)

                nb_rays = rays.size(0)
                num_threads = 256
                num_blocks_x = (nb_rays + 256 - 1) // 256
                num_blocks_y = NB_TRACKS

                output_tracks_timestamps_sum_and_count = output_tracks_timestamps_sum_and_count.contiguous()
                rays = rays.contiguous()
                rays_timestamps = rays_timestamps.contiguous()
                tracks_poses = tracks_poses.contiguous()
                tracks_timestamps = tracks_timestamps.contiguous()
                tracks_packinfo = tracks_packinfo.contiguous()
                cuboids_dimensions = cuboids_dimensions.contiguous()

                kernel(
                    output_tracks_timestamps_sum_and_count=output_tracks_timestamps_sum_and_count,
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frame_timestamp=0,
                ).launchRaw(
                    (num_threads, 1, 1),
                    (num_blocks_x, num_blocks_y, 1),
                )

                expected_output_tracks_timestamps_sum_and_count = torch.tensor(
                    [(test.expected_sum, test.expected_count)] * NB_TRACKS, dtype=torch.int64, device=device
                )
                self.assertTrue(
                    torch.equal(output_tracks_timestamps_sum_and_count, expected_output_tracks_timestamps_sum_and_count)
                )

            # Move the ray direction.
            for test in tests:
                # Ray origin is (0, 0, 0) and direction is (x, 0, 1.5)
                rays = torch.tensor([(0, 0, 0, x, 0, 1.5) for _, x in test.rays], dtype=torch.float32, device=device)
                rays_timestamps = torch.tensor(
                    [timestamp for timestamp, _ in test.rays], dtype=torch.int64, device=device
                )
                # Cube centered at (0,0,2) with no rotation, size 1.0)
                tracks_poses = torch.tensor([(0, 0, 2, 0, 0, 0, 1)] * 2 * NB_TRACKS, dtype=torch.float32, device=device)
                tracks_timestamps = torch.tensor([0, 1000] * NB_TRACKS, dtype=torch.int64, device=device)
                tracks_packinfo = torch.tensor([(0, 2)] * NB_TRACKS, dtype=torch.int32, device=device)
                cuboids_dimensions = torch.tensor([[1.0, 1.0, 1.0]] * NB_TRACKS, dtype=torch.float32, device=device)

                output_tracks_timestamps_sum_and_count = torch.zeros((NB_TRACKS, 2), dtype=torch.int64, device=device)

                nb_rays = rays.size(0)
                num_threads = 256
                num_blocks_x = (nb_rays + 256 - 1) // 256
                num_blocks_y = NB_TRACKS

                output_tracks_timestamps_sum_and_count = output_tracks_timestamps_sum_and_count.contiguous()
                rays = rays.contiguous()
                rays_timestamps = rays_timestamps.contiguous()
                tracks_poses = tracks_poses.contiguous()
                tracks_timestamps = tracks_timestamps.contiguous()
                tracks_packinfo = tracks_packinfo.contiguous()
                cuboids_dimensions = cuboids_dimensions.contiguous()

                kernel(
                    output_tracks_timestamps_sum_and_count=output_tracks_timestamps_sum_and_count,
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frame_timestamp=0,
                ).launchRaw(
                    (num_threads, 1, 1),
                    (num_blocks_x, num_blocks_y, 1),
                )

                expected_output_tracks_timestamps_sum_and_count = torch.tensor(
                    [(test.expected_sum, test.expected_count)] * NB_TRACKS, dtype=torch.int64, device=device
                )
                self.assertTrue(
                    torch.equal(output_tracks_timestamps_sum_and_count, expected_output_tracks_timestamps_sum_and_count)
                )

            # Move the cuboid position.
            for test in tests:
                if len(test.rays) != 1:
                    continue
                x = test.rays[0][1]

                # Ray origin is (0, 0, 0) and direction is (0, 0, 1)
                rays = torch.tensor([(0, 0, 0, 0, 0, 1) for _, _ in test.rays], dtype=torch.float32, device=device)
                rays_timestamps = torch.tensor(
                    [timestamp for timestamp, _ in test.rays], dtype=torch.int64, device=device
                )
                # Cube centered at (x,0,2) with no rotation, size 1.0)
                tracks_poses = torch.tensor([(x, 0, 2, 0, 0, 0, 1)] * 2 * NB_TRACKS, dtype=torch.float32, device=device)
                tracks_timestamps = torch.tensor([0, 1000] * NB_TRACKS, dtype=torch.int64, device=device)
                tracks_packinfo = torch.tensor([(0, 2)] * NB_TRACKS, dtype=torch.int32, device=device)
                cuboids_dimensions = torch.tensor([[1.0, 1.0, 1.0]] * NB_TRACKS, dtype=torch.float32, device=device)

                output_tracks_timestamps_sum_and_count = torch.zeros((NB_TRACKS, 2), dtype=torch.int64, device=device)

                nb_rays = rays.size(0)
                num_threads = 256
                num_blocks_x = (nb_rays + 256 - 1) // 256
                num_blocks_y = NB_TRACKS

                output_tracks_timestamps_sum_and_count = output_tracks_timestamps_sum_and_count.contiguous()
                rays = rays.contiguous()
                rays_timestamps = rays_timestamps.contiguous()
                tracks_poses = tracks_poses.contiguous()
                tracks_timestamps = tracks_timestamps.contiguous()
                tracks_packinfo = tracks_packinfo.contiguous()
                cuboids_dimensions = cuboids_dimensions.contiguous()

                kernel(
                    output_tracks_timestamps_sum_and_count=output_tracks_timestamps_sum_and_count,
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frame_timestamp=0,
                ).launchRaw(
                    (num_threads, 1, 1),
                    (num_blocks_x, num_blocks_y, 1),
                )

                expected_output_tracks_timestamps_sum_and_count = torch.tensor(
                    [(test.expected_sum, test.expected_count)] * NB_TRACKS, dtype=torch.int64, device=device
                )
                self.assertTrue(
                    torch.equal(output_tracks_timestamps_sum_and_count, expected_output_tracks_timestamps_sum_and_count)
                )

    def test_timestamps_overflow(self):
        # Choose a value that would overflow after adding ~1000 timestamps.
        MAX_INT64 = (1 << 63) - 1
        RAY_TIMESTAMP = MAX_INT64 // 1000
        RAY_COUNT = 1000000
        NB_TRACKS = 1

        for kernel_name, kernel in self.kernels.items():
            # Ray origin is (0, 0, 0) and direction is (0, 0, 1)
            rays = torch.tensor([(0, 0, 0, 0, 0, 1)] * RAY_COUNT, dtype=torch.float32, device=device)
            rays_timestamps = torch.tensor([RAY_TIMESTAMP] * RAY_COUNT, dtype=torch.int64, device=device)
            rays_timestamps += torch.arange(RAY_COUNT, dtype=torch.int64, device=device) * 2 - RAY_COUNT

            # Cube centered at (0,0,2) with no rotation, size 1.0)
            tracks_poses = torch.tensor([(0, 0, 2, 0, 0, 0, 1)] * 2 * NB_TRACKS, dtype=torch.float32, device=device)
            tracks_timestamps = torch.tensor(
                [RAY_TIMESTAMP - RAY_COUNT * 2, RAY_TIMESTAMP + RAY_COUNT * 2] * NB_TRACKS,
                dtype=torch.int64,
                device=device,
            )
            tracks_packinfo = torch.tensor([(0, 2)] * NB_TRACKS, dtype=torch.int32, device=device)
            cuboids_dimensions = torch.tensor([[1.0, 1.0, 1.0]] * NB_TRACKS, dtype=torch.float32, device=device)

            output_tracks_timestamps_sum_and_count = torch.zeros((NB_TRACKS, 2), dtype=torch.int64, device=device)

            nb_rays = rays.size(0)
            num_threads = 256
            num_blocks_x = (nb_rays + 256 - 1) // 256
            num_blocks_y = NB_TRACKS

            output_tracks_timestamps_sum_and_count = output_tracks_timestamps_sum_and_count.contiguous()
            rays = rays.contiguous()
            rays_timestamps = rays_timestamps.contiguous()
            tracks_poses = tracks_poses.contiguous()
            tracks_timestamps = tracks_timestamps.contiguous()
            tracks_packinfo = tracks_packinfo.contiguous()
            cuboids_dimensions = cuboids_dimensions.contiguous()

            kernel(
                output_tracks_timestamps_sum_and_count=output_tracks_timestamps_sum_and_count,
                rays=rays,
                rays_timestamps=rays_timestamps,
                tracks_poses=tracks_poses,
                tracks_timestamps=tracks_timestamps,
                tracks_packinfo=tracks_packinfo,
                cuboids_dimensions=cuboids_dimensions,
                frame_timestamp=RAY_TIMESTAMP,
            ).launchRaw(
                (num_threads, 1, 1),
                (num_blocks_x, num_blocks_y, 1),
            )

            # Make sure we have hits.
            self.assertTrue((output_tracks_timestamps_sum_and_count[:, 1] != 0).all())
            # Make sure the average is right.
            average_timestamp = (
                output_tracks_timestamps_sum_and_count[:, 0] / output_tracks_timestamps_sum_and_count[:, 1]
                + RAY_TIMESTAMP
            )
            self.assertTrue((average_timestamp == RAY_TIMESTAMP).all())

    def test_benchmark(self):
        with enable_torch_nvtx():
            # We run 2 versions of the benchmark: one with a more realistic dataset,
            # and one with a stress test for contention where basically all rays intersect
            # with all tracks.
            for with_contention in [False, True]:
                print(f"Running benchmark with contention: {with_contention}")

                NB_WARMUP = 10
                NB_MEASURE = 100
                NB_REPEAT = 2

                # When running with contention, all rays end up end up in a very concentrated area.
                if with_contention:
                    SPACE_SIZE = 0.5
                else:
                    SPACE_SIZE = 10.0

                TIMESTAMP_RANGE = (0, 1000000)

                # Create rays, all centered at (0,0,RAYS_Z) and point towards a grid centered at the origin.
                NB_RAYS_X = 1024
                NB_RAYS_Y = 1024
                RANGE_X = (-SPACE_SIZE, SPACE_SIZE)
                RANGE_Y = (-SPACE_SIZE, SPACE_SIZE)
                RAYS_Z = -10.0

                # Rays all have the same origin.
                origins = torch.tensor([[0, 0, RAYS_Z]], device=device, dtype=torch.float32)
                origins = origins.expand(NB_RAYS_X * NB_RAYS_Y, 3)

                # Rays direction is in a grid centered at the origin.
                x_values = torch.linspace(RANGE_X[0], RANGE_X[1], NB_RAYS_X, device=device, dtype=torch.float32)
                y_values = torch.linspace(RANGE_Y[0], RANGE_Y[1], NB_RAYS_Y, device=device, dtype=torch.float32)
                x, y = torch.meshgrid(x_values, y_values, indexing="ij")
                x = x.flatten()
                y = y.flatten()
                z = torch.tensor([-RAYS_Z], device=device, dtype=torch.float32).expand(NB_RAYS_X * NB_RAYS_Y)

                directions = torch.stack((x, y, z), dim=-1)
                directions = torch.nn.functional.normalize(directions, dim=-1)

                rays = torch.cat([origins, directions], dim=1)

                rays_timestamps = torch.linspace(
                    TIMESTAMP_RANGE[0], TIMESTAMP_RANGE[1], NB_RAYS_X * NB_RAYS_Y, device=device, dtype=torch.int64
                )

                # Create tracks centered at Z=0, parallel to the Y axis.
                NB_TRACKS = 10
                NB_POSES = (50, 150)
                CUBOID_SIZE = 1.0

                tracks_x = torch.linspace(RANGE_X[0], RANGE_X[1], NB_TRACKS, device=device, dtype=torch.float32)

                tracks_poses = torch.empty((0, 7), device=device, dtype=torch.float32)
                tracks_timestamps = torch.empty((0,), device=device, dtype=torch.int64)
                tracks_packinfo = torch.empty((0, 2), device=device, dtype=torch.int32)
                for track_id in range(NB_TRACKS):
                    nb_poses = random.randint(NB_POSES[0], NB_POSES[1])

                    track_x = tracks_x[track_id].expand(nb_poses)
                    track_y = torch.linspace(RANGE_Y[0], RANGE_Y[1], nb_poses, device=device, dtype=torch.float32)
                    track_z = torch.zeros_like(track_y, device=device, dtype=torch.float32)

                    # When running with contention, we just don't rotate the tracks.
                    if with_contention:
                        angles = torch.zeros((nb_poses,), device=device, dtype=torch.float32)
                    else:
                        angles = torch.linspace(0, 2 * torch.pi, nb_poses, device=device, dtype=torch.float32)

                    def compute_quaternion(axis, angle):
                        axis_normalized = axis / axis.norm(dim=1, keepdim=True)
                        half_angle = angle * 0.5
                        sin_half_angle = torch.sin(half_angle)
                        cos_half_angle = torch.cos(half_angle)
                        quat_xyz = axis_normalized * sin_half_angle.unsqueeze(1)
                        quat_w = cos_half_angle.unsqueeze(1)
                        return torch.cat([quat_xyz, quat_w], dim=1)

                    rotation_axis = torch.tensor([0, 1, 0], device=device, dtype=torch.float32).expand(nb_poses, 3)
                    rotation_angles = angles
                    rotations = compute_quaternion(rotation_axis, rotation_angles)
                    track_poses = torch.cat(
                        (track_x.unsqueeze(1), track_y.unsqueeze(1), track_z.unsqueeze(1), rotations), dim=1
                    )
                    track_timestamps = torch.linspace(
                        TIMESTAMP_RANGE[0], TIMESTAMP_RANGE[1], nb_poses, device=device, dtype=torch.int64
                    )
                    start_offset = tracks_poses.shape[0]
                    track_packinfo = torch.tensor([[start_offset, nb_poses]], device=device, dtype=torch.int32)

                    tracks_poses = torch.cat((tracks_poses, track_poses), dim=0)
                    tracks_timestamps = torch.cat((tracks_timestamps, track_timestamps), dim=0)
                    tracks_packinfo = torch.cat((tracks_packinfo, track_packinfo), dim=0)

                cuboids_dimensions = torch.tensor(
                    [CUBOID_SIZE, CUBOID_SIZE, CUBOID_SIZE], device=device, dtype=torch.float32
                ).expand(NB_TRACKS, 3)

                rays = rays.contiguous()
                rays_timestamps = rays_timestamps.contiguous()
                tracks_poses = tracks_poses.contiguous()
                tracks_timestamps = tracks_timestamps.contiguous()
                tracks_packinfo = tracks_packinfo.contiguous()
                cuboids_dimensions = cuboids_dimensions.contiguous()

                def call_kernel(
                    kernel,
                    rays,
                    rays_timestamps,
                    tracks_poses,
                    tracks_timestamps,
                    tracks_packinfo,
                    cuboids_dimensions,
                    frame_timestamp,
                ):
                    nb_tracks = tracks_packinfo.size(0)
                    output_tracks_timestamps_sum_and_count = torch.zeros(
                        (nb_tracks, 2), device=tracks_timestamps.device, dtype=tracks_timestamps.dtype
                    )

                    nb_rays = rays.size(0)

                    def div_up(a, b):
                        return (a + b - 1) // b

                    num_threads = (256, 1, 1)
                    num_blocks = (div_up(nb_rays, 256), nb_tracks, 1)

                    kernel.fn_handle(
                        num_threads,
                        num_blocks,
                        output_tracks_timestamps_sum_and_count,
                        rays,
                        rays_timestamps,
                        tracks_poses,
                        tracks_timestamps,
                        tracks_packinfo,
                        cuboids_dimensions,
                        frame_timestamp,
                    )

                    return output_tracks_timestamps_sum_and_count

                ground_truth = call_kernel(
                    kernel=next(iter(self.kernels.values())),
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frame_timestamp=0,
                )

                implementations = list(self.kernels.items()) * NB_REPEAT
                # random.shuffle(implementations)
                for kernel_name, kernel in implementations:
                    name = kernel_name
                    with profile(name):
                        with profile("Test"):
                            result = call_kernel(
                                kernel=kernel,
                                rays=rays,
                                rays_timestamps=rays_timestamps,
                                tracks_poses=tracks_poses,
                                tracks_timestamps=tracks_timestamps,
                                tracks_packinfo=tracks_packinfo,
                                cuboids_dimensions=cuboids_dimensions,
                                frame_timestamp=0,
                            )
                            assert torch.equal(result[0], ground_truth[0])
                            assert torch.equal(result[1], ground_truth[1])

                        with profile("Warmup"):
                            for i in range(NB_WARMUP):
                                result = call_kernel(
                                    kernel=kernel,
                                    rays=rays,
                                    rays_timestamps=rays_timestamps,
                                    tracks_poses=tracks_poses,
                                    tracks_timestamps=tracks_timestamps,
                                    tracks_packinfo=tracks_packinfo,
                                    cuboids_dimensions=cuboids_dimensions,
                                    frame_timestamp=0,
                                )

                        torch.cuda.synchronize()
                        with profile("Measure time"):
                            start_time = time.perf_counter()
                            for i in range(NB_MEASURE):
                                result = call_kernel(
                                    kernel=kernel,
                                    rays=rays,
                                    rays_timestamps=rays_timestamps,
                                    tracks_poses=tracks_poses,
                                    tracks_timestamps=tracks_timestamps,
                                    tracks_packinfo=tracks_packinfo,
                                    cuboids_dimensions=cuboids_dimensions,
                                    frame_timestamp=0,
                                )
                            torch.cuda.synchronize()
                            end_time = time.perf_counter()
                            print(f"{name:<60}{(end_time - start_time) / NB_MEASURE * 1e3:12.6f} ms")

                # To make it easier to compare implementations next to one another in a profile,
                # we run all implementations together.
                with profile("Run all implementations together"):
                    for i in range(NB_WARMUP):
                        for kernel_name, kernel in implementations:
                            torch.cuda.synchronize()
                            with profile(kernel_name):
                                torch.cuda.synchronize()
                                result = call_kernel(
                                    kernel=kernel,
                                    rays=rays,
                                    rays_timestamps=rays_timestamps,
                                    tracks_poses=tracks_poses,
                                    tracks_timestamps=tracks_timestamps,
                                    tracks_packinfo=tracks_packinfo,
                                    cuboids_dimensions=cuboids_dimensions,
                                    frame_timestamp=0,
                                )
                                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
