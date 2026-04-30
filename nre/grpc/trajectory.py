# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Provides geometry tools for working with vehicle trajectories, expressed as quaternion + translation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence


try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np

from scipy import spatial
from scipy.spatial.transform import Rotation as R

import nre.grpc.protos.common_pb2 as grpc_types


def assert_is_quat_shape(q: np.ndarray) -> None:
    if q.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {q.shape[-1]}")


def assert_is_vec3_shape(v: np.ndarray) -> None:
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {v.shape[-1]}")


def quat_vec3_multiply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert_is_quat_shape(q)
    assert_is_vec3_shape(v)

    q_ = R.from_quat(q)

    zeros_shape = v.shape[:-1] + (1,)
    zeros = np.zeros(zeros_shape)
    v_as_r = R.from_quat(np.concatenate([v, zeros], axis=-1))

    ret_as_r = q_ * v_as_r * q_.inv()
    return ret_as_r.as_quat()[..., :-1]


def uniform_length_resample(positions: np.ndarray, n_points: int) -> np.ndarray:
    """
    Takes an [T, 3] array of xyz positions and resamples it to `n_points` uniformly
    spaced *along the length of the trajectory*, instead of taking time (speed) into
    consideration.
    """
    # Compute the distances between consecutive points
    delta = positions[1:] - positions[:-1]
    segment_lengths = np.linalg.norm(delta, axis=1)
    # Compute the cumulative length along the trajectory
    cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]
    # Generate equally spaced points along the total length
    desired_lengths = np.linspace(0, total_length, n_points)

    # Handle possible duplicate cumulative lengths (when the vehicle is stationary)
    cumulative_length, unique_indices = np.unique(cumulative_length, return_index=True)
    positions = positions[unique_indices]

    # Interpolate the positions at the desired lengths
    return np.stack(
        [np.interp(x=desired_lengths, xp=cumulative_length, fp=positions[:, axis]) for axis in range(3)],
        axis=-1,
    )


@dataclass(kw_only=True)
class QVec:
    """
    Conventions:
        1. xyzw order on quat
        2. first apply translation, then rotation
    """

    vec3: np.ndarray
    quat: np.ndarray

    def __post_init__(self):
        assert_is_vec3_shape(self.vec3)
        assert_is_quat_shape(self.quat)
        assert self.vec3.shape[:-1] == self.quat.shape[:-1]

    def __repr__(self) -> str:
        return f"QVec(batch_size={self.batch_size})"

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            vec3=np.zeros((0, 3), dtype=np.float32),
            quat=np.zeros((0, 4), dtype=np.float32),
        )

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.vec3.shape[:-1]

    @staticmethod
    def stack(qvecs: Sequence[QVec], axis: int = 0) -> QVec:
        return QVec(
            vec3=np.stack([qvec.vec3 for qvec in qvecs], axis=axis),
            quat=np.stack([qvec.quat for qvec in qvecs], axis=axis),
        )

    def _apply(
        self,
        fn: Callable[
            [
                np.ndarray,
            ],
            np.ndarray,
        ],
    ) -> QVec:
        return QVec(vec3=fn(self.vec3), quat=fn(self.quat))

    def __getitem__(self, index) -> QVec:
        return self._apply(lambda arr: arr[index])

    def __len__(self) -> int:
        return self.batch_size[0]

    def __iter__(self) -> Iterator[QVec]:
        for i in range(len(self)):
            yield self[i]

    def __matmul__(self, other: QVec) -> QVec:
        R_self = R.from_quat(self.quat)
        if isinstance(other, QVec):
            vec3 = (R_self.as_matrix() @ other.vec3[..., None]).squeeze(-1) + self.vec3
            quat = (R_self * R.from_quat(other.quat)).as_quat()
            return QVec(vec3=vec3, quat=quat)
        elif isinstance(other, DynamicState):
            linear_velocity = (R_self.as_matrix() @ other.linear_velocity[..., None]).squeeze(-1)
            angular_velocity = (R_self.as_matrix() @ other.angular_velocity[..., None]).squeeze(-1)
            return DynamicState(angular_velocity=angular_velocity, linear_velocity=linear_velocity)

    def inverse(self) -> QVec:
        R_inv = R.from_quat(self.quat).inv()
        return QVec(
            vec3=-(R_inv.as_matrix() @ self.vec3),
            quat=R_inv.as_quat(),
        )

    def as_se3(self) -> np.ndarray:
        m = np.zeros((*self.batch_size, 4, 4))
        m[..., 3, 3] = 1

        m[..., :3, :3] = R.from_quat(self.quat).as_matrix()
        m[..., :3, 3] = self.vec3[..., :3]

        return m

    @staticmethod
    def from_se3(se3_mat: np.ndarray) -> QVec:
        quat = R.from_matrix(se3_mat[..., :3, :3]).as_quat()
        vec3 = se3_mat[..., :3, 3]
        return QVec(vec3=vec3, quat=quat)

    def append(self, other: QVec) -> QVec:
        assert len(self.batch_size) == 1
        assert other.batch_size == (), f"Can only append an unbatched QVec (got {other.batch_size})"

        other = other[None, :]  # add batch dimension

        return QVec(
            vec3=np.concatenate([self.vec3, other.vec3], axis=0),
            quat=np.concatenate([self.quat, other.quat], axis=0),
        )

    def as_grpc_pose(self) -> grpc_types.Pose:
        assert self.batch_size == ()
        return grpc_types.Pose(
            vec=grpc_types.Vec3(
                x=self.vec3[0],
                y=self.vec3[1],
                z=self.vec3[2],
            ),
            quat=grpc_types.Quat(
                x=self.quat[0],
                y=self.quat[1],
                z=self.quat[2],
                w=self.quat[3],
            ),
        )

    def to_grpc_pose_at_time(self, timestamp_us: int) -> grpc_types.PoseAtTime:
        pose_at_time = grpc_types.PoseAtTime(
            pose=self.as_grpc_pose(),
            timestamp_us=timestamp_us,
        )
        return pose_at_time

    def as_grpc_poses(self) -> Sequence[grpc_types.Pose]:
        assert len(self.batch_size) == 1
        return [qvec.as_grpc_pose() for qvec in self]

    @staticmethod
    def from_grpc_pose(grpc_pose: grpc_types.Pose) -> QVec:
        return QVec(
            vec3=np.array([getattr(grpc_pose.vec, dim) for dim in "xyz"]),
            quat=np.array([getattr(grpc_pose.quat, dim) for dim in "xyzw"]),
        )


@dataclass
class DynamicState:
    angular_velocity: np.ndarray
    linear_velocity: np.ndarray

    def __post_init__(self):
        assert_is_vec3_shape(self.angular_velocity)
        assert_is_vec3_shape(self.linear_velocity)
        assert self.angular_velocity.shape[:-1] == self.linear_velocity.shape[:-1]

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            angular_velocity=np.zeros((0, 3), dtype=np.float32),
            linear_velocity=np.zeros((0, 3), dtype=np.float32),
        )

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.linear_velocity.shape[:-1]

    @staticmethod
    def stack(states: Sequence[DynamicState], axis: int = 0) -> DynamicState:
        return DynamicState(
            angular_velocity=np.stack([state.angular_velocity for state in states], axis=axis),
            linear_velocity=np.stack([state.linear_velocity for state in states], axis=axis),
        )

    def _apply(
        self,
        fn: Callable[
            [
                np.ndarray,
            ],
            np.ndarray,
        ],
    ) -> DynamicState:
        return DynamicState(
            angular_velocity=fn(self.angular_velocity),
            linear_velocity=fn(self.linear_velocity),
        )

    def __getitem__(self, index) -> DynamicState:
        return self._apply(lambda arr: arr[index])

    def __len__(self) -> int:
        return self.batch_size[0]

    def __iter__(self) -> Iterator[DynamicState]:
        for i in range(len(self)):
            yield self[i]

    def append(self, other: DynamicState) -> DynamicState:
        if len(self.batch_size) != 1:
            raise ValueError("Can only append to a single state")
        if other.batch_size != ():
            raise ValueError("Can only append a single state")

        other = other[None, :]  # add batch dimension

        return DynamicState(
            angular_velocity=np.concatenate([self.angular_velocity, other.angular_velocity], axis=0),
            linear_velocity=np.concatenate([self.linear_velocity, other.linear_velocity], axis=0),
        )

    @staticmethod
    def from_grpc_state(grpc_state: grpc_types.DynamicState) -> DynamicState:
        return DynamicState(
            angular_velocity=np.array([getattr(grpc_state.angular_velocity, dim) for dim in "xyz"]),
            linear_velocity=np.array([getattr(grpc_state.linear_velocity, dim) for dim in "xyz"]),
        )

    def as_grpc_state(self) -> grpc_types.DynamicState:
        if self.batch_size != ():
            raise ValueError("Can only convert a single state to a grpc state")
        return grpc_types.DynamicState(
            angular_velocity=grpc_types.Vec3(
                x=self.angular_velocity[0],
                y=self.angular_velocity[1],
                z=self.angular_velocity[2],
            ),
            linear_velocity=grpc_types.Vec3(
                x=self.linear_velocity[0],
                y=self.linear_velocity[1],
                z=self.linear_velocity[2],
            ),
        )

    def to_grpc_pose_at_time(self, timestamp_us: int, qvec: QVec) -> grpc_types.PoseAtTime:
        return qvec.to_grpc_pose_at_time(timestamp_us)

    def as_grpc_states(self) -> Sequence[grpc_types.DynamicState]:
        assert len(self.batch_size) == 1
        return [state.as_grpc_state() for state in self]


@dataclass
class Trajectory:
    timestamps_us: np.ndarray
    poses: QVec

    def __post_init__(self):
        assert self.timestamps_us.ndim == 1
        assert self.timestamps_us.dtype == np.uint64
        assert self.poses.batch_size == self.timestamps_us.shape

        # check strict monotonicity
        delta = self.timestamps_us[1:] - self.timestamps_us[:-1]
        assert (delta > 0).all()

    def __len__(self) -> int:
        return self.timestamps_us.shape[0]

    def is_empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            timestamps_us=np.array([], dtype=np.uint64),
            poses=QVec.create_empty(),
        )

    def __repr__(self) -> str:
        return f"Trajectory(n_poses={self.timestamps_us.shape[0]}, time_range_us={self.time_range_us})"

    def transform(self, transform: QVec) -> Trajectory:
        if self.is_empty():
            return self
        return Trajectory(self.timestamps_us, transform[None, :] @ self.poses)

    def clip(self, start_us: int, end_us: int) -> Trajectory:
        """
        Subselect the portion of `self` which is between `start_us` and `end_us` (exclusive).
        Returns an empty trajectory if that selection is out of bounds of `self`.
        """
        assert start_us <= end_us
        if start_us == end_us or self.time_range_us.start > end_us or self.time_range_us.stop < start_us:
            return Trajectory.create_empty()

        # clamp the input time range to `self.time_range_us`
        start_us = max(start_us, self.time_range_us.start)
        last_timestamp_us = min(end_us, self.time_range_us.stop) - 1

        # interpolate the start and end poses, retain the poses in between
        first_pose, last_pose = self.interpolate_to_timestamps(
            np.array([start_us, last_timestamp_us], dtype=np.uint64)
        ).poses
        is_between_start_and_end = (self.timestamps_us > start_us) & (self.timestamps_us < last_timestamp_us)
        if start_us == last_timestamp_us:
            poses = [first_pose]
            timestamps_us = [start_us]
        else:
            poses = [first_pose, *list(self.poses[is_between_start_and_end]), last_pose]
            timestamps_us = [
                start_us,
                *self.timestamps_us[is_between_start_and_end],
                last_timestamp_us,
            ]

        return Trajectory(
            timestamps_us=np.array(timestamps_us, dtype=np.uint64),
            poses=QVec.stack(poses),
        )

    @property
    def time_range_us(self) -> range:
        if self.is_empty():
            return range(0, 0)

        return range(int(self.timestamps_us[0]), int(self.timestamps_us[-1]) + 1)

    @property
    def last_pose(self) -> QVec:
        return self.poses[-1]

    def update_absolute(self, timestamp: int, pose: QVec) -> None:
        assert timestamp > self.time_range_us.stop
        self.timestamps_us = np.concatenate([self.timestamps_us, np.array([timestamp], dtype=np.uint64)], axis=0)
        self.poses = self.poses.append(pose)

    def update_relative(self, timestamp: int, pose_delta: QVec) -> None:
        assert timestamp > self.time_range_us.stop
        self.update_absolute(timestamp, self.poses[-1] @ pose_delta)

    def interpolate_to_timestamps(self, ts_target: np.ndarray) -> Trajectory:
        if ts_target.dtype != np.uint64:
            raise TypeError(f"Expected np.uint64 got {ts_target.dtype=}.")

        if self.is_empty():
            raise ValueError("Trying to interpolate on an empty trajectory.")

        is_in_range = (ts_target >= self.time_range_us.start) & (ts_target < self.time_range_us.stop)
        if not is_in_range.all():
            raise ValueError(f"Interpolate @ {ts_target[~is_in_range]} outside of {self.time_range_us}.")

        if self.timestamps_us.shape == (1,):
            # Slerp will fail with a single pose. Since we already checked that all queries are
            # in range, we can just return that single pose replicated for each query
            poses = QVec.stack([self.last_pose] * ts_target.shape[0])
        else:
            slerp = spatial.transform.Slerp(self.timestamps_us, R.from_quat(self.poses.quat))(ts_target).as_quat()

            lerp = np.stack(
                [np.interp(ts_target, self.timestamps_us, vec_dim) for vec_dim in self.poses.vec3.T],
                axis=1,
            )

            poses = QVec(
                vec3=lerp,
                quat=slerp,
            )

        return Trajectory(timestamps_us=ts_target, poses=poses)

    def interpolate_pose(self, at_us: int) -> QVec:
        return self.interpolate_to_timestamps(np.array([at_us], dtype=np.uint64)).poses[0]

    def interpolate_delta(self, start_us: int, end_us: int) -> QVec:
        interp = self.interpolate_to_timestamps(np.array([start_us, end_us], dtype=np.uint64))
        return interp.poses[0].inverse() @ interp.poses[1]

    @staticmethod
    def from_grpc(trajectory: grpc_types.Trajectory) -> Trajectory:
        if len(trajectory.poses) == 0:
            return Trajectory.create_empty()

        timestamps_us = np.array([p.timestamp_us for p in trajectory.poses], dtype=np.uint64)
        poses = QVec.stack([QVec.from_grpc_pose(p.pose) for p in trajectory.poses], axis=0)
        return Trajectory(timestamps_us=timestamps_us, poses=poses)

    def to_grpc(self) -> grpc_types.Trajectory:
        poses = self.poses.as_grpc_poses()
        return grpc_types.Trajectory(
            poses=[
                grpc_types.PoseAtTime(
                    timestamp_us=ts,
                    pose=pose,
                )
                for ts, pose in zip(
                    self.timestamps_us,
                    poses,
                )
            ]
        )
