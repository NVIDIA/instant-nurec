# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import os
import tempfile
import zipfile

from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    List,
    Literal,
    Optional,
    OrderedDict,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)


try:
    # Python 3.11+ provides dataclass_transform and Self in standard library
    from typing import Self, dataclass_transform
except ImportError:
    # Fall-back to importing dataclass_transform and Self from typing extensions
    # for older toolchains
    from typing_extensions import Self, dataclass_transform

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum, IntFlag, auto
from functools import cached_property

import dataclasses_json
import lietorch as lt
import numpy as np
import numpy.typing as npt
import torch

from omegaconf import DictConfig, OmegaConf
from pxr import Usd

from ncore.data import (
    BBox3,
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
)
from nre.utils.fields import (
    field_camera_model_parameters,
    field_lidar_model_parameters,
    field_numpy_array,
    field_torch_tensor,
)
from nre.utils.misc import (
    assert_same_type,
    collate_fn,
    dataclass_items,
    dataclass_keys,
    flatten_list,
    unpack_optional,
)

M = TypeVar("M", bound=torch.nn.Module)
# ModuleRef is an annotation for a getter of torch Modules. It is used when we want to pass a module
# to another module as a reference, *without adding it to its state dict* and while *reflecting the changes
# to the original module*.
ModuleRef = Callable[[], M]

Checkpoint: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class HalfClosedInterval:
    """Represents a closed interval [start, end)"""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start <= self.end

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.end or other.end <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.end, other.end))

    @property
    def length(self) -> int:
        return self.end - self.start





@runtime_checkable
class Chunkable(Protocol):
    """Marks classes as chunkable"""

    def __getitem__(self: Chunkable, key: Any) -> Any:
        pass


@dataclass_transform(
    # Note that these `*_default` parameters are for static type hinting only, and actual runtime-
    # behavior is adapted in the body below
    eq_default=False,
    kw_only_default=True,
)
def chunkable_dataclass_decorator(cls=None, eq: Literal[False] = False, **kwargs: Any):
    """Custom decorator for all chunkable dataclasses, which have the restriction to not define their own __eq__ functions"""
    assert eq == False, "Chunkable dataclasses are not allowed to define their own __eq__ function"
    kwargs["eq"] = False
    return dataclass(cls, **kwargs)


@dataclass
class TorchChunkable(Chunkable):
    def _getitem_basedict(self, key: torch.Tensor | slice | int) -> dict[str, Any]:
        return {k: v[key] if isinstance(v, (torch.Tensor, TorchChunkable)) else v for k, v in dataclass_items(self)}

    def __getitem__(self, key: torch.Tensor | slice | int) -> Self:
        return type(self)(**self._getitem_basedict(key))

    def __setitem__(self, key: torch.Tensor | slice | int, val: Any) -> None:
        for _, v in dataclass_items(self):
            if isinstance(v, (torch.Tensor, TorchChunkable)):
                v[key] = val

    def __eq__(self, other) -> bool:
        if isinstance(other, TorchChunkable):
            for k, v in dataclass_items(self):
                if isinstance(v, torch.Tensor):
                    if not isinstance(other_tensor := getattr(other, k), torch.Tensor) or not torch.equal(
                        v, other_tensor
                    ):
                        return False
                else:
                    if v != getattr(other, k):
                        return False
            return True

        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @classmethod
    def _collate_fn_basedict(
        cls,
        item_or_seq: Union[TorchChunkable, Sequence[TorchChunkable]],
        device: torch.device = torch.device("cpu"),
        unsqueeze_if_zero_dim: bool = True,
        allow_partial_none: bool = True,
    ) -> dict[str, Any]:
        if isinstance(item_or_seq, TorchChunkable):
            return asdict(item_or_seq)

        assert isinstance(item_or_seq[0], cls), f"{cls.__name__} got invalid item type {type(item_or_seq[0])}"
        assert len(item_or_seq), f"Sequence of {cls.__name__} is empty"
        dict_of_lst = {k: [getattr(v, k) for v in item_or_seq] for k in dataclass_keys(cls)}

        def _collate_vals(vals: list[Any], k: str) -> Any:
            # Flatten the lists in case we are collating a single TorchChunkable with one that was collated before
            vals = flatten_list(vals)
            if (n_nones := sum([v is None for v in vals])) == len(vals):
                return None
            elif n_nones > 0:
                assert allow_partial_none, (
                    "TorchChunkable: set to not allowed: some of the item being None and some not"
                )

            if len(tensor_vals := [v for v in vals if isinstance(v, torch.Tensor)]):
                if unsqueeze_if_zero_dim:
                    tensor_vals = [v.unsqueeze(0) if v.dim() == 0 else v for v in tensor_vals]

                # Assert that all elements are of the same type
                assert_same_type(tensor_vals)

                return torch.cat([v.to(device, non_blocking=True) for v in tensor_vals], dim=0)
            elif len(chunkable_vals := [v for v in vals if isinstance(v, TorchChunkable)]):
                # Assert that all elements are of the same type
                assert_same_type(chunkable_vals)

                return type(chunkable_vals[0]).collate_fn(
                    chunkable_vals, device=device, unsqueeze_if_zero_dim=unsqueeze_if_zero_dim
                )
            else:
                # Assert that all elements are of the same type
                assert_same_type(vals)
                return collate_fn(vals, target_device=device, name_hint=k, return_list_if_unknown=True)

        return {k: _collate_vals(v, k) for k, v in dict_of_lst.items()}

    @classmethod
    def collate_fn(
        cls,
        item_or_seq: Union[TorchChunkable, Sequence[TorchChunkable]],
        device: torch.device,
        unsqueeze_if_zero_dim: bool = True,
    ) -> Self:
        if isinstance(item_or_seq, cls):
            return item_or_seq

        return cls(**cls._collate_fn_basedict(item_or_seq, device=device, unsqueeze_if_zero_dim=unsqueeze_if_zero_dim))

    @classmethod
    def concatenate(cls, seq: Sequence[TorchChunkable], dim: int = 0) -> Self:
        assert isinstance(seq[0], cls), f"{cls.__name__} got invalid item type {type(seq[0])}"
        assert len(seq), f"Sequence of {cls.__name__} is empty"
        dict_of_lst = {k: [getattr(v, k) for v in seq] for k in dataclass_keys(cls)}

        def _concat_vals(vals: list[Any]) -> Any:
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                return torch.cat(vals, dim=dim)
            elif isinstance(v0, TorchChunkable):
                return type(v0).concatenate(vals, dim=dim)
            elif v0 is None:
                return None
            else:
                return vals

        return cls(**{k: _concat_vals(v) for k, v in dict_of_lst.items()})

    @classmethod
    def stack(cls, seq: Sequence[TorchChunkable], dim: int = 0) -> Self:
        assert isinstance(seq[0], cls), f"{cls.__name__} got invalid item type {type(seq[0])}"
        assert len(seq), f"Sequence of {cls.__name__} is empty"
        dict_of_lst = {k: [getattr(v, k) for v in seq] for k in dataclass_keys(cls)}

        def _stack_vals(vals: list[Any]) -> Any:
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                return torch.stack(vals, dim=dim)
            elif isinstance(v0, TorchChunkable):
                return type(v0).stack(vals, dim=dim)
            elif v0 is None:
                return None
            else:
                return vals

        return cls(**{k: _stack_vals(v) for k, v in dict_of_lst.items()})

    def apply(
        self,
        fn: Callable[
            [
                torch.Tensor | TorchChunkable,
            ],
            torch.Tensor | TorchChunkable,
        ],
    ) -> Self:
        return type(self)(
            **{k: fn(v) if isinstance(v, (torch.Tensor, TorchChunkable)) else v for k, v in dataclass_items(self)}
        )

    def to(self, *args, **kwargs) -> Self:
        with torch.cuda.nvtx.range("TorchChunkable_to", color="red"):
            return self.apply(lambda t: t.to(*args, **kwargs))

    def to_device(self, device: torch.device) -> Self:
        return self.to(device)

    def detach(self) -> Self:
        return self.apply(lambda t: t.detach())

    def clone(self) -> Self:
        return self.apply(lambda t: t.clone())

    def tile(self, *dims) -> Self:
        return self.apply(lambda t: t.tile(*dims))

    def take_along_dim(self, index: torch.Tensor, dim: int) -> Self:
        return self.apply(lambda t: t.take_along_dim(index, dim))

    def gather(self, dim: int, index: torch.Tensor) -> Self:
        return self.apply(lambda t: t.gather(dim, index))




class RayFlags(IntFlag):
    """Bitmask flags of per-ray properties (note: limited to 32 variants)"""

    # general ray-associated attributes [non-mutually exclusive]
    RGB_LABEL = auto()  # set if the ray has associated RGB values
    VALID_SEMANTIC = (
        auto()
    )  # set if semantics of the ray can be considered to be valid (even if no specific class is set)
    SKY_SEMANTIC = auto()  # set if the ray is classified to be a sky ray
    ROAD_SEMANTIC = auto()  # set if the ray is classified to be a road ray
    VEHICLE_SEMANTIC = auto()  # set if the ray is classified to be a vehicle ray (non-ego)
    EGO_SEMANTIC = auto()  # set if the ray is classified to the ego car
    DROPPED = auto()  # set if the ray is classified to be a dropped ray
    VALID_NORMAL = auto()  # set if the ray has a valid normal

    INVALID = (
        auto()
    )  # rays can be invalid due to e.g. motion state, mask, ... - all training rays are usually considered to be valid,
    # but invalid rays can be produced in validation mode and should be discarded for, e.g., metric estimation

    DIFIXED = auto()  # set if the ray label has been processed by the Difix model (via TrainingDifixController)
    SYNTHETIC = auto()  # set if the ray is synthesized by e.g. Gen3C


@chunkable_dataclass_decorator
class ExtraSignal(TorchChunkable):
    """
    Contains all other ray-renderable signals besides [opacity, distance, radiance, transmittance] as in `AlphaCompositing`.
        Each potential field stores the sample-wise data
        Each potential field stores the ray-wise data, as in:
            - VolumeRenderingReturn.extra_ray_signals
    Contains:
        - dinov2_feats:     DINOv2 features [float]     (n_samples or n_rays, 64)        [float32]
        - dinov2_mask:      DINOv2 mask [bool]          (n_rays)                         [bool]
        - semantic: semantic labels                     (n_samples or n_rays, 64)        [float32]
            inferred for camera rays/pixels using a pretrained semantic seg network
        - semantic_logits:  semantic logits             (n_samples or n_rays, n_classes) [float32]
        - normals:  scene-space surface normals         (n_samples or n_rays, 3)         [float32]
            For sample-wise data, this represents the normalized gradients of scalar geometry (density or SDF)
            For ray-wise data, this represents the surface normals of the ray-hit surface element
        - intensity:  intensity for Lidar simulation    (n_samples or n_rays)            [float32]
        - raydrop:  raydrop for Lidar simulation        (n_samples or n_rays)            [float32]
            For sample-wise data, this represents the raydrop possibility of sample
            For ray-wise data, this represents the possibility be dropped of this ray
            Both probabilities range from [0, 1].
        - rgb_background: rgb originating from background model (n_rays, 3)              [float32]
            The foreground opacity is not substracted from the rgb
        - rgb_before_post_processing: rgb before applying (n_rays, 3)                    [float32]
          post-processing
        - velocity: velocity vector (also known as scene flow) of the point
            Unit is meters per second (mps)              (n_rays, 3)                     [float32]
    """

    normals: Optional[torch.Tensor] = None
    intensity: Optional[torch.Tensor] = None
    dinov2_feats: Optional[torch.Tensor] = None
    dinov2_mask: Optional[torch.Tensor] = None
    semantic: Optional[torch.Tensor] = None
    semantic_logits: Optional[torch.Tensor] = None
    raydrop: Optional[torch.Tensor] = None
    rgb_background: Optional[torch.Tensor] = None
    rgb_before_post_processing: Optional[torch.Tensor] = None
    velocity: Optional[torch.Tensor] = None

    def extend(self, other: ExtraSignal) -> None:
        for k, v in dataclass_items(self):
            if (other_v := getattr(other, k, None)) is None:
                continue
            assert v is None or v is other_v, f"Got conflict values for field {k}"
            setattr(self, k, other_v)

    def to_tuple(self) -> tuple[torch.Tensor | None, ...]:
        """Return a tuple of all items in the class, useful when the class is used as checkpoint output."""
        return tuple(v for _, v in dataclass_items(self))

    @classmethod
    def from_tuple(cls, tuple: tuple[torch.Tensor | None, ...]) -> ExtraSignal:
        """Create an ExtraSignal from a tuple, useful when the class is used as checkpoint input."""
        return cls(**dict(zip(dataclass_keys(cls), tuple)))

    @classmethod
    def from_packed_tensor(
        cls,
        extra_signal_tensor: torch.Tensor,
        extra_signal_infos: tuple[list[str], list[int], list[Callable]],
    ) -> ExtraSignal:
        extra_signals_names = extra_signal_infos[0]
        extra_signals_dims = extra_signal_infos[1]
        extra_signals_activations = extra_signal_infos[2]

        assert (
            sum(extra_signals_dims) == extra_signal_tensor.shape[-1]
            and len(extra_signals_names) == len(extra_signals_dims)
            and len(extra_signals_names) == len(extra_signals_activations)
        ), f"Incorrect extra_signal_infos: {extra_signal_infos}, {extra_signal_tensor.shape[-1]}"

        extra_signals_tensors = torch.split(extra_signal_tensor, extra_signals_dims, dim=-1)

        extra_signal = cls()
        for i in range(len(extra_signals_names)):
            setattr(extra_signal, extra_signals_names[i], extra_signals_activations[i](extra_signals_tensors[i]))

        return extra_signal


@chunkable_dataclass_decorator(slots=False, kw_only=True)
@dataclass(slots=True, kw_only=True)
class GaussiansRenderReturn(TorchChunkable):
    """
    Return of the rendering of the gaussians
    """

    rgb: Optional[torch.Tensor] = None
    opacity: torch.Tensor
    distance: torch.Tensor
    normal: Optional[torch.Tensor] = None
    extra_ray_signals: Optional[ExtraSignal] = None
    visibility: Optional[torch.Tensor] = None
    cumulated_weights: Optional[torch.Tensor] = None

    # Per-Gaussian scene-level fields (NOT per-ray). These are excluded from
    # per-ray indexing in _getitem_basedict to avoid shape mismatches.
    _SCENE_LEVEL_FIELDS: ClassVar[frozenset[str]] = frozenset({"visibility", "cumulated_weights"})

    def _getitem_basedict(self, key: torch.Tensor | slice | int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in dataclass_items(self):
            if k in self._SCENE_LEVEL_FIELDS:
                result[k] = None
            elif isinstance(v, (torch.Tensor, TorchChunkable)):
                result[k] = v[key]
            else:
                result[k] = v
        return result

    def to_tuple(self) -> tuple[torch.Tensor | None, ...]:
        """Return a tuple of all items in the class, useful when the class is used as checkpoint output."""
        return (self.rgb, self.opacity, self.distance, self.normal) + (
            self.extra_ray_signals.to_tuple() if self.extra_ray_signals is not None else ()
        )

    @classmethod
    def from_tuple(cls, tuple: tuple[torch.Tensor | None, ...]) -> GaussiansRenderReturn:
        """Create an GaussiansRenderReturn from a tuple, useful when the class is used as checkpoint input."""
        return cls(
            rgb=unpack_optional(tuple[0]),
            opacity=unpack_optional(tuple[1]),
            distance=unpack_optional(tuple[2]),
            normal=tuple[3],
            extra_ray_signals=ExtraSignal.from_tuple(tuple[4:]) if len(tuple) > 4 else None,
        )


@dataclass(slots=True, kw_only=True)
class GaussiansCompositeReturn:
    """
    Return of the GaussianComposite model
    """

    rendered_cam: Optional[GaussiansRenderReturn] = None
    rendered_lidar: Optional[GaussiansRenderReturn] = None
    deform_smoothness: Optional[torch.Tensor] = None
    deform_smoothness_mask: Optional[torch.Tensor] = None


@dataclass(slots=True, kw_only=True)
class FrameConversion(dataclasses_json.DataClassJsonMixin):
    """Represents parameters and functions to convert frame-associated data between different (potentially uniformly scaled) canonical 3d frames"""

    #: Homogeneous source -> target transformation matrix; its dtype declares the output dtype of this conversion.
    #:
    #: ⎡ R  -o ⎤
    #: ⎣ 0 1/s ⎦
    #:
    #: with
    #: - R: source -> target frame orientation with det(R)=1 (3,3)
    #: - o: origin of the target frame in the source frame (in source-frame units) (3,1)
    #: - s: the source -> target scale
    #:
    #: Any floating dtype is accepted (default: float32). Construct with a float64 matrix to enable
    #: double-precision transforms end-to-end (e.g. ECEF coordinate poses). JSON round-trip goes through
    #: field_numpy_array(np.float32, ...) and always decodes as float32; for f64 use-cases construct in-memory.
    matrix: npt.NDArray[np.floating] = field_numpy_array(np.float32, (4, 4))

    def __post_init__(self):
        assert self.matrix.shape == (4, 4)
        if not np.issubdtype(self.matrix.dtype, np.floating):
            raise TypeError(f"Expected floating point matrix dtype, but got {self.matrix.dtype}")
        assert self.matrix[3, 3] > 0.0
        assert np.isclose(np.linalg.det(self.matrix[:3, :3]), 1.0)

    @property
    def dtype(self) -> np.dtype:
        """Returns the declared output dtype of this conversion, taken from the underlying matrix."""
        return self.matrix.dtype

    @property
    def target_scale(self) -> float:
        """The uniform scale of the target frame relative to the source frame"""
        return 1 / self.matrix[3, 3]

    def get_transformation_matrices(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Returns scale-aware (4,4) matrices T / S, which can be used to transform

        - source *points* / *vectors* x_source to the target frame via

          x_target = T @ x_source

        - source *poses* P_source to the target frame via

          P_target = T @ P_source @ S

          Resulting poses have target frame scale when incorporating S, or source frame scale if omitting S

        Both returned matrices have dtype == self.dtype.
        """

        # T has the form
        # ⎡ s*R -s*o ⎤
        # ⎣ 0    1   ⎦
        T = self.matrix.copy()
        T *= self.target_scale

        # S has the form
        # ⎡ 1/s*I 0 ⎤
        # ⎣ 0     1 ⎦
        inv_s = self.matrix[3, 3]
        S = np.zeros((4, 4), dtype=self.dtype)
        np.fill_diagonal(S, [inv_s, inv_s, inv_s, 1.0])

        return (T, S)

    def transform_poses(
        self,
        T_poses_source: np.ndarray,
    ) -> np.ndarray:
        """Transforms poses in the source frame to corresponding poses in the target frame.

        Returned poses have target frame units and dtype == self.dtype. Inputs are cast to self.dtype
        before the matmul so the computation itself happens in the declared dtype (no silent numpy
        promotion, no post-hoc downcast).

        Supports both singular (4,4) and batched (N,4,4) input poses 'T_poses_source'
        """

        # Cast to self.dtype first so the matmul runs in the declared dtype.
        T_poses = T_poses_source.astype(self.dtype, copy=False).reshape((-1, 4, 4))  # (N,4,4)

        # apply transformation
        T, S = self.get_transformation_matrices()
        T_poses = T @ T_poses @ S

        # unbatch dimensions conditionally
        return T_poses.squeeze()  # (N,4,4) or (4,4)


@dataclass(slots=True, kw_only=True)
class RigTrajectories(dataclasses_json.DataClassJsonMixin):
    """Represents a list of rig trajectories (using NCore frame conventions)"""

    # NCore world frame -> base frame rigid transformation (potentially geo-located)
    T_world_base: torch.Tensor = field_torch_tensor(torch.float64, (4, 4), device="cpu", kw_only=True)

    # NCore world -> NRE frame conversion
    world_to_nre: FrameConversion

    @dataclass(slots=True, kw_only=True)
    class RigTrajectory(dataclasses_json.DataClassJsonMixin):
        """Represents a single rig trajectory with associated sensor and frame timestamps"""

        sequence_id: str  # the source sequence id of the current trajectory (might be shared with other trajectories)

        rig_bbox: Optional[
            BBox3 | None
        ]  # if available, the 3d bbox of the sensor rig (vehicle / robot) relative to the rig frame

        # maps of *unique* sensor ids to linear start frame indices (across all rig trajectories and cameras)
        # None if not available (backward compatibility with older artifacts)
        cameras_linear_start_frame_indices: Optional[dict[str, int]] = None
        lidars_linear_start_frame_indices: Optional[dict[str, int]] = None

        cameras_frame_timestamps_us: dict[str, torch.Tensor] = field_torch_tensor(
            torch.int64, (-1, 2), device="cpu", kw_only=True
        )  # map of *unique* camera sensor ids to start-/end-of-frame timestamps Nx2
        lidars_frame_timestamps_us: dict[str, torch.Tensor] = field_torch_tensor(
            torch.int64, (-1, 2), device="cpu", kw_only=True
        )  # map of *unique* lidar sensor indices to start-/end-of-frame timestamps Nx2

        # Timestamped trajectory of the rig frame in NCore world coordinates
        T_rig_worlds: torch.Tensor = field_torch_tensor(torch.float64, (-1, 4, 4), device="cpu", kw_only=True)  # Nx4x4
        T_rig_world_timestamps_us: torch.Tensor = field_torch_tensor(
            torch.int64, (-1,), device="cpu", kw_only=True
        )  # N, guaranteed to cover the *end-of-frame* timestamps for all sensors (start-of-frame may be out-of-time-bounds)

        # Timestamped trajectory of the per-camera rig frame in NCore world coordinates (free poses)
        cameras_frame_T_rig_worlds: Optional[dict[str, torch.Tensor]] = field_torch_tensor(
            torch.float64, (-1, 2, 4, 4), device="cpu", kw_only=True, default=None
        )  # map of *unique* camera sensor ids to start-/end-of-frame poses Nx2x4x4

        def __post_init__(self):
            assert self.T_rig_world_timestamps_us.ndim == 1, "T_rig_world_timestamps_us must be 1D"
            assert len(self.T_rig_worlds) == len(self.T_rig_world_timestamps_us)
            if self.cameras_linear_start_frame_indices is not None:
                assert self.cameras_linear_start_frame_indices.keys() == self.cameras_frame_timestamps_us.keys()
            assert all(
                camera_frame_timestamps_us.shape[1:] == (2,)
                for camera_frame_timestamps_us in self.cameras_frame_timestamps_us.values()
            )
            if self.lidars_linear_start_frame_indices is not None:
                assert self.lidars_linear_start_frame_indices.keys() == self.lidars_frame_timestamps_us.keys()
            assert all(
                lidar_frame_timestamps_us.shape[1:] == (2,)
                for lidar_frame_timestamps_us in self.lidars_frame_timestamps_us.values()
            )

    rig_trajectories: list[RigTrajectory]  # indexed by trajectory index

    @dataclass(slots=True, kw_only=True)
    class SensorCalibration(dataclasses_json.DataClassJsonMixin):
        """Represents a generic sensor-associated calibration"""

        sequence_id: str  # sequence id
        logical_sensor_name: str  # logical sensor name (potentially non-unique for multi-rig-trajectories)
        unique_sensor_idx: int  # unique sensor index (of this associated sensor type!)

        T_sensor_rig: torch.Tensor = field_torch_tensor(
            torch.float32, (4, 4), device="cpu", kw_only=True
        )  # extrinsics 4x4

    @dataclass(slots=True, kw_only=True)
    class CameraCalibration(SensorCalibration):
        """Represents a camera-associated calibration"""

        camera_model_parameters: ConcreteCameraModelParametersUnion = field_camera_model_parameters(
            kw_only=True
        )  # intrinsics [available unconditionally]

    camera_calibrations: OrderedDict[str, CameraCalibration]  # indexed by *unique* camera sensor ids

    @dataclass(slots=True, kw_only=True)
    class LidarCalibration(SensorCalibration):
        """Represents a lidar-associated calibration"""

        lidar_model_parameters: Optional[ConcreteLidarModelParametersUnion] = field_lidar_model_parameters(
            default=None, kw_only=True
        )  # intrinsics [available conditionally only]

    lidar_calibrations: OrderedDict[str, LidarCalibration]  # indexed by *unique* lidar sensor ids

    def __post_init__(self):
        # make sure sensors referenced by trajectories are available
        for rig_trajectory in self.rig_trajectories:
            for camera_id in rig_trajectory.cameras_frame_timestamps_us.keys():
                assert camera_id in self.camera_calibrations, f"Missing camera {camera_id} in camera calibrations"
            for lidar_id in rig_trajectory.lidars_frame_timestamps_us.keys():
                assert lidar_id in self.lidar_calibrations, f"Missing lidar {lidar_id} in lidar calibrations"



class TrackFlags(IntFlag):
    """Bitmask flags of per-track properties (note: limited to 32 variants)"""

    # Special value without any set flag
    NONE = 0

    # Dynamic flags in accordance with the dataset loader
    DYNAMIC = auto()

    # Controllable flags in accordance with the model
    CONTROLLABLE = auto()


@dataclass(kw_only=True, slots=True)
class TracksData:
    """
    Data-components of nre.datasets.tracks.Tracks.

    Args:
        tracks_id: list[str]  - (N_tracks) string identifiers of each track
        max_track_n_poses: int  - maximum number of poses for an individual track among all tracks (used within kernels for shared memory allocations)
        tracks_label_class: list[str]  - (N_tracks) semantic class of each track
        tracks_packinfo: torch.Tensor  - (N_tracks x 2 containing) with [track_start_idx, N_track_poses] each
        tracks_poses: lt.SE3  - (N_total_poses, ) containing SE3 poses
        tracks_timestamps_us: torch.Tensor  - (N_total_poses, ) containing per-pose timestamps
        tracks_flags: torch.Tensor  # (N_tracks) containing per-track flags int32 values (see TrackFlags)
    """

    tracks_id: list[str]
    max_track_n_poses: int
    tracks_label_class: list[str]
    tracks_packinfo: torch.Tensor
    tracks_poses: lt.SE3
    tracks_timestamps_us: torch.Tensor
    tracks_flags: torch.Tensor

    def __post_init__(self):
        """Post-init validation of the tracks data"""

        if len(self.tracks_label_class) != self.n_tracks:
            raise ValueError(
                f"Number of tracks ({self.n_tracks}) does not match number of track label classes ({len(self.tracks_label_class)})"
            )
        if self.tracks_packinfo.ndim != 2:
            raise ValueError(
                f"Track packinfo must have shape (N_tracks, 2), but has shape {self.tracks_packinfo.shape}"
            )
        if self.tracks_packinfo.shape[0] != self.n_tracks:
            raise ValueError(
                f"Number of tracks ({self.n_tracks}) does not match number of track packinfo ({self.tracks_packinfo.shape[0]})"
            )
        if self.tracks_packinfo.shape[1] != 2:
            raise ValueError(
                f"Track packinfo must have shape (N_tracks, 2), but has shape {self.tracks_packinfo.shape}"
            )

        n_total_poses = self.tracks_poses.shape[0]
        if self.tracks_timestamps_us.shape != (n_total_poses,):
            raise ValueError(
                f"Number of total poses ({n_total_poses}) does not match number of track timestamps ({self.tracks_timestamps_us.shape})"
            )
        if self.tracks_flags.shape != (self.n_tracks,):
            raise ValueError(
                f"Number of tracks ({self.n_tracks}) does not match number of track flags ({self.tracks_flags.shape})"
            )
        if self.tracks_flags.dtype != torch.int32:
            raise ValueError(f"Track flags must be of type torch.int32, but is {self.tracks_flags.dtype}")

    @property
    def n_tracks(self) -> int:
        return len(self.tracks_id)

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            tracks_id=self.tracks_id,
            max_track_n_poses=self.max_track_n_poses,
            tracks_label_class=self.tracks_label_class,
            tracks_packinfo=self.tracks_packinfo.to(device),
            tracks_poses=self.tracks_poses.to(device),
            tracks_timestamps_us=self.tracks_timestamps_us.to(device),
            tracks_flags=self.tracks_flags.to(device),
        )

    @classmethod
    def empty(cls, device: torch.device) -> Self:
        """A factory method to create an empty TracksData instance (length zero)."""
        return cls(
            tracks_id=[],
            max_track_n_poses=0,
            tracks_label_class=[],
            tracks_packinfo=torch.tensor([], dtype=torch.int32, device=device).reshape(0, 2),
            tracks_poses=lt.SE3.Identity(0, dtype=torch.float32, device=device),
            tracks_timestamps_us=torch.tensor([], dtype=torch.int64, device=device),
            tracks_flags=torch.tensor([], dtype=torch.int32, device=device),
        )


@dataclass(kw_only=True, slots=True)
class CuboidTracksData:
    """
    Data-components of nre.datasets.tracks.CuboidTracks.

    Args:
        cuboids_dims: torch.Tensor  - (N_tracks, 3) containing per-track dimensions in local track-frame
    """

    cuboids_dims: torch.Tensor

    def __post_init__(self):
        """Post-init validation of the cuboid tracks data"""
        if self.cuboids_dims.ndim != 2 or self.cuboids_dims.shape[1] != 3:
            raise ValueError(f"Cuboids dims must have shape (N_tracks, 3), but has shape {self.cuboids_dims.shape}")
        if self.cuboids_dims.dtype != torch.float32:
            raise ValueError(f"Cuboids dims must be of type torch.float32, but is {self.cuboids_dims.dtype}")

    @property
    def n_tracks(self) -> int:
        return self.cuboids_dims.shape[0]

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            cuboids_dims=self.cuboids_dims.to(device),
        )

    @classmethod
    def empty(cls, device: torch.device) -> Self:
        """A factory method to create an empty CuboidTracksData instance (length zero)."""
        return cls(
            cuboids_dims=torch.tensor([], dtype=torch.float32, device=device).reshape(0, 3),
        )


@dataclass(kw_only=True, slots=True)
class CuboidTracksDataPack:
    """Aggregation of a TracksData and CuboidTracksData pair.

    Args:
        tracks_data: TracksData  - (N_tracks) containing per-track flags int32 values (see TrackFlags)
        cuboidtracks_data: CuboidTracksData  - (N_tracks, 3) containing per-track dimensions in local track-frame
    """

    tracks_data: TracksData
    cuboidtracks_data: CuboidTracksData

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            tracks_data=self.tracks_data.to_device(device),
            cuboidtracks_data=self.cuboidtracks_data.to_device(device),
        )

    def __post_init__(self):
        """Post-init validation of the cuboid tracks data"""
        if self.tracks_data.n_tracks != self.cuboidtracks_data.n_tracks:
            raise ValueError(
                f"Number of tracks ({self.tracks_data.n_tracks}) does not match number of cuboid tracks ({self.cuboidtracks_data.n_tracks})"
            )

    @classmethod
    def empty(cls, device: torch.device) -> Self:
        """A factory method to create an empty CuboidTracksDataPack instance (length zero)."""
        return cls(
            tracks_data=TracksData.empty(device),
            cuboidtracks_data=CuboidTracksData.empty(device),
        )


