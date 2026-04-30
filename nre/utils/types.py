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
    Generic,
    Iterable,
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
# visualdebugger is replaced by an inline no-op at the only call site below
# (PointCloud.visualize); kept as a stub so polyscope is not in the predict
# image at all.
def get_visualdebugger():
    class _NullDebugger:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    return _NullDebugger()


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

    @staticmethod
    def from_series(sorted_series: np.ndarray | torch.Tensor) -> HalfClosedInterval:
        """
        Creates the shortest HalfClosedInterval which covers a *sorted* series (not validated) of integer values
        """
        if isinstance(sorted_series, torch.Tensor):
            return HalfClosedInterval.from_series(sorted_series.numpy())

        assert np.issubdtype(sorted_series.dtype, np.integer), sorted_series.dtype
        assert sorted_series.ndim == 1, sorted_series.ndim

        return HalfClosedInterval(
            int(sorted_series[0]),
            # make sure that the final value is included in the new interval
            int(sorted_series[-1]) + 1,
        )

    def __post_init__(self) -> None:
        assert self.start <= self.end

    def __contains__(self, item: int | HalfClosedInterval) -> bool:
        if isinstance(item, int):
            return self.start <= item < self.end
        elif isinstance(item, HalfClosedInterval):
            return (self.start <= item.start) and (item.end <= self.end)
        else:
            raise TypeError(f"Expected int or HalfClosedInterval, got {type(item).__name__}")

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.end or other.end <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.end, other.end))

    @staticmethod
    def union(intervals: Iterable[HalfClosedInterval]) -> HalfClosedInterval:
        """Creates the shortest HalfClosedInterval which contains all elements of `intervals`"""
        iterator = iter(intervals)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("intervals needs to contain at least one element.")

        start = first.start
        end = first.end

        for interval in iterator:
            start = min(start, interval.start)
            end = max(end, interval.end)

        return HalfClosedInterval(start, end)

    def overlaps(self, other: HalfClosedInterval) -> bool:
        """Checks if the interval has a non-zero overlap with an other closed interval"""
        return self.intersection(other) is not None

    def cover_range(self, sorted_series: np.ndarray) -> range:
        """Given a set of *sorted* series (not validated), return the corresponding range for samples
        that are within the interval"""
        assert np.any(self.start <= sorted_series), "All elements in the series are before the start of the range"
        assert np.any(sorted_series < self.end), "All elements in the series are after the end of the range"

        cover_range_start = np.argmax(self.start <= sorted_series).item()
        cover_range_stop = (
            np.argmin(sorted_series < self.end).item() if self.end < sorted_series[-1] else len(sorted_series)
        )  # full range of frames

        return range(cover_range_start, cover_range_stop)

    def restricted(self, sorted_series: np.ndarray) -> HalfClosedInterval:
        """Returns a restricted version of the interval that guarantees to cover a *sorted* series (not validated)"""
        return HalfClosedInterval.from_series(sorted_series[self.cover_range(sorted_series)])

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_local(self, value: int) -> int:
        """Maps from [start, end) to [0, self.length)"""
        if not value in self:
            raise ValueError(f"{value} is outside [start={self.start}, end={self.end})")
        return value - self.start

    def to_global(self, value: int) -> int:
        """Maps from [0, self.length) to [start, end)"""
        if not (0 <= value < self.length):
            raise ValueError(f"{value} is outside [0, length={self.length})")
        return value + self.start


@dataclass(kw_only=True)
class NamedSerialized:
    filename: str
    serialized: str | bytes

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(self.serialized, bytes) else "w"
        with open(out_dir / self.filename, mode) as f:
            f.write(self.serialized)
        logging.info(f"Saved file: {out_dir / self.filename}")

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        zip_file.writestr(self.filename, self.serialized)

    @classmethod
    def from_config(cls, config: DictConfig, filename: str = "parsed_config.yaml") -> Self:
        return cls(filename=filename, serialized=OmegaConf.to_yaml(config))


@dataclass(kw_only=True)
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save(self, out_dir: Path, preserve_references: bool = False):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / self.filename)
        if preserve_references:
            # Export root layer only - preserves reference arcs
            self.stage.GetRootLayer().Export(out_path)
        else:
            # Flatten and export composed result
            self.stage.Export(out_path)
        logging.info(f"Saved file: {out_dir / self.filename}")

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


ArtifactContents: TypeAlias = List[NamedSerialized | NamedUSDStage]


@dataclass
class SequenceChunk:
    """Represents a chunk (given by time-range) within a sequence"""

    sequence_id: str
    time_range_us: HalfClosedInterval

    def time_length_us(self) -> int:
        return self.time_range_us.length

    def time_length_sec(self) -> float:
        return self.time_length_us() / 1e6


class CameraFrustum:
    """
    We use the following convention for the frustum corners

              5--------6
             /|       /|
            1-+------2 |
            | |      | |
            | 4------+-7
            |/       |/
            0--------3

    """

    def __init__(self, corners: torch.Tensor, device: torch.device = torch.device("cpu")) -> None:
        """Represents a camera frustum through the near plane corners, the 4 vectors and depth along the normal"""
        self.corners: torch.Tensor = corners.to(device)  # [8,3]
        self.device = device
        self.edges: torch.Tensor = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        ).to(self.device)

        self.planes: torch.Tensor = torch.tensor(
            [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0], [4, 0, 3, 7], [3, 2, 6, 7], [1, 5, 6, 2]]
        ).to(self.device)
        assert self.corners.shape == (8, 3), "Frustum is defined by 8 corners"
        self.check_input_conformity()

    def check_input_conformity(self) -> None:
        """We make two assumptions on the input: 1) first and second 4 points define the near, far plane respectively. Hence,
        their corners need to be coplanar, 2) the two planes are parallel"""

        # Near plane
        v = self.corners[1:4] - self.corners[0:1]
        v /= v.norm(dim=-1, keepdim=True)

        # TODO[nischneider]: higher tolerance needed for tele cameras
        # probably a per camera tolerance or a more robust check is needed
        determinant_tolerance = 1e-2

        assert torch.isclose(torch.det(v), torch.zeros(1, device=self.device), atol=determinant_tolerance), (
            "The corners of near plane are not coplanar"
        )

        # Far plane
        v = self.corners[5:] - self.corners[4:5]
        v /= v.norm(dim=-1, keepdim=True)
        assert torch.isclose(torch.det(v), torch.zeros(1, device=self.device), atol=determinant_tolerance), (
            "The corners of far plane are not coplanar"
        )

        # Compute the normal vectors of all planes and check that the ones of near/far plane are parallel
        self.compute_normal_vectors()
        assert torch.isclose(torch.dot(self.normals[0], self.normals[0]), torch.ones(1, device=self.device)), (
            "The near and far plane are not parallel"
        )

    def compute_normal_vectors(self):
        """Get the indices of the points used to compute the normal vectors. Planes are defined such that the first three indices
        always form a normal vector that points into the frustum"""
        normal_idx = self.planes[:, :3].flatten()
        points = self.corners[normal_idx].reshape(6, 3, 3)
        vectors = points[:, [2, 0]] - points[:, 1:2]
        normals = torch.linalg.cross(vectors[:, 0], vectors[:, 1])
        self.normals = normals / normals.norm(dim=1, keepdim=True)

    def points_in_frustum(self, points: torch.Tensor) -> torch.Tensor:
        points_on_plane = self.corners[self.planes[:, 0]]
        w = points.unsqueeze(1).to(self.device) - points_on_plane
        dist_to_plane = torch.sum(w * self.normals.unsqueeze(0), dim=-1)

        return (dist_to_plane >= 0.0).all(dim=1)

    def get_aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.corners.min(0).values, self.corners.max(0).values

    def to(self, device: torch.device) -> CameraFrustum:
        self.corners = self.corners.to(device=device)
        self.edges = self.edges.to(device=self.device)
        self.planes = self.planes.to(device=self.device)
        self.device = device
        self.check_input_conformity()
        return self


class BoundingBox:
    def __init__(
        self,
        extent: torch.Tensor,
        T_rig_world: torch.Tensor = torch.eye(4),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Represents a bounding box through its 8 corners"""

        # TODO[ZG]: split into length, width, heigh and check if 1 or 2 values are passed
        self.device: torch.device = device
        self.extent: torch.Tensor = extent.to(device)  # [3,2]
        self.T_rig_world: torch.Tensor = T_rig_world.to(device)
        self.edges: torch.Tensor = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        ).to(device)
        self.get_corners()

    def get_corners(self) -> None:
        local_corners = torch.tensor(
            [
                [self.extent[0, 0], self.extent[1, 0], self.extent[2, 0]],  # Back bottom left
                [self.extent[0, 0], self.extent[1, 0], self.extent[2, 1]],  # Back top left
                [self.extent[0, 0], self.extent[1, 1], self.extent[2, 1]],  # Back top right
                [self.extent[0, 0], self.extent[1, 1], self.extent[2, 0]],  # Back bottom right
                [self.extent[0, 1], self.extent[1, 0], self.extent[2, 0]],  # Front bottom left
                [self.extent[0, 1], self.extent[1, 0], self.extent[2, 1]],  # Front top left
                [self.extent[0, 1], self.extent[1, 1], self.extent[2, 1]],  # Front top right
                [self.extent[0, 1], self.extent[1, 1], self.extent[2, 0]],  # Front bottom right
            ]
        ).to(self.device)

        self.corners = (self.T_rig_world[:3, :3] @ local_corners.transpose(0, 1) + self.T_rig_world[:3, 3:4]).transpose(
            0, 1
        )

    def to(self, device: torch.device) -> BoundingBox:
        self.corners = self.corners.to(device=device)
        self.edges = self.edges.to(device=self.device)
        self.T_rig_world = self.T_rig_world.to(device=self.device)
        self.extent = self.extent.to(device=self.device)
        self.device = device
        return self


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


@chunkable_dataclass_decorator(slots=True, kw_only=True)
class PointCloud(TorchChunkable):
    """
    Represents a 3d point cloud consisting of corresponding start and end points
    Optionally can contain per-point colors and flags (see RayFlags)
    and per point semantic class ids.
    """

    xyz_start: torch.Tensor  # [N,3] coordinates of the start points [float32]
    xyz_end: torch.Tensor  # [N,3] coordinates of the end points [float32]
    normal: Optional[torch.Tensor] = None  # [N,3] normal vector of each point [float32]
    color: Optional[torch.Tensor] = None  # [N,3] RGB color of each point [uint8]
    flags: Optional[torch.Tensor] = None  # [N,]-sized tensor of RayFlags
    semantic_class_id: Optional[torch.Tensor] = None  # [N,]-sized tensor of per point semantic class id
    intensity: Optional[torch.Tensor] = None  # [N, ]-sized tensor of per point intensity [float32]
    camera_footprint_scale: Optional[torch.Tensor] = None  # [N] observation scale of each point [float32]
    sensor_type: Optional[List[Literal["camera", "lidar"]]] = None  # one of "camera" or "lidar" if specified [str]

    def __post_init__(self) -> None:
        # dimension checks
        assert self.xyz_start.shape == (self.n_points, 3)
        assert self.xyz_end.shape == self.xyz_start.shape
        assert self.normal is None or self.normal.shape == self.xyz_start.shape
        assert self.color is None or self.color.shape == self.xyz_start.shape
        assert self.flags is None or self.flags.shape == (self.n_points,)
        assert self.semantic_class_id is None or self.semantic_class_id.shape == (self.n_points,)
        assert self.intensity is None or self.intensity.shape == (self.n_points,)
        assert self.camera_footprint_scale is None or self.camera_footprint_scale.shape == (self.n_points,)

        # type checks
        assert self.color is None or self.color.dtype == torch.uint8

    def visualize(self) -> None:
        visualizer = get_visualdebugger()
        visualizer.add_point_cloud(
            "pc",
            self.xyz_end.cpu().numpy(),
            colors_quantities=({"color": self.color.cpu().numpy()} if self.color is not None else None),
        )
        visualizer.show()

    def flipped(self, axis: AxisType = "Y") -> PointCloud:
        """Returns a new PointCloud with end-points flipped along the specified axis"""
        match axis:
            case "X":
                return PointCloud(
                    xyz_start=self.xyz_start,
                    xyz_end=self.xyz_end * torch.tensor([-1, 1, 1], device=self.xyz_start.device),
                    color=self.color,
                    flags=self.flags,
                    semantic_class_id=self.semantic_class_id,
                    intensity=self.intensity,
                    camera_footprint_scale=self.camera_footprint_scale,
                )
            case "Y":
                return PointCloud(
                    xyz_start=self.xyz_start,
                    xyz_end=self.xyz_end * torch.tensor([1, -1, 1], device=self.xyz_start.device),
                    color=self.color,
                    flags=self.flags,
                    semantic_class_id=self.semantic_class_id,
                    intensity=self.intensity,
                    camera_footprint_scale=self.camera_footprint_scale,
                )
            case "Z":
                return PointCloud(
                    xyz_start=self.xyz_start,
                    xyz_end=self.xyz_end * torch.tensor([1, 1, -1], device=self.xyz_end.device),
                    color=self.color,
                    flags=self.flags,
                    semantic_class_id=self.semantic_class_id,
                    intensity=self.intensity,
                    camera_footprint_scale=self.camera_footprint_scale,
                )
            case _:
                raise ValueError(f"Unsupported axis {axis}")

    @property
    def n_points(self) -> int:
        return len(self.xyz_end)


# Defining this inside PointCloud feels more elegant but it leads to this mypy error:
# error: Type aliases inside dataclass definitions are not supported at runtime  [misc]
# generic-data is for color loaded from sfm point data
PointCloudColorType: TypeAlias = Optional[Literal["camera-rgb", "semantics", "generic-data"]]

AxisType: TypeAlias = Literal["X", "Y", "Z"]


@dataclass(slots=True, kw_only=True)
class TrackPointCloud:
    """Represents a 3d point cloud associated with a track"""

    track_id: str
    point_cloud: PointCloud


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

    def fill_with_defaults(self, to_fill_with_defaults: dict[str, torch.Tensor], prefix_shape) -> None:
        for signal_type, default in to_fill_with_defaults.items():
            if getattr(self, signal_type) is None:
                setattr(self, signal_type, default.tile(prefix_shape, *[1] * default.dim()))

    def concatenated(self) -> Optional[torch.Tensor]:
        """Returns a concatenated tensor of all non-None tensors in the ExtraSignal class.
        Returns None if all tensors are None.
        """
        tensors = []
        for _, v in dataclass_items(self):
            if isinstance(v, torch.Tensor):
                assert v.dim() == 1 or v.dim() == 2, f"Tensor {v.shape} is not 1D or 2D"
                tensors.append(v.unsqueeze(-1) if v.dim() == 1 else v)

        return torch.cat(tensors, dim=-1) if tensors else None

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
class ExtraReturn(TorchChunkable):
    """
    Contains extra learned signals (different from ExtraSignal, these are necessarily ray-wise or sample-wise):
        - pe_map: optional learned positional embedding map [float] (n_rays, n_pe_features)
    """

    pe_map: Optional[torch.Tensor] = None

    def extend(self, other: ExtraReturn) -> None:
        for k, v in dataclass_items(self):
            if (other_v := getattr(other, k, None)) is None:
                continue
            assert v is None or v is other_v, f"Got conflict values for field {k}"
            setattr(self, k, other_v)


ModelType = TypeVar("ModelType")


@chunkable_dataclass_decorator(slots=False, kw_only=True)
class ModelParametersList(TorchChunkable, Generic[ModelType]):
    data: List[ModelType]

    def item(self) -> ModelType:
        assert len(self.data) == 1, "ModelParametersList must contain exactly one element to call item()"
        return self.data[0]

    def __getitem__(self, key: torch.Tensor | slice | int) -> Self:
        assert not isinstance(key, torch.Tensor), f"Indexing {self.__class__.__name__} with a tensor is not supported"
        if isinstance(key, int):
            key = slice(key, key + 1)
        return replace(self, data=self.data[key])

    @classmethod
    def collate_fn(
        cls,
        batch,
        device: torch.device = torch.device("cpu"),
        unsqueeze_if_zero_dim: bool = True,
    ) -> ModelParametersList[ModelType]:
        collated: List[ModelType] = []
        for c in batch:
            collated.extend(c.data)
        return cls(data=collated)


class RadianceEmbeddingType(IntEnum):
    """Type enumeration for the concrete type of radiance-embedding features"""

    RGB = auto()  # directly represent RBG radiance values K=3
    EMPTY = auto()  # Empty radiance. K=0. Required for e.g. Lidar-only processes and simulation


@dataclass(slots=True)
class ModelInput:
    """
    Contains:
        - xyzs: x,z,y coordinates of the samples along all rays [float] (n_samples, 3)
        - xyzs_unit_cube: x,z,y coordinates of the samples along all rays contracted to the unit cube [0,1]^3 [float] (n_samples, 3)
        - dirs: dirs direction vector for each of the samples  [float] (n_samples, 3)
        - ts: distance along the ray for each sample [float] (n_samples,)
        - deltas: step size for each sample [float] (n_samples,)
        - pack_info: packed auxiliary tensor containing starting indices and num samples for each pack [int] (n_packs, 2)

        - levels: per-sample level value for LoD behavior, within range [-inf, n_levels] [float] (n_samples,)
        - normals: per-sample surface normals [float] (n_samples,)

        - timestamps_us: per-sample time-of-measurement in microseconds [float] (n_samples, )
        - time_emb: per-sample temporal embedding [float] (n_samples, some_embedding_dim)

        - instance_idx: per-sample local instance indices ("local" meaning in the current object_layer) [long] (n_samples, )
        - global_instance_idx: per-sample global instance indices ("global" meaning within all the tracks in scenes) [long] (n_samples, )
        - instance_emb: per-sample instance embedding [float] (n_samples, some_embedding_dim)
    """

    xyzs: Optional[torch.Tensor] = None
    xyzs_unit_cube: Optional[torch.Tensor] = None
    dirs: Optional[torch.Tensor] = None
    ts: Optional[torch.Tensor] = None
    deltas: Optional[torch.Tensor] = None
    pack_info: Optional[torch.Tensor] = None

    levels: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None

    timestamps_us: Optional[torch.Tensor] = None
    time_emb: Optional[torch.Tensor] = None

    instance_idx: Optional[torch.Tensor] = None
    global_instance_idx: Optional[torch.Tensor] = None  # TODO: [JG] Always store?
    instance_emb: Optional[torch.Tensor] = None

    def detach(self):
        return replace(self, **{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in dataclass_items(self)})


@dataclass(slots=True, kw_only=True)
class VolumeRenderingReturn:
    """
    Contains:
        - n_vr_samples: total number samples used in volume rendering across all rays [int] (1,)
        - sample_weights: weights of each sample along the ray [float] (n_samples,)
        - sample_transmittance: transmittance of each sample along the ray [float] (n_samples,)
        - pack_info: packed auxiliary tensor containing starting indices and num samples for each ray [int] (n_rays, 2)
        - vr_samples: number of samples used in volume rendering per ray [int] (n_rays,)
        - opacity: accumulated opacity along each ray [float] (n_rays,)
        - distance: integral across distances along the ray for each ray [float] (n_rays,)
        - radiance_embedding_type: concrete type of the features stored in 'radiance_embedding', from which final radiance values can be decoded
        - radiance_embedding: alpha composited features targeting radiance computation with type-specific dimension K for each ray [float] (n_rays, K)
        - extra_ray_signals: optional extra signal per ray, see ExtraSignal (e.g., DINOv2 features or semantic logits)
    """

    # Sample-wise data
    n_vr_samples: int
    sample_weights: Optional[torch.Tensor]
    sample_transmittance: Optional[torch.Tensor]

    # Ray-wise data
    pack_info: torch.Tensor
    vr_samples: torch.Tensor
    opacity: torch.Tensor
    distance: torch.Tensor

    radiance_embedding_type: RadianceEmbeddingType
    radiance_embedding: torch.Tensor

    extra_ray_signals: Optional[ExtraSignal] = None

    @property
    def rgb(self) -> torch.Tensor:
        # TODO: [JG] Allow for radiance embedding type fallback/decode callback?
        assert self.radiance_embedding_type == RadianceEmbeddingType.RGB, (
            f"Can not get RGB from current radiance embedding type {self.radiance_embedding_type}"
        )
        return self.radiance_embedding


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

    @classmethod
    def from_origin_scale_axis(
        cls,
        target_origin: npt.NDArray[np.floating],
        target_scale: float,
        target_axis: list[int],
        dtype: npt.DTypeLike = np.float32,
    ):
        """Construct FrameConversion from
        - target_origin: origin of the target frame relative to the source frame (in source-frame units)
        - target_scale: uniform scale of the target frame relative to the source frame
        - target_axis: The target's frame axis order relative to the source frame using axis indices.
                       For instance, an axis conversion of xyz[source] -> yzx[target] would be represented by [1, 2, 0]
        - dtype: Floating dtype of the resulting matrix; becomes the output dtype of all transforms (default: float32)
        """
        # Construct homogeneous transformation matrix from translation / scale / orientation components
        matrix = np.eye(4, dtype=dtype)

        # translation
        matrix[:3, 3] = -target_origin

        # scale
        matrix[3, 3] = 1 / target_scale

        # axis swap
        assert len(np.unique(target_axis)) == 3
        matrix = matrix[target_axis + [3]]

        return cls(matrix=matrix)

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
    def target_origin(self) -> npt.NDArray[np.floating]:
        """The origin of the target frame relative to the source frame (in source-frame units)"""
        return -self.matrix[:3, 3]

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

    @cached_property
    def rotation_quat_tuple(self) -> tuple[float, float, float, float]:
        """Rotation as quaternion (x, y, z, w) format"""
        from nre.utils.geometry import so3_matrix_to_quat

        quat = so3_matrix_to_quat(self.matrix[:3, :3])
        return tuple(quat.tolist())

    @cached_property
    def translation_tuple(self) -> tuple[float, float, float]:
        """Translation component as tuple"""
        return tuple(self.matrix[:3, 3].tolist())

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

    def transform_points(
        self,
        points_source: np.ndarray,
    ) -> np.ndarray:
        """Transforms points in the source frame to points in the target frame.

        Returned points have target frame units and dtype == self.dtype. Inputs are cast to self.dtype
        before the matmul so the computation itself happens in the declared dtype.

        Support boths singular (3,) and batched (N, 3) input points `points_source`.
        """
        # Cast to self.dtype first so the matmul runs in the declared dtype.
        points = points_source.astype(self.dtype, copy=False).reshape((-1, 3))  # (N, 3)
        src2tgt, _ = self.get_transformation_matrices()
        aug_points = np.hstack((points, np.ones((points.shape[0], 1), dtype=self.dtype)))
        points = (src2tgt @ aug_points.T)[:3, :].T
        return points.squeeze()  # (N, 3) or (3,)

    def inverse(self) -> FrameConversion:
        """Return the FrameConversion from target to source.

        Homogeneous target -> source transformation of the form
        ⎡ inv(R)      inv(R) @ o * s ⎤
        ⎣   0                s       ⎦

        dtype is preserved: the returned FrameConversion has the same dtype as self.
        """
        matrix = np.eye(4, dtype=self.dtype)
        o = -self.matrix[:3, 3:]
        s = self.target_scale
        matrix[:3, :3] = (inv_R := self.matrix[:3, :3].T)
        matrix[:3, 3:] = inv_R @ o * s
        matrix[3, 3] = s
        return FrameConversion(matrix=matrix)


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


class AABB3D(torch.nn.Module):
    """Represents the 3D Axis Aligned Bounding Box where, The box assumes the following convention:

    - Extent: the extent of the bounding box along each of the axis [n, 3]
    - Center: the coordinates of the center point [n, 3]
    - blb: the coordinates of the bottom left back corner of the bounding boxes [n, 3]
    - trf: the coordinates of the top right front corner of the bounding boxes  [n, 3]

    """

    def __init__(self, blb: torch.Tensor, trf: torch.Tensor):
        super().__init__()

        if len(blb.shape) == 1:
            blb = blb.unsqueeze(0)
            trf = trf.unsqueeze(0)

        self.blb = torch.nn.Buffer(blb)
        self.trf = torch.nn.Buffer(trf)

        assert self.blb.shape == self.trf.shape
        assert self.blb.shape[1] == 3
        assert self.blb.dtype == torch.float32
        assert self.trf.dtype == torch.float32
        assert self.blb.device == self.trf.device

    def clone(self) -> AABB3D:
        return AABB3D(self.blb.clone(), self.trf.clone())

    @classmethod
    def from_center_extent(cls, center: torch.Tensor, extent: torch.Tensor):
        """Initializes AABBs from their center and extent"""

        assert center.shape == extent.shape

        return cls(blb=center - extent / 2, trf=center + extent / 2)

    @classmethod
    def from_compact(cls, compact: torch.Tensor):
        """Initializes AABBs from a compact representation, i.e. the AABBs coordinates of the {blb, trf} concatenated [n,6]"""

        assert len(compact.shape) in [1, 2]
        assert compact.shape[-1] == 6

        return cls(blb=compact[..., :3], trf=compact[..., 3:])

    def scale_aabb(self, scale: float | torch.Tensor) -> AABB3D:
        """Scales the AABBs with a given scale (multiplied by the scale factor)"""

        if isinstance(scale, torch.Tensor):
            assert scale.dtype == torch.float32
            assert len(scale.shape) == 2
            assert scale.shape[0] == self.blb.shape[0]
            assert scale.shape[1] in [1, 3]

        # Return a new instance
        return AABB3D(blb=self.blb * scale, trf=self.trf * scale)

    def offset_aabb(self, offset: float | torch.Tensor) -> AABB3D:
        """Offsets the AABBs with a given offset (offset added to the AABBs)"""

        if isinstance(offset, torch.Tensor):
            assert offset.dtype == torch.float32
            assert len(offset.shape) == 2
            assert offset.shape[0] == self.blb.shape[0]
            assert offset.shape[1] in [1, 3]

        return AABB3D(blb=self.blb + offset, trf=self.trf + offset)

    def points_to_unit_cube(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transforms (scales and shifts) the data from the AABB to a unit cube

        Input:
            points [torch.Tensor] points in the aabb world [n x 3]
        """

        assert self.blb.shape[0] == 1 or points.shape[0] == self.blb.shape[0]

        return (points - self.blb) / self.get_extent()

    def points_from_unit_cube(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transforms (scales and shifts) the data from a unit cube to the AABB
        Input:
            points [torch.Tensor] points in the aabb world [n x 3]
        """

        assert self.blb.shape[0] == 1 or points.shape[0] == self.blb.shape[0]

        return points * self.get_extent(unbatch=False) + self.blb

    def get_extent(self, unbatch: bool = True) -> torch.Tensor:
        """The extent of the AABBs along each axis"""

        extent = self.trf - self.blb
        return extent.squeeze(0) if unbatch else extent  # (N,3) or (3)

    def get_center(self, unbatch: bool = True) -> torch.Tensor:
        """The center of the AABBs"""

        center = self.trf + self.blb
        return center.squeeze(0) if unbatch else center  # (N,3) or (3)

    def get_compact(self, unbatch: bool = True) -> torch.Tensor:
        """The compact representation of the AABBs coordinates of the {blb, trf} concatenated [n,6]"""

        compact = torch.cat([self.blb, self.trf], dim=1)
        return compact.squeeze(0) if unbatch else compact  # (N,6) or (6)

    def points_within_aabbs(self, points: torch.Tensor, in_all=False) -> torch.Tensor:
        """
        Checks if the points lie within the AABBs
        Input:
            points [torch.Tensor] points in the aabb world [m x 3] or [n x m x 3]
            in_all [bool] if true, point needs to lie in all the AABBs otherwise at least in one
        """

        if points.dim() == 3:
            assert points.shape[0] == 1 or points.shape[0] == self.blb.shape[0]

        elif points.dim() == 2:
            points = points.unsqueeze(0)

        else:
            raise ValueError("Points must be have dim 2 or 3")

        # Subtract the center to get the local points [n x m x 3]
        local_points = points - self.get_center(unbatch=False).unsqueeze(1)

        # Check if all the point dimensions are smaller than half of the extent -> if yes, point is within the bbox
        local_points_within_aabbs = (torch.abs(local_points) < (self.get_extent(unbatch=False).unsqueeze(1) / 2)).all(
            dim=2
        )

        return local_points_within_aabbs.all(dim=0) if in_all else local_points_within_aabbs.any(dim=0)

    @property
    def device(self) -> torch.device:
        return self.blb.device

    def __getitem__(self, index: torch.Tensor) -> AABB3D:
        # We only assert on the shape to be 1 here such that the dimensions agree, other asserting will be done by the default indexing operation
        assert len(index.shape) == 1

        return AABB3D(blb=self.blb[index], trf=self.trf[index])

    def __len__(self):
        return len(self.blb)


class SceneContractor(torch.nn.Module):
    """Provides methods to contract the points from the aabb to the contracted space or vice versa:

    - aabb: axis aligned bounding box
    - degree: degree of the norm used for space contraction (2.0 - sphere as in MipNerf360, float("inf") bounds the space to a cube)
    - is_single: indicates if it represents a single scene or multiple instances/objects
    - is_merf: indicates if it is uses MERF's contraction (eq. 7 of the paper)

    """

    aabb: AABB3D
    degree: float | None
    is_single: bool = True

    def __init__(self, degree: float | None, aabb: AABB3D, is_single: bool = True, is_merf: bool = False):
        super().__init__()

        assert (not is_merf) or degree == float("inf"), "MERF contraction should only be used with infinity norm"
        self.aabb = aabb
        self.degree = degree
        self.is_single = is_single
        self.is_merf = is_merf

    @torch.no_grad()
    def is_coord_in_warped_distant(self, points: torch.Tensor) -> torch.Tensor:
        points = self.aabb.points_to_unit_cube(points)
        if self.degree is None:
            mask = (points < 0).any(dim=-1) | (points > 1).any(dim=-1)
        else:
            norm = torch.linalg.norm(points, ord=self.degree, dim=-1)
            mask = norm > 1
        return mask

    def to_contracted_space(self, points: torch.Tensor) -> torch.Tensor:
        points = self.aabb.points_to_unit_cube(points)

        if self.degree is None:
            return points

        points = 2 * points - 1  # points in [-1,1]
        norm = torch.linalg.norm(points, ord=self.degree, dim=-1, keepdim=True)
        if self.is_merf:
            scale = norm.clamp_min(1)  # Only affects points with a norm larger than 1
            is_largest_coord = points.abs() == norm
            contracted_points = torch.where(is_largest_coord, (points / scale) * (2 - 1 / scale), points / scale)
        else:
            contracted_points = torch.where(norm < 1, points, (2 - 1 / norm) * (points / norm))

        return contracted_points * 0.25 + 0.5  # [-inf, inf] is at [0, 1]

    def from_contracted_space(self, contracted_points: torch.Tensor) -> torch.Tensor:
        if self.degree is not None:
            contracted_points = (contracted_points - 0.5) * 4  # [0, 1]^3 -> [-2, 2]^3

            norm = torch.linalg.norm(contracted_points, ord=self.degree, dim=-1, keepdim=True)

            if self.is_merf:
                largest_coord = (1.0 / (2.0 - contracted_points.abs() + 1e-10)).max(dim=-1, keepdim=True)[0]
                scale = largest_coord.clamp_min(1)
                is_largest_contracted_coord = contracted_points.abs() == norm
                points = torch.where(
                    is_largest_contracted_coord,
                    (contracted_points / (2.0 - 1.0 / scale)) * scale,
                    contracted_points * scale,
                )
            else:
                points = torch.where(
                    norm < 1, contracted_points, contracted_points / (2.0 * norm - torch.square(norm) + 1e-10)
                )

            points = points * 0.5 + 0.5
        else:
            points = contracted_points

        return self.aabb.points_from_unit_cube(points)

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def __getitem__(self, index: torch.Tensor) -> SceneContractor:
        # We only assert on the shape to be 1 here such that the dimensions agree, other asserting will be done by the default indexing operation
        assert len(index.shape) == 1

        return SceneContractor(degree=self.degree, aabb=self.aabb[index], is_merf=self.is_merf)


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


@dataclass(slots=True)
class NovelViewOverrides:
    """Overrides for novel view sampling.
    These can be used with a custom sampler to sample novel views from a dataset.

    ie. see ParameterizedSequentialSampler for an example of how to use this
    """

    transl_delta_m: npt.NDArray[np.float32] | None
    rot_delta_deg: npt.NDArray[np.float32] | None
