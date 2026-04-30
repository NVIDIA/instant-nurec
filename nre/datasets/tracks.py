# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional, Self, Tuple, Union, cast

import dataclasses_json
import lietorch as lt
import numpy as np
import torch

from torch.autograd.function import once_differentiable

from libs.packed_ops.interface import packed_ops  # type: ignore
from libs.vren.interface import vren  # type: ignore
from nre.utils.fields import field_enum
from nre.utils.geometry import se3_matrix_to_tquat, tquat_to_se3_matrix
from nre.utils.misc import get_pack_info_from_n
from nre.utils.packed_ops import linstep_interleave
from nre.utils.types import CuboidTracksData, CuboidTracksDataPack, FrameConversion, TrackFlags, TracksData
def get_visualdebugger():
    class _Null:
        def __getattr__(self, _n):
            return lambda *a, **k: None
    return _Null()


@dataclass(kw_only=True, slots=True)
class Tracks(dataclasses_json.DataClassJsonMixin):
    """Manages time-dependent poses and related interpolation / intersection GPU operations for tracks

    This class uses the :class:`nre.utils.types.TracksData` class for storing data.
    """

    @dataclass(slots=True)
    class Unpacked(dataclasses_json.DataClassJsonMixin):
        """'Unpacked' serialization format of the internal packed representation that is easier for human / external modification"""

        tracks_id: list[str] = field(default_factory=list)
        tracks_poses: list[list] = field(default_factory=list)
        tracks_timestamps_us: list[list] = field(default_factory=list)
        tracks_label_class: list[str] = field(default_factory=list)
        tracks_flags: list[TrackFlags] = field_enum(TrackFlags, default_factory=list)

        @staticmethod
        def from_data(data: TracksData) -> Tracks.Unpacked:
            tracks_id = []
            tracks_poses = []
            tracks_timestamps_us = []
            tracks_label_class = []
            tracks_flags = []

            # Unpack data for each track
            for track_idx, track_id in enumerate(data.tracks_id):
                tracks_id.append(track_id)
                start_idx, n_poses = data.tracks_packinfo[track_idx].tolist()
                tracks_poses.append(data.tracks_poses[start_idx : start_idx + n_poses].data.tolist())
                tracks_timestamps_us.append(data.tracks_timestamps_us[start_idx : start_idx + n_poses].data.tolist())
                tracks_label_class.append(data.tracks_label_class[track_idx])
                tracks_flags.append(TrackFlags(int(data.tracks_flags[track_idx].item())))

            return Tracks.Unpacked(
                tracks_id,
                tracks_poses,
                tracks_timestamps_us,
                tracks_label_class,
                tracks_flags,
            )

    @staticmethod
    def encoder_tracks(input: TracksData) -> dict:
        return Tracks.Unpacked.from_data(input).to_dict()

    @staticmethod
    def decoder_tracks(input: dict) -> TracksData:
        unpacked = Tracks.Unpacked.from_dict(input)
        return Tracks.Factory.from_numpy(
            unpacked.tracks_id,
            [np.array(track_poses, dtype=np.float32) for track_poses in unpacked.tracks_poses],
            [np.array(track_timestamps_us, dtype=np.int64) for track_timestamps_us in unpacked.tracks_timestamps_us],
            unpacked.tracks_label_class,
            unpacked.tracks_flags,
            "tquat",
        ).tracks_data

    tracks_data: TracksData = field(metadata=dataclasses_json.config(encoder=encoder_tracks, decoder=decoder_tracks))

    @property
    def tracks_id(self) -> list[str]:
        return self.tracks_data.tracks_id

    @property
    def max_track_n_poses(self) -> int:
        return self.tracks_data.max_track_n_poses

    @property
    def tracks_label_class(self) -> list[str]:
        return self.tracks_data.tracks_label_class

    @property
    def tracks_packinfo(self) -> torch.Tensor:
        return self.tracks_data.tracks_packinfo

    @property
    def tracks_poses(self) -> lt.SE3:
        return self.tracks_data.tracks_poses

    @tracks_poses.setter
    def tracks_poses(self, value: lt.SE3):
        """Allow updating tracks poses with consistent values"""
        assert (
            self.tracks_data.tracks_poses.shape == value.shape
            and self.tracks_data.tracks_poses.device == value.device
            and self.tracks_data.tracks_poses.dtype == value.dtype
        )
        self.tracks_data.tracks_poses = value

    @property
    def tracks_timestamps_us(self) -> torch.Tensor:
        return self.tracks_data.tracks_timestamps_us

    @property
    def tracks_flags(self) -> torch.Tensor:
        return self.tracks_data.tracks_flags

    class Factory:
        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_label_class: list[str],
            tracks_flags: list[TrackFlags],
            pose_format: Literal["matrix", "tquat"] = "matrix",
            device: torch.device = torch.device("cuda"),
        ) -> Tracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 matrix or 7-dim tquat vector track to world pose transformations, depending on 'pose_format', N_tracks x (N_poses_i x [4 x 4 | 7]) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_label_class: semantic class of each track, N_tracks [str]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - pose_format: the format of the poses (either 4x4 'matrix'es or 7-dim 'tquat' vectors)
            - device: the device to store the data on [torch.device]
            """

            # Convert to compressed pose representation
            tracks_poses_sizes = [len(track_poses) for track_poses in tracks_poses]

            assert len(tracks_id) == len(tracks_poses_sizes), "Tracks: inconsistent track_ids / pose"
            assert len(tracks_id) == len(tracks_flags), "Tracks: inconsistent track_ids / flags"
            assert tracks_poses_sizes == [len(track_timestamps_us) for track_timestamps_us in tracks_timestamps_us], (
                "Tracks: inconsistent pose / timestamp pairs"
            )

            assert all(
                [
                    (track_poses.shape[-2:] == (4, 4) if pose_format == "matrix" else track_poses.shape[-1:] == (7,))
                    and track_poses.dtype == np.float32
                    for track_poses in tracks_poses
                ]
            ), "Tracks: invalid poses inputs"
            assert all([len(track_poses) >= 2 for track_poses in tracks_poses]), (
                "Tracks: require at least two poses per track"
            )
            assert all(
                [
                    len(track_timestamps_us.shape) == 1 and track_timestamps_us.dtype == np.int64
                    for track_timestamps_us in tracks_timestamps_us
                ]
            ), "Tracks: invalid tracks_timestamps_us inputs"
            assert len(tracks_id) == len(tracks_label_class), "Tracks: inconsistent track_ids / label_class"

            # create packed pose / timestamp representation and upload to GPU
            # (also handle special case of empty / non-existing tracks)
            n_poses = [track_poses.shape[0] for track_poses in tracks_poses]
            start_idxs = np.cumsum([0] + n_poses, dtype=np.int32)[:-1]

            max_track_n_poses = max(n_poses) if len(n_poses) else 0

            tracks_packinfo = torch.tensor(
                np.stack((start_idxs, np.array(n_poses, dtype=np.int32)), axis=1), device=device
            )

            N_total_poses = tracks_packinfo[:, 1].sum()

            tracks_poses_array = np.empty(
                (N_total_poses, 4, 4) if pose_format == "matrix" else (N_total_poses, 7), dtype=np.float32
            )
            tracks_timestamps_array_us = np.empty((N_total_poses,), dtype=np.int64)
            if N_total_poses > 0:
                np.concatenate(tracks_poses, out=tracks_poses_array)
                np.concatenate(tracks_timestamps_us, out=tracks_timestamps_array_us)

            tracks_poses_se3: lt.SE3
            if pose_format == "matrix":
                tracks_poses_se3 = lt.SE3(se3_matrix_to_tquat(tracks_poses_array).to(device))
            else:
                tracks_poses_se3 = lt.SE3(torch.from_numpy(tracks_poses_array).to(device))

            return Tracks(
                tracks_data=TracksData(
                    tracks_id=tracks_id,
                    tracks_packinfo=tracks_packinfo,
                    tracks_poses=tracks_poses_se3,
                    tracks_timestamps_us=torch.from_numpy(tracks_timestamps_array_us).to(device),
                    tracks_flags=torch.tensor(
                        [track_flags.value for track_flags in tracks_flags], dtype=torch.int32, device=device
                    ),
                    max_track_n_poses=max_track_n_poses,
                    tracks_label_class=tracks_label_class,
                )
            )

    def get_mask_flags_all(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *all* flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).eq(flags.value)

    def get_mask_flags_any(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *any* flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).ne(0)

    def get_mask_flags_none(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *none* of the flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).eq(0)

    @property
    def device(self) -> torch.device:
        return self.tracks_poses.device

    def to_device(self, device: torch.device) -> Self:
        return cast(Self, self.__class__(tracks_data=self.tracks_data.to_device(device)))

    @property
    def n_tracks(self) -> int:
        return len(self.tracks_id)


@dataclass(kw_only=True, slots=True)
class CuboidTracks(Tracks):
    """Manages time-dependent cuboid tracks related GPU intersection operations for tracks

    This class uses the :class:`nre.datasets.tracks.CuboidTracks.Data` class for storing data.

    """

    @dataclass(slots=True)
    class Unpacked(dataclasses_json.DataClassJsonMixin):
        """'Unpacked' serialization format of the internal packed representation that is easier for human / external modification"""

        cuboids_dims: list[list] = field(default_factory=list)

    @staticmethod
    def encoder_cuboidtracks(input: CuboidTracksData) -> dict:
        # combine both the base part and local part
        return CuboidTracks.Unpacked(input.cuboids_dims.tolist()).to_dict()

    @staticmethod
    def decoder_cuboidtracks(input: dict) -> CuboidTracksData:
        return CuboidTracksData(
            cuboids_dims=torch.tensor(
                CuboidTracks.Unpacked.from_dict(input).cuboids_dims,
                dtype=torch.float32,
                device=torch.device("cuda"),
            ).reshape(-1, 3)
        )

    cuboidtracks_data: CuboidTracksData = field(
        metadata=dataclasses_json.config(encoder=encoder_cuboidtracks, decoder=decoder_cuboidtracks)
    )

    @property
    def cuboids_dims(self) -> torch.Tensor:
        return self.cuboidtracks_data.cuboids_dims

    class Factory:
        @staticmethod
        def empty(device: torch.device = torch.device("cuda")) -> CuboidTracks:
            """
            Constructs an empty cuboid tracks container
            """
            return CuboidTracks.Factory.from_numpy(
                tracks_id=[],
                tracks_poses=[],
                tracks_timestamps_us=[],
                tracks_label_class=[],
                tracks_flags=[],
                cuboids_dims=[],
                device=device,
            )

        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_label_class: list[str],
            tracks_flags: list[TrackFlags],
            cuboids_dims: list[np.ndarray],
            device: torch.device = torch.device("cuda"),
        ) -> CuboidTracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 track to world pose transformations, N_tracks x (N_poses_i x 4 x 4) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_label_class: semantic class of each track, N_tracks [str]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - cuboids_dims: cuboid x/y/z extents (in local track frame), N_tracks x 3 [float]
            - device: the device to store the data on [torch.device]
            """

            # construct base tracks part
            tracks = Tracks.Factory.from_numpy(
                tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, device=device
            )

            # construct cuboid-tracks-specific part
            if len(cuboids_dims_array := np.empty((len(tracks.tracks_id), 3), dtype=np.float32)):
                np.stack(cuboids_dims, out=cuboids_dims_array)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=tracks.tracks_id,
                    tracks_packinfo=tracks.tracks_packinfo,
                    tracks_poses=tracks.tracks_poses,
                    tracks_timestamps_us=tracks.tracks_timestamps_us,
                    tracks_flags=tracks.tracks_flags,
                    max_track_n_poses=tracks.max_track_n_poses,
                    tracks_label_class=tracks.tracks_label_class,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.tensor(cuboids_dims_array, dtype=torch.float32, device=device)
                ),
            )

        @staticmethod
        def from_pack(pack: CuboidTracksDataPack) -> CuboidTracks:
            """
            Constructs cuboid tracks from packed data components
            """

            return CuboidTracks(tracks_data=pack.tracks_data, cuboidtracks_data=pack.cuboidtracks_data)

    class Ops:
        """
        Operations on CuboidTracks returning new CuboidTracks instances.
        Compared to classmethods, this forces the semantics to be NOT inplace.
        """

        @staticmethod
        def clone(cuboid_tracks: CuboidTracks):
            """Clones the cuboid tracks instance"""

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id[:],
                    tracks_packinfo=cuboid_tracks.tracks_packinfo.clone(),
                    tracks_poses=lt.SE3(cuboid_tracks.tracks_poses.data.clone()),
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us.clone(),
                    tracks_flags=cuboid_tracks.tracks_flags.clone(),
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class[:],
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims.clone(),
                ),
            )

        @staticmethod
        def transform_with_frame_conversion(
            cuboid_tracks: CuboidTracks,
            source_to_target_frame_conversion: FrameConversion | None,
            T_local_to_common_target: np.ndarray | torch.Tensor | None,
        ) -> CuboidTracks:
            """Transforms the cuboid tracks from their current frame to a target frame"""

            if not cuboid_tracks.n_tracks:
                # no need to transform empty tracks
                return cuboid_tracks

            # Convert from 7d to SE3 matrix and apply transformation
            tracks_poses = cuboid_tracks.tracks_poses.matrix()

            if isinstance(T_local_to_common_target, np.ndarray):
                T_local_to_common_target = torch.from_numpy(T_local_to_common_target)

            if T_local_to_common_target is not None:
                tracks_poses = T_local_to_common_target.to(tracks_poses) @ tracks_poses

            cuboids_dims = cuboid_tracks.cuboids_dims
            if source_to_target_frame_conversion is not None:
                T_np, S_np = source_to_target_frame_conversion.get_transformation_matrices()
                T = torch.from_numpy(T_np).to(tracks_poses)
                S = torch.from_numpy(S_np).to(tracks_poses)
                tracks_poses = T @ tracks_poses @ S

                # Apply transform's scale to the cuboid dimensions
                cuboids_dims = cuboids_dims / torch.diag(S)[None, :3]

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    tracks_poses=lt.SE3(se3_matrix_to_tquat(tracks_poses)),
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboids_dims,
                ),
            )

        @staticmethod
        def transform_with_delta_poses(
            cuboid_tracks: CuboidTracks,
            delta_poses: lt.SE3,
            left_multiply: bool = True,
        ) -> CuboidTracks:
            """Transforms the cuboid tracks according to the delta poses"""

            if left_multiply:
                tracks_poses = delta_poses * cuboid_tracks.tracks_poses
            else:
                tracks_poses = cuboid_tracks.tracks_poses * delta_poses

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    tracks_poses=tracks_poses,
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

        @staticmethod
        def freeze(
            cuboid_tracks: CuboidTracks,
            min_timestamps_us: torch.Tensor,
            max_timestamps_us: torch.Tensor,
            tracks_poses_start: lt.SE3,
            tracks_poses_end: lt.SE3 | None = None,
        ) -> CuboidTracks:
            """
            Freeze each of the tracks to one pose (optionally two if `tracks_poses_end` is set).

            Inputs:
            - cuboid_tracks: the input cuboid tracks
            - min_timestamps_us: Tensor (N_tracks, ) [int64] containing the first timestamp for each track
            - max_timestamps_us: Tensor (N_tracks, ) [int64] containing the second (and last) timestamp for each track
            - tracks_poses_start: Start SE3 poses to freeze the tracks to
            - tracks_poses_end: Optional end SE3 poses to freeze the tracks to

            Returns:
            - the frozen cuboid tracks
            """

            assert min_timestamps_us.size(0) == cuboid_tracks.n_tracks, "CuboidTracks: invalid min_timestamps_us size"
            assert max_timestamps_us.size(0) == cuboid_tracks.n_tracks, "CuboidTracks: invalid max_timestamps_us size"
            assert tracks_poses_start.shape[0] == cuboid_tracks.n_tracks, (
                "CuboidTracks: invalid tracks_poses_start size"
            )
            if tracks_poses_end is not None:
                assert tracks_poses_end.shape[0] == cuboid_tracks.n_tracks, (
                    "CuboidTracks: invalid tracks_poses_end size"
                )
            else:
                tracks_poses_end = tracks_poses_start

            n_poses = torch.full((cuboid_tracks.n_tracks,), 2, dtype=torch.int32, device=cuboid_tracks.device)
            tracks_timestamps_us = torch.stack((min_timestamps_us, max_timestamps_us), dim=1).view(-1)
            tracks_poses = lt.stack([tracks_poses_start, tracks_poses_end], dim=1).view((-1,))

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=get_pack_info_from_n(n_poses),
                    tracks_poses=tracks_poses,
                    tracks_timestamps_us=tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=2,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

        @staticmethod
        def concatenate(cuboid_tracks_list: Iterable[CuboidTracks]) -> CuboidTracks:
            """
            Concatenates a list of CuboidTracks instances into a new one
            """

            tracks_id_list = [cuboid_tracks.tracks_id for cuboid_tracks in cuboid_tracks_list]
            tracks_length_list = [cuboid_tracks.tracks_packinfo[:, 1] for cuboid_tracks in cuboid_tracks_list]
            tracks_poses_list = [cuboid_tracks.tracks_poses for cuboid_tracks in cuboid_tracks_list]
            tracks_timestamps_us_list = [cuboid_tracks.tracks_timestamps_us for cuboid_tracks in cuboid_tracks_list]
            tracks_flags_list = [cuboid_tracks.tracks_flags for cuboid_tracks in cuboid_tracks_list]
            cuboids_dims_list = [cuboid_tracks.cuboids_dims for cuboid_tracks in cuboid_tracks_list]
            max_track_n_poses = max([cuboid_tracks.max_track_n_poses for cuboid_tracks in cuboid_tracks_list])
            tracks_label_class_list = [cuboid_tracks.tracks_label_class for cuboid_tracks in cuboid_tracks_list]

            tracks_packinfo = get_pack_info_from_n(torch.cat(tracks_length_list, dim=0)).int()
            tracks_id: list[str] = sum(tracks_id_list, [])

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=tracks_id,
                    tracks_packinfo=tracks_packinfo,
                    tracks_poses=lt.cat(tracks_poses_list, dim=0),
                    tracks_timestamps_us=torch.cat(tracks_timestamps_us_list, dim=0),
                    tracks_flags=torch.cat(tracks_flags_list, dim=0),
                    max_track_n_poses=max_track_n_poses,
                    tracks_label_class=sum(tracks_label_class_list, []),
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.cat(cuboids_dims_list, dim=0),
                ),
            )

        @staticmethod
        def subset_from_indices(cuboid_tracks: CuboidTracks, indices: list[int] | torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches argument indices
            This operation keeps the gradient flow.
            """
            if isinstance(indices, list):
                indices_list = indices
                indices_tensor = torch.tensor(indices, device=cuboid_tracks.device, dtype=torch.long)

            else:
                indices_list = indices.cpu().numpy().tolist()
                indices_tensor = indices

            track_starts_counts = cuboid_tracks.tracks_packinfo[indices_tensor]
            track_starts, track_counts = track_starts_counts[:, 0], track_starts_counts[:, 1]
            packed_attr_ind = (
                linstep_interleave(track_starts, track_counts, 1).values
                if len(track_starts)
                else torch.empty(0, dtype=torch.long, device=cuboid_tracks.device)
            )
            new_tracks_packinfo = get_pack_info_from_n(track_counts)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=[cuboid_tracks.tracks_id[i] for i in indices_list],
                    tracks_packinfo=new_tracks_packinfo,
                    tracks_poses=cuboid_tracks.tracks_poses[packed_attr_ind],
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us[packed_attr_ind],
                    tracks_flags=cuboid_tracks.tracks_flags[indices_tensor],
                    max_track_n_poses=cast(int, track_counts.max().item()) if len(track_counts) else 0,
                    tracks_label_class=[cuboid_tracks.tracks_label_class[i] for i in indices],
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims[indices_tensor],
                ),
            )

        @staticmethod
        def subset_from_tracks_id(cuboid_tracks: CuboidTracks, tracks_id: list[str]) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches argument tracks_id
            This operation keeps the gradient flow.
            """
            indices = [cuboid_tracks.tracks_id.index(tid) for tid in tracks_id]
            return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, indices)

        @staticmethod
        def subset_from_mask(cuboid_tracks: CuboidTracks, mask: torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches the mask
            This operation keeps the gradient flow.
            """
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, indices)

        @staticmethod
        def clean_track_ids(cuboid_tracks: CuboidTracks, cleaner_fn: Callable[[str], str]) -> CuboidTracks:
            """
            Cleans the track IDs using the provided cleaner function.
            Returns a new CuboidTracks with cleaned track IDs.

            Args:
                cuboid_tracks: The cuboid tracks to clean
                cleaner_fn: A function that takes a track ID string and returns a cleaned version

            Returns:
                A new CuboidTracks with cleaned track IDs
            """
            cleaned_tracks_id = [cleaner_fn(tid) for tid in cuboid_tracks.tracks_id]

            return CuboidTracks(
                tracks_data=TracksData(
                    tracks_id=cleaned_tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    tracks_poses=cuboid_tracks.tracks_poses,
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                ),
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

    def to_device(self, device):
        return CuboidTracks(
            tracks_data=self.tracks_data.to_device(device),
            cuboidtracks_data=self.cuboidtracks_data.to_device(device),
        )

    @dataclass
    class RayIntersectionResult:
        intersections_cnt: torch.Tensor  # number of intersections of each ray, N_rays [int]
        intersections_tracks_idx: (
            torch.Tensor
        )  # for each intersection, the index of the intersected track, N_rays x max_intersections_per_ray [int]
        intersections_ts: Optional[
            torch.Tensor
        ]  # for each intersection, the start / end distances of each intersection, N_rays x max_intersections_per_ray x 2 [float]

    def ray_intersection(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
        max_intersections_per_ray: int = 32,
        with_intersections_ts: bool = True,
    ) -> CuboidTracks.RayIntersectionResult:
        """
        Computes the intersection of all cuboid tracks with timed world rays

        Inputs:
        - rays_o: ray origins / 3d world positions, N_rays x 3 [float]
        - rays_d: normalized 3d world directions, N_rays x 3 [float]
        - rays_timestamps_us: per ray timestamp, N_rays [int64]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]
        - max_intersections_per_ray: upper limit of intersections to return [int]
        - with_intersections_ts: if True, intersections_ts will be returned, otherwise only intersections_cnt and intersections_tracks_idx will be returned [bool]

        Returns:
        - RayIntersectionResult: result of the ray intersection
        """

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        intersection_result = vren.ray_cuboidtracks_intersection(
            rays_o,
            rays_d,
            rays_timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            cuboids_dims,
            self.max_track_n_poses,
            max_intersections_per_ray,
            with_intersections_ts,
        )

        return CuboidTracks.RayIntersectionResult(
            intersections_cnt=intersection_result[0],
            intersections_tracks_idx=intersection_result[1],
            intersections_ts=intersection_result[2] if with_intersections_ts else None,
        )

    def point_intersection_interpolate_pose(
        self,
        points: torch.Tensor,
        points_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
    ) -> tuple[lt.SE3, torch.Tensor]:
        """
        For each point, returns the interpolated pose of the tracks that it is inside, as well as the track idx.

        Inputs:
        - points: 3D points to check for inside check, [..., 3] [float]
        - points_timestamps_us: per point timestamp, [...] [int64]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]

        Returns:
        - interpolated_poses: for each point, the pose of the cuboid it is inside, [...] [SE3]
        - interpolated_tracks_idx: for each point, the index of the intersected track (-1 if no intersection), [...] [int]
        """

        data_shape = points.shape[:-1]
        points = points.reshape(-1, 3).contiguous()
        points_timestamps_us = points_timestamps_us.reshape(-1).contiguous()

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        interpolated_tracks_pose_data, interpolated_tracks_idx = vren.point_cuboidtracks_intersection_interpolate_pose(
            points,
            points_timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            cuboids_dims,
            self.max_track_n_poses,
        )
        interpolated_poses = lt.SE3(
            interpolated_tracks_pose_data.reshape(data_shape + (interpolated_tracks_pose_data.shape[-1],))
        )
        return interpolated_poses, interpolated_tracks_idx.reshape(data_shape)

    def frame_poses_interpolation(
        self,
        frame_timestamps_us: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the interpolated poses for a given frame that has been captured between the given timestamps pair

        Inputs:
        - frame_timestamps_us: frame start and end timestamps between which the frame has been captured , 2 [int64]

        Returns:
        - num_valid_tracks : number of track having a valid poses within the frame capture time, 1 [int]
        - track_ids : tracks mapping idx and unique ids corresponding to the return poses, num_valid_tracks [intx2]
        - start_poses : interpolated poses at the frame start timestamp, num_valid_tracks x 7 [float]
        - end_poses : interpolated poses at the frame end timestamp, num_valid_tracks x 7 [float]
        """

        return vren.cuboidtracks_frame_poses_interpolation(
            frame_timestamps_us, self.tracks_packinfo, self.tracks_poses.data, self.tracks_timestamps_us
        )

    def interpolate_tracks_poses(
        self,
        timestamps_us: torch.Tensor,
        tracks_idx: torch.Tensor,
    ) -> lt.SE3:
        return self.interpolate_tracks_poses_ex(timestamps_us, tracks_idx)[0]

    def interpolate_tracks_poses_ex(
        self,
        timestamps_us: torch.Tensor,
        tracks_idx: torch.Tensor,
    ) -> Tuple[lt.SE3, torch.Tensor]:
        """
        Compute the interpolated pose of the tracks at the given timestamps.
        This will not check if the timestamps are within the range of the tracks!

        Inputs:
        - timestamps_us: timestamps to interpolate the pose at, N_data [int64]
        - tracks_idx: indices of the tracks to interpolate the pose for, N_data [int]

        Returns:
        - interpolated_pose: interpolated pose of the tracks at the given timestamps, N_data [SE3]
        - interpolated_mask: mask of the tracks that are valid interpolations, N_data [bool]
        """

        tidx_right = packed_ops.packed_searchsorted_indexed_vals(
            self.tracks_timestamps_us,
            self.tracks_packinfo,
            timestamps_us,
            tracks_idx,
        )
        # Make sure right index is locally [1, N] indices
        # (usually this won't happen since we already filter in the filtering stage)
        packinfo_start = self.tracks_packinfo[tracks_idx, 0]
        packinfo_end = packinfo_start + self.tracks_packinfo[tracks_idx, 1]
        tidx_right[tidx_right == packinfo_start] += 1
        tidx_right[tidx_right == packinfo_end] -= 1

        tidx_left = tidx_right - 1
        time_diff = (self.tracks_timestamps_us[tidx_right] - self.tracks_timestamps_us[tidx_left]).to(torch.float32)
        alpha = (timestamps_us - self.tracks_timestamps_us[tidx_left]).to(torch.float32) / time_diff
        interpolated_mask = torch.logical_and(time_diff != 0, torch.logical_and(alpha >= 0, alpha <= 1))
        alpha = torch.where(interpolated_mask, alpha, torch.zeros_like(alpha))

        pose_start = self.tracks_poses[tidx_left]
        pose_end = self.tracks_poses[tidx_right]

        # Perform manifold-product interpolation (to be consistent with transform-filter)
        R_start = lt.SO3.InitFromVec(pose_start.vec()[:, 3:])
        R_end = lt.SO3.InitFromVec(pose_end.vec()[:, 3:])
        t_start, t_end = pose_start.translation(), pose_end.translation()

        R_alpha = R_start * lt.SO3.exp(alpha[:, None] * (R_start.inv() * R_end).log())
        t_alpha = t_start + alpha[:, None] * (t_end - t_start)

        interpolated_pose = lt.SE3.InitFromVec(torch.cat([t_alpha[:, :3], R_alpha.vec()], dim=1))

        return (interpolated_pose, interpolated_mask)

    @torch.autocast("cuda", enabled=False)
    def warp_world_points_to_timestamps(
        self,
        world_points: torch.Tensor,
        points_timestamps_us: torch.Tensor,
        target_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Warp the world points to the target timestamps using the cuboid tracks.

        Inputs:
        - world_points: The world points to warp. [..., 3] (float32)
        - points_timestamps_us: The timestamps of the world points. [...,] (int64)
        - target_timestamps_us: The target timestamps to warp to. [...,] (int64)

        Returns:
        - The warped world points. [..., 3] (float32)
        """
        data_shape = world_points.shape[:-1]

        world_points = world_points.reshape(-1, 3)
        points_timestamps_us = points_timestamps_us.reshape(-1)
        target_timestamps_us = target_timestamps_us.reshape(-1)

        interpolated_poses, interpolated_tracks_idx = self.point_intersection_interpolate_pose(
            points=world_points,
            points_timestamps_us=points_timestamps_us,
            cuboids_dims_padding=cuboids_dims_padding,
        )

        # Subselect dynamic points to save computation.
        new_world_points = world_points.clone()
        dynamic_mask = torch.where(interpolated_tracks_idx != -1)[0]
        if len(dynamic_mask) > 0:
            interpolated_poses = interpolated_poses[dynamic_mask]
            current_pose = self.interpolate_tracks_poses(
                timestamps_us=target_timestamps_us[dynamic_mask],
                tracks_idx=interpolated_tracks_idx[dynamic_mask],
            )
            xyz_rel_pose = current_pose * interpolated_poses.inv()
            new_world_points[dynamic_mask] = xyz_rel_pose * new_world_points[dynamic_mask]  # type: ignore

        return new_world_points.reshape(data_shape + (3,))

