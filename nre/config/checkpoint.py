# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Literal

from nre.config.base_schema import BaseConfigSchema, Field


class ArtifactConfig(BaseConfigSchema):
    enabled: bool = Field(default=False, description="Global enable/disable for exporting the output artifact.")

    class ArtifactCheckpointConfig(BaseConfigSchema):
        enabled: bool = Field(default=True, description="Enable/disable saving the checkpoint file into the artifact.")
        fp16: bool = Field(
            default=False,
            description="Cast checkpoint tensors from fp32 to fp16 to reduce storage. Not suitable for checkpoints used to resume training.",
        )
        strip_optimizer: bool = Field(
            default=True,
            description="Remove optimizer state, lr schedulers, loops, and callbacks from the checkpoint to reduce storage. "
            "Stripped checkpoints cannot be used with resume=; set to false if the artifact checkpoint should be resumable.",
        )

    checkpoint: ArtifactCheckpointConfig = Field(
        default_factory=ArtifactCheckpointConfig, description="Artifact export settings for the checkpoint file."
    )

    class MeshConfig(BaseConfigSchema):
        class GenericMeshConfig(BaseConfigSchema):
            """Settings of the Poisson meshing pipeline that computes a generic background mesh
            from the static point cloud"""

            enabled: bool = Field(
                default=False,
                description="Enable/disable running the generic Poisson mesh reconstruction and saving the mesh file(s) into the artifact.",
            )
            formats: List[Literal["ply", "usd"]] = Field(
                default=["ply", "usd"], description="Output formats. Any subset of ['ply', 'usd']"
            )
            step_frame: int = Field(gt=0, default=1, description="Frame step for point cloud generation")
            lidar_ids: List[str] | None = Field(
                default=None, description="Lidar sensors to use for point cloud generation, first LiDAR if None"
            )
            camera_ids: List[str] | None = Field(
                default=None,
                description="Camera sensors to use for point cloud generation, first point-cloud-capable camera if None",
            )
            max_num_points: int | None = Field(
                default=None,
                gt=0,
                description="Optional limit of the number of points fed to meshing, based on a random subsampling",
            )
            n_neighbors: int = Field(
                gt=0, default=200, description="Point cloud k-NNs used to compute normals prior to meshing"
            )
            trim_distance: float = Field(
                gt=0.0, default=0.225, description="Vertex-to-2nd-nearest-point distance threshold for mesh trimming"
            )
            smooth: bool = Field(default=False, description="Whether to apply smoothing to the Poisson mesh")
            apply_road_segmentation: bool = Field(
                default=False, description="Transfer point cloud road/non-road labels to the mesh"
            )
            export_disjoint_meshes: bool = Field(
                default=False, description="Export mesh road/non-road mesh segments into separate files"
            )
            mesh_path: str | None = Field(
                default=None,
                description="Optional path to an existing mesh PLY file to use instead of computing a mesh",
            )

        generic: GenericMeshConfig = Field(
            default_factory=GenericMeshConfig, description="Artifact export settings for the generic mesh."
        )

        class GroundMeshConfig(BaseConfigSchema):
            """Settings of the Delaunay meshing pipeline that aims to reconstruct a drivable ground without holes"""

            enabled: bool = Field(
                default=False,
                description="Enable/disable running the ground mesh reconstruction and saving the mesh file(s) into the artifact.",
            )
            formats: List[Literal["ply", "usd"]] = Field(
                default=["ply", "usd"], description="Output formats. Any subset of ['ply', 'usd']"
            )
            step_frame: int = Field(gt=0, default=1, description="Frame step for point cloud generation")
            lidar_ids: List[str] | None = Field(
                default=None, description="Lidar sensors to use for point cloud generation, first LiDAR if None"
            )
            camera_ids: List[str] | None = Field(
                default=None,
                description="Camera sensors to use for point cloud generation, first point-cloud-capable camera if None",
            )
            min_ray_length: float = Field(
                gt=0.0, default=0.1, description="Minimum length of rays to filter unwanted points near the sensor (>0)"
            )
            ground_compat_max_distance: float = Field(
                gt=0.0,
                default=0.5,
                description="Max distance between valid ground plane hypotheses and the ground control point under the ego vehicle",
            )
            ground_compat_max_angle_deg: float = Field(
                gt=0.0,
                le=90.0,
                default=60.0,
                description="Max angle between valid ground plane hypotheses and the Z-direction of the world",
            )
            num_plane_hypotheses: int = Field(
                gt=0, default=100, description="Number of plane hypotheses to evaluate (>0)"
            )
            plane_max_distance: float = Field(
                gt=0.0,
                default=0.3,
                description="Max distance of a point from a plane hypothesis to be considered inlier",
            )
            plane_max_angle_deg: float = Field(
                gt=0.0,
                le=90.0,
                default=30,
                description="Max angle between a point normal and the normal of a plane hypothesis for the point to be considered inlier",
            )
            plane_min_eval_points: int = Field(
                gt=0,
                default=1000,
                description="Min number of points to evaluate plane hypotheses on, skipping frame if not reached",
            )
            plane_max_eval_points: int = Field(
                gt=0,
                default=10000,
                description="Max number of points to evaluate plane hypotheses on, subsampling if exceeded",
            )
            plane_min_inliers: int = Field(
                gt=0,
                default=1000,
                description="Min number of inliers for accepting the dominant plane, skipping frame if not reached",
            )
            enable_plane_extension: bool = Field(
                default=False,
                description="Find plane inliers in the full point cloud per LiDAR spin, not just in points presegmented as road",
            )  # False to be conservative
            voxel_size: float = Field(
                gt=0,
                default=0.1,
                description="Voxel size (in world units) for point cloud downsampling to be applied before meshing",
            )
            min_points_per_voxel: int = Field(gt=0, default=1, description="Minimum number of points per voxel")
            smoothing_passes: int = Field(gt=0, default=10, description="Number of mesh smoothing passes to apply")
            verbose: bool = Field(default=False, description="Verbose mode")
            colored: bool = Field(
                default=False,
                description="Color mesh vertices by projecting camera RGB onto LiDAR points before meshing",
            )

        ground: GroundMeshConfig = Field(
            default_factory=GroundMeshConfig, description="Artifact export settings for the ground mesh."
        )

    mesh: MeshConfig = Field(default_factory=MeshConfig, description="Artifact export settings for the mesh(es).")

    class NRendConfig(BaseConfigSchema):
        enabled: bool = Field(
            default=False,
            description="Enable/disable saving the nrend checkpoint file into the artifact.",
        )
        use_legacy_model_dict: bool = Field(
            default=False,
            description="Export the legacy model dictionary for OV rendering support. Needed for backward compatibility.",
        )

    nrend: NRendConfig = Field(default_factory=NRendConfig, description="Artifact export settings for NRend.")

    class ParsedConfigConfig(BaseConfigSchema):
        enabled: bool = Field(default=True, description="Enable/disable saving the parsed config into the artifact.")

    parsed_config: ParsedConfigConfig = Field(
        default_factory=ParsedConfigConfig, description="Artifact export settings for the parsed config."
    )

    class RigTrajectoriesConfig(BaseConfigSchema):
        enabled: bool = Field(default=True, description="Enable/disable saving the rig trajectories into the artifact.")
        add_default_cameras: bool = Field(
            default=False,
            description="Option to add extra cameras with different camera models (e.g. OpenCV pinhole and fisheye) using some default intrinsics to the rig in the exported USD for the purpose of testing these specific camera models in downstream software, such as OV.",
        )
        match_datasource: bool = Field(
            default=False,
            description="Used only in NRM artifact export. If True, the rig trajectories will be matched to the original ncore data (no frame subsampling, with all existing cameras). Useful in //:run render target that wants to replicate original views.",
        )

    rig_trajectories: RigTrajectoriesConfig = Field(
        default_factory=RigTrajectoriesConfig, description="Artifact export settings for the rig trajectories."
    )

    class SequenceTracksConfig(BaseConfigSchema):
        enabled: bool = Field(default=True, description="Enable/disable saving the sequence tracks into the artifact.")
        controllable_only: bool = Field(default=False, description="Only export controllable sequence tracks.")

    sequence_tracks: SequenceTracksConfig = Field(
        default_factory=SequenceTracksConfig, description="Artifact export settings for the sequence tracks."
    )


class CheckpointConfig(BaseConfigSchema):
    every_n_train_steps: int | None
    save_top_k: int
    monitor: str | None
    mode: Literal["min", "max"]
    save_on_train_epoch_end: bool
    save_on_preemption: bool = Field(
        default=False, description="Save checkpoint on SLURM preemption for single-scene optimization"
    )
    fp16: bool = Field(
        default=False,
        description="Cast training checkpoint (.ckpt) tensors from fp32 to fp16 to reduce storage. "
        "fp16 checkpoints are automatically upcast to fp32 on load.",
    )
    strip_optimizer: bool = Field(
        default=False,
        description="Remove optimizer state, lr schedulers, loops, and callbacks from the training checkpoint (.ckpt). "
        "Stripped checkpoints cannot be used with resume=.",
    )
    artifact: ArtifactConfig
