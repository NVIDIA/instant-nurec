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

import logging

from typing import Any, Callable, ClassVar, Literal, Self, cast

import torch

from omegaconf import DictConfig
from torch import nn

from nre.difix.model import DifixModel
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.nrm.primitives.base import BaseGaussiansNRMPrimitive
from nre.utils.batch import DataAndRenderingBatch, RenderingData
from nre.utils.geometry import quat_mult_xyzw, so3_matrix_to_quat
from nre.utils.misc import unpack_optional
from nre.utils.types import Checkpoint, GaussiansRenderReturn, RayFlags, RigTrajectories


logger = logging.getLogger(__name__)


class ModulatedLinearLayer(nn.Module):
    """
    Modulated linear layer for direction-input, token-conditioned sky color rendering.
    Proposed in STORM [Yang et al. 2024]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, condition_dim: int = 768, out_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))
        self.condition_mapping = nn.Linear(condition_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)

    @classmethod
    def modulate(cls, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def get_checkpoint(self) -> Checkpoint:
        return {
            "state_dict": self.state_dict(),
            "constructor_args": {
                "input_dim": self.linear.in_features,
                "hidden_dim": self.linear.out_features,
                "condition_dim": self.condition_mapping.in_features,
                "out_dim": self.output.out_features,
            },
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint: Checkpoint, device: torch.device) -> Self:
        instance = cls(**checkpoint["constructor_args"])
        instance.load_state_dict(checkpoint["state_dict"])
        instance.to(device)
        return instance

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ..., input_dim) tensor
            c: (B, condition_dim) tensor
        Returns:
            (B, ..., out_dim) tensor
        """
        x = self.linear(x)
        c = self.condition_mapping(c)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)  # 2x (B, hidden_dim)
        x_shape = x.shape
        x = self.modulate(self.norm(x.reshape(x_shape[0], -1, x.shape[-1])), shift, scale)
        x = self.output(x)
        x = x.reshape(*x_shape[:-1], -1)
        return x


class CelsiusNRMPrimitive(BaseGaussiansNRMPrimitive):
    """
    Celsius NRM primitive that contains the following attribute:
        - positions:            Positions of the 3D Gaussians (x, y, z)                            [n_gaussians, 3]
        - rotations:            Rotation of each Gaussian represented as a unit quaternion         [n_gaussians, 4]
                                Note that this follows nrend's convention of wxyz format.
        - scales:               Anisotropic scale of each Gaussian                                 [n_gaussians, 3]
        - densities:            Density of each Gaussian                                           [n_gaussians, 1]
        - rgb:                  RGB color of each Gaussian                                         [n_gaussians, 3]
        - timestamps_us:        Timestamps of each Gaussian that originate from                    [n_gaussians, 1]
        - forward_speed_mps:    Velocity of each Gaussian going forward, unit is m/s               [n_gaussians, 3/6]
                                if None, the Gaussians are considered static.
        - falloff_sigma:        Sigma for the opacity falloff function, unit is s                  [n_gaussians, 1/2]
                                if None, the Gaussians will not fall off.
                                if dimension is 2, the first dimension is for forward falloff,
                                the second dimension is for backward falloff (in primitive merging strategy).
        - dynamic_bbox_mask:    Mask indicating whether this Gaussian belongs to a dynamic
                                bounding box -- mainly used in "static" scenarios to mask out
                                dynamic objects.                                                   [n_gaussians, 1]
        - road_mask:            Road mask indicating whether this Gaussian represents the road.    [n_gaussians, 1]
        - sky_mask:             Sky mask indicating whether this Gaussian represents the sky       [n_gaussians, 1]
                                if None, no Gaussians will be masked out if we enable sky mask.
        - sky_token:            The sky token that represents the sky color                        [D,]
                                if None, the background sky will not be rendered.
        - sky_head:             The head to predict the sky color from the sky token and ray direction
        - sky_rotation:         Rotation applied to the sky                                        [3, 3]
        - solid_background_color: The solid background color                                       [3,]
                                Setting this will override the color predicted by the sky head.
        - affine_matrix:        The affine transform matrix                                        [n_cameras, 3, 3]
                                if None, no per-camera ISP transform will be applied.
        - affine_bias:          The bias part                                                      [n_cameras, 3]
                                if None, no per-camera ISP transform will be applied.
    """

    # Exponential factor for generalized gaussian
    FALLOFF_GAUSSIAN_EXP = 4.0

    # Time difference threshold for static clipping
    STATIC_CLIP_TIME_DIFF_S = 0.05

    # List of attributes that have size of [n_gaussians, ...]
    GAUSSIAN_ATTRIBUTES_NAMES: ClassVar[list[str]] = [
        "positions",
        "rotations",
        "scales",
        "densities",
        "rgb",
        "timestamps_us",
        "forward_speed_mps",
        "falloff_sigma",
        "dynamic_bbox_mask",
        "sky_mask",
        "road_mask",
    ]

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    rgb: torch.Tensor
    timestamps_us: torch.Tensor
    forward_speed_mps: torch.Tensor | None
    falloff_sigma: torch.Tensor | None
    dynamic_bbox_mask: torch.Tensor | None
    sky_mask: torch.Tensor | None
    road_mask: torch.Tensor | None

    # List of other attributes related to background and cameras
    OTHER_ATTRIBUTES_NAMES: ClassVar[list[str]] = [
        "sky_token",
        "sky_head",
        "sky_rotation",
        "sky_mask_enabled",
        "solid_background_color",
        "affine_matrix",
        "affine_bias",
    ]

    # Background attributes
    sky_token: torch.Tensor | None
    sky_head: ModulatedLinearLayer | None
    sky_rotation: torch.Tensor | None
    sky_mask_enabled: bool
    solid_background_color: torch.Tensor | None

    # Per-camera ISP transform attributes
    affine_matrix: torch.Tensor | None
    affine_bias: torch.Tensor | None

    def __init__(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        rgb: torch.Tensor,
        timestamps_us: torch.Tensor,
        forward_speed_mps: torch.Tensor | None,
        falloff_sigma: torch.Tensor | None,
        dynamic_bbox_mask: torch.Tensor | None,
        road_mask: torch.Tensor | None,
        sky_mask: torch.Tensor | None,
        sky_token: torch.Tensor | None,
        sky_head: ModulatedLinearLayer | None,
        sky_rotation: torch.Tensor | None,
        affine_matrix: torch.Tensor | None,
        affine_bias: torch.Tensor | None,
        gaussians_renderer: BaseGaussianRenderer | None,
        checkpointing: Literal["render", "all", "none"] = "render",
        difix_model: DifixModel | None = None,
        sky_mask_enabled: bool = False,
        solid_background_color: torch.Tensor | None = None,
    ):
        share_gaussian_parameters: bool = (forward_speed_mps is None) and (dynamic_bbox_mask is None)
        super().__init__(
            gaussians_renderer=gaussians_renderer,
            checkpointing=checkpointing,
            shared_gaussian_parameters=share_gaussian_parameters,
            difix_model=difix_model,
        )

        # Assign attributes
        local_vars = locals()
        for attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES + self.OTHER_ATTRIBUTES_NAMES:
            setattr(self, attr_name, local_vars[attr_name])

        self._post_init_validation()

    def _post_init_validation(self):
        n_gaussians = self.positions.shape[0]
        assert self.rotations.shape == (n_gaussians, 4), "Rotations must have shape (n_gaussians, 4)"
        assert self.scales.shape == (n_gaussians, 3), "Scales must have shape (n_gaussians, 3)"
        assert self.densities.shape == (n_gaussians, 1), "Densities must have shape (n_gaussians, 1)"
        assert self.rgb.shape == (n_gaussians, 3), "RGB must have shape (n_gaussians, 3)"
        assert self.timestamps_us.shape == (n_gaussians, 1), "Timestamps must have shape (n_gaussians, 1)"
        if self.dynamic_bbox_mask is not None:
            assert self.dynamic_bbox_mask.shape == (
                n_gaussians,
                1,
            ), "Dynamic bbox mask must have shape (n_gaussians, 1)"
            assert self.forward_speed_mps is None, "Forward speed must be None if dynamic bbox mask is provided"
        if self.forward_speed_mps is not None:
            assert self.forward_speed_mps.shape == (n_gaussians, 3) or self.forward_speed_mps.shape == (
                n_gaussians,
                6,
            ), "Forward speed must have shape (n_gaussians, 3) or (n_gaussians, 6)"
        if self.falloff_sigma is not None:
            assert self.falloff_sigma.shape == (n_gaussians, 1) or self.falloff_sigma.shape == (
                n_gaussians,
                2,
            ), "Falloff sigma must have shape (n_gaussians, 1) or (n_gaussians, 2)"
        if self.road_mask is not None:
            assert self.road_mask.shape == (n_gaussians, 1), "Road mask must have shape (n_gaussians, 1)"
        if self.sky_mask is not None:
            assert self.sky_mask.shape == (n_gaussians, 1), "Sky mask must have shape (n_gaussians, 1)"
        if self.sky_rotation is not None:
            assert self.sky_rotation.shape == (3, 3), "Sky rotation must have shape (3, 3)"
        if self.solid_background_color is not None:
            assert self.solid_background_color.shape == (3,), "Solid background color must have shape (3,)"
        if self.affine_matrix is not None:
            assert self.affine_bias is not None, "Affine bias must be provided if affine matrix is provided"
            n_cameras = self.affine_matrix.shape[0]
            assert self.affine_bias.shape == (n_cameras, 3), "Affine bias must have shape (n_cameras, 3)"
            assert self.affine_matrix.shape == (n_cameras, 3, 3), "Affine matrix must have shape (n_cameras, 3, 3)"

    def device(self) -> torch.device:
        return self.positions.device

    def memory_bytes(self) -> int:
        # Iterate dataclass attributes and find all tensors, then sum up the memory usage
        memory_bytes = 0
        for attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES:
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                memory_bytes += attr.numel() * attr.element_size()
        return memory_bytes

    @staticmethod
    def from_positions_and_fixed_scale(
        positions: torch.Tensor, scale: float, velocity_dim: int = 0
    ) -> CelsiusNRMPrimitive:
        """
        Create a celsius primitive with fixed scale. Used to replace the original "random" factory function and for
        rendering from lidar points
        """
        device = positions.device
        n_gaussians = positions.shape[0]
        rotations = torch.zeros(n_gaussians, 4, device=device)
        rotations[:, 0] = 1.0  # identity in wxyz format

        return CelsiusNRMPrimitive(
            positions=positions,
            rotations=rotations,
            scales=torch.full((n_gaussians, 3), scale, device=device),
            densities=torch.ones(n_gaussians, 1, device=device),
            rgb=torch.ones(n_gaussians, 3, device=device),
            timestamps_us=torch.full((n_gaussians, 1), 1000000, device=device),
            forward_speed_mps=torch.zeros(n_gaussians, velocity_dim, device=device) if velocity_dim > 0 else None,
            falloff_sigma=None,
            dynamic_bbox_mask=None,
            road_mask=None,
            sky_mask=None,
            sky_token=None,
            sky_head=None,
            sky_rotation=None,
            affine_matrix=None,
            affine_bias=None,
            gaussians_renderer=None,
            checkpointing="render",
        )

    def state_dict_and_config(self) -> tuple[dict[str, Any], DictConfig]:
        # Here to support nrend (however only gaussians themselves will be rendered).
        # TODO [JH]: As this is rendered via render_nrend_sensor_rays() as opposed to forward(),
        # we need to inv-activate the values. Currently we don't do this since it's just a PH for initialize the renderer.
        n_gaussians = self.positions.size(0)
        extra_signal_dim = 0 if self.forward_speed_mps is None else self.forward_speed_mps.size(1)
        state_dict = {
            "positions": self.positions,
            "rotations": self.rotations,
            "scales": self.scales,
            "densities": self.densities,
            "extra_signal": torch.zeros(n_gaussians, extra_signal_dim, device=self.device()),
            "features_albedo": self.rgb,
            "features_specular": torch.zeros(n_gaussians, 0, device=self.device()),
            "n_active_features": torch.tensor(0, device=self.device()),
        }
        return state_dict, DictConfig(
            {
                "name": "sh-gaussians",
                "device": "cuda",
                "particle": {"radiance_sph_degree": 0, "radiance_sph_O0": True, "extra_signal_dim": extra_signal_dim},
            }
        )

    def get_checkpoint(self) -> Checkpoint:
        checkpoint: Checkpoint = {}
        for attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES + self.OTHER_ATTRIBUTES_NAMES:
            attr = getattr(self, attr_name)
            if isinstance(attr, ModulatedLinearLayer):
                checkpoint[attr_name] = attr.get_checkpoint()
            else:
                # Other attributes can be directly serialized
                checkpoint[attr_name] = attr
        return checkpoint

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint: Checkpoint, gaussians_renderer: BaseGaussianRenderer | None = None
    ) -> CelsiusNRMPrimitive:
        loaded_attributes: dict[str, Any] = {}
        device = cast(torch.Tensor, checkpoint["positions"]).device
        for attr_name in cls.GAUSSIAN_ATTRIBUTES_NAMES + cls.OTHER_ATTRIBUTES_NAMES:
            attr = checkpoint.get(attr_name)
            if attr_name == "sky_head":
                assert isinstance(attr, dict)
                loaded_attributes[attr_name] = ModulatedLinearLayer.load_from_checkpoint(attr, device)
            else:
                loaded_attributes[attr_name] = attr

        return cls(
            **loaded_attributes,
            gaussians_renderer=gaussians_renderer,
            checkpointing="render",
            difix_model=None,
        )

    def set_sky_mask_enabled(self, sky_mask_enabled: bool):
        self.sky_mask_enabled = sky_mask_enabled

    def set_solid_background_color(self, solid_background_color: torch.Tensor | None):
        self.solid_background_color = solid_background_color

    def get_gaussian_parameters(self, timestamps_us: torch.Tensor | None) -> dict[str, torch.Tensor]:
        densities = self.densities

        def get_time_diff_s() -> torch.Tensor:
            assert timestamps_us is not None, "Timestamp must be provided for dynamic primitives"
            assert timestamps_us.shape == (1, 2), "Timestamps must have shape (1, 2)"
            # Convert to CPU if on GPU to avoid GPU->CPU sync when calling .item()
            timestamps_cpu = timestamps_us.cpu() if timestamps_us.is_cuda else timestamps_us
            start_timestamp_us, end_timestamp_us = timestamps_cpu[0, 0].item(), timestamps_cpu[0, 1].item()
            timestamp_scene_us = (start_timestamp_us + end_timestamp_us) // 2
            time_diff_s = (timestamp_scene_us - self.timestamps_us).float() / 1e6  # (n_gaussians, 1)
            return time_diff_s

        if self.forward_speed_mps is not None:
            time_diff_s = get_time_diff_s()

            if self.forward_speed_mps.shape[1] == 3:
                # Uni-directional flow
                positions = self.positions + self.forward_speed_mps * time_diff_s
            else:
                # Bidirectional flow
                positions = (
                    self.positions
                    + torch.where(
                        time_diff_s > 0,
                        self.forward_speed_mps[..., :3] * time_diff_s,  # First 3 dimensions are for forward speed
                        self.forward_speed_mps[..., 3:] * time_diff_s,  # Last 3 dimensions are for backward speed
                    )
                )

            # Falloff is only applied in dynamic scenarios.
            if self.falloff_sigma is not None:
                if self.falloff_sigma.shape[1] == 1:
                    falloff_factor = torch.exp(-((time_diff_s / self.falloff_sigma) ** self.FALLOFF_GAUSSIAN_EXP))
                else:
                    falloff_factor = torch.where(
                        time_diff_s > 0,
                        torch.exp(-((time_diff_s / self.falloff_sigma[..., :1]) ** self.FALLOFF_GAUSSIAN_EXP)),
                        torch.exp(-((time_diff_s / self.falloff_sigma[..., 1:]) ** self.FALLOFF_GAUSSIAN_EXP)),
                    )
                densities = densities * falloff_factor

        else:
            positions = self.positions

            # For static scenarios when needed, we use a simple clipping instead.
            if self.dynamic_bbox_mask is not None:
                time_diff_s = get_time_diff_s()
                # Gaussians to be kept has to be either static or have a time difference similar (0.1s) to the current rendering.
                densities = (
                    densities
                    * torch.logical_or(
                        ~self.dynamic_bbox_mask, time_diff_s.abs() < self.STATIC_CLIP_TIME_DIFF_S
                    ).float()
                )

        if self.sky_mask is not None and self.sky_mask_enabled:
            # Detach sky mask to make it solely supervised by sky loss (not through renderer).
            densities = densities * (1 - self.sky_mask.detach())

        # Note: to detach gradients we should use nre.utils.misc.stop_gradient() instead of .detach()
        # Otherwise either DDP or nrend will be unhappy.

        gaussian_parameters = {
            "positions": positions.float(),
            "rotations": self.rotations.float(),
            "scales": self.scales.float(),
            "densities": densities.float(),
            # A change in !2312 supporting MGPU is now accepting RGB directly instead of SH.
            # So we no longer need to to RGB2SH conversion here.
            "features": self.rgb.float(),  # [n_gaussians, 3]
            "extra_signal": torch.zeros(positions.size(0), 0, device=self.device()),
            "camera_extra_signal": torch.zeros(positions.size(0), 0, device=self.device()),
            "lidar_extra_signal": torch.zeros(positions.size(0), 0, device=self.device()),
        }

        # As we might need to supervise this, we use the actual prediction instead of the warping gt.
        if self.forward_speed_mps is not None:
            gaussian_parameters["extra_signal"] = self.forward_speed_mps.float()

        return gaussian_parameters

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        # Don't allow visualizing gaussian positions for now.
        return None

    def get_extra_ray_signal_infos(self) -> tuple[list[str], list[int], list[Callable]]:
        if self.forward_speed_mps is not None:
            return (["velocity"], [self.forward_speed_mps.shape[1]], [nn.Identity()])
        return ([], [], [])

    @torch.autocast(device_type="cuda", enabled=False)
    def postprocess_rendering(
        self, out: GaussiansRenderReturn, rendering_data: RenderingData, unique_sensor_idx: int | None
    ) -> GaussiansRenderReturn:
        rays = rendering_data.rays.reshape(-1, 6)
        sky_color: torch.Tensor | None = None
        if self.sky_token is not None and self.sky_head is not None:
            rays_d = rays[None, :, 3:]
            if self.sky_rotation is not None:
                rays_d = rays_d @ self.sky_rotation
            sky_color = torch.sigmoid(self.sky_head(rays_d, self.sky_token[None])[0])

        # First assemble bg color
        if self.solid_background_color is not None:
            out.rgb = out.rgb + (1 - out.opacity[..., None]) * self.solid_background_color[None]
            # Zero out sky color (we do this to avoid DDP issues)
            sky_color = (sky_color * 0.0) if sky_color is not None else None

        if sky_color is not None:
            # blend with the Gaussian rendering (N, 3)
            out.rgb = out.rgb + sky_color * (1 - out.opacity[..., None])

        # Then apply affine transform according to sensor index
        if self.affine_matrix is not None and self.affine_bias is not None:
            assert unique_sensor_idx is not None and 0 <= unique_sensor_idx < self.affine_matrix.shape[0], (
                "Invalid sensor index"
            )
            out.rgb = torch.einsum("n p, q p -> n q", out.rgb, self.affine_matrix[unique_sensor_idx])
            out.rgb = torch.clamp(out.rgb + self.affine_bias[unique_sensor_idx], min=0.0, max=1.0)

        return out

    def detach(self) -> Self:
        detached_attributes: dict[str, torch.Tensor | ModulatedLinearLayer | bool | None] = {}
        for attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES + self.OTHER_ATTRIBUTES_NAMES:
            attr = getattr(self, attr_name)
            detached_attributes[attr_name] = attr.detach() if isinstance(attr, torch.Tensor) else attr
        return self.__class__(
            **detached_attributes,  # type: ignore
            gaussians_renderer=self.gaussians_renderer,
            checkpointing=self.checkpointing,
            difix_model=self.difix_model,
        )

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __repr__(self) -> str:
        return f"CelsiusNRMPrimitive(#GS={len(self) / 1e6:.2f}M)"

    def mask(self, mask: torch.Tensor) -> Self:
        masked_attributes: dict[str, torch.Tensor | ModulatedLinearLayer | bool | None] = {}
        for attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES + self.OTHER_ATTRIBUTES_NAMES:
            attr = getattr(self, attr_name)
            if attr_name in self.GAUSSIAN_ATTRIBUTES_NAMES and isinstance(attr, torch.Tensor):
                masked_attributes[attr_name] = attr[mask]
            else:
                masked_attributes[attr_name] = attr
        return self.__class__(
            **masked_attributes,  # type: ignore
            gaussians_renderer=self.gaussians_renderer,
            checkpointing=self.checkpointing,
            difix_model=self.difix_model,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
    ) -> Self:
        """
        Return a new primitive with Gaussians filtered by density (and optionally sky).

        Does not modify self in place: builds a boolean mask from config and GT labels,
        then returns a new primitive via self.mask(gaussians_mask) with road_mask set on
        the returned primitive for downstream use.
        Rigid transform is not applied here (merge does that when merging chunks).

        If config.project_to_z_offset is True (Celsius), projects each Gaussian onto
        the road plane (z_offset in rig space) along the ray that spawned it; requires
        context_rig and context_batch.rendering.camera.rays (1:1 with Gaussians).

        Returns:
            A new primitive containing only Gaussians passing the filter (density threshold,
            and sky removal when keep_sky_gaussians is False and sky_mask is present).
        """
        camera_data = unpack_optional(context_batch.data.camera)
        road_flags = camera_data.labels.get_mask_flags_all(RayFlags.ROAD_SEMANTIC)
        road_mask = road_flags.flatten().unsqueeze(1)

        projected_positions = None
        project_to_z_offset = getattr(config, "project_to_z_offset", False)
        if project_to_z_offset:
            if context_rig is None:
                logger.warning(
                    "[CelsiusNRMPrimitive] project_to_z_offset is True but context_rig was not passed; "
                    "skipping ray projection onto road plane."
                )
            else:
                rendering_cam = unpack_optional(unpack_optional(context_batch.rendering).camera)
                rays = rendering_cam.rays.reshape(-1, 6)  # [N, 6] in NRE (origin, direction)
                N = self.positions.shape[0]
                assert rays.shape[0] == N, (
                    f"Ray count {rays.shape[0]} must match number of Gaussians {N} for project_to_z_offset"
                )
                dev = self.positions.device
                dtype = self.positions.dtype

                # Rays and primitive positions are in NRE. To find the z=z_offset plane in
                # rig coordinates, we need the NRE -> rig transform at the context timestamp.
                # T_rig_worlds[i] is a rig-to-NRE matrix (rig frame expressed in NRE space);
                # inverting it gives the NRE -> rig transform for this context's frame.
                traj = context_rig.rig_trajectories[0]
                if traj.cameras_frame_T_rig_worlds is not None:
                    first_cam_id = next(iter(traj.cameras_frame_T_rig_worlds))
                    # Use end-of-frame (index 1) pose of the first camera frame.
                    T_rig_to_nre = traj.cameras_frame_T_rig_worlds[first_cam_id][0, 1]
                else:
                    # Pick the rig pose closest to the first camera's end-of-frame timestamp.
                    first_cam_id = next(iter(traj.cameras_frame_timestamps_us))
                    cam_ts = traj.cameras_frame_timestamps_us[first_cam_id][0, 1]  # end-of-frame
                    closest_idx = int((traj.T_rig_world_timestamps_us - cam_ts).abs().argmin().item())
                    T_rig_to_nre = traj.T_rig_worlds[closest_idx]
                T_nre_to_rig = torch.linalg.inv(T_rig_to_nre.double()).to(device=dev, dtype=dtype)

                # Ray origins and directions in NRE space.
                O_nre = rays[:, :3].to(device=dev, dtype=dtype)
                D_nre = rays[:, 3:].to(device=dev, dtype=dtype)
                D_nre = D_nre / (D_nre.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                # Transform ray into rig space to solve the plane intersection there.
                # The road plane is z = z_offset in rig coordinates, which is axis-aligned
                # and trivial to intersect. Since T_nre_to_rig is a rigid transform (no
                # scaling), the ray parameter t is the same in both frames:
                #   ray_nre(t) = O_nre + t * D_nre
                #   ray_rig(t) = O_rig + t * D_rig
                # So we solve for t in rig space and evaluate the intersection in NRE space.
                O_rig = (T_nre_to_rig[:3, :3] @ O_nre.T + T_nre_to_rig[:3, 3:4]).T  # [N, 3]
                D_rig = (T_nre_to_rig[:3, :3] @ D_nre.T).T  # [N, 3]

                # Solve ray_rig(t).z = z_offset  =>  t = (z_offset - O_rig.z) / D_rig.z
                z_offset = getattr(config, "z_offset", 0.0)
                eps = 1e-8
                D_rig_z = D_rig[:, 2]
                # Clamp near-zero D_rig_z to avoid division by zero (rays nearly parallel
                # to the road plane); these rays are excluded via use_proj below.
                D_rig_z_safe = torch.where(
                    D_rig_z.abs() >= eps,
                    D_rig_z,
                    torch.where(
                        D_rig_z >= 0,
                        torch.tensor(eps, device=dev, dtype=dtype),
                        torch.tensor(-eps, device=dev, dtype=dtype),
                    ),
                )
                t = (z_offset - O_rig[:, 2]) / D_rig_z_safe
                # Only project when the ray actually hits the plane ahead (t >= 0)
                # and the ray isn't nearly parallel to the plane.
                use_proj = (t >= 0) & (D_rig_z.abs() >= eps)
                assert road_mask is not None, "road_mask must be set for project_to_z_offset"
                road_flat = road_mask.squeeze(1)  # [N]
                apply_proj = road_flat & use_proj
                # Evaluate the intersection point in NRE space (same t, NRE-space ray).
                p_proj_nre = O_nre + t.unsqueeze(-1) * D_nre
                positions_out = self.positions.clone()
                positions_out[apply_proj] = p_proj_nre[apply_proj]
                projected_positions = positions_out
                n_road = road_flat.sum().item()
                n_road_proj = apply_proj.sum().item()
                if n_road > 0 and n_road_proj < n_road:
                    logger.info(
                        "[CelsiusNRMPrimitive] project_to_z_offset: applied to %d/%d road Gaussians "
                        "(%d road rays had t<0 or |D_rig_z|<eps)",
                        n_road_proj,
                        n_road,
                        n_road - n_road_proj,
                    )

        gaussians_mask = self.densities[:, 0] > config.density_prune_threshold
        if self.sky_mask is not None and not getattr(config, "keep_sky_gaussians", False):
            gaussians_mask = gaussians_mask & (self.sky_mask[:, 0] < 0.5)

        n_gaussians = self.densities.shape[0]
        if road_mask.shape[0] != n_gaussians:
            raise ValueError(
                f"road_mask has {road_mask.shape[0]} entries (from ray-level labels) but primitive has "
                f"{n_gaussians} Gaussians; they must match for per-Gaussian indexing."
            )

        result = self.mask(gaussians_mask)
        result.road_mask = road_mask[gaussians_mask]
        if projected_positions is not None:
            result.positions = projected_positions[gaussians_mask]
        return result

    @torch.autocast(device_type="cuda", enabled=False)
    def rigid_transform(self, T_new: torch.Tensor) -> Self:
        # Rigid transform in SE(3) applied to the Gaussians, this will invalidate the sky tokens.
        assert T_new.shape == (4, 4), "Transform must have shape (4, 4)"
        T_new = T_new.float()
        R_new, t_new = T_new[:3, :3], T_new[:3, 3]
        q_new = so3_matrix_to_quat(R_new, unbatch=False)

        positions = self.positions @ R_new.T + t_new
        # Note that self.rotations is wxyz so we need to convert it a bit back and forth.
        rotations = quat_mult_xyzw(q_new.repeat(self.rotations.shape[0], 1), self.rotations[:, [1, 2, 3, 0]])[
            :, [3, 0, 1, 2]
        ]
        sky_rotation = R_new if self.sky_rotation is None else R_new @ self.sky_rotation

        forward_speed_mps = None
        if self.forward_speed_mps is not None:
            if self.forward_speed_mps.shape[1] == 3:
                forward_speed_mps = self.forward_speed_mps.to(R_new.dtype) @ R_new.T
            else:
                forward_speed_mps = torch.concat(
                    [
                        self.forward_speed_mps[..., :3].to(R_new.dtype) @ R_new.T,
                        self.forward_speed_mps[..., 3:].to(R_new.dtype) @ R_new.T,
                    ],
                    dim=-1,
                )

        transformed_primitive = self.__class__(
            positions=positions,
            rotations=rotations,
            scales=self.scales,
            densities=self.densities,
            rgb=self.rgb,
            timestamps_us=self.timestamps_us,
            forward_speed_mps=forward_speed_mps,
            falloff_sigma=self.falloff_sigma,
            dynamic_bbox_mask=self.dynamic_bbox_mask,
            road_mask=self.road_mask,
            sky_mask=self.sky_mask,
            sky_token=self.sky_token,
            sky_head=self.sky_head,
            sky_rotation=sky_rotation,
            affine_matrix=self.affine_matrix,
            affine_bias=self.affine_bias,
            gaussians_renderer=self.gaussians_renderer,
            checkpointing=self.checkpointing,
            difix_model=self.difix_model,
            sky_mask_enabled=self.sky_mask_enabled,
            solid_background_color=self.solid_background_color,
        )
        return transformed_primitive

    @classmethod
    def concatenate_gaussians(cls, primitives: list[Self]) -> Self:
        # Concatenate the Gaussians attributes (i.e. tensors with size [n_gaussians, ...]),
        # all other attributes will be removed.
        assert len(primitives) > 0, "No primitives to concatenate"
        assert all(isinstance(p, cls) for p in primitives), "All primitives must be of the same type"

        first_primitive: CelsiusNRMPrimitive = primitives[0]
        concatenated_attributes: dict[str, torch.Tensor | None] = {}
        for attr_name in cls.GAUSSIAN_ATTRIBUTES_NAMES:
            if getattr(first_primitive, attr_name) is not None:
                assert all(getattr(p, attr_name) is not None for p in primitives), (
                    f"All primitives must have {attr_name}"
                )
                concatenated_attributes[attr_name] = torch.cat([getattr(p, attr_name) for p in primitives], dim=0)
            else:
                concatenated_attributes[attr_name] = None

        concatenated_primitive = cls(
            **concatenated_attributes,  # type: ignore
            sky_token=None,
            sky_head=None,
            sky_rotation=None,
            affine_matrix=None,
            affine_bias=None,
            gaussians_renderer=first_primitive.gaussians_renderer,
            checkpointing=first_primitive.checkpointing,
            difix_model=first_primitive.difix_model,
            sky_mask_enabled=first_primitive.sky_mask_enabled,
            solid_background_color=first_primitive.solid_background_color,
        )
        return concatenated_primitive
