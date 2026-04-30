# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Tuple, Union, cast, overload

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import ModuleList
from torch.nn.functional import sigmoid, softplus

from libs.losses.kernel.constants import DEFAULT_SOURCE_CHROMS_VALUES, EPSILON, NUM_VIGNETTING_ALPHA_TERMS
from nre.models.post_processings.ppisp.slang import PPISPSlangFunction


@torch.no_grad()
def softplus_inverse(x: float, epsilon: float = EPSILON) -> float:
    """Compute the inverse of softplus function for initialization."""
    clamped_value = max(epsilon, float(x))
    tensor_value = torch.tensor(clamped_value)
    return float((tensor_value.exp() - 1).log())


@torch.no_grad()
def sigmoid_inverse(x: float, epsilon: float = EPSILON) -> float:
    """Compute the inverse of sigmoid function (logit) for initialization."""
    clamped_value = max(epsilon, min(1 - epsilon, float(x)))
    tensor_value = torch.tensor(clamped_value)
    return float(torch.log(tensor_value / (1 - tensor_value)))


class ExposureOffset(nn.Module):
    def __init__(self, device: torch.device | str, n_frames_per_camera: list[int], num_frames: int) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_frames_per_camera = n_frames_per_camera
        self.num_cameras = len(n_frames_per_camera)
        self.num_frames = num_frames
        # Exposure offset in EVs
        # Initialize with shape (num_frames, 1) to avoid unsqueeze in forward
        self.exposure_params = nn.Parameter(torch.zeros(num_frames, 1, device=self.device))

    @staticmethod
    def apply_exposure_offset(rgb: Tensor, exposure_offset: Tensor) -> Tensor:
        """
        Apply exposure offset to RGB values.
        """
        return rgb * torch.exp2(exposure_offset)

    def forward(self, rgb: Tensor, frame_idcs: Tensor) -> Tensor:
        """
        Args:
            rgb: Input RGB values, shape (B, 3)
            frame_idcs: Frame indices, shape (B,)
        Returns:
            Adjusted RGB values, shape (B, 3)
        """
        assert rgb.shape[0] == frame_idcs.shape[0], "Batch sizes must match"
        assert frame_idcs.max() < self.num_frames, "Frame indices out of bounds"
        assert frame_idcs.min() >= -1, "Frame indices must be non-negative or -1"

        has_frame = frame_idcs != -1

        return torch.where(
            has_frame.unsqueeze(-1),
            ExposureOffset.apply_exposure_offset(rgb, self.exposure_params[frame_idcs.clamp(min=0)]),
            rgb,
        )


class RadialFalloff(nn.Module):
    def __init__(self, device: torch.device | str) -> None:
        super().__init__()
        self.device = torch.device(device)

        # Optical center for this specific channel
        self.optical_center = nn.Parameter(torch.full((2,), 0.5, device=self.device))
        # Polynomial coefficients for radial falloff
        self.alpha = nn.Parameter(torch.zeros(NUM_VIGNETTING_ALPHA_TERMS, device=self.device))

    class PackedParams:
        """
        Efficient data class for packed vignetting parameters.

        Can represent either:
        - Single curve: tensor shape (NUM_VIGNETTING_ALPHA_TERMS + 2,)
        - Multiple curves: tensor shape (N, NUM_VIGNETTING_ALPHA_TERMS + 2) where N is number of curves
        - Batched curves: tensor shape (..., NUM_VIGNETTING_ALPHA_TERMS + 2) for arbitrary batch dimensions
        """

        def __init__(self, data: torch.Tensor):
            """
            Args:
                data: Tensor of shape (..., NUM_VIGNETTING_ALPHA_TERMS + 2) containing packed parameters in order:
                     [center_x, center_y, alpha1, alpha2, ..., alphaN] where N = NUM_VIGNETTING_ALPHA_TERMS
            """
            assert data.shape[-1] == NUM_VIGNETTING_ALPHA_TERMS + 2, (
                f"Expected last dimension to be {NUM_VIGNETTING_ALPHA_TERMS + 2}, got {data.shape[-1]}"
            )
            self.data = data

        @property
        def optical_center(self) -> torch.Tensor:
            """Returns optical center as [center_x, center_y]."""
            return self.data[..., :2]

        @optical_center.setter
        def optical_center(self, value):
            self.data[..., :2] = value

        @property
        def alphas(self) -> torch.Tensor:
            """Returns alpha coefficients."""
            return self.data[..., 2 : 2 + NUM_VIGNETTING_ALPHA_TERMS]

        @alphas.setter
        def alphas(self, value):
            self.data[..., 2 : 2 + NUM_VIGNETTING_ALPHA_TERMS] = value

    @staticmethod
    def apply_radial_falloff(values: Tensor, coords_xy: Tensor, optical_center: Tensor, alphas: Tensor) -> Tensor:
        """
        Apply radial falloff (vignetting) to values using given parameters.

        Args:
            values: Input values, shape (B,) or (B, C) where C is number of channels
            coords_xy: 2D coordinates, shape (B, 2)
            optical_center: Optical center coordinates, shape (2,)
            alphas: Alpha coefficients, shape (NUM_VIGNETTING_ALPHA_TERMS,)
        Returns:
            Falloff-adjusted values, same shape as input values
        """
        # Compute squared radial distances from optical center
        diff = coords_xy - optical_center  # Shape: (B, 2)
        r2 = torch.sum(diff * diff, dim=1)  # Shape: (B,)

        # Compute falloff factors: 1 + alpha1*r^2 + alpha2*r^4 + ... + alphaN*r^(2N)
        powers = torch.arange(1, NUM_VIGNETTING_ALPHA_TERMS + 1, device=r2.device)
        r_powers = r2.unsqueeze(-1).pow(powers)  # Shape: (B, NUM_VIGNETTING_ALPHA_TERMS)
        vig = 1.0 + (alphas * r_powers).sum(dim=-1)  # Shape: (B,)

        # Clamp to [0, 1] for physical plausibility
        vig = torch.clamp(vig, min=0.0, max=1.0)

        # Apply vignetting - handle both 1D and 2D input cases
        if values.dim() == 1:
            return values * vig
        else:
            return values * vig.unsqueeze(-1)

    def compute_falloff_factors(self, r2: Tensor) -> Tensor:
        """
        Compute vignetting falloff factors from squared radial distances.

        Args:
            r2: Squared radial distances, shape (B,)
        Returns:
            Vignetting factors, shape (B,)
        """
        # Compute falloff factors
        powers = torch.arange(1, NUM_VIGNETTING_ALPHA_TERMS + 1, device=r2.device)
        r_powers = r2.unsqueeze(-1).pow(powers)
        vig = 1.0 + (self.alpha * r_powers).sum(dim=-1)

        # Clamp to [0, 1] for physical plausability
        vig = torch.clamp(vig, min=0.0, max=1.0)

        return vig

    def forward(self, values: Tensor, coords_xy: Tensor) -> Tensor:
        """
        Args:
            values: Input values, shape (B,)
            coords_xy: 2D coordinates, shape (B, 2)
        Returns:
            Falloff-adjusted values, shape (B,)
        """
        return RadialFalloff.apply_radial_falloff(values, coords_xy, self.optical_center, self.alpha)


# Add custom classes to improve readability and type safety, replacing nested torch.nn.ModuleList
# types.
class RGBRadialFalloff(ModuleList):
    def __init__(self, device: torch.device | str) -> None:
        super().__init__([RadialFalloff(device) for _ in range(3)])

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> RadialFalloff: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[RadialFalloff, ModuleList]:
        result = super().__getitem__(idx)
        if isinstance(idx, int):
            return cast(RadialFalloff, result)
        return cast(ModuleList, result)


class MultiCamRadialFalloff(ModuleList):
    def __init__(self, device: torch.device | str, num_cameras: int) -> None:
        super().__init__([RGBRadialFalloff(device) for _ in range(num_cameras)])

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> RGBRadialFalloff: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[RGBRadialFalloff, ModuleList]:
        result = super().__getitem__(idx)
        if isinstance(idx, int):
            return cast(RGBRadialFalloff, result)
        return cast(ModuleList, result)


class Vignetting(nn.Module):
    def __init__(self, device: torch.device | str, num_cameras: int) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_cameras = num_cameras

        self.falloff_curves = MultiCamRadialFalloff(device, num_cameras)

    def forward(self, rgb: Tensor, unique_camera_idcs: Tensor, coords_xy: Tensor) -> Tensor:
        """
        Args:
            rgb: RGB values, shape (B, 3)
            unique_camera_idcs: Camera indices, shape (B,)
            coords_xy: 2D coordinates, shape (B, 2)
        Returns:
            Vignetting-adjusted RGB values, shape (B, 3)
        """
        assert unique_camera_idcs.shape[0] == rgb.shape[0], "Batch sizes must match"
        assert coords_xy.shape[0] == rgb.shape[0], "Batch sizes must match"
        assert unique_camera_idcs.max() < self.num_cameras, "Camera indices out of bounds"

        output = torch.zeros_like(rgb)
        for cam_idx in range(self.num_cameras):
            cam_mask = unique_camera_idcs == cam_idx
            if not cam_mask.any():
                continue

            masked_coords = coords_xy[cam_mask]
            cam_curves = self.falloff_curves[cam_idx]
            for channel in range(3):
                masked_values = rgb[cam_mask, channel]
                output[cam_mask, channel] = cam_curves[channel](masked_values, masked_coords)

        unchanged_cam_mask = unique_camera_idcs == -1
        output[unchanged_cam_mask] = rgb[unchanged_cam_mask]

        return output


class ColorCorrection(nn.Module):
    """
    Color correction based on 2D color homography, following work by Finlayson, Gong, et al.
    """

    rgb_to_rgi: Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], device="cuda")
    rgi_to_rgb: Tensor = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device="cuda")
    _default_source_chroms: Tensor

    def __init__(self, device: torch.device | str, n_frames_per_camera: list[int], num_frames: int) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_frames_per_camera = n_frames_per_camera
        self.num_frames = num_frames

        # Store only the first 8 elements of each homography matrix as parameters
        # The 9th element (h[2,2]) is fixed at 1
        # Initialize with identity matrix values
        identity_flat = torch.eye(3, device=self.device).flatten()[:8]  # First 8 elements of identity matrix
        self.color_params = nn.Parameter(identity_flat.unsqueeze(0).repeat(num_frames, 1))

        # Register default source chromaticities as a buffer to avoid CUDA sync on each access
        self.register_buffer(
            "_default_source_chroms",
            torch.tensor(DEFAULT_SOURCE_CHROMS_VALUES, device=self.device),
        )

    @property
    def default_source_chroms(self) -> Tensor:
        """Return the default source chromaticities buffer."""
        return self._default_source_chroms

    @staticmethod
    def get_default_source_chroms(device: torch.device | str) -> Tensor:
        """
        Return the default source chromaticities tensor for color correction.

        Args:
            device: Device on which to create the tensor

        Returns:
            Tensor containing the source chromaticities with shape (NUM_SOURCE_CHROMS, 2)
        """
        return torch.tensor(DEFAULT_SOURCE_CHROMS_VALUES, device=device)

    @staticmethod
    def apply_color_correction_rgb(rgb: Tensor, h: Tensor) -> Tensor:
        """
        Apply color correction to RGB values using the given homography matrices.

        Args:
            rgb: RGB values, shape (B, 3)
            h: Homography matrices, shape (B, 3, 3)
        Returns:
            Color corrected RGB values, shape (B, 3)
        """
        # Get device from input tensor
        device = rgb.device
        h = h.to(device)

        # Get transformation matrices on the correct device
        rgb_to_rgi = ColorCorrection.rgb_to_rgi.to(device)
        rgi_to_rgb = ColorCorrection.rgi_to_rgb.to(device)

        # Convert RGB to Red-Green-Intensity (RGI) space
        rgi = torch.matmul(rgb, rgb_to_rgi)  # Shape: (B, 3)
        intensity = rgi[:, 2:3]

        # Apply homography
        rgi_transformed = torch.bmm(h, rgi.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)

        # Scale to maintain original intensity.
        # Use numerically robust division that doesn't fail when intensity is close to zero.
        rgi_transformed_scaled = rgi_transformed * intensity / (rgi_transformed[:, 2:3] + EPSILON)

        # Convert RGI back to RGB
        rgb_corrected = torch.matmul(rgi_transformed_scaled, rgi_to_rgb)  # Shape: (B, 3)

        return rgb_corrected

    @staticmethod
    def apply_color_correction_rg(rg: Tensor, h: Tensor) -> Tensor:
        """
        Apply color correction in rg space by applying homographies to source rg chromaticities.

        Args:
            rg: Source chromaticities in RG space, shape (N, 2)
            h: Homography matrices, shape (B, 3, 3), where B can be num_frames.
        Returns:
            Color corrected chromaticities, shape (B, N, 2)
        """
        device = rg.device
        batch_size = h.shape[0]

        # Convert source chromaticities to homogeneous coordinates
        rg_homogeneous = torch.cat([rg, torch.ones(rg.shape[0], 1, device=device)], dim=1)  # Shape: (N, 3)

        # Apply homographies to source chromaticities
        # Expand source points to match shape (B, N, 3)
        rg_expanded = rg_homogeneous.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply homographies for each frame
        rg_transformed = torch.bmm(
            h,  # Shape: (B, 3, 3)
            rg_expanded.transpose(1, 2),  # Shape: (B, 3, N)
        ).transpose(1, 2)  # Shape: (B, N, 3)

        # Convert back from homogeneous coordinates
        out_rg = rg_transformed[:, :, :2] / (rg_transformed[:, :, 2:3] + EPSILON)  # Shape: (B, N, 2)

        return out_rg

    @staticmethod
    def params_to_homography(color_params: Tensor) -> Tensor:
        """Constructs homography matrices from color parameters."""
        batch_size = color_params.shape[0]
        ones = torch.ones(batch_size, 1, device=color_params.device)
        homography_flat = torch.cat([color_params, ones], dim=1)
        return homography_flat.reshape(batch_size, 3, 3)

    def get_all_homographies(self) -> Tensor:
        """
        Get all homography matrices for all frames.
        Returns:
            Homography matrices, shape (num_frames, 3, 3)
        """
        return ColorCorrection.params_to_homography(self.color_params)

    def get_homographies_for_frames(self, frame_idcs: Tensor) -> Tensor:
        """
        Get the homography matrices for the given frame indices.
        Args:
            frame_idcs: Frame indices, shape (B,).
        Returns:
            Homography matrices, shape (B, 3, 3).
        """
        return ColorCorrection.params_to_homography(self.color_params[frame_idcs])

    def forward(self, rgb: Tensor, frame_idcs: Tensor) -> Tensor:
        """
        Apply color correction homographies to RGB values, using RGI space.

        Args:
            rgb: RGB values, shape (B, 3)
            frame_idcs: Frame indices, shape (B,)
        Returns:
            Color corrected RGB values, shape (B, 3)
        """
        assert frame_idcs.shape[0] == rgb.shape[0], "Batch sizes must match"
        assert frame_idcs.max() < self.num_frames, "Frame indices out of bounds"
        assert frame_idcs.min() >= -1, "Frame indices must be non-negative or -1"

        # When frame_idcs = -1, we don't care about the homography
        h = self.get_homographies_for_frames(frame_idcs.clamp(min=0))  # shape (B, 3, 3)

        has_frame = frame_idcs != -1
        output = torch.where(has_frame.unsqueeze(-1), ColorCorrection.apply_color_correction_rgb(rgb, h), rgb)

        return output

    @staticmethod
    def get_h_from_chrom_pairs(source_chroms: Tensor, target_chroms: Tensor) -> Tensor:
        """
        Get the homography matrix using the DLT algorithm based on source and target chromaticity pairs.

        Args:
            source_chroms: Source chromaticities in RG space, shape (N, 2) where N is the number of points
            target_chroms: Target chromaticities in RG space, shape (N, 2)
        Returns:
            Homography matrix, shape (3, 3)
        """
        # Ensure we have at least 4 pairs of points (the minimum needed for a robust homography)
        assert source_chroms.shape[0] >= 4, "At least 4 chromaticity pairs are required"
        assert source_chroms.shape == target_chroms.shape, "Source and target chromaticities must have the same shape"

        # Ensure both tensors are on the same device
        device = source_chroms.device
        target_chroms = target_chroms.to(device)

        num_points = source_chroms.shape[0]

        # Convert to homogeneous coordinates
        source_homogeneous = torch.cat(
            [source_chroms, torch.ones(num_points, 1, device=device)], dim=1
        )  # Shape: (N, 3)

        # Extract u, v coordinates
        u = target_chroms[:, 0]  # Shape: (N,)
        v = target_chroms[:, 1]  # Shape: (N,)

        # Create the matrix in one go
        A = torch.zeros(2 * num_points, 9, device=device)

        # Even rows: [0, 0, 0, -x, -y, -w, v*x, v*y, v*w]
        A[0::2, 3:6] = -source_homogeneous
        A[0::2, 6:9] = source_homogeneous * v.unsqueeze(1)

        # Odd rows: [x, y, w, 0, 0, 0, -u*x, -u*y, -u*w]
        A[1::2, 0:3] = source_homogeneous
        A[1::2, 6:9] = -source_homogeneous * u.unsqueeze(1)

        # Solve the homogeneous system of equations using SVD
        _, _, Vt = torch.linalg.svd(A)

        # The solution is the last row of Vt (corresponding to the smallest singular value)
        h_flat = Vt[-1]

        # Reshape to 3x3 matrix and normalize so that h[2,2] = 1
        h = h_flat.reshape(3, 3)
        h = h / h[2, 2]

        return h.to(device)


class PiecewisePowerFunction(nn.Module):
    """
    Piecewise function used to model camera response curve.
    It has five segments defined by four points: toe point, p0, p1, shoulder point.
    1. Below first point, return constant value equal to 0
    2. Between first and second point, return power curve of form: y = a*x^b
    3. Between second and third point, return power curve of form: y = (m*x + b)^gamma
    4. Between third and fourth point, return inverted power curve of form: y_max - y = a*(x_max - x)^b
    5. Above last point, return constant value equal to 1

    The initial curve parameters are chosen to satisfy inverse(1.0) ~= 1.0, i.e., dynamic range 1.
    """

    class RawParams:
        """
        Efficient data class for raw CRF parameters (before applying softplus/sigmoid).

        Can represent either:
        - Single curve: tensor shape (7,)
        - Multiple curves: tensor shape (N, 7) where N is number of curves
        - Batched curves: tensor shape (..., 7) for arbitrary batch dimensions
        """

        def __init__(self, data: torch.Tensor):
            """
            Args:
                data: Tensor of shape (..., 7) containing raw parameters in order:
                     [x0_offset_raw, y0_raw, y1_fract_raw, toe_length_raw,
                      shoulder_length_raw, shoulder_overshoot_raw, gamma_raw]
            """
            assert data.shape[-1] == 7, f"Expected last dimension to be 7, got {data.shape[-1]}"
            self.data = data

        @property
        def x0_offset_raw(self) -> torch.Tensor:
            return self.data[..., 0]

        @x0_offset_raw.setter
        def x0_offset_raw(self, value):
            self.data[..., 0] = value

        @property
        def y0_raw(self) -> torch.Tensor:
            return self.data[..., 1]

        @y0_raw.setter
        def y0_raw(self, value):
            self.data[..., 1] = value

        @property
        def y1_fract_raw(self) -> torch.Tensor:
            return self.data[..., 2]

        @y1_fract_raw.setter
        def y1_fract_raw(self, value):
            self.data[..., 2] = value

        @property
        def toe_length_raw(self) -> torch.Tensor:
            return self.data[..., 3]

        @toe_length_raw.setter
        def toe_length_raw(self, value):
            self.data[..., 3] = value

        @property
        def shoulder_length_raw(self) -> torch.Tensor:
            return self.data[..., 4]

        @shoulder_length_raw.setter
        def shoulder_length_raw(self, value):
            self.data[..., 4] = value

        @property
        def shoulder_overshoot_raw(self) -> torch.Tensor:
            return self.data[..., 5]

        @shoulder_overshoot_raw.setter
        def shoulder_overshoot_raw(self, value):
            self.data[..., 5] = value

        @property
        def gamma_raw(self) -> torch.Tensor:
            return self.data[..., 6]

        @gamma_raw.setter
        def gamma_raw(self, value):
            self.data[..., 6] = value

    @staticmethod
    @torch.no_grad()
    def get_crf_raw_param_values(
        x0: float = 0.166,
        y0: float = 0.25,
        y1: float = 0.9,
        toe_length: float = 1.0,
        shoulder_length: float = 1.0,
        shoulder_overshoot: float = 0.1,
        gamma: float = 1.0 / 2.2,
    ) -> "PiecewisePowerFunction.RawParams":
        """
        Convert user-friendly CRF parameters to raw parameter values for initialization.

        Args:
            x0: x-coordinate of p0 point
            y0: y-coordinate of p0 point
            y1: y-coordinate of p1 point
            toe_length: Length of toe region relative to x0
            shoulder_length: Length of shoulder region relative to remaining distance
            shoulder_overshoot: Amount of overshoot in shoulder region
            gamma: gamma correction value

        Returns:
            RawParams object containing raw parameter values
        """
        # Initialize raw params with zeros
        raw_params = PiecewisePowerFunction.RawParams(torch.zeros(7))

        # Calculate effective parameters and apply inverse functions using setters
        raw_params.x0_offset_raw = torch.tensor(softplus_inverse(x0 / (1.0 + toe_length)))
        raw_params.y0_raw = torch.tensor(sigmoid_inverse(y0))
        raw_params.y1_fract_raw = torch.tensor(sigmoid_inverse((y1 - y0) / (1.0 - y0)))
        raw_params.toe_length_raw = torch.tensor(softplus_inverse(toe_length))
        raw_params.shoulder_length_raw = torch.tensor(softplus_inverse(shoulder_length))
        raw_params.shoulder_overshoot_raw = torch.tensor(softplus_inverse(shoulder_overshoot))
        raw_params.gamma_raw = torch.tensor(softplus_inverse(gamma))

        return raw_params

    def __init__(
        self,
        device: torch.device | str,
        x0: float = 0.166,
        y0: float = 0.25,
        y1: float = 0.9,
        toe_length: float = 1.0,
        shoulder_length: float = 1.0,
        shoulder_overshoot: float = 0.1,
        gamma: float = 1.0 / 2.2,
    ) -> None:
        """
        Args:
            device: Device to place the parameters on
            x0: x-coordinate of p0 point
            y0: y-coordinate of p0 point
            y1: y-coordinate of p1 point
            toe_length: Length of toe region relative to x0
            shoulder_length: Length of shoulder region relative to remaining distance
            shoulder_overshoot: Amount of overshoot in shoulder region
            gamma: gamma correction value
        """
        super().__init__()
        self.device = torch.device(device)
        # Store parameter directly on module for proper registration
        initial_raw_params = self.get_crf_raw_param_values(
            x0, y0, y1, toe_length, shoulder_length, shoulder_overshoot, gamma
        )
        self.raw_params = nn.Parameter(initial_raw_params.data.to(device))

    class CurvePoints:
        """
        Efficient data class for computed CRF curve points.

        Can represent either:
        - Single curve: individual tensor arguments
        - Multiple curves: tensor arguments with shape (N,) where N is number of curves
        - Batched curves: tensor arguments with arbitrary batch dimensions
        """

        def __init__(
            self,
            x0: torch.Tensor,
            y0: torch.Tensor,
            slope_p0: torch.Tensor,
            y0_pre_gamma: torch.Tensor,
            slope_line: torch.Tensor,
            gamma: torch.Tensor,
            x1: torch.Tensor,
            y1: torch.Tensor,
            slope_p1: torch.Tensor,
            shoulder_x: torch.Tensor,
            shoulder_y: torch.Tensor,
        ):
            """
            Args:
                x0: x-coordinate of p0 point
                y0: y-coordinate of p0 point
                slope_p0: slope at p0 point
                y0_pre_gamma: y-coordinate of p0 before gamma correction
                slope_line: slope of linear segment
                gamma: gamma correction value
                x1: x-coordinate of p1 point
                y1: y-coordinate of p1 point
                slope_p1: slope at p1 point
                shoulder_x: x-coordinate of shoulder point
                shoulder_y: y-coordinate of shoulder point
            """
            self.x0 = x0
            self.y0 = y0
            self.slope_p0 = slope_p0
            self.y0_pre_gamma = y0_pre_gamma
            self.slope_line = slope_line
            self.gamma = gamma
            self.x1 = x1
            self.y1 = y1
            self.slope_p1 = slope_p1
            self.shoulder_x = shoulder_x
            self.shoulder_y = shoulder_y

    @staticmethod
    def crf_curve_points(raw_params: "PiecewisePowerFunction.RawParams") -> "PiecewisePowerFunction.CurvePoints":
        """
        Vectorized computation of curve points from raw parameters.

        Args:
            raw_params: RawParams object containing raw parameter tensors

        Returns:
            CurvePoints object containing computed curve points
        """
        # Apply activation functions to raw parameters
        x0_offset = softplus(raw_params.x0_offset_raw)
        y0 = sigmoid(raw_params.y0_raw)
        y1_fract = sigmoid(raw_params.y1_fract_raw)
        toe_length = softplus(raw_params.toe_length_raw)
        shoulder_length = softplus(raw_params.shoulder_length_raw)
        shoulder_overshoot = softplus(raw_params.shoulder_overshoot_raw)
        gamma = softplus(raw_params.gamma_raw)

        # Clamp gamma to prevent pow() underflow: when gamma < ~0.1,
        # y0^(1/gamma) underflows to 0 in float32, causing NaN cascade.
        gamma = gamma.clamp(min=0.1)

        # Compute p0 = (x0, y0)
        x0 = x0_offset * (1.0 + toe_length)

        # The slope at p0 is given by the line passing through (toe_length, 0) and p0
        slope_p0 = y0 / x0_offset

        # Before applying gamma, p0 and p1 are connected with a straight line
        y0_pre_gamma = y0.pow(1.0 / gamma)
        slope_line = slope_p0 / (gamma * y0_pre_gamma.pow(gamma - 1.0))

        # Find p1 = (x1, y1)
        y1 = y0 + (1.0 - y0) * y1_fract
        y1_pre_gamma = y1.pow(1.0 / gamma)
        x1 = x0 + (y1_pre_gamma - y0_pre_gamma) / slope_line

        # The slope at p1 is given from p1
        slope_p1 = gamma * slope_line * y1_pre_gamma.pow(gamma - 1.0)

        # The shoulder point is given by extending from p1 along slope_p1 and intersecting y = 1 + shoulder_overshoot
        remaining_y = 1.0 - y1
        shoulder_y = 1.0 + remaining_y * shoulder_overshoot
        shoulder_intercept_x_offset = (shoulder_y - y1) / slope_p1
        shoulder_x = x1 + shoulder_intercept_x_offset * (1.0 + shoulder_length)

        return PiecewisePowerFunction.CurvePoints(
            x0=x0,
            y0=y0,
            slope_p0=slope_p0,
            y0_pre_gamma=y0_pre_gamma,
            slope_line=slope_line,
            gamma=gamma,
            x1=x1,
            y1=y1,
            slope_p1=slope_p1,
            shoulder_x=shoulder_x,
            shoulder_y=shoulder_y,
        )

    @staticmethod
    def solve_ln_a_b(x0: Tensor, y0: Tensor, m: Tensor, epsilon: float = EPSILON) -> Tuple[Tensor, Tensor]:
        """
        Solve for ln(a) and b in the equation y = a*x^b given a point (x0, y0) and slope m.

        Args:
            x0: x-coordinate of point
            y0: y-coordinate of point
            m: slope at point
            epsilon: Small value to prevent numerical issues
        Returns:
            Tuple containing ln(a) and b parameters
        """
        b = (m * x0) / y0
        ln_a = torch.log(y0.clamp(min=epsilon)) - b * torch.log(x0.clamp(min=epsilon))
        return ln_a, b

    @staticmethod
    def apply_ppf(curve_points: "PiecewisePowerFunction.CurvePoints", x: Tensor, epsilon: float = EPSILON) -> Tensor:
        """
        Apply the piecewise power function using pre-computed curve points.

        Args:
            curve_points: Pre-computed curve points
            x: Input values
            epsilon: Small value to prevent numerical issues
        Returns:
            Output values after applying the piecewise function, same shape as input
        """
        # Compute all regions unconditionally to avoid CUDA syncs from .any() checks.
        # Use torch.where to select the appropriate values based on masks.

        # Toe region: y = a*x^b for x in [0, x0)
        ln_a_toe, b_toe = PiecewisePowerFunction.solve_ln_a_b(
            curve_points.x0, curve_points.y0, curve_points.slope_p0, epsilon
        )
        # Clamp x to epsilon to prevent log(0), safe for all regions
        x_safe = torch.clamp(x, min=epsilon)
        toe_y = torch.exp(ln_a_toe + b_toe * torch.log(x_safe)).to(dtype=x.dtype)

        # Middle region (gamma-corrected line): y = (y0_pre_gamma + slope_line * (x - x0))^gamma
        middle_y = (
            (curve_points.y0_pre_gamma + curve_points.slope_line * (x - curve_points.x0))
            .clamp(min=epsilon)  # Clamp to avoid negative values before pow
            .pow(curve_points.gamma)
            .to(dtype=x.dtype)
        )

        # Shoulder region: y = shoulder_y - a*(shoulder_x - x)^b
        ln_a_shoulder, b_shoulder = PiecewisePowerFunction.solve_ln_a_b(
            curve_points.shoulder_x - curve_points.x1,
            curve_points.shoulder_y - curve_points.y1,
            curve_points.slope_p1,
            epsilon,
        )
        shoulder_diff = curve_points.shoulder_x - x
        # Clamp to epsilon to prevent log of negative/zero values
        shoulder_diff_safe = torch.clamp(shoulder_diff, min=epsilon)
        shoulder_y = (
            curve_points.shoulder_y - torch.exp(ln_a_shoulder + b_shoulder * torch.log(shoulder_diff_safe))
        ).to(dtype=x.dtype)
        # Clamp to LDR range
        shoulder_y = shoulder_y.clamp(max=1.0)

        # Build masks for each region
        toe_mask = (x >= 0) & (x < curve_points.x0)
        middle_mask = (x >= curve_points.x0) & (x < curve_points.x1)
        shoulder_mask = (x >= curve_points.x1) & (x < curve_points.shoulder_x)
        above_shoulder_mask = x >= curve_points.shoulder_x

        # Combine results using torch.where (no CUDA sync required)
        # Start with zeros for x < 0
        y = torch.zeros_like(x)
        y = torch.where(toe_mask, toe_y, y)
        y = torch.where(middle_mask, middle_y, y)
        y = torch.where(shoulder_mask, shoulder_y, y)
        y = torch.where(above_shoulder_mask, 1.0, y)

        return y

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the piecewise power function to input values.

        Args:
            x: Input values, shape (B,) or (B, 1)
        Returns:
            Output values after applying the piecewise function, same shape as input
        """
        # Create lightweight typed wrapper for memory access
        raw_params_accessor = PiecewisePowerFunction.RawParams(self.raw_params)
        curve_points = PiecewisePowerFunction.crf_curve_points(raw_params_accessor)

        # Use the static method to apply the function
        return PiecewisePowerFunction.apply_ppf(curve_points, x, EPSILON)

    @staticmethod
    def inverse(curve_points: "PiecewisePowerFunction.CurvePoints", y: Tensor, epsilon: float = EPSILON) -> Tensor:
        """
        Apply the inverse of the piecewise power function.

        Args:
            curve_points: Pre-computed curve points. Can be single curve or batched.
            y: Input values, shape matching the curve_points (e.g., (B,), (num_cameras, 3), etc.)
            epsilon: Small value to prevent numerical issues
        Returns:
            Output values after applying the inverse function, same shape as input
        """
        # Compute all regions unconditionally to avoid CUDA syncs from .any() checks.
        # Use torch.where to select the appropriate values based on masks.

        # Toe region: y = a*x^b -> x = (y/a)^(1/b) = exp((log(y) - ln_a) / b)
        ln_a_toe, b_toe = PiecewisePowerFunction.solve_ln_a_b(
            curve_points.x0, curve_points.y0, curve_points.slope_p0, epsilon
        )
        y_safe = y.clamp(min=epsilon)
        toe_x = torch.exp((torch.log(y_safe) - ln_a_toe) / b_toe)

        # Middle region: y = (m*x + b)^gamma -> x = ((y^(1/gamma) - y0_pre_gamma) / slope_line) + x0
        middle_x = (
            y_safe.pow(1.0 / curve_points.gamma) - curve_points.y0_pre_gamma
        ) / curve_points.slope_line + curve_points.x0

        # Shoulder region: y = shoulder_y - a*(shoulder_x - x)^b
        # -> x = shoulder_x - ((shoulder_y - y)/a)^(1/b) = shoulder_x - exp((log(shoulder_y - y) - ln_a) / b)
        ln_a_shoulder, b_shoulder = PiecewisePowerFunction.solve_ln_a_b(
            curve_points.shoulder_x - curve_points.x1,
            curve_points.shoulder_y - curve_points.y1,
            curve_points.slope_p1,
            epsilon,
        )
        shoulder_diff = (curve_points.shoulder_y - y).clamp(min=epsilon)
        shoulder_x = curve_points.shoulder_x - torch.exp((torch.log(shoulder_diff) - ln_a_shoulder) / b_shoulder)

        # Above dynamic range (y > 1.0): return x where curve intersects y = 1.0
        # This uses the same shoulder formula with y = 1.0
        above_diff = (curve_points.shoulder_y - 1.0).clamp(min=epsilon)
        above_x = curve_points.shoulder_x - torch.exp((torch.log(above_diff) - ln_a_shoulder) / b_shoulder)

        # Build masks for each region
        toe_mask = (y >= 0.0) & (y < curve_points.y0)
        middle_mask = (y >= curve_points.y0) & (y < curve_points.y1)
        shoulder_mask = (y >= curve_points.y1) & (y <= 1.0)
        above_mask = y > 1.0

        # Combine results using torch.where (no CUDA sync required)
        # Start with zeros for y < 0
        x = torch.zeros_like(y)
        x = torch.where(toe_mask, toe_x, x)
        x = torch.where(middle_mask, middle_x, x)
        x = torch.where(shoulder_mask, shoulder_x, x)
        x = torch.where(above_mask, above_x, x)

        return x


class RGBPiecewisePowerFunction(ModuleList):
    def __init__(self, device: torch.device | str) -> None:
        super().__init__([PiecewisePowerFunction(device) for _ in range(3)])

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> PiecewisePowerFunction: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[PiecewisePowerFunction, ModuleList]:
        result = super().__getitem__(idx)
        if isinstance(idx, int):
            return cast(PiecewisePowerFunction, result)
        return cast(ModuleList, result)


class MultiCamPiecewisePowerFunction(ModuleList):
    def __init__(self, device: torch.device | str, num_cameras: int) -> None:
        super().__init__([RGBPiecewisePowerFunction(device) for _ in range(num_cameras)])

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> RGBPiecewisePowerFunction: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[RGBPiecewisePowerFunction, ModuleList]:
        result = super().__getitem__(idx)
        if isinstance(idx, int):
            return cast(RGBPiecewisePowerFunction, result)
        return cast(ModuleList, result)


class CRF(nn.Module):
    """
    The full camera response function (CRF) is modelled as a piecewise power function per channel.
    """

    def __init__(self, device: torch.device | str, num_cameras: int) -> None:
        super().__init__()
        self.device = torch.device(device)

        self.num_cameras = num_cameras
        self.curves = MultiCamPiecewisePowerFunction(device, num_cameras)

    def forward(self, rgb: Tensor, unique_camera_idcs: Tensor) -> Tensor:
        """
        Applies camera-specific and color channel-specific response curves to input RGB values.

        For each unique camera index, applies three separate piecewise power functions
        (one per RGB channel) to transform the corresponding RGB values. This models
        how different cameras may have different response characteristics per color channel.

        Args:
            rgb: Input RGB tensor of shape (B, 3)
            unique_camera_idcs: Camera indices tensor of shape (B,) indicating which camera
                response function to use for each element in the batch, can be -1 where to apply a simple gamma correction

        Returns:
            Tensor of same shape as input rgb after applying the camera-specific
            color response functions
        """
        assert unique_camera_idcs.shape[0] == rgb.shape[0], "Batch sizes must match"
        assert unique_camera_idcs.max() < self.num_cameras, "Camera indices out of bounds"
        assert unique_camera_idcs.min() >= -1, "Camera indices must be non-negative or -1"

        output = torch.zeros_like(rgb)
        for cam_idx in range(self.num_cameras):
            cam_mask = unique_camera_idcs == cam_idx
            if not cam_mask.any():
                continue

            cam_curves = self.curves[cam_idx]
            for channel in range(3):
                masked_values = rgb[cam_mask, channel]
                output[cam_mask, channel] = cam_curves[channel](masked_values)

        # Clamped changed pixels after the CRF, -1 means simple gamma correction
        has_camera = unique_camera_idcs != -1
        output = torch.where(has_camera.unsqueeze(-1), output.clamp(0.0, 1.0), rgb.clamp(0.0, 1.0).pow(1.0 / 2.2))

        return output


class BasePPISP(nn.Module, ABC):
    """
    An abstract base class for PPISP implementations. It defines a contract
    for providing parameters in a packed, vectorized format suitable for
    efficient, vectorized loss computation.
    """

    _smoothness_src_indices: torch.Tensor
    _smoothness_dst_indices: torch.Tensor
    n_frames_per_camera: list[int]
    num_cameras: int
    num_frames: int

    @property
    @abstractmethod
    def packed_exposure_params(self) -> torch.Tensor:
        """Returns packed exposure offsets. Shape: (num_frames,)."""
        ...

    @property
    @abstractmethod
    def packed_vignetting_params(self) -> "RadialFalloff.PackedParams":
        """
        Returns packed vignetting parameters.
        Shape: (num_cameras, 3, 5) -> [center_x, center_y, alpha1, alpha2, alpha3].
        """
        ...

    @property
    @abstractmethod
    def packed_color_params(self) -> torch.Tensor:
        """
        Returns packed color correction parameters.
        Shape: (num_frames, 8).
        """
        ...

    @property
    @abstractmethod
    def default_source_chroms(self) -> torch.Tensor:
        """
        Returns the default source chromaticities buffer.
        Shape: (4, 2).
        """
        ...

    @property
    @abstractmethod
    def crf_curve_points(self) -> PiecewisePowerFunction.CurvePoints:
        """
        Returns computed CRF curve points.
        Returns a CurvePoints object containing curve parameters in order:
        [x0, y0, slope_p0, y0_pre_gamma, slope_line, gamma, x1, y1, slope_p1, shoulder_x, shoulder_y].
        """
        ...

    @property
    @abstractmethod
    def smoothness_src_indices(self) -> torch.Tensor:
        """Source indices for calculating temporal smoothness loss."""
        ...

    @property
    @abstractmethod
    def smoothness_dst_indices(self) -> torch.Tensor:
        """Destination indices for calculating temporal smoothness loss."""
        ...

    # "Virtual" modules for optimizer config compatibility
    @property
    @abstractmethod
    def exposure(self) -> Union[nn.Module, nn.Parameter]: ...

    @property
    @abstractmethod
    def vignetting(self) -> Union[nn.Module, nn.Parameter]: ...

    @property
    @abstractmethod
    def color(self) -> Union[nn.Module, nn.Parameter]: ...

    @property
    @abstractmethod
    def crf(self) -> Union[nn.Module, nn.Parameter]: ...

    @abstractmethod
    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        unique_camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ) -> torch.Tensor:
        """The standard forward pass for rendering."""
        ...


class PPISP(BasePPISP):
    """
    Physically Plausible Image Signal Processing (PPISP) pipeline that models camera-specific image
    processing operations.

    The pipeline applies the following operations in sequence:
    1. Exposure compensation
    2. Vignetting correction
    3. Color correction (including white balance)
    4. Camera Response Function (CRF)
    """

    def __init__(
        self, device: torch.device | str, n_frames_per_camera: list[int], num_cameras: int, num_frames: int
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_frames_per_camera = n_frames_per_camera
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        # Initialize sub-models
        self._exposure = ExposureOffset(device, n_frames_per_camera, num_frames)
        self._vignetting = Vignetting(device, num_cameras)
        self._color = ColorCorrection(device, n_frames_per_camera, num_frames)
        self._crf = CRF(device, num_cameras)

        # Pre-compute indices for smoothness losses across frames, excluding boundaries across cameras.
        smoothness_src_indices_list = []
        smoothness_dst_indices_list = []

        current_idx = 0
        for n_frames in n_frames_per_camera:
            if n_frames > 1:
                src_indices = torch.arange(current_idx, current_idx + n_frames - 1, device=self.device)
                dst_indices = torch.arange(current_idx + 1, current_idx + n_frames, device=self.device)
                smoothness_src_indices_list.append(src_indices)
                smoothness_dst_indices_list.append(dst_indices)
            current_idx += n_frames

        # Concatenate all indices, handling empty lists and storing as non-parameter buffers
        if smoothness_src_indices_list:
            self.register_buffer("_smoothness_src_indices", torch.cat(smoothness_src_indices_list))
            self.register_buffer("_smoothness_dst_indices", torch.cat(smoothness_dst_indices_list))
        else:
            self.register_buffer("_smoothness_src_indices", torch.tensor([], device=self.device, dtype=torch.long))
            self.register_buffer("_smoothness_dst_indices", torch.tensor([], device=self.device, dtype=torch.long))

    @property
    def exposure(self) -> ExposureOffset:
        return self._exposure

    @property
    def vignetting(self) -> Vignetting:
        return self._vignetting

    @property
    def color(self) -> ColorCorrection:
        return self._color

    @property
    def crf(self) -> CRF:
        return self._crf

    @property
    def smoothness_src_indices(self) -> torch.Tensor:
        return self._smoothness_src_indices

    @property
    def smoothness_dst_indices(self) -> torch.Tensor:
        return self._smoothness_dst_indices

    @property
    def packed_exposure_params(self) -> torch.Tensor:
        """Gathers and packs exposure parameters."""
        return self.exposure.exposure_params.squeeze(-1)

    @property
    def packed_vignetting_params(self) -> RadialFalloff.PackedParams:
        """Gathers and packs vignetting parameters from sub-modules."""
        all_cam_params = []
        for cam_curves in self.vignetting.falloff_curves:
            all_channel_params = []
            for channel_curve in cam_curves:
                packed_channel_params = torch.cat([channel_curve.optical_center, channel_curve.alpha])
                all_channel_params.append(packed_channel_params)
            all_cam_params.append(torch.stack(all_channel_params, dim=0))
        return RadialFalloff.PackedParams(torch.stack(all_cam_params, dim=0))

    @property
    def packed_color_params(self) -> torch.Tensor:
        """Gathers and packs color correction parameters."""
        return self.color.color_params

    @property
    def default_source_chroms(self) -> torch.Tensor:
        """Returns the default source chromaticities buffer."""
        return self.color.default_source_chroms

    @property
    def crf_curve_points(self) -> PiecewisePowerFunction.CurvePoints:
        """
        Computes CRF curve points from raw parameters using vectorized method.
        Shape: (num_cameras, 3, 11)
        """
        # Collect curve points from all cameras and channels
        all_cam_x0, all_cam_y0, all_cam_slope_p0 = [], [], []
        all_cam_y0_pre_gamma, all_cam_slope_line, all_cam_gamma = [], [], []
        all_cam_x1, all_cam_y1, all_cam_slope_p1 = [], [], []
        all_cam_shoulder_x, all_cam_shoulder_y = [], []

        for cam_idx in range(self.num_cameras):
            cam_x0, cam_y0, cam_slope_p0 = [], [], []
            cam_y0_pre_gamma, cam_slope_line, cam_gamma = [], [], []
            cam_x1, cam_y1, cam_slope_p1 = [], [], []
            cam_shoulder_x, cam_shoulder_y = [], []

            for channel in range(3):
                # Get the raw params from this specific curve
                raw_params = PiecewisePowerFunction.RawParams(self.crf.curves[cam_idx][channel].raw_params)
                # Compute curve points
                curve_points = PiecewisePowerFunction.crf_curve_points(raw_params)

                # Collect individual tensors
                cam_x0.append(curve_points.x0)
                cam_y0.append(curve_points.y0)
                cam_slope_p0.append(curve_points.slope_p0)
                cam_y0_pre_gamma.append(curve_points.y0_pre_gamma)
                cam_slope_line.append(curve_points.slope_line)
                cam_gamma.append(curve_points.gamma)
                cam_x1.append(curve_points.x1)
                cam_y1.append(curve_points.y1)
                cam_slope_p1.append(curve_points.slope_p1)
                cam_shoulder_x.append(curve_points.shoulder_x)
                cam_shoulder_y.append(curve_points.shoulder_y)

            # Stack channel data for this camera
            all_cam_x0.append(torch.stack(cam_x0))
            all_cam_y0.append(torch.stack(cam_y0))
            all_cam_slope_p0.append(torch.stack(cam_slope_p0))
            all_cam_y0_pre_gamma.append(torch.stack(cam_y0_pre_gamma))
            all_cam_slope_line.append(torch.stack(cam_slope_line))
            all_cam_gamma.append(torch.stack(cam_gamma))
            all_cam_x1.append(torch.stack(cam_x1))
            all_cam_y1.append(torch.stack(cam_y1))
            all_cam_slope_p1.append(torch.stack(cam_slope_p1))
            all_cam_shoulder_x.append(torch.stack(cam_shoulder_x))
            all_cam_shoulder_y.append(torch.stack(cam_shoulder_y))

        # Stack camera data
        return PiecewisePowerFunction.CurvePoints(
            x0=torch.stack(all_cam_x0),
            y0=torch.stack(all_cam_y0),
            slope_p0=torch.stack(all_cam_slope_p0),
            y0_pre_gamma=torch.stack(all_cam_y0_pre_gamma),
            slope_line=torch.stack(all_cam_slope_line),
            gamma=torch.stack(all_cam_gamma),
            x1=torch.stack(all_cam_x1),
            y1=torch.stack(all_cam_y1),
            slope_p1=torch.stack(all_cam_slope_p1),
            shoulder_x=torch.stack(all_cam_shoulder_x),
            shoulder_y=torch.stack(all_cam_shoulder_y),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        unique_camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process input RGB values through the complete ISP pipeline.

        Args:
            rgb: Input RGB values of shape (B, 3)
            coords_xy: Normalized image coordinates of shape (B, 2)
            unique_camera_idcs: Camera indices of shape (B,)
            frame_idcs: Frame indices of shape (B,)

        Returns:
            Processed RGB values of same shape as input after applying the complete
            ISP pipeline including exposure, vignetting, color correction, and
            camera response function
        """
        assert rgb.shape[0] == coords_xy.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == unique_camera_idcs.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == frame_idcs.shape[0], "Batch sizes must match"

        assert unique_camera_idcs.max() < self.num_cameras, "Camera indices out of bounds"
        assert unique_camera_idcs.min() >= -1, "Camera indices must be non-negative or -1"

        assert frame_idcs.max() < self.num_frames, "Frame indices out of bounds"
        assert frame_idcs.min() >= -1, "Frame indices must be non-negative or -1"

        unique_camera_idcs = unique_camera_idcs.to(self.device, dtype=torch.int32)

        out = self.exposure(rgb, frame_idcs)
        out = self.vignetting(out, unique_camera_idcs, coords_xy)
        out = self.color(out, frame_idcs)
        out = self.crf(out, unique_camera_idcs)

        return out


class PPISPSlang(BasePPISP):
    _default_source_chroms: Tensor

    def __init__(
        self, device: torch.device | str, n_frames_per_camera: list[int], num_cameras: int, num_frames: int
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_frames_per_camera = n_frames_per_camera
        self.num_cameras = num_cameras
        self.num_frames = num_frames

        self.exposure_params = nn.Parameter(torch.zeros(num_frames, device=self.device))

        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5, device=self.device))

        # Initialize homography as identity matrix [1,0,0,0,1,0,0,0] for each frame
        identity_homography = torch.zeros(num_frames, 8, device=self.device)
        identity_homography[:, 0] = 1.0  # H[0,0] = 1
        identity_homography[:, 4] = 1.0  # H[1,1] = 1
        self.color_params = nn.Parameter(identity_homography)

        # Initialize CRF parameters using the shared function
        initial_raw_params = PiecewisePowerFunction.get_crf_raw_param_values()  # Use default values

        # Parameter order: [x0_offset, y0, y1_fract, toe_length, shoulder_length, shoulder_overshoot, gamma]
        # Broadcast the same parameters across all cameras and color channels
        self.crf_params = nn.Parameter(initial_raw_params.data.expand(num_cameras, 3, 7))

        # Pre-compute indices for smoothness losses across frames, excluding boundaries across cameras.
        smoothness_src_indices_list = []
        smoothness_dst_indices_list = []

        current_idx = 0
        for n_frames in n_frames_per_camera:
            if n_frames > 1:
                src_indices = torch.arange(current_idx, current_idx + n_frames - 1, device=self.device)
                dst_indices = torch.arange(current_idx + 1, current_idx + n_frames, device=self.device)
                smoothness_src_indices_list.append(src_indices)
                smoothness_dst_indices_list.append(dst_indices)
            current_idx += n_frames

        # Concatenate all indices, handling empty lists and storing as non-parameter buffers
        if smoothness_src_indices_list:
            self.register_buffer("_smoothness_src_indices", torch.cat(smoothness_src_indices_list))
            self.register_buffer("_smoothness_dst_indices", torch.cat(smoothness_dst_indices_list))
        else:
            self.register_buffer("_smoothness_src_indices", torch.tensor([], device=self.device, dtype=torch.long))
            self.register_buffer("_smoothness_dst_indices", torch.tensor([], device=self.device, dtype=torch.long))

        # Register default source chromaticities as a buffer to avoid CUDA sync on each access
        self.register_buffer(
            "_default_source_chroms",
            torch.tensor(DEFAULT_SOURCE_CHROMS_VALUES, device=self.device),
        )

        # Makes sure parameters are on the correct device, even when launched out of the main pipeline
        self.to(self.device)

    @property
    def packed_exposure_params(self) -> torch.Tensor:
        return self.exposure_params

    @property
    def packed_vignetting_params(self) -> RadialFalloff.PackedParams:
        return RadialFalloff.PackedParams(self.vignetting_params)

    @property
    def packed_color_params(self) -> torch.Tensor:
        return self.color_params

    @property
    def default_source_chroms(self) -> torch.Tensor:
        """Returns the default source chromaticities buffer."""
        return self._default_source_chroms

    @property
    def crf_curve_points(self) -> PiecewisePowerFunction.CurvePoints:
        """
        Computes CRF curve points from raw parameters using vectorized method.
        Shape: (num_cameras, 3, 11)
        """
        # Use vectorized computation to get curve points directly from raw parameters
        raw_params = PiecewisePowerFunction.RawParams(self.crf_params)
        return PiecewisePowerFunction.crf_curve_points(raw_params)

    # "Virtual" modules for optimizer config compatibility
    @property
    def exposure(self) -> nn.Parameter:
        return self.exposure_params

    @property
    def vignetting(self) -> nn.Parameter:
        return self.vignetting_params

    @property
    def color(self) -> nn.Parameter:
        return self.color_params

    @property
    def crf(self) -> nn.Parameter:
        return self.crf_params

    @property
    def smoothness_src_indices(self) -> torch.Tensor:
        return self._smoothness_src_indices

    @property
    def smoothness_dst_indices(self) -> torch.Tensor:
        return self._smoothness_dst_indices

    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        unique_camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ) -> torch.Tensor:
        # Shape validations
        assert rgb.shape[0] == coords_xy.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == unique_camera_idcs.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == frame_idcs.shape[0], "Batch sizes must match"
        assert rgb.shape[1] == 3, f"Expected rgb to have 3 channels, got {rgb.shape[1]}"
        assert coords_xy.shape[1] == 2, f"Expected coords_xy to have 2 dimensions (x,y), got {coords_xy.shape[1]}"

        return PPISPSlangFunction.apply(
            rgb.shape[0],
            self.num_cameras,
            self.num_frames,
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
            rgb,
            coords_xy,
            unique_camera_idcs,
            frame_idcs,
        )
