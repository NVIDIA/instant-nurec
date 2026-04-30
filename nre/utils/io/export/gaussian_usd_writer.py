# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
USD Gaussian prim writers for exporting Gaussian splatting data.

Provides a schema-agnostic interface for writing Gaussian splatting data to USD.
This abstraction allows switching between different USD schemas (UsdGeomPoints,
ParticleField3DGaussianSplat) without changing the export logic.

Writers handle the creation and population of USD prims with Gaussian attributes
including positions, rotations, scales, opacities, and appearance data.

Supported schemas:
- GaussianGeomPointsWriter: Legacy UsdGeomPoints schema with custom primvars
- GaussianLightFieldWriter: OpenUSD UsdVol ParticleField3DGaussianSplat schema
"""

import logging

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdVol, Vt

from nre.datasets.summary import DataSourceSummary
from nre.utils.io.export.gaussian_export_accessor import GaussianAttributes, ModelCapabilities
from nre.utils.io.utils import sanitize_usd_path
from nre.utils.misc import unpack_optional


log = logging.getLogger(__name__)


# =============================================================================
# Schema Type Enum
# =============================================================================


class GaussianUSDSchemaType(str, Enum):
    """Available USD schemas for Gaussian representation."""

    GEOM_POINTS = "geompoints"
    LIGHT_FIELD = "lightfield"


# =============================================================================
# Abstract Base Class
# =============================================================================


class GaussianUSDWriter(ABC):
    """Abstract base class for USD Gaussian prim writers.

    Writers are responsible for creating and populating USD prims with Gaussian
    splatting data. Different implementations can use different USD schemas while
    maintaining a common interface.

    The writer handles:
    - Prim creation and hierarchy
    - Material binding
    - Attribute authoring (static and animated)
    - Time sampling coordination

    Args:
        stage: USD stage to write prims to
        capabilities: Model capabilities descriptor
        track_id: Track identifier
        track_index: Track ordering index
        node_layer_name: Optional node layer name for composite models
        content_root_path: Root path for content in USD stage
    """

    def __init__(
        self,
        stage: Usd.Stage,
        capabilities: ModelCapabilities,
        track_id: str,
        track_index: int,
        node_layer_name: Optional[str] = None,
        content_root_path: str = "/World",
        linear_srgb: bool = False,
        has_post_processing: bool = False,
    ):
        self.stage = stage
        self.capabilities = capabilities
        self.track_id = track_id
        self.track_index = track_index
        self.node_layer_name = node_layer_name
        self.content_root_path = content_root_path
        self.linear_srgb = linear_srgb
        self.has_post_processing = has_post_processing
        self.prim = None
        self.track_xform = None
        self._track_visibility_attr = None

    @abstractmethod
    def create_prim(
        self,
        num_gaussians: int,
        track_transforms: Optional[Dict[float, Gf.Matrix4d]] = None,
    ) -> Usd.Prim:
        """Create the USD prim for this track.

        Args:
            num_gaussians: Number of gaussians in this track
            track_transforms: Optional dict mapping time codes to transformation matrices

        Returns:
            Created USD prim
        """
        pass

    @abstractmethod
    def write_static_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        do_not_cast_shadows: bool,
        force_sh_0: bool,
    ) -> None:
        """Write static attributes that don't change over time.

        Args:
            attributes: Gaussian attributes at initial timestamp
            gaussian_indices: Indices of gaussians in this track
            do_not_cast_shadows: Whether to disable shadow casting
            force_sh_0: Force SH degree to 0 (skip f_rest coefficients)
        """
        pass

    @abstractmethod
    def write_animated_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        time_code: float,
        animate_positions: bool,
        animate_rotations: bool,
        animate_scales: bool,
        animate_albedo: bool,
    ) -> None:
        """Write time-sampled attributes for animation.

        Args:
            attributes: Gaussian attributes at this timestamp
            gaussian_indices: Indices of gaussians in this track
            time_code: USD time code for this sample
            animate_positions: Whether positions are animated
            animate_rotations: Whether rotations are animated
            animate_scales: Whether scales are animated
            animate_albedo: Whether albedo/appearance is animated
        """
        pass

    @abstractmethod
    def write_first_frame_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
    ) -> None:
        """Write attributes for the first frame (static or first of animation).

        This is called once to write the initial state, whether the model
        is static or animated.

        Args:
            attributes: Gaussian attributes at initial timestamp
            gaussian_indices: Indices of gaussians in this track
        """
        pass

    @abstractmethod
    def finalize(self, all_positions: List[np.ndarray]) -> None:
        """Finalize the prim after all attributes are written.

        Args:
            all_positions: List of all position arrays for extent computation
        """
        pass

    @abstractmethod
    def write_track_transform(self, time_code: float, transform: Gf.Matrix4d) -> None:
        """Write a track transform at a specific time code.

        Args:
            time_code: USD time code for this transform
            transform: 4x4 transformation matrix
        """
        pass

    def write_track_visibility(self, time_code: float, visible: bool) -> None:
        """Write track prim visibility at a time code (so track is hidden when out of range, matching validation).

        Default no-op; overridden in writers that have a track xform.
        """
        if self.track_xform is None:
            return
        visibility_attr = self._track_visibility_attr
        if visibility_attr is None:
            visibility_attr = UsdGeom.Imageable(self.track_xform).GetVisibilityAttr()
            if not visibility_attr:
                visibility_attr = UsdGeom.Imageable(self.track_xform).CreateVisibilityAttr()
            self._track_visibility_attr = visibility_attr
        token = UsdGeom.Tokens.visible if visible else UsdGeom.Tokens.invisible
        visibility_attr.Set(token, time_code)

    @abstractmethod
    def set_documentation(self, doc: str) -> None:
        """Set prim documentation string.

        Args:
            doc: Documentation string
        """
        pass

    def apply_color_space_to_prim(self, prim: Usd.Prim) -> None:
        """Apply ColorSpaceAPI and set color space based on linear_srgb flag.

        According to USD color space conventions:
        - lin_rec709_scene: Linear Rec.709 color space (for post-processed/linear RGB data)
        - srgb_rec709_display: sRGB Rec.709 color space (for gamma-encoded data)

        Args:
            prim: USD prim to apply color space to
        """
        if not self.capabilities.has_spherical_harmonics:
            return

        color_space = "lin_rec709_scene" if self.linear_srgb else "srgb_rec709_display"
        color_space_api = Usd.ColorSpaceAPI.Apply(prim)
        color_space_api.CreateColorSpaceNameAttr().Set(color_space)


# =============================================================================
# GeomPoints Writer (Legacy Schema)
# =============================================================================


USD_LOOKS_PATH = "/World/Looks"
USD_GAUSSIAN_MATERIAL_PATH = USD_LOOKS_PATH + "/GaussianEmissive"
USD_GAUSSIAN_SHADER_PATH = USD_GAUSSIAN_MATERIAL_PATH + "/Shader"
GAUSSIAN_MATERIAL_MDL_FILE = "GaussianEmissive.mdl"
GAUSSIAN_MATERIAL_NAME = "GaussianEmissive"

# ParticleField material constants (for LightField schema)
USD_PARTICLEFIELD_MATERIAL_PATH = USD_LOOKS_PATH + "/ParticleFieldEmissive"
USD_PARTICLEFIELD_SHADER_PATH = USD_PARTICLEFIELD_MATERIAL_PATH + "/Shader"
PARTICLEFIELD_MATERIAL_MDL_FILE = "ParticleFieldEmissive.mdl"
PARTICLEFIELD_MATERIAL_NAME = "ParticleFieldEmissive"

SH_C0_COEFFICIENT = 0.28209479177387814

DEFAULT_GAUSSIAN_COLOR = 0.5
DEFAULT_GAUSSIAN_WIDTH = 1.0
DEFAULT_GAUSSIAN_TYPE = 1


class GaussianGeomPointsWriter(GaussianUSDWriter):
    """USD Gaussian writer using UsdGeomPoints schema."""

    def __init__(
        self,
        stage: Usd.Stage,
        capabilities: ModelCapabilities,
        track_id: str,
        track_index: int,
        node_layer_name: Optional[str] = None,
        content_root_path: str = "/World",
        linear_srgb: bool = False,
        has_post_processing: bool = False,
    ) -> None:
        super().__init__(
            stage,
            capabilities,
            track_id,
            track_index,
            node_layer_name,
            content_root_path,
            linear_srgb,
            has_post_processing,
        )
        self.points_prim: Optional[UsdGeom.Points] = None
        self.primvars_api: Optional[UsdGeom.PrimvarsAPI] = None
        self.type_primvar: Optional[UsdGeom.Primvar] = None
        self.scale_primvars: List[UsdGeom.Primvar] = []
        self.rot_primvars: List[UsdGeom.Primvar] = []
        self.opacity_primvar: Optional[UsdGeom.Primvar] = None
        self.shadow_primvar: Optional[UsdGeom.Primvar] = None
        self.f_dc_primvars: List[UsdGeom.Primvar] = []
        self.f_rest_primvars: List[UsdGeom.Primvar] = []
        self.display_color_primvar: Optional[UsdGeom.Primvar] = None
        self._transform_op: Optional[UsdGeom.XformOp] = None

    def create_prim(
        self,
        num_gaussians: int,
        track_transforms: Optional[Dict[float, Gf.Matrix4d]] = None,
    ) -> Usd.Prim:
        cleaned_track_id = DataSourceSummary._clean_track_id_str(self.track_id)
        safe_track_id = sanitize_usd_path(cleaned_track_id)
        track_path = f"{self.content_root_path}/track_{self.track_index:05d}_{safe_track_id}"
        if self.node_layer_name:
            safe_node_layer = sanitize_usd_path(self.node_layer_name)
            points_name = f"{safe_node_layer}_{safe_track_id}_points"
        else:
            points_name = f"{safe_track_id}_points"
        points_path = f"{track_path}/{points_name}"
        self.track_xform = UsdGeom.Xform.Define(self.stage, track_path)
        # Create transform op for animated track transforms
        if self.track_xform is not None:
            self._transform_op = self.track_xform.AddTransformOp()
            # If transforms provided upfront, write them now
            if track_transforms is not None:
                for time_code, transform_matrix in track_transforms.items():
                    self._transform_op.Set(transform_matrix, time_code)
        self.points_prim = UsdGeom.Points.Define(self.stage, points_path)
        self.prim = self.points_prim.GetPrim()
        self._create_and_bind_material()
        self.primvars_api = UsdGeom.PrimvarsAPI(self.prim)
        self._create_primvars()
        self.apply_color_space_to_prim(self.prim)
        return self.prim

    def write_track_transform(self, time_code: float, transform: Gf.Matrix4d) -> None:
        """Write a track transform at a specific time code."""
        if self._transform_op is not None:
            self._transform_op.Set(transform, time_code)

    def _create_and_bind_material(self):
        material_prim = self._create_gaussian_material()
        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI(self.prim)
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)

    def _create_gaussian_material(self) -> Usd.Prim:
        looks_prim = self.stage.GetPrimAtPath(USD_LOOKS_PATH)
        if not looks_prim.IsValid():
            looks_prim = self.stage.DefinePrim(USD_LOOKS_PATH, "Scope")
        material_prim = self.stage.DefinePrim(USD_GAUSSIAN_MATERIAL_PATH, "Material")
        shader_prim = self.stage.DefinePrim(USD_GAUSSIAN_SHADER_PATH, "Shader")
        shader_prim.CreateAttribute(
            "info:implementationSource", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        ).Set("sourceAsset")
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False, variability=Sdf.VariabilityUniform
        ).Set(Sdf.AssetPath(GAUSSIAN_MATERIAL_MDL_FILE))
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier",
            Sdf.ValueTypeNames.Token,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(GAUSSIAN_MATERIAL_NAME)
        # Configure MDL shader parameters when post-processing is enabled
        # Post-processing handles color correction, so disable MDL's built-in correction
        if self.has_post_processing:
            shader_prim.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool).Set(False)
            shader_prim.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool).Set(False)
        outputs_out = shader_prim.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token)
        outputs_out.SetMetadata("renderType", "material")
        material = UsdShade.Material(material_prim)
        shader = UsdShade.Shader(shader_prim)
        for output_name in ["mdl:displacement", "mdl:surface", "mdl:volume"]:
            output = material.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
            output.ConnectToSource(shader.GetOutput("out"))
        return material_prim

    def _create_primvars(self) -> None:
        primvars_api: UsdGeom.PrimvarsAPI = unpack_optional(self.primvars_api, msg="primvars_api must be initialized")
        self.type_primvar = primvars_api.CreatePrimvar("type", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant)
        for i in range(3):
            scale_primvar = primvars_api.CreatePrimvar(
                f"scale_{i}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
            )
            self.scale_primvars.append(scale_primvar)
        for i in range(4):
            rot_primvar = primvars_api.CreatePrimvar(f"rot_{i}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            self.rot_primvars.append(rot_primvar)
        self.opacity_primvar = primvars_api.CreatePrimvar(
            "opacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        )
        if self.capabilities.has_spherical_harmonics:
            for i in range(3):
                f_dc_primvar = primvars_api.CreatePrimvar(
                    f"f_dc_{i}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
                )
                self.f_dc_primvars.append(f_dc_primvar)
            self.display_color_primvar = primvars_api.CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )
        else:
            self.display_color_primvar = primvars_api.CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )

    def write_static_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        do_not_cast_shadows: bool,
        force_sh_0: bool,
    ) -> None:
        points_prim: UsdGeom.Points = unpack_optional(self.points_prim, msg="create_prim must be called first")
        primvars_api: UsdGeom.PrimvarsAPI = unpack_optional(self.primvars_api, msg="create_prim must be called first")
        type_primvar: UsdGeom.Primvar = unpack_optional(self.type_primvar, msg="create_prim must be called first")
        opacity_primvar: UsdGeom.Primvar = unpack_optional(self.opacity_primvar, msg="create_prim must be called first")
        np_indices = gaussian_indices.cpu().numpy()
        num_gaussians = len(np_indices)
        uniform_widths = np.full(num_gaussians, DEFAULT_GAUSSIAN_WIDTH, dtype=np.float32)
        points_prim.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(uniform_widths))
        type_primvar.Set(DEFAULT_GAUSSIAN_TYPE)
        # Use np_indices to filter densities
        densities_filtered = attributes.densities[np_indices]
        opacity_primvar.Set(Vt.FloatArray.FromNumpy(densities_filtered.flatten()))
        if do_not_cast_shadows:
            self.shadow_primvar = primvars_api.CreatePrimvar(
                "doNotCastShadows", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant
            )
            self.shadow_primvar.Set(True)
        if self.capabilities.has_spherical_harmonics and not force_sh_0:
            specular_coeffs = attributes.specular_coefficients
            if specular_coeffs is not None and specular_coeffs.size > 0:
                track_specular = specular_coeffs[np_indices]
                if track_specular.size > 0:
                    num_coeffs = track_specular.shape[-1] if track_specular.ndim > 1 else 0
                    if not self.f_rest_primvars:
                        for i in range(num_coeffs):
                            f_rest_primvar = primvars_api.CreatePrimvar(
                                f"f_rest_{i}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
                            )
                            self.f_rest_primvars.append(f_rest_primvar)
                    num_sh_coeffs = track_specular.shape[-1] // 3
                    track_specular_reshaped = track_specular.reshape((num_gaussians, num_sh_coeffs, 3))
                    track_specular_channel_major = (
                        track_specular_reshaped.transpose(2, 1, 0).reshape((num_sh_coeffs * 3, num_gaussians)).T
                    )
                    for i in range(len(self.f_rest_primvars)):
                        coeff_data = track_specular_channel_major[:, i]
                        self.f_rest_primvars[i].Set(Vt.FloatArray.FromNumpy(coeff_data))

    def write_animated_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        time_code: float,
        animate_positions: bool,
        animate_rotations: bool,
        animate_scales: bool,
        animate_albedo: bool,
    ) -> None:
        points_prim: UsdGeom.Points = unpack_optional(self.points_prim, msg="create_prim must be called first")
        display_color_primvar: UsdGeom.Primvar = unpack_optional(
            self.display_color_primvar, msg="create_prim must be called first"
        )
        # Filter attributes by gaussian_indices
        np_indices = gaussian_indices.cpu().numpy()
        positions_data = attributes.positions[np_indices]
        rotations_data = attributes.rotations[np_indices]
        scales_data = attributes.scales[np_indices]
        if animate_positions:
            points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions_data), time_code)
        if animate_scales:
            for i in range(3):
                self.scale_primvars[i].Set(Vt.FloatArray.FromNumpy(scales_data[:, i]), time_code)
        if animate_rotations:
            for i in range(4):
                self.rot_primvars[i].Set(Vt.FloatArray.FromNumpy(rotations_data[:, i]), time_code)
        if self.capabilities.has_spherical_harmonics and animate_albedo and attributes.albedo_coefficients is not None:
            albedo = attributes.albedo_coefficients[np_indices]
            for i in range(3):
                self.f_dc_primvars[i].Set(Vt.FloatArray.FromNumpy(albedo[:, i]), time_code)
            # Apply SH decode only if radiance_sph_O0 is False (coefficients need C0*x+0.5 transform)
            if self.capabilities.radiance_sph_O0:
                display_colors = np.maximum(0.0, albedo)
            else:
                display_colors = np.maximum(0.0, albedo * SH_C0_COEFFICIENT + DEFAULT_GAUSSIAN_COLOR)
            display_color_primvar.Set(Vt.Vec3fArray.FromNumpy(display_colors.astype(np.float32)), time_code)

    def write_first_frame_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
    ) -> None:
        points_prim: UsdGeom.Points = unpack_optional(self.points_prim, msg="create_prim must be called first")
        display_color_primvar: UsdGeom.Primvar = unpack_optional(
            self.display_color_primvar, msg="create_prim must be called first"
        )
        np_indices = gaussian_indices.cpu().numpy()
        num_gaussians = len(np_indices)
        # Filter attributes by gaussian_indices
        positions_data = attributes.positions[np_indices]
        rotations_data = attributes.rotations[np_indices]
        scales_data = attributes.scales[np_indices]
        points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions_data))
        for i in range(3):
            self.scale_primvars[i].Set(Vt.FloatArray.FromNumpy(scales_data[:, i]))
        for i in range(4):
            self.rot_primvars[i].Set(Vt.FloatArray.FromNumpy(rotations_data[:, i]))
        if self.capabilities.has_spherical_harmonics and attributes.albedo_coefficients is not None:
            albedo = attributes.albedo_coefficients[np_indices]
            for i in range(3):
                self.f_dc_primvars[i].Set(Vt.FloatArray.FromNumpy(albedo[:, i]))
            # Apply SH decode only if radiance_sph_O0 is False (coefficients need C0*x+0.5 transform)
            if self.capabilities.radiance_sph_O0:
                display_colors = np.maximum(0.0, albedo)
            else:
                display_colors = np.maximum(0.0, albedo * SH_C0_COEFFICIENT + DEFAULT_GAUSSIAN_COLOR)
            display_color_primvar.Set(Vt.Vec3fArray.FromNumpy(display_colors.astype(np.float32)))
        else:
            default_colors = np.full((num_gaussians, 3), DEFAULT_GAUSSIAN_COLOR, dtype=np.float32)
            display_color_primvar.Set(Vt.Vec3fArray.FromNumpy(default_colors))

    def finalize(self, all_positions: List[np.ndarray]) -> None:
        points_prim: UsdGeom.Points = unpack_optional(self.points_prim, msg="create_prim must be called first")
        if all_positions:
            all_positions_combined = np.vstack(all_positions)
            min_bounds = np.min(all_positions_combined, axis=0)
            max_bounds = np.max(all_positions_combined, axis=0)
            extent_range = Vt.Vec3fArray(
                [
                    Gf.Vec3f(float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2])),
                    Gf.Vec3f(float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])),
                ]
            )
            points_prim.GetExtentAttr().Set(extent_range)

    def set_documentation(self, doc: str) -> None:
        prim: Usd.Prim = unpack_optional(self.prim, msg="create_prim must be called first")
        prim.SetDocumentation(doc)


# =============================================================================
# LightField Writer (OpenUSD ParticleField Schema)
# =============================================================================


class GaussianLightFieldWriter(GaussianUSDWriter):
    """USD Gaussian writer using UsdVol ParticleField schema.

    Uses ParticleField3DGaussianSplat (3DGS) or ParticleField with applied API schemas
    (2DGS/surflet). Requires usd-core>=26.3 for UsdVol schema API.
    Reference: https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/pxr/usd/usdVol/schema.usda
    """

    def __init__(
        self,
        stage: Usd.Stage,
        capabilities: ModelCapabilities,
        track_id: str,
        track_index: int,
        node_layer_name: Optional[str] = None,
        content_root_path: str = "/World",
        projection_mode_hint: str = "perspective",
        sorting_mode_hint: str = "zDepth",
        half_precision: bool = False,
        half_geometry: Optional[bool] = None,
        half_features: Optional[bool] = None,
        linear_srgb: bool = False,
        has_post_processing: bool = False,
    ) -> None:
        super().__init__(
            stage,
            capabilities,
            track_id,
            track_index,
            node_layer_name,
            content_root_path,
            linear_srgb,
            has_post_processing,
        )
        self.particle_field_prim: Optional[Usd.Prim] = None
        self.positions_attr: Optional[Usd.Attribute] = None
        self.orientations_attr: Optional[Usd.Attribute] = None
        self.scales_attr: Optional[Usd.Attribute] = None
        self.opacities_attr: Optional[Usd.Attribute] = None
        self.sh_coeffs_attr: Optional[Usd.Attribute] = None
        self.sh_degree_attr: Optional[Usd.Attribute] = None
        self._transform_op: Optional[UsdGeom.XformOp] = None
        self._schema: Optional[UsdVol.ParticleField3DGaussianSplat] = None
        self.projection_mode_hint = projection_mode_hint
        self.sorting_mode_hint = sorting_mode_hint
        self.half_geometry = half_geometry if half_geometry is not None else half_precision
        self.half_features = half_features if half_features is not None else half_precision

    def create_prim(
        self,
        num_gaussians: int,
        track_transforms: Optional[Dict[float, Gf.Matrix4d]] = None,
    ) -> Usd.Prim:
        cleaned_track_id = DataSourceSummary._clean_track_id_str(self.track_id)
        safe_track_id = sanitize_usd_path(cleaned_track_id)
        track_path = f"{self.content_root_path}/track_{self.track_index:05d}_{safe_track_id}"
        if self.node_layer_name:
            safe_node_layer = sanitize_usd_path(self.node_layer_name)
            prim_name = f"{safe_node_layer}_{safe_track_id}_gaussians"
        else:
            prim_name = f"{safe_track_id}_gaussians"
        prim_path = f"{track_path}/{prim_name}"
        self.track_xform = UsdGeom.Xform.Define(self.stage, track_path)
        # Create transform op for animated track transforms
        if self.track_xform is not None:
            self._transform_op = self.track_xform.AddTransformOp()
            # If transforms provided upfront, write them now
            if track_transforms is not None:
                for time_code, transform_matrix in track_transforms.items():
                    self._transform_op.Set(transform_matrix, time_code)

        if self.capabilities.is_planar_gaussian:
            self.prim = UsdVol.ParticleField.Define(self.stage, prim_path).GetPrim()
            self._apply_surflet_kernel_schemas()
            log.info(f"Created ParticleField with GaussianSurfletAPI (2DGS/surfel) at {prim_path}")
        else:
            self.prim = UsdVol.ParticleField3DGaussianSplat.Define(self.stage, prim_path).GetPrim()
            self._schema = UsdVol.ParticleField3DGaussianSplat(self.prim)
            log.info(f"Created ParticleField3DGaussianSplat at {prim_path}")

        self._create_and_bind_material()
        self._create_attributes()
        self._set_rendering_hints()
        self.apply_color_space_to_prim(self.prim)
        return self.prim

    def write_track_transform(self, time_code: float, transform: Gf.Matrix4d) -> None:
        """Write a track transform at a specific time code."""
        if self._transform_op is not None:
            self._transform_op.Set(transform, time_code)

    def _apply_surflet_kernel_schemas(self) -> None:
        """Apply API schemas for 2DGS/surfel particles via UsdVol schema types."""
        prim: Usd.Prim = unpack_optional(self.prim, msg="create_prim must be called first")
        for api_schema in (
            UsdVol.ParticleFieldPositionAttributeAPI,
            UsdVol.ParticleFieldOrientationAttributeAPI,
            UsdVol.ParticleFieldScaleAttributeAPI,
            UsdVol.ParticleFieldOpacityAttributeAPI,
            UsdVol.ParticleFieldKernelGaussianSurfletAPI,
            UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI,
        ):
            prim.ApplyAPI(api_schema)

    def _create_and_bind_material(self):
        """Create and bind ParticleFieldEmissive material for LightField schema."""
        material_prim = self._create_particlefield_material()
        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI(self.prim)
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)

    def _create_particlefield_material(self) -> Usd.Prim:
        """Create a ParticleFieldEmissive material for LightField schema.

        This material uses spherical harmonics to render view-dependent Gaussian splats
        with proper MDL shader support for the ParticleField schema.

        Returns:
            The created Material prim
        """
        looks_prim = self.stage.GetPrimAtPath(USD_LOOKS_PATH)
        if not looks_prim.IsValid():
            looks_prim = self.stage.DefinePrim(USD_LOOKS_PATH, "Scope")

        material_prim = self.stage.DefinePrim(USD_PARTICLEFIELD_MATERIAL_PATH, "Material")
        shader_prim = self.stage.DefinePrim(USD_PARTICLEFIELD_SHADER_PATH, "Shader")

        shader_prim.CreateAttribute(
            "info:implementationSource", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        ).Set("sourceAsset")
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False, variability=Sdf.VariabilityUniform
        ).Set(Sdf.AssetPath(PARTICLEFIELD_MATERIAL_MDL_FILE))
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier",
            Sdf.ValueTypeNames.Token,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(PARTICLEFIELD_MATERIAL_NAME)

        # Configure MDL shader parameters when post-processing is enabled
        # Post-processing handles color correction, so disable MDL's built-in correction
        if self.has_post_processing:
            shader_prim.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool).Set(False)
            shader_prim.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool).Set(False)

        outputs_out = shader_prim.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token)
        outputs_out.SetMetadata("renderType", "material")

        material = UsdShade.Material(material_prim)
        shader = UsdShade.Shader(shader_prim)
        for output_name in ["mdl:displacement", "mdl:surface", "mdl:volume"]:
            output = material.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
            output.ConnectToSource(shader.GetOutput("out"))

        return material_prim

    def _create_attributes(self) -> None:
        """Create particle field attributes via UsdVol schema API.

        half_geometry: positions, orientations, scales use *h (half) attributes.
        half_features: opacities and SH coefficients use *h (half) attributes.
        """
        if self._schema is not None:
            self.positions_attr = (
                self._schema.CreatePositionshAttr() if self.half_geometry else self._schema.CreatePositionsAttr()
            )
            self.orientations_attr = (
                self._schema.CreateOrientationshAttr() if self.half_geometry else self._schema.CreateOrientationsAttr()
            )
            self.scales_attr = (
                self._schema.CreateScaleshAttr() if self.half_geometry else self._schema.CreateScalesAttr()
            )
            self.opacities_attr = (
                self._schema.CreateOpacitieshAttr() if self.half_features else self._schema.CreateOpacitiesAttr()
            )
            if self.capabilities.has_spherical_harmonics:
                self.sh_degree_attr = self._schema.CreateRadianceSphericalHarmonicsDegreeAttr()
                self.sh_coeffs_attr = (
                    self._schema.CreateRadianceSphericalHarmonicsCoefficientshAttr()
                    if self.half_features
                    else self._schema.CreateRadianceSphericalHarmonicsCoefficientsAttr()
                )
        else:
            pos_api = UsdVol.ParticleFieldPositionAttributeAPI(self.prim)
            orient_api = UsdVol.ParticleFieldOrientationAttributeAPI(self.prim)
            scale_api = UsdVol.ParticleFieldScaleAttributeAPI(self.prim)
            opacity_api = UsdVol.ParticleFieldOpacityAttributeAPI(self.prim)
            self.positions_attr = (
                pos_api.CreatePositionshAttr() if self.half_geometry else pos_api.CreatePositionsAttr()
            )
            self.orientations_attr = (
                orient_api.CreateOrientationshAttr() if self.half_geometry else orient_api.CreateOrientationsAttr()
            )
            self.scales_attr = scale_api.CreateScaleshAttr() if self.half_geometry else scale_api.CreateScalesAttr()
            self.opacities_attr = (
                opacity_api.CreateOpacitieshAttr() if self.half_features else opacity_api.CreateOpacitiesAttr()
            )
            if self.capabilities.has_spherical_harmonics:
                rad_api = UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI(self.prim)
                self.sh_degree_attr = rad_api.CreateRadianceSphericalHarmonicsDegreeAttr()
                self.sh_coeffs_attr = (
                    rad_api.CreateRadianceSphericalHarmonicsCoefficientshAttr()
                    if self.half_features
                    else rad_api.CreateRadianceSphericalHarmonicsCoefficientsAttr()
                )
        if self.half_geometry or self.half_features:
            log.info(
                "LightField precision: geometry=%s, features=%s",
                "half" if self.half_geometry else "float",
                "half" if self.half_features else "float",
            )

    def _set_rendering_hints(self) -> None:
        """Set rendering hints (same names/types as ParticleField3DGaussianSplat).

        projectionModeHint: 'perspective' or 'tangential'
        sortingModeHint: 'zDepth' or 'cameraDistance'
        For ellipsoid we use the schema; for surflet we create attributes on the prim so
        downstream can read them the same way.
        """
        if self._schema is not None:
            self._schema.CreateProjectionModeHintAttr().Set(self.projection_mode_hint)
            self._schema.CreateSortingModeHintAttr().Set(self.sorting_mode_hint)
        else:
            prim: Usd.Prim = unpack_optional(self.prim, msg="create_prim must be called first")
            prim.CreateAttribute(
                "projectionModeHint",
                Sdf.ValueTypeNames.Token,
                custom=False,
                variability=Sdf.VariabilityUniform,
            ).Set(self.projection_mode_hint)
            prim.CreateAttribute(
                "sortingModeHint",
                Sdf.ValueTypeNames.Token,
                custom=False,
                variability=Sdf.VariabilityUniform,
            ).Set(self.sorting_mode_hint)

    def write_static_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        do_not_cast_shadows: bool,
        force_sh_0: bool,
    ) -> None:
        opacities_attr: Usd.Attribute = unpack_optional(self.opacities_attr, msg="create_prim must be called first")
        np_indices = gaussian_indices.cpu().numpy()
        num_gaussians = len(np_indices)
        # Filter densities by gaussian_indices
        densities_filtered = attributes.densities[np_indices]
        densities_clamped = np.clip(densities_filtered, 0.0, 1.0).flatten()

        if self.half_features:
            opacities_attr.Set(Vt.HalfArray.FromNumpy(densities_clamped.astype(np.float16)))
        else:
            opacities_attr.Set(Vt.FloatArray.FromNumpy(densities_clamped.astype(np.float32)))

        if self.capabilities.has_spherical_harmonics:
            sh_degree = 0 if force_sh_0 else self.capabilities.sh_degree
            if sh_degree is not None and self.sh_degree_attr is not None:
                self.sh_degree_attr.Set(sh_degree)
            albedo_coeffs = attributes.albedo_coefficients
            specular_coeffs = attributes.specular_coefficients
            if albedo_coeffs is not None and self.sh_coeffs_attr is not None:
                track_albedo = albedo_coeffs[np_indices]
                if force_sh_0:
                    # SH degree 0: write only DC term (albedo)
                    all_coeffs_flat = track_albedo.reshape(-1, 3)
                    num_sh_coeffs = 1
                elif specular_coeffs is not None and sh_degree is not None:
                    # Full SH: write albedo + specular combined
                    track_specular = specular_coeffs[np_indices]
                    num_sh_coeffs = (sh_degree + 1) ** 2
                    num_rest_coeffs = num_sh_coeffs - 1
                    if track_specular.shape[-1] != num_rest_coeffs * 3:
                        return
                    track_specular_reshaped = track_specular.reshape((num_gaussians, num_rest_coeffs, 3))
                    track_albedo_expanded = track_albedo.reshape((num_gaussians, 1, 3))
                    all_coeffs = np.concatenate([track_albedo_expanded, track_specular_reshaped], axis=1)
                    all_coeffs_flat = all_coeffs.reshape(-1, 3)
                else:
                    return
                if self.half_features:
                    self.sh_coeffs_attr.Set(Vt.Vec3hArray.FromNumpy(all_coeffs_flat.astype(np.float16)))
                else:
                    self.sh_coeffs_attr.Set(Vt.Vec3fArray.FromNumpy(all_coeffs_flat.astype(np.float32)))
                self.sh_coeffs_attr.SetMetadata("elementSize", num_sh_coeffs)

    def _prepare_scales_for_export(self, scales_data: np.ndarray) -> np.ndarray:
        """Prepare scales for export, handling surflet kernel (z-component ignored).

        For surflet (planar) gaussians, the z-component of scale is irrelevant as the
        kernel is a 2D disk. We zero it out for clarity in the exported file.
        """
        if self.capabilities.is_planar_gaussian:
            scales_data = scales_data.copy()
            scales_data[:, 2] = 0.0  # Zero out z-component for surflets
        return scales_data

    def write_animated_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
        time_code: float,
        animate_positions: bool,
        animate_rotations: bool,
        animate_scales: bool,
        animate_albedo: bool,
    ) -> None:
        positions_attr: Usd.Attribute = unpack_optional(self.positions_attr, msg="create_prim must be called first")
        orientations_attr: Usd.Attribute = unpack_optional(
            self.orientations_attr, msg="create_prim must be called first"
        )
        scales_attr: Usd.Attribute = unpack_optional(self.scales_attr, msg="create_prim must be called first")
        # Filter attributes by gaussian_indices
        np_indices = gaussian_indices.cpu().numpy()
        positions_data = attributes.positions[np_indices]
        orientations_data = attributes.rotations[np_indices]
        scales_data = self._prepare_scales_for_export(attributes.scales[np_indices])

        if animate_positions:
            if self.half_geometry:
                positions_attr.Set(Vt.Vec3hArray.FromNumpy(positions_data.astype(np.float16)), time_code)
            else:
                positions_attr.Set(Vt.Vec3fArray.FromNumpy(positions_data.astype(np.float32)), time_code)
        if animate_rotations:
            if self.half_geometry:
                quats = [Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in orientations_data]
                orientations_attr.Set(Vt.QuathArray(quats), time_code)
            else:
                quats = [Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in orientations_data]
                orientations_attr.Set(Vt.QuatfArray(quats), time_code)
        if animate_scales:
            if self.half_geometry:
                scales_attr.Set(Vt.Vec3hArray.FromNumpy(scales_data.astype(np.float16)), time_code)
            else:
                scales_attr.Set(Vt.Vec3fArray.FromNumpy(scales_data.astype(np.float32)), time_code)
        if animate_albedo and self.capabilities.has_spherical_harmonics:
            # DC-term only for animated albedo here (schema supports full SH but not filled here)
            pass

    def write_first_frame_attributes(
        self,
        attributes: GaussianAttributes,
        gaussian_indices: torch.Tensor,
    ) -> None:
        positions_attr: Usd.Attribute = unpack_optional(self.positions_attr, msg="create_prim must be called first")
        orientations_attr: Usd.Attribute = unpack_optional(
            self.orientations_attr, msg="create_prim must be called first"
        )
        scales_attr: Usd.Attribute = unpack_optional(self.scales_attr, msg="create_prim must be called first")
        # Filter attributes by gaussian_indices
        np_indices = gaussian_indices.cpu().numpy()
        positions_data = attributes.positions[np_indices]
        orientations_data = attributes.rotations[np_indices]
        scales_data = self._prepare_scales_for_export(attributes.scales[np_indices])

        if self.half_geometry:
            positions_attr.Set(Vt.Vec3hArray.FromNumpy(positions_data.astype(np.float16)))
            quats = [Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in orientations_data]
            orientations_attr.Set(Vt.QuathArray(quats))
            scales_attr.Set(Vt.Vec3hArray.FromNumpy(scales_data.astype(np.float16)))
        else:
            positions_attr.Set(Vt.Vec3fArray.FromNumpy(positions_data.astype(np.float32)))
            quats = [Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in orientations_data]
            orientations_attr.Set(Vt.QuatfArray(quats))
            scales_attr.Set(Vt.Vec3fArray.FromNumpy(scales_data.astype(np.float32)))

    def finalize(self, all_positions: List[np.ndarray]) -> None:
        prim: Usd.Prim = unpack_optional(self.prim, msg="create_prim must be called first")
        if all_positions:
            all_positions_combined = np.vstack(all_positions)
            min_bounds = np.min(all_positions_combined, axis=0)
            max_bounds = np.max(all_positions_combined, axis=0)
            extent_range = Vt.Vec3fArray(
                [
                    Gf.Vec3f(float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2])),
                    Gf.Vec3f(float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])),
                ]
            )
            if self._schema is not None:
                self._schema.CreateExtentAttr().Set(extent_range)
            else:
                UsdGeom.Boundable(prim).CreateExtentAttr().Set(extent_range)

    def set_documentation(self, doc: str) -> None:
        prim: Usd.Prim = unpack_optional(self.prim, msg="create_prim must be called first")
        prim.SetDocumentation(doc)


# =============================================================================
# Factory Function
# =============================================================================


def create_gaussian_writer(
    schema_type: GaussianUSDSchemaType,
    stage: Usd.Stage,
    capabilities: ModelCapabilities,
    track_id: str,
    track_index: int,
    node_layer_name: Optional[str] = None,
    content_root_path: str = "/World",
    sorting_mode_hint: str = "cameraDistance",
    half_precision: bool = False,
    half_geometry: Optional[bool] = None,
    half_features: Optional[bool] = None,
    linear_srgb: bool = False,
    has_post_processing: bool = False,
) -> GaussianUSDWriter:
    """Factory function to create appropriate USD Gaussian writer.

    Args:
        schema_type: Type of USD schema to use
        stage: USD stage to write to
        capabilities: Model capabilities descriptor
        track_id: Track identifier
        track_index: Track ordering index
        node_layer_name: Optional node layer name
        content_root_path: Root path for content
        sorting_mode_hint: LightField sorting hint ("zDepth" or "cameraDistance", per UsdVol schema)
        half_precision: Use half for both geometry and features (backward compat)
        half_geometry: Half for positions/orientations/scales (overrides half_precision for geometry)
        half_features: Half for opacities and SH (overrides half_precision for features)
        linear_srgb: Whether to use linear sRGB color space (true if post-processing is attached)
        has_post_processing: Whether the model has post-processing (disables MDL color correction)

    Returns:
        Configured GaussianUSDWriter instance

    Raises:
        ValueError: If schema_type is not recognized
    """
    if schema_type == GaussianUSDSchemaType.GEOM_POINTS:
        return GaussianGeomPointsWriter(
            stage=stage,
            capabilities=capabilities,
            track_id=track_id,
            track_index=track_index,
            node_layer_name=node_layer_name,
            content_root_path=content_root_path,
            linear_srgb=linear_srgb,
            has_post_processing=has_post_processing,
        )
    elif schema_type == GaussianUSDSchemaType.LIGHT_FIELD:
        return GaussianLightFieldWriter(
            stage=stage,
            capabilities=capabilities,
            track_id=track_id,
            track_index=track_index,
            node_layer_name=node_layer_name,
            content_root_path=content_root_path,
            projection_mode_hint="perspective",
            sorting_mode_hint=sorting_mode_hint,
            half_precision=half_precision,
            half_geometry=half_geometry,
            half_features=half_features,
            linear_srgb=linear_srgb,
            has_post_processing=has_post_processing,
        )
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")
