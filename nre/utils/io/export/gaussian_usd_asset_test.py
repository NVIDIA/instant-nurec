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
Lightweight tests for USD Gaussian Asset export utilities.
"""

import tempfile
import unittest
import zipfile

from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from nre.models.background import EnvMapType, SkyEnvMapBackground
from nre.utils.io.export.gaussian_usd_asset import (
    DEFAULT_FRAME_RATE,
    SKY_DOME_LIGHT_PATH,
    USD_GAUSSIAN_MATERIAL_PATH,
    USD_GAUSSIAN_ROOT_PATH,
    USD_GAUSSIAN_SHADER_PATH,
    USD_WORLD_PATH,
    CameraRenderProductInfo,
    USDGaussianExportCache,
    apply_gaussian_subsampling,
    create_background_domelight,
    create_gaussian_material,
    create_gaussian_model_root,
    resample_timestamps,
    update_animation_settings,
)
from nre.utils.io.export.ppisp_usd_writer import (
    PPISP_INPUT_RENDER_VAR,
    PPISP_OUTPUT_RENDER_VAR,
    PPISP_SPG_LUA_FILE,
    PPISP_SPG_SLANG_FILE,
    PPISP_SPG_USDA_FILE,
    TrainingFrameFilterConfig,
    build_camera_frame_mappings,
    get_ppisp_spg_files,
)
from nre.utils.types import BBox3, FrameConversion, NamedSerialized, NamedUSDStage, RigTrajectories


class TestUtilityFunctions(unittest.TestCase):
    def test_resample_timestamps_basic(self):
        timestamps_us = [100, 200, 300, 400, 500]
        reference_timestamps_us = [0, 50, 150, 250, 350, 450, 550, 600]
        resampled = resample_timestamps(timestamps_us, reference_timestamps_us)
        self.assertGreaterEqual(resampled[0], 0)
        self.assertLessEqual(resampled[0], timestamps_us[0])
        self.assertGreaterEqual(resampled[-1], timestamps_us[-1])
        self.assertLessEqual(resampled[-1], reference_timestamps_us[-1])

    def test_resample_timestamps_empty(self):
        self.assertEqual(resample_timestamps([], [1, 2, 3]), [])
        self.assertEqual(resample_timestamps([1, 2, 3], []), [1, 2, 3])

    def test_apply_gaussian_subsampling_full(self):
        indices = torch.arange(1000)
        result = apply_gaussian_subsampling(indices, 100.0)
        self.assertEqual(len(result), 1000)
        torch.testing.assert_close(result, indices)

    def test_quaternion_xyzw_to_wxyz_conversion(self):
        # Test quaternion conversion (XYZW to WXYZ order)
        # This conversion is now handled in the writer implementations
        quat_xyzw = torch.tensor([[0.1, 0.2, 0.3, 0.9238]]).float()
        quat_xyzw = torch.nn.functional.normalize(quat_xyzw, dim=-1)
        # XYZW -> WXYZ reordering indices: [3, 0, 1, 2]
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        quat_wxyz = torch.nn.functional.normalize(quat_wxyz, dim=-1)
        self.assertAlmostEqual(quat_wxyz[0, 0].item(), quat_xyzw[0, 3].item(), places=5)


class TestUSDStageUtilities(unittest.TestCase):
    def test_create_gaussian_model_root_basic(self):
        stage = Usd.Stage.CreateInMemory()
        root_path = create_gaussian_model_root(
            stage, flip_x_axis=False, flip_y_axis=False, flip_z_axis=False, dataset_offset=None
        )
        self.assertEqual(root_path, USD_GAUSSIAN_ROOT_PATH)
        xform_prim = stage.GetPrimAtPath(root_path)
        self.assertTrue(xform_prim.IsValid())
        self.assertTrue(xform_prim.IsA(UsdGeom.Xform))

    def test_create_gaussian_material(self):
        stage = Usd.Stage.CreateInMemory()
        material_prim = create_gaussian_material(stage)
        self.assertTrue(material_prim.IsValid())
        self.assertTrue(material_prim.IsA(Usd.Typed))

    def test_update_animation_settings(self):
        root_stage = Usd.Stage.CreateInMemory()
        ref_stage = Usd.Stage.CreateInMemory()
        ref_layer = ref_stage.GetRootLayer()
        ref_layer.startTimeCode = 10.0
        ref_layer.endTimeCode = 100.0
        ref_layer.timeCodesPerSecond = 30.0
        ref_layer.customLayerData = {"absoluteTimeOffsetMicroSec": 500000}
        update_animation_settings(root_stage, ref_layer)
        self.assertEqual(root_stage.GetStartTimeCode(), 10.0)
        self.assertEqual(root_stage.GetEndTimeCode(), 100.0)
        self.assertEqual(root_stage.GetTimeCodesPerSecond(), 30.0)
        self.assertEqual(root_stage.GetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec"), 500000)


class TestUSDZPackaging(unittest.TestCase):
    def test_usdz_cache_initialization(self):
        cache = USDGaussianExportCache()
        self.assertEqual(len(cache.usd_stages), 0)
        self.assertEqual(len(cache.hdr_files), 0)

    def test_usdz_cache_add_stage(self):
        cache = USDGaussianExportCache()
        stage = Usd.Stage.CreateInMemory()
        named_stage = NamedUSDStage(filename="test.usda", stage=stage)
        cache.add_usd_stage(named_stage)
        self.assertEqual(len(cache.usd_stages), 1)
        self.assertEqual(cache.usd_stages[0], named_stage)

    def test_usdz_cache_write_to_usdz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = USDGaussianExportCache()
            stage = Usd.Stage.CreateInMemory()
            named_stage = NamedUSDStage(filename="test.usda", stage=stage)
            cache.add_usd_stage(named_stage)
            output_path = Path(tmpdir) / "test.usdz"
            cache.write_to_usdz(output_path)
            self.assertTrue(output_path.exists())
            self.assertTrue(zipfile.is_zipfile(output_path))
            with zipfile.ZipFile(output_path, "r") as zf:
                names = zf.namelist()
                self.assertIn("default.usda", names)
                self.assertIn("test.usda", names)


class TestRenderProductAndSpgExport(unittest.TestCase):
    """Tests for RenderProduct creation and SPG node support in USD export."""

    def test_render_products_with_render_vars_and_camera_refs(self):
        """Verify RenderProducts are created with correct structure for SPG pipeline."""
        cache = USDGaussianExportCache()
        stage = Usd.Stage.CreateInMemory()
        cache.add_usd_stage(NamedUSDStage(filename="test.usda", stage=stage))

        cam_infos = [
            CameraRenderProductInfo("camera_front", "rig_0", 1920, 1080),
            CameraRenderProductInfo("camera_rear", "rig_0", 1280, 720),
        ]
        cache.set_camera_render_products(cam_infos)
        composed_stage = cache.compose_default_usd_stage()

        # Verify Render scope and RenderProducts
        self.assertTrue(composed_stage.GetPrimAtPath("/Render").IsValid())

        for cam_info in cam_infos:
            rp_path = f"/Render/{cam_info.camera_name}"
            rp_prim = composed_stage.GetPrimAtPath(rp_path)
            self.assertTrue(rp_prim.IsValid())
            self.assertEqual(rp_prim.GetTypeName(), "RenderProduct")

            # Verify resolution and camera relationship
            resolution = rp_prim.GetAttribute("resolution").Get()
            self.assertEqual(resolution[0], cam_info.width)
            self.assertEqual(resolution[1], cam_info.height)
            expected_camera = f"/World/rig_trajectories/{cam_info.rig_name}/{cam_info.camera_name}"
            self.assertEqual(str(rp_prim.GetRelationship("camera").GetTargets()[0]), expected_camera)

            # Verify input RenderVar with correct sourceName
            input_var = composed_stage.GetPrimAtPath(f"{rp_path}/{PPISP_INPUT_RENDER_VAR}")
            self.assertTrue(input_var.IsValid())
            self.assertEqual(input_var.GetAttribute("sourceName").Get(), PPISP_INPUT_RENDER_VAR)

            # Verify orderedVars includes input RenderVar
            ordered_targets = [str(t) for t in rp_prim.GetRelationship("orderedVars").GetTargets()]
            self.assertIn(f"{rp_path}/{PPISP_INPUT_RENDER_VAR}", ordered_targets)

    def test_spg_files_loaded_and_packaged_in_usdz(self):
        """Verify SPG shader files are loaded and included in USDZ package."""
        # Verify files are loaded correctly
        spg_files = get_ppisp_spg_files()
        self.assertEqual(len(spg_files), 3)
        filenames = [f.filename for f in spg_files]
        self.assertIn(PPISP_SPG_SLANG_FILE, filenames)
        self.assertIn(PPISP_SPG_LUA_FILE, filenames)
        self.assertIn(PPISP_SPG_USDA_FILE, filenames)

        # Verify files are packaged in USDZ
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = USDGaussianExportCache()
            stage = Usd.Stage.CreateInMemory()
            cache.add_usd_stage(NamedUSDStage(filename="test.usda", stage=stage))
            cache.add_spg_files(spg_files)

            output_path = Path(tmpdir) / "test.usdz"
            cache.write_to_usdz(output_path)

            with zipfile.ZipFile(output_path, "r") as zf:
                names = zf.namelist()
                for expected_file in [PPISP_SPG_SLANG_FILE, PPISP_SPG_LUA_FILE, PPISP_SPG_USDA_FILE]:
                    self.assertIn(expected_file, names)

    def test_material_post_processing_disables_mdl_color_correction(self):
        """Verify MDL shader params are set correctly based on post-processing flag."""
        # Without post-processing: no MDL param overrides
        stage_no_pp = Usd.Stage.CreateInMemory()
        create_gaussian_material(stage_no_pp, has_post_processing=False)
        shader_no_pp = stage_no_pp.GetPrimAtPath(USD_GAUSSIAN_SHADER_PATH)
        self.assertFalse(shader_no_pp.GetAttribute("inputs:apply_srgb_linear").IsValid())
        self.assertFalse(shader_no_pp.GetAttribute("inputs:apply_inverse_tonemap").IsValid())

        # With post-processing: MDL params disabled
        stage_pp = Usd.Stage.CreateInMemory()
        create_gaussian_material(stage_pp, has_post_processing=True)
        shader_pp = stage_pp.GetPrimAtPath(USD_GAUSSIAN_SHADER_PATH)
        self.assertEqual(shader_pp.GetAttribute("inputs:apply_srgb_linear").Get(), False)
        self.assertEqual(shader_pp.GetAttribute("inputs:apply_inverse_tonemap").Get(), False)


class TestPPISPFrameMapping(unittest.TestCase):
    """Tests for PPISP frame index mapping alignment with training code."""

    def _create_rig_trajectories(self) -> RigTrajectories:
        """Create RigTrajectories with 2 cameras for testing."""
        T_identity = torch.eye(4, device="cpu", dtype=torch.float64)
        rig_trajectory = RigTrajectories.RigTrajectory(
            sequence_id="test",
            rig_bbox=BBox3(centroid=(0.0, 0.0, 0.0), dim=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0)),
            cameras_linear_start_frame_indices={"front": 0, "back": 3},
            lidars_linear_start_frame_indices=None,
            cameras_frame_timestamps_us={
                "front": torch.tensor([[900, 1000], [1900, 2000], [2900, 3000]], dtype=torch.int64),
                "back": torch.tensor([[1400, 1500], [2400, 2500]], dtype=torch.int64),
            },
            lidars_frame_timestamps_us={},
            T_rig_worlds=T_identity.repeat(5, 1, 1),
            T_rig_world_timestamps_us=torch.tensor([1000, 1500, 2000, 2500, 3000], dtype=torch.int64),
        )
        from collections import OrderedDict

        from ncore.data import FThetaCameraModelParameters, ShutterType

        ftheta = FThetaCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0] * 6, dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0] * 6, dtype=np.float32),
            max_angle=1.0,
            linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        calibs = OrderedDict(
            {
                name: RigTrajectories.CameraCalibration(
                    sequence_id="test",
                    logical_sensor_name=name,
                    unique_sensor_idx=i,
                    T_sensor_rig=T_identity.float(),
                    camera_model_parameters=ftheta,
                )
                for i, name in enumerate(["front", "back"])
            }
        )
        return RigTrajectories(
            T_world_base=T_identity,
            world_to_nre=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            rig_trajectories=[rig_trajectory],
            camera_calibrations=calibs,
            lidar_calibrations=OrderedDict(),
        )

    def test_ppisp_frame_index_calculation(self):
        """Verify PPISP frame index = linear_start_idx + local_frame_idx."""
        rig_traj = self._create_rig_trajectories()
        mappings = build_camera_frame_mappings(rig_traj)

        # front: linear_start=0, frames 0,1,2 -> PPISP 0,1,2
        self.assertEqual(mappings["front"].timestamp_to_ppisp_frame_idx[1000], 0)
        self.assertEqual(mappings["front"].timestamp_to_ppisp_frame_idx[2000], 1)
        self.assertEqual(mappings["front"].timestamp_to_ppisp_frame_idx[3000], 2)
        # back: linear_start=3, frames 0,1 -> PPISP 3,4
        self.assertEqual(mappings["back"].timestamp_to_ppisp_frame_idx[1500], 3)
        self.assertEqual(mappings["back"].timestamp_to_ppisp_frame_idx[2500], 4)

    def test_training_frame_filter(self):
        """Verify TrainingFrameFilterConfig filters validation frames correctly."""
        # val_frame_step=2 -> validation at [0, 2, 4], training at [1, 3]
        config = TrainingFrameFilterConfig(val_frame_start=0, val_frame_step=2)
        training = config.get_training_frame_local_indices(num_frames=5)
        self.assertEqual(training, [1, 3])

        # val_exclude_frame_step=2 -> training at [0, 2, 4]
        config2 = TrainingFrameFilterConfig(val_exclude_frame_start=0, val_exclude_frame_step=2)
        training2 = config2.get_training_frame_local_indices(num_frames=5)
        self.assertEqual(training2, [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
