# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import tempfile
import unittest

from pathlib import Path

import torch

from internal.scripts.experimental.ppisp_report.ppisp_report import (
    extract_ppisp_params,
    generate_camera_report,
)
from nre.models.post_processings.ppisp import PPISP


class TestPPISPReport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create a simple PPISP model for testing
        self.num_cameras = 2
        self.n_frames_per_camera = [3, 2]
        self.num_frames = sum(self.n_frames_per_camera)
        self.ppisp_model = PPISP(
            device=self.device,
            n_frames_per_camera=self.n_frames_per_camera,
            num_cameras=self.num_cameras,
            num_frames=self.num_frames,
        )

        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

        # Clean up GPU memory
        torch.cuda.empty_cache()

    def test_extract_ppisp_params(self):
        """Test that parameter extraction works without errors"""
        params = extract_ppisp_params(self.ppisp_model)
        self.assertEqual(params["num_cameras"], self.num_cameras)
        self.assertEqual(params["num_frames"], self.num_frames)
        self.assertEqual(params["n_frames_per_camera"], self.n_frames_per_camera)
        self.assertEqual(len(params["exposure_params"]), self.num_frames)
        self.assertEqual(len(params["crf"]["effective_values"]), self.num_cameras)
        self.assertEqual(len(params["color_params"]), self.num_frames)
        self.assertEqual(len(params["vignetting"]), self.num_cameras)

    def test_generate_camera_report(self):
        """Test that camera report generation works without errors"""
        for camera_idx in range(self.num_cameras):
            # Generate report for this camera
            report_path = self.output_path / f"camera_{camera_idx}_report.pdf"
            generate_camera_report(
                ppisp_model=self.ppisp_model,
                camera_idx=camera_idx,
                output_path=report_path,
            )

            # Check that report file was created
            self.assertTrue(report_path.exists(), f"Report file not created for camera {camera_idx}")


if __name__ == "__main__":
    unittest.main()
