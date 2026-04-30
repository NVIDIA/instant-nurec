# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import imageio as iio
import numpy as np
import pytest
import torch

from nre.grpc.ego_hood import EgocarHood, EgocarRig, EgocarRigBank
from nre.grpc.protos.sensorsim_pb2 import RGBRenderRequest


@pytest.fixture
def test_image_rgba():
    # Create a simple RGBA test image (100x100 with a red square in the middle)
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    # Set the middle 50x50 area to red with alpha=255
    img[25:75, 25:75, 0] = 255  # Red
    img[25:75, 25:75, 3] = 255  # Alpha
    return img


@pytest.fixture
def test_rig_bank_dir(test_image_rgba, tmp_path):
    rig_ids = ["rig1", "rig2"]
    camera_ids = ["front", "left", "right"]

    # Create multiple rig subdirectories in the pytest-provided temporary directory
    for rig_id in rig_ids:
        rig_dir = tmp_path / rig_id
        rig_dir.mkdir()

        # Create multiple hood images for each rig
        for camera_id in camera_ids:
            image_path = rig_dir / f"{camera_id}.png"
            iio.imwrite(image_path, test_image_rgba)

    # Return the base directory, rig IDs, camera IDs,
    # a specific hood image path, and a specific rig directory path
    hood_image_path = tmp_path / rig_ids[0] / f"{camera_ids[0]}.png"
    rig_dir_path = tmp_path / rig_ids[0]

    return {
        "base_dir": tmp_path,
        "rig_ids": rig_ids,
        "camera_ids": camera_ids,
        "hood_image_path": hood_image_path,
        "rig_dir_path": rig_dir_path,
    }


class TestEgocarHood:
    def test_load_from_file(self, test_rig_bank_dir):
        # Test creating an EgocarHood from a file
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        assert hood.camera_logical_id == "front"
        assert hood.rig_id == "rig1"
        assert hood._hood_rgba.shape[0] == 4  # RGBA
        assert hood._hood_rgba.dtype == torch.float32
        assert torch.max(hood._hood_rgba) <= 1.0  # Normalized to [0, 1]

    def test_draw_egocar(self, test_rig_bank_dir):
        # Test overlaying the hood on a render
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        # Create a test render (blue background)
        render = torch.zeros(100, 100, 3, dtype=torch.float32)
        render[:, :, 2] = 1.0  # Blue background

        # Apply the hood to the render
        result = hood.overlay_on_image(render)

        # The result should have red pixels where the hood was and blue pixels elsewhere
        assert result.shape == (100, 100, 3)

        # Check middle pixel (should be hood/red)
        assert result[50, 50, 0] > 0.9  # Red channel high
        assert result[50, 50, 1] < 0.1  # Green channel low
        assert result[50, 50, 2] < 0.1  # Blue channel low

        # Check corner pixel (should be render/blue)
        assert result[0, 0, 0] < 0.1  # Red channel low
        assert result[0, 0, 1] < 0.1  # Green channel low
        assert result[0, 0, 2] > 0.9  # Blue channel high

    def test_resize_cache(self, test_rig_bank_dir, monkeypatch):
        # Test that the resize cache works correctly
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        # Create a render with a different size
        render1 = torch.zeros(50, 50, 3, dtype=torch.float32)

        # Setup to monitor logger calls
        logger_calls = []

        def mock_info(message):
            logger_calls.append(message)

        # Patch the logger.info method
        monkeypatch.setattr("nre.grpc.ego_hood.logger.info", mock_info)

        # First draw should update the cache
        hood.overlay_on_image(render1)
        assert len(logger_calls) == 1

        # Cache should now be at the new size
        assert hood._resize_cache.shape[1:3] == (50, 50)

        # Clear the call log
        logger_calls.clear()

        # Second draw should use the cache
        hood.overlay_on_image(render1)
        assert len(logger_calls) == 0

    def test_metadata(self, test_rig_bank_dir):
        # Test that the metadata is correctly generated
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)
        metadata = hood.metadata

        assert metadata.ego_mask_id.camera_logical_id == "front"
        assert metadata.ego_mask_id.rig_config_id == "rig1"


class TestEgocarRig:
    def test_load_from_dir(self, test_rig_bank_dir):
        # Test creating an EgocarRig from a directory
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        assert rig.rig_id == "test_rig"
        assert len(rig._hoods) == len(camera_ids)
        for camera_id in camera_ids:
            assert camera_id in rig._hoods

    def test_get(self, test_rig_bank_dir):
        # Test the get method
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        # Existing camera_id
        hood = rig.get(camera_ids[0])
        assert hood is not None
        assert hood.camera_logical_id == camera_ids[0]

        # Non-existent camera_id
        hood = rig.get("nonexistent")
        assert hood is None

    def test_available_metadata(self, test_rig_bank_dir):
        # Test that available_metadata returns the correct list
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        metadata_list = rig.available_metadata()
        assert len(metadata_list) == len(camera_ids)

        camera_ids_in_metadata = [m.ego_mask_id.camera_logical_id for m in metadata_list]
        for camera_id in camera_ids:
            assert camera_id in camera_ids_in_metadata


class TestEgocarRigBank:
    def test_load_from_dir(self, test_rig_bank_dir):
        # Test creating an EgocarRigBank from a directory
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        assert len(bank._rigs) == len(rig_ids)
        for rig_id in rig_ids:
            assert rig_id in bank._rigs

    def test_empty(self):
        # Test creating an empty EgocarRigBank
        bank = EgocarRigBank.empty()

        assert len(bank._rigs) == 0

    def test_available_metadata(self, test_rig_bank_dir):
        # Test that available_metadata returns the correct list
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        metadata_list = bank.available_metadata()
        assert len(metadata_list) == len(rig_ids) * len(camera_ids)

    def test_select_from_request_specific(self, test_rig_bank_dir):
        # Test selecting a specific hood by ID
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with specific camera_logical_id and rig_config_id
        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = camera_ids[1]
        request.ego_mask_id.rig_config_id = rig_ids[1]

        hood = bank.select_from_request(request)
        assert hood is not None
        assert hood.camera_logical_id == camera_ids[1]
        assert hood.rig_id == rig_ids[1]

    def test_select_from_request_nonexistent_config(self, test_rig_bank_dir):
        # Test selecting a hood with nonexistent config_id
        bank_dir = test_rig_bank_dir["base_dir"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with nonexistent config_id
        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = camera_ids[0]
        request.ego_mask_id.rig_config_id = "nonexistent"

        with pytest.raises(KeyError):
            bank.select_from_request(request)

    def test_select_from_request_nonexistent_camera(self, test_rig_bank_dir):
        # Test selecting a hood with nonexistent camera_id
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with nonexistent camera_id
        request = RGBRenderRequest()
        request.ego_mask_id.camera_logical_id = "nonexistent"
        request.ego_mask_id.rig_config_id = rig_ids[0]

        hood = bank.select_from_request(request)
        assert hood is None

    def test_select_from_request_no_use_ego_mask(self, test_rig_bank_dir):
        bank_dir = test_rig_bank_dir["base_dir"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with empty camera_id
        request = RGBRenderRequest()
        request.camera_intrinsics.logical_id = "front"  # This should be ignored

        hood = bank.select_from_request(request)
        assert hood is None
