# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import shutil
import tempfile
import threading
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from omegaconf import DictConfig, OmegaConf

from nre.difix.model import DifixModel, DifixModelFactory
from nre.systems.utils import system_config_compatibility_check


class TestDifixModel(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory that will be writable in Bazel test environment
        self.cache_dir = tempfile.mkdtemp()
        self.model_url = "https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/difix_pretrained_models/3.0/difix_pretrained_model-e4253f9f.tar.gz"
        self.batch_size = 1
        self.ckpt_name = "difix.pt"
        self.default_cache_dir_child = ".cache/NRE/difix/checkpoints"
        self.model_resolution = (544, 960)

    def tearDown(self):
        # Clean up the temporary directory after each test
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_init(self):
        self.maxDiff = None
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        self.assertEqual(difix_model.model_url, self.model_url)
        self.assertIsNone(difix_model.model)
        self.assertEqual(difix_model.cache_dir, self.cache_dir)
        self.assertEqual(difix_model.model_filename, self.ckpt_name)
        self.assertEqual(type(difix_model._lock), type(threading.Lock()))

    @patch("nre.difix.model.create_model_registry")
    @patch("os.path.exists")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("shutil.move")
    def test_download_checkpoint_tar_gz(self, mock_move, mock_remove, mock_tarfile, mock_exists, mock_create_registry):
        """Test download_checkpoint method with a tar.gz file"""
        # Set up mocks
        mock_registry = MagicMock()
        tarball_path = os.path.join(self.cache_dir, "difix_pretrained_model-e4253f9f.tar.gz")
        mock_registry.get_model.return_value = tarball_path
        mock_create_registry.return_value = mock_registry

        # Mock .tar.gz handling - provide values for calls to os.path.exists
        # 1. Initial check if cached_checkpoint_file exists - False means we need to extract
        # 2. Check if downloaded_file exists - True to pass the assertion
        # 3. Check if cached_checkpoint_file exists after extraction - True means extraction worked
        mock_exists.side_effect = (
            lambda path: path == tarball_path
            or path == os.path.join(self.cache_dir, self.ckpt_name)
            and mock_move.called
        )

        # Create mock tar file with a single .pt file member
        mock_tar_instance = MagicMock()
        mock_member = MagicMock()
        mock_member.name = "difix.pt"
        mock_tar_instance.getmembers.return_value = [mock_member]
        mock_tarfile.return_value.__enter__.return_value = mock_tar_instance

        # Create instance and call method
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        result = difix_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_called_once_with(self.model_url, Path(self.cache_dir))
        mock_registry.get_model.assert_called_once()
        mock_tarfile.assert_called_once_with(tarball_path, "r:gz")
        mock_tar_instance.extract.assert_called_once_with(mock_member, path=self.cache_dir)
        mock_move.assert_called_once_with(
            os.path.join(self.cache_dir, mock_member.name), os.path.join(self.cache_dir, self.ckpt_name)
        )
        mock_remove.assert_called_once_with(tarball_path)
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("nre.difix.model.create_model_registry")
    @patch("os.path.exists")
    @patch("shutil.move")
    def test_download_checkpoint_checkpoint_exists(self, mock_move, mock_exists, mock_create_registry):
        """Test download_checkpoint method when checkpoint already exists"""
        # Mock checkpoint already exists
        mock_exists.return_value = True

        # Create instance and call method
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        result = difix_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_not_called()
        mock_move.assert_not_called()
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("nre.difix.model.create_model_registry")
    @patch("os.path.exists")
    @patch("shutil.move")
    def test_download_checkpoint_non_tar_file(self, mock_move, mock_exists, mock_create_registry):
        """Test download_checkpoint method with a non-tar file (e.g. direct checkpoint)"""
        # Set up mocks
        mock_registry = MagicMock()
        checkpoint_path = os.path.join(self.cache_dir, "some_direct_checkpoint.pt")
        mock_registry.get_model.return_value = checkpoint_path
        mock_create_registry.return_value = mock_registry

        # Mock the file existence check - first False for cached_checkpoint_file check,
        # then True for downloaded_file existence check
        mock_exists.side_effect = [False, True, True]

        # Create instance and call method
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        result = difix_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_called_once_with(self.model_url, Path(self.cache_dir))
        mock_registry.get_model.assert_called_once()
        mock_move.assert_called_once_with(checkpoint_path, os.path.join(self.cache_dir, self.ckpt_name))
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("torch.jit.load")
    @patch.object(DifixModel, "_download_checkpoint")
    def test_load_pretrained_model(self, mock_download_checkpoint, mock_torch_load):
        """Test the _load_pretrained_model method"""
        # Set up mocks
        mock_model = MagicMock()
        mock_torch_load.return_value = mock_model
        checkpoint_path = os.path.join(self.cache_dir, self.ckpt_name)
        mock_download_checkpoint.return_value = checkpoint_path

        # Create instance and call method
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        result = difix_model._load_pretrained_model()

        # Checks
        mock_download_checkpoint.assert_called_once_with(self.model_url, self.cache_dir, self.ckpt_name)
        mock_torch_load.assert_called_once_with(checkpoint_path)
        self.assertEqual(result, mock_model.cuda.return_value)

    def _make_cpu_dummy(self):
        """Create a CPU dummy tensor matching model_resolution, bypassing CUDA requirement."""
        return torch.zeros(1, 3, *self.model_resolution)

    def test_detect_needs_temporal_dim_false(self):
        """Model that accepts 4-D input sets needs_temporal_dim=False."""
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: x
        difix_model.model = mock_model

        with patch("nre.difix.model.torch.zeros", return_value=self._make_cpu_dummy()):
            difix_model._detect_needs_temporal_dim()

        self.assertFalse(difix_model.needs_temporal_dim)
        mock_model.assert_called_once()

    def test_detect_needs_temporal_dim_true(self):
        """Model that rejects 4-D but accepts 5-D input sets needs_temporal_dim=True."""
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        mock_model = MagicMock()

        def accept_only_5d(x):
            if x.dim() == 4:
                raise RuntimeError("expected 5-D input")
            return x

        mock_model.side_effect = accept_only_5d
        difix_model.model = mock_model

        with patch("nre.difix.model.torch.zeros", return_value=self._make_cpu_dummy()):
            difix_model._detect_needs_temporal_dim()

        self.assertTrue(difix_model.needs_temporal_dim)
        self.assertEqual(mock_model.call_count, 2)

    def test_detect_needs_temporal_dim_both_fail(self):
        """If both 4-D and 5-D inputs fail, the error from the 5-D attempt propagates."""
        difix_model = DifixModel(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("unsupported model")
        difix_model.model = mock_model

        with patch("nre.difix.model.torch.zeros", return_value=self._make_cpu_dummy()):
            with self.assertRaises(RuntimeError):
                difix_model._detect_needs_temporal_dim()


class TestDifixModelFactory(unittest.TestCase):
    def setUp(self):
        DifixModelFactory._initialized_model = None
        self.mock_model_url = MagicMock()
        self.mock_cache_dir = MagicMock()
        self.mock_model_filename = MagicMock()
        self.mock_instance = MagicMock(spec=DifixModel)
        self.mock_model_resolution = MagicMock()

    @patch("nre.difix.model.DifixModel._download_checkpoint")
    @patch("nre.difix.model.DifixModel")
    def test_get_creates_new_instance(self, mock_model, mock_download):
        mock_model.return_value = self.mock_instance
        mock_download.return_value = "/path/to/checkpoint"
        mock_loaded_model = MagicMock()
        self.mock_instance._load_pretrained_model.return_value = mock_loaded_model

        result = DifixModelFactory.get(
            self.mock_model_url, self.mock_cache_dir, self.mock_model_filename, self.mock_model_resolution
        )

        # Check that the factory returns the instance
        self.assertEqual(result, self.mock_instance)

        # Check that DifixModel was instantiated
        mock_model.assert_called_once_with(
            self.mock_model_url, self.mock_cache_dir, self.mock_model_filename, self.mock_model_resolution
        )

        # Check that download was called
        mock_download.assert_called_once_with(self.mock_model_url, self.mock_cache_dir, self.mock_model_filename)

        # Check that _load_pretrained_model was called on the instance
        self.mock_instance._load_pretrained_model.assert_called_once()

        # Check that model attribute was set
        self.assertEqual(self.mock_instance.model, mock_loaded_model)

        # Check that temporal dim detection was run
        self.mock_instance._detect_needs_temporal_dim.assert_called_once()

        # Reset for next test
        DifixModelFactory._initialized_model = None
        mock_model.reset_mock()

    @patch("nre.difix.model.DifixModel")
    def test_get_returns_existing_instance(self, mock_model):
        DifixModelFactory._initialized_model = self.mock_instance
        result = DifixModelFactory.get(
            self.mock_model_url, self.mock_cache_dir, self.mock_model_filename, self.mock_model_resolution
        )
        # checks
        self.assertEqual(result, self.mock_instance)
        mock_model.assert_not_called()

    def test_clear_cache(self):
        DifixModelFactory._initialized_model = self.mock_instance
        DifixModelFactory.clear_cache()
        # checks
        self.assertIsNone(DifixModelFactory._initialized_model)

    @patch("nre.difix.model.DifixModel._download_checkpoint")
    @patch("nre.difix.model.DifixModel")
    def test_thread_safety(self, mock_model, mock_download):
        DifixModelFactory.clear_cache()
        instances = []
        mock_model.return_value = self.mock_instance
        mock_download.return_value = "/path/to/checkpoint"
        mock_loaded_model = MagicMock()
        self.mock_instance._load_pretrained_model.return_value = mock_loaded_model
        config = DictConfig({"key": "value"})

        def create_instance(config):
            instance = DifixModelFactory.get(
                self.mock_model_url, self.mock_cache_dir, self.mock_model_filename, self.mock_model_resolution
            )
            instances.append(instance)

        threads = [threading.Thread(target=create_instance(config)) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Assert that all instances have the same ID
        self.assertTrue(all(id(instance) == id(instances[0]) for instance in instances))
        self.assertEqual(len(set(instances)), 1)
        self.assertEqual(mock_model.call_count, 1)
        # Verify download was called exactly once (thread-safe singleton)
        self.assertEqual(mock_download.call_count, 1)
        # Verify _load_pretrained_model was called exactly once on the instance
        self.assertEqual(self.mock_instance._load_pretrained_model.call_count, 1)


class TestDifixConfigCompatibility(unittest.TestCase):
    """Tests for Difix configuration validation in system_config_compatibility_check."""

    def setUp(self):
        """Set up test configuration with valid training-time Difix settings."""
        # Create a mock config with valid difix training settings
        self.config = OmegaConf.create(
            {
                "dataset": {
                    "name": "ncore",
                    "n_train_sample_camera_rays": 1024,
                    "val_lidar": False,
                    "n_train_sample_lidar_rays": 0,
                    "samplers": {
                        "batch_sampler": {
                            "name": "default",
                            "ratio_camera_samples": 1.0,
                            "update_n_epochs": 1,
                            "camera_pixel_sampler": {"name": "image-crop"},
                        }
                    },
                },
                "difix": {
                    "name": "difix",
                    "model_url": "https://example.com/model.tar.gz",
                    "model_resolution": [544, 960],
                    "model_filename": "difix.pt",
                    "cache_dir": "~/.cache/nre/difix",
                    "training": {
                        "enabled": True,
                        "p_scheduler": {
                            "p_init": 0.5,
                        },
                        "start_step": 15000,
                    },
                    "inference": {
                        "enabled": False,
                    },
                },
                "model": {
                    "calib": {
                        "name": "skip-calib"  # Skip calibration check
                    },
                    # Add minimal model config
                    "gaussians": {
                        "name": "default-gaussians"  # Not "sh-gaussians" to avoid extra assertions
                    },
                },
                "trainer": {
                    "precision": 32  # Required for gaussian checks
                },
                "system": {
                    "name": "default-system"  # For lidar validation check
                },
            }
        )

    def test_valid_difix_training_config(self):
        """Test valid training-time Difix configuration passes checks."""
        # Should not raise any exceptions for valid config
        system_config_compatibility_check(self.config)
