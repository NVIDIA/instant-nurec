# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from typing import Optional
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig

from nre.datasets.samplers.holdout import HoldOutFrameSampler


def create_deterministic_rng():
    """Create a mock np.random.Generator.choice() that returns subsequent indices of the non-zero elements of the
    probability array when called in sequence
    """
    current_index: int = 0

    def choice(
        frame_range: npt.ArrayLike,
        size: Optional[int] = None,
        replace: bool = False,
        p: Optional[npt.ArrayLike] = None,
        axis: Optional[int] = None,
        shuffle: bool = False,
    ):
        # This mock is limited to some specific inputs of np.random.Generator.choice().
        assert isinstance(frame_range, range)
        assert size == 1
        assert not replace
        assert not shuffle
        assert axis is None
        assert p is not None

        p_array = np.asarray(p)
        nonzero_indices = np.where(p_array > 0)[0]

        nonlocal current_index
        if current_index >= len(nonzero_indices):
            current_index = 0  # Reset to beginning for cycling

        selected_index = nonzero_indices[current_index]
        selected_frame = list(frame_range)[selected_index]

        current_index += 1
        return np.array([selected_frame])

    # Create a mock that behaves like np.random.Generator
    mock_rng = Mock(spec=np.random.Generator)
    mock_rng.choice = choice
    return mock_rng


class TestHoldOutFrameSampler(unittest.TestCase):
    """Test cases for HoldOutFrameSampler"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_dataset = Mock(spec="nre.datasets.ncore.NCORETrainDataset")
        self.rng = create_deterministic_rng()

        # Define frame_range as the indices { 20, 30, 40, 50, 60, 70, 80 }.
        self.frame_range = range(20, 81, 10)

    def test_sample_frame_exclude_mode(self):
        """Test whether sample_frame() correctly excludes every Nth frame in exclude mode"""

        # Define config to skip every 3rd frame, starting from index 2.
        # Frames 40, 70 are excluded, only 20, 30, 50, 60, 80 are included (5 indices).
        config = DictConfig(
            {
                # Notice: include_frames_start and include_every_n_frames are missing from the config.
                "exclude_frames_start": 2,
                "exclude_every_n_frames": 3,
            }
        )

        # Create the sampler to be tested.
        sampler = HoldOutFrameSampler(config, self.mock_dataset)

        # Sample a frame multiple times to verify behavior.
        sampled_frames = []
        for _ in range(10):  # Test 2 full cycles over the valid 5 frame indices.
            result = sampler.sample_frame(
                rng=self.rng, batch_idx=0, frame_range=self.frame_range, unique_sensor_id="test_sensor"
            )
            sampled_frames.append(result.sampled_frame_idx)

        expected_sequence = [20, 30, 50, 60, 80, 20, 30, 50, 60, 80]  # 2 full cycles over the valid 5 frame indices.
        self.assertEqual(sampled_frames, expected_sequence)

    def test_sample_frame_include_mode(self):
        """Test whether sample_frame() correctly includes every Nth frame in include mode"""

        # Define config to include every 3rd frame, starting from index 2.
        # Frames 40, 70 are included (2 indices), but 20, 30, 50, 60, 80 are excluded.
        config = DictConfig(
            {
                # Notice: exclude_frames_start and exclude_every_n_frames are missing from the config.
                "include_frames_start": 2,
                "include_every_n_frames": 3,
            }
        )
        # Create the sampler to be tested.
        sampler = HoldOutFrameSampler(config, self.mock_dataset)

        # Sample a frame multiple times to verify behavior.
        sampled_frames = []
        for _ in range(4):  # Test 2 full cycles over the valid 2 frame indices.
            result = sampler.sample_frame(
                rng=self.rng, batch_idx=0, frame_range=self.frame_range, unique_sensor_id="test_sensor"
            )
            sampled_frames.append(result.sampled_frame_idx)

        expected_sequence = [40, 70, 40, 70]  # 2 full cycles over the valid 2 frame indices.
        self.assertEqual(sampled_frames, expected_sequence)

    def test_construction_with_empty_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with completely missing config"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(config=DictConfig({}), dataset=self.mock_dataset)

    def test_construction_with_contradicting_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with contradicting include/exclude config"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_frames_start": 1,
                        "include_every_n_frames": 2,
                        "exclude_frames_start": 3,
                        "exclude_every_n_frames": 4,
                    }
                ),
                dataset=self.mock_dataset,
            )

    def test_construction_with_unspecific_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with unspecific config (all None values)"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_frames_start": None,
                        "include_every_n_frames": None,
                        "exclude_frames_start": None,
                        "exclude_every_n_frames": None,
                    }
                ),
                dataset=self.mock_dataset,
            )

    def test_construction_with_unspecific_and_incomplete_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with incomplete config (missing start parameters)"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_every_n_frames": None,
                        "exclude_every_n_frames": None,
                    }
                ),
                dataset=self.mock_dataset,
            )
