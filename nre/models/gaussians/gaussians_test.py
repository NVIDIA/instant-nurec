# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile
import unittest

import numpy as np
import torch
import yaml

from nre.models.gaussians.gaussians_model import compute_stable_merging_indices


class TestGaussiansModel(unittest.TestCase):
    def test_compute_stable_merging_indices(self):
        full_length = np.random.randint(100, 1000)
        world_size = np.random.randint(2, 32)
        device = torch.device("cuda")

        original_data = torch.arange(full_length, device=device)
        gathered_data = []
        for rank in range(world_size):
            gathered_data.append(original_data[rank::world_size])
        gathered_data = torch.cat(gathered_data)

        indices = compute_stable_merging_indices(full_length, world_size, device)
        assert torch.allclose(gathered_data[indices], original_data)


if __name__ == "__main__":
    unittest.main()
