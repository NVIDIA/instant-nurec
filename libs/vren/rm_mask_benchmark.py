# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from libs.vren.interface import vren  # type: ignore
from nre.models.custom_modules import weights_from_alphas


n_rays = 8192
n_samples = n_rays * 1024

# Generate a random number of samples per each ray
rand_n = np.random.rand(n_rays)
samples_per_ray = np.floor(rand_n * n_samples / np.sum(rand_n))
missing = np.random.choice(n_rays, int(n_samples - np.sum(samples_per_ray)), replace=False)
samples_per_ray[missing] += 1
samples_per_ray = torch.from_numpy(samples_per_ray).to(torch.int)

# Thresholds for early stopping and skipping of samples
transmittance_threshold = 0.01
alpha_threshold = 0.01

# Torch doesn't have exclusive cumsum so need to this manually
cumsum_samples = torch.cumsum(samples_per_ray, 0)[:, None].roll(1, 0)
cumsum_samples[0] = 0
pack_info = torch.cat([cumsum_samples, samples_per_ray[:, None]], dim=1).cuda().int()
alphas = torch.rand((n_samples,), requires_grad=True).cuda() * 0.03


def compute_mask_torch(
    pack_info: torch.Tensor, alphas: torch.Tensor, transmittance_threshold: float, alpha_threshold: float
) -> torch.Tensor:
    with torch.no_grad():
        _, _, weights = weights_from_alphas(alphas, pack_info, transmittance_threshold)

        transmittance = weights / alphas

    mask = transmittance > transmittance_threshold

    if alpha_threshold > 0:
        mask = mask & (alphas > alpha_threshold)

    return mask


def compute_mask_cuda(
    pack_info: torch.Tensor, alphas: torch.Tensor, transmittance_threshold: float, alpha_threshold: float
) -> torch.Tensor:
    with torch.no_grad():
        mask = vren.ray_samples_visibility_masks(alphas, pack_info, transmittance_threshold, alpha_threshold)

    return mask


# Ensure that both functions compute the same output. We only compare up to the 0.999 as some values might be different due to numerical precision
test_mask_cuda = compute_mask_cuda(pack_info, alphas, transmittance_threshold, alpha_threshold)
test_mask_torch = compute_mask_torch(pack_info, alphas, transmittance_threshold, alpha_threshold)
assert torch.eq(test_mask_cuda, test_mask_torch).sum() / test_mask_cuda.shape[0] > 0.999

num_threads = torch.get_num_threads()
print(f"Benchmarking on {num_threads} threads")

t0 = benchmark.Timer(
    stmt="compute_mask_torch(pack_info, alphas, transmittance_threshold, alpha_threshold)",
    setup="from __main__ import compute_mask_torch",
    globals={
        "pack_info": pack_info,
        "alphas": alphas,
        "transmittance_threshold": transmittance_threshold,
        "alpha_threshold": alpha_threshold,
    },
    num_threads=num_threads,
    label="Torch mask computation",
    sub_label="Getting the weights from the kernel and computing the transmittance in torch",
)

t1 = benchmark.Timer(
    stmt="compute_mask_cuda(pack_info, alphas, transmittance_threshold, alpha_threshold)",
    setup="from __main__ import compute_mask_cuda",
    globals={
        "pack_info": pack_info,
        "alphas": alphas,
        "transmittance_threshold": transmittance_threshold,
        "alpha_threshold": alpha_threshold,
    },
    num_threads=num_threads,
    label="Cuda kernel mask computation",
    sub_label="The kernel directly outputs the masks and skips other computations",
)

print(t0.timeit(1000))
print(t1.timeit(1000))
