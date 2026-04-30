# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import numpy as np
import torch

# C++ / CUDA libs
from libs.packed_ops.interface import packed_ops  # type: ignore
from nre.utils.misc import get_pack_info_from_n
from nre.utils.packed_ops import (
    arange_interleave,
    arange_interleave_simple,
    linstep_interleave,
    merge_two_packs_sorted_aligned,
    packed_add,
    packed_backward_diff,
    packed_cumprod,
    packed_cumsum,
    packed_diff,
    packed_div,
    packed_interp,
    packed_max,
    packed_min,
    packed_mul,
    packed_searchsorted,
    packed_searchsorted_packed_vals,
    packed_sub,
    packed_sum,
    packed_weighted_sum,
)
from nre.utils.tests import CommonTestCase, NonDeterministicTestCase


class TestMergeTwoPackedSortedArrays(CommonTestCase):
    def setUp(self):
        self.n_rays = 1000
        self.a_samples = torch.randint(0, 100, (self.n_rays,))
        self.b_samples = torch.randint(0, 100, (self.n_rays,))

        self.packed_a = torch.cat(
            [torch.cumsum(self.a_samples, 0)[:, None].roll(1, 0), self.a_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_b = torch.cat(
            [torch.cumsum(self.b_samples, 0)[:, None].roll(1, 0), self.b_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a[0, 0] = 0
        self.packed_b[0, 0] = 0

        self.data_a = []
        self.data_b = []

        for i in range(self.n_rays):
            self.data_a.append(
                torch.sort(torch.rand(int(self.packed_a[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
            self.data_b.append(
                torch.sort(torch.rand(int(self.packed_b[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
        self.data_a = torch.cat(self.data_a)
        self.data_b = torch.cat(self.data_b)

        # Data with guaranteed zero entries (no sample for that ray)
        self.n_rays_zero = 3
        self.a_samples_zero = torch.cat(
            [torch.randint(0, 20, (self.n_rays_zero - 2,)), torch.zeros((1,)), 20 * torch.ones((1,))]
        )
        self.b_samples_zero = torch.randint(0, 20, (3,))

        self.packed_a_zero = torch.cat(
            [torch.cumsum(self.a_samples_zero, 0)[:, None].roll(1, 0), self.a_samples_zero[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_b_zero = torch.cat(
            [torch.cumsum(self.b_samples_zero, 0)[:, None].roll(1, 0), self.b_samples_zero[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a_zero[0, 0] = 0
        self.packed_b_zero[0, 0] = 0

        self.data_a_zero = []
        self.data_b_zero = []
        for i in range(self.n_rays_zero):
            self.data_a_zero.append(
                torch.sort(torch.rand(int(self.packed_a_zero[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
            self.data_b_zero.append(
                torch.sort(torch.rand(int(self.packed_b_zero[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
        self.data_a_zero = torch.cat(self.data_a_zero)
        self.data_b_zero = torch.cat(self.data_b_zero)

    def test_compare_to_torch_sort_merge(self):
        assert isinstance(self.data_a, torch.Tensor)
        assert isinstance(self.data_b, torch.Tensor)

        torch_sort = []
        for i in range(self.n_rays):
            torch_sort.append(
                torch.sort(
                    torch.cat(
                        [
                            self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]],
                            self.data_b[self.packed_b[i, 0] : self.packed_b[i, 0] + self.packed_b[i, 1]],
                        ]
                    )
                ).values
            )
        torch_sort = torch.cat(torch_sort)

        # Merge using the packed_ops's merge packed tensors
        ranks_a, ranks_b, _ = packed_ops.merge_two_packs_sorted_aligned_fw(
            self.data_a, self.packed_a, self.data_b, self.packed_b
        )
        merged_packed_ops = torch.empty(self.data_a.shape[0] + self.data_b.shape[0]).cuda()

        merged_packed_ops[ranks_a] = self.data_a
        merged_packed_ops[ranks_b] = self.data_b

        self._compareTensor(merged_packed_ops.cpu(), torch_sort.cpu())

    def test_compare_to_torch_sort_merge_with_zero_entries(self):
        assert isinstance(self.data_a_zero, torch.Tensor)
        assert isinstance(self.data_b_zero, torch.Tensor)

        torch_sort = []
        for i in range(self.n_rays_zero):
            torch_sort.append(
                torch.sort(
                    torch.cat(
                        [
                            self.data_a_zero[
                                self.packed_a_zero[i, 0] : self.packed_a_zero[i, 0] + self.packed_a_zero[i, 1]
                            ],
                            self.data_b_zero[
                                self.packed_b_zero[i, 0] : self.packed_b_zero[i, 0] + self.packed_b_zero[i, 1]
                            ],
                        ]
                    )
                ).values
            )
        torch_sort = torch.cat(torch_sort)

        # Merge using the packed_ops's merge packed tensors
        ranks_a, ranks_b, _ = packed_ops.merge_two_packs_sorted_aligned_fw(
            self.data_a_zero, self.packed_a_zero, self.data_b_zero, self.packed_b_zero
        )
        merged_packed_ops = torch.empty(self.data_a_zero.shape[0] + self.data_b_zero.shape[0]).cuda()
        merged_packed_ops[ranks_a] = self.data_a_zero
        merged_packed_ops[ranks_b] = self.data_b_zero

        self._compareTensor(merged_packed_ops.cpu(), torch_sort.cpu())


class TestPackedArithmeticOperations(NonDeterministicTestCase):
    def setUp(self):
        self.n_rays = 1000
        self.n_features = 300
        self.a_samples = torch.randint(0, 100, (self.n_rays,))

        self.packed_a = torch.cat(
            [torch.cumsum(self.a_samples, 0)[:, None].roll(1, 0), self.a_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a[0, 0] = 0

        self.data_a = []
        for i in range(self.n_rays):
            self.data_a.append(
                torch.rand(int(self.packed_a[i, 1].item()), requires_grad=True).to(device="cuda", dtype=torch.float)
            )

        self.data_a = torch.cat(self.data_a)
        self.data_b = torch.rand((int(self.packed_a[-1].sum().item()), self.n_features), requires_grad=True).to(
            device="cuda", dtype=torch.float
        )
        self.weights_b = torch.rand((self.data_b.shape[0],), requires_grad=True).cuda()

        # Data with guaranteed zero entries (no sample for initial / middle / end ray)
        self.n_rays_zero = 5
        self.a_samples_zero = torch.cat(
            [
                torch.zeros((1,)),
                torch.randint(0, 20, (self.n_rays_zero - 4,)),
                torch.zeros((1,)),
                20 * torch.ones((1,)),
                torch.zeros((1,)),
            ]
        )

        self.packed_a_zero = torch.cat(
            [torch.cumsum(self.a_samples_zero, 0)[:, None].roll(1, 0), self.a_samples_zero[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a_zero[0, 0] = 0

        self.data_a_zero = []
        for i in range(self.n_rays_zero):
            self.data_a_zero.append(
                torch.rand(self.packed_a_zero[i, 1].item(), requires_grad=True).to(device="cuda", dtype=torch.float)
            )
        self.data_a_zero = torch.cat(self.data_a_zero)
        self.data_b_zero = torch.rand(
            (int(self.packed_a_zero[-1].sum().item()), self.n_features), requires_grad=True
        ).to(device="cuda", dtype=torch.float)
        self.weights_b_zero = torch.rand((self.data_b_zero.shape[0],), requires_grad=True).cuda()

    def test_packed_cumsum(self):
        def test(n_rays, data_a, packed_a):
            assert isinstance(data_a, torch.Tensor)

            torch_cumsum = []
            torch_cumsum_exclusive = []
            for i in range(n_rays):
                torch_cumsum.append(torch.cumsum(data_a[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1]], dim=0))

                tmp_cumsum = torch.cumsum(data_a[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1]], dim=0).roll(1, 0)
                # If there are some elements, set the first after the roll-over to 0.0
                if tmp_cumsum.shape[0] > 0:
                    tmp_cumsum[0] = 0.0
                torch_cumsum_exclusive.append(tmp_cumsum)

            torch_cumsum = torch.cat(torch_cumsum)
            torch_cumsum_exclusive = torch.cat(torch_cumsum_exclusive)

            ours_cumsum = packed_cumsum(data_a, packed_a, False, False)
            ours_cumsum_exclusive = packed_cumsum(data_a, packed_a, True, False)

            self._compareTensor(
                torch_cumsum.detach().cpu(), ours_cumsum.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )
            self._compareTensor(
                torch_cumsum_exclusive.detach().cpu(),
                ours_cumsum_exclusive.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
            )

            # Generate the GT cumulative sum: compute the loss and compare the gradients
            gt_cumsum = torch.rand_like(torch_cumsum, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_cumsum - gt_cumsum))
            loss_ours = torch.mean(torch.square(ours_cumsum - gt_cumsum))

            grad_torch_cumsum = torch.autograd.grad(loss_torch, (data_a), grad_outputs=torch.ones_like(loss_torch))[0]
            grad_ours_cumsum = torch.autograd.grad(loss_ours, (data_a), grad_outputs=torch.ones_like(loss_ours))[0]
            self._compareTensor(
                grad_torch_cumsum.detach().cpu(),
                grad_ours_cumsum.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=-1,
                ratio_of_permitted_failures=0.04,
            )

            grad_torch_cumsum_exclusive = torch.autograd.grad(
                torch_cumsum_exclusive, (data_a), grad_outputs=torch.ones_like(torch_cumsum_exclusive)
            )[0]
            grad_ours_cumsum_exclusive = torch.autograd.grad(
                ours_cumsum_exclusive, (data_a), grad_outputs=torch.ones_like(ours_cumsum_exclusive)
            )[0]
            self._compareTensor(
                grad_torch_cumsum_exclusive.detach().cpu(),
                grad_ours_cumsum_exclusive.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=-1,
                ratio_of_permitted_failures=0.04,
            )

        test(self.n_rays, self.data_a, self.packed_a)
        test(self.n_rays_zero, self.data_a_zero, self.packed_a_zero)

    def test_packed_cumprod(self):
        def test(n_rays, data_a, packed_a):
            assert isinstance(data_a, torch.Tensor)

            torch_cumprod = []
            torch_cumprod_exclusive = []
            for i in range(n_rays):
                torch_cumprod.append(torch.cumprod(data_a[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1]], dim=0))

                tmp_cumprod = torch.cumprod(data_a[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1]], dim=0).roll(1, 0)
                # If there are some elements, set the first after the roll-over to 1.0 in accordance with Tensorflow
                if tmp_cumprod.shape[0] > 0:
                    tmp_cumprod[0] = 1.0
                torch_cumprod_exclusive.append(tmp_cumprod)

            torch_cumprod = torch.cat(torch_cumprod)
            torch_cumprod_exclusive = torch.cat(torch_cumprod_exclusive)

            ours_cumprod = packed_cumprod(data_a, packed_a, False, False)
            ours_cumprod_exclusive = packed_cumprod(data_a, packed_a, True, False)

            self._compareTensor(
                torch_cumprod.detach().cpu(), ours_cumprod.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )
            self._compareTensor(
                torch_cumprod_exclusive.detach().cpu(),
                ours_cumprod_exclusive.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
            )

            # Generate the GT cumulative product: compute the loss and compare the gradients
            gt_cumprod = torch.rand_like(torch_cumprod, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_cumprod - gt_cumprod))
            loss_ours = torch.mean(torch.square(ours_cumprod - gt_cumprod))

            grad_torch_cumprod = torch.autograd.grad(loss_torch, (data_a), grad_outputs=torch.ones_like(loss_torch))[0]
            grad_ours_cumprod = torch.autograd.grad(loss_ours, (data_a), grad_outputs=torch.ones_like(loss_ours))[0]

            self._compareTensor(
                grad_torch_cumprod.detach().cpu(),
                grad_ours_cumprod.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=-1,
                ratio_of_permitted_failures=0.04,
            )

            grad_torch_cumprod_exclusive = torch.autograd.grad(
                torch_cumprod_exclusive, (data_a), grad_outputs=torch.ones_like(torch_cumprod_exclusive)
            )[0]
            grad_ours_cumprod_exclusive = torch.autograd.grad(
                ours_cumprod_exclusive, (data_a), grad_outputs=torch.ones_like(ours_cumprod_exclusive)
            )[0]
            self._compareTensor(
                grad_torch_cumprod_exclusive.detach().cpu(),
                grad_ours_cumprod_exclusive.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=-1,
                ratio_of_permitted_failures=0.04,
            )

        test(self.n_rays, self.data_a, self.packed_a)
        test(self.n_rays_zero, self.data_a_zero, self.packed_a_zero)

    def test_packed_sum(self):
        def test(data_b, packed_a):
            assert isinstance(data_b, torch.Tensor)

            torch_sum = []
            for i in range(len(packed_a)):
                torch_sum.append(torch.sum(data_b[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1]], dim=0))

            torch_sum = torch.stack(torch_sum, dim=0)
            ours_sum = packed_sum(data_b, packed_a)

            self._compareTensor(
                torch_sum.detach().cpu(), ours_sum.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )

            # Generate the GT sum: compute the loss and compare the gradients
            gt_sum = torch.rand_like(torch_sum, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_sum - gt_sum))
            loss_ours = torch.mean(torch.square(ours_sum - gt_sum))

            grad_torch_sum = torch.autograd.grad(loss_torch, (data_b), grad_outputs=torch.ones_like(loss_torch))[0]
            grad_ours_sum = torch.autograd.grad(loss_ours, (data_b), grad_outputs=torch.ones_like(loss_ours))[0]
            self._compareTensor(
                grad_torch_sum.detach().cpu(),
                grad_ours_sum.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )

        test(self.data_b, self.packed_a)
        test(self.data_b_zero, self.packed_a_zero)
        test(self.data_b, self.packed_a[::2].contiguous())

    def test_packed_add(self):
        def test(data_b, packed_a):
            assert isinstance(data_b, torch.Tensor)

            n_rays = len(packed_a)

            other = torch.rand((n_rays, data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float) - 0.5

            torch_add = data_b.clone()
            for i in range(n_rays):
                torch_add[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1], :].add_(other[i][None, :])

            ours_add = packed_add(data_b, other, packed_a)
            self._compareTensor(
                torch_add.detach().cpu(), ours_add.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )

            # Generate the GT addition result: compute the loss and compare the gradients
            gt_add = torch.rand_like(torch_add, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_add - gt_add))
            loss_ours = torch.mean(torch.square(ours_add - gt_add))

            grad_torch_add_a, grad_torch_add_other = torch.autograd.grad(
                loss_torch, (data_b, other), grad_outputs=torch.ones_like(loss_torch)
            )
            grad_ours_add_a, grad_ours_add_other = torch.autograd.grad(
                loss_ours, (data_b, other), grad_outputs=torch.ones_like(loss_ours)
            )
            self._compareTensor(
                grad_torch_add_a.detach().cpu(),
                grad_ours_add_a.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )

            self._compareTensor(
                grad_torch_add_other.detach().cpu(),
                grad_ours_add_other.detach().cpu(),
                absolute_decimal=0,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

        test(self.data_b, self.packed_a)
        test(self.data_b_zero, self.packed_a_zero)
        test(self.data_b, self.packed_a[::2].contiguous())

    def test_packed_sub(self):
        def test(data_b, packed_a):
            assert isinstance(data_b, torch.Tensor)

            n_rays = len(packed_a)

            other = torch.rand((n_rays, data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float) - 0.5

            torch_sub = data_b.clone()
            for i in range(n_rays):
                torch_sub[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1], :].sub_(other[i][None, :])

            ours_sub = packed_sub(data_b, other, packed_a)
            self._compareTensor(
                torch_sub.detach().cpu(), ours_sub.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )

            # Generate the GT div: compute the loss and compare the gradients
            gt_sub = torch.rand_like(torch_sub, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_sub - gt_sub))
            loss_ours = torch.mean(torch.square(ours_sub - gt_sub))

            grad_torch_sub_a, grad_torch_sub_other = torch.autograd.grad(
                loss_torch, (data_b, other), grad_outputs=torch.ones_like(loss_torch)
            )
            grad_ours_sub_a, grad_ours_sub_other = torch.autograd.grad(
                loss_ours, (data_b, other), grad_outputs=torch.ones_like(loss_ours)
            )
            self._compareTensor(
                grad_torch_sub_a.detach().cpu(),
                grad_ours_sub_a.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )

            self._compareTensor(
                grad_torch_sub_other.detach().cpu(),
                grad_ours_sub_other.detach().cpu(),
                absolute_decimal=0,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

        test(self.data_b, self.packed_a)
        test(self.data_b_zero, self.packed_a_zero)
        test(self.data_b, self.packed_a[::2].contiguous())

    def test_packed_mul(self):
        def test(data_b, packed_a):
            assert isinstance(data_b, torch.Tensor)

            n_rays = len(packed_a)

            other = torch.rand((n_rays, data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float) - 0.5

            torch_mul = data_b.clone()
            for i in range(n_rays):
                torch_mul[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1], :].mul_(other[i][None, :])

            ours_mul = packed_mul(data_b, other, packed_a)
            self._compareTensor(
                torch_mul.detach().cpu(), ours_mul.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )

            # Generate the GT div: compute the loss and compare the gradients
            gt_mul = torch.rand_like(torch_mul, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_mul - gt_mul))
            loss_ours = torch.mean(torch.square(ours_mul - gt_mul))

            grad_torch_mul_a, grad_torch_mul_other = torch.autograd.grad(
                loss_torch, (data_b, other), grad_outputs=torch.ones_like(loss_torch)
            )
            grad_ours_mul_a, grad_ours_mul_other = torch.autograd.grad(
                loss_ours, (data_b, other), grad_outputs=torch.ones_like(loss_ours)
            )
            self._compareTensor(
                grad_torch_mul_a.detach().cpu(),
                grad_ours_mul_a.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )

            self._compareTensor(
                grad_torch_mul_other.detach().cpu(),
                grad_ours_mul_other.detach().cpu(),
                absolute_decimal=0,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

        test(self.data_b, self.packed_a)
        test(self.data_b_zero, self.packed_a_zero)
        test(self.data_b, self.packed_a[::2].contiguous())

    def test_packed_div(self):
        def test(data_b, packed_a):
            assert isinstance(data_b, torch.Tensor)

            n_rays = len(packed_a)

            other = torch.rand((n_rays, data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float) - 0.5

            torch_div = data_b.clone()
            for i in range(n_rays):
                torch_div[packed_a[i, 0] : packed_a[i, 0] + packed_a[i, 1], :].div_(other[i][None, :])

            ours_div = packed_div(data_b, other, packed_a)
            self._compareTensor(
                torch_div.detach().cpu(), ours_div.detach().cpu(), absolute_decimal=5, relative_decimal=5
            )

            # Generate the GT div: compute the loss and compare the gradients
            gt_div = torch.rand_like(torch_div, requires_grad=False)
            loss_torch = torch.mean(torch.square(torch_div - gt_div))
            loss_ours = torch.mean(torch.square(ours_div - gt_div))

            grad_torch_div_a, grad_torch_div_other = torch.autograd.grad(
                loss_torch, (data_b, other), grad_outputs=torch.ones_like(loss_torch)
            )
            grad_ours_div_a, grad_ours_div_other = torch.autograd.grad(
                loss_ours, (data_b, other), grad_outputs=torch.ones_like(loss_ours)
            )
            self._compareTensor(
                grad_torch_div_a.detach().cpu(),
                grad_ours_div_a.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )

            self._compareTensor(
                grad_torch_div_other.detach().cpu(),
                grad_ours_div_other.detach().cpu(),
                absolute_decimal=0,
                relative_decimal=5,
                ratio_of_permitted_failures=0.04,
            )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

        test(self.data_b, self.packed_a)
        test(self.data_b_zero, self.packed_a_zero)
        test(self.data_b, self.packed_a[::2].contiguous())

    def test_inverted_cdf(self):
        n_rays = 1000
        n_samples = 128
        samples = torch.full((n_rays,), n_samples, device="cuda", dtype=torch.int)

        bins = torch.sort(torch.rand((n_rays, n_samples)), dim=1).values.to(device="cuda")

        bins_pack_info = get_pack_info_from_n(samples)
        u_pack_info = get_pack_info_from_n(samples)

        # TODO: Below is the Zian's code for reference (check if Zian if ok to keep the sample size for cdf (removing one sample and normalizing))
        weights = (torch.rand((n_rays, n_samples)) + 1e-5).to(device="cuda")
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1).roll(1, 0)
        cdf[:, 0] = 0.0
        cdf /= cdf[:, -1:]

        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device).contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        # Check if our cuda implementation returns the same samples
        samples_ours, bin_idx_ours = packed_ops.packed_invert_cdf(
            bins.flatten(), cdf.flatten(), bins_pack_info, u.flatten(), u_pack_info, 1.0e-7
        )

        self._compareTensor(
            samples.detach().cpu(),
            samples_ours.view(n_rays, n_samples).detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.01,
        )

        self._compareTensor(
            inds.detach().cpu(),
            bin_idx_ours.view(n_rays, n_samples).detach().cpu(),
            absolute_decimal=6,
            relative_decimal=6,
            ratio_of_permitted_failures=0.01,
        )

    def test_interp(self):
        bins_np = np.array([1, 2, 3], dtype=np.float32)
        vals_np = np.array([3, 2, 0], dtype=np.float32)
        query_np = np.array([0, 1, 1.5, 2.72, 3.14], dtype=np.float32)
        interpolated_np = np.interp(query_np, bins_np, vals_np)
        interpolated_torch = torch.from_numpy(interpolated_np)

        bins = torch.tensor(bins_np, device="cuda")
        vals = torch.tensor(vals_np, device="cuda")
        query = torch.tensor(query_np, device="cuda")
        bins_pack_info = torch.tensor([[0, len(bins_np)]], dtype=torch.int, device="cuda")
        query_pack_info = torch.tensor([[0, len(query_np)]], dtype=torch.int, device="cuda")
        interpolated_ours, _ = packed_interp(bins, vals, bins_pack_info, query, query_pack_info)

        self._compareTensor(
            interpolated_torch,
            interpolated_ours.cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.01,
        )

    def test_compare_weighted_sum_to_torch(self):
        def test(data_b, weights_b, packed_a):
            assert isinstance(data_b, torch.Tensor)
            assert isinstance(weights_b, torch.Tensor)

            # Reference torch implementation
            weighted_sum_torch = []
            for ray in packed_a:
                weights = weights_b[ray[0] : ray[0] + ray[1]].reshape(-1, 1)
                data = data_b[ray[0] : ray[0] + ray[1], :]
                weighted_sum_torch.append(torch.sum(weights * data, dim=0))
            weighted_sum_torch = torch.vstack(weighted_sum_torch)

            # Compare the weights
            weighted_sum_ours = packed_weighted_sum(data_b, weights_b, packed_a)
            self._compareTensor(
                weighted_sum_torch.detach().cpu(),
                weighted_sum_ours.detach().cpu(),
                absolute_decimal=5,
                relative_decimal=5,
            )

            # Generate the GT weighted sum: compute the loss and compare the gradients
            gt_weighted_sum = torch.rand_like(weighted_sum_torch, requires_grad=False)
            loss_ours = torch.mean(torch.square(weighted_sum_ours - gt_weighted_sum))
            loss_torch = torch.mean(torch.square(weighted_sum_torch - gt_weighted_sum))

            grad_torch = torch.autograd.grad(loss_torch, (data_b, weights_b), grad_outputs=torch.ones_like(loss_torch))
            grad_ours = torch.autograd.grad(loss_ours, (data_b, weights_b), grad_outputs=torch.ones_like(loss_ours))
            for a, b in zip(grad_ours, grad_torch):
                self._compareTensor(
                    a.detach().cpu(), b.detach().cpu(), absolute_decimal=4, ratio_of_permitted_failures=0.04
                )

        test(self.data_b, self.weights_b, self.packed_a)
        test(self.data_b_zero, self.weights_b_zero, self.packed_a_zero)
        test(self.data_b, self.weights_b, self.packed_a[::2].contiguous())

    def test_packed_min_and_packed_max(self):
        # fmt: off
        v_packed = torch.tensor(
            [3.5,1.2,-2,9, #
             # [empty]
             2.2,-1,3.2, #
             6.0,7.0, #
             1.0,3.0,5.0,6.0,7.0 #
            ],
            device=torch.device("cuda"),
            dtype=torch.float,
        )
        # fmt: on
        v_pack_info = get_pack_info_from_n(torch.tensor([4, 0, 3, 2, 5], device=torch.device("cuda"), dtype=torch.long))
        v_max, argmax_idx = packed_max(v_packed, v_pack_info)
        v_min, argmin_idx = packed_min(v_packed, v_pack_info)

        self._compareTensor(
            v_max.cpu(), torch.tensor([9, 0, 3.2, 7.0, 7.0], dtype=torch.float, device=torch.device("cpu"))
        )
        self._compareTensor(
            argmax_idx.cpu(), torch.tensor([3, -1, 6, 8, 13], dtype=torch.int, device=torch.device("cpu"))
        )
        self._compareTensor(
            v_min.cpu(), torch.tensor([-2, 0, -1, 6.0, 1.0], dtype=torch.float, device=torch.device("cpu"))
        )
        self._compareTensor(
            argmin_idx.cpu(), torch.tensor([2, -1, 5, 7, 9], dtype=torch.int, device=torch.device("cpu"))
        )

    def test_packed_diff(self):
        def packed_diff_torch_simple(
            v: torch.Tensor,
            pack_info: torch.Tensor,
            appends: torch.Tensor | None = None,
            last_fill: torch.Tensor | None = None,
        ):
            n_max = int(pack_info[:, 1].max().item()) + 1
            v_ex = torch.zeros([len(pack_info), n_max], dtype=v.dtype, device=v.device).flatten()

            indices_start_in_ex = torch.arange(len(pack_info), device=v.device) * n_max
            indices_end_in_ex = indices_start_in_ex + pack_info[:, 1]
            indices_in_ex = linstep_interleave(indices_start_in_ex, pack_info[:, 1], 1).values
            non_empty_packs = pack_info[:, 1] > 0
            v_ex[indices_in_ex] = v

            if appends is not None:
                v_ex[indices_end_in_ex[non_empty_packs]] = appends[non_empty_packs]

            v_ex_diff = torch.cat(
                [
                    torch.diff(v_ex.view(len(pack_info), n_max), n=1, dim=-1),
                    torch.zeros([len(pack_info), 1], dtype=v.dtype, device=v.device),
                ],
                dim=-1,
            ).flatten()

            if last_fill is not None:
                v_ex_diff[(indices_end_in_ex - 1)[non_empty_packs]] = last_fill[non_empty_packs]
            elif appends is None:
                v_ex_diff[(indices_end_in_ex - 1)[non_empty_packs]] = 0
            return v_ex_diff[indices_in_ex]

        def packed_backward_diff_torch_simple(
            v: torch.Tensor,
            pack_info: torch.Tensor,
            prepends: torch.Tensor | None = None,
            first_fill: torch.Tensor | None = None,
        ):
            n_max = int(pack_info[:, 1].max().item()) + 1
            v_ex = torch.zeros([len(pack_info), n_max], dtype=v.dtype, device=v.device).flatten()

            indices_start_in_ex = torch.arange(len(pack_info), device=v.device) * n_max
            indices_in_ex = linstep_interleave(indices_start_in_ex + 1, pack_info[:, 1], 1).values
            non_empty_packs = pack_info[:, 1] > 0
            v_ex[indices_in_ex] = v

            if prepends is not None:
                v_ex[indices_start_in_ex[non_empty_packs]] = prepends[non_empty_packs]

            v_ex_diff = torch.cat(
                [
                    torch.diff(v_ex.view(len(pack_info), n_max), n=1, dim=-1),
                    torch.zeros([len(pack_info), 1], dtype=v.dtype, device=v.device),
                ],
                dim=-1,
            ).flatten()
            indices_in_ex_diff = linstep_interleave(indices_start_in_ex, pack_info[:, 1], 1).values

            if first_fill is not None:
                v_ex_diff[indices_start_in_ex[non_empty_packs]] = first_fill[non_empty_packs]
            elif prepends is None:
                v_ex_diff[indices_start_in_ex[non_empty_packs]] = 0
            return v_ex_diff[indices_in_ex_diff]

        # Prepare data
        # fmt: off
        v_packed = torch.tensor(
            [3.5,1.2,-2,9, #
             # [empty]
             7.7, #
             2.2,-1,3.2, #
             6.0,7.0, #
             1.0,3.0,5.0,6.0,7.0 #
            ],
            
            device=torch.device("cuda"),
            dtype=torch.float,
            requires_grad=True,
        )
        # fmt: on
        v_pack_info = get_pack_info_from_n(
            torch.tensor([4, 0, 1, 3, 2, 5], device=torch.device("cuda"), dtype=torch.long)
        )
        out_grad = torch.randn_like(v_packed)

        # Forward difference with no appends / last fill
        out_ours = packed_diff(v_packed, v_pack_info, None, None)
        out_torch = packed_diff_torch_simple(v_packed, v_pack_info, None, None)
        # fmt: off
        self._compareTensor(
            out_ours.data.cpu(),
            torch.tensor(
                [-2.3,-3.2,11.0,0.0, #
                # [empty]
                0.0, #
                -3.2,4.2,0.0, #
                1.0,0.0, #
                2.0,2.0,1.0,1.0,0.0 #
                ],
                device=torch.device("cpu"),
                dtype=torch.float,
            ),
        )
        # fmt: on
        self._compareTensor(out_ours.data.cpu(), out_torch.data.cpu())
        in_grad_ours = torch.autograd.grad(out_ours, v_packed, out_grad, retain_graph=True)[0]
        in_grad_torch = torch.autograd.grad(out_torch, v_packed, out_grad, retain_graph=True)[0]
        self._compareTensor(in_grad_ours.data.cpu(), in_grad_torch.data.cpu())

        # Forward difference with appends
        appends = torch.tensor(
            [10.0, 0.0, 3.0, -2.0, 8.0, 8.0], dtype=torch.float, device=torch.device("cuda"), requires_grad=True
        )
        out_ours = packed_diff(v_packed, v_pack_info, appends, None)
        out_torch = packed_diff_torch_simple(v_packed, v_pack_info, appends, None)
        # fmt: off
        self._compareTensor(
            out_ours.data.cpu(),
            torch.tensor(
                [-2.3,-3.2,11.0,1.0, #
                # [empty]
                -4.7, #
                -3.2,4.2,-5.2, #
                1.0,1.0, #
                2.0,2.0,1.0,1.0,1.0 #
                ],
                device=torch.device("cpu"),
                dtype=torch.float,
            ),
        )
        # fmt: on
        self._compareTensor(out_ours.data.cpu(), out_torch.data.cpu())
        in_grad_ours = torch.autograd.grad(out_ours, v_packed, out_grad, retain_graph=True)[0]
        in_grad_torch = torch.autograd.grad(out_torch, v_packed, out_grad, retain_graph=True)[0]
        self._compareTensor(in_grad_ours.data.cpu(), in_grad_torch.data.cpu())
        appends_grad_ours = torch.autograd.grad(out_ours, appends, out_grad, retain_graph=True)[0]
        appends_grad_torch = torch.autograd.grad(out_torch, appends, out_grad, retain_graph=True)[0]
        self._compareTensor(appends_grad_ours.data.cpu(), appends_grad_torch.data.cpu())

        # Backward difference with no prepends / first fill
        out_ours = packed_backward_diff(v_packed, v_pack_info, None, None)
        out_torch = packed_backward_diff_torch_simple(v_packed, v_pack_info, None, None)
        # fmt: off
        self._compareTensor(
            out_ours.data.cpu(),
            torch.tensor(
                [0.0,-2.3,-3.2,11.0, #
                # [empty]
                0.0, 
                0.0,-3.2,4.2, #
                0.0,1.0, #
                0.0,2.0,2.0,1.0,1.0 #
                ],
                device=torch.device("cpu"),
                dtype=torch.float,
            ),
        )
        # fmt: on
        self._compareTensor(out_ours.data.cpu(), out_torch.data.cpu())
        in_grad_ours = torch.autograd.grad(out_ours, v_packed, out_grad, retain_graph=True)[0]
        in_grad_torch = torch.autograd.grad(out_torch, v_packed, out_grad, retain_graph=True)[0]
        self._compareTensor(in_grad_ours.data.cpu(), in_grad_torch.data.cpu())

        # Backward difference with prepends
        prepends = torch.tensor(
            [10.0, 0.0, 3.0, -2.0, 8.0, 8.0], dtype=torch.float, device=torch.device("cuda"), requires_grad=True
        )
        out_ours = packed_backward_diff(v_packed, v_pack_info, prepends, None)
        out_torch = packed_backward_diff_torch_simple(v_packed, v_pack_info, prepends, None)
        # fmt: off
        self._compareTensor(
            out_ours.data.cpu(),
            torch.tensor(
                [-6.5,-2.3,-3.2,11.0, #
                # [empty]
                4.7,
                4.2,-3.2,4.2, #
                -2.0,1.0, #
                -7.0,2.0,2.0,1.0,1.0 #
                ],
                device=torch.device("cpu"),
                dtype=torch.float,
            ),
        )
        # fmt: on
        self._compareTensor(out_ours.data.cpu(), out_torch.data.cpu())
        in_grad_ours = torch.autograd.grad(out_ours, v_packed, out_grad, retain_graph=True)[0]
        in_grad_torch = torch.autograd.grad(out_torch, v_packed, out_grad, retain_graph=True)[0]
        self._compareTensor(in_grad_ours.data.cpu(), in_grad_torch.data.cpu())
        prepends_grad_ours = torch.autograd.grad(out_ours, prepends, out_grad, retain_graph=True)[0]
        prepends_grad_torch = torch.autograd.grad(out_torch, prepends, out_grad, retain_graph=True)[0]
        self._compareTensor(prepends_grad_ours.data.cpu(), prepends_grad_torch.data.cpu())


class TestPackedSearchSorted(CommonTestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")

        # packed and individually sorted data ranges
        # fmt: off
        self.bins = torch.tensor(
            [2, 3, 7, #
             1, #
             0, 4, #
             8, #
             3, 6, #
             ],
            device=self.device,
            dtype=torch.float,
        )
        # fmt: on
        self.pack_info = get_pack_info_from_n(torch.tensor([3, 1, 2, 1, 2], device=self.device, dtype=torch.long))

    def test_packed_searchsorted(self):
        self._compareTensor(
            # test case for initial + intermediate indices
            packed_searchsorted(
                self.bins,
                torch.tensor([4, 0, 2, 7, 1], device=self.device, dtype=torch.float).unsqueeze(-1),
                self.pack_info,
            ),
            torch.tensor([[2], [3], [5], [6], [7]], device=torch.device("cpu"), dtype=torch.long),
        )

        self._compareTensor(
            # test case for end + past-the-end indices
            packed_searchsorted(
                self.bins,
                torch.tensor([7, 1, 10, 9, 7], device=self.device, dtype=torch.float).unsqueeze(-1),
                self.pack_info,
            ).cpu(),
            torch.tensor([[3], [4], [6], [7], [9]], device=torch.device("cpu"), dtype=torch.long),
        )

        self._compareTensor(
            # test case for combined multi searches
            packed_searchsorted(
                self.bins,
                torch.tensor([[4, 7], [0, 1], [2, 10], [7, 9], [1, 7]], device=self.device, dtype=torch.float),
                self.pack_info,
            ).cpu(),
            torch.tensor([[2, 3], [3, 4], [5, 6], [6, 7], [7, 9]], device=torch.device("cpu"), dtype=torch.long),
        )

    def test_packed_searchsorted_packed_vals(self):
        # fmt: off
        u_packed = torch.tensor(
            [3.5,1.2,-2,9, #
             # [empty]
             2.2,-1,3.2, #
             6.0,7.0, #
             1.0,3.0,5.0,6.0,7.0 #
            ],
            device=self.device,
            dtype=torch.float,
        )
        # fmt: on
        u_pack_info = get_pack_info_from_n(torch.tensor([4, 0, 3, 2, 5], device=self.device, dtype=torch.long))

        # fmt: off
        self._compareTensor(
            # test case for combined multi searches (initial + intermediate + end + past-the-end indices indices)
            packed_searchsorted_packed_vals(self.bins, self.pack_info, u_packed, u_pack_info).cpu(),
            torch.tensor(
                [2, 0, 0, 3, #
                 # [empty]
                 5, 4, 5, #
                 6, 6, #
                 7, 8, 8, 9, 9 #
                ],
                device=torch.device("cpu"),
                dtype=torch.long,
            ),
        )
        # fmt: on


class TestInterleaveOps(CommonTestCase):
    def test_arange_interleave(self):
        device = torch.device("cuda")

        def arange_interleave_pt(n_per_pack: torch.Tensor) -> torch.Tensor:
            """
            A slower pytorch-based implementation of arange_interleave that returns a packed tensor with each pack_i being the result of torch.arange(n_per_pack[i])
            """
            assert n_per_pack.dim() == 1, "Only works for 1D Tensors."
            cumsum = n_per_pack.cumsum(0)
            pack_inds = cumsum - n_per_pack
            # BUG in pytorch-1.11. Fixed in pytorch-1.12
            # https://github.com/pytorch/pytorch/issues/78787
            # return torch.arange(cumsum[-1], device=n_per_pack.device)-pack_inds.repeat_interleave(n_per_pack)
            return torch.arange(cumsum[-1].item(), device=n_per_pack.device) - torch.repeat_interleave(
                pack_inds, n_per_pack
            )

        n_per_pack = torch.randint(32, 96, [4096], device=device)
        starts = n_per_pack.new_full(n_per_pack.shape, 0)
        step_size = n_per_pack.new_full(n_per_pack.shape, 1)
        stops = starts + step_size * n_per_pack

        y0 = arange_interleave_pt(n_per_pack).cpu()
        y1, _ = arange_interleave_simple(n_per_pack, False)
        y2, _ = arange_interleave(starts, stops, step_size, False)
        y3, _ = linstep_interleave(starts, n_per_pack, step_size, False)

        self._compareTensor(y0, y1.cpu())
        self._compareTensor(y0, y2.cpu())
        self._compareTensor(y0, y3.cpu())


class TestEmptyInputs(CommonTestCase):
    def test_everything(self, n_rays: int = 4096, feat_dim: int = 3):
        device = torch.device("cuda")
        non_empty_n_per_pack = torch.randint(0, 4, (n_rays,), dtype=torch.long, device=device)
        n_samples = int(non_empty_n_per_pack.sum().item())

        # Pack info for non-empty data
        pack_info_r_n = get_pack_info_from_n(non_empty_n_per_pack)
        # Pack info for empty data on `n_rays` ray packs
        pack_info_r_0n = torch.zeros((n_rays, 2), dtype=torch.long, device=device)
        # Pack info on 0 ray packs
        pack_info_0r = torch.empty((0, 2), dtype=torch.long, device=device)

        # Empty data with feature dimension
        data_0n_d = torch.empty((0, feat_dim), dtype=torch.float, device=device)
        # Empty data without feature dimension
        data_0n = torch.empty((0,), dtype=torch.float, device=device)
        data_0n_int = torch.empty((0,), dtype=torch.long, device=device)

        # Per-sample data with empty feature dimension (0-feature-dimension)
        data_n_0d = torch.empty((n_samples, 0), dtype=torch.float, device=device)
        # Per-sample data with feature dimension
        data_n_d = torch.randn((n_samples, feat_dim), dtype=torch.float, device=device)
        # Per-sample data without feature dimension
        data_n = torch.randn((n_samples,), dtype=torch.float, device=device)

        # Per-ray data with feature dimension
        data_r_d = torch.randn((n_rays, feat_dim), dtype=torch.float, device=device)
        # Per-ray data with empty feature dimension
        data_r_0d = torch.randn((n_rays, 0), dtype=torch.float, device=device)

        # Unary operators
        out = packed_diff(data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_diff(data_n_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_diff(data_n_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_backward_diff(data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_backward_diff(data_n_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_backward_diff(data_n_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_cumsum(data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_cumsum(data_n_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_cumsum(data_n_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_cumprod(data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_cumprod(data_n_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_cumprod(data_n_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        # Binary operators
        out = packed_add(data_0n_d, data_r_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_add(data_n_0d, data_r_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_add(data_n_d, data_r_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_sub(data_0n_d, data_r_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_sub(data_n_0d, data_r_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_sub(data_n_d, data_r_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_mul(data_0n_d, data_r_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_mul(data_n_0d, data_r_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_mul(data_n_d, data_r_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_div(data_0n_d, data_r_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_div(data_n_0d, data_r_0d, pack_info_r_n)
        assert [*out.shape] == [n_samples, 0]
        out = packed_div(data_n_d, data_r_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        # Reduction operators
        # NOTE: For reduction operators, always expect outputs to have batch_size = number of packs
        out = packed_sum(data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [n_rays, feat_dim]
        assert (out == 0).all()
        out = packed_sum(data_n_0d, pack_info_r_n)
        assert [*out.shape] == [n_rays, 0]
        out = packed_sum(data_n_d, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_weighted_sum(data_0n_d, data_0n, pack_info_r_0n)
        assert [*out.shape] == [n_rays, feat_dim] and (out == 0).all()
        out = packed_weighted_sum(data_n_0d, data_0n, pack_info_r_n)
        assert [*out.shape] == [n_rays, 0]
        out = packed_weighted_sum(data_n_d, data_n, pack_info_0r)
        assert [*out.shape] == [0, feat_dim]

        out = packed_max(data_0n, pack_info_r_0n)[0]
        assert [*out.shape] == [n_rays] and (out == 0).all()
        out = packed_max(data_n, pack_info_0r)[0]
        assert [*out.shape] == [0]

        out = packed_min(data_0n, pack_info_r_0n)[0]
        assert [*out.shape] == [n_rays] and (out == 0).all()
        out = packed_min(data_n, pack_info_0r)[0]
        assert [*out.shape] == [0]

        # Misc
        out = arange_interleave(data_0n, data_0n, 1).values
        assert [*out.shape] == [0]

        out = arange_interleave_simple(data_0n_int).values
        assert [*out.shape] == [0]

        out = linstep_interleave(data_0n, data_0n, 1).values
        assert [*out.shape] == [0]

        p, i, r_a, r_b = merge_two_packs_sorted_aligned(data_n, pack_info_r_n, data_0n, pack_info_r_0n)
        assert torch.equal(p, pack_info_r_n) and [*r_b.shape] == [0]
        p, i, r_a, r_b = merge_two_packs_sorted_aligned(data_0n, pack_info_r_0n, data_n, pack_info_r_n)
        assert torch.equal(p, pack_info_r_n) and [*r_a.shape] == [0]

        # searchsorted: Empty bins & empty search vals
        out = packed_searchsorted(data_0n, data_r_0d, pack_info_r_0n)
        assert [*out.shape] == [n_rays, 0]
        out = packed_searchsorted(data_0n, data_0n_d, pack_info_r_0n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_searchsorted_packed_vals(data_0n, pack_info_r_0n, data_0n, pack_info_r_0n)
        assert [*out.shape] == [0]
        out = packed_searchsorted_packed_vals(data_0n, pack_info_r_0n, data_n, pack_info_0r)
        assert [*out.shape] == [0]

        # searchsorted: Non-empty bins & empty search vals
        out = packed_searchsorted(data_n, data_r_0d, pack_info_r_n)
        assert [*out.shape] == [n_rays, 0]
        out = packed_searchsorted(data_n, data_0n_d, pack_info_r_n)
        assert [*out.shape] == [0, feat_dim]
        out = packed_searchsorted_packed_vals(data_n, pack_info_r_n, data_0n, pack_info_r_0n)
        assert [*out.shape] == [0]
        out = packed_searchsorted_packed_vals(data_n, pack_info_r_n, data_n, pack_info_0r)
        assert [*out.shape] == [0]


if __name__ == "__main__":
    unittest.main()
