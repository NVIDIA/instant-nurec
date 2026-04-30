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
import time
import unittest

from collections import namedtuple

import slangtorch
import torch

from libs.slang_utils.utils import add_ninja_to_path


def get_slang_module_path():
    from python.runfiles import runfiles

    r = runfiles.Create()
    path = r.Rlocation("nre_repo/libs/slang_gaussians/common/search_test.slang-module")
    return path


class TestSlangSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_ninja_to_path()
        slang_module_path = get_slang_module_path()
        slang_module = slangtorch.loadModule(slang_module_path)
        cls.slang_module = slang_module

    def test_search(self):
        TestArrays = namedtuple("TestArrays", ["values", "value_to_search", "expected_index", "expected_inside"])

        tests = [
            # Empty array.
            TestArrays([], 1, [-1, -1], False),
            # Single value array.
            TestArrays([5], 1, [0, 0], False),
            TestArrays([5], 5, [0, 0], True),
            TestArrays([5], 9, [0, 0], False),
            # More than one value array.
            TestArrays([2, 5], -1, [0, 0], False),
            TestArrays([2, 5], 1, [0, 0], False),
            TestArrays([2, 5], 2, [0, 0], True),
            TestArrays([2, 5], 3, [0, 1], True),
            TestArrays([2, 5], 4, [0, 1], True),
            TestArrays([2, 5], 5, [1, 1], True),
            # Multiple values arrays.
            TestArrays([1, 3, 6, 8, 9, 13], -1, [0, 0], False),
            TestArrays([1, 3, 6, 8, 9, 13], 0, [0, 0], False),
            TestArrays([1, 3, 6, 8, 9, 13], 1, [0, 0], True),
            TestArrays([1, 3, 6, 8, 9, 13], 2, [0, 1], True),
            TestArrays([1, 3, 6, 8, 9, 13], 3, [1, 1], True),
            TestArrays([1, 3, 6, 8, 9, 13], 4, [1, 2], True),
            TestArrays([1, 3, 6, 8, 9, 13], 5, [1, 2], True),
            TestArrays([1, 3, 6, 8, 9, 13], 6, [2, 2], True),
            TestArrays([1, 3, 6, 8, 9, 13], 7, [2, 3], True),
            TestArrays([1, 3, 6, 8, 9, 13], 8, [3, 3], True),
            TestArrays([1, 3, 6, 8, 9, 13], 9, [4, 4], True),
            TestArrays([1, 3, 6, 8, 9, 13], 10, [4, 5], True),
            TestArrays([1, 3, 6, 8, 9, 13], 11, [4, 5], True),
            TestArrays([1, 3, 6, 8, 9, 13], 12, [4, 5], True),
            TestArrays([1, 3, 6, 8, 9, 13], 13, [5, 5], True),
            TestArrays([1, 3, 6, 8, 9, 13], 14, [5, 5], False),
        ]

        # Build the arrays to pass to the kernel.
        values = []
        arrays = []
        values_to_search = []
        expected_out_indices = []
        expected_out_inside = []

        for test in tests:
            start_index = len(values)
            length = len(test.values)
            values.extend(test.values)
            arrays.append([start_index, length])
            values_to_search.append(test.value_to_search)
            expected_out_indices.append(test.expected_index)
            expected_out_inside.append(test.expected_inside)

        device = torch.device("cuda")
        values = torch.tensor(values, device=device, dtype=torch.int64)
        arrays = torch.tensor(arrays, device=device, dtype=torch.int32)
        values_to_search = torch.tensor(values_to_search, device=device, dtype=torch.int64)
        expected_out_indices = torch.tensor(expected_out_indices, device=device, dtype=torch.int32)
        expected_out_inside = torch.tensor(expected_out_inside, device=device, dtype=torch.bool)

        count = len(values_to_search)
        threads_per_block = 256
        blocks_per_grid = (count + threads_per_block - 1) // threads_per_block

        # Linear search, with int64 values and int32 indices.
        out_indices = torch.full_like(expected_out_indices, -2)
        out_inside = torch.full_like(expected_out_inside, False)
        self.slang_module.search_linear_int64(
            values=values,
            arrays=arrays,
            values_to_search=values_to_search,
            out_indices=out_indices,
            out_inside=out_inside,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))
        self.assertTrue(torch.equal(out_indices, expected_out_indices))
        self.assertTrue(torch.equal(out_inside, expected_out_inside))

        # Binary search, with int64 values and int32 indices.
        out_indices = torch.full_like(expected_out_indices, -2)
        out_inside = torch.full_like(expected_out_inside, False)
        self.slang_module.search_binary_int64(
            values=values,
            arrays=arrays,
            values_to_search=values_to_search,
            out_indices=out_indices,
            out_inside=out_inside,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))
        self.assertTrue(torch.equal(out_indices, expected_out_indices))
        self.assertTrue(torch.equal(out_inside, expected_out_inside))

        # Linear search, with float values and int64 indices.
        values = values.to(dtype=torch.float32)
        values_to_search = values_to_search.to(dtype=torch.float32)
        arrays = arrays.to(dtype=torch.int64)
        expected_out_indices = expected_out_indices.to(dtype=torch.int64)
        expected_out_inside = expected_out_inside.to(dtype=torch.bool)
        out_indices = torch.full_like(expected_out_indices, -2)
        out_inside = torch.full_like(expected_out_inside, False)
        self.slang_module.search_linear_float(
            values=values,
            arrays=arrays,
            values_to_search=values_to_search,
            out_indices=out_indices,
            out_inside=out_inside,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))
        self.assertTrue(torch.equal(out_indices, expected_out_indices))
        self.assertTrue(torch.equal(out_inside, expected_out_inside))

        # Binary search, with float values and int64 indices.
        out_indices = torch.full_like(expected_out_indices, -2)
        out_inside = torch.full_like(expected_out_inside, False)
        self.slang_module.search_binary_float(
            values=values,
            arrays=arrays,
            values_to_search=values_to_search,
            out_indices=out_indices,
            out_inside=out_inside,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))
        self.assertTrue(torch.equal(out_indices, expected_out_indices))
        self.assertTrue(torch.equal(out_inside, expected_out_inside))

    def test_unsigned_compilation_error(self):
        # We need to get the original source because it needs to be compiled
        # with the TRIGGER_UNSIGNED_COMPILATION_ERROR define, and the slang module was not.
        slang_source_path = os.path.join(os.path.dirname(__file__), "search_test.slang")
        with self.assertRaises(RuntimeError):
            slang_module = slangtorch.loadModule(slang_source_path, defines={"TRIGGER_UNSIGNED_COMPILATION_ERROR": "1"})

    def test_benchmark(self):
        # Build the arrays to pass to the kernel.
        values = []
        arrays = []
        values_to_search = []

        NB_ARRAYS = 1000
        NB_VALUES = (50, 150)
        VALUES_RANGE = (0, 1000)

        NB_WARMUP = 10
        NB_MEASURE = 1000

        import random

        random.seed(123)
        for _ in range(NB_ARRAYS):
            start_index = len(values)
            length = random.randint(NB_VALUES[0], NB_VALUES[1])

            vals = random.sample(range(VALUES_RANGE[0], VALUES_RANGE[1]), length)
            vals.sort()
            assert len(set(vals)) == length

            values.extend(vals)
            arrays.append([start_index, length])
            values_to_search.append(random.randint(VALUES_RANGE[0], VALUES_RANGE[1]))

        device = torch.device("cuda")
        values = torch.tensor(values, device=device, dtype=torch.int64)
        arrays = torch.tensor(arrays, device=device, dtype=torch.int32)
        values_to_search = torch.tensor(values_to_search, device=device, dtype=torch.int64)
        out_indices = torch.full_like(values_to_search, -2, dtype=torch.int32)
        out_inside = torch.full_like(values_to_search, False, dtype=torch.bool)

        count = len(values_to_search)
        threads_per_block = 256
        blocks_per_grid = (count + threads_per_block - 1) // threads_per_block

        for name, kernel in [
            ("linear", self.slang_module.search_linear_int64),
            ("binary", self.slang_module.search_binary_int64),
        ]:
            for _ in range(NB_WARMUP):
                kernel(
                    values=values,
                    arrays=arrays,
                    values_to_search=values_to_search,
                    out_indices=out_indices,
                    out_inside=out_inside,
                ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(NB_MEASURE):
                kernel(
                    values=values,
                    arrays=arrays,
                    values_to_search=values_to_search,
                    out_indices=out_indices,
                    out_inside=out_inside,
                ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"{name} took {(end_time - start_time) / NB_MEASURE * 1e3:.6f} ms")


if __name__ == "__main__":
    unittest.main()
