# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import sys
import tempfile
import threading
import types
import unittest

from unittest import mock


if "slangtorch" not in sys.modules:
    sys.modules["slangtorch"] = types.SimpleNamespace(
        util=types.SimpleNamespace(wrapModule=lambda module: module),
    )

from libs.slang_gaussians.collector import collector as collector_module
from libs.slang_gaussians.collector.codegen import CollectorConfiguration, CollectorKernelCode


class TestCollectorKernelCache(unittest.TestCase):
    def setUp(self):
        self._cached_kernels_backup = collector_module._cached_kernels.copy()
        self._prebuilt_configs_loaded_backup = collector_module._prebuilt_configs_loaded
        collector_module._cached_kernels.clear()
        collector_module._prebuilt_configs_loaded = False

    def tearDown(self):
        collector_module._cached_kernels.clear()
        collector_module._cached_kernels.update(self._cached_kernels_backup)
        collector_module._prebuilt_configs_loaded = self._prebuilt_configs_loaded_backup

    def test_load_prebuilt_configs_raises_for_duplicate_configuration(self):
        configuration = CollectorConfiguration(parameters=("Collector_Copy<3>",))
        collector_module._cached_kernels[configuration] = collector_module.CollectorKernel(
            slang_module=object(), kernel=object()
        )
        fake_extension = object()

        with tempfile.NamedTemporaryFile("w", suffix=".json") as tf:
            json.dump(
                [
                    {
                        "configuration": list(configuration.parameters),
                        "kernel_name": "collect_parameters_0",
                    }
                ],
                tf,
            )
            tf.flush()
            with mock.patch.object(
                collector_module.slangtorch.util,
                "wrapModule",
                return_value=types.SimpleNamespace(collect_parameters_0=object()),
            ):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    collector_module.load_prebuilt_configs(fake_extension, tf.name)

    def test_get_slang_kernels_compiles_once_for_concurrent_callers(self):
        configuration = CollectorConfiguration(parameters=("Collector_Copy<3>",))
        kernel_object = object()
        compile_calls = 0
        compile_gate = threading.Event()
        compile_started = threading.Event()
        results = []

        def fake_generate(configurations):
            self.assertEqual(configurations, [configuration])
            return CollectorKernelCode(code="// fake slang", kernel_names=("collect_parameters_0",))

        def fake_load_module(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            compile_started.set()
            compile_gate.wait(timeout=5)
            return types.SimpleNamespace(collect_parameters_0=kernel_object)

        def worker():
            results.append(collector_module.get_slang_kernels([configuration]))

        with (
            mock.patch.object(collector_module, "_load_prebuilt_configs_if_not_loaded", return_value=None),
            mock.patch.object(collector_module, "generate_collector_code", side_effect=fake_generate),
            mock.patch.object(collector_module, "add_ninja_to_path", return_value=None),
            mock.patch.object(collector_module, "get_slang_module_path", return_value="/tmp/collector.slang-module"),
            mock.patch.object(collector_module.slangtorch, "loadModule", side_effect=fake_load_module, create=True),
        ):
            first = threading.Thread(target=worker)
            second = threading.Thread(target=worker)
            first.start()
            self.assertTrue(compile_started.wait(timeout=5))
            second.start()
            compile_gate.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertEqual(compile_calls, 1)
        self.assertEqual(len(results), 2)
        self.assertIs(results[0][0], results[1][0])


if __name__ == "__main__":
    unittest.main()
