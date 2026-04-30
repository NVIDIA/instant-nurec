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

import slangtorch
import torch

from libs.slang_gaussians.collector import get_slang_module_path
from libs.slang_utils.utils import add_ninja_to_path


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(123)

NB_GAUSSIANS = 1234
OFFSET = 567
THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = (NB_GAUSSIANS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

KERNEL_PRELUDE = """
import collector;

"""

KERNEL_TEMPLATE = """
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void {kernel_name}(
    no_diff uint out_offset,
    no_diff uint count,
    {collector_type} collector,
) {{
    run_tasks(out_offset, count, collector);
}}

"""


class TestSlangCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kernels_to_generate = [
            ("test_positions_copy", "Collector_Copy<3>"),
            ("test_rotations_normalize", "RotationsCollector_Normalize"),
            ("test_scales_exp", "ScalesCollector_Exp"),
            ("test_densities_sigmoid", "DensitiesCollector_Sigmoid"),
            ("test_extra_signal_copy", "Collector_Copy<5>"),
            ("test_camera_extra_signal_copy", "Collector_Copy<20>"),
            ("test_lidar_extra_signal_copy", "Collector_Copy<10>"),
            ("test_spherical_features_copy", "SphericalFeaturesCollector_Copy<3,45>"),
            (
                "test_spherical_features_fourier_individual",
                "SphericalFeaturesCollector_Fourier_Individual<3,45,20>",
            ),
            (
                "test_spherical_features_fourier_holistic",
                "SphericalFeaturesCollector_Fourier_Holistic<3,45,20>",
            ),
            (
                "test_spherical_features_fourier_individual_step",
                "SphericalFeaturesCollector_Fourier_IndividualStep<3,45,20,15>",
            ),
        ]
        compiled_kernels = {}
        code = KERNEL_PRELUDE
        for kernel_name, collector_type in kernels_to_generate:
            code += KERNEL_TEMPLATE.format(kernel_name=kernel_name, collector_type=collector_type)
        with tempfile.NamedTemporaryFile("w", suffix=".slang", delete=False) as tf:
            slang_file_path = tf.name
            tf.write(code)
        try:
            add_ninja_to_path()
            slang_module_path = get_slang_module_path()
            include_paths = [os.path.dirname(slang_module_path)]
            slang_module = slangtorch.loadModule(slang_file_path, verbose=False, includePaths=include_paths)
        finally:
            os.unlink(slang_file_path)

        for kernel_name, _ in kernels_to_generate:
            compiled_kernels[kernel_name] = (slang_module, getattr(slang_module, kernel_name))
        cls.kernels = compiled_kernels

    @staticmethod
    def _launch(kernel, collector):
        return kernel(out_offset=OFFSET, count=NB_GAUSSIANS, collector=collector).launchRaw(
            blockSize=(THREADS_PER_BLOCK, 1, 1), gridSize=(BLOCKS_PER_GRID, 1, 1)
        )

    def _test_inputs_to_output(
        self, input_functions, output_nb_components, kernel_name, parameter_type, ground_truth_function, atol
    ):
        def compare(a, b):
            if atol is None:
                return torch.equal(a, b)
            else:
                return torch.allclose(a, b, atol=atol)

        slang_module, kernel = self.kernels[kernel_name]

        # Forward pass.
        inputs = {}
        for func in input_functions:
            name, shape = func()
            input = torch.nn.Parameter(torch.randn(*shape, device=device))
            inputs[name] = input
        output = torch.randn(OFFSET + NB_GAUSSIANS, output_nb_components, device=device)
        output_untouched = output[:OFFSET].clone()
        ground_truth = ground_truth_function(inputs)

        collector = getattr(slang_module, parameter_type)(**inputs, output=output)

        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
        self._launch(kernel, collector)
        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertTrue(compare(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))

        # Backward pass.
        input_grads = {name: (input, torch.randn_like(input)) for name, input in inputs.items()}
        output_grad_ground_truth = torch.randn_like(output)

        collector = getattr(slang_module, parameter_type)(**input_grads, output=(output, output_grad_ground_truth))

        self._launch(kernel.bwd, collector)

        for name, (input, input_grad) in input_grads.items():
            input.grad = None
        torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)
        for name, (input, input_grad) in input_grads.items():
            self.assertTrue(compare(input.grad, input_grad))

    def _test_input_to_output(self, nb_components, kernel_name, parameter_type, ground_truth_function, atol):
        self._test_inputs_to_output(
            [lambda: ("input", (NB_GAUSSIANS, nb_components))],
            nb_components,
            kernel_name,
            parameter_type,
            lambda x: ground_truth_function(x["input"]),
            atol,
        )

    def test_positions_copy(self):
        self._test_input_to_output(3, "test_positions_copy", "Collector_Copy", lambda x: x.clone(), None)

    def test_rotations_normalize(self):
        self._test_input_to_output(
            4,
            "test_rotations_normalize",
            "RotationsCollector_Normalize",
            lambda x: torch.nn.functional.normalize(x),
            1e-6,
        )

    def test_scales_exp(self):
        self._test_input_to_output(3, "test_scales_exp", "ScalesCollector_Exp", lambda x: torch.exp(x), 1e-6)

    def test_densities_sigmoid(self):
        self._test_input_to_output(
            1, "test_densities_sigmoid", "DensitiesCollector_Sigmoid", lambda x: torch.sigmoid(x), 1e-6
        )

    def test_extra_signal_copy(self):
        self._test_input_to_output(5, "test_extra_signal_copy", "Collector_Copy", lambda x: x.clone(), None)

    def test_camera_extra_signal_copy(self):
        self._test_input_to_output(
            20,
            "test_camera_extra_signal_copy",
            "Collector_Copy",
            lambda x: x.clone(),
            None,
        )

    def test_lidar_extra_signal_copy(self):
        self._test_input_to_output(
            10,
            "test_lidar_extra_signal_copy",
            "Collector_Copy",
            lambda x: x.clone(),
            None,
        )

    def test_spherical_features_copy(self):
        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        self._test_inputs_to_output(
            [lambda: ("albedo", (NB_GAUSSIANS, ALBEDO_DIM)), lambda: ("specular", (NB_GAUSSIANS, SPECULAR_DIM))],
            ALBEDO_DIM + SPECULAR_DIM,
            "test_spherical_features_copy",
            "SphericalFeaturesCollector_Copy",
            lambda x: torch.cat([x["albedo"], x["specular"]], dim=1),
            None,
        )

    def test_spherical_features_fourier(self):
        atol = 1e-5

        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        FOURIER_DIM = 20

        def _get_individual_embedding():
            track_size = NB_GAUSSIANS // 3
            instance_idx = torch.zeros(NB_GAUSSIANS, dtype=torch.int32, device=device)
            instance_idx[track_size:] = 1
            timestamps_ranges = torch.tensor([[0, 1000000], [500000, 1500000]], dtype=torch.int64, device=device)
            remap_min = 0.0
            remap_max = 1.0
            time_embedding = slang_module.IndividualRemapTimeInputEmbedding(
                instance_idx=instance_idx,
                timestamps_ranges=timestamps_ranges,
                remap_min=remap_min,
                remap_max=remap_max,
            )
            return time_embedding

        def _evaluate_individual_embedding(time_embedding, timestamp):
            instance_idx = time_embedding.instance_idx
            timestamps_ranges = time_embedding.timestamps_ranges
            remap_min = time_embedding.remap_min
            remap_max = time_embedding.remap_max
            timestamps_us_ranges = timestamps_ranges[instance_idx]
            ratio = (timestamp - timestamps_us_ranges[:, 0]) / (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0])
            time_emb = (ratio.clamp(0, 1) * (remap_max - remap_min) + remap_min).unsqueeze(-1)
            return time_emb

        def _get_holistic_embedding():
            timestamps_us_min = 0
            timestamps_us_max = 1000000
            remap_min = 0.0
            remap_max = 1.0
            time_embedding = slang_module.HolisticRemapTimeInputEmbedding(
                timestamps_us_min=timestamps_us_min,
                timestamps_us_max=timestamps_us_max,
                remap_min=remap_min,
                remap_max=remap_max,
            )
            return time_embedding

        def _evaluate_holistic_embedding(time_embedding, timestamp):
            timestamps_us_min = time_embedding.timestamps_us_min
            timestamps_us_max = time_embedding.timestamps_us_max
            remap_min = time_embedding.remap_min
            remap_max = time_embedding.remap_max
            ratio = (timestamp - timestamps_us_min) / (timestamps_us_max - timestamps_us_min)
            ratio = torch.tensor([ratio], dtype=torch.float32, device=device)
            time_emb = (ratio.clamp(0, 1) * (remap_max - remap_min) + remap_min).unsqueeze(-1)
            return time_emb

        for kernel_name, get_embedding, evaluate_embedding, collector_type in [
            (
                "test_spherical_features_fourier_individual",
                _get_individual_embedding,
                _evaluate_individual_embedding,
                "SphericalFeaturesCollector_Fourier_Individual",
            ),
            (
                "test_spherical_features_fourier_holistic",
                _get_holistic_embedding,
                _evaluate_holistic_embedding,
                "SphericalFeaturesCollector_Fourier_Holistic",
            ),
        ]:
            slang_module, kernel = self.kernels[kernel_name]

            timestamp = 500000

            # Forward pass.
            time_embedding = get_embedding()
            albedo = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, FOURIER_DIM, ALBEDO_DIM, device=device))
            specular = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, SPECULAR_DIM, device=device))
            output = torch.randn(OFFSET + NB_GAUSSIANS, ALBEDO_DIM + SPECULAR_DIM, device=device)
            output_untouched = output[:OFFSET].clone()

            # Compute ground truth.
            time_embed = evaluate_embedding(time_embedding, timestamp)

            t = time_embed.view(-1, 1).float()
            dim = FOURIER_DIM
            idft = torch.zeros(t.shape[0], dim, device=device)
            indices = torch.arange(dim, device=device)
            even_indices = indices[::2]
            odd_indices = indices[1::2]
            idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)
            idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))

            ground_truth = torch.cat(
                [
                    torch.sum(albedo * idft.unsqueeze(-1), dim=1),
                    specular,
                ],
                -1,
            )

            collector = getattr(slang_module, collector_type)(
                time_embedding=time_embedding,
                timestamp=timestamp,
                albedo=albedo,
                specular=specular,
                output=output,
            )

            self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
            self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
            self._launch(kernel, collector)
            self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
            self.assertTrue(
                torch.allclose(
                    output[OFFSET : OFFSET + NB_GAUSSIANS, 0:ALBEDO_DIM], ground_truth[:, 0:ALBEDO_DIM], atol=atol
                )
            )
            self.assertTrue(
                torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS, ALBEDO_DIM:], ground_truth[:, ALBEDO_DIM:])
            )

            # Backward pass.
            albedo_grad = (albedo, torch.randn_like(albedo))
            specular_grad = (specular, torch.randn_like(specular))
            output_grad_ground_truth = torch.randn_like(output)

            collector = getattr(slang_module, collector_type)(
                time_embedding=time_embedding,
                timestamp=timestamp,
                albedo=albedo_grad,
                specular=specular_grad,
                output=(output, output_grad_ground_truth),
            )

            self._launch(kernel.bwd, collector)

            albedo.grad = None
            specular.grad = None
            torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)
            self.assertTrue(torch.allclose(albedo.grad, albedo_grad[1], atol=atol))
            self.assertTrue(torch.allclose(specular.grad, specular_grad[1], atol=atol))

    def test_spherical_features_fourier_individual_step(self):
        """Test IndividualStepTimeInputEmbedding following IndividualStepTimeInputEmbedding from input_embedding.py."""
        atol = 1e-5

        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        FOURIER_DIM = 20
        N_STEPS = 15
        N_TRACKS = 2

        slang_module, kernel = self.kernels["test_spherical_features_fourier_individual_step"]

        # Setup track assignments: first third -> track 0, rest -> track 1
        track_size = NB_GAUSSIANS // 3
        instance_idx = torch.zeros(NB_GAUSSIANS, dtype=torch.int32, device=device)
        instance_idx[track_size:] = 1

        # Per-track timestamp ranges
        timestamps_ranges = torch.tensor([[0, 1000000], [500000, 1500000]], dtype=torch.int64, device=device)

        # Learnable u parameter: [n_tracks, 1, n_steps]
        # Initialize similar to Python: linspace then reshape
        u_init = torch.linspace(0.0, 1.0, N_STEPS + 2)[1:-1].view(1, 1, N_STEPS)
        u = torch.nn.Parameter(u_init.repeat(N_TRACKS, 1, 1).to(device))

        # Beta parameter (stepness in microseconds)
        beta = torch.tensor(100000.0, dtype=torch.float32, device=device)

        timestamp = 500000

        def _evaluate_individual_step_embedding(instance_idx, timestamps_ranges, u, beta, timestamp):
            """Python ground truth following IndividualStepTimeInputEmbedding.forward() exactly."""
            timestamps_us_ranges = timestamps_ranges[instance_idx]  # [N, 2]

            # ratio = (timestamp - range_min) / (range_max - range_min)
            ratio = (timestamp - timestamps_us_ranges[:, 0].float()) / (
                timestamps_us_ranges[:, 1].float() - timestamps_us_ranges[:, 0].float()
            )

            # beta_scaled = beta / time_range, shape [N, 1, n_steps]
            time_range = (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0]).float()
            beta_scaled = (beta / time_range)[:, None, None].repeat(1, 1, N_STEPS)

            # output = ratio[:, None, None] - u[instance_idx]
            # u[instance_idx] has shape [N, 1, n_steps]
            output = ratio[:, None, None] - u[instance_idx]

            # Step function:
            #   if diff <= 0: 0.5 * exp(diff / clamp(|beta_scaled|, 1e-3))
            #   else: 1 - 0.5 * exp(-diff / clamp(|beta_scaled|, 1e-3))
            abs_beta = torch.clamp_min(torch.abs(beta_scaled), 1e-3)
            msk = output <= 0.0
            result = torch.empty_like(output)
            result[msk] = 0.5 * torch.exp(output[msk] / abs_beta[msk])
            result[~msk] = 1 - 0.5 * torch.exp(-output[~msk] / abs_beta[~msk])

            # Mean over steps: [N, 1]
            time_emb = result.mean(dim=-1)
            return time_emb

        # Forward pass
        albedo = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, FOURIER_DIM, ALBEDO_DIM, device=device))
        specular = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, SPECULAR_DIM, device=device))
        output = torch.randn(OFFSET + NB_GAUSSIANS, ALBEDO_DIM + SPECULAR_DIM, device=device)
        output_untouched = output[:OFFSET].clone()

        # Compute ground truth time embedding
        time_emb = _evaluate_individual_step_embedding(instance_idx, timestamps_ranges, u, beta, timestamp)

        # Compute Fourier features ground truth
        t = time_emb.view(-1, 1).float()
        dim = FOURIER_DIM
        idft = torch.zeros(t.shape[0], dim, device=device)
        indices = torch.arange(dim, device=device)
        even_indices = indices[::2]
        odd_indices = indices[1::2]
        idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)
        idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))

        ground_truth = torch.cat(
            [
                torch.sum(albedo * idft.unsqueeze(-1), dim=1),
                specular,
            ],
            -1,
        )

        # Create Slang time embedding struct
        time_embedding = slang_module.IndividualStepTimeInputEmbedding(
            instance_idx=instance_idx,
            timestamps_ranges=timestamps_ranges,
            u=u,
            beta=beta,
        )

        collector = slang_module.SphericalFeaturesCollector_Fourier_IndividualStep(
            time_embedding=time_embedding,
            timestamp=timestamp,
            albedo=albedo,
            specular=specular,
            output=output,
        )

        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
        self._launch(kernel, collector)
        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertTrue(
            torch.allclose(
                output[OFFSET : OFFSET + NB_GAUSSIANS, 0:ALBEDO_DIM], ground_truth[:, 0:ALBEDO_DIM], atol=atol
            )
        )
        self.assertTrue(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS, ALBEDO_DIM:], ground_truth[:, ALBEDO_DIM:]))

        # Backward pass
        u_grad = (u, torch.zeros_like(u))  # Zero-init because loadEx accumulates
        albedo_grad = (albedo, torch.randn_like(albedo))
        specular_grad = (specular, torch.randn_like(specular))
        output_grad_ground_truth = torch.randn_like(output)

        time_embedding_bwd = slang_module.IndividualStepTimeInputEmbedding(
            instance_idx=instance_idx,
            timestamps_ranges=timestamps_ranges,
            u=u_grad,
            beta=beta,
        )

        collector = slang_module.SphericalFeaturesCollector_Fourier_IndividualStep(
            time_embedding=time_embedding_bwd,
            timestamp=timestamp,
            albedo=albedo_grad,
            specular=specular_grad,
            output=(output, output_grad_ground_truth),
        )

        self._launch(kernel.bwd, collector)

        # Compute ground truth gradients
        u.grad = None
        albedo.grad = None
        specular.grad = None
        torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)

        self.assertTrue(torch.allclose(albedo.grad, albedo_grad[1], atol=atol))
        self.assertTrue(torch.allclose(specular.grad, specular_grad[1], atol=atol))
        # u gradient uses larger tolerance because:
        # 1. loadEx atomic accumulation can introduce small ordering-dependent errors
        # 2. Gradient flows through step function -> Fourier basis -> output (error amplification)
        # 3. Many Gaussians (1234) accumulate to few u elements (2 tracks * 15 steps = 30)
        u_atol = 5e-4
        self.assertTrue(torch.allclose(u.grad, u_grad[1], atol=u_atol))


if __name__ == "__main__":
    unittest.main()
