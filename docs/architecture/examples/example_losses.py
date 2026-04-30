#!/usr/bin/env python3

"""
Example script demonstrating various loss functions in the NuRec losses module.

This script provides simple, runnable examples of different loss function implementations:

1. Python example: Demonstrates basic Python-based loss functions (relu_sum, bce_loss_clipped)
   that work on CPU or CUDA without requiring specialized kernels.

2. CUDA fused example: Demonstrates ModuleLosses, a fused CUDA kernel implementation that computes
   multiple losses simultaneously for improved performance. This example shows how to:
   - Create a ModuleLosses instance
   - Configure individual losses (RGB L1, background MSE)
   - Call the forward pass with model results and target data

3. CUDA example: Demonstrates RoadGaussiansLoss, a CUDA-accelerated loss function that
   constrains the height and rotation variance of gaussians in a specified layer.
   This example shows the CUDA implementation of the road gaussians distortion loss.

All examples use minimal mock data and are designed to be self-contained and easy to understand.

Read more in the README.md file.

"""

import logging
import time

import torch

from libs.losses.models import ModuleLosses, SlangBaseLoss
from libs.losses.models.loss_fns import bce_loss_clipped, relu_sum
from libs.losses.models.render_losses import RoadGaussiansLoss
from libs.losses.orchestration.config import LossItemConfig
from nre.config.trainer import TrainerConfig
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.gaussians_model import BaseGaussianModel
from nre.models.nn_extensions import TypedModuleDict, TypedModuleList
from nre.models.post_processing import BasePostProcessing
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta
from nre.utils.types import GaussiansCompositeReturn, GaussiansRenderReturn, RayFlags


logging.getLogger().setLevel(logging.ERROR)  # suppress verbose logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer_config = TrainerConfig(
    max_epochs=1,
    check_val_every_n_epoch=1,
    precision="32",
    log_every_n_steps=1,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
)

torch.manual_seed(42)  # for reproducibility


def python_example():
    """Run Python-based losses examples."""
    start_time = time.time()
    print("Python example:")

    input_tensor = torch.tensor([0.2, 0.7, 1.5], device=device, requires_grad=True)
    target_tensor = torch.tensor([0.0, 1.0, 1.0], device=device)

    relu_loss = relu_sum(input_tensor, eps=0.5)
    print("  ReLU sum loss:", relu_loss.cpu())

    bce_loss = bce_loss_clipped(input_tensor.sigmoid(), target_tensor, eps=0.01)
    print("  BCE clipped loss:", bce_loss.cpu())

    elapsed_time = time.time() - start_time
    print(f"  Time to run: {elapsed_time:.3f} seconds")


def slang_example():
    """Demonstrate Slang-based losses example."""
    start_time = time.time()
    print("\nSlang example:")

    # ModuleLosses requires CUDA
    if device.type != "cuda":
        print("WARNING: Slang requires CUDA. Skipping this example.")
        return

    # Create ModuleLosses instance
    module_losses = ModuleLosses()

    # Configure RGB L1 Mean loss
    loss_config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})

    # Append SlangBaseLoss to module_losses.losses (like LossAggregator does)
    module_losses.losses.append(SlangBaseLoss("rgb", loss_config, trainer_config))

    # Add background loss configuration
    bg_loss_config = LossItemConfig.model_validate({"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}})
    module_losses.losses.append(SlangBaseLoss("background", bg_loss_config, trainer_config))

    # Create minimal GaussiansComposite model using mock pattern from test
    # Intentionally skip super().__init__() to avoid needing config
    model = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({})
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.background = None

    # Create dummy data structures with random values
    B, H, W, C = 1, 2, 2, 3
    n_rays = B * H * W

    # Use random values for ground truth and predictions to get non-zero loss

    rgb_gt = torch.rand((B, H, W, C), device=device) * 0.5  # Random values [0, 0.5]
    rgb_pred = torch.rand((n_rays, C), device=device, requires_grad=True) * 0.8  # Random values [0, 0.8]
    flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)

    # Mark first pixel as sky for background loss
    flags_flat = flags.view(-1)
    flags_flat[0] |= RayFlags.SKY_SEMANTIC.value

    # Create opacity predictions for background loss
    opacity_pred = torch.rand(n_rays, device=device, requires_grad=True)

    # Create results and target
    rendered_cam = GaussiansRenderReturn(
        rgb=rgb_pred,
        opacity=opacity_pred,
        distance=torch.zeros(n_rays, device=device),
    )
    results = GaussiansCompositeReturn(rendered_cam=rendered_cam)

    camera_labels = CameraFrameLabels(flags=flags, rgb=rgb_gt)
    camera = DataBatch.Camera(meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)], labels=camera_labels)
    data_batch = DataBatch(idx=0, worker_id=None, sequence_id=["example"], camera=camera, lidar=None)
    target = DataAndRenderingBatch(data=data_batch)

    # Call ModuleLosses forward via call operator (as LossAggregator does)
    loss_returns = module_losses(step=0, model=model, results=results, target=target)

    print(f"  Computed {len(loss_returns)} loss(es):")
    for loss_name, loss_ret in loss_returns.items():
        print(f"    {loss_name}: {loss_ret.reduced_value.item():.6f}")

    elapsed_time = time.time() - start_time
    print(f"  Time to run: {elapsed_time:.3f} seconds")


def cuda_example():
    """Demonstrate RoadGaussiansLoss with CUDA implementation."""
    start_time = time.time()
    print("\nCUDA example: RoadGaussiansLoss")

    # RoadGaussiansLoss requires CUDA
    if device.type != "cuda":
        print("WARNING: RoadGaussiansLoss example requires CUDA. Skipping this example.")
        return

    # Create simple test data
    n_points = 50

    positions_world = torch.randn(n_points, 3, device=device, dtype=torch.float32)
    positions_world[:, 2] = torch.abs(positions_world[:, 2]) * 5  # Ensure positive z

    rotations_world = torch.randn(n_points, 4, device=device, dtype=torch.float32)

    # Create a simple camera pose (translation + quaternion)
    pose_tquat = torch.randn(7, device=device, dtype=torch.float32)
    pose_tquat[3:] = pose_tquat[3:] / torch.norm(pose_tquat[3:], dim=0, keepdim=True)  # Normalize quaternion

    # Create loss configuration
    loss_config = LossItemConfig.model_validate(
        {
            "fn": "mse",
            "lambda_": 1.0,
            "reduce": {"name": "mean"},
            "layer_name": "road",
            "n_samples": 3,
            "grid_len": 0.5,
            "min": -5.0,
            "range": 4.0,
            "rotation_lambda": 10.0,
        }
    )

    # Create CUDA implementation
    loss_cuda = RoadGaussiansLoss(loss_config, trainer_config, use_cuda=True)

    cuda_result = loss_cuda.forward_direct(positions_world, rotations_world, pose_tquat)

    print(f"  road_gaussians_abs_mean: {cuda_result.item():.6f}")

    elapsed_time = time.time() - start_time
    print(f"  Time to run: {elapsed_time:.3f} seconds")


def main():
    print(f"Using device: {device}")
    python_example()
    slang_example()
    cuda_example()


if __name__ == "__main__":
    main()
