#!/usr/bin/env python3
"""Test that gsplat can be imported and CUDA extension works."""

import torch


def test_gsplat_basic_import():
    """Test basic gsplat import."""
    import gsplat

    assert gsplat.__version__ is not None
    print(f"gsplat version: {gsplat.__version__}")


def test_gsplat_rendering_import():
    """Test gsplat.rendering import."""
    from gsplat import rendering

    assert rendering is not None
    print("gsplat.rendering imported")


def test_gsplat_cuda_extension():
    """Test that CUDA extension actually loads and works with rasterization."""
    import gsplat

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA extension test")
        return

    # Create minimal test data for rasterization
    num_points = 100
    height, width = 64, 64

    means = torch.randn(1, num_points, 3, device="cuda")
    quats = torch.randn(1, num_points, 4, device="cuda")
    quats = torch.nn.functional.normalize(quats, dim=-1)  # Normalize quaternions
    scales = torch.rand(1, num_points, 3, device="cuda") * 0.1
    opacities = torch.rand(1, num_points, device="cuda")
    colors = torch.rand(1, num_points, 3, device="cuda")  # RGB colors

    # Simple camera parameters
    viewmats = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    Ks = torch.tensor([[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]], device="cuda").unsqueeze(
        0
    )  # (1, 1, 3, 3)

    # Call rasterization - this is what the actual renderer uses!
    render_colors, render_alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
    )

    assert render_colors is not None
    assert render_colors.shape == (1, 1, height, width, 3)
    assert render_alphas is not None
    assert render_alphas.shape == (1, 1, height, width, 1)


if __name__ == "__main__":
    print("Testing gsplat imports...")
    test_gsplat_basic_import()
    test_gsplat_rendering_import()
    test_gsplat_cuda_extension()
    print("\nAll tests successful!")
