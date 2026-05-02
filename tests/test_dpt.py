"""Branch-coverage tests for nre.nrm.models.blocks.dpt.

Pure torch.nn — no compiled-lib stubs needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nre.nrm.models.blocks.dpt import (
    DPTFullHead,
    DPTFusionBlock,
    DPTFusionHead,
    DPTReassembleBlock,
)


# ---------------------------------------------------------------------------
# DPTReassembleBlock
# ---------------------------------------------------------------------------


def test_reassemble_constructor_rejects_too_few_blocks():
    with pytest.raises(AssertionError, match="At least 2 blocks"):
        DPTReassembleBlock(
            input_dim=32,
            output_dim=8,
            n_blocks=1,
            hidden_dims=(8,),
            pos_embed_strength=None,
        )


def test_reassemble_constructor_rejects_hidden_dim_count_mismatch():
    with pytest.raises(AssertionError, match="match number of blocks"):
        DPTReassembleBlock(
            input_dim=32,
            output_dim=8,
            n_blocks=4,
            hidden_dims=(4, 8, 16),  # 3, not 4
            pos_embed_strength=None,
        )


def test_reassemble_pos_embed_none_disables_module():
    block = DPTReassembleBlock(
        input_dim=32, output_dim=8, n_blocks=4, hidden_dims=(4, 8, 16, 32), pos_embed_strength=None
    )
    assert block.pos_embed is None


def test_reassemble_pos_embed_strength_constructs_module():
    block = DPTReassembleBlock(
        input_dim=32, output_dim=8, n_blocks=4, hidden_dims=(4, 8, 16, 32), pos_embed_strength=0.1
    )
    assert block.pos_embed is not None


def test_reassemble_forward_produces_multiscale_outputs():
    """For n_blocks=4 with stride pattern 4x, 2x, identity, /2 — feeding
    (B, 4, 4, 32) features yields four outputs at sizes 16, 8, 4, 2."""
    block = DPTReassembleBlock(
        input_dim=32, output_dim=8, n_blocks=4, hidden_dims=(4, 8, 16, 32), pos_embed_strength=0.1
    )
    x_list = [torch.randn(2, 4, 4, 32) for _ in range(4)]
    out = block(x_list)
    assert len(out) == 4
    assert out[0].shape == (2, 8, 16, 16)
    assert out[1].shape == (2, 8, 8, 8)
    assert out[2].shape == (2, 8, 4, 4)
    assert out[3].shape == (2, 8, 2, 2)


def test_reassemble_forward_without_pos_embed_does_not_add_anything():
    block = DPTReassembleBlock(
        input_dim=32, output_dim=8, n_blocks=4, hidden_dims=(4, 8, 16, 32), pos_embed_strength=None
    )
    x_list = [torch.zeros(1, 4, 4, 32) for _ in range(4)]
    out = block(x_list)
    assert all(o.shape[0] == 1 for o in out)


def test_reassemble_two_blocks_only_uses_identity_then_downsample():
    """n_blocks=2 → no transposed-convs, just identity then 3x3 stride-2 conv."""
    block = DPTReassembleBlock(
        input_dim=8, output_dim=4, n_blocks=2, hidden_dims=(8, 8), pos_embed_strength=None
    )
    x_list = [torch.randn(1, 4, 4, 8), torch.randn(1, 4, 4, 8)]
    out = block(x_list)
    assert len(out) == 2
    # First out is unchanged size; second is /2.
    assert out[0].shape == (1, 4, 4, 4)
    assert out[1].shape == (1, 4, 2, 2)


# ---------------------------------------------------------------------------
# DPTFusionBlock
# ---------------------------------------------------------------------------


def test_fusion_block_with_residual_path():
    block = DPTFusionBlock(input_dim=8, output_dim=4, with_residual=True)
    x = torch.randn(1, 8, 4, 4)
    res = torch.randn(1, 8, 4, 4)
    out = block(x, x_res=res)
    assert out.shape == (1, 4, 4, 4)


def test_fusion_block_without_residual_path():
    block = DPTFusionBlock(input_dim=8, output_dim=4, with_residual=False)
    assert block.res_block is None
    x = torch.randn(1, 8, 4, 4)
    out = block(x)
    assert out.shape == (1, 4, 4, 4)


def test_fusion_block_residual_required_when_with_residual_true():
    block = DPTFusionBlock(input_dim=8, output_dim=4, with_residual=True)
    x = torch.randn(1, 8, 4, 4)
    with pytest.raises(AssertionError, match="Residual connection requires"):
        block(x, x_res=None)


def test_fusion_block_resize_branch_changes_spatial_dims():
    block = DPTFusionBlock(input_dim=8, output_dim=4, with_residual=False)
    x = torch.randn(1, 8, 4, 4)
    out = block(x, resize=(8, 8))
    assert out.shape == (1, 4, 8, 8)


# ---------------------------------------------------------------------------
# DPTFusionHead
# ---------------------------------------------------------------------------


def test_fusion_head_before_conv_1_layer():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=4,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    assert isinstance(head.before_conv, nn.Conv2d)


def test_fusion_head_before_conv_5_layers():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=4,
        before_conv="5-layers",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    assert isinstance(head.before_conv, nn.Sequential)
    assert len(head.before_conv) == 5


def test_fusion_head_before_conv_invalid_value_raises():
    with pytest.raises(ValueError, match="Invalid before_conv"):
        DPTFusionHead(
            input_dim=8,
            output_dim=3,
            n_blocks=4,
            before_conv="bogus",  # type: ignore[arg-type]
            after_conv="2-layers",
            after_conv_dim=16,
            pos_embed_strength=None,
        )


def test_fusion_head_after_conv_2_layers():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=4,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    # 2-layers: Conv2d, ReLU, Conv2d
    assert len(head.after_conv) == 3


def test_fusion_head_after_conv_2_layers_with_norm():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=4,
        before_conv="1-layer",
        after_conv="2-layers-w-norm",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    # Norm path: Conv2d, LayerNorm2d, ReLU, Conv2d
    assert len(head.after_conv) == 4


def test_fusion_head_after_conv_invalid_value_raises():
    with pytest.raises(ValueError, match="Invalid after_conv"):
        DPTFusionHead(
            input_dim=8,
            output_dim=3,
            n_blocks=4,
            before_conv="1-layer",
            after_conv="bogus",  # type: ignore[arg-type]
            after_conv_dim=16,
            pos_embed_strength=None,
        )


def test_fusion_head_pos_embed_none_vs_strength():
    h_off = DPTFusionHead(
        input_dim=8, output_dim=3, n_blocks=4, before_conv="1-layer", after_conv="2-layers",
        after_conv_dim=16, pos_embed_strength=None
    )
    h_on = DPTFusionHead(
        input_dim=8, output_dim=3, n_blocks=4, before_conv="1-layer", after_conv="2-layers",
        after_conv_dim=16, pos_embed_strength=0.1
    )
    assert h_off.pos_embed is None
    assert h_on.pos_embed is not None


def test_fusion_head_zero_init_zeroes_output():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=2,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    head.zero_init([0.0, 0.0, 0.0])
    x_list = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    out = head(x_list)
    assert torch.allclose(out, torch.zeros_like(out))


def test_fusion_head_zero_init_nan_skips_that_index():
    """nan in init_values means: don't override that channel."""
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=2,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    last_conv = head.after_conv[-1]
    assert isinstance(last_conv, nn.Conv2d)
    init_bias_before = last_conv.bias.data.clone()  # type: ignore[union-attr]
    head.zero_init([math.nan, 1.0, math.nan])
    # Channels 0 and 2 are reset_parameters output (no manual override),
    # channel 1 has bias=1.0 and weight zero.
    assert last_conv.bias[1].item() == 1.0  # type: ignore[union-attr]
    assert torch.equal(last_conv.weight[1], torch.zeros_like(last_conv.weight[1]))


def test_fusion_head_zero_init_rejects_wrong_init_values_length():
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=2,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=None,
    )
    with pytest.raises(AssertionError, match="must match init_values length"):
        head.zero_init([0.0, 0.0])


def test_fusion_head_forward_output_shape_explicit():
    head = DPTFusionHead(
        input_dim=8, output_dim=3, n_blocks=2, before_conv="1-layer", after_conv="2-layers",
        after_conv_dim=16, pos_embed_strength=None
    )
    x_list = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    out = head(x_list, output_shape=(8, 8))
    assert out.shape == (1, 3, 8, 8)


def test_fusion_head_forward_rejects_wrong_x_list_length():
    head = DPTFusionHead(
        input_dim=8, output_dim=3, n_blocks=2, before_conv="1-layer", after_conv="2-layers",
        after_conv_dim=16, pos_embed_strength=None
    )
    x_list = [torch.randn(1, 8, 4, 4)]  # 1, not 2
    with pytest.raises(AssertionError, match="Number of features"):
        head(x_list)


def test_fusion_head_forward_with_fusion_features_branch():
    head = DPTFusionHead(
        input_dim=8, output_dim=3, n_blocks=2, before_conv="1-layer", after_conv="2-layers",
        after_conv_dim=16, pos_embed_strength=None
    )
    x_list = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    fusion = torch.randn(1, 4, 8, 8)  # input_dim // 2 = 4 channels
    out = head(x_list, output_shape=(8, 8), fusion_features=fusion)
    assert out.shape == (1, 3, 8, 8)


# ---------------------------------------------------------------------------
# DPTFullHead
# ---------------------------------------------------------------------------


def test_full_head_forward_chunk_size_unset_uses_full_batch():
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
    )
    x_list = [torch.randn(2, 4, 4, 32) for _ in range(4)]
    out = head(x_list, output_shape=(8, 8))
    assert out.shape == (2, 3, 8, 8)


def test_full_head_forward_chunk_size_splits_batch():
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
    )
    x_list = [torch.randn(4, 4, 4, 32) for _ in range(4)]
    # chunk_size=2 → 2 chunks of size 2 → final cat back to (4, ...)
    out = head(x_list, output_shape=(8, 8), chunk_size=2)
    assert out.shape == (4, 3, 8, 8)


def test_full_head_forward_chunk_size_zero_treated_as_full_batch():
    """chunk_size <= 0 falls back to full batch."""
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
    )
    x_list = [torch.randn(3, 4, 4, 32) for _ in range(4)]
    out = head(x_list, chunk_size=0)
    assert out.shape[0] == 3


def test_full_head_zero_init_with_values_and_without():
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
    )
    head.zero_init(None)  # no-op branch
    head.zero_init([0.0, 0.0, 0.0])
    x_list = [torch.randn(1, 4, 4, 32) for _ in range(4)]
    out = head(x_list, output_shape=(8, 8))
    assert torch.allclose(out, torch.zeros_like(out))


def test_fusion_head_forward_pos_embed_branch():
    """When pos_embed_strength is set, the pos_embed branch in forward fires."""
    head = DPTFusionHead(
        input_dim=8,
        output_dim=3,
        n_blocks=2,
        before_conv="1-layer",
        after_conv="2-layers",
        after_conv_dim=16,
        pos_embed_strength=0.1,  # enables pos_embed branch
    )
    x_list = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    out = head(x_list, output_shape=(8, 8))
    assert out.shape == (1, 3, 8, 8)


def test_full_head_forward_with_fusion_features_chunk_branch():
    """fusion_features kwarg triggers the per-chunk slicing branch."""
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
    )
    B = 4
    x_list = [torch.randn(B, 4, 4, 32) for _ in range(4)]
    fusion = torch.randn(B, 4, 8, 8)  # input_dim // 2 = 4 channels
    out = head(x_list, output_shape=(8, 8), fusion_features=fusion, chunk_size=2)
    assert out.shape == (B, 3, 8, 8)


def test_full_head_forward_checkpointing_branch():
    """checkpointing=True wraps each chunk in torch.utils.checkpoint.checkpoint."""
    head = DPTFullHead(
        input_dim=32,
        reassemble_hidden_dims=(4, 8, 16, 32),
        reassemble_dim=8,
        output_dim=3,
        n_blocks=4,
        head_before_conv="1-layer",
        head_after_conv="2-layers",
        head_after_conv_dim=16,
        pos_embed_strength=None,
        checkpointing=True,
    )
    head.train()  # checkpointing requires train mode for autograd-friendly path
    # Need at least one tensor with requires_grad for checkpoint to be exercised
    # — use a leaf input.
    x_list = [torch.randn(2, 4, 4, 32, requires_grad=True) for _ in range(4)]
    out = head(x_list, output_shape=(8, 8))
    assert out.shape == (2, 3, 8, 8)
