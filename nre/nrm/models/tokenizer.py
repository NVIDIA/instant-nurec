# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
COSMOS-1 Video Tokenizer Utilities

Utilities for loading the COSMOS-1 video tokenizer (VAE) that encodes video frames
into latent representations and decodes them back to pixel space.

Terminology:
    - Tokenizer/VAE: Complete model with encoder and decoder
    - Encoder: Video frames (B, C, T, H, W) → latents (B, C_latent, T', H', W')
    - Decoder: Latents → video frames

Reference: Adapted from https://github.com/nv-tlabs/lyra
"""

import os
import sys

import torch

from einops import rearrange


# Mock missing dependencies before importing cosmos_predict1
# These dependencies are required by cosmos_predict1 but may not be available in all environments
try:
    import mediapy  # type: ignore[import-not-found]
except ImportError:
    from unittest.mock import Mock

    sys.modules["mediapy"] = Mock()

try:
    import IPython  # type: ignore[import-not-found]
except ImportError:
    from unittest.mock import Mock

    sys.modules["IPython"] = Mock()
    sys.modules["IPython.display"] = Mock()

try:
    import loguru  # type: ignore[import-not-found]
except ImportError:
    from unittest.mock import Mock

    mock_loguru = Mock()
    mock_loguru.logger = Mock()
    sys.modules["loguru"] = mock_loguru

from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer  # type: ignore[import-not-found]
from cosmos_predict1.tokenizer.networks import TokenizerConfigs, TokenizerModels  # type: ignore[import-not-found]


def _get_tokenizer_config(checkpoint_path: str) -> dict:
    """
    Parse tokenizer configuration from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint (e.g., "/models/Cosmos-Tokenize1-CV8x8x8")

    Returns:
        dict: Config with 'name', 'latent_channels', 'temporal_compression', etc.
    """
    model_name = os.path.basename(checkpoint_path)
    model_name = model_name.split("Cosmos-Tokenize1-")[1].replace("-", "_")
    tokenizer_config = TokenizerConfigs[model_name].value
    return tokenizer_config


def load_cosmos_1_tokenizer(
    checkpoint_path: str,
    load_encoder: bool = True,
    load_decoder: bool = False,
    load_jit: bool = True,
    return_tokenizer_config: bool = False,
    add_tokenizer_kwargs: dict | None = None,
):
    """
    Load COSMOS-1 video tokenizer (VAE) with configurable components.

    Args:
        checkpoint_path: Path to checkpoint directory (contains encoder.jit, decoder.jit, mean_std.pt)
        load_encoder: Load encoder component: (B, 3, T, H, W) → (B, C_latent, T', H', W')
        load_decoder: Load decoder component: (B, C_latent, T', H', W') → (B, 3, T, H, W)
        load_jit: Use JIT-compiled models (faster inference) vs regular models (for training)
        return_tokenizer_config: Return (tokenizer, config_dict) instead of just tokenizer
        add_tokenizer_kwargs: Override default config parameters

    Returns:
        CausalVideoTokenizer or tuple[CausalVideoTokenizer, dict]

    Example:
        >>> tokenizer = load_cosmos_1_tokenizer("/models/Cosmos-Tokenize1-CV8x8x8")
        >>> latents = tokenizer.encode(video)  # (B, 3, T, H, W) -> (B, C, T', H', W')
    """
    tokenizer_kwargs = {}
    tokenizer_config: dict | None = None
    tokenizer_name: str | None = None

    if return_tokenizer_config or not load_jit:
        tokenizer_config = _get_tokenizer_config(checkpoint_path)
        tokenizer_name = tokenizer_config["name"]

    if load_encoder:
        tokenizer_kwargs["checkpoint_enc"] = f"{checkpoint_path}/encoder.jit"
    if load_decoder:
        tokenizer_kwargs["checkpoint_dec"] = f"{checkpoint_path}/decoder.jit"

    if not load_jit:
        # When load_jit=False, tokenizer_config and tokenizer_name are guaranteed to be set
        assert tokenizer_config is not None, "tokenizer_config must be set when load_jit=False"
        assert tokenizer_name is not None, "tokenizer_name must be set when load_jit=False"

        if add_tokenizer_kwargs:
            for k, v in add_tokenizer_kwargs.items():
                tokenizer_config[k] = v
        tokenizer = TokenizerModels[tokenizer_name].value(**tokenizer_config)
    else:
        tokenizer = CausalVideoTokenizer(**tokenizer_kwargs)

    if return_tokenizer_config:
        return tokenizer, tokenizer_config
    else:
        return tokenizer


def load_cosmos_latent_statistics(
    vae_path: str,
    pixel_chunk_duration: int = 121,
    device: torch.device | str = "cpu",
    weight_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load latent normalization statistics for diffusion models.

    Args:
        vae_path: Path to checkpoint directory (contains mean_std.pt)
        pixel_chunk_duration: Number of frames (e.g., 121) to compute latent temporal dimension
        device: Device for tensors ("cpu", "cuda", etc.)
        weight_dtype: Data type (e.g., torch.float16), defaults to checkpoint dtype

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (latent_mean, latent_std) with shape (1, C, T, 1, 1)
            where the leading 1 enables broadcasting across batch dimensions

    Example:
        >>> mean, std = load_cosmos_latent_statistics("/models/Cosmos-Tokenize1-CV8x8x8", device="cuda")
        >>> normalized = (latents - mean) / std  # For diffusion
    """
    tokenizer_config = _get_tokenizer_config(vae_path)
    latent_chunk_duration = (pixel_chunk_duration - 1) // tokenizer_config["temporal_compression"] + 1
    latent_mean, latent_std = _get_cosmos_diffusion_mean_std(
        vae_path, weight_dtype, tokenizer_config["latent_channels"], latent_chunk_duration
    )
    if isinstance(device, str):
        device = torch.device(device)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    return latent_mean, latent_std


def _get_cosmos_diffusion_mean_std(
    vae_dir: str, dtype: torch.dtype | None, latent_ch: int, latent_chunk_duration: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and reshape mean_std.pt to (1, C, T, 1, 1) format.

    Args:
        vae_dir: Directory containing mean_std.pt
        dtype: Target dtype, defaults to checkpoint dtype
        latent_ch: Number of latent channels
        latent_chunk_duration: Temporal dimension of latent

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (mean, std) with shape (1, latent_ch, latent_chunk_duration, 1, 1)
            where singleton dimensions (1) enable broadcasting
    """
    latent_mean, latent_std = torch.load(os.path.join(vae_dir, "mean_std.pt"), weights_only=True)
    if dtype is None:
        dtype = latent_mean.dtype
    target_shape = [1, latent_ch, latent_chunk_duration, 1, 1]
    latent_mean = latent_mean.view(latent_ch, -1)
    latent_std = latent_std.view(latent_ch, -1)
    latent_mean = latent_mean.to(dtype).reshape(*target_shape)
    latent_std = latent_std.to(dtype).reshape(*target_shape)
    return latent_mean, latent_std


def denormalize_latents(
    model_input: torch.Tensor,
    latent_std: torch.Tensor,
    latent_mean: torch.Tensor,
    num_input_multi_views: int = 1,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Denormalize diffusion model output to VAE decoder input format.

    Reverses normalization and converts latents from diffusion format to COSMOS decoder format,
    handling multi-view inputs and coordinate transformations.

    Args:
        model_input: Latent tensor, shape (B, V*T, C, H, W) or (V*T, C, H, W), range ~[-3, 3]
        latent_std: Per-channel std, shape (1, T, C, 1, 1) where 1 is for broadcasting across batches
        latent_mean: Per-channel mean, shape (1, T, C, 1, 1) where 1 is for broadcasting across batches
        num_input_multi_views: Number of views (V), same stats applied to all views
        sigma_data: Diffusion scaling factor (EDM σ_data parameter, default 0.5)

    Returns:
        torch.Tensor: Denormalized latents for decoder, shape (B, V*T, C, H, W) or (V*T, C, H, W)
            Data range depends on VAE statistics, typically wider than input after (x / sigma_data) * std + mean

    Example:
        >>> latents = torch.randn(1, 32, 16, 64, 64)  # V=4, T=8
        >>> decoder_input = denormalize_latents(latents, std, mean, num_input_multi_views=4)
        >>> print(decoder_input.shape)  # (1, 32, 16, 64, 64) -> (B, V*T, C, H, W)
    """
    # Validate input shapes
    assert len(model_input.shape) in [4, 5], (
        f"model_input must be 4D (V*T, C, H, W) or 5D (B, V*T, C, H, W), got shape {model_input.shape}"
    )
    assert len(latent_mean.shape) == 5 and latent_mean.shape[0] == 1, (
        f"latent_mean must have shape (1, T, C, 1, 1) for broadcasting, got {latent_mean.shape}"
    )
    assert len(latent_std.shape) == 5 and latent_std.shape[0] == 1, (
        f"latent_std must have shape (1, T, C, 1, 1) for broadcasting, got {latent_std.shape}"
    )
    assert (latent_std > 0).all(), "latent_std must contain only positive values"
    assert num_input_multi_views >= 1, f"num_input_multi_views must be >= 1, got {num_input_multi_views}"
    assert sigma_data > 0, f"sigma_data must be positive, got {sigma_data}"

    # Add batch dimension if needed
    if len(model_input.shape) == 4:
        model_input = model_input.unsqueeze(0)
        unsqueeze = True
    else:
        unsqueeze = False

    # Validate that temporal dimension is divisible by num_input_multi_views
    total_frames = model_input.shape[1]
    assert total_frames % num_input_multi_views == 0, (
        f"Total frames ({total_frames}) must be divisible by num_input_multi_views ({num_input_multi_views})"
    )

    # Use same statistics across views: (B, V*T, C, H, W) -> (B*V, T, C, H, W)
    model_input = rearrange(model_input, "b (v t) c h w -> (b v) t c h w", v=num_input_multi_views)
    # Undo sigma_data scaling
    model_input = model_input / sigma_data
    # Denormalize using VAE statistics (stats shape: 1, T, C, 1, 1)
    model_input = model_input * latent_std + latent_mean
    # Reshape frames and views again in one dimension: (B*V, T, C, H, W) -> (B, V*T, C, H, W)
    model_input = rearrange(model_input, "(b v) t c h w -> b (v t) c h w", v=num_input_multi_views)
    # Remove batch dimension
    if unsqueeze:
        model_input = model_input.squeeze(0)
    return model_input
