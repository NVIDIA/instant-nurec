# Copyright (c) 2024-2025 NVIDIA CORPORATION.  All rights reserved.
"""
Convert legacy TokenGS checkpoint state_dict to Kelvin model keys.

Legacy TokenGS checkpoints use a different module layout and naming than the Kelvin
model (encoder / decoder / sky layout). This module detects such checkpoints and
converts key names and tensor shapes so that load_state_dict(..., strict=True) can
be used.

Key mappings (legacy TokenGS → Kelvin encoder/decoder)
------------------------------------------------------
1. activation_head.deconv → decoder.token_to_gs_linear

2. enc_dec_backbone.encoder.{i}.* → encoder.blocks.{i}.*
   - Per-block norm1, attn, norm2, mlp, ls1, ls2.
   - encoder_norm → decoder.feature_norm (legacy applied norm before kv_proj; new decoder applies feature_norm before kv_projector).

3. enc_dec_backbone.kv_proj → decoder.kv_projector (split into to_k, to_v).
   enc_dec_backbone.k_proj_norm / k_norm → decoder.kv_projector.k_norm.

4. enc_dec_backbone.decoder_blocks.* → decoder.blocks.*
   - gs_cross_attn.* → ca.*, gs_self_attn.* → sa.*, mlp.* → mlp_norm/mlp/ls_mlp.

5. tokenizer (9 ch) → encoder.patch_embed_img.proj + encoder.patch_embed_ray.proj
   input_layernorm → encoder.patch_embed_img.norm and encoder.patch_embed_ray.norm.

6. gs_tokens → decoder.gaussian_tokens

7. backbone_norm → decoder.norm (when use_decoder_norm=True).

8. Filled defaults for missing biases; gaussian_activations.xyz._z_offset_vec is
   not in legacy checkpoints — the caller injects it from the model before loading.
"""

from __future__ import annotations

import logging
import math
import re

from typing import Any

import torch


logger = logging.getLogger(__name__)


def is_legacy_tokengs_format_state_dict(state_dict: dict[str, Any]) -> bool:
    """Return True if the state_dict uses legacy TokenGS layout (enc_dec_backbone.*)."""
    return any(k.startswith("enc_dec_backbone.") for k in state_dict)


def infer_legacy_tokengs_depths(state_dict: dict[str, Any]) -> tuple[int, int]:
    """
    Infer encoder and decoder block counts from legacy TokenGS state_dict keys.
    Returns (encoder_depth, decoder_depth).
    """
    enc_depth = -1
    dec_depth = -1
    for k in state_dict:
        if m := re.match(r"enc_dec_backbone\.encoder\.(\d+)\.", k):
            enc_depth = max(enc_depth, int(m.group(1)) + 1)
        if m := re.match(r"enc_dec_backbone\.decoder_blocks\.(\d+)\.", k):
            dec_depth = max(dec_depth, int(m.group(1)) + 1)
    return (enc_depth if enc_depth >= 0 else 12, dec_depth if dec_depth >= 0 else 12)


def infer_legacy_tokengs_num_gaussian_tokens(state_dict: dict[str, Any]) -> int | None:
    """
    Infer number of Gaussian tokens from legacy TokenGS state_dict (gs_tokens shape).
    Returns None if gs_tokens is missing.
    """
    gs = state_dict.get("gs_tokens")
    if gs is None:
        return None
    return int(gs.shape[0])


def _infer_legacy_tokengs_layout(state_dict: dict[str, Any]) -> tuple[int, int, int, int, tuple[int, int]]:
    """
    Infer embed_dim, n_gaussian_params, n_gaussians_per_token, head_dim, patch_shape
    from a legacy TokenGS state_dict. Used by convert_legacy_tokengs_state_dict_to_kelvin.
    """
    sd = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # embed_dim from gs_tokens or tokenizer (check key presence; tensors are not bool-able)
    embed_dim = None
    gs = sd.get("gs_tokens")
    if gs is not None:
        embed_dim = int(gs.shape[1])
    w = sd.get("tokenizer.0.weight")
    if embed_dim is None and w is not None:
        embed_dim = int(w.shape[0])
    if embed_dim is None:
        embed_dim = 1024

    # patch_shape from tokenizer.0.weight
    ph = pw = 8
    if w is not None:
        if w.ndim == 4:
            ph, pw = int(w.shape[2]), int(w.shape[3])
        else:
            in_dim = int(w.shape[1])
            if in_dim % 9 == 0:
                patch_size = int(math.isqrt(in_dim // 9))
                if patch_size * patch_size == in_dim // 9:
                    ph = pw = patch_size

    n_gaussians_per_token = ph * pw

    # n_gaussian_params from activation_head.deconv.weight
    n_gaussian_params = 14
    deconv = sd.get("activation_head.deconv.weight")
    if deconv is not None:
        out_features = int(deconv.shape[0])
        if out_features % n_gaussians_per_token == 0:
            n_gaussian_params = out_features // n_gaussians_per_token

    num_heads = embed_dim // 64 if embed_dim % 64 == 0 else 16
    head_dim = embed_dim // num_heads
    return embed_dim, n_gaussian_params, n_gaussians_per_token, head_dim, (ph, pw)


def load_state_dict_keys_for_inference(path: str) -> dict[str, Any]:
    """
    Load a state dict from path (safetensors or .ckpt) with minimal memory.
    For .ckpt uses map_location='meta' to avoid loading tensor data.
    Returns state_dict suitable for is_legacy_tokengs_format_state_dict and
    infer_legacy_tokengs_depths.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    try:
        ckpt = torch.load(path, map_location="meta", weights_only=False)
        return ckpt.get("state_dict", ckpt)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("state_dict", ckpt)


def _encoder_key_legacy_to_kelvin(key: str) -> str | None:
    """enc_dec_backbone.encoder.{i}.x -> encoder.blocks.{i}.x"""
    if not key.startswith("enc_dec_backbone.encoder."):
        return None
    rest = key[len("enc_dec_backbone.encoder.") :]
    if (
        rest.startswith("norm.")
        or rest.startswith("kv_proj.")
        or rest.startswith("k_norm.")
        or rest.startswith("v_norm.")
    ):
        return None
    m = re.match(r"(\d+)\.(.+)", rest)
    if not m:
        return None
    i, suffix = m.group(1), m.group(2)
    return f"encoder.blocks.{i}.{suffix}"


def _decoder_suffix_legacy_to_kelvin(suffix: str) -> str:
    """Map legacy TokenGS decoder block sub-keys to Kelvin names."""
    for old, new in [
        ("gs_cross_attn.gs_token_norm.", "ca_norm."),
        ("gs_cross_attn.q_proj.", "ca.to_q."),
        ("gs_cross_attn.q_norm.", "ca.q_norm."),
        ("gs_cross_attn.out_proj.", "ca.proj."),
        ("gs_cross_attn_scale.gamma", "ls_ca.gamma"),
        ("gs_self_attn.norm.", "sa_norm."),
        ("gs_self_attn.gs_self_attn.qkv.", "sa.qkv."),
        ("gs_self_attn.gs_self_attn.q_norm.", "sa.q_norm."),
        ("gs_self_attn.gs_self_attn.k_norm.", "sa.k_norm."),
        ("gs_self_attn.gs_self_attn.proj.", "sa.proj."),
        ("gs_self_attn_scale.gamma", "ls_sa.gamma"),
        ("mlp.norm.", "mlp_norm."),
        ("mlp.mlp.fc1.", "mlp.fc1."),
        ("mlp.mlp.fc2.", "mlp.fc2."),
        ("mlp_scale.gamma", "ls_mlp.gamma"),
    ]:
        if suffix == old or suffix.startswith(old):
            return suffix.replace(old, new, 1)
    return suffix


def convert_legacy_tokengs_state_dict_to_kelvin(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a legacy TokenGS state_dict to Kelvin key names and tensor shapes.

    embed_dim, patch_shape, n_gaussian_params, and n_gaussians_per_token are inferred
    from the checkpoint (tokenizer, gs_tokens, activation_head.deconv). gs_tokens
    is passed through unchanged; load_state_dict(..., strict=True) will raise if
    its shape does not match the model. Encoder k_norm is kept per-head (head_dim)
    for legacy compatibility.

    The result is intended to be loaded with load_state_dict(..., strict=True).
    The only key that may still be missing is gaussian_activations.xyz._z_offset_vec
    (Kelvin-only); the caller should inject it from the model's current state_dict
    before loading so that all keys match.

    Parameters
    ----------
    state_dict : dict
        Raw state_dict from a legacy TokenGS checkpoint (may include "model." prefix).

    Returns
    -------
    dict
        State dict with Kelvin key names and compatible tensor shapes.
    """
    embed_dim, n_gaussian_params, n_gaussians_per_token, head_dim, patch_shape = _infer_legacy_tokengs_layout(
        state_dict
    )
    ph, pw = patch_shape

    out: dict[str, Any] = {}
    filled_in_conversion: list[str] = []  # Keys filled with defaults (e.g. missing biases), for logging

    for key, value in state_dict.items():
        k = key.replace("model.", "")

        # 1) activation_head.deconv -> decoder.token_to_gs_linear
        if k == "activation_head.deconv.weight":
            out["decoder.token_to_gs_linear.weight"] = value
            continue
        if k == "activation_head.deconv.bias":
            out["decoder.token_to_gs_linear.bias"] = value
            continue

        # 2) enc_dec_backbone.encoder.{i}.* -> encoder.blocks.{i}.*
        if new_k := _encoder_key_legacy_to_kelvin(k):
            out[new_k] = value
            if new_k.endswith(".weight") and (".norm1." in new_k or ".norm2." in new_k):
                bias_k = new_k.replace(".weight", ".bias")
                if bias_k not in out:
                    out[bias_k] = torch.zeros(value.shape[0], device=value.device, dtype=value.dtype)
                    filled_in_conversion.append(bias_k)
            continue

        # 2b) encoder_norm -> decoder.feature_norm (in legacy, norm was applied before kv_proj;
        #     in the new layout the decoder applies feature_norm before kv_projector, so load it there)
        if k == "enc_dec_backbone.encoder_norm.weight":
            out["decoder.feature_norm.weight"] = value
            if "decoder.feature_norm.bias" not in out:
                out["decoder.feature_norm.bias"] = torch.zeros(value.shape[0], device=value.device, dtype=value.dtype)
                filled_in_conversion.append("decoder.feature_norm.bias")
            continue
        if k == "enc_dec_backbone.encoder_norm.bias":
            out["decoder.feature_norm.bias"] = value
            continue

        # 2c) kv_proj -> decoder.kv_projector (split into to_k, to_v). k_norm -> decoder.kv_projector.k_norm.
        if k == "enc_dec_backbone.kv_proj.weight":
            # Legacy: one Linear (2*embed_dim, embed_dim) outputting [k; v]. Split for decoder.kv_projector.
            w = value
            dim = w.shape[1]
            assert w.shape[0] == 2 * dim, f"Expected kv_proj weight (2*dim, dim), got {w.shape}"
            out["decoder.kv_projector.to_k.weight"] = w[:dim].contiguous()
            out["decoder.kv_projector.to_v.weight"] = w[dim:].contiguous()
            continue
        if k == "enc_dec_backbone.kv_proj.bias":
            b = value
            dim = b.shape[0] // 2
            out["decoder.kv_projector.to_k.bias"] = b[:dim].contiguous()
            out["decoder.kv_projector.to_v.bias"] = b[dim:].contiguous()
            continue
        if k == "enc_dec_backbone.k_proj_norm.weight":
            out["decoder.kv_projector.k_norm.weight"] = value
            if "decoder.kv_projector.k_norm.bias" not in out:
                out["decoder.kv_projector.k_norm.bias"] = torch.zeros(head_dim, device=value.device, dtype=value.dtype)
                filled_in_conversion.append("decoder.kv_projector.k_norm.bias")
            continue
        if k == "enc_dec_backbone.k_norm.weight":
            out["decoder.kv_projector.k_norm.weight"] = value
            continue
        if k == "enc_dec_backbone.k_norm.bias":
            out["decoder.kv_projector.k_norm.bias"] = value
            continue

        if k.startswith("enc_dec_backbone.decoder_blocks."):
            continue

        # 4) gs_tokens -> decoder.gaussian_tokens
        if k == "gs_tokens":
            out["decoder.gaussian_tokens"] = value.contiguous()
            continue

        # 5) tokenizer (9ch) -> encoder.patch_embed_img + encoder.patch_embed_ray
        # Legacy: Linear(9*ph*pw, embed_dim) with input order (B,N,576) = position-major [pos0_rgb_pluck, pos1_...]
        # So weight (embed_dim, 576) -> reshape (embed_dim, ph, pw, 9); img = ch 0:3, ray = ch 3:9 per position.
        if k == "tokenizer.0.weight":
            w = value
            if w.ndim == 4:
                out["encoder.patch_embed_img.proj.weight"] = w[:, :3].contiguous()
                out["encoder.patch_embed_ray.proj.weight"] = w[:, 3:9].contiguous()
            else:
                emb, in_dim = w.shape
                assert in_dim == 9 * ph * pw, f"tokenizer weight last dim should be 9*ph*pw={9 * ph * pw}, got {in_dim}"
                w_patches = w.reshape(emb, ph, pw, 9)
                out["encoder.patch_embed_img.proj.weight"] = w_patches[:, :, :, :3].permute(0, 3, 1, 2).contiguous()
                out["encoder.patch_embed_ray.proj.weight"] = w_patches[:, :, :, 3:9].permute(0, 3, 1, 2).contiguous()
            continue
        if k == "tokenizer.0.bias":
            out["encoder.patch_embed_img.proj.bias"] = value.clone()
            out["encoder.patch_embed_ray.proj.bias"] = torch.zeros_like(value, device=value.device, dtype=value.dtype)
            continue

        # 6) input_layernorm -> encoder.patch_embed norms
        if k == "input_layernorm.weight":
            out["encoder.patch_embed_img.norm.weight"] = value.clone()
            out["encoder.patch_embed_ray.norm.weight"] = value.clone()
            continue
        if k == "input_layernorm.bias":
            out["encoder.patch_embed_img.norm.bias"] = value.clone()
            out["encoder.patch_embed_ray.norm.bias"] = value.clone()
            continue

        # 7) backbone_norm (after decoder in legacy) -> decoder.norm
        if k == "backbone_norm.weight":
            out["decoder.norm.weight"] = value
            continue
        if k == "backbone_norm.bias":
            out["decoder.norm.bias"] = value
            continue

        if (
            k.startswith("enc_dec_backbone.")
            or k.startswith("activation_head.")
            or k.startswith("tokenizer.")
            or k.startswith("input_layernorm.")
            or k == "gs_tokens"
        ):
            continue

        out[k] = value

    # Decoder blocks with converted suffix
    for key, value in state_dict.items():
        k = key.replace("model.", "")
        if not k.startswith("enc_dec_backbone.decoder_blocks."):
            continue
        rest = k[len("enc_dec_backbone.decoder_blocks.") :]
        m = re.match(r"(\d+)\.(.+)", rest)
        if not m:
            continue
        i, suffix = m.group(1), m.group(2)
        new_suffix = _decoder_suffix_legacy_to_kelvin(suffix)
        out[f"decoder.blocks.{i}.{new_suffix}"] = value

    # Fill missing LayerNorm biases and linear biases (encoder/decoder prefix)
    for key in list(out.keys()):
        if key.endswith(".weight"):
            dim = out[key].shape[0]
            bias_key = key.replace(".weight", ".bias")
            if bias_key not in out and (
                "norm1" in key
                or "norm2" in key
                or "decoder.norm" in key
                or "ca_norm" in key
                or "sa_norm" in key
                or "mlp_norm" in key
                or "ca.q_norm" in key
            ):
                out[bias_key] = torch.zeros(dim, device=out[key].device, dtype=out[key].dtype)
                filled_in_conversion.append(bias_key)
    if "encoder.patch_embed_img.proj.weight" in out and "encoder.patch_embed_img.proj.bias" not in out:
        ref = out["encoder.patch_embed_img.proj.weight"]
        out["encoder.patch_embed_img.proj.bias"] = torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
        filled_in_conversion.append("encoder.patch_embed_img.proj.bias")
    if "encoder.patch_embed_ray.proj.weight" in out and "encoder.patch_embed_ray.proj.bias" not in out:
        ref = out["encoder.patch_embed_ray.proj.weight"]
        out["encoder.patch_embed_ray.proj.bias"] = torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
        filled_in_conversion.append("encoder.patch_embed_ray.proj.bias")
    if "decoder.token_to_gs_linear.bias" not in out and "decoder.token_to_gs_linear.weight" in out:
        ref = out["decoder.token_to_gs_linear.weight"]
        out["decoder.token_to_gs_linear.bias"] = torch.zeros(
            ref.shape[0],
            device=ref.device,
            dtype=ref.dtype,
        )
        filled_in_conversion.append("decoder.token_to_gs_linear.bias")
    if "encoder.patch_embed_img.norm.weight" in out and "encoder.patch_embed_img.norm.bias" not in out:
        ref = out["encoder.patch_embed_img.norm.weight"]
        out["encoder.patch_embed_img.norm.bias"] = torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
        filled_in_conversion.append("encoder.patch_embed_img.norm.bias")
    if "encoder.patch_embed_ray.norm.weight" in out and "encoder.patch_embed_ray.norm.bias" not in out:
        ref = out["encoder.patch_embed_ray.norm.weight"]
        out["encoder.patch_embed_ray.norm.bias"] = torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
        filled_in_conversion.append("encoder.patch_embed_ray.norm.bias")

    if filled_in_conversion:
        logger.info(
            "Conversion filled missing parameters (e.g. biases not in legacy checkpoint): %s",
            sorted(filled_in_conversion),
        )
    return out
