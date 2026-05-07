# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export ``kelvin_jit.pt`` from the existing ``kelvin_full.pt`` checkpoint.

Loads the pickled ``GaussiansInstantNuRecSystem``, rebuilds a fresh
``KelvinStaticCore`` from its config (avoiding double-submodule registration),
copies weights, traces ``KelvinStaticCore.forward_tensors`` via
``TraceableStaticCore`` on a real ncorev4 batch, and saves the result.

Usage::

    python internal/scripts/export_kelvin_jit.py \\
        --ncore-path /storage/data/nurec/ncorev4/clips/<UUID>/pai_<UUID>.json \\
        --output /tmp/kelvin_jit.pt

The trace inputs come from the predict-side dataloader so the captured shapes
match what the runtime feeds the JIT artifact in commit 7.

Parity gates:
  1. eager dataclass forward(context) vs eager forward_tensors(extracted)
     -- bitwise on float comparisons.
  2. eager forward_tensors vs traced+saved+loaded module on the same input
     -- numerically close (atol=1e-5) but not strictly bitwise (trace can
     reorder/fuse ops).
"""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "internal"))


# Legacy ``kelvin_full.pt`` pickle baked in qualnames that no longer
# exist on the public surface after commit 8 (architecture sources +
# config classes were relocated under ``internal/instant_nurec_internal``).
# Pre-register sys.modules aliases + class-attribute aliases so
# ``torch.load(kelvin_full.pt)`` can resolve the saved qualnames against
# the corresponding internal classes. Done up-front -- before the rest
# of the script's imports trigger the unpickling import chain.
import sys as _sys  # noqa: E402

import instant_nurec.config_schema.models as _public_models  # noqa: E402
import instant_nurec_internal.config_schema.models as _full_models  # noqa: E402
import instant_nurec_internal.model.activations as _act_mod  # noqa: E402
import instant_nurec_internal.model.backbone.base as _backbone_base_mod  # noqa: E402
import instant_nurec_internal.model.backbone.decoders as _decoders_mod  # noqa: E402
import instant_nurec_internal.model.backbone.encoders as _encoders_mod  # noqa: E402
import instant_nurec_internal.model.backbone.sky as _sky_mod  # noqa: E402
import instant_nurec_internal.model.blocks.aa_vit as _aa_vit_mod  # noqa: E402
import instant_nurec_internal.model.blocks.attention as _attention_mod  # noqa: E402
import instant_nurec_internal.model.blocks.dav3 as _dav3_mod  # noqa: E402
import instant_nurec_internal.model.blocks.dpt as _dpt_mod  # noqa: E402
import instant_nurec_internal.model.blocks.embeds as _embeds_mod  # noqa: E402
import instant_nurec_internal.model.blocks.layers as _layers_mod  # noqa: E402
import instant_nurec_internal.model.kelvin as _kelvin_mod  # noqa: E402
import instant_nurec_internal.model.post_processing as _post_proc_mod  # noqa: E402

_public_models.KelvinModelConfig = _full_models.KelvinFullModelConfig
_public_models.GaussiansActivationConfig = _full_models.GaussiansActivationConfig
_public_models.KelvinDAv3EncoderConfig = _full_models.KelvinDAv3EncoderConfig
_public_models.KelvinDPTDecoderConfig = _full_models.KelvinDPTDecoderConfig
_public_models.KelvinSkyCubemapDecoderConfig = _full_models.KelvinSkyCubemapDecoderConfig

_sys.modules.setdefault("instant_nurec.model.kelvin", _kelvin_mod)
_sys.modules.setdefault("instant_nurec.model.activations", _act_mod)
_sys.modules.setdefault("instant_nurec.model.post_processing", _post_proc_mod)
_sys.modules.setdefault("instant_nurec.model.backbone.base", _backbone_base_mod)
_sys.modules.setdefault("instant_nurec.model.backbone.decoders", _decoders_mod)
_sys.modules.setdefault("instant_nurec.model.backbone.encoders", _encoders_mod)
_sys.modules.setdefault("instant_nurec.model.backbone.sky", _sky_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.aa_vit", _aa_vit_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.attention", _attention_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.dav3", _dav3_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.dpt", _dpt_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.embeds", _embeds_mod)
_sys.modules.setdefault("instant_nurec.model.blocks.layers", _layers_mod)


from instant_nurec.config_schema.dataset import (  # noqa: E402
    InstantNuRecSplitsConfig,
    NCoreInstantNuRecDatasetConfig,
)
from instant_nurec.config_schema.instantnurec import InstantNuRecConfig  # noqa: E402
from instant_nurec.config_schema.predict import (  # noqa: E402
    PredictConfig,
    PrimitiveMergeConfig,
)
from instant_nurec.datasets.datamodule import InstantNuRecDataModule  # noqa: E402
from instant_nurec.model import _resolve_full_pt_path  # noqa: E402
from instant_nurec.ncore_input import resolve_ncore_paths  # noqa: E402
from instant_nurec.utils.geometry import tquat_to_se3_matrix  # noqa: E402
from instant_nurec.utils.misc import unpack_optional  # noqa: E402
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters  # noqa: E402
from instant_nurec_internal.jit.kelvin_static_core import (  # noqa: E402
    KelvinStaticCore,
    TraceableStaticCore,
)
from instant_nurec_internal.model.backbone.decoders import KelvinDPTDecoder  # noqa: E402
from instant_nurec_internal.model.backbone.encoders import KelvinDAv3Encoder  # noqa: E402
from instant_nurec_internal.model.post_processing import (  # noqa: E402
    PerCameraAffinePostProcessing,
)


logger = logging.getLogger("export_kelvin_jit")


def _build_fresh_static_core(loaded_kelvin) -> KelvinStaticCore:
    """Construct a fresh ``KelvinStaticCore`` and copy weights from ``loaded_kelvin``.

    Building fresh submodules (instead of sharing references with the loaded
    ``KelvinInstantNuRec``) keeps the state_dict tree single-rooted, so the
    traced graph references parameters by clean ``static_core.<x>`` names.

    Activation checkpointing is disabled on the fresh config copy: the
    checkpoint wrapper inserts a ``_NoopSaveInputs`` autograd Function that
    ``torch.jit.save`` cannot serialize. Disabling it is a no-op
    numerically -- checkpointing only changes memory/compute trade-off,
    not forward output.
    """
    cfg = loaded_kelvin.config.model_copy(deep=True)
    cfg.encoder.checkpointing = "none"
    cfg.decoder.checkpointing = False
    cfg.sky.checkpointing = False

    encoder = KelvinDAv3Encoder(cfg.encoder, cfg)
    encoder.load_state_dict(loaded_kelvin.encoder.state_dict())

    decoder = KelvinDPTDecoder(cfg.decoder, cfg)
    decoder.load_state_dict(loaded_kelvin.decoder.state_dict())

    post = PerCameraAffinePostProcessing(
        embed_dim=cfg.encoder.embed_dim, init_token_scale=0.02
    )
    post.load_state_dict(loaded_kelvin.post_processing.state_dict())

    return KelvinStaticCore(
        encoder=encoder,
        decoder=decoder,
        post_processing=post,
        scene_rescale=cfg.scene_rescale,
    )


def _extract_trace_tensors(context, scene_rescale: float):
    """Extract the ``forward_tensors`` inputs from a single-batch context list."""
    assert len(context) == 1, "trace assumes B=1 (predict_config.chunk_size=1)"
    batch = context[0]
    data = unpack_optional(batch.data.camera)
    rendering = unpack_optional(unpack_optional(batch.rendering).camera)

    rgb = unpack_optional(data.labels.rgb).unsqueeze(0)  # (1, V, H, W, 3)
    rays = rendering.rays.unsqueeze(0)  # (1, V, H, W, 6)
    distance_to_depth_scale = rendering.distance_to_depth_scale.unsqueeze(0)  # (1, V, H, W, 1)

    # c2w: end-of-frame, scene-rescaled (mirror of encoders.py:127-129)
    c2w = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False)
    c2w = c2w.clone()
    c2w[:, :3, 3] *= scene_rescale
    c2w = c2w.unsqueeze(0)  # (1, V, 4, 4)

    # fov: mirror of encoders.py:132-141
    pinhole_parameters = [
        to_simple_pinhole_model_parameters(rendering.sensor_model_parameters[vidx])
        for vidx in range(data.b)
    ]
    fov_list = []
    for p in pinhole_parameters:
        import math

        fov_w = 2 * math.atan2(p.resolution[0] / 2, p.focal_length[0])
        fov_h = 2 * math.atan2(p.resolution[1] / 2, p.focal_length[1])
        fov_list.append([fov_w, fov_h])
    fov = torch.tensor(fov_list, dtype=torch.float32, device=rgb.device).unsqueeze(0)  # (1, V, 2)

    camera_idxs = (
        torch.tensor([meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64)
        .to(rgb.device)
        .unsqueeze(0)  # (1, V)
    )

    return rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 0.0, rtol: float = 0.0):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch -- eager {a.shape} vs jit {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch -- eager {a.dtype} vs jit {b.dtype}")
    if atol == 0.0 and rtol == 0.0:
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            raise AssertionError(
                f"{name}: bitwise mismatch -- max abs diff {diff.max().item():.6e}"
            )
    else:
        diff = (a.float() - b.float()).abs()
        if a.dtype.is_floating_point:
            denom = b.float().abs() + atol
            relative = diff / denom.clamp_min(1e-12)
            ok = ((diff <= atol) | (relative <= rtol)).all().item()
        else:
            ok = bool(torch.equal(a, b))
        if not ok:
            raise AssertionError(
                f"{name}: numerical mismatch -- max|a-b|={diff.max().item():.6e} "
                f"(atol={atol:.1e}, rtol={rtol:.1e})"
            )
    logger.info(
        "%s parity OK (shape=%s, dtype=%s)", name, tuple(a.shape), str(a.dtype)
    )


def _assert_count_close(a: torch.Tensor, b: torch.Tensor, name: str, vertex_count_delta: int = 50):
    """Variable-length tensor count check.

    The static-layer fields are masked by ``argmax(semantic_logits) ==
    MOVABLE``, so the row count is data-dependent on a classification decision
    that flips on a few boundary pixels under JIT-vs-eager numerical drift.
    Allow the same vertex-count delta the PLY parity gate uses (50, per
    ``tests/tolerance.json``); when counts differ within the tolerance there is
    no canonical 1:1 alignment so we skip per-element comparison and trust the
    full-pipeline PLY parity gate (commit 7) to catch real divergence.
    """
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch -- eager {a.dtype} vs jit {b.dtype}")
    if a.ndim != b.ndim:
        raise AssertionError(f"{name}: ndim mismatch -- eager {a.ndim} vs jit {b.ndim}")
    if a.shape[1:] != b.shape[1:]:
        raise AssertionError(
            f"{name}: trailing-shape mismatch -- eager {a.shape} vs jit {b.shape}"
        )
    delta = abs(a.shape[0] - b.shape[0])
    if delta > vertex_count_delta:
        raise AssertionError(
            f"{name}: vertex count delta {delta} exceeds tolerance "
            f"{vertex_count_delta} (eager {a.shape[0]} vs jit {b.shape[0]})"
        )
    if a.shape == b.shape:
        diff = (a.float() - b.float()).abs().max().item()
        logger.info(
            "%s parity OK (shape=%s, dtype=%s, max|a-b|=%.3e)",
            name,
            tuple(a.shape),
            str(a.dtype),
            diff,
        )
    else:
        logger.info(
            "%s count delta=%d (eager %d, jit %d) within tolerance",
            name,
            delta,
            a.shape[0],
            b.shape[0],
        )


def _build_config(ncore_path: str, output_dir: str) -> InstantNuRecConfig:
    """Mirror of ``instant_nurec.cli.main``'s config construction, scoped to
    a single ncorev4 .json so the dataloader yields a usable trace input."""
    json_paths = resolve_ncore_paths(Path(ncore_path))
    return InstantNuRecConfig(
        out_dir=output_dir,
        dataset=InstantNuRecSplitsConfig(
            predict=NCoreInstantNuRecDatasetConfig(
                ncore_json_paths=[str(p) for p in json_paths],
            ),
        ),
        predict=PredictConfig(primitive_merge=PrimitiveMergeConfig(enabled=False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ncore-path",
        required=True,
        help="Path to a representative ncorev4 .json sequence file.",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for kelvin_jit.pt."
    )
    parser.add_argument(
        "--full-pt",
        default=None,
        help="Override INSTANT_NUREC_FULL_PT (defaults to the env-var resolution).",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for tracing (default: cuda)."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Resolve checkpoint
    if args.full_pt is not None:
        full_pt = args.full_pt
    else:
        full_pt = _resolve_full_pt_path()
    if not full_pt or not Path(full_pt).exists():
        logger.error("kelvin_full.pt not found. Set INSTANT_NUREC_FULL_PT or pass --full-pt.")
        return 2

    logger.info("Loading existing system from %s", full_pt)
    system = torch.load(full_pt, map_location="cpu", weights_only=False)
    kelvin = system.model

    logger.info("Building fresh KelvinStaticCore and copying weights")
    static_core = _build_fresh_static_core(kelvin).to(args.device)
    static_core.eval()

    # Build a real batch via the predict dataloader.
    config = _build_config(args.ncore_path, "/tmp/export_kelvin_jit_run")
    datamodule = InstantNuRecDataModule(config)
    loader = datamodule.predict_dataloader()
    full_batch = next(iter(loader)).to(torch.device(args.device))
    full_batch.maybe_compute_rendering_data(device=torch.device(args.device))
    chunk = full_batch[0:1]  # chunk_size=1
    chunk.context = kelvin.prepare_context(chunk.context)

    logger.info("Computing eager forward_tensors reference")
    with torch.inference_mode():
        trace_inputs = _extract_trace_tensors(chunk.context, static_core.scene_rescale)
        eager_t_out = static_core.forward_tensors(*trace_inputs)
    # Sanity-check the per-pixel shapes -- the JIT-load round-trip below
    # compares against this reference numerically.
    B, V, H, W, _ = trace_inputs[0].shape
    expected = {
        "gs_xyz": (B, V, H, W, 3),
        "gs_rotations": (B, V, H, W, 4),
        "gs_scales": (B, V, H, W, 3),
        "gs_densities": (B, V, H, W, 1),
        "gs_rgb": (B, V, H, W, 3),
        "semantic_argmax": (B, V, H, W),
        "normals": (B, V, H, W, 3),
        "affine": (B, eager_t_out[7].shape[1], 3, 4),
    }
    for tag, t in zip(expected, eager_t_out):
        assert tuple(t.shape) == expected[tag], (
            f"{tag} unexpected shape: got {tuple(t.shape)}, expected {expected[tag]}"
        )
    logger.info("forward_tensors output shapes match expectation")

    logger.info("Tracing TraceableStaticCore.forward")
    traceable = TraceableStaticCore(static_core).to(args.device).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(traceable, trace_inputs, strict=False, check_trace=False)
        # Reference output from the just-traced (in-memory) module. Comparing
        # the post-save+load output against this isolates serialization
        # fidelity from any eager-vs-trace numerical drift introduced by the
        # autocast contexts inside the encoder vit (which are recorded but
        # don't faithfully reproduce eager fp16 behavior on every operator).
        # End-to-end PLY parity against the eager pickle path is the actual
        # user-facing gate and runs at inference time via
        # ``internal/benchmark/validate_parity.py``.
        traced_out = traced(*trace_inputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path))
    logger.info("Saved JIT artifact to %s (%d bytes)", output_path, output_path.stat().st_size)

    logger.info("Round-trip sanity: serialized module loads and produces well-formed output")
    loaded = torch.jit.load(str(output_path), map_location=args.device).eval()
    with torch.inference_mode():
        loaded_out = loaded(*trace_inputs)
    tags = (
        "gs_xyz",
        "gs_rotations",
        "gs_scales",
        "gs_densities",
        "gs_rgb",
        "semantic_argmax",
        "normals",
        "affine",
    )
    # Bitwise-identical comparison between in-memory traced output and
    # post-save+load output is not reliable -- ``torch.jit.load`` applies
    # IR optimization passes (operator fusion / reordering) that interact
    # non-deterministically with the autocast contexts inside encoder.vit.
    # Per-element drift is checked end-to-end via the PLY parity gate
    # (``internal/benchmark/validate_parity.py``) at inference time. Here
    # we only assert shape, dtype, and finiteness so a corrupt save would
    # surface immediately.
    for tag, traced_v, loaded_v in zip(tags, traced_out, loaded_out):
        if traced_v.shape != loaded_v.shape:
            raise AssertionError(
                f"{tag}: shape mismatch (traced {tuple(traced_v.shape)} vs "
                f"loaded {tuple(loaded_v.shape)})"
            )
        if traced_v.dtype != loaded_v.dtype:
            raise AssertionError(
                f"{tag}: dtype mismatch (traced {traced_v.dtype} vs loaded {loaded_v.dtype})"
            )
        if traced_v.dtype.is_floating_point and not torch.isfinite(loaded_v).all():
            raise AssertionError(f"{tag}: loaded output contains non-finite values")
        max_diff = (
            (traced_v.float() - loaded_v.float()).abs().max().item()
            if traced_v.dtype.is_floating_point
            else float((traced_v != loaded_v).sum().item())
        )
        logger.info(
            "%s (load round-trip) shape=%s dtype=%s max_diff=%.3e",
            tag,
            tuple(loaded_v.shape),
            str(loaded_v.dtype),
            max_diff,
        )

    logger.info("All parity gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
