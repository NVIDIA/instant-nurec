# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.transforms.functional import gaussian_blur

import instant_nurec.training.renderer as renderer_module
import instant_nurec.training.run as training_run_module
import instant_nurec.training.losses as losses_module
from instant_nurec.config_schema.dataset import (
    InstantNuRecSplitsConfig,
    NCoreInstantNuRecDatasetConfig,
)
from instant_nurec.config_schema.train import (
    TrainingConfig,
    TrainingLoggerConfig,
    TrainingLossConfig,
    TrainingSchedulerConfig,
    TrainingSystemConfig,
)
from instant_nurec.model.post_processing import PerCameraAffinePostProcessing
from instant_nurec.model.blocks.layers import LayerNorm2d
from instant_nurec.model.blocks.dav3 import convert_dav3_state_dict
from instant_nurec.model.supervision import SupervisionPack
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive, KelvinStaticLayer
from instant_nurec.training.losses import TrainingLosses
from instant_nurec.training.optim import CosineWithWarmupPBScheduler
from instant_nurec.training.renderer import RenderOutput, render_supervision
from instant_nurec.training.run import (
    initialize_resume_logger_before_distributed_setup,
    make_checkpoint_callback,
    make_training_logger,
    resolve_training_strategy,
)
from instant_nurec.training.system import TrainingSystem
from instant_nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    RenderingBatch,
    RenderingData,
)
from instant_nurec.utils.types import RayFlags
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType


def _dataset_config() -> NCoreInstantNuRecDatasetConfig:
    return NCoreInstantNuRecDatasetConfig(ncore_json_paths=["/tmp/example.json"])


def test_render_phase_selects_official_loss_profile_and_ddp_strategy(tmp_path):
    config = TrainingConfig(
        out_dir=tmp_path,
        phase="render",
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )
    assert config.system.enable_render_global_step == 0
    assert config.system.strategy == "ddp_find_unused_parameters_true"
    assert config.loss == TrainingLossConfig.render_phase()


def test_context_phase_keeps_official_optimizer_defaults(tmp_path):
    config = TrainingConfig(out_dir=tmp_path, dataset=InstantNuRecSplitsConfig(train=_dataset_config()))
    assert config.system == TrainingSystemConfig()
    assert config.system.optimizer.lr == 1.0e-4
    assert config.system.optimizer.eps == 1.0e-15
    assert config.system.optimizer.betas == (0.9, 0.99)
    assert config.system.precision == "bf16-mixed"
    assert config.logger == TrainingLoggerConfig()
    assert config.loss == TrainingLossConfig.context_phase()


def test_layer_norm_2d_uses_channel_last_fused_layer_norm(monkeypatch):
    module = LayerNorm2d(3, eps=1.0e-5)
    value = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    calls = []
    original = torch.nn.functional.layer_norm

    def tracking_layer_norm(input_value, normalized_shape, weight, bias, eps):
        calls.append((tuple(input_value.shape), normalized_shape, eps))
        return original(input_value, normalized_shape, weight, bias, eps)

    monkeypatch.setattr(torch.nn.functional, "layer_norm", tracking_layer_norm)

    actual = module(value)
    expected = original(value.permute(0, 2, 3, 1), (3,), module.weight, module.bias, module.eps).permute(0, 3, 1, 2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [((1, 2, 4, 3), (3,), 1.0e-5)]


def test_train_epoch_reseeds_runtime_rng_by_epoch_and_rank(monkeypatch):
    seeds = []
    monkeypatch.setenv("PL_GLOBAL_SEED", "38")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch, "manual_seed", seeds.append)

    TrainingSystem.on_train_epoch_start(SimpleNamespace(current_epoch=3))

    assert seeds == [52]


def test_train_epoch_does_not_reseed_without_distributed_runtime(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: pytest.fail(f"unexpected seed {seed}"))

    TrainingSystem.on_train_epoch_start(SimpleNamespace(current_epoch=3))


def test_wandb_logger_uses_stable_run_id_for_resume(tmp_path, monkeypatch):
    captured = {}

    class FakeWandbLogger:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(training_run_module, "WandbLogger", FakeWandbLogger)
    config = TrainingConfig(
        out_dir=tmp_path,
        run_id="stable-run-id",
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
        logger=TrainingLoggerConfig(
            name="wandb",
            project="instant-nurec",
            entity="example-org",
            run_name="training-test",
            group="training",
            tags=["training"],
            job_type="render",
        ),
    )

    logger = make_training_logger(config, tmp_path / config.run_id)

    assert isinstance(logger, FakeWandbLogger)
    assert captured["id"] == "stable-run-id"
    assert captured["resume"] == "allow"
    assert captured["project"] == "instant-nurec"
    assert captured["entity"] == "example-org"


def test_resumed_wandb_initializes_on_external_global_rank_zero_before_ddp(tmp_path, monkeypatch):
    accesses = []

    class FakeLogger:
        @property
        def experiment(self):
            accesses.append("experiment")
            return object()

    checkpoint = tmp_path / "last.ckpt"
    config = TrainingConfig(
        out_dir=tmp_path,
        resume_from_checkpoint=checkpoint,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
        logger=TrainingLoggerConfig(name="wandb"),
    )
    monkeypatch.setenv("SLURM_PROCID", "0")

    assert initialize_resume_logger_before_distributed_setup(config, FakeLogger())
    assert accesses == ["experiment"]


def test_resumed_wandb_remains_lazy_on_nonzero_external_rank(tmp_path, monkeypatch):
    class FakeLogger:
        @property
        def experiment(self):
            raise AssertionError("nonzero ranks must not initialize W&B")

    checkpoint = tmp_path / "last.ckpt"
    config = TrainingConfig(
        out_dir=tmp_path,
        resume_from_checkpoint=checkpoint,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
        logger=TrainingLoggerConfig(name="wandb"),
    )
    monkeypatch.setenv("SLURM_PROCID", "3")

    assert not initialize_resume_logger_before_distributed_setup(config, FakeLogger())


def test_fresh_wandb_run_keeps_lightning_lazy_initialization(tmp_path, monkeypatch):
    class FakeLogger:
        @property
        def experiment(self):
            raise AssertionError("fresh runs should keep Lightning's normal lazy initialization")

    config = TrainingConfig(
        out_dir=tmp_path,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
        logger=TrainingLoggerConfig(name="wandb"),
    )
    monkeypatch.setenv("SLURM_PROCID", "0")

    assert not initialize_resume_logger_before_distributed_setup(config, FakeLogger())


@pytest.mark.parametrize(("resume", "expected_autolog"), [(False, True), (True, False)])
def test_trainer_autologs_hparams_only_for_fresh_runs(tmp_path, monkeypatch, resume, expected_autolog):
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, *args, **kwargs):
            captured["fit_ckpt_path"] = kwargs["ckpt_path"]

    checkpoint = tmp_path / "last.ckpt"
    config = TrainingConfig(
        out_dir=tmp_path,
        resume_from_checkpoint=checkpoint if resume else None,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )
    config.system.accelerator = "cpu"
    config.system.devices = 1
    monkeypatch.setattr(training_run_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(training_run_module, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_run_module, "TrainingDataModule", lambda _config: object())
    monkeypatch.setattr(training_run_module, "TrainingSystem", lambda _config: object())
    monkeypatch.setattr(training_run_module, "make_training_logger", lambda *args: object())
    monkeypatch.setattr(training_run_module, "make_training_callbacks", lambda *args: [])
    monkeypatch.setattr(
        training_run_module,
        "initialize_resume_logger_before_distributed_setup",
        lambda *args: False,
    )

    training_run_module.run_training(config)

    assert captured["enable_autolog_hparams"] is expected_autolog
    assert captured["fit_ckpt_path"] == (str(checkpoint) if resume else None)


def test_training_run_id_can_be_shared_across_distributed_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTANT_NUREC_RUN_ID", "shared-run")

    config = TrainingConfig(
        out_dir=tmp_path,
        run_id="rank-local-yaml-value",
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )

    assert config.run_id == "shared-run"


def test_epoch_checkpoint_tracks_official_validation_psnr(tmp_path):
    config = TrainingConfig(out_dir=tmp_path, dataset=InstantNuRecSplitsConfig(train=_dataset_config()))

    checkpoint = make_checkpoint_callback(config, tmp_path / config.run_id)

    assert checkpoint.monitor == "val/psnr"
    assert checkpoint.mode == "max"
    assert checkpoint.save_top_k == 2
    assert checkpoint.save_last


def test_fresh_weights_initialize_before_lightning_sanity_validation():
    calls = []
    system = SimpleNamespace(
        _weights_initialized=False,
        trainer=SimpleNamespace(ckpt_path=None),
        initialize_weights=lambda: calls.append("initialized"),
    )

    TrainingSystem.on_fit_start(system)

    assert calls == ["initialized"]


def test_resume_skips_component_initialization_before_sanity_validation():
    calls = []
    system = SimpleNamespace(
        _weights_initialized=False,
        trainer=SimpleNamespace(ckpt_path="last.ckpt"),
        initialize_weights=lambda: calls.append("initialized"),
    )

    TrainingSystem.on_fit_start(system)

    assert calls == []


def test_multigpu_context_forces_find_unused_ddp(tmp_path, monkeypatch):
    config = TrainingConfig(
        out_dir=tmp_path,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )
    monkeypatch.setattr(training_run_module.torch.cuda, "device_count", lambda: 4)

    assert resolve_training_strategy(config) == "ddp_find_unused_parameters_true"


def test_single_gpu_context_keeps_lightning_auto_strategy(tmp_path, monkeypatch):
    config = TrainingConfig(
        out_dir=tmp_path,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )
    monkeypatch.setattr(training_run_module.torch.cuda, "device_count", lambda: 1)

    assert resolve_training_strategy(config) == "auto"


def test_integer_all_devices_context_forces_find_unused_ddp(tmp_path, monkeypatch):
    config = TrainingConfig(
        out_dir=tmp_path,
        dataset=InstantNuRecSplitsConfig(train=_dataset_config()),
    )
    config.system.devices = -1
    monkeypatch.setattr(training_run_module.torch.cuda, "device_count", lambda: 4)

    assert resolve_training_strategy(config) == "ddp_find_unused_parameters_true"


@pytest.mark.parametrize(
    ("filename", "phase", "enable_render_step"),
    [
        ("full_model_front_context.yaml", "context", 1_000_000),
        ("full_model_front_render.yaml", "render", 0),
    ],
)
def test_checked_in_training_configs_validate(filename, phase, enable_render_step):
    path = Path(__file__).resolve().parents[1] / "configs" / "training" / filename
    config = TrainingConfig.model_validate(yaml.safe_load(path.read_text()))

    assert config.phase == phase
    assert config.system.enable_render_global_step == enable_render_step
    assert config.system.deterministic == "warn"
    assert config.system.strategy == "ddp_find_unused_parameters_true"
    assert config.system.checkpoint_monitor == "val/psnr"
    assert config.system.checkpoint_mode == "max"
    assert config.system.save_top_k == 2
    assert config.logger.name == "wandb"
    assert config.logger.project == "instant-nurec"
    assert isinstance(config.dataset.train, NCoreInstantNuRecDatasetConfig)
    dataset = config.dataset.train
    assert dataset.frame_batch_sampler.name == "uniform"
    assert dataset.frame_batch_sampler.frame_gap_timestamp_us == 500_000
    assert dataset.frame_batch_sampler.n_samples_per_sequence == 10
    assert dataset.supervision_frame_batch.n_frames_per_camera == 6
    assert dataset.supervision_frame_batch.prepend_timestamps_us == 100_000
    assert dataset.supervision_frame_batch.append_timestamps_us == 100_000


def test_dav3_state_dict_conversion_covers_encoder_and_depth_head():
    source = {
        "model.backbone.pretrained.cls_token": torch.arange(4.0).reshape(1, 1, 4),
        "model.backbone.pretrained.pos_embed": torch.arange(20.0).reshape(1, 5, 4),
        "model.backbone.pretrained.patch_embed.proj.weight": torch.ones(4, 3, 2, 2),
        "model.backbone.pretrained.blocks.0.norm1.weight": torch.ones(4),
        "model.head.projects.0.weight": torch.ones(2, 4, 1, 1),
        "model.head.scratch.output_conv1.weight": torch.ones(2, 2, 3, 3),
        "model.gs_head.weight": torch.ones(1),
    }

    converted = convert_dav3_state_dict(
        source,
        patch_embed_name="patch_embed_img",
        backbone_name="vit",
        dpt_reassemble_name="depth_head.reassemble",
        dpt_depth_head_name="depth_head.fusion_head",
        dpt_rays_head_name=None,
        camera_encoder_name=None,
    )

    assert set(converted) == {
        "vit.cls_tokens",
        "vit.cls_pos_embed",
        "vit.img_pos_embed",
        "patch_embed_img.proj.weight",
        "vit.blocks.0.norm1.weight",
        "depth_head.reassemble.proj_layers.0.weight",
        "depth_head.fusion_head.before_conv.weight",
    }
    assert converted["vit.img_pos_embed"].shape == (2, 2, 4)


def test_progress_scheduler_matches_reference_initial_and_warmup_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    scheduler = CosineWithWarmupPBScheduler(optimizer, TrainingSchedulerConfig())
    assert optimizer.param_groups[0]["lr"] == 1.0e-10
    scheduler.set_progress(epoch=0, total_epochs=40, local_step=0, total_local_steps=1000)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)
    scheduler.set_progress(epoch=0, total_epochs=40, local_step=99, total_local_steps=1000)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0e-4


def test_quantile_reduce_matches_reference_single_value_edge_case():
    from instant_nurec.training.losses import _quantile_mean

    value = torch.tensor([3.0], requires_grad=True)
    reduced = _quantile_mean(value, 0.98)
    torch.testing.assert_close(reduced, torch.tensor(0.0))


def test_affine_zero_init_and_delayed_gradient_contract():
    module = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.02)
    module.zero_init()
    latent = torch.randn(1, 2, 32, requires_grad=True)
    matrix, bias = module.decode_affine(latent)
    torch.testing.assert_close(matrix, torch.eye(3)[None, None].expand(1, 2, 3, 3))
    torch.testing.assert_close(bias, torch.zeros(1, 2, 3))
    assert matrix.requires_grad

    module.set_detach_linear_grad(True)
    detached_matrix, detached_bias = module.decode_affine(latent)
    assert not detached_matrix.requires_grad
    assert not detached_bias.requires_grad


def test_no_aux_context_rgb_loss_remains_trainable():
    rgb = torch.ones(1, 2, 2, 3)
    labels = CameraFrameLabels(
        rgb=rgb,
        flags=torch.full((1, 2, 2, 1), int(RayFlags.RGB_LABEL), dtype=torch.int32),
    )
    context = DataAndRenderingBatch(
        data=DataBatch(camera=DataBatch.Camera(meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)], labels=labels)),
        rendering=SimpleNamespace(camera=SimpleNamespace(distance_to_depth_scale=torch.ones(1, 2, 2, 1))),
    )
    prediction = torch.zeros_like(rgb, requires_grad=True)
    pack = SupervisionPack(context_rgb=prediction)
    config = TrainingLossConfig(
        primitive_sky_cubemap=0,
        primitive_rgb=0.1,
        primitive_distance=0,
        primitive_distance_gradient=0,
        primitive_semantics=0,
        primitive_normal=0,
        primitive_velocity=0,
    )
    result = TrainingLosses(config)(pack, context)
    torch.testing.assert_close(result.values["primitive_rgb"], torch.tensor(1.0))
    torch.testing.assert_close(result.total_value, torch.tensor(0.1))
    result.total_value.backward()
    assert prediction.grad is not None


def _camera_batch(labels: CameraFrameLabels) -> DataAndRenderingBatch:
    return DataAndRenderingBatch(
        data=DataBatch(
            camera=DataBatch.Camera(
                meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)],
                labels=labels,
            )
        ),
        rendering=SimpleNamespace(camera=SimpleNamespace(distance_to_depth_scale=torch.ones_like(labels.flags))),
    )


def _render_only_config(**weights) -> TrainingLossConfig:
    defaults = {
        "primitive_sky_cubemap": 0,
        "primitive_rgb": 0,
        "primitive_distance": 0,
        "primitive_distance_gradient": 0,
        "primitive_semantics": 0,
        "primitive_normal": 0,
        "primitive_velocity": 0,
    }
    defaults.update(weights)
    return TrainingLossConfig(**defaults)


def test_semantic_targets_use_reference_priority_and_all_pixels():
    flags = torch.tensor(
        [[[[int(RayFlags.ROAD_SEMANTIC | RayFlags.SKY_SEMANTIC | RayFlags.VEHICLE_SEMANTIC | RayFlags.EGO_SEMANTIC | RayFlags.INVALID)]]]],
        dtype=torch.int32,
    )
    labels = CameraFrameLabels(flags=flags)
    logits = torch.tensor([[[[0.0, 8.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    config = _render_only_config(primitive_semantics=1.0)

    values, skipped = TrainingLosses(config)._context_losses(
        SupervisionPack(context_semantic_logits=logits),
        _camera_batch(labels),
    )

    expected = torch.nn.functional.cross_entropy(logits.moveaxis(-1, 1), torch.ones((1, 1, 1), dtype=torch.long))
    torch.testing.assert_close(values["primitive_semantics"], expected)
    assert not skipped


def test_render_rgb_compacts_valid_pixels_before_semantic_weighting(monkeypatch):
    flags = torch.tensor(
        [
            [
                [
                    [int(RayFlags.RGB_LABEL)],
                    [int(RayFlags.RGB_LABEL | RayFlags.SYNTHETIC)],
                    [int(RayFlags.RGB_LABEL | RayFlags.INVALID)],
                    [0],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    predicted = torch.tensor(
        [[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]]],
        requires_grad=True,
    )
    labels = CameraFrameLabels(rgb=torch.zeros_like(predicted), flags=flags)
    output = RenderOutput(
        rgb=predicted,
        opacity=torch.zeros((1, 1, 4, 1)),
        distance=torch.ones((1, 1, 4, 1)),
        sky_rgb=torch.zeros((1, 1, 4, 3)),
    )
    calls = []
    original_mse_loss = losses_module.F.mse_loss

    def tracking_mse_loss(actual, target, *, reduction):
        calls.append((actual.detach().clone(), target.detach().clone(), reduction))
        return original_mse_loss(actual, target, reduction=reduction)

    monkeypatch.setattr(losses_module.F, "mse_loss", tracking_mse_loss)

    values, skipped = TrainingLosses(_render_only_config(rgb=1.0))._render_losses(output, _camera_batch(labels))

    torch.testing.assert_close(values["rgb"], torch.tensor(1.0))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], predicted.detach()[0, 0, :2])
    torch.testing.assert_close(calls[0][1], torch.zeros((2, 3)))
    assert calls[0][2] == "none"
    values["rgb"].backward()
    assert predicted.grad is not None
    torch.testing.assert_close(predicted.grad[0, 0, 0], torch.full((3,), 1.0 / 3.0))
    torch.testing.assert_close(predicted.grad[0, 0, 1], torch.full((3,), 1.0 / 6.0))
    torch.testing.assert_close(predicted.grad[0, 0, 2:], torch.zeros((2, 3)))
    assert not skipped


def test_render_rgb_empty_valid_set_preserves_nan_and_zero_gradient():
    predicted = torch.ones((1, 1, 2, 3), requires_grad=True)
    labels = CameraFrameLabels(
        rgb=torch.zeros_like(predicted),
        flags=torch.zeros((1, 1, 2, 1), dtype=torch.int32),
    )
    output = RenderOutput(
        rgb=predicted,
        opacity=torch.zeros((1, 1, 2, 1)),
        distance=torch.ones((1, 1, 2, 1)),
        sky_rgb=torch.zeros((1, 1, 2, 3)),
    )

    values, skipped = TrainingLosses(_render_only_config(rgb=1.0))._render_losses(output, _camera_batch(labels))

    assert torch.isnan(values["rgb"])
    values["rgb"].backward()
    assert predicted.grad is not None
    torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))
    assert not skipped


def test_render_psnr_uses_only_valid_rgb_labeled_rays():
    flags = torch.tensor(
        [
            [
                [
                    [int(RayFlags.RGB_LABEL)],
                    [int(RayFlags.RGB_LABEL | RayFlags.INVALID)],
                    [0],
                    [int(RayFlags.RGB_LABEL | RayFlags.SYNTHETIC)],
                    [int(RayFlags.RGB_LABEL | RayFlags.HARMONIZED)],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    labels = CameraFrameLabels(rgb=torch.zeros((1, 1, 5, 3)), flags=flags)
    output = RenderOutput(
        rgb=torch.tensor([[[[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.75, 0.75, 0.75], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]]),
        opacity=torch.zeros((1, 1, 5, 1)),
        distance=torch.zeros((1, 1, 5, 1)),
        sky_rgb=torch.zeros((1, 1, 5, 3)),
    )

    predicted, target = TrainingSystem._render_psnr_inputs(output, _camera_batch(labels))

    torch.testing.assert_close(predicted, torch.full((3, 3), 0.1))
    torch.testing.assert_close(target, torch.zeros((3, 3)))
    torch.testing.assert_close(PeakSignalNoiseRatio(data_range=1)(predicted, target), torch.tensor(20.0))


def test_psnr_accumulates_pixels_globally_and_context_uses_placeholder():
    logged = []
    system = SimpleNamespace(
        train_psnr=PeakSignalNoiseRatio(data_range=1),
        validation_psnr=PeakSignalNoiseRatio(data_range=1),
        current_epoch=3,
        log=lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    predicted = [torch.ones((1, 3)), torch.full((9, 3), 0.1)]
    target = [torch.zeros_like(value) for value in predicted]

    TrainingSystem._log_psnr(system, predicted, target, "val")

    metric = logged[0][0][1]
    torch.testing.assert_close(
        metric.compute(), torch.tensor(-10.0 * np.log10(0.109), dtype=torch.float32), rtol=1e-5, atol=1e-5
    )
    assert logged[0][0][0] == "val/psnr"

    logged.clear()
    TrainingSystem._log_psnr(system, [], [], "val")
    assert logged == [(("val/psnr", pytest.approx(0.3)), {"prog_bar": True})]


def test_render_distance_matches_harmonized_and_synthetic_masks():
    flags = torch.tensor(
        [
            [
                [
                    [0],
                    [int(RayFlags.SYNTHETIC)],
                    [int(RayFlags.HARMONIZED)],
                    [int(RayFlags.INVALID)],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    labels = CameraFrameLabels(metric_distance=torch.full((1, 1, 4, 1), 2.0), flags=flags)
    output = RenderOutput(
        rgb=torch.zeros((1, 1, 4, 3)),
        opacity=torch.ones((1, 1, 4, 1)),
        distance=torch.ones((1, 1, 4, 1)),
        sky_rgb=torch.zeros((1, 1, 4, 3)),
    )

    values, skipped = TrainingLosses(_render_only_config(distance=1.0))._render_losses(
        output,
        _camera_batch(labels),
    )

    # The synthetic ray is zero weighted but remains in the two-ray denominator.
    torch.testing.assert_close(values["distance"], torch.tensor(0.125))
    assert not skipped


def test_pinhole_camera_kwargs_match_reference_batch_axes(monkeypatch):
    class FakeRollingShutterType(Enum):
        GLOBAL = 0

    fake_rendering = ModuleType("gsplat.rendering")
    setattr(fake_rendering, "RollingShutterType", FakeRollingShutterType)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", fake_rendering)
    monkeypatch.setattr(renderer_module, "_configure_gsplat_build", lambda: None)
    monkeypatch.setattr(renderer_module, "_external_distortion_gsplat_parameters", lambda parameters: None)
    parameters = OpenCVPinholeCameraModelParameters(
        resolution=np.array([2, 2], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([1.0, 1.0], dtype=np.float32),
        focal_length=np.array([2.0, 3.0], dtype=np.float32),
        radial_coeffs=np.arange(6, dtype=np.float32),
        tangential_coeffs=np.arange(2, dtype=np.float32),
        thin_prism_coeffs=np.ones(4, dtype=np.float32),
    )

    K, kwargs = renderer_module._camera_kwargs(parameters, torch.device("cpu"))

    torch.testing.assert_close(K, torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]]))
    assert kwargs["radial_coeffs"].shape == (1, 1, 6)
    assert kwargs["tangential_coeffs"].shape == (1, 1, 2)
    assert "thin_prism_coeffs" not in kwargs


def test_renderer_checkpoints_full_frame_and_matches_reference_postprocess_order(monkeypatch):
    captured = {}
    checkpoint_calls = []
    inverse_calls = []

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        rendered = torch.zeros((1, 1, 2, 2, 4))
        rendered[..., :3] = -0.5
        rendered[..., 3] = 0.75
        opacity = torch.full((1, 1, 2, 2, 1), 0.5)
        return rendered, opacity, {}

    def fake_checkpoint(function, *args, use_reentrant):
        checkpoint_calls.append((function, args[-1], use_reentrant))
        return function(*args)

    def fake_inverse(matrix, *, unbatch):
        inverse_calls.append((matrix.clone(), unbatch))
        return torch.eye(4, dtype=matrix.dtype)

    monkeypatch.setattr(renderer_module, "require_gsplat", lambda: fake_rasterization)
    monkeypatch.setattr(renderer_module, "checkpoint", fake_checkpoint)
    monkeypatch.setattr(renderer_module, "se3_matrix_inverse", fake_inverse)
    monkeypatch.setattr(renderer_module, "_reference_renderer_config", lambda: object())
    monkeypatch.setattr(renderer_module, "_reference_ut_parameters", lambda: object())
    monkeypatch.setattr(
        renderer_module,
        "sample_sky_cubemap",
        lambda cubemap, directions: torch.full_like(directions, 0.5),
    )
    monkeypatch.setattr(
        renderer_module,
        "_camera_kwargs",
        lambda parameters, device: (
            torch.eye(3, device=device),
            {
                "camera_model": "pinhole",
                "rolling_shutter": "global",
                "_global_shutter": "global",
            },
        ),
    )
    parameters = OpenCVPinholeCameraModelParameters(
        resolution=np.array([2, 2], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([1.0, 1.0], dtype=np.float32),
        focal_length=np.array([1.0, 1.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    directions = torch.zeros((1, 2, 2, 3))
    directions[..., 2] = 1.0
    rays = torch.cat([torch.zeros_like(directions), directions], dim=-1)
    tquat = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]).expand(1, 2, 7).clone()
    timestamps = torch.tensor([[0, 1]], dtype=torch.int64)
    labels = CameraFrameLabels(
        rgb=torch.zeros((1, 2, 2, 3)),
        flags=torch.full((1, 2, 2, 1), int(RayFlags.RGB_LABEL), dtype=torch.int32),
    )
    supervision = DataAndRenderingBatch(
        data=DataBatch(
            camera=DataBatch.Camera(
                meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)],
                labels=labels,
            )
        ),
        rendering=RenderingBatch(
            camera=RenderingData(
                rays=rays,
                sensor_model_parameters=[parameters],
                poses_tquat_startend=tquat,
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps,
            )
        ),
    )
    primitive = KelvinInstantNuRecPrimitive(
        static_layer=KelvinStaticLayer(
            positions=torch.zeros((1, 3)),
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            scales=torch.ones((1, 3)),
            densities=torch.ones((1, 1)),
            rgb=torch.zeros((1, 3)),
        ),
        dynamic_layers=[],
        sky_cubemap=torch.zeros((6, 2, 2, 3)),
        affine_matrix=torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]]),
    )

    output = render_supervision(primitive, supervision)

    assert checkpoint_calls == [(renderer_module._render_supervision_frame, 0, False)]
    assert len(inverse_calls) == 2
    assert captured["render_mode"] == "RGB-d"
    assert captured["means"].shape == (1, 1, 3)
    assert captured["viewmats"].shape == (1, 1, 4, 4)
    assert captured["Ks"].shape == (1, 1, 3, 3)
    assert captured["rays"].shape == (2, 2, 6)
    assert captured["radius_clip"] == 0.0
    assert captured["eps2d"] == pytest.approx(0.3)
    assert captured["sparse_grad"] is False
    assert captured["absgrad"] is False
    # clamp(-0.5) + (1 - 0.5) * sky(0.5) = 0.25, then affine scale 2 = 0.5.
    # Clamping only after composition would instead produce zero.
    torch.testing.assert_close(output.rgb, torch.full((1, 2, 2, 3), 0.5))
    torch.testing.assert_close(output.distance, torch.full((1, 2, 2, 1), 0.75))


def test_background_does_not_require_valid_semantic_flag():
    flags = torch.tensor([[[[int(RayFlags.SKY_SEMANTIC)], [0]]]], dtype=torch.int32)
    labels = CameraFrameLabels(flags=flags)
    output = RenderOutput(
        rgb=torch.zeros((1, 1, 2, 3)),
        opacity=torch.tensor([[[[1.0], [0.0]]]]),
        distance=torch.zeros((1, 1, 2, 1)),
        sky_rgb=torch.zeros((1, 1, 2, 3)),
    )

    values, skipped = TrainingLosses(_render_only_config(background=1.0))._render_losses(
        output,
        _camera_batch(labels),
    )

    torch.testing.assert_close(values["background"], torch.tensor(1.0))
    assert not skipped


def test_sky_cubemap_smooth_region_matches_reference_and_backpropagates():
    predicted = torch.zeros((6, 31, 31, 3), requires_grad=True)
    with torch.no_grad():
        predicted[0, 15, 15] = 1.0
    pack = SupervisionPack(
        predicted_sky_cubemap=predicted,
        reference_sky_cubemap=torch.zeros_like(predicted),
        reference_sky_cubemap_mask=torch.zeros((6, 31, 31, 1), dtype=torch.bool),
    )
    labels = CameraFrameLabels(flags=torch.zeros((1, 1, 1, 1), dtype=torch.int32))
    config = _render_only_config(primitive_sky_cubemap=1.0)

    values, skipped = TrainingLosses(config)._context_losses(pack, _camera_batch(labels))

    expected_target = gaussian_blur(predicted.permute(0, 3, 1, 2), kernel_size=[29, 29]).permute(0, 2, 3, 1)
    torch.testing.assert_close(values["primitive_sky_cubemap"], (predicted - expected_target).abs().mean())
    assert not skipped
    values["primitive_sky_cubemap"].backward()
    assert predicted.grad is not None
