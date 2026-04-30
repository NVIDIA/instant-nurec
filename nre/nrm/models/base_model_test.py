# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for models supporting the BaseNRM API."""

from typing import cast

import lietorch as lt
import numpy as np
import pytest
import torch

from omegaconf import DictConfig

from nre.config.model import GSplatRendererConfig
from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.models import (
    CelsiusModelConfig,
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinSkySolidColorConfig,
    KelvinTokenGSDecoderConfig,
    KelvinTokenGSEncoderConfig,
    _CelsiusModelAffineModuleConfig,
    _CelsiusModelEncoderConfig,
    _CelsiusModelMotionModuleConfig,
    _CelsiusModelSkyModuleConfig,
)
from nre.nrm.models.base import BaseNRM
from nre.nrm.models.celsius_model import CelsiusNRM
from nre.nrm.models.kelvin_backbone.base import KelvinNRMSupervisionPack
from nre.nrm.models.kelvin_model import KelvinNRM
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.primitives.celsius_primitive import CelsiusNRMPrimitive
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive
from nre.utils.batch import (
    CameraFrameLabels,
    ConcreteSensorModelParametersUnion,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    RenderingBatch,
    RenderingData,
)
from nre.utils.types import CuboidTracksData, TracksData


def _get_primitive_positions(primitive: BaseNRMPrimitive) -> torch.Tensor:
    """Positions from primitive (static_layer for Kelvin, else direct)."""
    if isinstance(primitive, KelvinNRMPrimitive):
        return primitive.static_layer.positions
    elif isinstance(primitive, CelsiusNRMPrimitive):
        return primitive.positions
    else:
        raise ValueError(f"Unsupported primitive type: {type(primitive)}")


def _get_primitive_densities(primitive: BaseNRMPrimitive) -> torch.Tensor:
    """Densities from primitive (static_layer for Kelvin, else direct)."""
    if isinstance(primitive, KelvinNRMPrimitive):
        return primitive.static_layer.densities
    elif isinstance(primitive, CelsiusNRMPrimitive):
        return primitive.densities
    else:
        raise ValueError(f"Unsupported primitive type: {type(primitive)}")


@pytest.fixture
def device() -> torch.device:
    """Device for running tests."""
    return torch.device("cuda")


def create_minimal_celsius_config() -> CelsiusModelConfig:
    """Create a minimal Celsius model configuration for testing."""
    return CelsiusModelConfig(
        name="celsius",
        renderer=GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",
                "rasterize_mode": "classic",
                "log_level": "info",
            }
        ),
        track_padding_m=[0.1, 0.1, 0.1],
        init_token_scale=0.02,
        patch_shape=(8, 8),
        encoder=_CelsiusModelEncoderConfig(
            depth=2,  # Small depth for testing
            embed_dim=128,
            n_heads=4,
            mlp_ratio=2.0,
            block_pattern="T",
        ),
        sky_module=_CelsiusModelSkyModuleConfig(enabled=False),
        motion_module=_CelsiusModelMotionModuleConfig(
            enabled=False,
            n_motion_tokens=0,
            motion_qkv_dim=128,
            falloff=False,
        ),
        affine_module=_CelsiusModelAffineModuleConfig(
            enabled=False,
            n_affine_tokens=-1,
        ),
        use_deferred_bp=False,
        activation_checkpointing=False,
        centroid_prediction="distance",
    )


def _kelvin_common_config() -> dict:
    """Common Kelvin config fields (renderer, sky, patch_shape, etc.)."""
    return {
        "name": "kelvin",
        "renderer": GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",
                "rasterize_mode": "classic",
                "log_level": "info",
            }
        ),
        "prepare_normal_supervision": False,
        "use_2dgs": False,
        "patch_shape": (8, 8),
        "sky": KelvinSkySolidColorConfig(name="solid-color", color=(0.5, 0.5, 0.5), cubemap_size=16),
    }


def create_minimal_kelvin_tokengs_config() -> KelvinModelConfig:
    """Create a minimal Kelvin model configuration with TokenGS encoder/decoder."""
    return KelvinModelConfig(
        **_kelvin_common_config(),
        encoder=KelvinTokenGSEncoderConfig(name="token-gs-encoder", depth=2, n_heads=4, embed_dim=64),
        decoder=KelvinTokenGSDecoderConfig(name="token-gs-decoder", depth=2),
    )


def create_minimal_kelvin_dav3_config() -> KelvinModelConfig:
    """Create a minimal Kelvin model configuration with DAv3 encoder and DPT decoder."""
    take_block_indices = [0, 1]
    return KelvinModelConfig(
        **_kelvin_common_config(),
        encoder=KelvinDAv3EncoderConfig(
            name="dav3-encoder",
            depth=2,
            n_heads=4,
            embed_dim=64,
            take_block_indices=take_block_indices,
            aa_start_block_idx=0,
            ffn_type="mlp",
        ),
        decoder=KelvinDPTDecoderConfig(
            name="dpt-decoder",
            dpt_dim=32,
            dpt_reassemble_hidden_dims=[32, 32],  # length must match len(take_block_indices)
        ),
    )


def create_minimal_rendering_batch(
    device: torch.device,
    batch_size: int = 1,
    height: int = 64,
    width: int = 64,
) -> RenderingBatch:
    """Create a minimal RenderingBatch for testing."""
    # Create simple camera rays (origin + direction)
    rays_o = torch.tensor([0.0, 0.0, 0.0], device=device).reshape(1, 1, 1, 3)
    rays_o = rays_o.expand(batch_size, height, width, 3)

    # Create rays pointing in different directions based on pixel position
    rays_d = torch.zeros(batch_size, height, width, 3, device=device)
    rays_d[..., 0] = torch.linspace(-0.5, 0.5, width, device=device).reshape(1, 1, width)
    rays_d[..., 1] = torch.linspace(-0.5, 0.5, height, device=device).reshape(1, height, 1)
    rays_d[..., 2] = 1.0
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays = torch.cat([rays_o, rays_d], dim=-1)

    # Create timestamps
    timestamps_startend_us = torch.tensor(
        [[i * 100000, i * 100000 + 50000] for i in range(batch_size)],
        dtype=torch.int64,
        device=device,
    )
    rays_timestamps_us = (
        timestamps_startend_us[:, 0:1].reshape(batch_size, 1, 1, 1).expand(batch_size, height, width, 1)
    )

    # Create poses (identity)
    poses_tquat_startend = torch.zeros(batch_size, 2, 7, device=device)
    poses_tquat_startend[..., 6] = 1.0  # qw = 1 for identity quaternion

    # Create sensor model parameters
    from ncore.data import FThetaCameraModelParameters, ShutterType

    sensor_model_parameters = [
        FThetaCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([width / 2.0, height / 2.0], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=1.7,
            linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        for _ in range(batch_size)
    ]

    # Compute ray footprints (needed by DAv3 decoder for pixel scale).
    # Footprint ≈ angular extent of a pixel ≈ 1/focal_length.  Use a simple uniform value here.
    rays_footprints = torch.ones(batch_size, height, width, 1, device=device) * (1.0 / height)

    return RenderingBatch(
        camera=RenderingData(
            rays=rays,
            sensor_model_parameters=cast(list[ConcreteSensorModelParametersUnion], sensor_model_parameters),
            poses_tquat_startend=poses_tquat_startend,
            timestamps_startend_us=timestamps_startend_us,
            rays_timestamps_us=rays_timestamps_us.contiguous(),
            timestamps_startend_us_cpu=timestamps_startend_us.cpu(),
            _rays_footprints=rays_footprints,
        )
    )


def create_minimal_data_batch(
    device: torch.device,
    batch_size: int = 1,
    height: int = 64,
    width: int = 64,
    sequence_id: str = "test_sequence",
) -> DataBatch:
    """Create a minimal DataBatch for testing."""
    # Create RGB images
    rgb = torch.rand(batch_size, height, width, 3, device=device)

    # Create camera metadata
    camera_meta = []
    for i in range(batch_size):
        meta = FrameMeta(
            unique_sensor_idx=i % 3,  # Simulate multiple cameras
            unique_frame_idx=i,
        )
        camera_meta.append(meta)

    # Create labels (metric_distance needed for Kelvin prepare_supervision when normals are computed)
    flags = torch.zeros(batch_size, height, width, 1, dtype=torch.int32, device=device)
    metric_distance = torch.ones(batch_size, height, width, 1, device=device)

    camera_labels = CameraFrameLabels(
        rgb=rgb,
        flags=flags,
        metric_distance=metric_distance,
        velocity=None,
    )

    return DataBatch(
        idx=0,
        sequence_id=[sequence_id] * batch_size,
        worker_id=[0] * batch_size,
        camera=DataBatch.Camera(
            meta=camera_meta,
            labels=camera_labels,
        ),
    )


def create_minimal_cuboid_tracks(device: torch.device) -> CuboidTracks:
    """Create minimal cuboid tracks for testing."""
    # Create a simple static track
    tracks_id = ["track_0"]
    n_poses = 2

    # Create identity poses
    tracks_poses_tquat = torch.zeros(n_poses, 7, dtype=torch.float32, device=device)
    tracks_poses_tquat[:, 0] = torch.tensor([1.0, 2.0], device=device)  # x translation
    tracks_poses_tquat[:, 6] = 1.0  # qw = 1 for identity quaternion

    tracks_timestamps_us = torch.tensor([0, 1000000], dtype=torch.int64, device=device)
    tracks_packinfo = torch.tensor([[0, n_poses]], dtype=torch.int32, device=device)  # int32 required by CUDA kernel
    tracks_poses = lt.SE3(tracks_poses_tquat)

    cuboids_dims = torch.ones(1, 3, dtype=torch.float32, device=device)

    tracks_data = TracksData(
        tracks_id=tracks_id,
        tracks_poses=tracks_poses,
        tracks_timestamps_us=tracks_timestamps_us,
        tracks_packinfo=tracks_packinfo,
        max_track_n_poses=n_poses,
        tracks_label_class=["automobile"],
        tracks_flags=torch.zeros(1, dtype=torch.int32, device=device),
    )

    cuboidtracks_data = CuboidTracksData(cuboids_dims=cuboids_dims)

    return CuboidTracks(
        tracks_data=tracks_data,
        cuboidtracks_data=cuboidtracks_data,
    )


@pytest.fixture
def minimal_data_and_rendering_batch(device: torch.device) -> DataAndRenderingBatch:
    """Create a minimal DataAndRenderingBatch for testing."""
    data_batch = create_minimal_data_batch(device)
    rendering_batch = create_minimal_rendering_batch(device)
    return DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)


@pytest.fixture
def minimal_cuboid_tracks(device: torch.device) -> CuboidTracks:
    """Create minimal cuboid tracks for testing."""
    return create_minimal_cuboid_tracks(device)


@pytest.fixture
def celsius_model(device: torch.device) -> BaseNRM:
    """Create a CelsiusNRM instance for testing."""
    config = create_minimal_celsius_config()
    model = CelsiusNRM(config).to(device)
    model.eval()  # Set to eval mode for deterministic behavior
    return model


@pytest.fixture
def kelvin_tokengs_model(device: torch.device) -> BaseNRM:
    """Create a KelvinNRM instance with TokenGS encoder/decoder for testing."""
    config = create_minimal_kelvin_tokengs_config()
    model = KelvinNRM(config).to(device)
    model.eval()  # Set to eval mode for deterministic behavior
    return model


@pytest.fixture
def kelvin_dav3_model(device: torch.device) -> BaseNRM:
    """Create a KelvinNRM instance with DAv3 encoder and DPT decoder for testing."""
    # Seed so the semantic head's NaN-sentinel zero_init path is deterministic;
    # otherwise argmax(semantic_logits) varies run-to-run and may filter all gaussians.
    torch.manual_seed(0)
    config = create_minimal_kelvin_dav3_config()
    model = KelvinNRM(config).to(device)
    model.eval()  # Set to eval mode for deterministic behavior
    return model


# Parameterize tests by model type (Celsius + two Kelvin variants)
@pytest.fixture(
    params=["celsius_model", "kelvin_tokengs_model", "kelvin_dav3_model"],
    ids=["celsius", "kelvin_tokengs", "kelvin_dav3"],
)
def model(request) -> BaseNRM:
    """Parametrized fixture for all model types."""
    return request.getfixturevalue(request.param)


class TestBaseNRMAPI:
    """Test suite for the BaseNRM API."""

    def test_model_initialization(self, model: BaseNRM) -> None:
        """Test that model can be initialized properly."""
        assert isinstance(model, BaseNRM)
        assert model.config is not None

        # Check that model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model should have parameters"

    def test_update_step_train_batch_start(self, model: BaseNRM) -> None:
        """Test update_step_train_batch_start hook."""
        # Call with different epoch and step values
        result = model.update_step_train_batch_start(epoch=0, global_step=0, system=None)
        assert isinstance(result, dict)

        result = model.update_step_train_batch_start(epoch=5, global_step=100, system=None)
        assert isinstance(result, dict)

    def test_on_train_from_scratch_start(self, model: BaseNRM) -> None:
        """Test on_train_from_scratch_start hook."""
        # Should not raise any exceptions
        model.on_train_from_scratch_start(system=None)

        # Verify model is still functional after initialization
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_serialize_to_json_dict(self, model: BaseNRM) -> None:
        """Test serialize_to_json_dict method."""
        # Test with state dict
        result = model.serialize_to_json_dict(with_state_dict=True)
        assert isinstance(result, dict)

        # Test without state dict
        result = model.serialize_to_json_dict(with_state_dict=False)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "compute_supervision_pack, use_tracks, expect_supervision",
        [
            # basic reconstruction, no supervision, no tracks
            (False, False, False),
            # supervision pack requested, no tracks
            (True, False, True),
            # tracks given, no supervision pack
            (False, True, False),
            # tracks given, supervision pack requested
            (True, True, True),
        ],
    )
    def test_reconstruct_variants(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
        minimal_cuboid_tracks: CuboidTracks,
        compute_supervision_pack: bool,
        use_tracks: bool,
        expect_supervision: bool,
    ) -> None:
        """Test reconstruction with various combinations of parameters."""
        context = [minimal_data_and_rendering_batch]
        cuboid_tracks = [minimal_cuboid_tracks] if use_tracks else None

        with torch.no_grad():
            primitives, supervision_packs = model.reconstruct(
                context=context,
                cuboid_tracks=cuboid_tracks,
                media_logger=None,
                compute_supervision_pack=compute_supervision_pack,
            )

        # structure checks
        assert isinstance(primitives, list)
        assert len(primitives) == len(context)
        primitive = primitives[0]
        # Kelvin is layer-based: use static_layer; Celsius has positions/densities/rgb directly
        if isinstance(primitive, KelvinNRMPrimitive):
            assert primitive.static_layer is not None
            assert primitive.static_layer.positions.shape[0] > 0
            assert primitive.static_layer.densities.shape[0] > 0
            assert primitive.static_layer.rgb.shape[0] > 0
        else:
            assert hasattr(primitive, "positions")
            assert hasattr(primitive, "densities")
            assert hasattr(primitive, "rgb")

        if expect_supervision:
            assert supervision_packs is not None
            assert isinstance(supervision_packs, list)
            assert len(supervision_packs) == len(context)
        else:
            assert supervision_packs is None

    def test_reconstruct_multiple_views(
        self,
        model: BaseNRM,
        device: torch.device,
    ) -> None:
        """Test reconstruction with multiple views."""
        # Create batch with multiple views
        batch_size = 3
        data_batch = create_minimal_data_batch(device, batch_size=batch_size)
        rendering_batch = create_minimal_rendering_batch(device, batch_size=batch_size)
        batch = DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)

        context = [batch]

        with torch.no_grad():
            primitives, _ = model.reconstruct(
                context=context,
                cuboid_tracks=None,
                media_logger=None,
                compute_supervision_pack=False,
            )

        assert len(primitives) == 1
        # Verify primitive contains data from multiple views
        assert _get_primitive_positions(primitives[0]).shape[0] > 0

    def test_prepare_supervision_without_tracks(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
    ) -> None:
        """Test prepare_supervision without cuboid tracks."""
        context = [minimal_data_and_rendering_batch]
        supervision = [minimal_data_and_rendering_batch]
        with torch.no_grad():
            _, supervision_packs = model.reconstruct(
                context=context,
                cuboid_tracks=None,
                media_logger=None,
                compute_supervision_pack=True,
            )
        assert supervision_packs is not None
        prepared_supervision, _ = model.prepare_supervision(
            context=context,
            supervision=supervision,
            cuboid_tracks=None,
            supervision_packs=supervision_packs,
            media_logger=None,
        )

        assert isinstance(prepared_supervision, list)
        assert len(prepared_supervision) == len(supervision)

    def test_prepare_supervision_with_tracks(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
        minimal_cuboid_tracks: CuboidTracks,
    ) -> None:
        """Test prepare_supervision with cuboid tracks."""
        context = [minimal_data_and_rendering_batch]
        supervision = [minimal_data_and_rendering_batch]
        cuboid_tracks = [minimal_cuboid_tracks]
        with torch.no_grad():
            _, supervision_packs = model.reconstruct(
                context=context,
                cuboid_tracks=cuboid_tracks,
                media_logger=None,
                compute_supervision_pack=True,
            )
        assert supervision_packs is not None
        prepared_supervision, _ = model.prepare_supervision(
            context=context,
            supervision=supervision,
            cuboid_tracks=cuboid_tracks,
            supervision_packs=supervision_packs,
            media_logger=None,
        )

        assert isinstance(prepared_supervision, list)
        assert len(prepared_supervision) == len(supervision)

        # Verify the output is a valid DataAndRenderingBatch
        for sup_batch in prepared_supervision:
            assert isinstance(sup_batch, DataAndRenderingBatch)
            assert sup_batch.data.camera is not None

    def test_prepare_context_without_tracks(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
    ) -> None:
        """Test prepare_context without cuboid tracks."""
        context = [minimal_data_and_rendering_batch]

        prepared_context = model.prepare_context(
            context=context,
            cuboid_tracks=None,
        )

        assert isinstance(prepared_context, list)
        assert len(prepared_context) == len(context)

    def test_prepare_context_with_tracks(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
        minimal_cuboid_tracks: CuboidTracks,
    ) -> None:
        """Test prepare_context with cuboid tracks."""
        context = [minimal_data_and_rendering_batch]
        cuboid_tracks = [minimal_cuboid_tracks]

        prepared_context = model.prepare_context(
            context=context,
            cuboid_tracks=cuboid_tracks,
        )

        assert isinstance(prepared_context, list)
        assert len(prepared_context) == len(context)

    def test_forward_backward_pass(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
    ) -> None:
        """Test forward and backward pass through the model."""
        model.train()  # Set to training mode

        context = [minimal_data_and_rendering_batch]

        # Forward pass
        primitives, _ = model.reconstruct(
            context=context,
            cuboid_tracks=None,
            media_logger=None,
            compute_supervision_pack=False,
        )

        # Create a simple loss (layer-based for Kelvin: use static_layer)
        primitive = primitives[0]
        positions = _get_primitive_positions(primitive)
        densities = _get_primitive_densities(primitive)
        loss = positions.mean() + densities.mean()

        # Backward pass
        loss.backward()

        # Check that at least one parameter received a gradient
        grad_param_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradients for parameter {name} should be finite"
                grad_param_count += 1
        assert grad_param_count > 0, "At least one parameter should have received a gradient"

    def test_model_device_consistency(
        self,
        model: BaseNRM,
        device: torch.device,
    ) -> None:
        """Test that model maintains device consistency."""
        # Check model parameters are on correct device (type must match, index can differ)
        for param in model.parameters():
            assert param.device.type == device.type, (
                f"Parameter device type mismatch: {param.device.type} != {device.type}"
            )

    def test_eval_mode_determinism(
        self,
        model: BaseNRM,
        minimal_data_and_rendering_batch: DataAndRenderingBatch,
    ) -> None:
        """Test that eval mode produces deterministic outputs."""
        context = [minimal_data_and_rendering_batch]

        # Run inference twice
        with torch.no_grad():
            primitives1, _ = model.reconstruct(
                context=context,
                cuboid_tracks=None,
                media_logger=None,
                compute_supervision_pack=False,
            )

            primitives2, _ = model.reconstruct(
                context=context,
                cuboid_tracks=None,
                media_logger=None,
                compute_supervision_pack=False,
            )

        # Check outputs are identical (or very close due to numerical precision)
        p1, p2 = primitives1[0], primitives2[0]
        torch.testing.assert_close(_get_primitive_positions(p1), _get_primitive_positions(p2), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(_get_primitive_densities(p1), _get_primitive_densities(p2), rtol=1e-5, atol=1e-6)
