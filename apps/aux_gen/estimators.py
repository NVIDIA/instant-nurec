# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import functools
import math
import weakref

from contextlib import contextmanager
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Protocol

# We are type-ignoring most 3rdparty as these
# wheels are not part of the main toolchains requirement file
import mmseg.apis  # type: ignore
import numpy as np
import point_cloud_utils as pcu
import torch
import torch.nn.functional as F

from depthanythingv2 import dpt as relative_depth_anything  # type: ignore
from depthanythingv2.metric import dpt as metric_depth_anything  # type: ignore
from PIL import Image as PILImage
from python.runfiles import runfiles
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_dilation, median_filter
from torchvision import transforms

import ncore.data

from apps.aux_gen.utils import LidarSpinMesh, _quantile, plot_points_on_image_with_color
from libs.vren.ensemble import ensemble_cuda, ensemble_numba  # type: ignore
from ncore.impl.common.transformations import transform_point_cloud
from ncore.sensors import CameraModel
from nre.utils.geometry import se3_matrix_inverse
from nre.utils.ncore_utils import AuxDataCameraSemSegProvider


RUNFILES = runfiles.Create()


def weak_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def find_network_file_path(filename: str, search_roots: Optional[List[Path]] = None) -> Path:
    # external/ is for standard runfiles

    default_search_roots: list[Path] = []

    for search_path in [
        "mmseg_repo/configs/dinov2_sam",
        "pretrained_models_repo/segmentation",
        "pretrained_models_repo/depth",
        "trt_pretrained_models_repo/segmentation",
        "pretrained_models_repo/auto_egomask",
    ]:
        # Add runfiles-based search-roots
        rpath = RUNFILES.Rlocation(f"{search_path}/{filename}")
        if rpath is not None:
            default_search_roots.append(Path(rpath).parent)

        # Add obfuscated runfiles-based search-roots
        default_search_roots.append(Path("../" + search_path))

    if search_roots is None:
        search_roots = default_search_roots

    for path in search_roots:
        if path.exists():
            file_matches = list(path.rglob(f"**/{filename}"))

            # Take only first
            if len(file_matches) > 0:
                return file_matches[0]

    raise FileNotFoundError(f"Could not find file: {filename} in {default_search_roots}")


def get_gpu_info():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        return name, f"{capability[0]}.{capability[1]}"
    else:
        return None, None


class SegmentationReturn(NamedTuple):
    semantic_seg: Optional[PILImage.Image]
    semantic_seg_logits: Optional[np.ndarray]
    semantic_dinov2_feats: Optional[np.ndarray]
    instance_seg: Optional[dict[str, np.ndarray]]


class SegmentationEstimator(Protocol):
    def predict(self, image: PILImage.Image, **kwargs) -> SegmentationReturn: ...

    def get_instance_metadata(self) -> dict: ...

    def get_semantic_metadata(self) -> dict: ...


@contextmanager
def temporary_torch_load_override():
    """Helper function for applying local override to the torch.load() API"""
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


class Mask2FormerSegmentationEstimator(SegmentationEstimator):
    """
    Save semantic segmentation logits. Use a constant resolution set by segmentor's config.
    """

    def __init__(
        self,
        resolution: list,
        dataset_name: str = "PrivateDataset_20c",
        config: str = "mask2former_dino_nv_redim256_private12k_aug_backboneTTA.py",
        pretrained_checkpoint: str = "mask2former_dinov2_nv_private12k_backboneTTA_76000.pth",
        estimate_logits: bool = False,
        estimate_dinov2_feats: bool = False,
        backbone_feats_dim: int = -1,
        enable_trt: bool = True,
    ) -> None:
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.config = config
        self.method = "mask2former"
        self.estimate_logits = estimate_logits
        self.estimate_dinov2_feats = estimate_dinov2_feats
        self.backbone_feats_dim = backbone_feats_dim

        self.trt_checkpoints = {
            "A6000": "mask2former_fp16_no_split_gemm_dynamic_4k_8.6.A6000.engine",
            "A100": "mask2former_fp16_no_split_gemm_dynamic_4k_8.0.A100.engine",
            "V100": "mask2former_fp16_no_split_gemm_dynamic_4k_7.0.V100.engine",
            "A40": "mask2former_fp16_no_split_gemm_dynamic_4k_8.6.A40.engine",
            "5880": "mask2former_fp16_no_const_ppg_dynamic_4k_8.9.5880.engine",
            "4070": "mask2former_fp16_no_const_ppg_dynamic_4k_8.9.4070.engine",
        }

        config_path = find_network_file_path(self.config)
        checkpoint_path = find_network_file_path(pretrained_checkpoint)

        gpu_name, sm_version = get_gpu_info()

        trt_checkpoint = next(
            (
                self.trt_checkpoints[shortname]
                for shortname in self.trt_checkpoints
                if enable_trt and gpu_name and shortname in gpu_name
            ),
            None,
        )
        trt_checkpoint_path = find_network_file_path(trt_checkpoint) if trt_checkpoint else None

        args = {
            # config path
            "config": str(config_path),
            # ckpt path
            "checkpoint": str(checkpoint_path),
            "device": "cuda",
            "cfg_options": {"model.inference_backbone_feats_dim": backbone_feats_dim},
            "trt_checkpoint": trt_checkpoint_path,
            "resolution": resolution if trt_checkpoint_path else None,
        }

        # Set torch.load(weights_only=False) to allow loading the full
        # checkpoint including HistoryBuffer. Minimise scope of this overload
        # with save/restore of original torch.load() function.
        with temporary_torch_load_override():
            self.predictor = mmseg.apis.init_model(
                args["config"],
                args["checkpoint"],
                device=args["device"],
                cfg_options=args["cfg_options"],
                resolution=args["resolution"],
                trt_checkpoint=args["trt_checkpoint"],
            )

        self.pretrained_checkpoint = args["checkpoint"]
        self.trt_checkpoint = args["trt_checkpoint"]

        # Get the semantic metadata
        self.stuff_classes = self.predictor.dataset_meta["classes"]
        self.stuff_colors = self.predictor.dataset_meta["palette"]

    def post_process_logits(self, predictions):
        logits_tensor = predictions.seg_logits.data

        # Check if tensor is torch tensor and convert if needed
        if not isinstance(logits_tensor, torch.Tensor):
            logits_tensor = torch.tensor(logits_tensor)

        # Ensure it's on the right device and dtype
        if logits_tensor.device != torch.device("cuda") or logits_tensor.dtype != torch.float32:
            logits_tensor = logits_tensor.to(device="cuda", dtype=torch.float32)  # Shape: (num_classes, H, W)

        probs = F.softmax(logits_tensor, dim=0)  # Shape: (num_classes, H, W)

        entropy = -torch.sum(probs * torch.log(probs), dim=0)
        quantile = 0.90  # worst 10% of cases, marked as uncertain

        if entropy.numel() <= 4096**2:
            entropy_threshold = torch.quantile(entropy, quantile)  # Adaptive
        else:
            entropy_threshold = _quantile(entropy, quantile)

        uncertainty_mask = entropy > entropy_threshold

        building_id = self.stuff_classes.index("building")

        # make uncertain objects into a background class like "building"
        pred_labels = predictions.pred_sem_seg.data
        if not isinstance(pred_labels, torch.Tensor):
            pred_labels = torch.tensor(pred_labels).to(device="cuda")

        pred_labels = pred_labels.squeeze(0)
        if pred_labels.device != torch.device("cuda"):
            pred_labels.to(device="cuda")

        pred_labels[uncertainty_mask] = building_id

        # Update the predictions object with modified labels
        predictions.pred_sem_seg.data = pred_labels.unsqueeze(0)

        # Update the logits to match the new semantic map if required
        if self.estimate_logits:
            logits_tensor[:, uncertainty_mask] = 0
            logits_tensor[building_id, uncertainty_mask] = 1
            predictions.seg_logits.data = logits_tensor

        return uncertainty_mask

    def predict(self, image: PILImage.Image, ego_mask: Optional[np.ndarray] = None, **kwargs) -> SegmentationReturn:
        with torch.no_grad():
            # Convert from RGB to BGR, mmseg requires bgr. H, W, C
            image_arr = np.asarray(image)[:, :, ::-1]
            predictions = mmseg.apis.inference_model(self.predictor, image_arr)

            # Post process uncertain logits
            uncertainty_mask = self.post_process_logits(predictions)

            # Convert predictions to numpy for further processing
            predictions = self.convert_tensor_predictions_to_numpy(predictions)

            # Get semantic segmentation
            semantic_seg = predictions.pred_sem_seg.data.squeeze(0)
            if ego_mask is not None and "egocar" in self.stuff_classes:
                ego_class_id = self.stuff_classes.index("egocar")
                semantic_seg[ego_mask] = ego_class_id
                if isinstance(semantic_seg, torch.Tensor):
                    ego_mask_tensor = torch.tensor(
                        ego_mask, dtype=torch.bool, device=predictions.pred_sem_seg.data.device
                    )
                    predictions.pred_sem_seg.data[ego_mask_tensor.unsqueeze(0)] = ego_class_id
                else:
                    predictions.pred_sem_seg.data[np.expand_dims(ego_mask, axis=0)] = ego_class_id
            semantic_seg = PILImage.fromarray(semantic_seg, mode="P")

            # loading a P image removes the color palette, so we need to put it back to prevent losing data on save
            palettedata = np.linspace(0, 255, 256, dtype=np.uint8).tolist()
            semantic_seg.putpalette(palettedata)

            if self.estimate_logits:
                semantic_seg_logits = np.transpose(predictions.seg_logits.data, (1, 2, 0))
                if ego_mask is not None and "egocar" in self.stuff_classes:
                    semantic_seg_logits[ego_mask] = 0
                    semantic_seg_logits[ego_mask][:, ego_class_id] = 1
            else:
                semantic_seg_logits = None

            if self.estimate_dinov2_feats:
                dinov2_feats = np.transpose(predictions.backbone_feats.data, (1, 2, 0))
            else:
                dinov2_feats = None

            if vis_path := kwargs.get("vis_path"):
                vis_path = Path(vis_path)
                mmseg.apis.show_result_pyplot(
                    self.predictor,
                    image_arr,
                    predictions,
                    show=False,
                    title=vis_path.name,
                    save_dir=str(vis_path.parent),
                )

                # Save the uncertainty mask beside the main segmentation map with timestamp
                uncertainty_mask_img = (uncertainty_mask).cpu().numpy().astype(np.uint8) * 255
                uncertainty_image = PILImage.fromarray(uncertainty_mask_img, mode="L")
                # Use the timestamp from vis_path.name and save in the parent directory
                uncertainty_filename = f"{vis_path.name}_uncertainty_mask.png"
                uncertainty_save_path = vis_path.parent / uncertainty_filename
                uncertainty_image.save(uncertainty_save_path)

            return SegmentationReturn(
                semantic_seg=semantic_seg,
                semantic_seg_logits=semantic_seg_logits,
                semantic_dinov2_feats=dinov2_feats,
                # No instance-seg yet
                instance_seg=None,
            )

    def convert_tensor_predictions_to_numpy(self, predictions):
        if isinstance(predictions.pred_sem_seg.data, torch.Tensor):
            predictions.pred_sem_seg.data = predictions.pred_sem_seg.data.to("cpu").numpy().astype(np.uint8)
        if self.estimate_logits:
            if isinstance(predictions.seg_logits.data, torch.Tensor):
                predictions.seg_logits.data = predictions.seg_logits.data.to("cpu").numpy()
        if self.estimate_dinov2_feats:
            if isinstance(predictions.backbone_feats.data, torch.Tensor):
                predictions.backbone_feats.data = predictions.backbone_feats.data.to("cpu").numpy()
        return predictions

    def set_resolution(self, resolution: list) -> None:
        self.resolution = resolution

    def get_semantic_metadata(self) -> dict:
        return {
            "resolution": self.resolution,
            "dataset_name": self.dataset_name,
            "method": self.method,
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "stuff_classes": self.stuff_classes,
            "stuff_colors": self.stuff_colors,
        }

    def get_instance_metadata(self) -> dict:
        # No instance-seg yet
        return {}


class DINOv2Estimator:
    """
    DINOv2 feature estimator for feature extraction.
    Images are padded to the nearest patch size (one could also test direct rescaling outside).
    """

    PATCH_SIZE = (14, 14)

    def __init__(self, resolution: list, backend: str, facet: Literal["token", "key"] = "token") -> None:
        self.backend = backend
        self.facet = facet

        if backend.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", backend)
            self.model.eval().cuda()

        elif backend == "nv_dinov2":
            # TODO [JH]: Figure out a more elegant way to load this model.
            #   Currently we re-use the mask2former backbone since it is frozen during training.
            segm_model = Mask2FormerSegmentationEstimator(resolution, enable_trt=False)
            assert segm_model.predictor.backbone_dino is not None
            self.model = segm_model.predictor.backbone_dino.timm_model  # type: ignore

        else:
            raise ValueError(f"Backend {backend} not supported")

        self.last_extracted_facet = None
        if self.facet == "token":
            self.model.blocks[-1].register_forward_hook(self.extract_token_hook)

        elif self.facet == "key":
            self.model.blocks[-1].attn.register_forward_hook(self.extract_key_hook)

        # Patch size sanity check
        self.patch_size = self.model.patch_embed.patch_size[::-1]
        assert self.patch_size == self.PATCH_SIZE, f"Patch size mismatch: {self.patch_size} vs {self.PATCH_SIZE}"

        self.set_resolution(resolution)

    @classmethod
    def resolution_from_dino_width(cls, original_resolution: list, dino_width: int) -> list:
        dino_height = math.ceil(dino_width * original_resolution[1] / original_resolution[0])
        return [dino_width * cls.PATCH_SIZE[0], dino_height * cls.PATCH_SIZE[1]]

    def set_resolution(self, resolution: list) -> None:
        self.resolution = resolution
        pad_w = self.patch_size[0] - self.resolution[0] % self.patch_size[0]
        pad_h = self.patch_size[1] - self.resolution[1] % self.patch_size[1]
        if pad_w == self.patch_size[0]:
            pad_w = 0
        if pad_h == self.patch_size[1]:
            pad_h = 0

        self.dino_resolution = (
            (self.resolution[0] + pad_w) // self.patch_size[0],
            (self.resolution[1] + pad_h) // self.patch_size[1],
        )
        self.padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution[::-1]),  # W, H -> H, W
                transforms.Pad(self.padding, padding_mode="reflect"),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution[::-1]),
                transforms.Pad(self.padding, padding_mode="constant", fill=0),
            ]
        )

    def extract_key_hook(self, module, input, output):
        input = input[0]
        B, N, C = input.shape
        qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        # 0 is query, 1 is key, 2 is value
        feature = qkv[1]  # B, H, T, C
        # remove cls token, and permute to B, HC, T
        feature = feature[:, :, 1:].moveaxis(2, -1).flatten(1, 2)
        feature = feature.reshape(B, -1, *self.dino_resolution[::-1])
        self.last_extracted_facet = feature

    def extract_token_hook(self, module, input, output):
        B, _, C = output.shape
        feature = output[:, 1:].reshape(B, *self.dino_resolution[::-1], C)
        self.last_extracted_facet = feature.moveaxis(3, 1)

    @torch.no_grad()
    def predict(
        self, image: PILImage.Image, ego_mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image = self.preprocess(image).unsqueeze(0).cuda()

        self.model(image)
        assert self.last_extracted_facet is not None
        facet = self.last_extracted_facet[0].permute(1, 2, 0).cpu().numpy()

        # Ego mask is used to determine which tokens are usable.
        if ego_mask is not None:
            ego_mask = self.mask_preprocess(ego_mask.astype(np.float32))[0]
            assert ego_mask is not None

            # Patchify and compute mean mask, > 1 pixels to be considered containing ego information.
            ego_mask = (
                ego_mask.reshape(
                    self.dino_resolution[1], self.patch_size[1], self.dino_resolution[0], self.patch_size[0]
                ).sum([1, 3])
                > 1.0
            )

            ego_mask = ego_mask.numpy()

        return facet, ego_mask

    def get_metadata(self) -> dict:
        return {
            "resolution": self.resolution,
            "facet": self.facet,
            "backend": self.backend,
            "window_left": self.padding[0] / self.patch_size[0],
            "window_top": self.padding[1] / self.patch_size[1],
            "window_width": self.dino_resolution[0] - (self.padding[0] + self.padding[2]) / self.patch_size[0],
            "window_height": self.dino_resolution[1] - (self.padding[1] + self.padding[3]) / self.patch_size[1],
        }


class LidarSegmentationByProjectEstimator:
    """Lidar semantic segmentation by projecting image segmentation results."""

    def __init__(
        self,
        loader: ncore.data.SequenceLoaderProtocol,
        aux_camera_semseg_provider: AuxDataCameraSemSegProvider,
        camera_ids: list[str],
        ensemble_cuda: bool = True,
        method: str = "camera-sseg-projection-ensemble",
        ignore_label: int = 255,  # -1 for uint8
        min_ray_length: float = 0.1,
    ) -> None:
        self.loader = loader
        self.aux_camera_semseg_provider = aux_camera_semseg_provider
        self.ignore_label = ignore_label
        self.ensemble_cuda = ensemble_cuda
        self.method = method
        assert len(camera_ids), "Require at least a single active camera"
        self.camera_ids = camera_ids
        self.min_ray_length = min_ray_length

    def set_meta(self) -> None:
        self.stuff_colors = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(self.camera_ids[0])[
            "stuff_colors"
        ]
        for camera_id in self.camera_ids[1:]:
            stuff_color = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)["stuff_colors"]
            assert np.array(stuff_color).shape == np.array(self.stuff_colors).shape
            assert np.abs(np.array(stuff_color) - np.array(self.stuff_colors)).sum() == 0, (
                "Stuff colors of all cameras should be consistent."
            )
        self.stuff_colors.append([128, 128, 128])  # for ignore label
        self.stuff_classes = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(self.camera_ids[0])[
            "stuff_classes"
        ]
        for camera_id in self.camera_ids[1:]:
            stuff_class = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
            assert stuff_class == self.stuff_classes, "Stuff calsses of all cameras should be consistent."
        self.stuff_classes.append("ignore")

        # check resolution
        for camera_id in self.camera_ids:
            resolution_semseg_provider = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)[
                "resolution"
            ]
            camera_sensor = self.loader.get_camera_sensor(camera_id)
            camera_model = CameraModel.from_parameters(
                camera_sensor.model_parameters, device="cpu", dtype=torch.float32
            )
            assert torch.all(camera_model.resolution == torch.tensor(resolution_semseg_provider)), (
                "The resolution from CameraModel and AuxDataCameraSemSegProvider do not match."
            )

        # lidar-camera-visibility order
        self.lidar_camera_visibility_order = {camera_id: idx for idx, camera_id in enumerate(self.camera_ids)}

    def predict(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
        frame_timestamp_us: int,
        vis_path: Path | None = None,
    ) -> tuple[PILImage.Image, np.ndarray]:
        lidar_frame_idx = lidar_sensor.get_closest_frame_index(frame_timestamp_us)

        # load motion-compensated point clouds (represented in end-of-frame sensor-frame)
        pc = lidar_sensor.get_frame_point_cloud(lidar_frame_idx, motion_compensation=True, with_start_points=True)
        xyz_s = pc.xyz_m_start  # start
        assert isinstance(xyz_s, np.ndarray)
        xyz_e = pc.xyz_m_end  # end

        ray_length = np.linalg.norm(xyz_e - xyz_s, axis=-1)
        valid_mask = ray_length >= self.min_ray_length
        xyz_s_valid = xyz_s[valid_mask]
        xyz_e_valid = xyz_e[valid_mask]

        # lidar to world
        T_lidar_to_world = lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_idx)
        xyz_e_valid_world = transform_point_cloud(xyz_e_valid, T_lidar_to_world)
        xyz_e_valid_world = torch.from_numpy(xyz_e_valid_world).to("cuda", dtype=torch.float32)
        num_points = len(xyz_e_valid)
        lidar_seg_ensemble = np.full((num_points, len(self.camera_ids)), self.ignore_label)
        visibility_mask = np.full((num_points, len(self.camera_ids)), False)

        # create camera mesh from lidar spin to calculate occluded points
        spin_mesh = LidarSpinMesh(xyz_s_valid, xyz_e_valid)
        spin_mesh_vertices = np.ascontiguousarray(spin_mesh.vertices)
        intersector = pcu.RayMeshIntersector(spin_mesh_vertices, spin_mesh.faces)

        visual_dict = {}
        colormap = np.array(self.stuff_colors)

        ego_class_id = self.stuff_classes.index("egocar") if "egocar" in self.stuff_classes else self.ignore_label
        for camera_id, ci in self.lidar_camera_visibility_order.items():
            camera_sensor = self.loader.get_camera_sensor(camera_id)
            camera_frame_idx = camera_sensor.get_closest_frame_index(frame_timestamp_us)
            frame_timestamp_us_camera = camera_sensor.get_frame_timestamp_us(camera_frame_idx)
            image_seg = np.array(
                self.aux_camera_semseg_provider.get_semantic_segmentation(camera_id, frame_timestamp_us_camera)
            )
            image_seg[image_seg == ego_class_id] = self.ignore_label

            camera_model = CameraModel.from_parameters(
                camera_sensor.model_parameters, device="cuda", dtype=torch.float32
            )
            lidar_seg = np.full(num_points, self.ignore_label)

            T_world_sensor_startend = torch.from_numpy(
                camera_sensor.get_frames_T_source_sensor(
                    source_node="world",
                    frame_indices=camera_frame_idx,
                    frame_timepoint=None,  # both start and end
                )  # 2x4x4
            ).to("cuda", dtype=torch.float32)

            projection = camera_model.world_points_to_image_points_shutter_pose(
                xyz_e_valid_world,
                T_world_sensor_startend[0],
                T_world_sensor_startend[1],
                return_valid_indices=True,
                return_all_projections=False,
                return_T_world_sensors=True,
            )
            assert projection.valid_indices is not None
            assert (
                projection.T_world_sensors is not None
            )  # world-coordinates of the camera-frame at the time of projection

            # find non occluded points
            # cast rays from camera to lidar points in camera frustrum
            T_world_to_cameras = projection.T_world_sensors.cpu().numpy().astype(np.float32)
            T_lidar_to_cameras = T_world_to_cameras @ T_lidar_to_world
            # camera positions in lidar frame
            camera_positions_lidar = np.ascontiguousarray(
                se3_matrix_inverse(T_lidar_to_cameras).numpy()[:, :3, -1]
            )  # interpolated positions of the camera at the instant when the 2D projection of each Lidar point is recorded by the rolling shutter.
            camera_pc_rays = xyz_e_valid[projection.valid_indices.cpu()] - camera_positions_lidar
            face_indices, bary_coords, ray_intersection_distances = intersector.intersect_rays(
                camera_positions_lidar, camera_pc_rays
            )
            if len(face_indices.shape) == 0:
                face_indices = np.expand_dims(face_indices, 0)
                bary_coords = np.expand_dims(bary_coords, 0)
                ray_intersection_distances = np.expand_dims(ray_intersection_distances, 0)

            # initialize non-occlusion mask
            non_occluded_points: np.ndarray = face_indices < 0
            # depth check
            valid_rays = face_indices >= 0  # True only at the rays which intersected the shape
            ray_mesh_intersections = pcu.interpolate_barycentric_coords(
                spin_mesh.faces, face_indices[valid_rays], bary_coords[valid_rays], spin_mesh_vertices
            )
            intersection_tol = 0.001
            pc_camera_dist = np.linalg.norm(
                camera_pc_rays[valid_rays],
                axis=-1,
            )
            intersection_camera_dist = np.linalg.norm(
                camera_positions_lidar[valid_rays] - ray_mesh_intersections, axis=-1
            )
            non_occluded_points[valid_rays] = pc_camera_dist <= intersection_camera_dist + intersection_tol

            ij = np.floor(projection.image_points.cpu().numpy()).astype(np.int32)

            assert projection.valid_indices is not None
            valid_indices = projection.valid_indices.cpu()
            lidar_seg[valid_indices[non_occluded_points]] = image_seg[
                ij[non_occluded_points, 1], ij[non_occluded_points, 0]
            ]
            lidar_seg_ensemble[:, ci] = lidar_seg
            visibility_mask[:, ci][valid_indices] = True
            if vis_path:
                visual_dict[camera_id] = [ij, valid_indices]

        # ensamble labels from all cameras
        if self.ensemble_cuda:
            lidar_seg_ensemble = (
                ensemble_cuda(lidar_seg_ensemble, device=torch.device("cuda"), ignore_label=self.ignore_label)
                .cpu()
                .numpy()
            )
        else:
            lidar_seg_ensemble = ensemble_numba(lidar_seg_ensemble.astype(np.uint8), self.ignore_label)

        lidar_seg_result = np.full((len(xyz_e),), self.ignore_label, dtype=np.uint8)
        lidar_seg_result[valid_mask] = lidar_seg_ensemble

        if vis_path:
            for camera_id in self.camera_ids:
                camera_sensor = self.loader.get_camera_sensor(camera_id)
                camera_frame_idx = camera_sensor.get_closest_frame_index(frame_timestamp_us)
                img = camera_sensor.get_frame_image_array(camera_frame_idx)
                ij, indices = visual_dict[camera_id]
                seg = lidar_seg_result[valid_mask][indices]
                seg[seg == self.ignore_label] = self.stuff_classes.index("ignore")
                color = colormap[seg]
                img_vis = plot_points_on_image_with_color(img.copy(), ij, color)
                save_path = vis_path / camera_id / f"{camera_frame_idx}.png"
                PILImage.fromarray(img_vis).save(save_path)

        # Decode semantic segmentation as Image
        decode_seg = PILImage.fromarray(lidar_seg_result.reshape(-1, 1), mode="P")

        # loading a P image removes the color palette, so we need to put it back to prevent losing data on save
        palettedata = np.linspace(0, 255, 256, dtype=np.uint8).tolist()
        decode_seg.putpalette(palettedata)

        return decode_seg, visibility_mask

    def get_semantic_metadata(self) -> dict:
        return {
            "ignore_label": self.ignore_label,
            "method": self.method,
            "stuff_classes": self.stuff_classes,
            "stuff_colors": self.stuff_colors,
        }

    def get_visibility_metadata(self) -> dict:
        return {"camera_visibility_order": self.lidar_camera_visibility_order}


class DepthAnythingV2Estimator:
    """
    Depth estimatior using DepthAnythingV2 small model
    """

    def __init__(
        self,
        pretrained_checkpoint: str = "depth_anything_v2_vits.pth",
        max_depth_m: float = 1.0,
        input_resolution: int = 1036,
    ) -> None:
        checkpoint_path = find_network_file_path(pretrained_checkpoint)
        self.model = relative_depth_anything.init_model(checkpoint=checkpoint_path)
        self.max_depth_m = max_depth_m
        self.input_resolution = input_resolution

        # Used to store the metadata of the depth estimation
        self.method = "DepthAnythingV2"

    def predict(self, image: np.ndarray) -> np.ndarray:
        depth = self.model.infer_image(image, input_size=self.input_resolution)

        # Normalize the depth into the range [0, self.max_depth_m]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * self.max_depth_m

        return depth


class MetricDepthAnythingV2Estimator:
    """
    Depth estimator using DepthAnythingV2 small model finetuned for metric depth estimation
    """

    def __init__(
        self,
        pretrained_checkpoint: str = "depth_anything_v2_metric_vkitti_vits.pth",
        max_depth_m: float = 150.0,
        input_resolution: int = 1036,
    ) -> None:
        checkpoint_path = find_network_file_path(pretrained_checkpoint)
        self.model = metric_depth_anything.init_model(checkpoint=checkpoint_path, max_depth=max_depth_m)
        self.max_depth_m = max_depth_m
        self.input_resolution = input_resolution

        # Used to store the metadata of the depthestimation
        self.method = "MetricDepthAnythingV2"

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.model.infer_image(image, input_size=self.input_resolution)


class PromptlessSAM2EgoMaskEstimator:
    """
    Promptless SAM2-based ego mask estimator for auxiliary data generation.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l",
        pretrained_checkpoint: str = "sam2_hiera_l_avfinetuned_v1.pt",
        device: str = "cuda",
    ) -> None:
        self.dataset_name = model_cfg
        self.method = "promptless_avsam2"
        self.pretrained_checkpoint = str(find_network_file_path(pretrained_checkpoint))

        # Initialize SAM2 predictor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image predictor
        self.sam2_model = build_sam2(model_cfg, None, device=device)
        self.sam2_model.load_state_dict(torch.load(self.pretrained_checkpoint, weights_only=True))
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def predict(
        self,
        image: PILImage.Image,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform promptless ego-mask prediction using a dummy point prompt.
        """

        # Downsample image (stride-based slice) to minimise check compute
        small_patch_sum = np.asarray(image)[::16, ::16].mean(axis=(0, 1))
        # handle blank / very dark images early, 8 is a heuristic threshold
        if small_patch_sum.mean() < 8:
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        # Set image embeddings
        self.predictor.set_image(image)
        # Dummy point prompt: single background point at top-left
        prompt_coords = np.zeros((1, 1, 2), dtype=np.int32)
        prompt_labels = -1 * np.ones((1, 1), dtype=np.int32)

        # Predict masks and scores with point prompt
        masks, scores, _ = self.predictor.predict(
            point_coords=prompt_coords,
            point_labels=prompt_labels,
            multimask_output=True,
            return_logits=False,
        )

        # Select highest-score mask
        if len(masks) > 0:
            idx = int(np.argmax(scores))
            mask = masks[idx].astype(np.uint8)

            # Apply median filter to smoothen and reduce any noise/holes in the mask
            mask = median_filter(mask, size=15, mode="reflect")

            # Apply binary dilation to expand the mask slightly
            structure = np.ones((11, 11), dtype=np.uint8)
            mask = binary_dilation(mask, structure=structure).astype(np.uint8)
        else:
            raise ValueError("No masks predicted by Estimator")

        return mask
