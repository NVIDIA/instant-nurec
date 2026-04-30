# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import shutil
import tarfile
import threading

from pathlib import Path
from typing import Optional

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange

from nre.utils.model_registry import create_model_registry


logger = logging.getLogger(__name__)


def color_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Transfers the color distribution from the target image to the source image
    using the method of Reinhard et al., 2001.

    Parameters:
    - source: tensor of the source image.
    - target: tensor of the target image.

    Returns:
    - Color transferred image as a tensor.
    """
    # Convert images from BGR to L*a*b* color space
    source_lab = kornia.color.rgb_to_lab(source.permute(2, 0, 1)).permute(1, 2, 0)
    target_lab = kornia.color.rgb_to_lab(target.permute(2, 0, 1)).permute(1, 2, 0)

    src_mean = source_lab.view(-1, 1, 3).mean(dim=0, keepdim=True)
    src_std = source_lab.view(-1, 1, 3).std(dim=0, keepdim=True)

    target_mean = target_lab.view(-1, 1, 3).mean(dim=0, keepdim=True)
    target_std = target_lab.view(-1, 1, 3).std(dim=0, keepdim=True)

    # Subtract the mean from the source image
    lab = source_lab - src_mean

    # Scale by the standard deviation ratio
    lab = lab * (target_std / src_std)

    # Add the target mean
    lab += target_mean

    # Clip the values to valid range
    lab = lab.clamp(-128, 127)

    # Convert back to BGR color space
    transferred = kornia.color.lab_to_rgb(lab.permute(2, 0, 1)).permute(1, 2, 0)

    return transferred


class DifixModel(nn.Module):
    def __init__(
        self,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
    ):
        """ """
        super().__init__()
        self.model_url = model_url
        self.model = None
        self.cache_dir = cache_dir
        self.model_filename = model_filename
        self.needs_temporal_dim = False
        self._lock = threading.Lock()

        if (
            not isinstance(model_resolution, tuple)
            or len(model_resolution) != 2
            or not all(isinstance(x, int) for x in model_resolution)
        ):
            raise ValueError(f"{self.__class__.__name__}: model_resolution must be a tuple of two integers")
        self.model_resolution = torch.Size(model_resolution)

    @staticmethod
    def _download_checkpoint(model_url: str, cache_dir: str, model_filename: str) -> str:
        """Download the model using the model_registry module."""
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        cache_dir_path = Path(cache_dir)
        cached_checkpoint_file = os.path.join(cache_dir, model_filename)

        if os.path.exists(cached_checkpoint_file):
            logger.info("Using cached Difix model (%s).", cached_checkpoint_file)
            return cached_checkpoint_file

        # If the checkpoint file does not exist, create the model registry and download the model
        registry = create_model_registry(model_url, cache_dir_path)
        downloaded_file = registry.get_model()

        assert os.path.exists(downloaded_file), "Difix: Model file not found"

        # If the downloaded file is a tarball, extract it
        if downloaded_file.endswith(".tar.gz"):
            # Extract file
            with tarfile.open(downloaded_file, "r:gz") as tar:
                # Get model files from the tarball
                model_files = [member for member in tar.getmembers() if member.name.endswith(".pt")]
                # Assert that there is only one .pt file in the tarball
                if not len(model_files) == 1:
                    raise ValueError(
                        "Difix: Expected exactly one .pt file in the tarball, got %d",
                        len(model_files),
                    )
                # Extract the model file
                logger.debug("Extracting model file from %s to %s", downloaded_file, cache_dir)
                tar.extract(model_files[0], path=cache_dir)

            # Move the model file to the cached_checkpoint_file
            logger.debug("Moving model file to %s", cached_checkpoint_file)
            shutil.move(os.path.join(cache_dir, model_files[0].name), cached_checkpoint_file)

            # Remove the tarball
            os.remove(downloaded_file)
        else:
            # If not a tarball, assume it's already the checkpoint
            # Move the model file to the cached_checkpoint_file
            shutil.move(downloaded_file, cached_checkpoint_file)

        return cached_checkpoint_file

    def _load_pretrained_model(self):
        cached_checkpoint = self._download_checkpoint(self.model_url, self.cache_dir, self.model_filename)
        # load file
        # TODO: Investigate and identify the underlying reason why importing transformer_engine causes CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
        import transformer_engine

        # Ensure TE TorchScript custom ops (tex_ts::*) are registered before loading the JIT model
        logger.info(f"Loading model with TE saved at {cached_checkpoint}, {os.path.exists(cached_checkpoint)}")
        if hasattr(torch.ops, "tex_ts"):
            logger.info(f"has rmsnorm_fwd_inf_ts: {hasattr(torch.ops.tex_ts, 'rmsnorm_fwd_inf_ts')}")

        model = torch.jit.load(cached_checkpoint)
        model = model.cuda()
        # logger.info('------ model', logger.info(model.inlined_graph))
        logger.info(f"model {cached_checkpoint} loaded successfully")
        return model

    def _detect_needs_temporal_dim(self) -> None:
        """Probe the loaded model with a dummy input to determine whether it expects a temporal dimension."""
        assert self.model is not None, "Model must be loaded before calling _detect_needs_temporal_dim"
        dummy = torch.zeros(1, 3, *self.model_resolution, device="cuda")
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            try:
                self.model(dummy)
                self.needs_temporal_dim = False
                logger.info("Fixer model accepts 4-D input (no temporal dim).")
            except RuntimeError:
                self.model(dummy.unsqueeze(2))
                self.needs_temporal_dim = True
                logger.info("Fixer model requires 5-D input (temporal dim detected).")

    @staticmethod
    def _preprocess_input(
        img: torch.Tensor, img_size: torch.Size, model_resolution: torch.Size
    ) -> tuple[torch.Tensor, torch.dtype]:
        """preprocess input tensors to correspond to pre-defined resolution, rescale to [-1,1]"""

        # Check if input is fp16 and convert to bf16 if needed (SANA model does not support fp16)
        img_dtype = img.dtype
        if img_dtype == torch.float16:
            img = img.to(torch.bfloat16)
        img = rearrange(img, "(h w) c -> 1 c h w", h=img_size[0])

        transforms = T.Compose(
            [
                #  T.GaussianBlur(kernel_size=(11, 11), sigma=3.0), # TODO - add blurring to avoid aliasing!!!
                T.Resize(
                    model_resolution,
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.Lambda(lambda x: x * 2 - 1),
            ]
        )
        return transforms(img), img_dtype

    @staticmethod
    def _postprocess_output(img: torch.Tensor, img_size: torch.Size, img_dtype: torch.dtype) -> torch.Tensor:
        if img.dtype == torch.bfloat16 and img_dtype != torch.bfloat16:
            img = img.to(img_dtype)
        if img_size != torch.Size(img.shape[-2:]):
            logger.debug(f"Difix processed image will be resampled to its original resolution: {img_size}.")
        transforms = T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.Lambda(lambda x: (x + 1) / 2),
            ]
        )
        img = rearrange(transforms(img), "1 c h w -> (h w) c")

        return img

    def forward(self, input: torch.Tensor, img_size: torch.Size, use_color_transfer: bool) -> torch.Tensor:
        if self.model is not None:
            preprocessed_input, img_dtype = self._preprocess_input(input, img_size, self.model_resolution)
            if self.needs_temporal_dim:
                preprocessed_input = preprocessed_input.unsqueeze(2)
            # Disable autocast to avoid mixed input types into the model (SANA model does not support fp16)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                result = self.model(preprocessed_input)
            if self.needs_temporal_dim:
                result = result.squeeze(2)
            result = self._postprocess_output(result, img_size, img_dtype)
            if use_color_transfer:
                result = color_transfer(result.view(*img_size, -1), input.view(*img_size, -1)).view(input.shape)
            return result
        else:
            raise ValueError("Difix: failed to find model instance.")

    # TODO __call__ = module_call_type(forward) - currently module_call_type lives in nre.model.base which causes circular dependencies


class DifixModelFactory:
    """Currently assumes a maximum of a single initialized DifixModel"""

    _initialized_model: Optional[DifixModel] = None
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
    ) -> DifixModel:
        if not cls._initialized_model:
            with cls._lock:
                if not cls._initialized_model:
                    model_instance = DifixModel(model_url, cache_dir, model_filename, model_resolution)
                    # Eagerly download checkpoint
                    cached = DifixModel._download_checkpoint(model_url, cache_dir, model_filename)
                    logger.info(f"DifixModelFactory: checkpoint ready at {cached}")
                    # Eagerly load pretrained model
                    model_instance.model = model_instance._load_pretrained_model()
                    logger.info("DifixModelFactory: model loaded successfully")
                    model_instance._detect_needs_temporal_dim()
                    cls._initialized_model = model_instance
        # At this point _initialized_model cannot be None
        assert cls._initialized_model is not None, "DifixModel initialization failed"
        return cls._initialized_model

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._initialized_model = None
