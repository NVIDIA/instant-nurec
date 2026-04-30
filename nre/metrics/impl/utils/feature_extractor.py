# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generic feature extractor system for metrics computation."""

from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import transformers

from PIL import Image
from torch._prims_common import DeviceLikeType


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(
        self,
        pretrained_path: str,
        cache_dir: str | None = None,
        device: DeviceLikeType | None = None,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            pretrained_path: Path or model id for pretrained weights.
            cache_dir: Directory to cache pretrained models.
            device: Device string ("cuda" or "cpu"); autodetect if None.
        """
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def extract_features_batch(
        self, images: torch.Tensor, return_numpy: bool = False, batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract features from a batch of images or sequence.

        Args:
            images: Tensor of shape (B, C, H, W) or (T, C, H, W) in range [0, 1].
                   B = batch size, T = sequence length.
            return_numpy: Whether to return numpy array instead of tensor.
            batch_size: Optional batch size for processing. If None, process all
                       at once. If specified, process in chunks of this size.

        Returns:
            features: Features tensor or array of shape (B, feature_dim) or
                     (T, feature_dim).
        """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        pass


class SegformerFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using SegFormer model for semantic feature extraction."""

    # List of valid pretrained models to prevent arbitrary model selection
    VALID_MODELS = [
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b3-finetuned-ade-512-512",
        "nvidia/segformer-b4-finetuned-ade-512-512",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    ]

    def __init__(
        self,
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        device: DeviceLikeType | None = None,
    ) -> None:
        """Initialize the SegFormer feature extractor.

        Args:
            pretrained_path: Path or model id for pretrained weights.
            cache_dir: Directory to cache pretrained models.
            device: Device string ("cuda" or "cpu"); autodetect if None.

        Raises:
            ValueError: If pretrained_path is not in the list of valid models.
        """
        # Validate pretrained_path against allowed models
        if pretrained_path not in self.VALID_MODELS:
            raise ValueError(f"Invalid pretrained_path: {pretrained_path}. Must be one of: {self.VALID_MODELS}")
        logging.info("Using pretrained path: %s", pretrained_path)

        super().__init__(pretrained_path, cache_dir, device)
        self._feature_dim = 512  # SegFormer-B2 encoder dimension

        # Ensure cache directory exists if provided
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.processor = transformers.SegformerImageProcessor.from_pretrained(pretrained_path, cache_dir=cache_dir)
        self.model = transformers.SegformerForSemanticSegmentation.from_pretrained(pretrained_path, cache_dir=cache_dir)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Create adaptive pooling for global features
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1)).to(self.device)

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self._feature_dim

    def _convert_tensor_to_pil(self, images: torch.Tensor) -> list[Image.Image]:
        """Convert tensor images to PIL images.

        Args:
            images: Tensor of shape (B, C, H, W) or (T, C, H, W).

        Returns:
            List of PIL images.
        """
        batch_size = images.shape[0]
        pil_images = []

        for i in range(batch_size):
            img_tensor = images[i]
            pil_images.append(torchvision.transforms.functional.to_pil_image(img_tensor.detach().clamp(0, 1).cpu()))

        return pil_images

    def _extract_features_common(
        self, images: torch.Tensor, return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """Common feature extraction logic.

        Args:
            images: Tensor of shape (B, C, H, W) or (T, C, H, W).
            return_numpy: Whether to return numpy array instead of tensor.

        Returns:
            features: Features tensor or array of shape (B, 512) or (T, 512).
        """
        batch_size = images.shape[0]

        with torch.no_grad():
            # Convert tensor images to PIL for processor
            pil_images = self._convert_tensor_to_pil(images)

            # Process with SegFormer processor
            inputs = self.processor(images=pil_images, return_tensors="pt")

            # Move tensors to device
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Get model outputs with hidden states
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract features from the last encoder layer
            encoder_features = outputs.hidden_states[-1]

            # Global average pooling to get (B, 512) or (T, 512)
            pooled_features = self.adaptive_pool(encoder_features)
            pooled_features = pooled_features.view(batch_size, -1)

            # L2 normalize for consistency
            pooled_features = F.normalize(pooled_features, p=2, dim=1)

            if return_numpy:
                return pooled_features.cpu().numpy()
            return pooled_features

    def extract_features_batch(
        self, images: torch.Tensor, return_numpy: bool = False, batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract features from a batch of images or sequence.

        Args:
            images: Tensor of shape (B, C, H, W) or (T, C, H, W) in range [0, 1].
                   B = batch size, T = sequence length.
            return_numpy: Whether to return numpy array instead of tensor.
            batch_size: Optional batch size for processing. If None, process all
                       at once. If specified, process in chunks of this size.

        Returns:
            features: Features tensor or array of shape (B, feature_dim) or
                     (T, feature_dim).
        """
        if batch_size is None or images.shape[0] <= batch_size:
            # Process all at once if batch_size is None or data fits
            return self._extract_features_common(images, return_numpy=return_numpy)

        # Process in batches
        total_samples = images.shape[0]
        all_features: list[torch.Tensor] = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_images = images[start_idx:end_idx]

            batch_features = self._extract_features_common(batch_images, return_numpy=False)
            if isinstance(batch_features, torch.Tensor):
                all_features.append(batch_features)
            else:
                all_features.append(torch.from_numpy(batch_features))

        # Concatenate all batch results
        concatenated_features = torch.cat(all_features, dim=0)

        if return_numpy:
            return concatenated_features.cpu().numpy()
        return concatenated_features


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 for semantic object-level features.

    This extractor is designed for feature extraction with multi-layer feature aggregation.
    """

    VALID_MODELS = [
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov2-giant",
    ]

    def __init__(
        self,
        pretrained_path: str = "facebook/dinov2-base",
        cache_dir: str | None = None,
        device: DeviceLikeType | None = None,
        feature_layers: list[int] | None = None,
    ) -> None:
        """Initialize the DINOv2 feature extractor.

        Args:
            pretrained_path: Path or model id for pretrained weights.
            cache_dir: Directory to cache pretrained models.
            device: Device string ("cuda" or "cpu"); autodetect if None.
            feature_layers: Layer indices to extract features from.
                Default: [6, 9, 12] for multi-layer aggregation.

        Raises:
            ValueError: If pretrained_path is not in valid models list.
        """
        if pretrained_path not in self.VALID_MODELS:
            raise ValueError(f"Invalid pretrained_path: {pretrained_path}. Must be one of: {self.VALID_MODELS}")
        logging.info("Using DINOv2 pretrained path: %s", pretrained_path)

        super().__init__(pretrained_path, cache_dir, device)

        self.feature_layers = feature_layers if feature_layers is not None else [6, 9, 12]
        # DINOv2-base: 768 (small: 384, large: 1024, giant: 1536)
        self._feature_dim = 768
        self._logged_tensor_shape = False  # Log processed tensor shape once

        # Ensure cache directory exists if provided
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Define target image size for DINOv2 (224÷14=16 patches per side)
        self._target_size = 224

        # Load model and processor
        # Configure processor to use padding instead of center cropping
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            pretrained_path,
            cache_dir=cache_dir,
            do_center_crop=False,  # Disable center cropping
            do_resize=True,  # Keep resizing enabled
            size={"shortest_edge": self._target_size},
        )
        self.model = transformers.AutoModel.from_pretrained(pretrained_path, cache_dir=cache_dir)

        # Log basic configuration
        if hasattr(self.model, "config"):
            hidden_size = getattr(self.model.config, "hidden_size", "unknown")
            logging.info("DINOv2 loaded: hidden_size=%s, processor shortest_edge=%d", hidden_size, self._target_size)
        else:
            logging.info("DINOv2 processor configured: shortest_edge=%d", self._target_size)

        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self._feature_dim

    def _convert_tensor_to_pil(self, images: torch.Tensor) -> list[Image.Image]:
        """Convert tensor images to PIL images.

        Args:
            images: Tensor of shape (B, C, H, W) or (T, C, H, W).

        Returns:
            List of PIL images.
        """
        batch_size = images.shape[0]
        pil_images = []

        for i in range(batch_size):
            img_tensor = images[i]
            pil_images.append(torchvision.transforms.functional.to_pil_image(img_tensor.detach().clamp(0, 1).cpu()))

        return pil_images

    def _pad_to_square(self, pil_images: list[Image.Image]) -> list[Image.Image]:
        """Pad images to square aspect ratio (preserving aspect ratio).

        Args:
            pil_images: List of PIL images.

        Returns:
            List of square PIL images with black padding.
        """
        padded_images = []
        for img in pil_images:
            width, height = img.size
            max_dim = max(width, height)

            # Create a new square image with black padding
            new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))

            # Paste original image centered
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            new_img.paste(img, (paste_x, paste_y))

            padded_images.append(new_img)

        return padded_images

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using multiple layers.

        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            Features of shape (B, feature_dim * len(feature_layers)).
        """
        with torch.no_grad():
            # Log input and output shapes once (for debugging/verification)
            if not self._logged_tensor_shape:
                in_shape = images.shape
                logging.info("Input image size: [%d, %d, %d, %d]", in_shape[0], in_shape[1], in_shape[2], in_shape[3])

            # Convert tensor images to PIL for processor
            pil_images = self._convert_tensor_to_pil(images)

            # Check if images are already square - if so, skip padding
            _, _, H, W = images.shape
            already_square = H == W

            if not already_square:
                # Pad images to square (preserves aspect ratio)
                pil_images = self._pad_to_square(pil_images)

            # Check if images are already the target size
            # If so, disable processor resizing to avoid double-resize
            if already_square and H == self._target_size:
                # Images already preprocessed - skip resizing
                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                    do_resize=False,
                )
            else:
                # Images need resizing - use default processor behavior
                inputs = self.processor(images=pil_images, return_tensors="pt")

            if not self._logged_tensor_shape and "pixel_values" in inputs:
                out_shape = inputs["pixel_values"].shape
                logging.info(
                    "Processed tensor size: [%d, %d, %d, %d]", out_shape[0], out_shape[1], out_shape[2], out_shape[3]
                )
                self._logged_tensor_shape = True

            # Move tensors to device
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Get model outputs with hidden states
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract features from specified layers and concatenate
            layer_features = []
            for layer_idx in self.feature_layers:
                # Get hidden state from specified layer
                hidden_state = outputs.hidden_states[layer_idx]
                # Global average pooling (not CLS token!)
                if len(hidden_state.shape) == 3:  # [B, N, D]
                    pooled = hidden_state.mean(dim=1)  # [B, D]
                else:
                    pooled = hidden_state
                layer_features.append(pooled)

            # Concatenate features from different layers
            combined_features = torch.cat(layer_features, dim=1)

            return combined_features

    def extract_features_batch(
        self, images: torch.Tensor, return_numpy: bool = False, batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract multi-layer features from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1].
            return_numpy: Return numpy array instead of tensor.
            batch_size: Batch size for processing. If None, process all.

        Returns:
            Features tensor or array of shape (B, feature_dim).
        """
        if batch_size is None or images.shape[0] <= batch_size:
            # Process all at once
            features = self._extract_features(images)
            if return_numpy:
                return features.cpu().numpy()
            return features

        # Process in batches
        total_samples = images.shape[0]
        all_features: list[torch.Tensor] = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_images = images[start_idx:end_idx]
            batch_features = self._extract_features(batch_images)
            all_features.append(batch_features)

        # Concatenate all batch results
        concatenated_features = torch.cat(all_features, dim=0)

        if return_numpy:
            return concatenated_features.cpu().numpy()
        return concatenated_features


class FeatureExtractorFactory:
    """Factory class for creating feature extractors."""

    _EXTRACTORS = {
        "segformer": SegformerFeatureExtractor,
        "dinov2": DINOv2FeatureExtractor,
    }

    @classmethod
    def create_extractor(
        cls,
        extractor_type: str,
        pretrained_path: str,
        cache_dir: str | None = None,
        device: DeviceLikeType | None = None,
        **kwargs: Any,
    ) -> BaseFeatureExtractor:
        """Create a feature extractor of the specified type.

        Args:
            extractor_type: Type of feature extractor to create.
            pretrained_path: Path or model id for pretrained weights.
            cache_dir: Directory to cache pretrained models.
            device: Device string ("cuda" or "cpu"); autodetect if None.
            **kwargs: Additional keyword arguments for the extractor.

        Returns:
            Feature extractor instance.

        Raises:
            ValueError: If extractor_type is not supported.
        """
        if extractor_type not in cls._EXTRACTORS:
            available_types = list(cls._EXTRACTORS.keys())
            raise ValueError(f"Unsupported extractor type: {extractor_type}. Available types: {available_types}")

        extractor_class = cls._EXTRACTORS[extractor_type]
        return extractor_class(pretrained_path=pretrained_path, cache_dir=cache_dir, device=device, **kwargs)

    @classmethod
    def get_available_extractors(cls) -> list[str]:
        """Get list of available feature extractor types.

        Returns:
            List of available extractor type names.
        """
        return list(cls._EXTRACTORS.keys())
