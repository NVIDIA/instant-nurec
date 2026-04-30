# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import logging

from collections.abc import Mapping, Sequence

import torch

from nre.utils.types import Checkpoint, NamedSerialized


def serialize_checkpoint(checkpoint: Checkpoint, filename: str = "checkpoint.ckpt") -> NamedSerialized:
    with io.BytesIO() as buff:
        torch.save(checkpoint, buff)
        return NamedSerialized(filename=filename, serialized=buff.getvalue())


def strip_optimizer_state(checkpoint: Checkpoint) -> Checkpoint:
    keys_to_pop = ["loops", "callbacks", "optimizer_states", "lr_schedulers", "MixedPrecisionPlugin"]
    return Checkpoint({k: v for k, v in checkpoint.items() if k not in keys_to_pop})


def upcast_fp16_to_fp32(checkpoint: Checkpoint) -> Checkpoint:
    """Recursively upcast all fp16 tensors to fp32. Inverse of reduce_precision_to_fp16."""

    def upcast_recursive(item):
        if isinstance(item, torch.Tensor):
            return item.to(dtype=torch.float32) if item.dtype == torch.float16 else item
        elif isinstance(item, Mapping):
            return {k: upcast_recursive(v) for k, v in item.items()}
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [upcast_recursive(v) for v in item]
        return item

    return upcast_recursive(checkpoint)


def reduce_precision_to_fp16(checkpoint: Checkpoint) -> Checkpoint:
    logger = logging.getLogger(__name__)

    def reduce_precision_recursive(item):
        if isinstance(item, torch.Tensor):
            if item.dtype == torch.float32:
                if not (
                    in_fp16_range := (
                        (item < torch.finfo(torch.float16).max).all() and (item > torch.finfo(torch.float16).min).all()
                    )
                ):
                    logger.warning(
                        f"WARNING: Elements of tensor {item} of type torch.float32 is outside the range of torch.float16, skipping the casting operation."
                    )
                return item.to(dtype=torch.float16) if in_fp16_range else item

            return item
        elif isinstance(item, Mapping):
            return {k: reduce_precision_recursive(v) for k, v in item.items()}
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [reduce_precision_recursive(v) for v in item]
        return item

    return reduce_precision_recursive(checkpoint)
