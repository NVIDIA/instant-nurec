# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dataclasses

from enum import Enum, Flag
from typing import Any, Optional, Type

import dataclasses_json
import numpy as np
import numpy.typing as npt
import torch

import ncore.data


def field_numpy_array(dtype_like: npt.DTypeLike, shape: tuple[int, ...], *args, **kwargs):
    """Provides encoder / decoder functionality for numpy arrays or dicts or arrays into field types compatible with dataclass-JSON

    Args:
    - dtype_like: required for encoding / decoding (encoded inputs are verified to have this datatype, and decoded arrays will be returned with this type)
    - shape: tuple of array dimensions, used for consistency checks (one dimension is allowed to be -1, similar to reshape)
    - remaining args/kwargs are forward to dataclasses.field()
    """

    dtype = np.dtype(dtype_like)

    def validate_shape(array: np.ndarray) -> np.ndarray:
        if array.size == 0 and any(dim == -1 for dim in shape):
            array = array.reshape(*shape)  # special case: 0-size arrays can take on any shape with a -1 dim

        assert len(array.shape) == len(shape) and all(
            [dim == expected_dim if expected_dim != -1 else True for (dim, expected_dim) in zip(array.shape, shape)]
        ), f"Array {array} not having expected shape {shape}"
        return array

    def decoder(input: list | dict[Any, list]) -> np.ndarray | dict[Any, np.ndarray]:
        match input:
            case dict():
                # decode dict[<key-type>, list]
                return {key: validate_shape(np.array(value, dtype=dtype)) for (key, value) in input.items()}
            case list():
                # encode list
                return validate_shape(np.array(input, dtype=dtype))
            case _:
                raise ValueError(f"field_numpy_array: unsupported decoder input type {type(input)}")

    def encoder(input: np.ndarray | dict[Any, np.ndarray]) -> list | dict[Any, list]:
        match input:
            case dict():
                # encode as dict[<key-type>, list]
                assert all([array.dtype == dtype for array in input.values()]), (
                    f"Not all arrays in {input} of expected dtype {dtype}"
                )
                return {key: np.ndarray.tolist(validate_shape(array)) for (key, array) in input.items()}
            case np.ndarray():
                # encode as list
                assert input.dtype == dtype, f"Provided array {input} is not of expected dtype {dtype}"
                return np.ndarray.tolist(validate_shape(input))
            case _:
                raise ValueError(f"field_numpy_array: unsupported encoder input type {type(input)}")

    return dataclasses.field(metadata=dataclasses_json.config(encoder=encoder, decoder=decoder), *args, **kwargs)


def field_torch_tensor(dtype: torch.dtype, shape: tuple[int, ...], device: Optional[str] = None, *args, **kwargs):
    """Provides encoder / decoder functionality for torch tensors or dicts of tensors into field types compatible with dataclass-JSON

    Args:
    - dtype: required for encoding / decoding (encoded inputs are verified to have this datatype, and decoded arrays will be returned with this type)
    - shape: tuple of tensor dimensions, used for consistency checks (one dimension is allowed to be -1, similar to reshape)
    - device: optional device for the deserialized data
    - remaining args/kwargs are forward to dataclasses.field()
    """

    def validate(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size == 0 and any(dim == -1 for dim in shape):
            tensor = tensor.reshape(*shape)  # special case: 0-size tensors can take on any shape with a -1 dim

        assert len(tensor.shape) == len(shape) and all(
            [dim == expected_dim if expected_dim != -1 else True for (dim, expected_dim) in zip(tensor.shape, shape)]
        ), f"Tensor {tensor} not having expected shape {shape}"
        if device is not None:
            assert tensor.device.type == torch.device(device).type, (
                f"Tensor {tensor} not having expected device '{device}'"
            )
        return tensor

    def decoder(input: list | dict[Any, list]) -> torch.Tensor | dict[Any, torch.Tensor]:
        match input:
            case dict():
                # decode dict[<key-type>, list]
                return {
                    key: validate(torch.tensor(value, dtype=dtype, device=device)) for (key, value) in input.items()
                }
            case list():
                # encode list
                return validate(torch.tensor(input, dtype=dtype, device=device))
            case _:
                raise ValueError(f"field_torch_tensor: unsupported decoder input type {type(input)}")

    def encoder(input: torch.Tensor | dict[Any, torch.Tensor] | None) -> list | dict[Any, list] | None:
        match input:
            case dict():
                # encode as dict[<key-type>, list]
                assert all([tensor.dtype == dtype for tensor in input.values()]), (
                    f"Not all tensors in {input} of expected dtype {dtype}"
                )
                return {key: validate(tensor).tolist() for (key, tensor) in input.items()}
            case torch.Tensor():
                # encode as list
                assert input.dtype == dtype, f"Provided tensor {input} is not of expected dtype {dtype}"
                return validate(input).tolist()
            case None:
                return None
            case _:
                raise ValueError(f"field_torch_tensor: unsupported encoder input type {type(input)}")

    return dataclasses.field(metadata=dataclasses_json.config(encoder=encoder, decoder=decoder), *args, **kwargs)


def field_camera_model_parameters(*args, **kwargs):
    """Provides encoder / decoder functionality for NCore CameraModelParameters types compatible with dataclass-JSON

    Encoded camera-model-parameters will be encoded as

    "camera_model": {
        "parameters": {
            <TYPE_SPECIFIC_PARAMETERS>
        },
        "type": "<TYPE_ID>"
    },
    """

    def decoder(input: dict) -> ncore.data.ConcreteCameraModelParametersUnion:
        # deserialize based on encoded camera model type
        match type := input["type"]:
            case "ftheta":
                return ncore.data.FThetaCameraModelParameters.from_dict(input["parameters"])
            case "opencv-pinhole":
                return ncore.data.OpenCVPinholeCameraModelParameters.from_dict(input["parameters"])
            case "opencv-fisheye":
                return ncore.data.OpenCVFisheyeCameraModelParameters.from_dict(input["parameters"])
            case _:
                raise ValueError(f"field_camera_model_parameters: unknown camera model_type '{type}'")

    def encoder(input: ncore.data.ConcreteCameraModelParametersUnion) -> dict:
        # serialize camera model type and parameters
        return {"type": input.type(), "parameters": input.to_dict()}

    return dataclasses.field(
        metadata=dataclasses_json.config(encoder=encoder, decoder=decoder, field_name="camera_model"), *args, **kwargs
    )


def field_lidar_model_parameters(*args, **kwargs):
    """Provides encoder / decoder functionality for NCore LidarModelParameters types compatible with dataclass-JSON

    Encoded lidar-model-parameters will be encoded as

    "lidar_model": {
        "parameters": {
            <TYPE_SPECIFIC_PARAMETERS>
        },
        "type": "<TYPE_ID>"
    },
    """

    def decoder(input: dict) -> ncore.data.ConcreteLidarModelParametersUnion:
        # deserialize based on encoded lidar model type
        match type := input["type"]:
            case "row-offset-spinning":
                return ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_dict(input["parameters"])
            case _:
                raise ValueError(f"field_lidar_model_parameters: unknown lidar model_type '{type}'")

    def encoder(input: ncore.data.ConcreteLidarModelParametersUnion | None) -> dict | None:
        if input is None:
            return None

        # serialize lidar model type and parameters
        return {"type": input.type(), "parameters": input.to_dict()}

    return dataclasses.field(
        metadata=dataclasses_json.config(encoder=encoder, decoder=decoder, field_name="lidar_model"), *args, **kwargs
    )


def field_enum(enum_class, *args, **kwargs):
    """Provides encoder / decoder functionality for enum types or lists of enum types into field types compatible with dataclass-JSON"""

    def encoder(input: Enum | list[Enum]) -> str | list[str]:
        """Encode enum as `name`-based strings. This way values in JSON are "human-readable".
        `Flag`-derived enum types will automatically be encoded as '|'-joined strings"""
        match input:
            case list():
                # encode as list
                return [variant.name for variant in input]
            case Enum():
                # encode as variant
                return input.name
            case _:
                raise ValueError(f"field_enum: unsupported encoder input type {type(input)}")

    def decoder(input: str | list[str]) -> Enum | list[Enum]:
        def decode_str(input: str, enum_class: Type[Enum]) -> Enum:
            """Decode string-representations of enums with a special case for
            `Flag`-derived enum types, which are encoded as '|'-joined strings"""
            if issubclass(enum_class, Flag):
                ret = enum_class(0)
                for name in input.split("|"):
                    ret |= enum_class.__members__[name]
                return ret
            else:
                return enum_class.__members__[input]

        match input:
            case list():
                # decode from list
                return [decode_str(variant, enum_class) for variant in input]
            case str():
                # decode from string
                return decode_str(input, enum_class)
            case _:
                raise ValueError(f"field_enum: unsupported decoder input type {type(input)}")

    return dataclasses.field(metadata=dataclasses_json.config(encoder=encoder, decoder=decoder), *args, **kwargs)
