# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from nre.models.nn_extensions import BufferList


def test_buffer_list() -> None:
    class Example(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.blist = BufferList([])

    example = Example()
    example.blist = BufferList([torch.randn(10, 10) for i in range(10)])

    example.to("cuda")
    for i in range(10):
        assert example.blist[i].device.type == "cuda"

    example.to("cpu")
    for i in range(10):
        assert example.blist[i].device.type == "cpu"
        assert example.blist[i].dtype == torch.float32

    example.blist.to(dtype=torch.float16)
    for i in range(10):
        assert example.blist[i].dtype == torch.float16
