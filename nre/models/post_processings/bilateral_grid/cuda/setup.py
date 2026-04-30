# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import pathlib

import torch.version

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


torch_version = torch.version.__version__.split(".")[:2]
cuda_version = torch.version.cuda

# This will be e.g. "+pt23cu121"
assert cuda_version is not None, "Pytorch CUDA is required for this installation."
version_suffix = f"+pt{torch_version[0]}{torch_version[1]}cu{cuda_version.replace('.', '')}"


sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(
    name="bilateral_grid_cuda",
    version="0.1" + version_suffix,
    author="nvidia",
    author_email="nre@nvidia.com",
    description="CUDA bilateral grid operations",
    long_description="CUDA bilateral grid operations",
    ext_modules=[
        CUDAExtension(
            name="bilateral_grid_cuda",
            sources=sources,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
            # Register header include path for kernel_utils interface library
            include_dirs=[pathlib.Path(__file__).parent.parent.absolute() / "kernel_utils" / "inc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
