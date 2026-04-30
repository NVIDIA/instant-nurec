# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import platform
import re
import subprocess
import sys
import warnings

from distutils.version import LooseVersion
from typing import List, Optional

import setuptools

from setuptools.command.build_ext import build_ext


def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir="", cmake_args=(), exclude_arch=False):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args
        self.exclude_arch = exclude_arch


class CMakeBuild(build_ext):
    def run(self) -> None:
        if os.path.exists(".git"):
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"], check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to update git submodules: {e.stderr}")

        try:
            result = subprocess.run(["cmake", "--version"], check=True, capture_output=True, text=True)
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", result.stdout).group(1))  # type: ignore
            if cmake_version < "3.2.0":
                raise RuntimeError("CMake >= 3.2.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir)
        cmake_args: List[str] = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DPYTHON_EXECUTABLE=" + sys.executable]
        cmake_args.extend(ext.cmake_args)

        cfg = "Debug" if self.debug or os.environ.get("PCU_DEBUG") else "Release"
        build_args = ["--config", cfg]

        if cfg == "Debug":
            warnings.warn("Building extension %s in debug mode" % ext.name)

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            if os.environ.get("CMAKE_GENERATOR") != "NMake Makefiles":
                if sys.maxsize > 2**32 and not ext.exclude_arch:
                    cmake_args += ["-A", "x64"]
                build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        try:
            subprocess.run(
                ["cmake"] + cmake_args + [ext.sourcedir],
                cwd=self.build_temp,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["cmake", "--build", "."] + build_args, cwd=self.build_temp, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build extension: {e.stderr}")

        print()  # Add an empty line for cleaner output


def main():
    cmake_args = []
    exclude_arch = False

    setuptools.setup(
        name="ray_utils",
        version="0.1.0",
        author="Zan Gojcic",
        author_email="zan.gojcic@gmail.com  ",
        description="Ray utility function for torch-ngp-extension",
        long_description="Utility function for av data preprocessing",
        long_description_content_type="text/markdown",
        url="..",
        packages=[""],
        classifiers=["Programming Language :: C++", "Programming Language :: Python :: 3"],
        ext_modules=[CMakeExtension("", cmake_args=cmake_args, exclude_arch=exclude_arch)],
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False,
        install_requires=["numpy", "scipy"],
    )


if __name__ == "__main__":
    main()
