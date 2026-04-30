#!/usr/bin/bash -ex

# This build script considers itself running inside a docker image built by Dockerfile_nrend_deps.build at NREND folder

# The following two variables may be defined prior of running this script

NRE_DIR=${NRE_DIR:-$(readlink -f "../../")}
NREND_DIR=${NREND_DIR:-$(readlink -f "./")}

FLAVOUR=Release

#
# Build NREND package for linux-x86_64
#

cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=${FLAVOUR} \
      -DNREND_TCNN_DIR=/tiny-cuda-nn \
      -DSLANG_DIR=/slang \
      -DCMAKE_INSTALL_PREFIX=out/nrend .

cmake --build build --verbose

cmake --install build --verbose

git config --global --add safe.directory ${NRE_DIR}

${NRE_DIR}/bazel/version/version_string.sh > VERSION

cp -r /repo_man_temp/tools /repo_man_temp/repo.sh ./

./repo.sh upload -p

mv _build/packages package_x86_64

#
# Build NREND package for aarch64
#

cmake -B build-cross -G Ninja \
      -DNREND_OBFUSCATOR_PATH=${NREND_DIR}/build \
      -DCMAKE_TOOLCHAIN_FILE=/Toolchain_aarch64.cmake \
      -DCMAKE_BUILD_TYPE=${FLAVOUR} \
      -DNREND_TCNN_DIR=/tiny-cuda-nn \
      -DSLANG_DIR=/slang \
      -DCMAKE_INSTALL_PREFIX=out-cross/nrend .

cmake --build build-cross --verbose

cmake --install build-cross --verbose

sed -i 's/${platform}/linux-aarch64/' repo.toml
sed -i 's/out/out-cross/' repo.toml

./repo.sh upload -p

mv _build/packages package_aarch64
