#!/usr/bin/bash -ex

# The following variable may be defined prior of running this script

DOCKER_URL=${DOCKER_URL:-"gitlab-master.nvidia.com:5005/omniverse/nrend/nrend-deps"}

DOCKER_TAG=$(grep 'NREND_DEPS_VERSION:' nrend-deps.yml | awk '{print $2}' | tr -d '"')

DOCKER_IMAGE=$DOCKER_URL:$DOCKER_TAG

NRE_PATH=$(git rev-parse --show-toplevel)
NRE_ROOT=$(basename ${NRE_PATH})

rm -rf /tmp/${NRE_ROOT}
cp -r ${NRE_PATH} /tmp/

docker run -i --rm --network=host --entrypoint bash \
       --mount type=bind,source=/tmp,target=/tmp \
       -w /tmp/${NRE_ROOT}/libs/nrend \
       ${DOCKER_IMAGE} build_linux.sh
