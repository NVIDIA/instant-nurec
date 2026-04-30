#!/usr/bin/bash -ex

# The following 3 variables may be defined prior of running this script, where at least gitlab token must be defined

GITLAB_USER=${GITLAB_USER:-oauth2}
GITLAB_TOKEN=${GITLAB_TOKEN:-""}
DOCKER_URL=${DOCKER_URL:-"gitlab-master.nvidia.com:5005/omniverse/nrend/nrend-deps"}

DOCKER_TAG=$(grep 'NREND_DEPS_VERSION:' nrend-deps.yml | awk '{print $2}' | tr -d '"')

DOCKER_IMAGE=$DOCKER_URL:$DOCKER_TAG

# Extract Slang version from MODULE.bazel at repo root
NRE_PATH=$(git rev-parse --show-toplevel)
MODULE_BAZEL_PATH="${NRE_PATH}/MODULE.bazel"

if [ ! -f "${MODULE_BAZEL_PATH}" ]; then
    echo "Error: MODULE.bazel not found at ${MODULE_BAZEL_PATH}"
    exit 1
fi

SLANG_VERSION=$(grep -oP 'releases/download/\Kv[0-9]+\.[0-9]+\.[0-9]+' "${MODULE_BAZEL_PATH}" | head -n 1)

if [ -z "$SLANG_VERSION" ]; then
    echo "Error: Could not extract Slang version from ${MODULE_BAZEL_PATH}"
    exit 1
fi

echo "Extracted Slang version: ${SLANG_VERSION}"

DOCKER_BUILDKIT=0 docker image build --network=host --tag ${DOCKER_IMAGE} \
    --build-arg GITLAB_USER=$GITLAB_USER \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    --build-arg SLANG_VERSION=$SLANG_VERSION \
    -f Dockerfile_nrend_deps.build .

docker push ${DOCKER_IMAGE}
