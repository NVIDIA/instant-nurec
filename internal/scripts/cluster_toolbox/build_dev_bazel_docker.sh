#!/bin/bash

# The Bazel-mode environment is provided by a Docker image defined in this Dockerfile: `Dockerfile-dev-bazel{_arm64}.build`.
# This image is manually maintained by the NRE team and is generally stable. If you need to update the environment, rebuild the Docker image
# by running this script, then export it to a filesystem image (enroot .sqsh) using the command generated at the end of the script.
#
# A prebuilt version of the Docker image is available at:
# - nvcr.io: `nvidian/ct-toronto-ai/nre_run_ord_bazel_dev`
# - ORD: `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/nre-ord-dev-v0_2_650_d2affc3c.sqsh`

# NOTE: CLUSTER_NAME is an environment variable defined by the `ADLR_UTILS` setup script that points to the cluster you are running on. You can
#       check its definition in the `ADLR_UTILS` setup script.
declare -A machines=(
    ["CS_OCI_ORD"]="amd64" # https://confluence.nvidia.com/display/HWINFCSSUP/CS-OCI-ORD+HWInf+Compute+Node+Information
    ["CS_OCI_ORD_002"]="amd64" # https://confluence.nvidia.com/display/HWINFCSSUP/oci-ord-cs-002+HWInf+Compute+Node+Information
    ["AWS_IAD_CS_002"]="amd64" # https://confluence.nvidia.com/display/HWINFCSSUP/aws-iad-cs-002+HWInf+Compute+Node+Information
    ["OCI_HSG_CS_001"]="arm64" # https://confluence.nvidia.com/display/HWINFCSSUP/oci-hsg-cs-001+HWInf+Node+Information
)
PLATFORM=${PLATFORM:-${machines[$CLUSTER_NAME]}}
PLATFORM=${PLATFORM:-"amd64"} # Available platforms: amd64, arm64
echo "PLATFORM: linux/${PLATFORM}"

USE_CBUILD=${USE_CBUILD:-0}

if [ -z "$1" ]; then
    IMAGE=nvidian/ct-toronto-ai/nre_run_ord_bazel_dev:nre_dev_$(bazel/version/version_string.sh | tr '-' '_')
    SQUASH=nre-ord-dev-v$(bazel/version/version_string.sh | tr '-' '_').sqsh
    # check if docker is installed
    if [ $USE_CBUILD -eq 1 ]; then
        cbuild build --platform=linux/${PLATFORM} --dockerfile=internal/scripts/cluster_toolbox/Dockerfile-dev-bazel_${PLATFORM}.build --name=nvcr.io/${IMAGE} --no-cache
    else
        DOCKER_BUILDKIT=1 docker image build -t nvcr.io/$IMAGE -f internal/scripts/cluster_toolbox/Dockerfile-dev-bazel_${PLATFORM}.build . || exit 1
        docker push nvcr.io/$IMAGE
    fi
else
    # remove "nvcr.io/" from the image name if it exists
    IMAGE=${1#"nvcr.io/"}
    SQUASH=$(basename $1).sqsh
fi

echo 
echo "Please run the following command to import the image into an enroot .sqsh file:"
echo srun -A nvr_torontoai_3dscenerecon --partition cpu --nodes 1 --pty --pty enroot import \
    --output /lustre/fsw/portfolios/nvr/users/$USER/containers/$SQUASH \''docker://$oauthtoken@nvcr.io#'$IMAGE\'
echo
echo "Alternatively, you can use the cbuild utility to export the image directly to an enroot .sqsh file:"
echo "  cbuild export --force nvcr.io/${IMAGE} /lustre/fsw/portfolios/nvr/users/$USER/containers/$SQUASH"
echo
