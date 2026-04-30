#!/bin/bash

# The Conda Mode environment is provided by a Docker container defined in this Dockerfile: `scripts/cluster_toolbox/Dockerfile-dev-python.build`.
# This container is maintained to facilitate development (mainly for NRM training) now.
# If you need to update the environment, rebuild the Docker container
# by running this script, and then perform an `enroot` squashing on ORD with the command generated at the end of the script.
#
# WARNING: This script has the potential to be quite error-prone if one forgets to update the image to include the changes in C++/CUDA libs (e.g. vren / # ray-utils). One has to be aware of the implications on the environment, and end-users should always use the `bazel`-based environment for productions.
# 
# Current latest version of the Docker container is available at:
# - nvcr.io: `nvcr.io/nvidian/nre:python-dev-0.2.765_0ce3eab2`
# - ORD: `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/nre:python-dev-0.2.765_0ce3eab2.sqsh`

REPO_COMMIT=$(git rev-parse --short HEAD) 

IMAGE_NAME=nre:python-dev-$(bazel/version/version_string.sh | tr '-' '_')
SQUASH=python-dev-$(bazel/version/version_string.sh | tr '-' '_').sqsh

IMAGE=nvidian/$IMAGE_NAME
DOCKER_BUILDKIT=1 docker image build -t nvcr.io/$IMAGE -f scripts/cluster_toolbox/Dockerfile-dev-python.build --ssh default --secret id=netrc,src=$HOME/.netrc --build-arg REPO_COMMIT=$REPO_COMMIT . || exit 1
docker push nvcr.io/$IMAGE

echo 
echo "Please run the following command to import the container:"
echo srun -A nvr_torontoai_3dscenerecon --partition interactive --gpus 1 --pty enroot import --output /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/$IMAGE_NAME.sqsh \''docker://$oauthtoken@nvcr.io#'$IMAGE\'
echo
echo "Alternatively, you can build the squash files locally and then upload it:"
echo enroot import --output /tmp/$IMAGE_NAME.sqsh \'dockerd://nvcr.io/$IMAGE\'
echo scp /tmp/$IMAGE_NAME.sqsh cs-oci-ord-dc-01.nvidia.com:/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/$IMAGE_NAME.sqsh
echo 
