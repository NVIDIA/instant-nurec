#!/bin/bash

# This script creates a pip cache for the current repository and stores it in a shared cache directory. The cache is 
# used to pre-fill the pip cache for ORD jobs. To generate a new cache, modify the user setup and run this script.

REPO_COMMIT=$(git rev-parse --short HEAD) 
CACHE_NAME=nre-pip-cache-$(bazel/version/version_string.sh | tr '-' '_')
# Use Qi's cache path for now as it is accessible by all users across different projects.
CACHE_PATH=/lustre/fsw/portfolios/nvr/users/qiwu/caches

bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml \
  submit-job user=$(whoami) team=nvr_torontoai_3dscenerecon \
  git_commit=$REPO_COMMIT \
  --job-name nre-dev \
  --command "export PIP_CACHE_DIR=/scratch/pip_cache && bazel build //:run && tar -cvf $CACHE_PATH/$CACHE_NAME.tar /scratch/pip_cache"
