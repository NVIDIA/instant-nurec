#!/bin/bash

# This script creates a Bazel cache for the current repository and stores it in a shared cache directory. The cache is 
# used to pre-fill the Bazel cache for ORD jobs. To generate a new cache, modify the user setup and run this script.

REPO_COMMIT=$(git rev-parse --short HEAD) 
CACHE_LINK=nre-bazel-cache-latest
CACHE_NAME=nre-bazel-cache-$(bazel/version/version_string.sh | tr '-' '_')
# Use Qi's cache path for now as it is accessible by all users across different projects.
CACHE_PATH=/lustre/fsw/portfolios/nvr/users/qiwu/caches  

bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml \
  submit-job user=$(whoami) team=nvr_torontoai_3dscenerecon partition=[interactive] bazel.cache_prefill="empty" \
  git_commit=$REPO_COMMIT num_nodes=1 num_gpus=8 \
  --job-name nre-dev \
  --command "bazel build //:run && \
    tar -cvf $CACHE_PATH/$CACHE_NAME-rank\${SLURM_LOCALID}.tar \${RUNCACHE} && \
    ln  -fs  $CACHE_PATH/$CACHE_NAME-rank\${SLURM_LOCALID}.tar $CACHE_PATH/$CACHE_LINK-rank\${SLURM_LOCALID}.tar"
