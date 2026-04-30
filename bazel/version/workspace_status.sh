#!/usr/bin/env bash

# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

# This script will be run by bazel at build process starts to
# generate key-value information that represents the status of the
# workspace. The output should be like
#
# KEY1 VALUE1
# KEY2 VALUE2
#
# If the script exits with non-zero code, it's considered as a failure
# and the output will be discarded.
#
# Variables prefixed with 'STABLE_' are assumed to not change often /
# will result in rebuild / cache invalidations.

set -eo pipefail # exit immediately if any command fails.

function remove_url_credentials() {
  which perl > /dev/null && perl -pe 's#//.*?:.*?@#//#' || cat
}

repo_url=$(git config --get remote.origin.url | remove_url_credentials)
echo "STABLE_GIT_REPO_URL $repo_url"

commit_sha=$(git rev-parse HEAD)
echo "STABLE_GIT_COMMIT_SHA $commit_sha"

commit_date=$(git show -s --format=%cI)
echo "STABLE_GIT_COMMIT_DATE $commit_date"

git_branch=$(git rev-parse --abbrev-ref HEAD)
echo "STABLE_GIT_BRANCH $git_branch"

git_tree_status=$(git diff-index --quiet HEAD -- && echo 'clean' || echo 'modified')
echo "STABLE_GIT_TREE_STATUS $git_tree_status"

version_string="$($(dirname ${BASH_SOURCE[0]})/version_string.sh)"
echo "STABLE_VERSION_STRING $version_string"
