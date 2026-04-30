#!/usr/bin/env bash

# Copyright (c) 2023-2025 NVIDIA CORPORATION.  All rights reserved.

# This script will generate a version string according to the pattern
#
# MAJOR.MINOR.PATCH-<GIT_COMMIT_SHA_SHORT[+GIT_TREE_DIRTY][DEV_SUFFIX]>
#
# with
#
# - MAJOR/MINOR versions are stored explicitly in 'VERSION_FILE'
# - PATCH versions automatically deduced from the number of merge commits
#   since the last change to 'VERSION_FILE'
# - current repo's short git commit hash GIT_COMMIT_SHA_SHORT (amended by
#   a '+dirty' suffix in case the current workspace has uncommited changes
#   to tracked files)
# - optional '-dev' suffix marking unofficial builds (local build, unprotected
#   branch, non-RC tag)

# Path to version file (in same folder as current script)
version_file="$(dirname ${BASH_SOURCE[0]})/VERSION_FILE"

# Check if the version file exists
if [ ! -f "$version_file" ]; then
  echo "Error: File not found - $version_file"
  exit 1
fi

# Declare an associative array to store key-value pairs loaded from version file
declare -A key_value_pairs

# Use grep to find all lines with KEY=VALUE pattern and extract them as key-value pairs
while IFS= read -r line; do
  key=$(echo "$line" | cut -d= -f1)
  value=$(echo "$line" | cut -d= -f2)
  key_value_pairs["$key"]="$value"
done < <(grep -oE '[[:alnum:]_]+=[^[:space:]]+' "$version_file")

MAJOR=${key_value_pairs[VERSION_MAJOR]}
MINOR=${key_value_pairs[VERSION_MINOR]}

# Function to get the last commit hash where a specific file was changed
get_last_commit_for_file() {
  file=$1
  git log -n 1 --format=format:%H -- $file
}

# Function to count the number of merge commits since a specific commit
count_merge_commits_since() {
  commit_hash=$1
  git rev-list --count $commit_hash..HEAD --merges
}

# Get the last commit hash where the version file was changed
last_version_file_commit=$(get_last_commit_for_file "$version_file")

if [ -n "$last_version_file_commit" ]; then
  # Count the number of merge commits since the last commit and use this as patch version
  PATCH=$(count_merge_commits_since "$last_version_file_commit")
else
  echo "Version file not found or not committed"
  exit 1
fi

# Parse git commit hash and dirty flag
GIT_COMMIT_SHA_SHORT=$(git rev-parse --short=8 HEAD)
GIT_TREE_DIRTY=$(git diff-index --quiet HEAD -- && echo '' || echo '+dirty')

# Compute "-dev" suffix
# - Set the suffix by default
# - Remove it for protected branches in CI (main/release branches) and RC tags
# - Skip it if the tree is dirty, since the "+dirty" suffix is stronger (no unique mapping from version string to
#   state of the source tree, prevents container upload)
DEV_SUFFIX="-dev"
if [ -n "$GIT_TREE_DIRTY" ] || [ "$CI_COMMIT_REF_PROTECTED" == "true" ] || [[ "$CI_COMMIT_TAG" =~ ^[0-9]{2}\.(0[1-9]|1[0-2])-rc[1-9][0-9]*$ ]]; then
  DEV_SUFFIX=""
fi

# Output final version string
echo "$MAJOR.$MINOR.$PATCH-$GIT_COMMIT_SHA_SHORT$GIT_TREE_DIRTY$DEV_SUFFIX"
