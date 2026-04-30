#!/usr/bin/env bash

# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

# Script to fetch SHA256 hashes for Python versions from Astral's python-build-standalone
# Usage: ./get_python_hashes.sh <python_version> <release_date>
# Example: ./get_python_hashes.sh 3.11.13 20250604

if [ $# -ne 2 ]; then
    echo "Usage: $0 <python_version> <release_date>"
    echo "Example: $0 3.11.13 20250604"
    exit 1
fi

PYTHON_VERSION=$1
RELEASE_DATE=$2

echo "Python $PYTHON_VERSION SHA256 Hashes from python-build-standalone"
echo "=========================================================="
echo ""
echo "Release: https://github.com/astral-sh/python-build-standalone/releases/tag/$RELEASE_DATE"
echo ""

# Define the platforms
platforms=(
    "aarch64-apple-darwin"
    "aarch64-unknown-linux-gnu"
    "ppc64le-unknown-linux-gnu"
    "riscv64-unknown-linux-gnu"
    "s390x-unknown-linux-gnu"
    "x86_64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "x86_64-unknown-linux-gnu"
    "x86_64-unknown-linux-musl"
)
declare -A hashes

# Base URL for the release
base_url="https://github.com/astral-sh/python-build-standalone/releases/download/$RELEASE_DATE"

echo "Fetching hashes for each platform..."
echo ""

for platform in "${platforms[@]}"; do
    filename="cpython-${PYTHON_VERSION}+${RELEASE_DATE}-${platform}-install_only.tar.gz"
    file_url="${base_url}/${filename}"
    # Fetch the SHA256 hash
    hash=$(curl -sL "$file_url" 2>/dev/null | sha256sum | awk '{print $1}')
    hashes["$platform"]="$hash"
    
    echo "Platform: $platform"
    echo "File: $filename"
    
    if [ -n "$hash" ]; then
        echo "SHA256: $hash"
    else
        echo "SHA256: ERROR - Could not fetch hash from $sha256_url"
    fi
    echo ""
done

echo "=========================================================="
echo ""
echo "For rules_python versions.bzl format:"
echo ""
echo "\"$PYTHON_VERSION\": {"
echo "    \"url\": \"$RELEASE_DATE/cpython-{python_version}+$RELEASE_DATE-{platform}-{build}.tar.gz\","
echo "    \"sha256\": {"

for platform in "${platforms[@]}"; do
    hash="${hashes[$platform]}"
    echo "        \"$platform\": \"$hash\","
done

echo "    },"
echo "    \"strip_prefix\": \"python\","
echo "},"
