#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Simple script to run 1 example script using run-script command

set -e

# Parse arguments
TAG=""
RUNFILES_EXECUTABLE=""
BAZEL=false
RUN_SCRIPT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --tag)
        TAG="$2"
        shift
        ;;
    --runfiles)
        RUNFILES_EXECUTABLE="$2"
        shift
        ;;
    --bazel)
        BAZEL=true
        ;;
    --run-script)
        RUN_SCRIPT="$2"
        shift
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$RUN_SCRIPT" ]]; then
    echo "Error: --run-script is required"
    exit 1
fi

# Docker configuration (always non-obfuscated)
NGC_IMAGES="nvcr.io/nvidian/ct-toronto-ai"
NRE_RUN_NAME="nre_run"
NRE_IMAGE="${NGC_IMAGES}/${NRE_RUN_NAME}:${TAG}"

echo "========================================================================"
echo "Running example script: $RUN_SCRIPT"
echo "========================================================================"

# Determine execution method and run
if [[ "$BAZEL" == true ]]; then
    # Bazel run (always non-obfuscated)
    BAZEL_TARGET="//:run"

    CMD="bazel run ${BAZEL_TARGET} -- run-script ${RUN_SCRIPT}"
    echo "EXECUTE COMMAND: $CMD"
    eval "$CMD"

elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
    # Runfiles execution
    CMD="${RUNFILES_EXECUTABLE} run-script ${RUN_SCRIPT}"
    echo "EXECUTE COMMAND: $CMD"
    eval "$CMD"

else
    # Docker execution
    if [[ -z "$TAG" ]]; then
        echo "Error: --tag is required for Docker execution"
        exit 1
    fi

    echo "Using Docker image: $NRE_IMAGE"

    # Check if NGC_API_KEY is set
    if [[ -z "${NGC_API_KEY}" ]]; then
        echo "NGC_API_KEY environment variable is not set. Try to read from ~/.netrc file"
        NGC_API_KEY=$(awk '
        /machine api.ngc.nvidia.com/ { 
            flag=1; next 
        } 
        flag && /machine/ { 
            print "Error: Another machine entry found before password." > "/dev/stderr"; exit 1 
        } 
        flag && /password/ { 
            print $2; exit 
        } 
        END { 
            if (!flag) { 
                print "Error: machine api.ngc.nvidia.com not found." > "/dev/stderr"; exit 1 
            } 
        }
    ' ~/.netrc) || exit 1
        echo "Found NGC_API_KEY"
    fi

    # Check if running in WSL
    IS_WSL=false
    if [[ -f /proc/version ]] && grep -q Microsoft /proc/version || [[ -d /usr/lib/wsl ]]; then
        IS_WSL=true
    fi

    # Set environment variables for Docker
    DOCKER_ENVVARS=("-e NGC_API_KEY=${NGC_API_KEY}")
    if [[ "$IS_WSL" == true ]]; then
        DOCKER_ENVVARS+=("-e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so")
    fi

    DOCKER_ARGS=(
        run
        --shm-size=8g
        -it
        --user $(id -u):$(id -g)
        --rm
        --gpus all
        ${DOCKER_ENVVARS[@]}
    )
   
    CMD="docker ${DOCKER_ARGS[@]} ${NRE_IMAGE} run-script ${RUN_SCRIPT}"
    echo "EXECUTE COMMAND: $CMD"
    docker "${DOCKER_ARGS[@]}" "${NRE_IMAGE}" run-script "${RUN_SCRIPT}"
fi

echo ""
echo "Example script completed successfully"
