#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# SQA Scripts Utilities
# Common functions for Docker volume mounting and container detection

# Function to get volume info for a given path if it's under a named volume mount
# Returns: "volume_name|mount_point" or empty string if not on a named volume
get_volume_info_for_path() {
    local path="$1"

    if [ -f "/.dockerenv" ]; then
        # Find container by name pattern
        # In GitLab CI the hostname is a prefix of the container name, not an exact match
        local container_id=$(docker ps --format "{{.Names}}" | grep "^$(hostname)" | head -1 2>/dev/null || true)
        if [ -z "$container_id" ]; then
            return
        fi

        # Check each mount to find the longest matching one
        local best_match=""
        local best_mount_point=""
        local best_length=0
        local mounts=($(docker inspect "$container_id" --format='{{range .Mounts}}{{.Destination}}|{{.Name}} {{end}}' 2>/dev/null || true))
        for mount_info in "${mounts[@]}"; do
            local mount_dest="${mount_info%|*}"    # Everything before |
            local volume_name="${mount_info#*|}"   # Everything after |

            # Check if path starts with this mount destination
            if [[ "$path"/ == "$mount_dest"/* ]] || [[ "$path" == "$mount_dest" ]]; then
                local mount_length=${#mount_dest}
                if [ $mount_length -gt $best_length ]; then
                    best_match="$volume_name"
                    best_mount_point="$mount_dest"
                    best_length=$mount_length
                fi
            fi
        done

        # Return both volume name and mount point (separated by |)
        if [ -n "$best_match" ]; then
            echo "$best_match|$best_mount_point"
        fi
    fi
}

# Function to setup volume mounts (handles both bind mounts and named volume subdirectories)
# Usage: setup_volume_mounts path1 path2 path3 ...
setup_volume_mounts() {
    local volume_args=()

    # Process each path argument
    for path in "$@"; do
        # Create directory in case it doesn't exist yet (needed for both bind mounts and volume subpaths)
        mkdir -p "$path"

        # Check if path is on a named volume to decide the type of mount to use
        local volume_info=$(get_volume_info_for_path "$path")
        if [ -n "$volume_info" ]; then
            # Parse: volume_name|mount_point
            local volume_name="${volume_info%|*}" # Everything before |
            local mount_point="${volume_info#*|}" # Everything after |

            if [ "$path" != "$mount_point" ]; then
                # Calculate subpath: remove mount point prefix and leading slash
                local volume_subpath="${path#$mount_point/}"
                echo "Path $path is on named Docker volume $volume_name - mounting volume subdirectory" >&2
                volume_args+=("--mount" "type=volume,src=$volume_name,dst=$path,volume-subpath=$volume_subpath")
            else
                # If no subpath (path is exactly the mount point), mount the entire volume
                echo "Path $path matches the root of named Docker volume $volume_name - mounting entire volume" >&2
                volume_args+=("--mount" "type=volume,src=$volume_name,dst=$path")
            fi
        else
            echo "Path $path is not on a named Docker volume - using bind mount" >&2
            volume_args+=("--volume" "$path:$path")
        fi
    done

    echo "${volume_args[@]}"
}

