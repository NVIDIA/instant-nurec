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

# Usage: ./image_analysis.sh <image:tag>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image:tag>"
    echo "Example: $0 nvcr.io/nvidian/ct-toronto-ai/nre_run:latest"
    exit 1
fi

IMAGE_REF="$1"

# Function to check if a command exists
check_command() {
    local cmd="$1"
    local install_instructions="$2"
    
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: '$cmd' is not installed or not in PATH"
        echo ""
        echo "Installation instructions:"
        echo "  $install_instructions"
        echo ""
        exit 1
    fi
}

# Function to check if image exists locally
check_local_image() {
    if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
        echo "Image '$IMAGE_REF' not found locally. Pulling from registry..."
        if ! docker pull "$IMAGE_REF"; then
            echo "Error: Failed to pull image '$IMAGE_REF'"
            exit 1
        fi
    fi
}

# Function to get registry manifest
get_registry_manifest() {
    docker manifest inspect "$IMAGE_REF" 2>/dev/null || {
        echo "Error: Could not fetch registry manifest for '$IMAGE_REF'"
        echo "This might be a local-only image or you may not have access to the registry"
        exit 1
    }
}

# Function to format bytes to human readable using numfmt
format_bytes() {
    local bytes="$1"
    # Handle empty or invalid input
    if [[ -z "$bytes" ]] || [[ "$bytes" =~ [^0-9] ]]; then
        printf "N/A"
        return
    fi

    # Use numfmt for clean IEC formatting (KiB, MiB, GiB)
    numfmt --to=iec "$bytes" 2>/dev/null || printf "%d B" "$bytes"
}

# Main script
main() {
    echo "Mapping layers for: $IMAGE_REF"
    echo "=================================================="

    # Check dependencies
    echo "Checking dependencies..."
    check_command "docker" "Install Docker from https://docs.docker.com/get-docker/"
    check_command "dive" "Install Dive from https://github.com/wagoodman/dive#installation"
    check_command "jq" "Install jq with: sudo apt install jq"

    echo "Validating image and fetching metadata..."
    # Check if image exists locally
    check_local_image

    # Get registry manifest
    local manifest_json
    manifest_json=$(get_registry_manifest 2>/dev/null)

    # Create output directory for analysis with sanitized image name
    local sanitized_image_name
    sanitized_image_name=$(echo "$IMAGE_REF" | sed 's/[^a-zA-Z0-9._-]/_/g' | sed 's/__*/_/g' | sed 's/^_//' | sed 's/_$//')
    local output_dir="image_analysis_results/$sanitized_image_name"
    mkdir -p "$output_dir"

    # Create temporary files for processing
    local temp_dir
    temp_dir=$(mktemp -d)
    # Cleanup function
    cleanup() {
        if [ -n "${temp_dir:-}" ] && [ -d "${temp_dir:-}" ]; then
            rm -rf "${temp_dir:-}" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    # Extract registry layers
    echo "$manifest_json" | jq -r '.layers[] | "\(.digest)|\(.size)"' > "$temp_dir/registry_layers"

    # Get dive analysis for detailed layer information
    local dive_json
    local dive_file
    dive_file="/tmp/dive_${sanitized_image_name}.json"

    # Check if dive file already exists and ask user
    if [ -f "$dive_file" ]; then
        echo "Dive analysis file already exists: $dive_file"
        read -p "Do you want to refresh it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Using existing dive analysis file."
        else
            echo "Refreshing dive analysis..."
            dive --json "$dive_file" "$IMAGE_REF" >/dev/null 2>&1
        fi
    else
        echo "Running dive analysis..."
        dive --json "$dive_file" "$IMAGE_REF" >/dev/null 2>&1
    fi
    dive_json=$(cat "$dive_file")

    echo "Processing layer information..."
    # Extract dive layer information (digest|size|command)
    echo "$dive_json" | jq -r '.layer[] | "\(.digestId)|\(.sizeBytes)|\(.command)"' > "$temp_dir/dive_layers"

    # Extract file lists for each layer and store them
    local layer_index=0
    local total_layers
    total_layers=$(wc -l < "$temp_dir/dive_layers")
    echo "Processing $total_layers layers in parallel..."

    # Process layers in parallel using subshells
    while IFS='|' read -r digest size command; do
        if [ -n "$digest" ]; then
            local current_index=$layer_index
            local file_list_file="$output_dir/files_layer_$(printf "%02d" $current_index).txt"

            # Process layer in background subshell
            (
                # Extract file list for this layer, filtering out entries with empty paths
                echo "$dive_json" | jq -r ".layer[$current_index].fileList[] | select(.path != null and .path != \"\") | \"\(.path)|\(.size // 0)|\(.typeFlag // 0)\"" > "$file_list_file"

                # Add header to file list
                {
                    echo "# Layer $current_index: $digest"
                    echo "# Command: $command"
                    echo "# Size: $(format_bytes "$size")"
                    echo "# File count: $(wc -l < "$file_list_file")"
                    echo "# Format: filename|size|type"
                    echo ""
                    cat "$file_list_file"
                } > "$file_list_file.tmp" && mv "$file_list_file.tmp" "$file_list_file"

                # Store result for sequential reporting
                echo "$current_index|$(format_bytes "$size")|$(wc -l < "$file_list_file")" > "$temp_dir/layer_${current_index}_result"
            ) &

            layer_index=$((layer_index + 1))
        fi
    done < "$temp_dir/dive_layers"

    # Wait for all background processes to complete
    wait

    echo "Generating analysis summary..."
    # Count layers
    local registry_layer_count
    local dive_layer_count
    local total_image_size=0
    local total_file_count=0
    registry_layer_count=$(wc -l < "$temp_dir/registry_layers")
    dive_layer_count=$(wc -l < "$temp_dir/dive_layers")

    # Calculate total image size and file count by summing local layer sizes
    while IFS='|' read -r digest size command; do
        if [ -n "$size" ] && [[ "$size" =~ ^[0-9]+$ ]]; then
            total_image_size=$((total_image_size + size))
        fi
    done < "$temp_dir/dive_layers"

    # Calculate total file count from all layer files
    for i in $(seq 0 $((dive_layer_count - 1))); do
        local layer_file="$output_dir/files_layer_$(printf "%02d" $i).txt"
        if [ -f "$layer_file" ]; then
            local file_count=$(wc -l < "$layer_file")
            # Subtract header lines (6 lines) to get actual file count
            file_count=$((file_count - 6))
            if [ $file_count -ge 0 ]; then
                total_file_count=$((total_file_count + file_count))
            fi
        fi
    done

    echo ""
    # Capture output to both stdout and file
    {
        # Print header for registry vs dive layers
        echo "=== Filesystem Layers ==="
        printf "%-10s | %-20s | %-15s | %-20s | %-15s | %-15s | %-50s\n" "Idx" "Registry Digest" "Registry Size" "Local Digest" "Local Size" "File Count" "Command"
        printf "%-10s-|-%-20s-|-%-15s-|-%-20s-|-%-15s-|-%-15s-|-%-50s\n" "----------" "--------------------" "---------------" "--------------------" "---------------" "---------------" "--------------------------------------------------"

        # Process each layer (registry vs dive)
        for i in $(seq 0 $((registry_layer_count - 1))); do
            local registry_line
            local dive_line

            registry_line=$(sed -n "$((i + 1))p" "$temp_dir/registry_layers")
            dive_line=$(sed -n "$((i + 1))p" "$temp_dir/dive_layers")

            # Parse registry line
            local registry_digest
            local registry_size
            IFS='|' read -r registry_digest registry_size <<< "$registry_line"

            # Parse dive line (digest|size|command)
            local dive_digest
            local dive_size
            local dive_command
            IFS='|' read -r dive_digest dive_size dive_command <<< "$dive_line"

            # Truncate digest for display (first 12 chars)
            local registry_digest_short
            local local_digest_short
            registry_digest_short="${registry_digest#sha256:}"
            registry_digest_short="${registry_digest_short:0:12}"
            local_digest_short="${dive_digest#sha256:}"
            local_digest_short="${local_digest_short:0:12}"

            # Format sizes
            local registry_size_formatted
            local local_size_formatted
            registry_size_formatted=$(format_bytes "$registry_size")
            local_size_formatted=$(format_bytes "$dive_size")

            # Truncate command for display
            local command_short
            command_short=$(echo "$dive_command" | cut -c1-50)
            if [ ${#dive_command} -gt 50 ]; then
                command_short="${command_short}..."
            fi

            # Print layer info
            printf "%-10s | %-20s | %-15s | %-20s | %-15s | %-15s | %-50s\n" \
                "$i" \
                "$registry_digest_short" \
                "$registry_size_formatted" \
                "$local_digest_short" \
                "$local_size_formatted" \
                "$(($(wc -l < "$output_dir/files_layer_$(printf "%02d" $i).txt") - 6))" \
                "$command_short"
        done

        echo ""
        echo "=== Other Blobs ==="
        printf "%-10s | %-20s | %-15s\n" "Name" "Registry Digest" "Registry Size"
        printf "%-10s-|-%-20s-|-%-15s\n" "----------" "--------------------" "---------------"
        local config_size=$(echo "$manifest_json" | jq -r '.config.size')
        local config_digest=$(echo "$manifest_json" | jq -r '.config.digest')
        local config_digest_short="${config_digest#sha256:}"
        config_digest_short="${config_digest_short:0:12}"
        printf "%-10s | %-20s | %-15s\n" "Config" "$config_digest_short" "$(format_bytes "$config_size")"

        echo ""
        echo "Summary:"
        echo "  Total image size: $(format_bytes "$total_image_size")"
        echo "  Total file count: $total_file_count"
        echo ""
        echo "Analysis stored in: $output_dir/"
        echo "  Summary: $output_dir/summary.txt"
        echo "  Layer files: $output_dir/files_layer_<index>.txt"

    } | tee "$output_dir/summary.txt"

    echo ""
    echo "Image analysis completed successfully!"
}

# Run main function
main "$@"