#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -e

# Source utilities
source "$(dirname "$0")/utils.sh"

# Global usage function
function usage_global() {
    echo "Usage: $0 <command> [OPTIONS]"
    echo "Available commands:"
    echo "  eval   Export ground truth and evaluate rendering metrics"
}

function usage_eval() {
    echo "Evaluates rendering metrics between eval images and reference images."
    echo "Usage: $0 eval [OPTIONS]"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>               Docker image tag"
    echo "  --runfiles <PATH>         Path to runfiles executable"
    echo "  --bazel                   Use bazel run"
    echo ""
    echo "Required arguments:"
    echo "  --reference-dir <PATH>       Path to reference images directory"
    echo "  --eval-images-dir <PATH>     Path to directory containing images to evaluate"
    echo "  --output-dir <PATH>          Path to output directory"
    echo ""
    echo "Ego masks (exactly one required):"
    echo "  --egocar-hood-dir <PATH>     Path to ego-hood images"
    echo "  --shard-file-pattern <PATH>  Path to .zarr.itar file to export ego masks from dataset"
    echo ""
    echo "Optional arguments:"
    echo "  --gif-tool <PATH>         Path to GIF creation tool (falls back to ImageMagick 'convert' if not specified)"
    echo "  --no-obfuscated           Run on non-obfuscated image. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <SUFFIX>         Docker image suffix, ex. 'grpc' (default: none)"
    echo "  --help                    Show this help message"
}

function safe_print() {
    # Replace the pattern "KEY=value" with "KEY=******"
    local CMD_STR="$1"
    local KEY_TO_HIDE="$2"

    local SANITIZED_STR=$(echo "$CMD_STR" | sed "s/${KEY_TO_HIDE}=[^ \"']*/${KEY_TO_HIDE}=******/g")
    SANITIZED_STR=$(echo "$SANITIZED_STR" | sed "s/${KEY_TO_HIDE}=\"[^\"]*\"/${KEY_TO_HIDE}=\"******\"/g")
    SANITIZED_STR=$(echo "$SANITIZED_STR" | sed "s/${KEY_TO_HIDE}='[^']*'/${KEY_TO_HIDE}='******'/g")

    echo "$SANITIZED_STR"
}

# Checks the NGC_API_KEY env var and gets it from the netrc file if not set
function check_and_get_ngc_api_key() {
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
}

# Check if running in Windows Subsystem for Linux (WSL)
function is_wsl() {
    if [[ -f /proc/version ]] && grep -q Microsoft /proc/version || [[ -d /usr/lib/wsl ]]; then
        return 0
    fi
    return 1
}

setup_docker_args() {
    local tag=$1
    local ngc_api_key=$2
    local no_obfuscated=$3
    local suffix=$4
    local mount_volumes=("${@:5}")

    local docker_envvars=("-e NGC_API_KEY=${ngc_api_key}")

    local obfuscated="_obfuscated"
    if [[ "$no_obfuscated" == true ]]; then
        obfuscated=""
    fi
    local suffix_part=""
    if [[ -n "$suffix" ]]; then
        suffix_part="_${suffix}"
    fi
    local image_name="nre${obfuscated}_run${suffix_part}"
    local nre_image="nvcr.io/nvidian/ct-toronto-ai/${image_name}:${tag}"

    echo "Preparing container launch:"
    echo "  Docker image: ${nre_image}"
    echo "  Forwarding NGC_API_KEY to the container"
    echo "  Mount volumes:"
    for volume in "${mount_volumes[@]}"; do
        echo "    $volume"
    done

    local volume_mounts=($(setup_volume_mounts "${mount_volumes[@]}"))

    if is_wsl; then
        docker_envvars+=("-e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so")
    fi

    DOCKER_ARGS=(
        -it
        --user $(id -u):$(id -g)
        --rm
        --gpus all
        --shm-size=2g
        --net=host
        --privileged
        ${docker_envvars[@]}
        "${volume_mounts[@]}"
        "${nre_image}"
    )
}


# Check if at least one argument is provided
if [ "$#" -le 0 ]; then
    usage_global
    exit 1
fi

command=$1
shift

case $command in
eval)
    TAG=""
    RUNFILES_EXECUTABLE=""
    SHARD_FILE_PATTERN=""
    EGOCAR_HOOD_DIR=""
    REFERENCE_DIR=""
    EVAL_IMAGES_DIR=""
    OUTPUT_DIR=""
    GIF_TOOL=""
    NO_OBFUSCATED=false
    BAZEL=false
    SUFFIX=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --tag)
            if [ -n "$TAG" ]; then
                echo "Error: --tag is already set."
                usage_eval
                exit 1
            fi
            TAG="$2"
            shift
            ;;
        --runfiles)
            RUNFILES_EXECUTABLE="$2"
            shift
            ;;
        --shard-file-pattern)
            if [ -n "$SHARD_FILE_PATTERN" ]; then
                echo "Error: --shard-file-pattern is already set."
                usage_eval
                exit 1
            fi
            SHARD_FILE_PATTERN="$2"
            shift
            ;;
        --egocar-hood-dir)
            if [ -n "$EGOCAR_HOOD_DIR" ]; then
                echo "Error: --egocar-hood-dir is already set."
                usage_eval
                exit 1
            fi
            EGOCAR_HOOD_DIR="$2"
            shift
            ;;
        --reference-dir)
            if [ -n "$REFERENCE_DIR" ]; then
                echo "Error: --reference-dir is already set."
                usage_eval
                exit 1
            fi
            REFERENCE_DIR="$2"
            shift
            ;;
        --eval-images-dir)
            if [ -n "$EVAL_IMAGES_DIR" ]; then
                echo "Error: --eval-images-dir is already set."
                usage_eval
                exit 1
            fi
            EVAL_IMAGES_DIR="$2"
            shift
            ;;
        --output-dir)
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Error: --output-dir is already set."
                usage_eval
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --gif-tool)
            GIF_TOOL="$2"
            shift
            ;;
        --no-obfuscated)
            NO_OBFUSCATED=true
            ;;
        --bazel)
            BAZEL=true
            ;;
        --suffix)
            SUFFIX="$2"
            shift
            ;;
        --help)
            usage_eval
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_eval
            exit 1
            ;;
        esac
        shift
    done

    # Validate mutually exclusive execution modes
    execution_modes=0
    if [[ -n "$TAG" ]]; then
        execution_modes=$((execution_modes + 1))
    fi
    if [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        execution_modes=$((execution_modes + 1))
    fi
    if [[ "$BAZEL" == true ]]; then
        execution_modes=$((execution_modes + 1))
    fi

    if [[ $execution_modes -eq 0 ]]; then
        echo "Error: Exactly one of --tag, --runfiles, or --bazel must be provided."
        usage_eval
        exit 1
    elif [[ $execution_modes -gt 1 ]]; then
        echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
        usage_eval
        exit 1
    fi

    # Validate required arguments
    if [[ -z "$REFERENCE_DIR" ]]; then
        echo "Error: --reference-dir is required."
        usage_eval
        exit 1
    fi

    if [ -z "$EVAL_IMAGES_DIR" ]; then
        echo "Error: --eval-images-dir is required."
        usage_eval
        exit 1
    fi

    if [ -z "$OUTPUT_DIR" ]; then
        echo "Error: --output-dir is required."
        usage_eval
        exit 1
    fi

    # Ego masks: exactly one of --egocar-hood-dir or --shard-file-pattern
    if [[ -n "$EGOCAR_HOOD_DIR" ]] && [[ -n "$SHARD_FILE_PATTERN" ]]; then
        echo "Error: Only one of --egocar-hood-dir or --shard-file-pattern may be provided."
        usage_eval
        exit 1
    fi
    if [[ -z "$EGOCAR_HOOD_DIR" ]] && [[ -z "$SHARD_FILE_PATTERN" ]]; then
        echo "Error: One of --egocar-hood-dir or --shard-file-pattern is required."
        usage_eval
        exit 1
    fi

    echo "Evaluating rendering metrics with the following parameters:"
    echo "  TAG: $TAG"
    echo "  REFERENCE_DIR: $REFERENCE_DIR"
    echo "  EVAL_IMAGES_DIR: $EVAL_IMAGES_DIR"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    if [[ -n "$EGOCAR_HOOD_DIR" ]]; then
        echo "  EGOCAR_HOOD_DIR: $EGOCAR_HOOD_DIR"
    else
        echo "  SHARD_FILE_PATTERN: $SHARD_FILE_PATTERN (for ego mask export)"
    fi
    echo "  NO_OBFUSCATED: ${NO_OBFUSCATED}"
    echo "  BAZEL: ${BAZEL}"
    echo "  SUFFIX: ${SUFFIX}"
    echo ""

    # Defines NGC_API_KEY if not set
    check_and_get_ngc_api_key

    # Set bazel run target based on obfuscation flag
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    # Validate directories exist
    if [[ ! -d "${EVAL_IMAGES_DIR}" ]]; then
        echo "Error: Eval images directory not found: ${EVAL_IMAGES_DIR}"
        exit 1
    fi

    if [[ ! -d "${REFERENCE_DIR}" ]]; then
        echo "Error: Reference directory not found: ${REFERENCE_DIR}"
        exit 1
    fi

    if [[ -n "$EGOCAR_HOOD_DIR" ]] && [[ ! -d "${EGOCAR_HOOD_DIR}" ]]; then
        echo "Error: Egocar hood directory not found: ${EGOCAR_HOOD_DIR}"
        exit 1
    fi

    # Get image sequence names from eval images directory
    EVAL_IMAGE_SEQS=($(find "${EVAL_IMAGES_DIR}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))

    if [[ ${#EVAL_IMAGE_SEQS[@]} -eq 0 ]]; then
        echo "Error: No image sequence folders found in eval images directory: ${EVAL_IMAGES_DIR}"
        exit 1
    fi

    # Step 1: Ego masks for eval — use --egocar-hood-dir or export from dataset via --shard-file-pattern
    if [[ -n "$EGOCAR_HOOD_DIR" ]]; then
        echo "Step 1: Using ego masks from --egocar-hood-dir"
        if [[ -d "${EGOCAR_HOOD_DIR}/default" ]]; then
            # WAR bug 5555788: grpc preprocess uses a rig subfolder 'default'
            EGOCAR_HOOD_DIR="${EGOCAR_HOOD_DIR}/default"
        fi
    else
        echo "Step 1: Exporting ego masks from dataset..."

        CMD_EXPORT_EGO_MASK=(
            export-ego-mask
            --shard-file-pattern "${SHARD_FILE_PATTERN}"
            --output-dir "${OUTPUT_DIR}" # The call will create a subfolder 'ego-hoods'
            --camera-frame-idx 10
        )

        if [[ "$BAZEL" == true ]]; then
            CMD_EXPORT_EGO_MASK_FULL="bazel run ${BAZEL_RUN} -- ${CMD_EXPORT_EGO_MASK[@]}"
        elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
            CMD_EXPORT_EGO_MASK_FULL="${RUNFILES_EXECUTABLE} ${CMD_EXPORT_EGO_MASK[@]}"
        else
            SHARD_DIR=$(dirname "${SHARD_FILE_PATTERN}")
            MOUNT_VOLUMES_EGO=(
                "$SHARD_DIR"
                "$OUTPUT_DIR"
            )
            setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES_EGO[@]}"
            CMD_EXPORT_EGO_MASK_FULL="docker run ${DOCKER_ARGS[@]} ${CMD_EXPORT_EGO_MASK[@]}"
        fi

        echo "EXECUTE COMMAND: $(safe_print "${CMD_EXPORT_EGO_MASK_FULL}" "NGC_API_KEY")"
        eval ${CMD_EXPORT_EGO_MASK_FULL}
        EGOCAR_HOOD_DIR="${OUTPUT_DIR}/ego-hoods"
    fi

    # Step 2: Create reference structure by copying files
    # eval-rendering-metrics expects:
    #   <gt-dir>/camera_images/<image-seq>/<frame-timestamp>.<ext>
    #   <gt-dir>/camera_ego_masks/<image-seq>.png
    # Our reference dir only has: <image-seq>/<frame-timestamp>.<ext>
    # Image sequence names may differ between eval and reference (e.g., camera_front_wide_120fov vs cam_00)
    echo ""
    echo "Step 2: Creating reference structure..."

    REFERENCE_PREPARED_DIR="${OUTPUT_DIR}/reference"
    rm -rf "${REFERENCE_PREPARED_DIR}"
    mkdir -p "${REFERENCE_PREPARED_DIR}/camera_images"
    mkdir -p "${REFERENCE_PREPARED_DIR}/camera_ego_masks"

    # Get reference image sequence names (may differ from eval image sequence names)
    REF_IMAGE_SEQS=($(find "${REFERENCE_DIR}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))

    if [[ ${#REF_IMAGE_SEQS[@]} -eq 0 ]]; then
        echo "Error: No image sequence folders found in reference directory: ${REFERENCE_DIR}"
        exit 1
    fi

    # Determine if reference sequences use indexed names (e.g. cam_00) or actual camera names.
    # When reference uses indexed names (legacy archived artifacts, until 26.03), we need to fall back to
    # an alphabetical remapping of camera and frame names.
    USE_DIRECT_REF_MATCH=true
    for _ref_seq in "${REF_IMAGE_SEQS[@]}"; do
        if [[ "$_ref_seq" =~ ^cam_[0-9]+$ ]]; then
            USE_DIRECT_REF_MATCH=false
            break
        fi
    done

    if [[ "$USE_DIRECT_REF_MATCH" == true ]]; then
        echo "  Using direct name-based matching for reference image sequences"
    else
        echo "  Warning: Reference image sequences use indexed names (cam_NN) — falling back to alphabetical-order remapping"
        if [[ ${#EVAL_IMAGE_SEQS[@]} -ne ${#REF_IMAGE_SEQS[@]} ]]; then
            # With the fallback remapping approach, we can only handle a one-to-one mapping of image sequences.
            echo "Error: Number of image sequences in eval (${#EVAL_IMAGE_SEQS[@]}) does not match reference (${#REF_IMAGE_SEQS[@]})"
            echo "  Eval image sequences: ${EVAL_IMAGE_SEQS[*]}"
            echo "  Reference image sequences: ${REF_IMAGE_SEQS[*]}"
            exit 1
        fi
    fi

    for i in "${!EVAL_IMAGE_SEQS[@]}"; do
        eval_seq="${EVAL_IMAGE_SEQS[$i]}"

        if [[ "$USE_DIRECT_REF_MATCH" == true ]]; then
            ref_seq="$eval_seq"
        else
            # Fallback remap of image sequences by index
            # We assume that image sequence names in each folder map to each other in alphabetical order.
            # This is temporary and does not generalize to all possible use cases.
            ref_seq="${REF_IMAGE_SEQS[$i]}"
        fi
        # Ego masks are always named by camera name and match eval sequences directly.
        ego_mask_seq="$eval_seq"

        echo "  Mapping eval '${eval_seq}' to reference '${ref_seq}', ego mask '${ego_mask_seq}'"

        # Copy reference images under camera_images/<eval_seq>/
        ref_seq_dir="${REFERENCE_DIR}/${ref_seq}"
        if [[ -d "${ref_seq_dir}" ]]; then
            dest_dir="${REFERENCE_PREPARED_DIR}/camera_images/${eval_seq}"
            mkdir -p "${dest_dir}"

            if [[ "$USE_DIRECT_REF_MATCH" == false ]]; then
                # Reference images use sequential indices; eval images use timestamps.
                # Copy reference images to dest_dir, renaming each to match the
                # corresponding eval filename in alphabetical order, so that
                # eval-rendering-metrics can pair them correctly.
                #
                # The reference and eval sets may have different frame counts (ex. different
                # frame steps used in val and rendering). We derive the remapping step
                # from the frame count ratio.
                mapfile -t eval_imgs < <(find "${EVAL_IMAGES_DIR}/${eval_seq}" -maxdepth 1 -name "*.png" | sort)
                mapfile -t ref_imgs  < <(find "${ref_seq_dir}"                 -maxdepth 1 -name "*.png" | sort)

                eval_count=${#eval_imgs[@]}
                ref_count=${#ref_imgs[@]}

                if (( ref_count % eval_count != 0 )); then
                    echo "Error: Reference frame count (${ref_count}) is not a multiple of \
eval frame count (${eval_count}) for sequence '${eval_seq}'"
                    exit 1
                fi

                ref_frame_step=$(( ref_count / eval_count ))

                for (( j=0; j<eval_count; j++ )); do
                    ref_img_idx=$(( j * ref_frame_step ))
                    eval_filename=$(basename "${eval_imgs[$j]}")
                    cp "${ref_imgs[$ref_img_idx]}" "${dest_dir}/${eval_filename}"
                done

                echo "    Copied ${eval_count} reference images (step=${ref_frame_step}) to match eval filenames"
            else
                cp "${ref_seq_dir}"/* "${dest_dir}/"
            fi

            file_count=$(find "${dest_dir}" -maxdepth 1 -type f | wc -l)
            echo "    Copied ${file_count} images to camera_images/${eval_seq}/"
        else
            echo "Error: Reference image sequence directory not found: ${ref_seq_dir}"
            exit 1
        fi

        # Copy ego mask under camera_ego_masks/<eval_seq>.png (using ego_mask_seq from dataset)
        ego_mask_file="${EGOCAR_HOOD_DIR}/${ego_mask_seq}.png"
        if [[ -f "${ego_mask_file}" ]]; then
            cp "${ego_mask_file}" "${REFERENCE_PREPARED_DIR}/camera_ego_masks/${eval_seq}.png"
            echo "    Copied ego mask to camera_ego_masks/${eval_seq}.png"
        else
            echo "Error: Ego mask not found: ${ego_mask_file}"
            exit 1
        fi
    done

    echo "  Reference structure created at: ${REFERENCE_PREPARED_DIR}"

    # Step 3: Evaluate rendering metrics
    echo ""
    echo "Step 3: Evaluating rendering metrics..."

    CMD_EVAL_METRICS=(
        eval-rendering-metrics
        --render-dir "${EVAL_IMAGES_DIR}"
        --gt-dir "${REFERENCE_PREPARED_DIR}"
        --output-dir "${OUTPUT_DIR}"
        --metrics psnr
        --metrics ssim
        --save-yaml
        --visualize
    )

    if [[ "$BAZEL" == true ]]; then
        CMD_EVAL_METRICS_FULL="bazel run ${BAZEL_RUN} -- ${CMD_EVAL_METRICS[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        CMD_EVAL_METRICS_FULL="${RUNFILES_EXECUTABLE} ${CMD_EVAL_METRICS[@]}"
    else
        MOUNT_VOLUMES_EVAL=(
            "$EVAL_IMAGES_DIR"
            "$REFERENCE_DIR"
            "$OUTPUT_DIR"
        )
        setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES_EVAL[@]}"
        CMD_EVAL_METRICS_FULL="docker run ${DOCKER_ARGS[@]} ${CMD_EVAL_METRICS[@]}"
    fi

    echo "EXECUTE COMMAND: $(safe_print "${CMD_EVAL_METRICS_FULL}" "NGC_API_KEY")"
    eval ${CMD_EVAL_METRICS_FULL}

    # Step 4: Create comparison GIFs
    echo ""
    echo "Step 4: Creating comparison GIFs (eval vs reference)..."

    # Check for GIF creation tool
    if [[ -n "$GIF_TOOL" ]] && [[ -x "$GIF_TOOL" ]]; then
        echo "  Using GIF creation tool: $GIF_TOOL"
        USE_GIF_TOOL=true
    else
        echo "  Using ImageMagick 'convert' for GIF creation"
        USE_GIF_TOOL=false
    fi

    gif_count=0
    for eval_seq in "${EVAL_IMAGE_SEQS[@]}"; do
        echo "  Processing image sequence: ${eval_seq}"
        mkdir -p "${OUTPUT_DIR}/${eval_seq}"

        # Get eval images for this image sequence
        for eval_img in "${EVAL_IMAGES_DIR}/${eval_seq}"/*.png; do
            [[ -f "$eval_img" ]] || continue

            img_filename=$(basename "$eval_img")
            img_basename="${img_filename%.*}"

            # Find corresponding reference image in the prepared reference structure (with matching folders and files)
            ref_img="${REFERENCE_PREPARED_DIR}/camera_images/${eval_seq}/${img_filename}"
            if [[ ! -f "$ref_img" ]]; then
                echo "    Warning: Reference image not found for ${img_filename}, skipping"
                continue
            fi

            out_gif="${OUTPUT_DIR}/${eval_seq}/${img_basename}_eval_vs_ref.gif"

            if [[ "$USE_GIF_TOOL" == true ]]; then
                "${GIF_TOOL}" "${eval_img}" "${ref_img}" "${out_gif}" --delay 500 --loop 0
            else
                convert -delay 50 -loop 0 "${eval_img}" "${ref_img}" "${out_gif}"
            fi

            gif_count=$((gif_count + 1))
        done
    done

    echo "  Created ${gif_count} comparison GIFs in: ${OUTPUT_DIR}"

    echo ""
    echo "Done. Results saved to: ${OUTPUT_DIR}"
    ;;
--help)
    usage_global
    exit 0
    ;;
*)
    echo "Invalid command: $command"
    usage_global
    exit 1
    ;;
esac
