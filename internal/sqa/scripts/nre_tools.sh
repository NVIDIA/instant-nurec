#!/usr/bin/bash

set -e

# Source utilities
source "$(dirname "$0")/utils.sh"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Usage function
function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>                Docker image tag [Taken from release instructions]"
    echo "  --runfiles <PATH>          Path to runfiles executable"
    echo "  --bazel                    Use bazel run //apps:nre_tools"
    echo ""
    echo "Required arguments:"
    echo "  --dataset-path <PATH>      Path to .zarr.itar file [Taken from release instructions]"
    echo "  --output-dir <PATH>        Output directory"
    echo "  --camera-ids <IDS>         Comma-separated list of camera IDs [Taken from release instructions]"
    echo "  --filename <STR>           String that defines filename where timings are saved. ONLY the name, it's saved in output directory, eg timings.txt"
    echo ""
    echo "Optional arguments:"
    echo "  --no-obfuscated <BOOL>     Run on non-obfuscated image. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <STR>             Suffix for the Docker image name (default: none)"
    echo ""
    echo "Examples:"
    echo ""
    echo "  Docker mode:"
    echo "    $0 --tag \"X.X.XXX-SHAXXXXX\" \\"
    echo "       --dataset-path \"path/to/dataset/as/described/in/release/instructions.zarr.itar\" \\"
    echo "       --output-dir \"path/to/some/dir/for/release_XXXX\" \\"
    echo "       --camera-ids \"camera_cross_right_120fov,camera_cross_left_120fov,camera_front_wide_120fov,camera_front_tele_30fov\" \\"
    echo "       --filename \"timings.txt\""
    echo ""
    echo "  Runfiles mode:"
    echo "    $0 --runfiles \"/path/to/resolved/executable\" \\"
    echo "       --dataset-path \"path/to/dataset/as/described/in/release/instructions.zarr.itar\" \\"
    echo "       --output-dir \"path/to/some/dir/for/release_XXXX\" \\"
    echo "       --camera-ids \"camera_cross_right_120fov,camera_cross_left_120fov,camera_front_wide_120fov,camera_front_tele_30fov\" \\"
    echo "       --filename \"timings.txt\""
    echo ""
}

# Convert elapsed time to HH:MM:SS.ms format
function convert_time() {
    local elapsed_time=$1
    local milliseconds=$((elapsed_time % 1000))
    local seconds=$((elapsed_time / 1000 % 60))
    local minutes=$((elapsed_time / 60000 % 60))
    local hours=$((elapsed_time / 3600000))
    local formatted_time=$(printf "%02d:%02d:%02d.%03d" $hours $minutes $seconds $milliseconds)
    echo "$formatted_time"
}

function safe_print() {
    # Replace the pattern "KEY=value" with "KEY=******"
    # This handles both -e KEY=value and KEY=value formats
    local CMD_STR="$1"
    local KEY_TO_HIDE="$2"
    
    local SANITIZED_STR=$(echo "$CMD_STR" | sed "s/${KEY_TO_HIDE}=[^ \"']*/${KEY_TO_HIDE}=******/g")
    
    # Also handle the case where the key might be in quotes
    SANITIZED_STR=$(echo "$SANITIZED_STR" | sed "s/${KEY_TO_HIDE}=\"[^\"]*\"/${KEY_TO_HIDE}=\"******\"/g")
    SANITIZED_STR=$(echo "$SANITIZED_STR" | sed "s/${KEY_TO_HIDE}='[^']*'/${KEY_TO_HIDE}='******'/g")
    
    echo "$SANITIZED_STR"
}

# Parse input arguments
TAG=""
RUNFILES_EXECUTABLE=""
DATASET_PATH=""
OUTPUT_DIR=""
CAMERA_IDS=""
FILENAME=""
BAZEL=false
NO_OBFUSCATED=false
SUFFIX=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --tag)
        if [ -n "$TAG" ]; then
            echo "Error: --tag is already set."
            usage
            exit 1
        fi
        TAG="$2"
        shift
        ;;
    --runfiles)
        RUNFILES_EXECUTABLE="$2"
        shift
        ;;
    --dataset-path)
        if [ -n "$DATASET_PATH" ]; then
            echo "Error: --dataset-path is already set."
            usage
            exit 1
        fi
        DATASET_PATH="$2"
        shift
        ;;
    --output-dir)
        if [ -n "$OUTPUT_DIR" ]; then
            echo "Error: --output-dir is already set."
            usage
            exit 1
        fi
        OUTPUT_DIR="$2"
        shift
        ;;
    --camera-ids)
        if [ -n "$CAMERA_IDS" ]; then
            echo "Error: --camera-ids is already set."
            usage
            exit 1
        fi
        CAMERA_IDS="$2"
        shift
        ;;
    --filename)
        if [ -n "$FILENAME" ]; then
            echo "Error: --filename is already set."
            usage
            exit 1
        fi
        FILENAME="$2"
        shift
        ;;
    --bazel)
        BAZEL=true
        ;;
    --no-obfuscated)
        NO_OBFUSCATED=true
        ;;
    --suffix)
        SUFFIX="$2"
        shift
        ;;
    --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown parameter passed: $1"
        usage
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
    usage
    exit 1
elif [[ $execution_modes -gt 1 ]]; then
    echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
    usage
    exit 1
fi

# Validate required arguments
if [[ -z "$DATASET_PATH" || -z "$OUTPUT_DIR" || -z "$CAMERA_IDS" || -z "$FILENAME" ]]; then
    echo "Error: --dataset-path, --output-dir, --camera-ids, and --filename are required."
    usage
    exit 1
else
    echo "TAG=${TAG}"
    echo "  DATASET_PATH=${DATASET_PATH}"
    echo "  OUTPUT_DIR=${OUTPUT_DIR}"
    echo "  CAMERA_IDS=${CAMERA_IDS}"
    echo "  FILENAME=${FILENAME}"
    echo "  BAZEL=${BAZEL}"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  SUFFIX=${SUFFIX}"
fi
# Dataset directory
DATASET_DIR=$(dirname "${DATASET_PATH}")

# Setup Docker arguments only if we're using Docker (TAG is set)
if [[ -n "$TAG" ]]; then
    # Get volume mount arguments
    VOLUME_MOUNTS=($(setup_volume_mounts "$DATASET_DIR" "$OUTPUT_DIR"))

    DOCKER_ARGS=(
        --shm-size=2g
        -it
        --user $(id -u):$(id -g)
        --rm
        --gpus all
        "${VOLUME_MOUNTS[@]}"
    )
fi

# Docker image information
NGC_IMAGES=nvcr.io/nvidian/ct-toronto-ai
OBFUSCATED="_obfuscated"
if [[ "$NO_OBFUSCATED" == true ]]; then
    OBFUSCATED=""
fi
SUFFIX_PART=""
if [[ -n "$SUFFIX" ]]; then
    SUFFIX_PART="_${SUFFIX}"
fi
NRE_TOOLS_IMAGE_NAME="nre${OBFUSCATED}_tools${SUFFIX_PART}"
NRE_IMAGE="${NGC_IMAGES}/${NRE_TOOLS_IMAGE_NAME}:${TAG}"

CMD_NRE_TOOLS=(
    --shard-file-pattern=${DATASET_PATH}
    --output-dir=$OUTPUT_DIR
    $(echo $CAMERA_IDS | tr ',' '\n' | xargs -n1 printf "--camera-id %s ")
    --store-meta
    --no-seg-logits
    --lidar-seg-camvis
)

echo "Running nre-tools..."
echo "  Mounting dataset directory: $DATASET_DIR"
echo "  Mounting output directory: $OUTPUT_DIR"
echo "  NRE-Tools command: ${CMD_NRE_TOOLS[@]}"

if [[ "$BAZEL" == true ]]; then
    FINAL_CMD_NRE_TOOLS="bazel run //apps:nre_tools -- ${CMD_NRE_TOOLS[@]}"
elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
    FINAL_CMD_NRE_TOOLS="${RUNFILES_EXECUTABLE} ${CMD_NRE_TOOLS[@]}"
else
    FINAL_CMD_NRE_TOOLS="docker run ${DOCKER_ARGS[@]} ${NRE_IMAGE} ${CMD_NRE_TOOLS[@]}"
fi
        
# Measure elapsed time for final command
echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_NRE_TOOLS}" "NGC_API_KEY")"
start_time=$(date +%s%3N)
eval ${FINAL_CMD_NRE_TOOLS}
end_time=$(date +%s%3N)

elapsed_time=$((end_time - start_time))
formatted_time=$(convert_time $elapsed_time)
echo "  Elapsed time for NRE-Tools generation: ${formatted_time}"
echo "[${TIMESTAMP}] Elapsed time for NRE-Tools generation: ${formatted_time}" >>${OUTPUT_DIR}/${FILENAME}
echo "  Wrote file: ${OUTPUT_DIR}/${FILENAME}"
