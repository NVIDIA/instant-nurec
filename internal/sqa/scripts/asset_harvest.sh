#!/usr/bin/bash

set -e

# Source utilities
source "$(dirname "$0")/utils.sh"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Usage function, please see here for file usage: https://jirasw.nvidia.com/browse/NRE-1327
function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>                 Docker image tag [Taken from release instructions]"
    echo "  --runfiles <PATH>           Path to runfiles executable"
    echo "  --bazel                     Use bazel run //apps/asset_harvester:asset_harvester"
    echo ""
    echo "Required arguments:"
    echo "  --component-store <PATH>    Path to V4 component store [Taken from release instructions]"
    echo "  --output-dir <PATH>         Output directory"
    echo ""
    echo "Optional arguments:"
    echo "  --track-ids <IDS>           Comma-separated list of track IDs to process (default: process all)"
    echo "  --cache-dir <PATH>          Directory for downloaded model files (default: ~/.cache/nre/)"
    echo "  --no-obfuscated <BOOL>      Run on non-obfuscated image. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --ngc-api-key <KEY>         NGC API Key (for models download, can also be set as an environment variable)"
    echo "  --extra-args <ARGS>         Additional arguments to append to the asset harvest command"
    echo "  --suffix <SUFFIX>           Suffix for the Docker image name (default: none)"
    echo ""
    echo "Examples:"
    echo ""
    echo "  Docker mode:"
    echo "    $0 --tag \"X.X.XXX-SHAXXXXX\" \\"
    echo "       --component-store \"path/to/input/shard/file\" \\"
    echo "       --output-dir \"path/to/some/dir/for/release_XXXX\" \\"
    echo "       --track-ids \"track1,track2,track3\" \\"
    echo "       --ngc-api-key \"your_ngc_api_key\" \\"
    echo "       --extra-args \"--some-flag --another-flag=value\""
    echo ""
    echo "  Runfiles mode:"
    echo "    $0 --runfiles \"/path/to/resolved/executable\" \\"
    echo "       --component-store \"path/to/input/shard/file\" \\"
    echo "       --output-dir \"path/to/some/dir/for/release_XXXX\" \\"
    echo "       --track-ids \"track1,track2,track3\" \\"
    echo "       --ngc-api-key \"your_ngc_api_key\" \\"
    echo "       --extra-args \"--some-flag --another-flag=value\""
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
SHARD_PATH=""
OUTPUT_DIR=""
TRACK_IDS=""
CACHE_DIR=""
BAZEL=false
NO_OBFUSCATED=false
EXTRA_ARGS=""
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
    --component-store)
        if [ -n "$SHARD_FILE_PATTERN" ]; then
            echo "Error: --component-store is already set."
            usage
            exit 1
        fi
        SHARD_FILE_PATTERN="$2"
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
    --track-ids)
        if [ -n "$TRACK_IDS" ]; then
            echo "Error: --track-ids is already set."
            usage
            exit 1
        fi
        TRACK_IDS="$2"
        shift
        ;;
    --cache-dir)
        if [ -n "$CACHE_DIR" ]; then
            echo "Error: --cache-dir is already set."
            usage
            exit 1
        fi
        CACHE_DIR="$2"
        shift
        ;;
    --bazel)
        BAZEL=true
        ;;
    --no-obfuscated)
        NO_OBFUSCATED=true
        ;;
    --ngc-api-key)
        if [ -n "$NGC_API_KEY" ]; then
            echo "Error: --ngc-api-key is already set."
            usage
            exit 1
        fi
        NGC_API_KEY="$2"
        shift
        ;;
    --extra-args)
        EXTRA_ARGS="$2"
        shift
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

# Check if NGC_API_KEY is set as an environment variable if not provided as an argument
if [[ -z "$NGC_API_KEY" ]]; then
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
        ' ~/.netrc) || {
            echo "Error: --ngc-api-key is required either as an argument, environment variable, or in ~/.netrc file."
            usage
            exit 1
        }
        echo "Found NGC_API_KEY from ~/.netrc"
    else
        NGC_API_KEY="${NGC_API_KEY}"
    fi
fi

# Expand ~ to the full home directory path for CACHE_DIR
if [[ -n "$CACHE_DIR" ]]; then
    CACHE_DIR="${CACHE_DIR/#\~/$HOME}"
fi

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
if [[ -z "$SHARD_FILE_PATTERN" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --component-store and --output-dir are required."
    usage
    exit 1
else
    echo "TAG=${TAG}"
    echo "  SHARD_FILE_PATTERN=${SHARD_FILE_PATTERN}"
    echo "  OUTPUT_DIR=${OUTPUT_DIR}"
    echo "  TRACK_IDS=${TRACK_IDS}"
    echo "  CACHE_DIR=${CACHE_DIR}"
    echo "  BAZEL=${BAZEL}"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  NGC_API_KEY=******"
    echo "  EXTRA_ARGS=${EXTRA_ARGS}"
    echo "  SUFFIX=${SUFFIX}"
fi

# Shard directory
SHARD_DIR=$(dirname "${SHARD_FILE_PATTERN}")

# Setup Docker arguments only if we're using Docker (TAG is set)
if [[ -n "$TAG" ]]; then
    # Get volume mount arguments
    VOLUME_MOUNTS=($(setup_volume_mounts "$SHARD_DIR" "$OUTPUT_DIR"))

    DOCKER_ARGS=(
        --shm-size=2g
        -it
        --user $(id -u):$(id -g)
        --rm
        --gpus all
        "${VOLUME_MOUNTS[@]}"
        --env NGC_API_KEY=${NGC_API_KEY}
    )
fi

# Add cache directory mount if specified
if [[ -n "$TAG" ]] && [[ -n "$CACHE_DIR" ]]; then
    CACHE_MOUNT=($(setup_volume_mounts "$CACHE_DIR"))
    DOCKER_ARGS+=("${CACHE_MOUNT[@]}")
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
NRE_ASSET_HARVEST_NAME="nre${OBFUSCATED}_tools${SUFFIX_PART}"
NRE_IMAGE="${NGC_IMAGES}/${NRE_ASSET_HARVEST_NAME}:${TAG}"

# Build asset harvest command
CMD_ASSET_HARVEST=(
    asset-harvester
    --component-store=${SHARD_FILE_PATTERN}
    --output-dir=${OUTPUT_DIR}
    ncore_parser.camera_ids=["camera_front_wide_120fov","camera_cross_right_120fov","camera_cross_left_120fov"]
)

# Add optional parameters if provided
if [[ -n "$TRACK_IDS" ]]; then
    CMD_ASSET_HARVEST+=(--track-ids=${TRACK_IDS})
fi

if [[ -n "$CACHE_DIR" ]]; then
    CMD_ASSET_HARVEST+=(--cache-dir=${CACHE_DIR})
fi

# Append extra arguments if provided
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD_ASSET_HARVEST+=($EXTRA_ARGS)
fi

# Create output directory now to allow bind-mounting it to the container
mkdir -p ${OUTPUT_DIR}

# Create cache directory if specified
if [[ -n "$CACHE_DIR" ]]; then
    mkdir -p ${CACHE_DIR}
fi

echo "Running asset-harvest..."
echo "  Mounting shard file pattern: $SHARD_FILE_PATTERN"
echo "  Mounting output directory: $OUTPUT_DIR"
if [[ -n "$CACHE_DIR" ]]; then
    echo "  Mounting cache directory: $CACHE_DIR"
fi
echo "  Asset-Harvest command: ${CMD_ASSET_HARVEST[@]}"

if [[ "$BAZEL" == true ]]; then
    FINAL_CMD_ASSET_HARVEST="bazel run //apps/asset_harvester:asset_harvester -- ${CMD_ASSET_HARVEST[@]}"
elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
    FINAL_CMD_ASSET_HARVEST="${RUNFILES_EXECUTABLE} ${CMD_ASSET_HARVEST[@]}"
else
    FINAL_CMD_ASSET_HARVEST="docker run ${DOCKER_ARGS[@]} ${NRE_IMAGE} ${CMD_ASSET_HARVEST[@]}"
fi
        
# Measure elapsed time for final command
echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_ASSET_HARVEST}" "NGC_API_KEY")"
start_time=$(date +%s%3N)
eval ${FINAL_CMD_ASSET_HARVEST}
end_time=$(date +%s%3N)

elapsed_time=$((end_time - start_time))
formatted_time=$(convert_time $elapsed_time)
echo "  Elapsed time for Asset Harvesting: ${formatted_time}"
echo "[${TIMESTAMP}] Elapsed time for Asset Harvesting: ${formatted_time}" >>${OUTPUT_DIR}/asset_harvest_timings.txt
echo "  Wrote timing file: ${OUTPUT_DIR}/asset_harvest_timings.txt"
