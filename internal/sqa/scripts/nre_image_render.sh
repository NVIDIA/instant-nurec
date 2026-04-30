#!/usr/bin/bash

set -e

# Source utilities
source "$(dirname "$0")/utils.sh"

# Global usage function
function usage_global() {
    echo "Usage: $0 <command> [OPTIONS]"
    echo "Available commands:"
    echo "  render-training-views   Render training views"
}

function usage_render_training_views() {
    echo "Tests the render subcommand with rendering training views."
    echo "The quality of rendered frames will generally be assessed by comparing against validation images through " \
         "script nre_eval_rendering_metrics.sh."
    echo ""
    echo "Usage: $0 render-training-views [OPTIONS]"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>               Docker image tag"
    echo "  --runfiles <PATH>         Path to runfiles executable"
    echo "  --bazel                   Use bazel run"
    echo ""
    echo "Required arguments:"
    echo "  --artifact-path <PATH>    Path to last.usdz file"
    echo "  --output-dir <PATH>       Path to output directory"
    echo "  --camera-id <ID>          Camera ID"
    echo ""
    echo "Optional arguments:"
    echo "  --image-scale <FLOAT>     Desired resolution of the rendered frames in proportion of the camera resolution (default: 0.25)"
    echo "  --frame-step <INT>        Frame step (default: 100)"
    echo "  --no-obfuscated <BOOL>    Run on non-obfuscated image. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <SUFFIX>         Docker image suffix, ex. 'grpc' (default: none)"
    echo "  --help                    Show this help message"
}

function is_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
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
        return 0 # true in bash
    fi
    return 1 # false in bash
}

setup_docker_args() {
    # Output DOCKER_ARGS
    local tag=$1
    local ngc_api_key=$2
    local no_obfuscated=$3
    local suffix=$4
    local mount_volumes=("${@:5}")

    local docker_envvars=("-e NGC_API_KEY=${ngc_api_key}")

    # Docker image information
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

    # Use utility function to setup volume mounts
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
render-training-views)
    TAG=""
    RUNFILES_EXECUTABLE=""
    ARTIFACT_PATH=""
    OUTPUT_DIR=""
    CAMERA_ID=""
    FRAME_STEP=100
    IMAGE_SCALE=0.25
    NO_OBFUSCATED=false
    BAZEL=false
    TEST_ACTOR_CONTROL=false
    SUFFIX=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --tag)
            if [ -n "$TAG" ]; then
                echo "Error: --tag is already set."
                usage_render_training_views
                exit 1
            fi
            TAG="$2"
            shift
            ;;
        --runfiles)
            RUNFILES_EXECUTABLE="$2"
            shift
            ;;
        --artifact-path)
            if [ -n "$ARTIFACT_PATH" ]; then
                echo "Error: --artifact-path is already set."
                usage_render_training_views
                exit 1
            fi
            ARTIFACT_PATH="$2"
            shift
            ;;
        --output-dir)
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Error: --output-dir is already set."
                usage_render_training_views
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --camera-id)
            if [ -n "$CAMERA_ID" ]; then
                echo "Error: --camera-id is already set."
                usage_render_training_views
                exit 1
            fi
            CAMERA_ID="$2"
            shift
            ;;
        --image-scale)
            IMAGE_SCALE="$2"
            shift
            ;;
        --frame-step)
            if ! is_integer "$2"; then
                echo "Error: --frame-step must be an integer"
                usage_render_training_views
                exit 1
            fi
            FRAME_STEP="$2"
            shift
            ;;
        --test-actor-control)
            TEST_ACTOR_CONTROL=true
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
            usage_render_training_views
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_render_training_views
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
        usage_render_training_views
        exit 1
    elif [[ $execution_modes -gt 1 ]]; then
        echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
        usage_render_training_views
        exit 1
    fi

    if [ -z "$ARTIFACT_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CAMERA_ID" ]; then
        echo "Error: --artifact-path, --output-dir, and --camera-id are required."
        usage_render_training_views
        exit 1
    fi

    mkdir -p $OUTPUT_DIR

    echo "Running render with the following parameters:"
    echo "  TAG: $TAG"
    echo "  ARTIFACT_PATH: $ARTIFACT_PATH"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  CAMERA_ID: $CAMERA_ID"
    echo "  IMAGE_SCALE: $IMAGE_SCALE"
    echo "  FRAME_STEP: $FRAME_STEP"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  BAZEL=${BAZEL}"
    echo "  SUFFIX=${SUFFIX}"
    echo ""

    # Defines NGC_API_KEY if not set
    check_and_get_ngc_api_key

    CMD_RENDER=(
        render
        --artifact-path "$ARTIFACT_PATH"
        --output-dir "${OUTPUT_DIR}"
        --camera-id "${CAMERA_ID}"
        --image-format png
        --image-scale ${IMAGE_SCALE}
        --frame-step ${FRAME_STEP}
        --frame-naming frame-end-timestamp
    )
    if [[ "$TEST_ACTOR_CONTROL" == true ]]; then
       CMD_RENDER+=("--enable-editing-actors" "--demo-actor-transform")
    fi

    echo "NRE command: ${CMD_RENDER[@]}"

    # Set bazel run target based on obfuscation flag
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    if [[ "$BAZEL" == true ]]; then
        CMD_RENDER="bazel run ${BAZEL_RUN} -- ${CMD_RENDER[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        CMD_RENDER="${RUNFILES_EXECUTABLE} ${CMD_RENDER[@]}"
    else
        ARTIFACT_DIR=$(dirname "${ARTIFACT_PATH}")
        MOUNT_VOLUMES=(
          "$ARTIFACT_DIR"
          "$OUTPUT_DIR"
        )

        setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES[@]}"

        CMD_RENDER="docker run ${DOCKER_ARGS[@]} ${CMD_RENDER[@]}"
    fi

    echo "EXECUTE COMMAND: $(safe_print "${CMD_RENDER}" "NGC_API_KEY")"
    eval ${CMD_RENDER}

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
