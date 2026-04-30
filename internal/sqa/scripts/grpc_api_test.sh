#!/usr/bin/bash

set -e

# Source utilities
source "$(dirname "$0")/utils.sh"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Global usage function
function usage_global() {
    echo "Usage: $0 <command> [OPTIONS]"
    echo "Available commands:"
    echo "  preprocess   Preprocess data for the gRPC API"
    echo "  run-server   Run the gRPC server"
    echo "  test-shim    Test the shim with synthetic frames"
    echo "  render-grpc  Test the render-grpc subcommand"
    echo "  --help       Show this help message"
}

# Preprocess usage function
function usage_preprocess() {
    echo "Usage: $0 preprocess [OPTIONS]"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>               Docker image tag"
    echo "  --runfiles <PATH>         Path to runfiles executable"
    echo "  --bazel                   Use bazel run"
    echo ""
    echo "Required arguments:"
    echo "  --artifact-path <PATH>    Path to last.usdz file"
    echo "  --output-dir <PATH>       Path to output directory"
    echo "  --dataset-path <PATH>     Path to .zarr.itar file"
    echo "  --camera-ids <IDS>        Comma-separated list of camera IDs, eg \"camera_cross_right_120fov,camera_cross_left_120fov,camera_front_wide_120fov,camera_front_tele_30fov\""
    echo "  --lidar-id <ID>           Lidar ID, eg \"lidar_gt_top_p128_v4p5\""
    echo ""
    echo "Optional arguments:"
    echo "  --camera-frame-idx <IDX>  Camera frame index to use for ego mask export"
    echo "  --no-obfuscated <BOOL>    Run on non-obfuscated image or bazel run. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <SUFFIX>         Docker image suffix, ex. 'grpc' (default: none)"
    echo "  --help                    Show this help message"
}

# Run server usage function
function usage_run_server() {
    echo "Usage: $0 run-server [OPTIONS]"
    echo ""
    echo "Execution mode (exactly one required):"
    echo "  --tag <TAG>                 Docker image tag"
    echo "  --runfiles <PATH>           Path to runfiles executable"
    echo "  --bazel                     Use bazel run"
    echo ""
    echo "Required arguments:"
    echo "  --artifact-path <PATH>      Path to last.usdz file"
    echo "  --egocar-hood-dir <PATH>    Path to ego-hood images"
    echo ""
    echo "Optional arguments:"
    echo "  --no-obfuscated <BOOL>      Run on non-obfuscated image or bazel run. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <SUFFIX>           Docker image suffix, ex. 'grpc' (default: none)"
    echo "  --metrics-output-dir <PATH> Path to output directory for metrics"
    echo "  --enable-editing-actors     Enable editing actors in serve-grpc (needed for testing actor control)"
    echo "  --edit-assets <PATH>        Path to edit-assets JSON used to mount external asset directories"
    echo "  --enable-difix <BOOL>       Enable DiFix in serve-grpc"
    echo "  --use-gsplat                Enable GSplat renderer"
    echo "  --port <INT>                gRPC server port"
    echo "  --help                      Show this help message"
}

function usage_render_grpc() {
    echo "Tests the render-grpc gRPC client shipped as part of the standard NRE docker image." \
         "The client is used to render frames from a given USDZ artifact from a training camera " \
         "along the training trajectory."
    echo "The quality of rendered frames will generally be assessed by comparing against validation images through " \
         "script nre_eval_rendering_metrics.sh."
    echo ""
    echo "Usage: $0 render-grpc [OPTIONS]"
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
    echo "  --frame-height <INT>      Desired height of the rendered frames in pixels (default: 720)"
    echo "  --frame-step <INT>        Frame step (default: 100)"
    echo "  --no-obfuscated <BOOL>    Run on non-obfuscated image or bazel run. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <SUFFIX>         Docker image suffix, ex. 'grpc' (default: none)"
    echo "  --port <INT>              gRPC server port"
    echo "  --enable-editing-actors   Enable sending dynamic actor updates in render requests"
    echo "  --edit-assets <PATH>      Path to edit-assets JSON generated for the test"
    echo "  --help                    Show this help message"
}

# Test shim usage function
function usage_test_shim() {
    echo "Usage: $0 test-shim [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --shim-path <PATH>           Path to shim code"
    echo "  --sensor-tracks-json <PATH>  Path to sensor tracks JSON"
    echo "  --camera-ids <IDS>           Comma-separated list of camera IDs"
    echo "  --lidar-id <ID>              Lidar ID"
    echo "  --scene-id <ID>              Scene ID"
    echo "  --output-dir <PATH>          Path to output directory where timings should be saved"
    echo "  --filename <STR>             String that defines filename where timings are saved. ONLY the name, it's saved in output directory, eg timings.txt"
    echo ""
    echo "Optional arguments:"
    echo "  --lidar-type <TYPE>          Lidar type: AT128 or Pandar128 (default: Pandar128)"
    echo "  --port <INT>                 gRPC server port"
    echo "  --help                       Show this help message"
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

function extract_edit_asset_dirs() {
    local edit_assets_path="$1"

    if ! command -v jq >/dev/null 2>&1; then
        echo "Error: jq is required when using --edit-assets in Docker mode."
        exit 1
    fi

    jq -r '
      [
        (.replace[]?.replacement_id),
        (.insert.asset_ids[]?)
      ]
      | map(select(type == "string" and startswith("/")) | sub("/[^/]+$"; ""))
      | unique[]
    ' "$edit_assets_path"
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
    local tag=$1
    local ngc_api_key=$2
    local no_obfuscated=$3
    local suffix=$4
    shift 4
    local mount_volumes=("$@")

    local docker_envvars=("-e NGC_API_KEY=${ngc_api_key}")

    local obfuscated_suffix="_obfuscated"
    if [[ "$no_obfuscated" == true ]]; then obfuscated_suffix=""; fi
    local suffix_part=""
    if [[ -n "$suffix" ]]; then
        suffix_part="_${suffix}"
    fi
    local image_name="nre${obfuscated_suffix}_run${suffix_part}"
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
preprocess)
    TAG=""
    RUNFILES_EXECUTABLE=""
    ARTIFACT_PATH=""
    OUTPUT_DIR=""
    DATASET_PATH=""
    CAMERA_IDS=""
    LIDAR_ID=""
    CAMERA_FRAME_IDX="default"
    BAZEL=false
    NO_OBFUSCATED=false
    SUFFIX=""


    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --tag)
            if [ -n "$TAG" ]; then
                echo "Error: --tag is already set."
                usage_preprocess
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
                usage_preprocess
                exit 1
            fi
            ARTIFACT_PATH="$2"
            shift
            ;;
        --output-dir)
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Error: --output-dir is already set."
                usage_preprocess
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --dataset-path)
            if [ -n "$DATASET_PATH" ]; then
                echo "Error: --dataset-path is already set."
                usage_preprocess
                exit 1
            fi
            DATASET_PATH="$2"
            shift
            ;;
        --camera-ids)
            if [ -n "$CAMERA_IDS" ]; then
                echo "Error: --camera-ids is already set."
                usage_preprocess
                exit 1
            fi
            CAMERA_IDS="$2"
            shift
            ;;
        --lidar-id)
            if [ -n "$LIDAR_ID" ]; then
                echo "Error: --lidar-id is already set."
                usage_preprocess
                exit 1
            fi
            LIDAR_ID="$2"
            shift
            ;;
        --camera-frame-idx)
            CAMERA_FRAME_IDX="$2"
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
            usage_preprocess
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_preprocess
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
        usage_preprocess
        exit 1
    elif [[ $execution_modes -gt 1 ]]; then
        echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
        usage_preprocess
        exit 1
    fi

    if [ -z "$ARTIFACT_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$DATASET_PATH" ] || [ -z "$CAMERA_IDS" ] || [ -z "$LIDAR_ID" ]; then
        echo "Error: --artifact-path, --output-dir, --dataset-path, --camera-ids, and --lidar-id are required."
        usage_preprocess
        exit 1
    fi

    echo "Running preprocessing with the following parameters:"
    echo "  TAG: $TAG"
    echo "  ARTIFACT_PATH: $ARTIFACT_PATH"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  DATASET_PATH: $DATASET_PATH"
    echo "  CAMERA_IDS: $CAMERA_IDS"
    echo "  LIDAR_ID: $LIDAR_ID"
    echo "  CAMERA_FRAME_IDX: $CAMERA_FRAME_IDX"
    echo "  BAZEL=${BAZEL}"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  SUFFIX=${SUFFIX}"
    echo ""

    # Dataset directory
    DATASET_DIR=$(dirname "${DATASET_PATH}")

    # Create output directory now to allow bind-mounting it to the container
    mkdir -p ${OUTPUT_DIR}

    # Bazel run information
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    if [[ "$BAZEL" != true ]] && [[ -z "$RUNFILES_EXECUTABLE" ]]; then
        check_and_get_ngc_api_key
        ARTIFACT_DIR=$(dirname "${ARTIFACT_PATH}")
        MOUNT_VOLUMES=(
            "$ARTIFACT_DIR"
            "$DATASET_DIR"
            "$OUTPUT_DIR"
        )
        setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES[@]}"
    fi

    # Step 0: Upgrade usdz artifact
    UPGRADED_ARTIFACT_PATH="${OUTPUT_DIR}/artifacts/last_upgraded.usdz"

    # Check if input artifact exists
    if [[ ! -f "$ARTIFACT_PATH" ]]; then
        echo "Error: usdz artifact file not found: $ARTIFACT_PATH"
        exit 1
    fi

    mkdir -p ${OUTPUT_DIR}/artifacts

    CMD_UPGRADE_ARTIFACT=(
        upgrade-artifact
        --input "$ARTIFACT_PATH"
        --output "$UPGRADED_ARTIFACT_PATH"
    )

    echo "0. Running upgrade-artifact..."
    echo "  Input artifact: $ARTIFACT_PATH"
    echo "  Upgraded artifact: $UPGRADED_ARTIFACT_PATH"
    echo "  NRE command: ${CMD_UPGRADE_ARTIFACT[@]}"
    
    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_UPGRADE="bazel run ${BAZEL_RUN} -- ${CMD_UPGRADE_ARTIFACT[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_UPGRADE="${RUNFILES_EXECUTABLE} ${CMD_UPGRADE_ARTIFACT[@]}"
    else
        FINAL_CMD_UPGRADE="docker run ${DOCKER_ARGS[@]} ${CMD_UPGRADE_ARTIFACT[@]}"
    fi
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_UPGRADE}" "NGC_API_KEY")"
    eval ${FINAL_CMD_UPGRADE}
    echo "  Wrote upgraded artifact: ${UPGRADED_ARTIFACT_PATH}"
    echo ""

    # Step 1: Export ego mask
    CMD_EXPORT_EGO_MASK=(
        export-ego-mask
        --shard-file-pattern "$DATASET_PATH"
        --output-dir "$OUTPUT_DIR"
        $(echo $CAMERA_IDS | tr ',' '\n' | xargs -n1 printf "--camera-ids %s ")
    )
    if [[ "$CAMERA_FRAME_IDX" != "default" ]]; then
        CMD_EXPORT_EGO_MASK+=(--camera-frame-idx "$CAMERA_FRAME_IDX")
    fi

    echo "1. Running export-ego-mask..."
    echo "  NRE command: ${CMD_EXPORT_EGO_MASK[@]}"
    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_EGO_MASK="bazel run ${BAZEL_RUN} -- ${CMD_EXPORT_EGO_MASK[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_EGO_MASK="${RUNFILES_EXECUTABLE} ${CMD_EXPORT_EGO_MASK[@]}"
    else
        FINAL_CMD_EGO_MASK="docker run ${DOCKER_ARGS[@]} ${CMD_EXPORT_EGO_MASK[@]}"
    fi
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_EGO_MASK}" "NGC_API_KEY")"
    eval ${FINAL_CMD_EGO_MASK}
    echo "$(find "${OUTPUT_DIR}/ego-hoods" -name "*.png" | xargs -n1 echo "  Wrote")"
    echo ""

    # WAR bug 5555788
    # GRPC server expects '<ego-hoods>/<rig_id>/<camera_id>.png' but export-ego-mask writes flat
    # '<ego-hoods>/<camera_id>.png'. Move mask images into 'default' subfolder so the gRPC server sees a single rig.
    EGO_HOODS_DIR="${OUTPUT_DIR}/ego-hoods"
    mkdir -p "${EGO_HOODS_DIR}/default"
    mv "${EGO_HOODS_DIR}"/*.png "${EGO_HOODS_DIR}/default/"
    echo "  Moved ego mask images into ego-hoods/default for gRPC server (WAR for bug 5555788)."
    echo ""

    # Step 2: Export sequence tracks
    # Extract config and checkpoint from upgraded artifact
    UPGRADED_EXTRACT_DIR="${OUTPUT_DIR}/artifacts/last_upgraded_extracted"
    UPGRADED_CONFIG_PATH="${UPGRADED_EXTRACT_DIR}/parsed_config.yaml"
    UPGRADED_CHECKPOINT_PATH="${UPGRADED_EXTRACT_DIR}/checkpoint.ckpt"

    rm -rf "${UPGRADED_EXTRACT_DIR}"
    mkdir -p "${UPGRADED_EXTRACT_DIR}"
    unzip "${UPGRADED_ARTIFACT_PATH}" "checkpoint.ckpt" "parsed_config.yaml" -d "${UPGRADED_EXTRACT_DIR}"

    # Massage upgraded config: for archived artifacts the original dataset path may not match what we use at
    # test execution time. Replace any path in the config that ends with the dataset JSON basename by our
    # DATASET_JSON path (DATASET_PATH with .zarr.itar replaced by .json).
    DATASET_JSON="${DATASET_PATH%.zarr.itar}.json"
    DATASET_JSON_BASENAME=$(basename "$DATASET_JSON")
    sed -i "s#[^[:space:]]*${DATASET_JSON_BASENAME}#${DATASET_JSON}#g" "$UPGRADED_CONFIG_PATH"

    CMD_EXPORT_SEQUENCE_TRACKS=(export-sequence-tracks
        --config-name "$UPGRADED_CONFIG_PATH"
        --checkpoint-path "$UPGRADED_CHECKPOINT_PATH"
        --format=json
        --controllable-only
        --output-dir "$OUTPUT_DIR"
    )

    echo "2. Running export-sequence-tracks..."
    echo "  Using upgraded config: $UPGRADED_CONFIG_PATH"
    echo "  Using upgraded checkpoint: $UPGRADED_CHECKPOINT_PATH"
    echo "  NRE command: ${CMD_EXPORT_SEQUENCE_TRACKS[@]}"
    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_SEQUENCE_TRACKS="bazel run ${BAZEL_RUN} -- ${CMD_EXPORT_SEQUENCE_TRACKS[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_SEQUENCE_TRACKS="${RUNFILES_EXECUTABLE} ${CMD_EXPORT_SEQUENCE_TRACKS[@]}"
    else
        FINAL_CMD_SEQUENCE_TRACKS="docker run ${DOCKER_ARGS[@]} ${CMD_EXPORT_SEQUENCE_TRACKS[@]}"
    fi
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_SEQUENCE_TRACKS}" "NGC_API_KEY")"
    eval ${FINAL_CMD_SEQUENCE_TRACKS}
    echo "  Wrote ${OUTPUT_DIR}/sequence_tracks.json"
    echo ""

    # Step 3: Export ncore tracks
    CMD_EXPORT_NCORE_TRACKS=(export-ncore-tracks
        --shard-file-pattern "$DATASET_PATH"
        --model-tracks-json "${OUTPUT_DIR}/sequence_tracks.json"
        $(echo $CAMERA_IDS | tr ',' '\n' | xargs -n1 printf "--camera-id %s ")
        --lidar-id "$LIDAR_ID"
        --output-dir "$OUTPUT_DIR"
    )

    echo "3. Running export-ncore-tracks..."
    echo "  NRE command: ${CMD_EXPORT_NCORE_TRACKS[@]}"
    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_NCORE_TRACKS="bazel run ${BAZEL_RUN} -- ${CMD_EXPORT_NCORE_TRACKS[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_NCORE_TRACKS="${RUNFILES_EXECUTABLE} ${CMD_EXPORT_NCORE_TRACKS[@]}"
    else
        FINAL_CMD_NCORE_TRACKS="docker run ${DOCKER_ARGS[@]} ${CMD_EXPORT_NCORE_TRACKS[@]}"
    fi
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_NCORE_TRACKS}" "NGC_API_KEY")"
    eval ${FINAL_CMD_NCORE_TRACKS}
    echo "  Wrote $(find "$OUTPUT_DIR" -name "sensor_tracks_*.json")"
    echo ""
    ;;
run-server)
    TAG=""
    RUNFILES_EXECUTABLE=""
    ARTIFACT_PATH=""
    EGOCAR_HOOD_DIR=""
    METRICS_OUTPUT_DIR=""
    BAZEL=false 
    NO_OBFUSCATED=false
    ENABLE_EDITING_ACTORS=false
    EDIT_ASSETS=""
    ENABLE_DIFIX=false
    USE_GSPLAT=false
    SUFFIX=""
    PORT=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --tag)
            if [ -n "$TAG" ]; then
                echo "Error: --tag is already set."
                usage_run_server
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
                usage_run_server
                exit 1
            fi
            ARTIFACT_PATH="$2"
            shift
            ;;
        --egocar-hood-dir)
            if [ -n "$EGOCAR_HOOD_DIR" ]; then
                echo "Error: --egocar-hood-dir is already set."
                usage_run_server
                exit 1
            fi
            EGOCAR_HOOD_DIR="$2"
            shift
            ;;
        --metrics-output-dir)
            if [ -n "$METRICS_OUTPUT_DIR" ]; then
                echo "Error: --metrics-output-dir is already set."
                usage_run_server
                exit 1
            fi
            METRICS_OUTPUT_DIR="$2"
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
        --enable-editing-actors)
            ENABLE_EDITING_ACTORS=true
            ;;
        --edit-assets)
            EDIT_ASSETS="$2"
            shift
            ;;
        --enable-difix)
            ENABLE_DIFIX=true
            ;;
        --use-gsplat)
            USE_GSPLAT=true
            ;;
        --port)
            if ! is_integer "$2"; then
                echo "Error: --port must be an integer"
                usage_run_server
                exit 1
            fi
            PORT="$2"
            shift
            ;;
        --help)
            usage_run_server
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_run_server
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
        usage_run_server
        exit 1
    elif [[ $execution_modes -gt 1 ]]; then
        echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
        usage_run_server
        exit 1
    fi

    if [ -z "$ARTIFACT_PATH" ] || [ -z "$EGOCAR_HOOD_DIR" ]; then
        echo "Error: --artifact-path and --egocar-hood-dir are required."
        usage_run_server
        exit 1
    fi

    echo "Running gRPC server with the following parameters:"
    echo "  TAG: $TAG"
    echo "  ARTIFACT_PATH: $ARTIFACT_PATH"
    echo "  EGOCAR_HOOD_DIR: $EGOCAR_HOOD_DIR"
    echo "  BAZEL=${BAZEL}"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  ENABLE_EDITING_ACTORS=${ENABLE_EDITING_ACTORS}"
    echo "  EDIT_ASSETS=${EDIT_ASSETS}"
    echo "  ENABLE_DIFIX=${ENABLE_DIFIX}"
    echo "  USE_GSPLAT=${USE_GSPLAT}"
    echo "  SUFFIX=${SUFFIX}"
    echo "  PORT=${PORT}"

    # Defines NGC_API_KEY if not set
    check_and_get_ngc_api_key

    # Build gRPC serve command, skip serve-grpc command if suffix is "grpc" (grpc image is fixed to grpc and does not accept serve-grpc command)
    CMD_SERVE_GRPC=()
    if [[ "$SUFFIX" != "grpc" ]]; then
        CMD_SERVE_GRPC+=(serve-grpc)
    fi
    CMD_SERVE_GRPC+=(
        --artifact-glob "$ARTIFACT_PATH"
        --egocar-hood-dir "$EGOCAR_HOOD_DIR"
        --no-enable-nrend # Turned off until nrend supports bilarf/ppisp
    )
    if [[ "$ENABLE_EDITING_ACTORS" == true ]]; then
        CMD_SERVE_GRPC+=(--enable-editing-actors)
    fi
    if [[ -n "$PORT" ]]; then
        CMD_SERVE_GRPC+=(--port "$PORT")
    fi
    if [[ "$USE_GSPLAT" == true ]]; then
        CMD_SERVE_GRPC+=(--use-gsplat)
    fi
    if [[ "$ENABLE_DIFIX" == true ]]; then
        DIFIX_URL="https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut/files/cosmos_3dgut.pt"
        DIFIX_CACHE="/tmp/difix-cache"
        DIFIX_MODEL_FILENAME="cosmos_3dgut.pt"
        CMD_SERVE_GRPC+=(
            --enable-difix
            --difix-url "$DIFIX_URL"
            --difix-cache "$DIFIX_CACHE"
            --difix-model-filename "$DIFIX_MODEL_FILENAME"
        )
    fi
    if [ ! -z "$METRICS_OUTPUT_DIR" ]; then
        CMD_SERVE_GRPC+=(--metrics-output-dir "$METRICS_OUTPUT_DIR")
    fi
    echo "  NRE command: ${CMD_SERVE_GRPC[@]}"

    mkdir -p "$EGOCAR_HOOD_DIR"

    # Bazel run information
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_SERVE_GRPC="bazel run ${BAZEL_RUN} -- ${CMD_SERVE_GRPC[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        if [[ -n "$TEST_TMPDIR" ]]; then
            # The server requires a writable directory. We give a temporary sandbox path to match the behavior
            # achieved with Docker.
            CMD_SERVE_GRPC+=(--download-cache-dir "$TEST_TMPDIR/nre/scene_cache")
        fi
        FINAL_CMD_SERVE_GRPC="${RUNFILES_EXECUTABLE} ${CMD_SERVE_GRPC[@]}"
    else
        ARTIFACT_DIR=$(dirname "${ARTIFACT_PATH}")
        MOUNT_VOLUMES=(
          "$ARTIFACT_DIR"
          "$EGOCAR_HOOD_DIR"
        )
        if [[ -n "$EDIT_ASSETS" ]]; then
          MOUNT_VOLUMES+=("$(dirname "${EDIT_ASSETS}")")
          mapfile -t EDIT_ASSET_DIRS < <(extract_edit_asset_dirs "$EDIT_ASSETS")
          if [[ ${#EDIT_ASSET_DIRS[@]} -gt 0 ]]; then
            MOUNT_VOLUMES+=("${EDIT_ASSET_DIRS[@]}")
          fi
        fi
        if [[ "$ENABLE_DIFIX" == true ]]; then
          MOUNT_VOLUMES+=("$DIFIX_CACHE")
        fi
        setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES[@]}"

        FINAL_CMD_SERVE_GRPC="docker run ${DOCKER_ARGS[@]} ${CMD_SERVE_GRPC[@]}"
    fi
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_SERVE_GRPC}" "NGC_API_KEY")"
    eval ${FINAL_CMD_SERVE_GRPC}
    ;;
render-grpc)
    TAG=""
    RUNFILES_EXECUTABLE=""
    ARTIFACT_PATH=""
    OUTPUT_DIR=""
    CAMERA_ID=""
    FRAME_STEP=100
    FRAME_HEIGHT=720
    NO_OBFUSCATED=false
    BAZEL=false
    TEST_ACTOR_CONTROL=false
    ENABLE_EDITING_ACTORS=false
    EDIT_ASSETS=""
    SUFFIX=""
    PORT=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --tag)
            if [ -n "$TAG" ]; then
                echo "Error: --tag is already set."
                usage_render_grpc
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
                usage_render_grpc
                exit 1
            fi
            ARTIFACT_PATH="$2"
            shift
            ;;
        --output-dir)
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Error: --output-dir is already set."
                usage_render_grpc
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --camera-id)
            if [ -n "$CAMERA_ID" ]; then
                echo "Error: --camera-id is already set."
                usage_render_grpc
                exit 1
            fi
            CAMERA_ID="$2"
            shift
            ;;
        --frame-height)
            if ! is_integer "$2"; then
                echo "Error: --frame-height must be an integer"
                usage_render_grpc
                exit 1
            fi
            FRAME_HEIGHT="$2"
            shift
            ;;
        --frame-step)
            if ! is_integer "$2"; then
                echo "Error: --frame-step must be an integer"
                usage_render_grpc
                exit 1
            fi
            FRAME_STEP="$2"
            shift
            ;;
        --test-actor-control)
            TEST_ACTOR_CONTROL=true
            ;;
        --enable-editing-actors)
            ENABLE_EDITING_ACTORS=true
            ;;
        --edit-assets)
            EDIT_ASSETS="$2"
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
        --port)
            if ! is_integer "$2"; then
                echo "Error: --port must be an integer"
                usage_render_grpc
                exit 1
            fi
            PORT="$2"
            shift
            ;;
        --help)
            usage_render_grpc
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_render_grpc
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
        usage_render_grpc
        exit 1
    elif [[ $execution_modes -gt 1 ]]; then
        echo "Error: --tag, --runfiles, and --bazel are mutually exclusive. Only one can be provided."
        usage_render_grpc
        exit 1
    fi

    if [ -z "$ARTIFACT_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CAMERA_ID" ]; then
        echo "Error: --artifact-path, --output-dir, and --camera-id are required."
        usage_render_grpc
        exit 1
    fi

    mkdir -p $OUTPUT_DIR

    echo "Running gRPC render client with the following parameters:"
    echo "  TAG: $TAG"
    echo "  ARTIFACT_PATH: $ARTIFACT_PATH"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  CAMERA_ID: $CAMERA_ID"
    echo "  FRAME_HEIGHT: $FRAME_HEIGHT"
    echo "  FRAME_STEP: $FRAME_STEP"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  BAZEL=${BAZEL}"
    echo "  ENABLE_EDITING_ACTORS=${ENABLE_EDITING_ACTORS}"
    echo "  EDIT_ASSETS=${EDIT_ASSETS}"
    echo "  SUFFIX=${SUFFIX}"
    echo "  PORT=${PORT}"
    echo ""

    # Defines NGC_API_KEY if not set
    check_and_get_ngc_api_key

    CMD_RENDER_GRPC=(
        render-grpc
        --artifact-path "$ARTIFACT_PATH"
        --output-dir "${OUTPUT_DIR}"
        --camera-id "$CAMERA_ID"
        --image-format png
        --height ${FRAME_HEIGHT}
        --frame-step ${FRAME_STEP}
        --frame-naming frame-end-timestamp
        --shutdown-server-on-completion
    )
    if [[ -n "$PORT" ]]; then
        CMD_RENDER_GRPC+=(--port ${PORT})
    fi
    if [[ "$ENABLE_EDITING_ACTORS" == true ]]; then
        CMD_RENDER_GRPC+=(--enable-editing-actors)
    fi
    if [[ -n "$EDIT_ASSETS" ]]; then
        CMD_RENDER_GRPC+=(--edit-assets "$EDIT_ASSETS")
    fi
    if [[ "$TEST_ACTOR_CONTROL" == true ]]; then
        CMD_RENDER_GRPC+=(--demo-actor-transform)
        CMD_RENDER_GRPC+=(--enable-editing-actors)
    fi

    echo "NRE command: ${CMD_RENDER_GRPC[@]}"

    # Bazel run information
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    if [[ "$BAZEL" == true ]]; then
        CMD_RENDER_GRPC="bazel run ${BAZEL_RUN} -- ${CMD_RENDER_GRPC[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        CMD_RENDER_GRPC="${RUNFILES_EXECUTABLE} ${CMD_RENDER_GRPC[@]}"
    else
        ARTIFACT_DIR=$(dirname "${ARTIFACT_PATH}")
        MOUNT_VOLUMES=(
          "$ARTIFACT_DIR"
          "$OUTPUT_DIR"
        )
        if [[ -n "$EDIT_ASSETS" ]]; then
          MOUNT_VOLUMES+=("$(dirname "${EDIT_ASSETS}")")
          mapfile -t EDIT_ASSET_DIRS < <(extract_edit_asset_dirs "$EDIT_ASSETS")
          if [[ ${#EDIT_ASSET_DIRS[@]} -gt 0 ]]; then
            MOUNT_VOLUMES+=("${EDIT_ASSET_DIRS[@]}")
          fi
        fi

        setup_docker_args "$TAG" "$NGC_API_KEY" "$NO_OBFUSCATED" "$SUFFIX" "${MOUNT_VOLUMES[@]}"

        CMD_RENDER_GRPC="docker run ${DOCKER_ARGS[@]} ${CMD_RENDER_GRPC[@]}"
    fi

    echo "EXECUTE COMMAND: $(safe_print "${CMD_RENDER_GRPC}" "NGC_API_KEY")"
    eval ${CMD_RENDER_GRPC}

    ;;
test-shim)
    SHIM_PATH=""
    SENSOR_TRACKS_JSON=""
    CAMERA_IDS=""
    LIDAR_ID=""
    LIDAR_TYPE="Pandar128"
    SCENE_ID=""
    OUTPUT_DIR=""
    FILENAME=""
    PORT=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --shim-path)
            if [ -n "$SHIM_PATH" ]; then
                echo "Error: --shim-path is already set."
                usage_test_shim
                exit 1
            fi
            SHIM_PATH="$2"
            shift
            ;;
        --sensor-tracks-json)
            if [ -n "$SENSOR_TRACKS_JSON" ]; then
                echo "Error: --sensor-tracks-json is already set."
                usage_test_shim
                exit 1
            fi
            SENSOR_TRACKS_JSON="$2"
            shift
            ;;
        --camera-ids)
            if [ -n "$CAMERA_IDS" ]; then
                echo "Error: --camera-ids is already set."
                usage_test_shim
                exit 1
            fi
            CAMERA_IDS="$2"
            shift
            ;;
        --lidar-id)
            if [ -n "$LIDAR_ID" ]; then
                echo "Error: --lidar-id is already set."
                usage_test_shim
                exit 1
            fi
            LIDAR_ID="$2"
            shift
            ;;
        --lidar-type)
            if [ -n "$LIDAR_TYPE" ]; then
                echo "Error: --lidar-type is already set."
                usage_test_shim
                exit 1
            fi
            LIDAR_TYPE="$2"
            shift
            ;;
        --scene-id)
            if [ -n "$SCENE_ID" ]; then
                echo "Error: --scene-id is already set."
                usage_test_shim
                exit 1
            fi
            SCENE_ID="$2"
            shift
            ;;
        --output-dir)
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Error: --output-dir is already set."
                usage_test_shim
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --filename)
            if [ -n "$FILENAME" ]; then
                echo "Error: --filename is already set."
                usage_test_shim
                exit 1
            fi
            FILENAME="$2"
            shift
            ;;
        --port)
            if ! is_integer "$2"; then
                echo "Error: --port must be an integer"
                usage_test_shim
                exit 1
            fi
            PORT="$2"
            shift
            ;;
        --help)
            usage_test_shim
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage_test_shim
            exit 1
            ;;
        esac
        shift
    done

    if [ -z "$SHIM_PATH" ] || [ -z "$SENSOR_TRACKS_JSON" ] || [ -z "$CAMERA_IDS" ] || [ -z "$LIDAR_ID" ] || [ -z "$SCENE_ID" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$FILENAME" ]; then
        echo "Error: --shim-path, --sensor-tracks-json, --camera-ids, --lidar-id, --scene-id, --output-dir, --filename are required."
        usage_test_shim
        exit 1
    fi

    # Test the shim
    CMD_SHIM=(src.test.test_liauto_shim
        --sensor_tracks_json "$SENSOR_TRACKS_JSON"
        --camera_ids "$CAMERA_IDS"
        --lidar_id "$LIDAR_ID"
        --lidar_type "$LIDAR_TYPE"
        --scene_id "$SCENE_ID"
    )
    if [[ -n "$PORT" ]]; then
        CMD_SHIM+=(--port "$PORT")
    fi

    mkdir -p $OUTPUT_DIR

    echo "Testing shim with the following parameters:"
    echo "  SENSOR_TRACKS_JSON: $SENSOR_TRACKS_JSON"
    echo "  CAMERA_IDS: $CAMERA_IDS"
    echo "  LIDAR_ID: $LIDAR_ID"
    echo "  LIDAR_TYPE: $LIDAR_TYPE"
    echo "  SCENE_ID: $SCENE_ID"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  FILENAME: $FILENAME"
    echo "  PORT: $PORT"
    echo "  Shim command: ${CMD_SHIM[@]}"

    cd $SHIM_PATH

    FINAL_CMD_SHIM="python -m ${CMD_SHIM[@]}"
    echo "EXECUTE COMMAND: ${FINAL_CMD_SHIM}"
    # Measure elapsed time for shim run
    start_time=$(date +%s%3N)
    eval ${FINAL_CMD_SHIM}
    end_time=$(date +%s%3N)
    cd -

    elapsed_time=$((end_time - start_time))
    formatted_time=$(convert_time $elapsed_time)
    echo "  Elapsed time for shim run: ${formatted_time}"
    echo "[${TIMESTAMP}] Elapsed time for shim run: ${formatted_time}" >>${OUTPUT_DIR}/${FILENAME}
    echo "  Wrote file: ${OUTPUT_DIR}/${FILENAME}"
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
