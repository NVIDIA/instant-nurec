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
    echo "  --bazel                    Use bazel run (if --no-obfuscated is set, 'bazel run //:run', else 'bazel run //internal/scripts/pycena/runtime:pycena')"
    echo ""
    echo "Required arguments:"
    echo "  --config-path <PATH>       Path to the configuration file [Taken from release instructions]"
    echo "  --dataset-path <PATH>      Path to the dataset JSON file [Taken from release instructions]"
    echo "  --output-dir <PATH>        Output directory"
    echo "  --filename <STR>           String that defines filename where timings are saved. ONLY the name, it's saved in output directory, eg timings.txt"
    echo ""
    echo "Optional arguments:"
    echo "  --world-size <INT>         World size (total number of GPUs across all nodes to use for training or validation). Value 0 means using all available GPUs"
    echo "  --num-nodes <INT>          Number of nodes. Value 0 means using all available nodes"
    echo "  --train-append <ARG>       Additional arguments to append to the train command (can be used multiple times)"
    echo "  --val-append <ARG>         Additional arguments to append to the validation command (can be used multiple times)"
    echo "  --debug                    Enables some debug flags. Should not be used during SQA run"
    echo "  --no-obfuscated            Run on non-obfuscated image. Should not be used during SQA run (no-op with --runfiles)"
    echo "  --suffix <STR>             Suffix to append to the NRE image name, ex. 'grpc' (default: none)"
    echo "  --mode <STR>               Either 'trainval', 'train' or 'val' (default: 'trainval'). Should not be used during SQA run"
    echo "  --force                    Force the command to run even if the output directory already exists"
    echo ""
    echo "Examples:"
    echo ""
    echo "  Docker mode:"
    echo "    $0 --tag \"X.X.XXX-SHAXXXXX\" \\"
    echo "       --config-path \"path/to/config.yaml\" \\"
    echo "       --dataset-path \"path/to/dataset.json\" \\"
    echo "       --output-dir \"path/to/output\" \\"
    echo "       --filename \"timings.txt\""
    echo ""
    echo "  Runfiles mode:"
    echo "    $0 --runfiles \"/path/to/resolved/executable\" \\"
    echo "       --config-path \"path/to/config.yaml\" \\"
    echo "       --dataset-path \"path/to/dataset.json\" \\"
    echo "       --output-dir \"path/to/output\" \\"
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

TAG=""
CONFIG_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""
FILENAME=""
DEBUG=false
BAZEL=false
NO_OBFUSCATED=false
MODE=trainval
WORLD_SIZE=1
NUM_NODES=1
TRAIN_APPEND=()
VAL_APPEND=()
FORCE=false
SUFFIX=""

# Parse input arguments
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
    --runfiles=*)
        RUNFILES_EXECUTABLE="${1#*=}"
        ;;
    --config-path)
        CONFIG_PATH="$2"
        shift
        ;;
    --dataset-path)
        DATASET_PATH="$2"
        shift
        ;;
    --output-dir)
        OUTPUT_DIR="$2"
        shift
        ;;
    --filename)
        FILENAME="$2"
        shift
        ;;
    --debug)
        DEBUG=true
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
    --mode)
        MODE="$2"
        shift
        ;;
    --world-size)
        WORLD_SIZE="$2"
        shift
        ;;
    --num-nodes)
        NUM_NODES="$2"
        shift
        ;;
    --train-append)
        TRAIN_APPEND+=("$2")
        shift
        ;;
    --val-append)
        VAL_APPEND+=("$2")
        shift
        ;;
    --force)
        FORCE=true
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

if [[ "$MODE" != "trainval" && "$MODE" != "val" && "$MODE" != "train" ]]; then
    echo "Invalid MODE: $MODE"
    exit 1
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
if [[ -z "$CONFIG_PATH" || -z "$DATASET_PATH" || -z "$OUTPUT_DIR" || -z "$FILENAME" ]]; then
    echo "Error: --config-path, --dataset-path, --output-dir, and --filename are required."
    usage
    exit 1
else
    if [[ -n "$TAG" ]]; then
        echo "  TAG=${TAG}"
    fi
    if [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        echo "  RUNFILES_EXECUTABLE=${RUNFILES_EXECUTABLE}"
    fi
    echo "  CONFIG_PATH=${CONFIG_PATH}"
    echo "  DATASET_PATH=${DATASET_PATH}"
    echo "  OUTPUT_DIR=${OUTPUT_DIR}"
    echo "  FILENAME=${FILENAME}"
    echo "  DEBUG=${DEBUG}"
    echo "  BAZEL=${BAZEL}"
    echo "  NO_OBFUSCATED=${NO_OBFUSCATED}"
    echo "  MODE=${MODE}"
    echo "  WORLD_SIZE=${WORLD_SIZE}"
    echo "  NUM_NODES=${NUM_NODES}"
    echo "  TRAIN_APPEND=${TRAIN_APPEND[*]}"
    echo "  VAL_APPEND=${VAL_APPEND[*]}"
fi

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

# Dataset directory
DATASET_DIR=$(dirname "${DATASET_PATH}")

# Setup Docker arguments only if we're using Docker (TAG is set)
if [[ -n "$TAG" ]]; then
    # Get volume mount arguments
    VOLUME_MOUNTS=($(setup_volume_mounts "$DATASET_DIR" "$OUTPUT_DIR"))

    DOCKER_ARGS=(
        --shm-size=8g
        -it
        --user $(id -u):$(id -g)
        --rm
        --gpus all
        ${DOCKER_ENVVARS[@]}
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
NRE_RUN_NAME="nre${OBFUSCATED}_run${SUFFIX_PART}"
NRE_IMAGE="${NGC_IMAGES}/${NRE_RUN_NAME}:${TAG}"

##################
# Training phase #
##################
if [[ "$MODE" == "trainval" || "$MODE" == "train" ]]; then
    # Don't start training if there are alreadz ckpt or parsed.yaml files present, as those will interfere with validation mode, see below.
    if [[ "$FORCE" == false ]] && find ${OUTPUT_DIR} \( -name "parsed.yaml" -o -name "last.ckpt" \) | grep -q .; then
        FOUND_FILES=$(find ${OUTPUT_DIR} -name "parsed.yaml" -o -name "last.ckpt")
        echo "Error: Found at least one file matching 'parsed.yaml' or 'last.ckpt'."
        echo "Please rm those directories:"
        echo "$(for f in ${FOUND_FILES}; do echo rm -rf $(dirname $(dirname $f)); done)"
        echo "OR mv, if you like to keep them"
        echo "$(for f in ${FOUND_FILES}; do echo mv $(dirname $(dirname $f)) path/to/dir; done)"
        exit 1
    fi
    CMD_TRAIN=(
        --config-name=${CONFIG_PATH}
        mode=train
        dataset.path=${DATASET_PATH}
        out_dir=${OUTPUT_DIR}
        trainer.world_size=${WORLD_SIZE}
        trainer.num_nodes=${NUM_NODES}  # TODO: multi-node training not tested outside SLURM for now.
    )

    if [[ "$DEBUG" == true ]]; then
        CMD_TRAIN+=(
            dataset.n_samples_per_epoch=4
            checkpoint.artifact.mesh.generic.enabled=false
            checkpoint.artifact.mesh.ground.enabled=false
            dataset.n_train_sample_camera_rays=128
        )
    fi
    
    # Add any additional train arguments
    if [ ${#TRAIN_APPEND[@]} -gt 0 ]; then
        for arg in "${TRAIN_APPEND[@]}"; do
            CMD_TRAIN+=("$arg")
        done
    fi

    echo "1. Running NRE mode=train"
    if [[ "$BAZEL" == true ]]; then
        echo "  Using Bazel execution"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        echo "  Using runfiles executable: $RUNFILES_EXECUTABLE"
    else
        echo "  Forwarding NGC_API_KEY"
        echo "  Mounting dataset directory: $DATASET_DIR"
        echo "  Mounting nre output directory: $OUTPUT_DIR"
    fi
    echo "  NRE command: ${CMD_TRAIN[@]}"

    # Set bazel run target based on obfuscation flag
    BAZEL_RUN="//internal/scripts/pycena/runtime:pycena_run"
    if [[ "$NO_OBFUSCATED" == true ]]; then
        BAZEL_RUN="//:run"
    fi

    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_TRAIN="bazel run ${BAZEL_RUN} -- ${CMD_TRAIN[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_TRAIN="${RUNFILES_EXECUTABLE} ${CMD_TRAIN[@]}"
    else
        FINAL_CMD_TRAIN="docker run ${DOCKER_ARGS[@]} ${NRE_IMAGE} ${CMD_TRAIN[@]}"
    fi

    # Measure elapsed time for final command
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_TRAIN}" "NGC_API_KEY")"
    start_time_train=$(date +%s%3N)
    eval $FINAL_CMD_TRAIN
    end_time_train=$(date +%s%3N)

    elapsed_time_train=$((end_time_train - start_time_train))
    formatted_time_train=$(convert_time $elapsed_time_train)
    echo "  Elapsed time for train run: ${formatted_time_train}"
    echo "[${TIMESTAMP}] Elapsed time for train run: ${formatted_time_train}" >>${OUTPUT_DIR}/${FILENAME}
    echo "  Wrote file: ${OUTPUT_DIR}/${FILENAME}"
fi


####################
# Validation phase #
####################
if [[ "$MODE" == "trainval" || "$MODE" == "val" ]]; then
    # Find parsed.yaml and last.ckpt and get its path - throw an error if multiple a found.
    PARSED_CONF_FILES=$(find ${OUTPUT_DIR} -name "parsed.yaml")

    if [[ -z "${PARSED_CONF_FILES}" ]]; then
        echo "Error: No 'parsed.yaml' file found."
        exit 1
    fi

    PARSED_CONF_COUNT=$(echo "${PARSED_CONF_FILES}" | wc -l)
    if [[ ${PARSED_CONF_COUNT} -ne 1 ]]; then
        echo "Error: Expected exactly one 'parsed.yaml' file, but found ${PARSED_CONF_COUNT}."
        echo "Files found:"
        echo "${PARSED_CONF_FILES}"
        exit 1
    fi
    PARSED_CONF=$(echo "${PARSED_CONF_FILES}")

    LAST_CKPT_FILES=$(find ${OUTPUT_DIR} -name "last.ckpt")

    if [[ -z "${LAST_CKPT_FILES}" ]]; then
        echo "Error: No 'last.ckpt' file found."
        exit 1
    fi

    LAST_CKPT_COUNT=$(echo "${LAST_CKPT_FILES}" | wc -l)
    if [[ ${LAST_CKPT_COUNT} -ne 1 ]]; then
        echo "Error: Expected exactly one 'last.ckpt' file, but found ${LAST_CKPT_COUNT}."
        echo "Files found:"
        echo "${LAST_CKPT_FILES}"
        exit 1
    fi
    CHECKPOINT=$(echo "${LAST_CKPT_FILES}")

    NRE_OUTPUT_DIR=$(dirname $(dirname "${PARSED_CONF}")) # /path/to/output/uuid/config/parsed.yaml -> /path/to/output/uuid/config -> /path/to/output/uuid
    RUN_ID=$(basename "${NRE_OUTPUT_DIR}") # /path/to/output/uuid -> uuid

    CMD_VAL=(
        --config-name=${PARSED_CONF}
        dataset.path=${DATASET_PATH}
        mode=val
        resume=${CHECKPOINT}
        out_dir=${OUTPUT_DIR}
        logger.run_id=${RUN_ID}
        trainer.world_size=${WORLD_SIZE}
        trainer.num_nodes=${NUM_NODES}  # TODO: multi-node training not tested outside SLURM for now.
        # Makes sure that validation starts from the first frame per sensor.
        dataset.val_camera_frame_start=0
        dataset.val_lidar_frame_start=0
        # Use camera names and frame-end timestamps for output file naming so that validation
        # outputs are directly comparable to rendered outputs without remapping.
        system.test.use_camera_name_dirs=true
        system.test.frame_naming=frame-end-timestamp
    )

    if [[ "$DEBUG" == true ]]; then
        CMD_VAL+=(
            dataset.val_camera_frame_step=100
            dataset.val_lidar_frame_step=100
        )
    fi
    
    # Add any additional validation arguments
    if [ ${#VAL_APPEND[@]} -gt 0 ]; then
        for arg in "${VAL_APPEND[@]}"; do
            CMD_VAL+=("$arg")
        done
    fi

    echo "2. Running NRE mode=val"
    if [[ "$BAZEL" == true ]]; then
        echo "  Using Bazel execution"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        echo "  Using runfiles executable: $RUNFILES_EXECUTABLE"
    else
        echo "  Forwarding NGC_API_KEY"
        echo "  Mounting dataset directory: $DATASET_DIR"
        echo "  Mounting output directory: $OUTPUT_DIR"
    fi
    echo "  NRE command: ${CMD_VAL[@]}"

    if [[ "$BAZEL" == true ]]; then
        FINAL_CMD_VAL="bazel run ${BAZEL_RUN} -- ${CMD_VAL[@]}"
    elif [[ -n "$RUNFILES_EXECUTABLE" ]]; then
        FINAL_CMD_VAL="${RUNFILES_EXECUTABLE} ${CMD_VAL[@]}"
    else
        FINAL_CMD_VAL="docker run ${DOCKER_ARGS[@]} ${NRE_IMAGE} ${CMD_VAL[@]}"
    fi
        
    # Measure elapsed time for final command
    echo "EXECUTE COMMAND: $(safe_print "${FINAL_CMD_VAL}" "NGC_API_KEY")"
    start_time_val=$(date +%s%3N)
    eval $FINAL_CMD_VAL
    end_time_val=$(date +%s%3N)

    elapsed_time_val=$((end_time_val - start_time_val))
    formatted_time_val=$(convert_time $elapsed_time_val)
    echo "  Elapsed time for val run: ${formatted_time_val}"
    echo "[${TIMESTAMP}] Elapsed time for val run: ${formatted_time_val}" >>${OUTPUT_DIR}/${FILENAME}
    echo "  Wrote file: ${OUTPUT_DIR}/${FILENAME}"
fi
