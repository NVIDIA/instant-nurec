# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

GITLAB_MASTER_URL = "ssh://git@gitlab-master.nvidia.com:12051/nrs/nre.git"

ORD_DEFS = """#!/bin/bash
# - Job Configuration
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
# - Node Configuration
#SBATCH --nodes={num_nodes}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --array=1-{num_jobs}
# - Misc
#SBATCH --output={log}_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@300 # Send SIGUSR1 300 seconds before preemption.
{segment_string}

set -eu -o pipefail
{job_info}

# ----------------------------------------------------
# ORD definitions

ACCOUNT="{account}"
PARTITION="{partition}"
NUM_NODES="$SLURM_NNODES"
NUM_GPUS="$SLURM_NTASKS_PER_NODE"

export SRUN_LOG={log}
export GIT_COMMIT="{git_commit}"
export DOCKER_LINK="{docker_link}"
export DOCKER_SQSH="{docker_sqsh}"
export DOCKER_PYTHON_SQSH="{docker_python_sqsh}"
export DEV_IMAGE="{bazel_dev_image}"
export DEV_CACHE="{bazel_dev_cache}"
export TMP_CACHE="{bazel_tmp_cache}"
export SCRATCH="{scratch_prefix}/scratch_$UNIQUE_JOB_ID"
export SCRATCH_IN_CONTAINER="{scratch_in_container}"

export BAZEL_CACHE_PREFILL="{bazel_cache_prefill}"

TIMEOUT="{timeout}"

MOUNT="{mount}"
LAUNCH_SCRIPT="{launch_script}"

# Make sure the script is executable
chmod +x $LAUNCH_SCRIPT
"""

ORD_JOBINFO = """
if [[ -z $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_JOB_ID="${SLURM_JOB_ID}"
else
    export UNIQUE_JOB_ID="${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
fi
echo "SLURM JOB ID: $UNIQUE_JOB_ID"
cd "{remote_dir}"

# check resources
scontrol show job -d $UNIQUE_JOB_ID | grep GRES

# The file used to log the current process ID (PID). This file will be referenced later to send a SIGUSR1 signal for gracefully terminating the wandb process.
export PID_PATH="/scratch/$UNIQUE_JOB_ID.pid"

# Resolve the host-side path backing the container's /scratch dir.
# In default mode this is the bind-mounted $SCRATCH. In SCRATCH_IN_CONTAINER=1 mode
# there is no bind mount; instead /scratch lives inside the per-job pyxis container
# rootfs on node-local /raid (auto-cleaned by pyxis when the container exits), so we
# glob for the rootfs path. Must be called while the container is running.
function resolve_scratch_host_path()
{
    if [ "${SCRATCH_IN_CONTAINER:-0}" = "1" ]; then
        # pyxis names the container rootfs dir with SLURM_JOB_ID (the per-task job id),
        # not UNIQUE_JOB_ID (which is the array-composite). The .<STEP> suffix varies
        # with srun call order; globbing picks the current job's rootfs unambiguously.
        local rootfs
        rootfs=$(ls -dt /raid/enroot/data/user-$(id -u)/pyxis_${SLURM_JOB_ID}_${SLURM_JOB_ID}.* 2>/dev/null | head -1)
        if [ -z "$rootfs" ]; then
            return 1
        fi
        echo "${rootfs}/scratch"
    else
        echo "$SCRATCH"
    fi
}

# Set up a handler to pass the SIGUSR1 to the training process, as the wandb agent otherwise does
# not propagate signals to the training process (https://github.com/wandb/wandb/issues/3667).
function term_handler()
{
    PID=$(cat "$(resolve_scratch_host_path)/$UNIQUE_JOB_ID.pid")
    while kill -SIGUSR1 "$PID" 2>/dev/null
    do
        echo "Sent SIGUSR1 signal to $PID"
        sleep 10
        # Check the job is about to end, if so, stop trying.
        TIME_LEFT=$(squeue -j $UNIQUE_JOB_ID -o "%L" -h | awk -F: '{ if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1 }')
        if [[ $TIME_LEFT -lt 10 ]]; then
            echo "Job is about to end, stopping signal loop."
            break
        fi
    done

    scontrol requeue $UNIQUE_JOB_ID
}
"""

ORD_PREPARE_SHARED_WORKSPACE = """
# Prepare a shared workspace
echo ----------------------------------------
WORKSPACE=$PWD/workspace # NOTE $PWD is the job directory

# Create a critical section to prevent multiple jobs from cloning the workspace at the same time.
exec {FD}<>$WORKSPACE.lock && flock $FD

if [ -d "$WORKSPACE" ]; then
    echo "Workspace already exists. Skipping clone."
else
    # Use an ssh-agent in case multiple SSH keys exist (we find in ~/.ssh and add all private keys with a .pub public key).
    eval "$(ssh-agent -s)" && for k in ~/.ssh/*.pub; do [[ -f "${k%.pub}" ]] && ssh-add "${k%.pub}"; done
    git clone {GITLAB_MASTER_URL} "$WORKSPACE"
    pushd "$WORKSPACE"
    git checkout "$GIT_COMMIT"
    echo "commit=" $(git rev-parse HEAD)
    popd
fi

# Release the lock
flock -u $FD && exec {FD}<&-

echo ----------------------------------------
""".replace("{GITLAB_MASTER_URL}", GITLAB_MASTER_URL)

ORD_PREPARE_PRIVATE_WORKSPACE = """
# Set up a _partial_ clone of the target commit on each node following the solutiuon proposed here: https://stackoverflow.com/a/43136160
# However, we can't fetch with --depth 1 as otherwise we'd be missing the merge commit history to compute the version via the status commands.
echo ----------------------------------------

WORKSPACE=$(dirname $SCRATCH)/workspace_$UNIQUE_JOB_ID
cat > setup_workspace_$UNIQUE_JOB_ID.sh << EOF
#!/bin/bash
set -eu -o pipefail

# Remove WORKSPACE if it exists (in case of a requeued job)
rm -fr $WORKSPACE

mkdir -p $WORKSPACE
cd $WORKSPACE
git init .
git remote add origin {GITLAB_MASTER_URL}
git fetch origin $GIT_COMMIT  # GIT_COMMIT must be a branch name or a long hash code
git checkout FETCH_HEAD
EOF

srun --ntasks-per-node 1 --ntasks $NUM_NODES bash setup_workspace_$UNIQUE_JOB_ID.sh

echo ----------------------------------------
""".replace("{GITLAB_MASTER_URL}", GITLAB_MASTER_URL)

ORD_LAUNCH_PRE_COMMANDS_DOCKER = """
"""

ORD_TEMPLATE_DOCKER = """
# 1. If the container cache directory doesn't, bail out
if [ ! -e $(dirname $DOCKER_SQSH) ]; then
    echo "Container cache directory doesn't exist: $(dirname $DOCKER_SQSH)"
    exit 1
fi

# 2. Create a squash image if it doesn't exist in the container cache directory
if [ -e $DOCKER_SQSH ]; then
    echo "Using existing squash image: $DOCKER_SQSH"
else
    echo "Creating squash image: $DOCKER_SQSH"
    enroot import --output $DOCKER_SQSH docker://$DOCKER_LINK
fi

# 3. Mount the data directory
mkdir -p "$PWD/results" # make sure the results directory exists
srun --ntasks-per-node 1 --ntasks $NUM_NODES bash -c "mkdir -p $SCRATCH $TMP_CACHE"

# 4. Launch the job script in the container
MOUNT="${MOUNT:+$MOUNT,}$PWD/results:/results"
MOUNT="${MOUNT:+$MOUNT,}$SCRATCH:/scratch"

set +e  # Cancel exit immediately mode
srun --container-image=$DOCKER_SQSH --container-mounts=$MOUNT --job-name="$SLURM_JOB_NAME" bash -c "$LAUNCH_SCRIPT" &

child="$!"
trap term_handler SIGUSR1  # program killed by timeout - requeue
wait "$child"
"""

ORD_LAUNCH_PRE_COMMANDS_BAZEL = """
# Ensure /scratch and /runcache exist inside the container. When SCRATCH_IN_CONTAINER
# is set the outer script skips the host-side mkdir and bind mount, so these dirs are
# just regular subdirectories of the container rootfs. mkdir -p is a no-op when the
# bind-mount flow created them as mount points already.
mkdir -p /scratch /runcache

# Create per-rank local bazel cache
export RUNCACHE=/runcache/cache_${SLURM_LOCALID}

# In case of a requeued job with same job id and node, the cache might already exist, we need to skip so that tar doesn't fail.
if [ -d "$RUNCACHE" ] && [ "$(ls -A $RUNCACHE)" ]; then
    echo "RUNCACHE directory $RUNCACHE already exists and is not empty, skipping cache population."
else
    mkdir -p $RUNCACHE

    # Prepopulate bazel cache using a good heuristic
    BAZEL_CACHE_FILE=${BAZEL_CACHE_PREFILL}-rank${SLURM_LOCALID}.tar
    if [ -f "$BAZEL_CACHE_FILE" ]; then
        echo "Populating Bazel cache using $BAZEL_CACHE_FILE" ...
        tar --strip-components=2 -xf $BAZEL_CACHE_FILE -C $RUNCACHE
    else
        echo "Bazel cache file $BAZEL_CACHE_FILE do not exist, skip it!!"
    fi
fi

# Create per-rank local configuration
cat > $RUNCACHE/bazelrc << EOF
startup  --host_jvm_args=-Xlog:os+container=trace
build    --noexperimental_collect_system_network_usage
build    --nogenerate_json_trace_profile
build    --disk_cache=/keepcache/disk_cache
test     --disk_cache=/keepcache/disk_cache
build    --repository_cache=
test     --repository_cache=
build    --repo_contents_cache=
test     --repo_contents_cache=
build    --config=skip_nrend_obfuscation
test     --config=skip_nrend_obfuscation
startup  --output_base=/runcache/cache_main
build    --symlink_prefix=/tmp/rank${SLURM_LOCALID}_
test     --symlink_prefix=/tmp/rank${SLURM_LOCALID}_
EOF

# Overwrite system executable
bazel() {
  /usr/local/bin/bazelisk --bazelrc $RUNCACHE/bazelrc "$@"
}
export -f bazel

bazelisk() {
  /usr/local/bin/bazelisk --bazelrc $RUNCACHE/bazelrc "$@"
}
export -f bazelisk

# Avoid race conditions when downloading bazel
flock /tmp/bazel_install.lock -c "/usr/local/bin/bazelisk --version"

# Done
echo "Preparing Bazel environment, took $((SECONDS))s"
"""

ORD_TEMPLATE_BAZEL = """
# 1. Setup directories (we have to use srun to create directories local to each node, e.g., directories under /raid/scratch)
mkdir -p "$PWD/results"
if [ "${SCRATCH_IN_CONTAINER:-0}" = "1" ]; then
    # In SCRATCH_IN_CONTAINER mode /scratch and /runcache live inside the per-job
    # pyxis container rootfs (auto-cleaned by pyxis on container exit), so we do not
    # create or bind-mount $SCRATCH/$TMP_CACHE on the host. $DEV_CACHE still lives on
    # shared (lustre) storage, so one mkdir suffices.
    mkdir -p $DEV_CACHE
else
    srun --ntasks-per-node 1 --ntasks $NUM_NODES bash -c "mkdir -p $SCRATCH $TMP_CACHE $DEV_CACHE"
fi

# Create a lock directory to prevent race conditions with wandb agent (which can otherwise dequeue the same preempted job multiple times)
mkdir -p "$PWD/wandb_agent_lock"
touch "$PWD/wandb_agent_lock/lock"
export SHOULD_REQUEUE_PATH="/scratch/$UNIQUE_JOB_ID.should_requeue"

# 2. Run the job
MOUNT="${MOUNT:+$MOUNT,}$PWD/results:/results"
if [ "${SCRATCH_IN_CONTAINER:-0}" != "1" ]; then
    MOUNT="${MOUNT:+$MOUNT,}$SCRATCH:/scratch"
fi
MOUNT="${MOUNT:+$MOUNT,}$WORKSPACE:/workspace"
# - bazel specific file structure
if [ "${SCRATCH_IN_CONTAINER:-0}" != "1" ]; then
    MOUNT="${MOUNT:+$MOUNT,}$TMP_CACHE:/runcache"
fi
MOUNT="${MOUNT:+$MOUNT,}$DEV_CACHE:/keepcache"
# - authentication specific file structure
MOUNT="${MOUNT:+$MOUNT,}$HOME/.netrc:$HOME/.netrc"
if [ -f $HOME/.aws/config ]; then
    MOUNT="${MOUNT:+$MOUNT,}$HOME/.aws/config:$HOME/.aws/config"
fi
if [ -f $HOME/.aws/credentials ]; then
    MOUNT="${MOUNT:+$MOUNT,}$HOME/.aws/credentials:$HOME/.aws/credentials"
fi
# - wandb specific file structure
MOUNT="${MOUNT:+$MOUNT,}$PWD/wandb_agent_lock:/wandb_agent_lock"

set +e  # Cancel exit immediately
# --job-name: srun step defaults to the command name ("bash") which makes
# pytorch_lightning's SLURMEnvironment.detect() treat the run as interactive
# and fall back to LightningEnvironment (MASTER_ADDR=127.0.0.1). Inherit the
# sbatch job name so PL's SLURM auto-detection fires and MASTER_ADDR is set
# to the head node hostname.
srun --container-image="$DEV_IMAGE" \
     --container-mounts=$MOUNT \
     --container-workdir=/workspace \
     --no-container-remap-root \
     --no-container-mount-home \
     --job-name="$SLURM_JOB_NAME" \
     bash -c "$LAUNCH_SCRIPT" &

child="$!"
trap term_handler SIGUSR1  # program killed by timeout - requeue
wait "$child"

# Read the requeue marker from the host-side scratch path. In SCRATCH_IN_CONTAINER
# mode this is inside the pyxis rootfs which may already be cleaned up by the time
# we reach this point; resolve_scratch_host_path will return empty and the -f test
# will simply be false (so the marker-based requeue path is effectively not usable
# in SCRATCH_IN_CONTAINER mode — rely on the term_handler path instead).
SCRATCH_CHECK_DIR=$(resolve_scratch_host_path || true)
if [ -n "$SCRATCH_CHECK_DIR" ] && [ -f "${SCRATCH_CHECK_DIR}/${UNIQUE_JOB_ID}.should_requeue" ]; then
    rm "${SCRATCH_CHECK_DIR}/${UNIQUE_JOB_ID}.should_requeue"
    echo "Requeuing job $UNIQUE_JOB_ID"
    scontrol requeue $UNIQUE_JOB_ID
fi
"""

ORD_LAUNCH_PRE_COMMANDS_DOCKER_PYTHON = """
# Since we remove project root in nre.run.__init__.py, we need to make sure code is available for python.
# Print error only if this fails.
flock pip_install.lock -c "rm -rf nre.egg-info; pip install -e ." >/tmp/install.log 2>&1 || { cat /tmp/install.log; exit 1; }
"""

ORD_TEMPLATE_DOCKER_PYTHON = """
# 1. Setup directories (we have to use srun to create directories local to each node, e.g., directories under /raid/scratch)
mkdir -p "$PWD/results" # make sure the results directory exists
srun --ntasks-per-node 1 --ntasks $NUM_NODES bash -c "mkdir -p $SCRATCH $TMP_CACHE"

# 2. Run the job
MOUNT="${MOUNT:+$MOUNT,}$PWD/results:/results"
MOUNT="${MOUNT:+$MOUNT,}$SCRATCH:/scratch"
MOUNT="${MOUNT:+$MOUNT,}$WORKSPACE:/workspace"

set +e  # Cancel exit immediately mode
srun --container-image="$DOCKER_PYTHON_SQSH" \
     --container-mounts=$MOUNT \
     --container-workdir=/workspace \
     --no-container-mount-home \
     --no-container-remap-root \
     --job-name="$SLURM_JOB_NAME" \
     bash -c "$LAUNCH_SCRIPT" &

child="$!"
trap term_handler SIGUSR1  # program killed by timeout - requeue
wait "$child"
"""
