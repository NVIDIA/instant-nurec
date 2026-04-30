<!-- Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved. -->

# Cluster Toolbox

## Prerequisites

Make sure to set-up the NRE development environment by following the instructions in the general [README](../../README.md).

### Maglev

- Maglev CLI executable installed ([Maglev Basic Installation](https://maglev.nvda.ai/docs/getting-started/basic-install/installation)), but this is automatic when running via Bazel.
- The Maglev workflow needs to be authenticated against the W&B server in order to access W&B.
  To this end, register a valid W&B API key as follows via the Maglev CLI executable:
  ```
  maglev secrets set wandb-nvidia-toronto -k WANDB_API_KEY -v <YOUR_WANDB_API_KEY>
  ```
- If your Maglev workflow uses data from SwiftStack, S3 secret keys need to be configured. You can do this via the Maglev CLI executable:
  ```
  maglev storage-secrets set <secret_name_in_workflow> --access-key-id <s3_access_name> --secret-access-key <s3_secret_key>
  ```
  The S3 access name and secret key for your SwiftStack storage can be found on the Core Storage Portal if you click My Storage > click storage in the list > Auth Info tab.

### ORD

- Set up your ORD account by following the instruction in this [guide](https://gitlab-master.nvidia.com/nvr/ord_cluster/-/blob/main/docs/oci_ord_primer.md).
  - For general information about ORD, check their [Confluence](https://confluence.nvidia.com/display/HWINFCSSUP/CS-OCI-ORD)
- Make sure to enable passwordless `ssh` access and have `enroot` setup.

  NOTE: If you still face issues w/ cloning the NRE Repo like

  ```
  git@gitlab-master.nvidia.com: Permission denied (publickey).
  fatal: Could not read from remote repository.

  Please make sure you have the correct access rights
  and the repository exists.
  git@gitlab-master.nvidia.com: Permission denied (publickey).
  fatal: Could not read from remote repository.

  Please make sure you have the correct access rights
  and the repository exists.
  ```

  It's best to create a new SSH Key in your ORD `~/.ssh` following [README](../../README.md#accessing-gitlab)

- Follow the `Authentication` instructions from the root [README](../../README.md#authentication). This should be setup in your ORD `~/.netrc`
- Set up W&B on your ORD account to enable running sweeps. This can be done by adding your W&B API key to your `~/.netrc` file:

  ```bash
  machine api.wandb.ai
  login user
  password <SECRET>
  ```

- A simple mount function also helps viewing / copying files in and out of the ORD Cluster, you can add this to your local `~/.bashrc`
  ```bash
    mount_ord() {
      local remote_path="${1:-/lustre/fsw/portfolios/nvr/users/<nvidia-username>}"
      sshfs <nvidia-username>@cs-oci-ord-login-02.nvidia.com:"$remote_path" /path/to/local/folder
    }
  ```

### NGC

_TBD once this is integrated._

## Usage

NRE provides a simple CLI to submit training jobs on the clusters available to us (see [prerequisites](#prerequisites)).

This is intended to be used from your local workstation as follows:

- via Python:

  ```bash
  python workflows/cluster_toolbox/run_cluster_toolbox.py
      --cluster-name <maglev/ord/ngc> \
      --config-name <cluster-config> \
      <COMMAND> [ARGS]...
  ```

- via Bazel:

  ```bash
  bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
      --cluster-name <maglev/ord/ngc> \
      --config-name <cluster-config> \
      <COMMAND> [ARGS]...
  ```

There are three main commands defining the supported functionalities:

1. [submit-job](#submitting-a-job)
2. [submit-wandb-sweep-job](#submitting-a-weights-and-biases-sweep)
3. [submit-ray-sweep-job](#submitting-a-ray-sweep-job)

## Supported use-cases

### Execution Modes

#### Docker mode

Docker mode will rely on utilizing a docker image that contains the code version you want to run, either by:

1. building and pushing the current state of the repo on your local machine and push that to the `nvcr.io`
2. or utilizing a predefined image (when setting `docker.build_push=false` and choosing `docker.image=nvcr.io/<YOUR-IMAGE>`)

_Note that you can use the images that are built from CI/CD._

#### Bazel mode (currently only on ORD)

Bazel mode will clone the repo on the cluster and checkout the desired `git_commit` commit and rely on `bazel run` with a persistent cache on the cluster to run the code. This avoids building and pushing new Docker images for every code change meaning the jobs will start faster. Results in bazel mode are written to the job directory automatically mounted at `/results`.

The Bazel Mode environment is provided by a Docker container defined in this [Dockerfile](../../scripts/cluster_toolbox/Dockerfile-dev-bazel.build). This container is manually maintained by the NRE team and is generally stable.

If you need to update the environment, rebuild the Docker container by running this [script](../../scripts/cluster_toolbox/build_dev_bazel_docker.sh), and then perform an `enroot` squashing on ORD with the following command:

```bash
srun -A <ACCOUNT> --partition interactive --gpus 1 --pty enroot import --output /lustre/fsw/portfolios/nvr/users/<USER>/containers/<SQUASH> 'docker://$oauthtoken@nvcr.io#<IMAGE>'
```

Built docker files are encouraged to be put under `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/` for easy access and sharing.

Prebuilt versions of the Docker container are also available at:

- nvcr.io: `nvcr.io/nvidian/ct-toronto-ai/nre_run_ord_bazel_dev`
- ORD: `/lustre/fsw/portfolios/nvr/users/qiwu/containers`
- ORD: `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers`

**Important** In this Bazel mode container, the Bazel executable is made available through the exported `bazelisk` or `bazel` function wrappers. It is strongly recommended not to invoke the Bazel executable using its absolute path, as this bypasses the cluster-specific `bazelrc` settings. Doing so may result in improperly configured Bazel cache locations or application crashes.

#### Docker-Python mode (currently only on ORD)

Docker-Python mode will pre-install all necessary python packages and pre-compile all the C++/CUDA libraries in a `conda` environment inside the Docker container. When launching the job, the conda environment is fixed and the target commit (set via `git_commit`) will be cloned and mounted under `/workspace`. This mode is useful for development and NRM training process as it enables fast iteration and debugging where the python code changes frequently but the C++/CUDA part changes less frequently.

If you need to build such a Docker container environment, please refer to this [script](../../scripts/cluster_toolbox/build_dev_python_docker.sh), and then perform an `enroot` squashing on ORD with the following command:

```bash
srun -A <ACCOUNT> --partition interactive --gpus 1 --pty enroot import --output /lustre/fsw/portfolios/nvr/users/<USER>/containers/<SQUASH> 'docker://$oauthtoken@nvcr.io#<IMAGE>'
```

Built docker files are encouraged to be put under `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers/` for easy access and sharing.

Prebuilt versions of the Docker container are also available at:

- nvcr.io: `nvcr.io/nvidian/nre`
- ORD: `/lustre/fsw/portfolios/nvr/users/qiwu/containers`
- ORD: `/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/containers`

### Submitting a job

The `submit-job` command of the toolbox automates the launch of a job to your selected cluster.

#### On Maglev

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
    --cluster-name maglev \
    --config-name maglev.yaml \
    submit-job \
    --job-template-path workflows/cluster_toolbox/job_templates/maglev/workflow_template.yaml \
    --job-name test_maglev
```

#### On ORD in bazel mode:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord \
  --config-name ord.yaml \
  submit-job user=<YOUR-USERNAME> git_commit=<COMMIT-OR-BRANCH> \
  --job-name test_ord \
  --command "bazelisk run //:run -- --version"
```

#### On ORD in docker mode:

Build and push docker image:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord \
  --config-name ord.yaml \
  submit-job user=<YOUR-USERNAME> git_commit=<COMMIT-OR-BRANCH> exec_mode=docker docker.build_push=true \
  --job-name test_ord \
  --command "./run_image.binary --config-name apps/AV/NV/ncore_dnsg_calib_nonrigid.yaml out_dir=/results dataset.path=/lustre/fsw/portfolios/nvr/users/rdelutio/data/nre/76b46042-ba8f-11eb-a9b2-00044baf74dc@1621641716100213-1621641729800003/76b46042-ba8f-11eb-a9b2-00044baf74dc@1621641716100213-1621641729800003/76b46042-ba8f-11eb-a9b2-00044baf74dc@1621641716100213-1621641729800003.json"
```

Using an existing docker image:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml \
  submit-job user=<YOUR-USERNAME> exec_mode=docker docker.build_push=false \
  docker.image=<YOUR-DOCKER-IMAGE> \
  --job-name test_ord \
  --command "./run_image.binary --version"
```

#### On ORD in docker-python mode:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord \
  --config-name ord.yaml \
  submit-job user=<YOUR-USERNAME> exec_mode=docker_python num_nodes=2 num_gpus=8 git_commit=aed8ce \
  --job-name test_ord \
  --command "python run.py \
    --config-name=configs/nrm/celsius/static_ld.yaml \
    trainer.num_nodes=2 \
    logger.run_id=test_test \
    out_dir=/results \
    resume=auto \
    dataset.train.ncore_data_list_path=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/nre-ff/data_split/train-accum.lst \
    dataset.val.ncore_data_list_path=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/nre-ff/data_split/val-accum.lst \
    dataset.test.ncore_data_list_path=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/nre-ff/data_split/val-accum.lst"
```

> Remember to `scancel` your job if you are no longer interested in training this model.

### Submitting a Weights and Biases sweep

The `submit-wandb-sweep-job` command of the toolbox automates the launch of a [W&B sweep](<(https://docs.wandb.ai/guides/sweeps)>) and submitting the corresponding jobs executing the agents to your selected cluster.

#### General overview of a sweep

1. **Define sweep configuration** (typically in a YAML file): specify hyperparameter values to sweep through, the sampling strategy, and an execution command or training script for each sample of hyperparameters.
2. **Create sweep** on W&B from the sweep configuration (typically a YAML file).
   Launches a Sweep Controller (on the W&B server) that samples individual configurations (hyperparameter sets) according to the sweep configuration and waits for connecting agents that each execute the command or training script for a sample of hyperparameter values.
3. **Launch agents** on your machine or cluster in our case.
   Each connected agent receives a run configuration (hyperparameter sample) and the command or script to run from the Sweep Controller. Agents keep receiving runs to execute until all configurations sampled by the Controller are exhausted.
   In our case, each cluster job executes one instance of the W&B agent that can receive zero or more runs to execute.

#### Launching W&B sweeps on Maglev:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name maglev \
  --config-name maglev.yaml \
  submit-wandb-sweep-job \
  --sweep-name test_maglev_sweep \
  --sweep-conf-path workflows/cluster_toolbox/wandb_sweep_configs/maglev/example.yaml \
  --num-agents 2
```

#### Launching W&B sweeps on ORD:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord \
  --config-name ord.yaml \
  submit-wandb-sweep-job \
  --sweep-name test_ord_sweep \
  --sweep-conf-path workflows/cluster_toolbox/wandb_sweep_configs/ord/example.yaml \
  --num-agents 2

bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml \
  submit-wandb-sweep-job user=$USER wandb.tags=[a100,mipnerf360] partition=[interactive,polar3,polar4] \
    --sweep-name 3dgut/mipnerf360_$(date +%Y%m%d%H%M%S) \
    --sweep-conf-path workflows/cluster_toolbox/wandb_sweep_configs/ord/3dgut/mipnerf.yaml \
    --num-agents 9

bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml \
  submit-wandb-sweep-job user=$USER wandb.tags=[a100,waymo] \
    --sweep-name 3dgut/waymo_$(date +%Y%m%d%H%M%S) \
    --sweep-conf-path workflows/cluster_toolbox/wandb_sweep_configs/ord/3dgut/waymo.yaml \
    --num-agents 4
```

Note: Most ORD partitions have a 4 hour runtime limit. Shortly before that limit, the cluster toolbox sends a signal to the NRE training process that is used to requeue the associated slurm job to resume from the latest saved checkpoint. Please refer to [this template](wandb_sweep_configs/ord/autoresume_example.yaml) for an example of how to configure your sweeps to take advantage of this auto-resume feature.

_Note: for convenience the [workflows/cluster_toolbox/\*/personal/] path are git ignored, in order to store any personal configs not intended to be shared as the path is ignored by git._

### Submitting a Ray sweep job

W&B sweeps are useful for small number of runs, but for larger number of runs (e.g. data processing for NRM), it becomes buggy. Please refer to [this document](nre/nrm/docs/BATCH_DATA.md) for more details on how to use the ray backend to scale up the jobs.

## Utilities

### `wandb` cli script

For convenience, the `wandb` cli script of the currently used `wandb`
package is available as `//internal/workflows/cluster_toolbox:wandb` and can be run with

```bash
bazel run //internal/workflows/cluster_toolbox:wandb -- <cli options>
```
