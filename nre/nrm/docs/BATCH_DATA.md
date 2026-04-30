# Batch Data Processing

In many cases we want to process data in large scale on ORD. This document describes a basic workflow to leverage the ORD toolbox's to scale jobs on the cluster.

Historically, we leverage the wandb sweep agents to perform the jobs. However it suffers from the following limitations:

- The scheduler becomes buggy and often dequeue duplicated runs for preempted jobs. For >3K runs it will even dequeue un-scheduled jobs redundantly.
- Users have no explicit control over the scheduler's behavior.

Therefore, we recommend to use the `ray` backend to scale up the jobs. However, if you still prefer the `wandb` sweep approach, you can still use it. This file contains a workflow that incorporates both approaches.

## Step 1: Define your bazel target

The first step is to define a bazel target that can be triggered via `bazel run`. This target should incorporate the minimal working chunk of your large-scale processing, such as processing a specific sequence.
Of course, you can use existing bazel targets such as `//:run`

## Step 2: Define your `wandb`-compatible sweep config

Follow `internal/workflows/cluster_toolbox/wandb_sweep_configs/ord/3dgut/*.yaml` as examples to define your sweep. It can be something like:

```yaml
method: grid
parameters:
  clip_ids:
    values:
      - clipgt-a4fc9340-5ab6-4a93-b83a-dfe856b057b8
  traj_ids:
    values:
      - 0
      - 1
command:
  - /bin/bash
  - -euxc
  - |
    eval "$0"
    eval "$1"

    # Your main command
    bazel run //apps/aux_gen:ncore_aux_data -- \
      --shard-file-pattern /lustre/data/${clip_ids}/gen3c/${traj_ids}.zarr.itar \
      --output-dir /lustre/data/${clip_ids}/gen3c \
      --no-lidar-seg-camvis \
      --no-ego-mask

    # Two environment variables are available to control the job logic:

    # Touch this file if you think the ray worker should be re-queued.
    # We recommend creating this file so that one doesn't need to manually launch ray workers
    # after preemption.
    touch $SHOULD_REQUEUE_PATH

    # Touch this file if the ray worker should immediately die. It is useful if each of your 
    # job is very long (e.g. 3 hours), so the rest time before preemption will not cover another
    # job and you don't have a good resume mechanism.
    # touch $RAY_SHOULD_STOP_PATH

  - ${args_no_hyphens}

# This section is only read by the ray backend. It specifies the resource request for each job.
# You can use fractions (e.g. num_gpus=0.5) so that 2 jobs can be run concurrently on the same GPU.
ray_resource_request:
  num_cpus: 8
  num_gpus: 1
```

## Step 3: Launch the sweep (using the `wandb` backend)

Launch the toolbox such as:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml submit-wandb-sweep-job \
  user=YOUR_USERNAME team=YOUR_PPP partition=[grizzly,polar,polar3,polar4] \
  keep_workspace=true wandb.api_key=YOUR_API_KEY wandb.max_job_per_agent=null \
  --num-agents 2 \
  --sweep-name SWEEP_NAME \
  --sweep-conf-path PATH_TO_ABOVE_YAML.yaml
```

The above script is going to:

1. Submit a sweep job to `wandb` server.
2. Launch 2 single-node jobs and trigger `wandb agent <SWEEP_ID>` on each node, so that each node could receive jobs from wandb server and execute the above bazel commands.

> The agents could be elastically added by running `sbatch submit.sh` on ORD working directory, or killed by `scancel` the corresponding jobs.

## Step 3: Launch the sweep (using the `ray` backend)

First we need to have a long-running head node to run the `ray` scheduler. The scheduler plays the same role as the `wandb` server, but run locally on the head node. The head node has to be accessible from the compute nodes on the cluster. A good choice is to use a long-running CPU node on ORD cluster, which could be launched by:

```bash
#!/bin/bash
#SBATCH --job-name=nre-cpu-long
#SBATCH --time=6-23:59:00
#SBATCH --account=YOUR_PPP
#SBATCH --partition=cpu_long
#SBATCH --cpus-per-task=96

sleep 6d 23h
scontrol requeue $SLURM_JOB_ID
```

After the job is launched, you will need to obtain the IP address of the head node. One can also use the VSCode node if available on the cluster.

Then launch the toolbox such as:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
  --cluster-name ord --config-name ord.yaml submit-ray-sweep-job \
  user=YOUR_USERNAME team=YOUR_PPP partition=[grizzly,polar,polar3,polar4] \
  keep_workspace=true wandb.api_key=YOUR_API_KEY \
  --num-workers 2 \
  --job-name JOB_NAME \
  --head-node-address HEAD_NODE_IP \
  --sweep-conf-path PATH_TO_ABOVE_YAML.yaml
```

> The last argument is optional, and if not provided it will just launch the ray workers instead.

The above script is going to:

1. SSH into the above node, download the scheduler code & bazel binary under `exec_path.remote_ray_head`.
2. Launch the ray daemon on the head node.
3. Launch the scheduler on the head node based on the sweep configuration.
4. Launch the ray workers on ORD with `ray start --address=<HEAD_NODE_IP>:6379`.
5. Logs for running the jobs will be put under `exec_path.remote_ray_head/<JOB_NAME>-<TIMESTAMP>/`.
