# NRE Quality Benchmarking

## Objective

The objective of benchmarking is to track the quality of the reconstructions (and rendering) as the NRE codebase evolves.

The benchmark currently evaluates a deployed Docker image - containing Neural Reconstruction executables - on a fixed set of benchmark datasets, each with a fixed set of settings (a configuration). Each dataset and configuration can be evaluated independently, providing an opportunity for parallelization.

## Benchmarking as W&B sweeps on Maglev

The benchmarking is based on Weights & Biases (W&B) sweeps. Each sweep is launched as a workflow on Maglev by using the [Cluster Toolbox](../README.md). The benchmark runs in a Docker image, built from a specific version of the code and tagged with that version. A benchmark is implemented as a special W&B sweep:

- The sweep is a "grid search" over a fixed set of datasets and configurations, defined by the `benchmark_dataset` and `benchmark_config` variables in the [sweep configuration](../wandb_sweep_configs/maglev/benchmark_sweep.yaml), which are added to the hyperparameters of each run on W&B.
- Each experiment or run of a sweep is an NRE training (`mode=train`) and a testing (`mode=val`) launched on a single dataset by using a single config. Training logs into a W&B run, and testing logs additional test metrics into that same run.
- The name of the sweep (submitted to the W&B server) is automatically generated to reflect the name of the Docker image being evaluated and its version tag (which contains a semantic version and a short git commit hash).
- The [sweep configuration](../wandb_sweep_configs/maglev/benchmark_sweep.yaml) and the [Maglev workflow specification](../job_templates/maglev/benchmark_workflow.yaml) are predefined and are not supposed to be changed by the user.
- The parsed NRE training configuration is forwarded to W&B directly as [wandb.config](https://docs.wandb.ai/guides/track/config/).
- Benchmark datasets swept over are in a predefined container in the PDX cluster (SwiftStack storage at swift://pdx.s8k.io/team-nre-benchmarking/benchmark_data), organized under `<dataset_version>/<clip_name>`.
- [W&B tags](https://docs.wandb.ai/guides/runs/tags/) are labelling each run of a sweep and are used to filter runs when summarizing results on W&B. All W&B runs of a benchmark sweep will be tagged with a `benchmark` label, and further predefined tags are added as well, e.g. to distunguish results from `stage` vs. `dev` images.
- Code version information (incl. commit date and version string) is automatically parsed from a `version_file.yaml` stored inside the Docker image under test (see Bazel rule `//bazel/version:version_file_yaml`). It is appended to the program configuration (`config.version`) and submitted to W&B. `config.version` is used to visualize the history of benchmark results over GitLab commits.
- The Maglev workflow is named `benchmark-<image_name>-<image_version>-sweep-<sweep_id>` (all special characters replaced with dashes), e.g. `benchmark-nre-run-stage-0-1-19-be1cbf17-sweep-xq7kz1s8` when evaluating the `nre_run_stage:0.1.19-be1cbf17` image in a W&B sweep with ID `xq7kz1s8`. Maglev workflows can be browsed at https://maglev.nvda.ai/ide/workflows.

## Authentication

Any user launching a benchmark must provide a valid Maglev API key to the [Cluster Toolbox](../README.md) in an environment variable `MAGLEV_API_KEY` or in a config option (see [maglev.yaml](../cluster_configs/maglev.yaml)) so that the Toolbox can autheticate against Maglev and submit jobs. The Toolbox interacts with Maglev by using the Maglev CLI. In a Bazel environment, the Maglev CLI is made available automatically as a dependency of the Bazel target launching the benchmark. (For a manual installation of the Maglev CLI, follow the [Maglev basic installation guide](https://maglev.nvda.ai/docs/getting-started/basic-install/installation#log-in-to-maglev).)

The running [Maglev workflow](../job_templates/maglev/benchmark_workflow.yaml) needs access to the benchmark datasets in the SwiftStack container.
The corresponding S3 secret key is shown at the [Core Storage Portal](https://cssportal.sre.nsv.nvidia.com:4443/) for those having access to the`team-nre-benchmarking` storage space.
The key needs to be set only once per user, it will be stored in the cluster, and made available to Maglev workflows triggered by the same user:

```
maglev storage-secrets set swift-team-nre-benchmarking \
 --access-key-id swift-team-nre-benchmarking \
 --secret-access-key <s3_secret_key>
```

All W&B agents inside the Maglev workflow need access to the W&B server and project. The agents pick up the W&B API key from the `WANDB_API_KEY` environment variable automatically (see [workflow](../job_templates/maglev/benchmark_workflow.yaml)). The latter needs to be set once per user triggering the benchmark:

```
maglev secrets set wandb-nvidia-toronto -k WANDB_API_KEY -v <YOUR_WANDB_API_KEY>
```

Assuming you have a W&B account, and you logged in at [wandb.ai](https://wandb.ai), you can find your W&B API key [here](https://wandb.ai/authorize). You need to be in the `nvidia-toronto` team on W&B to successfully submit results to the `nre-benchmark` project.

## Running a benchmark sweep

Assuming [authentication](#authentication) is set up correctly, and [prerequisites](../README.md#prerequisites) are met, a benchmark sweep can be launched on Maglev by running the following from the repo root:

```
bazel run //internal/workflows/cluster_toolbox/benchmark:run_benchmark
```

This rule launches the script [run_benchmark.py] which uses the [Cluster Toolbox](../README.md) with the benchmark settings composed from [benchmark.yaml](../cluster_configs/benchmark.yaml), [maglev.yaml](../cluster_configs/maglev.yaml) and [base.yaml](../cluster_configs/base.yaml) via [Hydra](https://hydra.cc/).

These settings define the Maglev resource share, the number of workers (W&B sweep agents), the required Maglev cluster node type (hence, the GPU model to run training on), the W&B group and project for logging, login credentials, and W&B tags to tag the experiments. The default settings in the config files can be overridden via command line using Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)), for example:

```
bazel run //internal/workflows/cluster_toolbox/benchmark:run_benchmark -- \
  --num-agents ${MAGLEV_WORKERS} \
  --verbose \
  docker.image=${IMAGE} \
  docker.build_push=false \
  wandb.api_key=${WANDB_API_KEY} \
  resource_share=${MAGLEV_RESOURCE_SHARE} \
  node_type=${MAGLEV_NODE_TYPE} \
  wandb.tags=\[${SWEEP_TAG}\]
```

## Benchmark launch from CI/CD

### CI/CD jobs

- `benchmark_stage_image` job: Scheduled to automatically run **nightly** on the latest published `stage` Docker image from the `main` branch, and adds the W&B tag `stage` to each W&B run of the benchmark sweep. (The `stage` image is published from every merge commit of `main`.) The nightly run is because running the full benchmark for every merge commit to `main` would be too expensive.
- `benchmark_dev_image` job: Meant to trigger a benchmark manually from any commit of a feature branch or MR before merging to `main`, and adds the W&B tag `dev` to each W&B run of the benchmark sweep

> :warning: Every manual launch uses cluster resources (currently 4 GPUs for several hours), so think twice before triggering a benchmark manually.

### Authentication in CI/CD

The [keys used for authentication](#authentication) are injected through masked GitLab CI/CD variables into the benchmarking CI/CD jobs. The username showing up for each submitted benchmark workflow on Maglev and corresponding sweep on W&B depends on these injected credentials and not e.g. on the user manually triggering the CI/CD job. See the [the GitLab config](../../../.gitlab-ci.yml) for more info.

## Benchmark results and summarization

Results of the sweep can be accessed at https://wandb.ai/nvidia-toronto/nre-benchmark/sweeps by members of the `nvidia-toronto` group on W&B, and are also summarized in the [NRE Quality Benchmark Dashboard](https://wandb.ai/nvidia-toronto/nre-benchmark/reports/NRE-Quality-Benchmark-Dashboard--Vmlldzo3MzAyNzU2).

The dashboard is implemented as a live W&B Report, using [Custom charts](https://docs.wandb.ai/guides/app/features/custom-charts#how-to-edit-vega) defined in [Vega-Lite](https://vega.github.io/vega-lite/).
The charts visualize the performance (in terms of quality metrics) of neural reconstruction in function of version (commit date/time) of the software. The visualization is broken down per metric and per Docker image.
[W&B run tags](https://docs.wandb.ai/guides/app/features/tags) (e.g. `benchmark`, `stage`, `dev`) assigned via the sweep launch script are used to filter down results required for specific charts. Experiments of sweeps using the `nre_run_stage` Docker image must be tagged `stage` and those of `nre_run_dev` image must be tagged `dev`, so that results end up in corresponding chart(s) in the dashboard. Forgotten tags can also be added manually to all runs of a sweep in the W&B web UI ([https://docs.wandb.ai/guides/app/features/tags](how)).

**Development of the dashboard**. The summary charts in the Dashboard can be changed. Edit W&B Report > Edit panel > hit Edit in the top left will show the chart's JSON specification for editing. The JSONs used in the dashboard are also backed up in the code repo under [wandb_vega_lite_charts](wandb_vega_lite_charts). They have a stand-alone `_dev` version that replaces the W&B data source with dummy data, and can be directly copy-pasted into [Vega Editor](https://vega.github.io/editor/#/custom/vega-lite) for further development.

> :warning: Whenever you change the JSON specs of the charts in the W&B report, please make sure to update these checked-in copies as well.

## Updating the datasets

When a new dataset version becomes available, the clips need to be uploaded to the SwiftStack location fed to the benchmark.
Old versions of the same clips are retained on SwiftStack, so that it is possible to rerun the benchmark results of an older commit. Old commits certainly digest the clips in the format available at the time of commit, but not necessarily future dataset versions.

You can use the Maglev workflow in [benchmark_dataset_upload_wf.yaml](benchmark_dataset_upload_wf.yaml) to upload a new version of the benchmark clips from their original location. This is meant as a one-off workflow per upload. Please make sure to edit the inputs of each task in the YAML, and to specify a new output directory as target on SwiftStack.
By convention, the output directory is the version of the dataset, defined as the date the input was generated at the source in YYYY-MM-DD format.
After editing the YAML, the upload can be triggered as follows.

```
maglev workflows run -f benchmark_dataset_upload_wf.yaml -n nre-benchmark-dataset-upload-<username>
```

The workflow name contains `-<username>` because users can not submit into workflows created by other users on Maglev.
This workflow needs to be authenticated against the SwiftStack namespace, as well (see the section about authentication above).

Once the upload finished, the dataset link in the benchmark workflow needs to be updated to point to the new location (`inputs` field in [benchmark_wf.yaml](benchmark_wf.yaml)). It makes sense to launch a benchmark from a dev branch with this change, and validate on W&B that clips are processed successfully in the sweep. Once the test benchmark succeeded on the new data, submit both changes as an MR.
