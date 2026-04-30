# NDAS Workflows

## Objective

run_ndas_workflow.py is a script that generates (and launches) a maglev WF consisting of Car2Sim, AMO CLE, and NRE Eval maglev WF components, using a unified NuRec docker tag. It achieves this by reusing maglev yaml specs from already existing, finished WFs of each type above, and doing some merging/editing of the yaml specs.

Because it is reusing workflow components, any tools that crawl and analyze these existing workflows should be compatible with workflows produced by this script.

For more details see [this doc.](https://docs.google.com/document/d/1VZ_99rSRXf1d4MEMIKiPyTHH6_AxY_o0eq6BFv5SLz0/edit?tab=t.nuau63w8eqoi#heading=h.i539u8ff1q4)

## Authentication

The following ngc orgs must be joined:

```
nvidian/ct-toronto-ai
nv-maglev/dlav
```

The following maglev teams must be joined:

```
dlav
```

The following maglev secrets are used for various features. The script will remove secrets that you do not own but this may result in certain features/tasks failing.

```
kratos-creds-generate
ngc-creds
s3-creds
kratos-creds-gt
avpc-eval-kratos-api-key
nre-scenes-swiftstack
llm-gateway-client-id
llm-gateway-secret
```

## Running a workflow

Basic:

```
bazel run //internal/workflows/cluster_toolbox/ndas_workflows:run_ndas_workflow -- \
  --docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_run:<version_string> \
  --tools-docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_tools:<version_string>
```

Overwrite source wf:

```
bazel run //internal/workflows/cluster_toolbox/ndas_workflows:run_ndas_workflow -- \
  --docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_run:<version_string> \
  --tools-docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_tools:<version_string> \
  car2sim.source_wf=kzampogianni-nre-test-7clips-rno-daily/2025.05.08-0232-3jph00lnrdtgz \
  amo_cle.source_wf=diff-workflow-avpc-eval-pacsim-1967-amo-model/2025.05.12-2359-31slu3uv90hy8 \
  nre_eval.source_wf=ddworakowski-e2e-eval-release-2505/2025.05.13-1812-1rk7oyue47fze
```

### Lidarfree Sauron

Run the lidarfree workflow using Sauron for 3D object detection/tracking instead of camera-DCP:

```
bazel run //internal/workflows/cluster_toolbox/ndas_workflows:run_ndas_workflow -- \
  --docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_run:<version_string> \
  --tools-docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_tools:<version_string> \
  --ndas-workflow-config=lidarfree_sauron_15.yaml
```

To override the Sauron docker image (replaces `sauron-inference` and `sauron-detection-integration` images in the workflow):

```
bazel run //internal/workflows/cluster_toolbox/ndas_workflows:run_ndas_workflow -- \
  --docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_run:<version_string> \
  --tools-docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_tools:<version_string> \
  --ndas-workflow-config=lidarfree_sauron_15.yaml \
  --sauron-docker-image=nvcr.io/nvidian/sauron-inference:<sauron_version>
```

From NRE CI/CD, there are two modes for triggering the sauron workflow:

1. **Standalone (pinned NRE version)**: Set `AUTO_RUN_STANDALONE_MAGLEV_WF=true`. This runs the `ndaswf_standalone` job which uses pre-built NRE images from a pinned `STANDALONE_NRE_DEFAULT_TAG`. Does not rebuild NRE from source — faster. To pass a custom Sauron image, set `NDAS_WORKFLOW_ARGS` to `--sauron-docker-image=nvcr.io/nvidian/sauron-inference:<sauron_version>`.

2. **Full rebuild from NRE top-of-tree**: Set `AUTO_RUN_MAGLEV_NDAS_WORKFLOW=true` and `NDAS_WORKFLOW_CONFIG_ARG=--ndas-workflow-config=lidarfree_sauron_15.yaml`. This triggers the `images` job to build `nre_run`/`nre_tools`/`nre_nrm_run` from the target branch, then runs `maglev_launch_ndas_workflow` with those fresh images. `NDAS_WORKFLOW_ARGS` (e.g., `--sauron-docker-image`) is passed through to the benchmark job.

#### Launching from the Sauron repo

The Sauron repo triggers the NRE workflow via a downstream pipeline (`run-nurec-wf-image-upload` → `run-nurec-wf-trigger`). The `run-nurec-wf-image-upload` job uploads the sauron inference image to `nvcr.io/nvidian/dvl/sauron-inference-dev:<commit_sha>`, resolves trigger variables, then `run-nurec-wf-trigger` triggers a downstream NRE pipeline on `nrs/nre`.

Both jobs are manual on main, tags, and MRs.

The `NUREC_WF_NRE_BRANCH_OR_DOCKER_TAG` pipeline variable controls which NRE version is used:

1. **Empty / unset (default)**: Standalone mode. Uses a pinned NRE tag (`PINNED_NRE_TAG` in the sauron CI config) via `AUTO_RUN_STANDALONE_MAGLEV_WF`. Fastest — no NRE rebuild.

2. **NRE docker tag** (e.g., `26.3.78-66d2beab` or `26.3.78-66d2beab-dev`): Standalone mode with a custom NRE image version. Matches pattern `<major>.<minor>.<patch>-<hex>[-dev]`.

3. **Anything else** (treated as an NRE branch name): Full rebuild mode. Triggers the NRE pipeline on that branch with `AUTO_RUN_MAGLEV_NDAS_WORKFLOW=true`, which builds `nre_run`/`nre_tools`/`nre_nrm_run` from source. Slower.

The workflow config defaults to `lidarfree_sauron_15.yaml` and can be overridden via `NDAS_WORKFLOW_CONFIG_ARG`.

### Overriding the Sauron model

To use a custom Sauron model, download it from NGC and upload it as a Maglev volume:

```bash
# Download model from NGC
ngc registry model download-version nvidian/dvl/sauron -d /tmp/sauron_model

# Upload as a Maglev volume
maglev volumes create -n sauron_model -p /tmp/sauron_model
```

Example output:

```
Creating volume: name='sauron_model', version='84a691fa-8810-406c-b463-361f3bba7d82'
Start traversing '/tmp/sauron_model'
Finish traversing "/tmp/sauron_model"
Number of files collected: 2, total size: 49 KiB
Start uploading files to swiftstack
Number of files waiting to be processed: 1
Number of files waiting to be processed: 0
Finish uploading files to swiftstack
Start constructing and uploading vdisc
Finish constructing and uploading vdisc
Successfully created new volume. -n sauron_model -v 84a691fa-8810-406c-b463-361f3bba7d82
```

Then pass the model override as a compact hydra arg:

```
bazel run //internal/workflows/cluster_toolbox/ndas_workflows:run_ndas_workflow -- \
  --docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_run:<version_string> \
  --tools-docker-image=nvcr.io/nvidian/ct-toronto-ai/nre_tools:<version_string> \
  --ndas-workflow-config=lidarfree_sauron_15.yaml \
  car2sim.sauron_model=volume:sauron_model:84a691fa-8810-406c-b463-361f3bba7d82:my_checkpoint.ckpt
```

The compact format is `volume:<name>:<version>:<checkpoint_path>`. The individual fields can also be set separately:

```
  car2sim.sauron_model.volume_name=sauron_model \
  car2sim.sauron_model.volume_version=84a691fa-8810-406c-b463-361f3bba7d82 \
  car2sim.sauron_model.checkpoint_path=my_checkpoint.ckpt
```

From the Sauron repo CI/CD, set the `NUREC_WF_SAURON_MODEL` pipeline variable in the "Run Pipeline" form:

```
NUREC_WF_SAURON_MODEL=volume:sauron_model:84a691fa-8810-406c-b463-361f3bba7d82:my_checkpoint.ckpt
```

If unset, the sauron CI defaults to a pinned model defined in the sauron `.gitlab-ci.yml` (`SAURON_MODEL` variable in `run-nurec-wf-image-upload`).

## Cross-project metadata

When a workflow is triggered from an external repo (e.g. Sauron), the calling pipeline can attach its own git metadata to the resulting Maglev workflow tags by setting environment variables of the form:

```
NUREC_WF_METADATA_<SOURCE>=<json>
```

where `<SOURCE>` identifies the upstream project (e.g. `SAURON`) and `<json>` is a JSON object whose keys become tag suffixes. For example:

```
NUREC_WF_METADATA_SAURON='{"commit_sha": "abc123", "commit_ref_name": "main", "pipeline_id": "456"}'
```

This produces the following tags on the Maglev workflow:

```
sauron_gitlab_commit_sha = abc123
sauron_gitlab_commit_ref_name = main
sauron_gitlab_pipeline_id = 456
```

Fields with null or empty values are skipped. If the value is not valid JSON, a warning is logged and the variable is ignored.

This makes it possible to trace a workflow back to the exact upstream commit/pipeline that triggered it, which is useful for cross-repo debugging and result attribution.

## Launch a workflow from CI/CD (Self-serve gitlab pipeline)

See [this guide](https://docs.google.com/document/d/1VZ_99rSRXf1d4MEMIKiPyTHH6_AxY_o0eq6BFv5SLz0/edit?tab=t.v0mxg17hb24o).

Special GitLab Pipeline Env Vars:

- `AUTO_RUN_MAGLEV_NDAS_WORKFLOW=true`: Setting this will automatically queue `images` and `maglev_launch_ndas_workflow` so that users do not have to manually launch each job
- `AUTO_RUN_MAGLEV_NDAS_WORKFLOW_LIDARFREE=true`: Setting this will automatically queue `images` and `maglev_launch_ndas_workflow_lidarfree` for lidarfree testing (runs independently of `AUTO_RUN_MAGLEV_NDAS_WORKFLOW`)
- `NDAS_WORKFLOW_ARGS`: Setting this will allow users to customize the behavior or `run_ndas_workflows.py` by allowing users to inject hydra command line arguments. View guide above for more details.

### Authentication in CI/CD

The [keys used for authentication](#authentication) are injected through masked GitLab CI/CD variables into the benchmarking CI/CD jobs. The username showing up for each submitted benchmark workflow on Maglev depends on these injected credentials and not e.g. on the user manually triggering the CI/CD job. See the [the GitLab config](../../../.gitlab-ci.yml) for more info.
