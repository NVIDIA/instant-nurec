# NRE SQA Testing Guide

## JIRA Task(s)

The main JIRA epic for SQA testing is [NRE-884](https://jirasw.nvidia.com/browse/NRE-884). This epic will be deprecated as we aim to have all test-related information within this repository. While the JIRA epic provides a comprehensive list of tests, some information is outdated - particularly script locations. The correct scripts are located in the `internal/sqa/scripts` directory of this repository.

## Scripts to be Used

The scripts to test the datasets and configs are located in the `internal/sqa/scripts` directory. To understand each script's purpose, please run the help command for each:

```bash
# View help for nre_image_trainval.sh
./internal/sqa/scripts/nre_image_trainval.sh --help

# View help for nre_image_trainval_novel_view.sh
./internal/sqa/scripts/nre_image_trainval_novel_view.sh --help

# View help for nre_tools.sh
./internal/sqa/scripts/nre_tools.sh --help

# View help for grpc_api_test.sh
./internal/sqa/scripts/grpc_api_test.sh --help
```

**Important**: Please take the time to understand each script's purpose and parameters before running tests, as incorrect usage may lead to false test failures.

## Data Sets Location

The test datasets are available at the following locations:

- Hyperion-8.1 data: `pdx-team-ncore:/sqa/dataset/H81/*`
- Waymo data: `pdx-team-ncore:/sqa/dataset/Waymo/11017034898130016754_697_830_717_830/*`
- DriveSim data: `pdx-team-ncore:/sqa/dataset/DriveSim/*`

### Downloading Datasets with rclone

#### Set up rclone

```bash
# Install rclone
sudo apt-get update && sudo apt-get install -y curl
curl https://rclone.org/install.sh | sudo bash

# Set your Swift keys here
# To get these keys:
# - For pbss-team-ncore-waymo: See below
# - For pdx-team-ncore: See below
PBSS_TEAM_NCORE_WAYMO_KEY="your_pbss_team_ncore_waymo_key_here"
PDX_TEAM_NCORE_KEY="your_pdx_team_ncore_key_here"

# Create rclone config directory if it doesn't exist
mkdir -p ~/.config/rclone

# Create rclone.conf file with the Swift endpoints
cat > ~/.config/rclone/rclone.conf << EOF
[pbss-team-ncore-waymo]
type = swift
env_auth = false
user = team-ncore-waymo
key = ${PBSS_TEAM_NCORE_WAYMO_KEY}
auth = https://pbss.s8k.io/auth/v1.0
auth_version = 1

[pdx-team-ncore]
type = swift
env_auth = false
user = team-ncore
key = ${PDX_TEAM_NCORE_KEY}
auth = https://pdx.s8k.io/auth/v1.0
auth_version = 1
EOF

# Set appropriate permissions for the config file
chmod 600 ~/.config/rclone/rclone.conf
```

For the necessary keys, please do the following:
In [CSS](<https://cssportal.sre.nsv.nvidia.com:4443/user/(userRouter:home)>) (needs to be opened in Google Chrome), go to My Storage → Join Namespace and request access for

- Cluster: `PBSS` | Storage Namespace Name: `team-ncore-waymo`
- Cluster: `PDX` | Storage Namespace Name: `team-ncore`

Once granted: [CSS](<https://cssportal.sre.nsv.nvidia.com:4443/user/(userRouter:home)>) → My Storage → {team-ncore|team-ncore-waymo} → Auth Info → Copy "Storage Space User Password (For Swift API)" and paste them to the above keys.

#### Download data sets

To download these datasets, you can use rclone. First, ensure you have rclone configured with the appropriate remote endpoints.

```bash
# Create a local directory for the datasets
mkdir -p ~/nre_test_data/{hyperion81,waymo,drivesim}

# Download Hyperion-8.1 data
rclone copy pdx-team-ncore:/sqa/dataset/H81/ ~/nre_test_data/hyperion81/ --progress

# Download Waymo data
rclone copy pdx-team-ncore:/sqa/dataset/Waymo/11017034898130016754_697_830_717_830/ ~/nre_test_data/waymo/ --progress

# Download DriveSim data
rclone copy pdx-team-ncore:/sqa/dataset/DriveSim/ ~/nre_test_data/drivesim/ --progress
```

## Configs to Test

The configs that are being used for testing can be found in the `configs/apps/prod/{DriveSim,Hyperion-8.1,Waymo}/*.yaml` directories. To list all the configuration files:

```bash
# List all production configuration files
find configs/apps/prod/* -maxdepth 1 -path "configs/apps/prod/options" -prune -o -name "*.yaml" -print | sort
```

## Artifacts

Some tests rely on artifacts generated through `nre_image_trainval.sh` with a previous release of NuRec.

The process to capture these artifacts is:

- Run training with the parameters documented for "Render Tests" on the chosen base release of NuRec
  - Ex. Running one of the `nre_image_render*` test cases with the automation infra described later will do this, though
    the infra is only guaranteed to be compatible with the current branch.
- Package or upload contents of the randomly named folder created by the training and validation process
  - For full tests, the whole output folder can be uploaded to `pdx-team-ncore`
  - For lite tests, we restrict the data to keep the size reasonable and store the package in the GitLab registry

Sample commands to create an output package for lite tests and upload to GitLab's package registry:

```bash
bazel run //internal/sqa:run_tests -- --tag <container_tag> --test-identifiers nre_image_render--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val
TRAIN_VAL_OUTPUT=bazel-bin/internal/sqa/run_tests.runfiles/_main/results/nre_image_render--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val/<random_folder>
# Package name and version in the registry
# Remember to bump the version if uploading a new revision of a package
PACKAGE_NAME=test_data_ncore_sqa_default_<nre_major.minor>_artifacts
PACKAGE_VERSION=0.1
tar -I pigz -cf $PACKAGE_NAME.tgz -C $TRAIN_VAL_OUTPUT artifacts/last.usdz val/input_rgb val/pred_rgb
# Rely on GitLab Personal Access Token stored in ~/.netrc
curl --netrc --upload-file $PACKAGE_NAME.tgz \
    "https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/$PACKAGE_NAME/$PACKAGE_VERSION/$PACKAGE_NAME.tgz"
```

## Expected Results

The expected results are listed on the JIRA epic's subtasks as well as the (to be deprecated) confluence page: [NRE - Release 25.04 Test Report](https://confluence.nvidia.com/display/NUREC/NRE+-+Release+25.04+Test+Report). This information will also be embedded within the NRE repo in the future.

## Testing Workflow

```
┌─────────────┐
│  Run Test   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     Yes     ┌───────────────────────────────────────────────────┐
│   Crash?    ├────────────►│ Is crash outside NRE image or due to incorrect    │
└──────┬──────┘             │ usage (config not found, dataset not found, etc)? │
       │ No                 └───────────┬───────────────────────┬───────────────┘
       │                                │ Yes                   │ No
       │                                ▼                       ▼
       │                    ┌───────────────────────────┐     ┌───────────────────────────┐
       │                    │ SQA owns the test scripts │     │ SQA files NVBug in format │
       │                    │ and is expected to fix    │     │ [NuRec][SQA][YY.MM]       │
       │                    │ that, but notify the dev  │     │ <script-name> fails with  │
       │                    │ team about necessary fix  │     │ error <error-msg>         │
       │                    └───────────────────────────┘     └───────────────┬───────────┘
       ▼                                                                      │
┌─────────────┐                                                               ▼
│   Success   │                                                 ┌───────────────────────────┐
└──────┬──────┘                                                 │ Bug triaged and assigned  │
                                                                │ by dev team and fixed in  │
       │                                                        │ in next RC if necessary   │
       ▼                                                        └───────────────────────────┘
┌─────────────┐
│ Report KPIs │
│ (PSNR,      │
│ runtime,    │
│ etc.)       │
└─────────────┘
```

When a test crashes due to an issue within the NRE image, an NVBug should be filed in the format: `[NuRec][SQA][YY.MM] <script-name> fails with error <error-msg>`. The bug description should contain all necessary information to reproduce the issue. Template: [5621830](https://nvbugspro.nvidia.com/bug/5621830)

This bug will be handled by the development team and, once fixed, should be cherry-picked for the next Release Candidate (RC). Ideally, the next RC should only contain fixes to bugs found during the release cycle, not new features.

This process requires close collaboration between the development team and SQA to reproduce issues, implement fixes, and verify that the fixes resolve the problems.

## Test plan execution frontend

A layer has been built on top of the individual test scripts to document SQA's test plan as code and automate its execution. This is still WIP and does not exactly match SQA usage yet.

### Test case generation

The system generates test cases based on the available datasets and configurations. Test cases are defined in:

- `internal/sqa/test_cases/test_plan.yml`: Test plan definition, listing the individual test cases
- `internal/sqa/test_cases/test_cases.py`: Test case parsing logic from above YAML
- `internal/sqa/test_cases/datasets.py`: Dataset configurations
- `internal/sqa/test_cases/commands.py`: Command generation logic

### Test plan fields

Each entry in `test_cases/test_plan.yml` supports the following fields:

- **`name`** (required): Structured identifier encoding all test parameters, e.g.
  `nre_image_trainval--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default`

- **`owner`** (optional): NV login or DL name of the test owner, for example `achauveau` or `nre-team`.

- **`description`** (optional): Human-readable description of the test case. Supports:

  - Single-line string: `description: "Short description"`
  - Multi-line YAML block scalar:

    ```yaml
    description: |
      First paragraph.

      Second paragraph with **markdown** and a [link](#anchor).
    ```

  - Inline markdown including links to other sections in this README (e.g.
    `[see validation details](#psnr-thresholds)`) or other repo files. Relative paths must be
    relative to `internal/sqa/` (the README location). For example, to link to
    `internal/sqa/test_cases/foo.md`, write `[foo](test_cases/foo.md)`.

- **`manual_validation`** (optional): Anchor link to a section in this README.md describing the steps required to manually validate the test results.

- **`parallel_execution`** (optional, default `true`): Set to `false` to disable parallel
  execution for this test case (e.g. when runtime checks require exclusive access).

- **`ci_runtime_limit_<step>`** (optional): Maximum allowed wall-clock time for the named step in CI,
  e.g. `ci_runtime_limit_train: "4m20s"`. Supported formats: `180s`, `3m`, `1h`, `2m30s`.

- **`eval_psnr_threshold_<camera_id>`** (optional): Minimum acceptable PSNR value for the given
  camera, e.g. `eval_psnr_threshold_cam_00: 22.0`.

### Running through the plan

You can run the test plan using `internal/sqa/scripts/run_tests.py`:

```bash
# Run all available test cases on NuRec images tagged CONTAINER_TAG
# - By default each test defines whether it uses obfuscated or non-obfuscated SW
# - Default 'nre_run' and 'nre_tools' images or their obfuscated counterparts are used unless a suffix is passed in
bazel run //internal/sqa:run_tests -- --tag <CONTAINER_TAG> [--suffix <CONTAINER_SUFFIX>]

# Run specific test cases
# The input identifiers list is comma-separated, and supports shell-style wildcards like `*`
bazel run //internal/sqa:run_tests -- --tag <CONTAINER_TAG> [--suffix <CONTAINER_SUFFIX>] --test-identifiers <TEST1>,<TEST2>

# Refer to help target for full list of optional parameters
bazel run //internal/sqa:run_tests -- --help
```

## Lite tests for CI integration and easy debug

Tests with `--mode-lite` have been introduced into the test plan. They follow the structure of typical SQA tests and use
the same underlying scripts, but provide lighter datasets and execution parameters to keep runtime low. These tests
are primarily meant to offer CI protection on end-to-end NuRec use cases. They can also be used for easier reproduction
and debug of some SQA-reported issues.

### Design restrictions

For ease of use and efficient CI integration, we mandate that all datasets used by "lite" tests be installed at build
time through Bazel dependencies and not downloaded at runtime. The tests enforce this requirement during execution.

### Bazel test integration

"lite" tests are exposed as part of the SQA test plan and can be executed through `bazel run //internal/sqa:run_tests` as
described above.

In addition these tests are exposed as Bazel test targets, which offer sandboxing guarantees and native Bazel results
reporting. Since the tests are heavier than unit tests, they are tagged `manual` and won't be run by default. By default
the tests can execute in parallel with each other, but this behavior can be turned off on a case-by-case basis by
adding a `parallel_execution: false` attribute to test entries in `test_plan.yml`, which backends to Bazel's `exclusive`
tag on test targets.

Two types of test targets are generated for each test case:

Runfiles-based tests (`sqa_test--<test_identifier>`):

- Execute natively using Bazel runfiles
- Benefit from Bazel caching for faster execution
- Suitable for pre-merge CI checks and local development
- Depend on the appropriate executable target (ex. `//:run` or `//internal/scripts/pycena/runtime:pycena_run`)
- Can not currently download models from NGC at runtime

Docker-based tests (`sqa_docker_test--<test_identifier>`):

- Execute using Docker containers
- Tagged "external" to avoid Bazel caching due to Docker dependency
- Suitable for periodic CI checks and validation
- Require users to provide additional parameters for container and model download

#### Keeping Bazel targets up to date

The design for integration to Bazel avoids duplicating information with the Python-based test plan. Test names, Bazel
dataset dependencies and executable dependencies are automatically extracted from the test plan, and converted into
Bazel test targets.

- To update the list of tests known to Bazel, run: `bazel run //internal/sqa/test_cases:sync_test_plan`
- To verify whether tests are in sync between Bazel and Python test plan, run: `bazel test
//internal/sqa/test_cases:sync_test_plan_test`

#### Running tests

##### Runfiles-based tests (recommended for development and pre-merge CI)

Users can launch all runfiles-based tests with:

```
bazel test //internal/sqa:sqa_test_suite
```

Or run individual runfiles-based tests:

```
bazel test //internal/sqa:sqa_test--<test_identifier>
```

##### Docker-based tests (for post-merge CI and validation)

Users can launch all Docker-based tests with:

```
bazel test //internal/sqa:sqa_docker_test_suite --test_arg=--tag=<docker_tag> [--test_arg=--suffix=<docker_suffix>] --test_env=NGC_API_KEY=<your_ngc_key>
```

Or run individual Docker-based tests:

```
bazel test //internal/sqa:sqa_docker_test--<test_identifier> --test_arg=--tag=<docker_tag> [--test_arg=--suffix=<docker_suffix>] --test_env=NGC_API_KEY=<your_ngc_key>
```

Docker-specific inputs:

- `<docker_tag>`: Tests use Docker containers generated by CI job `images*`. Pass a tag defining the version to use.
  - Precise version tags of the form `25.10.153-8541a14e` are recommended.
- `<docker_suffix>`: Optional suffix for the Docker images, ex. `grpc`. By default, no suffix is used.
- `<your_ngc_key>`: Personal NGC API key used to download models.

##### Finding test identifiers

- Find test names by running: `bazel run //internal/sqa:run_tests`
- Query runfiles-based targets: `bazel query "tests(//internal/sqa:sqa_test_suite)"`
- Query Docker-based targets: `bazel query "tests(//internal/sqa:sqa_docker_test_suite)"`

##### Viewing outputs

Because output files can be key to understanding the results of SQA lite tests, we are storing them in a folder which is
retained after test sandbox teardown.

The results folder seen during test execution is of the form:

```
Results path: /<sandbox_path>/execroot/_main/bazel-out/k8-opt/testlogs/internal/sqa/<test_name>/test.outputs
```

After the test, it can be found on the filesystem at the following relative paths from the repo's root folder:

```
# Counterpart to the sandbox path
bazel-out/k8-opt/testlogs/internal/sqa/<test_name>/test.outputs

# Shorter path through convenience symlink
bazel-testlogs/internal/sqa/<test_name>/test.outputs
```

Since Bazel test outputs are cached, potentially remotely, outputs larger than 50 MB in size are replaced by marker
files. This keeps network traffic and remote storage usage small in CI.

In local execution it's possible to run the tests through `bazel run //internal/sqa:<test_name>` instead of
`bazel test //internal/sqa:<test_name>` if all outputs need to be observed. In this case the logs are printed directly
to the console, while non-truncated output files go to the same path as above.

For `sqa_lite*` CI jobs, outputs are exposed as job artifacts. They can be browsed or downloaded from a job's details
page and are also accessible as a direct link from the Merge Request UI.

#### Performance gates

In CI, we validate runtime for some of the lite tests. The performance gates are defined through `ci_runtime_limit_<step>`
entries in test plan file `test_plan.yml`, and use format `<n>h:<n>m:<n>s`. The steps correspond to separate measurements
output into file `timings.txt` by the SQA scripts during test execution. Validation is opt-in, enabled by explicitly
passing parameter `--test_arg=--validate_ci_runtime_limits` on the `bazel test` command line.

When tuning these performance gates, it is important to pay attention to the following:

- For stable results, tests with performance gates assigned are not allowed to run in parallel with any other test
  cases. This is validated by test plan integrity checks.
- Avoid assigning performance gates to any test which downloads and caches a large amount of information during
  execution (ex. models), since it can make runtime vary greatly from run to run.
- We only need to tune against values observed on CI machines for runfiles-based execution, ie. `sqa_lite` job.
  Validation against performance gates is requested solely for this job.
- Some margin needs to be left in the performance gates to allow for a reasonable amount of performance variation
  between machines in the CI pool and from run to run. Gather multiple reference data points when making updates, see
  previous section to access logs and `timings.txt` files of passing CI tests.

#### Quality evaluation

To evaluate the quality of training and rendering, we generate various evaluation images and compute associated metrics.

Evaluation of these steps in the pipeline is performed independently, as follows:

- Training is evaluated by comparing validation predictions `val/pred_rgb` to the ground truth `val/input_rgb`.
  - Training is expected to converge towards the ground truth, but the shorter we train (especially in lite tests), the
    less closely we approach it.
- Rendering of training scenes is evaluated by comparing rendered images under `render` to validation predictions
  `val/pred_rgb`.
  - Here we always expect nearly identical pixels between the renderer's output and the reference image. This close
    match is expected independently of training quality.

The main files and folders produced by the evaluation logic are:

- `eval/<camera_id>`: Visualization of the evaluation
  - `.jpg` showing side-by-side comparison, diff and 5x diff between evaluation and reference images
  - `.gif` showing an animated loop between evaluation and reference images
- `eval/ego-hoods`: Contains the ego hood mask used to restrict the evaluation to meaningful output pixels
- `eval/reference`: Copy of the reference images, laid out as needed by CLI command `eval-rendering-metrics`
- `eval/metrics.yaml`: Contains PSNR and SSIM metrics, both frame-per-frame and aggregates for the scene

##### Quality gates (PSNR thresholds)

For tests that include quality evaluation, we may validate that the PSNR metrics meet minimum thresholds. This helps
catch regressions in training or rendering quality.

Quality gates are defined through `eval_psnr_threshold_<camera_id>` entries in the test plan file `test_plan.yml`.
The camera ID corresponds to the camera names in the `eval/metrics.yaml` output file, under `aggregate_metrics.psnr`.

Unlike performance gates, PSNR validation is always enabled when thresholds are defined - no additional command-line
flag is required.

When tuning PSNR thresholds:

- Set thresholds with sufficient margin below typical values to avoid flaky failures from normal variation.
- The threshold represents a minimum acceptable PSNR value; actual values should typically be higher.
- Different test types may use different camera naming conventions (e.g., `cam_00` vs `camera_front_wide_120fov`).

## Manual tests and manual validation

There are tests that are not yet automated either due to the difficulty or lack of time. There are also tests that are automated but still need manual validation to verify the correctness of the results. This section will record the steps for both of these cases.

### Manual tests

**TODO**: Add steps for manual tests.

### Automated tests with manual validation

The outputs of full tests can be found at: `<NRE_repo_root>/bazel-bin/internal/sqa/run_tests.runfiles/_main/results/<test_name>`.

For example, the result of `asset_harvest--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar` test can be found at: `<NRE_repo_root>/bazel-bin/internal/sqa/run_tests.runfiles/_main/results/asset_harvest--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar/`

#### Rendering of training views test

Applies to both the `nre_image_render` (render CLI) and `nre_render_grpc` (gRPC client) test cases, which share the same validation steps.

**TODO**: Once a PSNR gate is added for full tests, this manual validation step can be removed.

1. Navigate to the output folder (see path above).
2. Under eval/ folder, there could be one or multiple camera id folders. For example, `camera_front_wide_120fov`. Look into all of the camera id folders and check all the GIF files generated. The generated anim GIFs should show no large areas (background or vehicles) being visibly moved or stretched.

An example GIF that shows a mismatch that should be reported:

![Example GIF](attachments/000001_render_grpc_vs_val.gif)

#### Control actor test

1. Navigate to the output folder (see path above).
2. Under eval/ folder, there could be one or multiple camera id folders. For example, `camera_front_wide_120fov`. Look into all of the camera id folders and check all the generated GIFs to verify that:
   - Static actors (non-moving or very slow-moving objects) stay in place.
   - A majority of clearly moving vehicles on the road are rotated along the vertical axis (not lifted). It is acceptable for parked cars or slow-moving cars not to show rotation.

An example GIF that shows correct behaviour:

![Example GIF](attachments/000001_render_grpc_vs_val_actor_control.gif)

Sometimes objects are not marked dynamic in the input (e.g. they are not moving or just moving too slowly). When objects are rotated, they often have some artifacts due to being viewed from a direction that the object had no observations from. These cases should be considered normal for the sake of this test.

#### Asset edit test

This validation applies to the `nre_render_grpc` asset-edit scenario test case.

1. Navigate to the output folder (see path above).
2. Open `internal/sqa/edit-assets/scenarios/sqa_scene_6b0e750d.json` and inspect the checked-in `metadata` section. This is the full scenario template that documents the dataset and high-level description.
3. Open `edit_assets/edit_assets.json` and confirm the runtime resolver replaced the `asset://...` references in `replace[*].replacement_id` and `insert.asset_ids` with absolute paths to the sample PLYs under `internal/sqa/edit-assets/samples/`.
4. Under `render/`, open the first frame (000000.png) of the rendered camera and verify that:
   - The scene vehicle replacement is visible as a grey van.
   - Two inserted pedestrians are visible on the right side of the scene.
   - Two inserted sedan vehicles are visible on the left side of the scene.

As with the control-actor test, minor view-dependent artifacts on edited actors are acceptable. The goal of this test is to verify API correctness and obvious visual plausibility, not render quality against a PSNR baseline.

#### Asset harvester test

1. Go to the output folder (see path above).
2. Check that there are subfolders, each corresponding to a track id.
3. Under each subfolder, verify that you see both `<track_id>.gif` and `<track_id>.ply` files.

For each dataset, the track ids are different. Generally, you should expect to see five subfolders with integer names.

Example output:

![Example](attachments/asset_harvester_output.png)

## Test Plan Details

This section documents the input resources and command lines of test cases that form the SQA test plan.
Command lines are documented for test execution on Docker images, and placeholders are used for paths and values that vary between executions:

- `<DOCKER_TAG>`: Docker image tag (e.g., `25.10.153-8541a14e`)
- `<DATASET_DIR>`: Local directory containing downloaded datasets
- `<ARTIFACTS_DIR>`: Local directory containing archived artifacts
- `<OUTPUT_DIR>`: Base output directory for test results
- `<TEST_NAME>`: Name of the test case (used as subdirectory under `OUTPUT_DIR`)
- `<TRAIN_OUTPUT_SUBDIR>`: Random subdirectory created by training (e.g., `<OUTPUT_DIR>/<TEST_NAME>/DV2tpKaZgbZnJcA2VwFUNN`)
- `<GIF_TOOL_PATH>`: Path to the optional GIF generation tool

<!-- BEGIN AUTO-GENERATED TEST PLAN DOCUMENTATION -->
<!-- DO NOT EDIT THIS SECTION MANUALLY -->
<!-- Run: bazel run //internal/sqa/test_cases:sync_test_plan -->

### Resources

#### Datasets

| Name                                                     | Source                                                                                    | Local Path                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `H81_0d59b8c8_4cam_lidarfree`                            | `pdx-team-ncore:/sqa/dataset/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`  | `H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`  |
| `H81_6b0e750d_4cam_lidarfree`                            | `pdx-team-ncore:/sqa/dataset/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`  | `H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`  |
| `H81_AT128_0d59b8c8_1cam_1lidar`                         | `pdx-team-ncore:/sqa/dataset/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`       | `H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`       |
| `H81_AT128_6b0e750d_1cam_1lidar`                         | `pdx-team-ncore:/sqa/dataset/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`       | `H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`       |
| `H81_Panda128_0d59b8c8_1cam_1lidar`                      | `pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`    | `H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`    |
| `H81_Panda128_0d59b8c8_4cam_1lidar`                      | `pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`    | `H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840`    |
| `H81_Panda128_0d59b8c8_4cam_1lidar_v4`                   | `pdx-team-ncore:/sqa/dataset/v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840` | `v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840` |
| `H81_Panda128_6b0e750d_1cam_1lidar`                      | `pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`    | `H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`    |
| `H81_Panda128_6b0e750d_4cam_1lidar`                      | `pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`    | `H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8`    |
| `H81_Panda128_6b0e750d_4cam_1lidar_v4`                   | `pdx-team-ncore:/sqa/dataset/v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8` | `v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8` |
| `Waymo_11017034898130016754_697_830_717_830_3cam_1lidar` | `pdx-team-ncore:/sqa/dataset/Waymo/11017034898130016754_697_830_717_830`                  | `Waymo/11017034898130016754_697_830_717_830`                  |
| `test_data_ncore`                                        | `@test_data_ncore//:ncore_clipgt_1sec`                                                    | `test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711` |

#### Artifacts

| Name                                                            | Source                                                                                                          | Local Path                                                      |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts` | `pdx-team-ncore:scratch-adrajeev/25_07/L20/RC5/sqa_default_6b/fP3DV4HAYAcQ9yP65Rcnsb`                           | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts` |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts` | `pdx-team-ncore:scratch-adrajeev/25_08/L40S/RC5/train_val_sqa_default_6b/9b68oNgdZeGdeAHRMWXqwP`                | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts` |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts` | `pdx-team-ncore:scratch-adrajeev/25_09/A100/RC10/TrainVal_GRPC/train_val_sqa_default_6b/86nfd6RbvJMFNiQr26tEX4` | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts` |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts` | `pdx-team-ncore:scratch-adrajeev/25_11/L40S/RC5/TrainVal_GRPC/train_val_sqa_default_6b/7Cfeq2NzXnDokBHAVYwjFi`  | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts` |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts` | `pdx-team-ncore:scratch-adrajeev/25_12/A40/RC2/TrainVal_GRPC/train_val_sqa_default_6b/75gJzJ6YHNHP9PA3mBqEnr`   | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts` |
| `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts` | `pdx-team-ncore:scratch-adrajeev/26_01/L40S/RC7/TrainVal_GRPC/train_val_sqa_default_6b/RZY4L2b4SK9xnjSYENX4Z9`  | `H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts` |
| `test_data_ncore_sqa_default_25.06_artifacts`                   | `@test_data_ncore_sqa_default_25.06_artifacts//:all`                                                            | `test_data_ncore_sqa_default_25.06_artifacts`                   |

### SQA Lite

#### nre_image_trainval

##### Test identifier `nre_image_trainval--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default`

Lite training and validation test.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=1500 \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --val-append system.test.save_videos=false \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-lite--obfuscation-yes--dataset-test_data_ncore--config-hyperion8.1_sqa_default`

Lite training and validation test on obfuscated SW.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=1500 \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --val-append system.test.save_videos=false \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --tag <DOCKER_TAG>
  ```

#### nre_render_grpc

##### Test identifier `nre_render_grpc--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val`

Lite gRPC rendering test after minimal training.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=50 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --camera-frame-idx 10 \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8042 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8042 \
      --frame-height 135 \
      --frame-step 3 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val--test_control_actor-yes`

Lite gRPC rendering test after minimal training with rotation of dynamic actors.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=50 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --camera-frame-idx 10 \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8043 \
      --no-obfuscated \
      --enable-editing-actors \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8043 \
      --frame-height 135 \
      --frame-step 3 \
      --no-obfuscated \
      --test-actor-control \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val--use_gsplat-yes`

Lite gRPC rendering test after minimal training with rendering handled through Gsplat.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=50 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --camera-frame-idx 10 \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8044 \
      --no-obfuscated \
      --use-gsplat \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8044 \
      --frame-height 135 \
      --frame-step 3 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val--edit_assets_scenario-sqa_scene_lite`

**Manual validation required:** [Asset edit test](README.md#asset-edit-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=50 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --camera-frame-idx 10 \
    --no-obfuscated
  ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/resolve_edit_assets_scenario.py \
    --scenario-id sqa_scene_lite \
    --dataset-name test_data_ncore \
    --output-edit-file <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json
  ```

- **Step 4** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8045 \
      --no-obfuscated \
      --enable-editing-actors \
      --edit-assets <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8045 \
      --frame-height 135 \
      --frame-step 3 \
      --no-obfuscated \
      --enable-editing-actors \
      --edit-assets <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json \
      --tag <DOCKER_TAG>
    ```

- **Step 5**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-lite--obfuscation-no--dataset-test_data_ncore--artifact_source-test_data_ncore_sqa_default_25.06_artifacts`

Lite backward-compatibility test for gRPC rendering using artifacts generated by minimal training with release 25.06.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --camera-frame-idx 10 \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8047 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8047 \
      --frame-height 135 \
      --frame-step 3 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### nre_image_render

##### Test identifier `nre_image_render--mode-lite--obfuscation-no--dataset-test_data_ncore--config-hyperion8.1_sqa_default--artifact_source-train_val`

Lite direct rendering test after minimal training.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --train-append dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2 \
    --train-append dataset.n_train_sequential_image_subsample=2 \
    --train-append dataset.n_samples_per_epoch=50 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.n_val_image_subsample=2 \
    --val-append dataset.val_camera_frame_step=3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_image_render.sh \
    render-training-views \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --camera-id camera_front_wide_120fov \
    --no-obfuscated \
    --image-scale 0.5 \
    --frame-step 3 \
    --tag <DOCKER_TAG>
  ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_render--mode-lite--obfuscation-no--dataset-test_data_ncore--artifact_source-test_data_ncore_sqa_default_25.06_artifacts`

Lite backward-compatibility test for direct rendering using artifacts generated by minimal training with release 25.06.

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_render.sh \
    render-training-views \
    --artifact-path <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --camera-id camera_front_wide_120fov \
    --no-obfuscated \
    --image-scale 0.5 \
    --frame-step 3 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/test_data_ncore_sqa_default_25.06_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### nre_tools

##### Test identifier `nre_tools--mode-lite--obfuscation-no--dataset-test_data_ncore`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov \
    --filename timings.txt \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-lite--obfuscation-yes--dataset-test_data_ncore`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711/clipgt-9048443e-c482-4228-8326-5b3dff3be711.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov \
    --filename timings.txt \
    --tag <DOCKER_TAG>
  ```

#### run_example

##### Test identifier `run_example--mode-lite--obfuscation-no--script_filename-example_losses`

**Owner:** `amaximo`

Lite test validating that the example_losses.py example script runs without errors.

- **Single command**:
  ```bash
  ./internal/sqa/scripts/run_examples.sh \
    --run-script docs/architecture/examples/example_losses.py \
    --tag <DOCKER_TAG>
  ```

### Full SQA

#### nre_image_trainval

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_distill`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_distill`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_distill_and_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill_and_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_difix_distill_and_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill_and_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_default_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_default_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_distill`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_distill`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_distill_and_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill_and_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_difix_distill_and_inference`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_difix_distill_and_inference.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_AT128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_AT128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_AT128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_AT128_0d59b8c8_1cam_1lidar--config-hyperion8.1_sqa_lidar_default`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml \
    --dataset-path <DATASET_DIR>/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_4cam_1lidar--config-car2sim`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar--config-car2sim`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar--config-car2sim_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_4cam_1lidar--config-car2sim`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_4cam_1lidar--config-car2sim`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_4cam_1lidar--config-car2sim_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_6b0e750d_4cam_lidarfree--config-car2sim_lidarfree`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_6b0e750d_4cam_lidarfree--config-car2sim_lidarfree`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_6b0e750d_4cam_lidarfree--config-car2sim_lidarfree_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-yes--dataset-H81_0d59b8c8_4cam_lidarfree--config-car2sim_lidarfree`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_0d59b8c8_4cam_lidarfree--config-car2sim_lidarfree`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_image_trainval--mode-full--obfuscation-no--dataset-H81_0d59b8c8_4cam_lidarfree--config-car2sim_lidarfree_gsplat`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/car2sim_lidarfree_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/input_rgb \
    --eval-images-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### nre_render_grpc

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default--artifact_source-train_val`

**Manual validation required:** [Rendering of training views test](README.md#rendering-of-training-views-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8049 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8049 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default--artifact_source-train_val--use_gsplat-yes`

**Manual validation required:** [Rendering of training views test](README.md#rendering-of-training-views-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8050 \
      --no-obfuscated \
      --use-gsplat \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8050 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default_gsplat--artifact_source-train_val--use_gsplat-yes`

**Manual validation required:** [Rendering of training views test](README.md#rendering-of-training-views-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8051 \
      --no-obfuscated \
      --use-gsplat \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8051 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default--artifact_source-train_val--test_control_actor-yes`

**Manual validation required:** [Control Actor Test](README.md#control-actor-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 3** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8052 \
      --no-obfuscated \
      --enable-editing-actors \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8052 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --test-actor-control \
      --tag <DOCKER_TAG>
    ```

- **Step 4**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default--artifact_source-train_val--edit_assets_scenario-sqa_scene_6b0e750d`

**Manual validation required:** [Asset edit test](README.md#asset-edit-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/resolve_edit_assets_scenario.py \
    --scenario-id sqa_scene_6b0e750d \
    --dataset-name H81_Panda128_6b0e750d_1cam_1lidar \
    --output-edit-file <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json
  ```

- **Step 4** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8053 \
      --no-obfuscated \
      --enable-editing-actors \
      --edit-assets <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8053 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --enable-editing-actors \
      --edit-assets <OUTPUT_DIR>/<TEST_NAME>/edit_assets/edit_assets.json \
      --tag <DOCKER_TAG>
    ```

- **Step 5**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8055 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8055 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8056 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8056 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8057 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8057 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8058 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8058 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8059 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8059 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_render_grpc--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--artifact_source-H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts`

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/grpc_api_test.sh \
    preprocess \
    --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --camera-ids camera_front_wide_120fov \
    --lidar-id lidar_gt_top_p128 \
    --tag <DOCKER_TAG> \
    --no-obfuscated
  ```

- **Step 2** (parallel):

  - **[Background]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      run-server \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts/artifacts/last.usdz \
      --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
      --port 8060 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```
  - **[Foreground after 20s]**:
    ```bash
    ./internal/sqa/scripts/grpc_api_test.sh \
      render-grpc \
      --artifact-path <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts/artifacts/last.usdz \
      --output-dir <OUTPUT_DIR>/<TEST_NAME>/render/camera_front_wide_120fov \
      --camera-id camera_front_wide_120fov \
      --port 8060 \
      --frame-height 540 \
      --frame-step 99 \
      --no-obfuscated \
      --tag <DOCKER_TAG>
    ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <ARTIFACTS_DIR>/H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --egocar-hood-dir <OUTPUT_DIR>/<TEST_NAME>/preprocess/ego-hoods \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### nre_image_render

##### Test identifier `nre_image_render--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_1cam_1lidar--config-hyperion8.1_sqa_default--artifact_source-train_val`

**Manual validation required:** [Rendering of training views test](README.md#rendering-of-training-views-test)

- **Step 1**:

  ```bash
  ./internal/sqa/scripts/nre_image_trainval.sh \
    --config-path configs/apps/prod/Hyperion-8.1/sqa_default.yaml \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --filename timings.txt \
    --no-obfuscated \
    --world-size 0 \
    --train-append dataset.n_samples_per_epoch=5000 \
    --val-append system.test.save_inputs=true \
    --val-append system.test.save_videos=false \
    --val-append dataset.val_camera_frame_step=99 \
    --tag <DOCKER_TAG>
  ```

- **Step 2**:

  ```bash
  ./internal/sqa/scripts/nre_image_render.sh \
    render-training-views \
    --artifact-path <TRAIN_OUTPUT_SUBDIR>/artifacts/last.usdz \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --camera-id camera_front_wide_120fov \
    --no-obfuscated \
    --frame-step 99 \
    --tag <DOCKER_TAG>
  ```

- **Step 3**:

  ```bash
  ./internal/sqa/scripts/nre_eval_rendering_metrics.sh \
    eval \
    --reference-dir <TRAIN_OUTPUT_SUBDIR>/val/pred_rgb \
    --eval-images-dir <OUTPUT_DIR>/<TEST_NAME>/render \
    --output-dir <OUTPUT_DIR>/<TEST_NAME>/eval \
    --gif-tool <GIF_TOOL_PATH> \
    --shard-file-pattern <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### nre_tools

##### Test identifier `nre_tools--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_4cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov \
    --filename timings.txt \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov \
    --filename timings.txt \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_4cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov \
    --filename timings.txt \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_4cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov \
    --filename timings.txt \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-full--obfuscation-yes--dataset-Waymo_11017034898130016754_697_830_717_830_3cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/Waymo/11017034898130016754_697_830_717_830/11017034898130016754_697_830_717_830.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_right_50fov,camera_front_left_50fov,camera_front_50fov \
    --filename timings.txt \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `nre_tools--mode-full--obfuscation-no--dataset-Waymo_11017034898130016754_697_830_717_830_3cam_1lidar`

- **Single command**:
  ```bash
  ./internal/sqa/scripts/nre_tools.sh \
    --dataset-path <DATASET_DIR>/Waymo/11017034898130016754_697_830_717_830/11017034898130016754_697_830_717_830.zarr.itar \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --camera-ids camera_front_right_50fov,camera_front_left_50fov,camera_front_50fov \
    --filename timings.txt \
    --no-obfuscated \
    --tag <DOCKER_TAG>
  ```

#### asset_harvest

##### Test identifier `asset_harvest--mode-full--obfuscation-yes--dataset-H81_Panda128_6b0e750d_4cam_1lidar_v4`

**Manual validation required:** [Asset Harvester Test](README.md#asset-harvester-test)

- **Single command**:
  ```bash
  ./internal/sqa/scripts/asset_harvest.sh \
    --component-store <DATASET_DIR>/v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --track-ids 39,44,1,3,52 \
    --cache-dir ~/.cache/nre \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `asset_harvest--mode-full--obfuscation-no--dataset-H81_Panda128_6b0e750d_4cam_1lidar_v4`

**Manual validation required:** [Asset Harvester Test](README.md#asset-harvester-test)

- **Single command**:
  ```bash
  ./internal/sqa/scripts/asset_harvest.sh \
    --component-store <DATASET_DIR>/v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --track-ids 39,44,1,3,52 \
    --no-obfuscated \
    --cache-dir ~/.cache/nre \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `asset_harvest--mode-full--obfuscation-yes--dataset-H81_Panda128_0d59b8c8_4cam_1lidar_v4`

**Manual validation required:** [Asset Harvester Test](README.md#asset-harvester-test)

- **Single command**:
  ```bash
  ./internal/sqa/scripts/asset_harvest.sh \
    --component-store <DATASET_DIR>/v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --track-ids 39,38,42,46,29 \
    --cache-dir ~/.cache/nre \
    --tag <DOCKER_TAG>
  ```

##### Test identifier `asset_harvest--mode-full--obfuscation-no--dataset-H81_Panda128_0d59b8c8_4cam_1lidar_v4`

**Manual validation required:** [Asset Harvester Test](README.md#asset-harvester-test)

- **Single command**:
  ```bash
  ./internal/sqa/scripts/asset_harvest.sh \
    --component-store <DATASET_DIR>/v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840.json \
    --output-dir <OUTPUT_DIR>/<TEST_NAME> \
    --track-ids 39,38,42,46,29 \
    --no-obfuscated \
    --cache-dir ~/.cache/nre \
    --tag <DOCKER_TAG>
  ```

<!-- END AUTO-GENERATED TEST PLAN DOCUMENTATION -->
