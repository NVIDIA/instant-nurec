<!-- Copyright (c) 2024-2026 NVIDIA CORPORATION.  All rights reserved. -->

# NeuralReconstruction (NRE)

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC [zgojcic@nvidia.com](mailto:zgojcic@nvidia.com) or JANICK MARTINEZ ESTURO [janickm@nvidia.com](mailto:janickm@nvidia.com) .

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.

## Contents

- [NeuralReconstruction (NRE)](#neuralreconstruction-nre)
  - [Contents](#contents)
  - [Building the project](#building-the-project)
    - [Prerequisites](#prerequisites)
    - [Accessing GitLab](#accessing-gitlab)
    - [Cloning the repo](#cloning-the-repo)
    - [Authentication](#authentication)
    - [Build systems](#build-systems)
    - [Ansible Playbook automated system setup](#ansible-playbook-automated-system-setup)
    - [Bazel builds](#bazel-builds)
      - [Installation of `bazelisk`](#installation-of-bazelisk)
      - [Tuning of `bazel` build settings](#tuning-of-bazel-build-settings)
        - [Host compiler](#host-compiler)
        - [CUDA compiler](#cuda-compiler)
        - [CUDA runtime](#cuda-runtime)
        - [Execution of building / running a target with `bazel`](#execution-of-building--running-a-target-with-bazel)
        - [Disabling internal modules](#disabling-internal-modules)
        - [Execution using docker image](#execution-using-docker-image)
        - [Docker image storage in nvcr.io registry](#docker-image-storage-in-nvcrio-registry)
  - [Development](#development)
    - [Merge Request workflow](#merge-request-workflow)
      - [Code Ownership and MR approval](#code-ownership-and-mr-approval)
        - [Adding new CODEOWNERS groups and members](#adding-new-codeowners-groups-and-members)
      - [Merging MRs](#merging-mrs)
        - [Special case of release branches](#special-case-of-release-branches)
    - [Backward Compatibility](#backward-compatibility)
    - [Code structure](#code-structure)
    - [Configuring the IDE](#configuring-the-ide)
    - [Formatting code via `bazel`](#formatting-code-via-bazel)
    - [Changing Python dependencies](#changing-python-dependencies)
    - [Debugging Python targets](#debugging-python-targets)
    - [Large-file support](#large-file-support)
    - [Static code analysis](#static-code-analysis)
    - [Testing](#testing)
      - [Disable Caching](#disable-caching)
      - [Show logging](#show-logging)
      - [Running multiple iterations](#running-multiple-iterations)
      - [Performance related tests](#performance-related-tests)
      - [Multi-GPU support](#multi-gpu-support)
    - [Test sandboxes and writable cache paths](#test-sandboxes-and-writable-cache-paths)
      - [Cache sharing](#cache-sharing)
    - [Code coverage](#code-coverage)
      - [Unified Coverage (with CUDA & Slang)](#unified-coverage-with-cuda--slang)
    - [Profiling](#profiling)
      - [Adjusting Training Parameters](#adjusting-training-parameters)
    - [Benchmarking](#benchmarking)
  - [CI](#ci)
    - [Bazel test logs](#bazel-test-logs)
    - [Bazel telemetry](#bazel-telemetry)
    - [Analytics/Dashboard](#analyticsdashboard)
    - [CI Pipelines](#ci-pipelines)
  - [Container image analysis](#container-image-analysis)
  - [Usage](#usage)
    - [Datasets](#datasets)
    - [Configuration](#configuration)
    - [Logging](#logging)
    - [Running the code](#running-the-code)
    - [Rendering via the `render` command](#rendering-via-the-render-command)
    - [Remote rendering via gRPC](#remote-rendering-via-grpc)
    - [Exporting to USD](#exporting-to-usd)
    - [Exporting artifacts (mesh, rig trajectories and sequence tracks)](#exporting-artifacts-mesh-rig-trajectories-and-sequence-tracks)
    - [Resuming a training run from a pretrained checkpoint](#resuming-a-training-run-from-a-pretrained-checkpoint)
    - [Multi-GPU Training](#multi-gpu-training)
    - [Upgrading Artifacts](#upgrading-artifacts)
    - [Environment Variables](#environment-variables)
      - [NRE_ENV_RUN_ID](#nre_env_run_id)

## Building the project

### Prerequisites (can be skipped if using Ansible playbook, see below)

The project requires Linux (tested on Ubuntu 22.04), NVIDIA drivers (>=570), the CUDA Toolkit (>=12.8) (see [CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#id3)), cuDNN and the following system packages in order to build:

```bash
sudo apt-get install gcc-11 g++-11

# Install cuDNN 9 (when using CUDA 12)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cudnn9-cuda-12 libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

_Note: The CUDA Toolkit is needed by slang at runtime._

> The project may build successfully with earlier driver versions, but you may experience runtime issues when enabling certain features. If you have no permission to upgrade the GPU driver of your runtime environment, then installing the `cuda-compat` library of the required CUDA version could be a workaround (see [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#forward-compatibility)).

### Accessing GitLab

It is recommended to set up a SSH key pair to access `gitlab-master.nvidia.com` following [these instructions](https://docs.gitlab.com/ee/user/ssh.html). For example:

```bash
# Generates key pair, enter new password to avoid clear text storage.
ssh-keygen -t ed25519 -f .ssh/id_gitlab_nvidia_ed25519 -C "Gitlab SSH key"
eval $(ssh-agent -s)                  # Starts OpenSSH agent (run if ssh-add reports an error without it).
ssh-add .ssh/id_gitlab_nvidia_ed25519 # Adds your private key to the OpenSSH agent.
```

You can protect your SSH private key with a password locally, as prompted by `ssh-keygen`. Normally, you will only need to enter this password on the first interaction with the git repo per terminal session, provided that the `ssh-agent` is running (it is if `$SSH_AUTH_SOCK` is not empty).

Next, add the followings to the OpenSSH client config file `~/.ssh/config` (see [GitLab-Master FAQ](https://confluence.nvidia.com/x/vmSEGQ)):

```
Host gitlab-master.nvidia.com
    Hostname gitlab-master.nvidia.com
    PreferredAuthentications publickey
    AddKeysToAgent yes
    Port 12051
    User git
    IdentityFile ~/.ssh/id_gitlab_nvidia_ed25519
```

This allows to set different SSH identities for different servers. It is also possible to do this per repo ([instructions](https://docs.gitlab.com/ee/user/ssh.html#use-different-keys-for-different-repositories)).

### Cloning the repo

Once SSH access to the repo is configured as described [earlier](#accessing-gitlab), the repo can be cloned by

```bash
git clone ssh://gitlab-master.nvidia.com/nrs/nre.git
```

You should specify the user and the port as well if you do not have these in `~/.ssh/config`:

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/nrs/nre.git
```

### Authentication (can be skipped if using Ansible playbook, see below)

To be able to install dependencies, two personal access tokens are needed:

- **GitLab access token**:
  Create a `gitlab-master` personal access token with `api` scope at [link](https://gitlab-master.nvidia.com/-/profile/personal_access_tokens) (the token needs to have all possible permissions), and register the new token token in `~/.netrc` file, and replacing `<GITLAB_TOKEN>` with the created token string:

  ```
  machine gitlab-master.nvidia.com
  login oauth2
  password <GITLAB_TOKEN>
  ```

- **Artifactory access token** :
  Create an Artifactory identity token at [link](https://urm.nvidia.com/ui/user_profile) and register the new token in `~/.netrc` as follows and replacing `<YOUR_USER>` with your domain user name and `<ARTIFACTORY_TOKEN>` with the created token string:

  ```
  machine urm.nvidia.com
  login <YOUR_USER>
  password <ARTIFACTORY_TOKEN>
  ```

- **GitLab Container Registry**:
  If you also need to access the docker image(s) in the GitLab Contanier Registry (e.g. because you would like to access the `dev` docker image used in the CI jobs), make sure your local docker daemon is authenticated against GitLab using the same `<GITLAB_TOKEN>`:
  ```bash
  docker login gitlab-master.nvidia.com:5005 -u oauth2
  ```

> :warning: `~/.netrc` does not support comments, lines starting with `#` lead to errors like `User for gitlab-master.nvidia.com: ERROR: Exception: [...] EOFError: EOF when reading a line`. Also make sure that end of lines are just LF otherwise it won't be parsed correctly.

- **AWS credentials**:
  If you need to access S3 storage resources, make sure your local AWS credentials are configured in `~/.aws/config` and `~/.aws/credentials`. For example:
  In `~/.aws/config`:
  ```
  [profile pdx-team-ncore]
  region = us-east-1
  endpoint_url = https://pdx.s8k.io
  ```
  In `~/.aws/credentials`:
  ```
  [pdx-team-ncore]
  aws_access_key_id = team-ncore
  aws_secret_access_key = <YOUR_AWS_SECRET_ACCESS_KEY>
  ```
  Here you need to obtain `<YOUR_AWS_SECRET_ACCESS_KEY>` from the [Core Storage Portal](https://cssportal.sre.nsv.nvidia.com:4443/). You can verify if the setup is correct using `aws s3 --profile pdx-team-ncore ls s3://`. In supported places (e.g. NRM), you can use a path such as `s3@pdx-team-ncore://bucket-name/path/to/data` to access the data.

### Build systems

The project uses Bazel as the build system for builds, CI, testing and deployment. Bazel automatically ensures that the correct version of the dependencies are used upon each build and run, and offers additional development utilities (e.g. code formatting, dependency update automation).

### Ansible Playbook automated system setup

If you want to set up your machine automatically, please see the [ansible README.md](internal/scripts/ansible/README.md#development-setup) for more details. After running ansible, you should be able to directly jump to [building nre](#execution-of-building--running-a-target-with-bazel). Just make sure that [host](#host-compiler) and [CUDA](#cuda-compiler) compiler are correctly set.

### Bazel builds

#### Installation of `bazelisk` (can be skipped if using Ansible playbook, see above)

The repository uses `bazel` as the core build-system, in particular for CI / unit testing, building and deploying Docker images, launching benchmarks.
You do not need to set up a Python environment manually in this case.
Each code commit may require a different Bazel version based on `.bazelversion` included in the repo.
The correct `bazel` version is automatically invoked when using the `bazelisk` wrapper. See [bazelisk installation methods](https://github.com/bazelbuild/bazelisk#installation), or simply run the following.

```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

#### Tuning of `bazel` build settings

##### Host compiler

Bazel automatically selects the latest system compiler available in `PATH`. To select different C++ compilers, specify the followings in `<repo-root>/.bazelrc.user`, for example.

```
build --repo_env=CC=/usr/bin/gcc-11
build --repo_env=CXX=/usr/bin/g++-11
```

##### CUDA compiler

Bazel automatically selects the latest `nvcc` compiler available in `PATH`. To select a different CUDA toolchain, specify the following in `<repo-root>/.bazelrc.user`, for example.

```
build --repo_env=CUDA_PATH=/usr/local/cuda-12.8
```

##### CUDA runtime

Having multiple cuda toolchains installed can lead to runtime JIT compilation issues (e.g., in tiny-cuda-nn).
If this is the case, export the toolchain's library path to the runtime library search path via
the environment variable `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64` before executing a binary.
Additionally, cleaning the tiny-cuda-nn RTC cache via `bazel clean`.

##### Execution of building / running a target with `bazel`

Build targets can be seamlessly built and executed using `bazel` or `bazelisk`. Example usage:

```bash
bazel run //:run -- --help
```

The first `run` is the subcommand to run a target (common subcommands are `run`, `build` and `test`). `//:run` specifies the target named `run`, living in the root package `//`. Bazel targets are specified as `//<path>:<name>`, and are defined in `<repo-root>/<path>/BUILD.bazel`. Arguments that follow the first `--` (such as `--help` above) are passed to the target being executed. Note that the `run` and `test` subcommands also invoke the `build` subcommand on the same target.

> **Note:** Instead of using `//:run`, you can also launch with `//internal/scripts/pycena/runtime:pycena` to test the obfuscated build.

##### Disabling internal modules

By default the internal modules are included in the build. To exclude them, specify the following in `<repo-root>/.bazelrc.user`, for example.

```
build --config=no-internal
```

Not that the internal modules are not part of the pycena build, so obfuscation build targets do test the no-internal code path in any case.

##### Execution using docker image

The main `run`-associated docker image entrypoint can be invoked by loading the built docker image into the docker daemon and running it using:

```bash
bazel run //:load_run_image_oci
docker run -it --rm --gpus all nvcr.io/nvidian/ct-toronto-ai/nre_run:latest --help
```

If you are running from WSL2 you might have to force the use of the regular CUDA library (that is redirected to the host) via LD_PRELOAD as follows:

```bash
bazel run //:load_run_image_oci
docker run -it --rm --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so --gpus all nvcr.io/nvidian/ct-toronto-ai/nre_run:latest --help
```

##### Docker image storage in nvcr.io registry

CI pushes the Docker images that it produces into the `nvcr.io` registry. These images can then be pulled by other CI jobs, for local use, or for workflows on GPU clusters.

The naming and tagging conventions are as follows.

Image names:

- `nvcr.io/nvidian/ct-toronto-ai/nre_run`: unobfuscated run image
- `nvcr.io/nvidian/ct-toronto-ai/nre_obfuscated_run`: obfuscated run image
- `nvcr.io/nvidian/ct-toronto-ai/nre_tools`: unobfuscated tools image
- `nvcr.io/nvidian/ct-toronto-ai/nre_obfuscated_tools`: obfuscated tools image

Image tags:

- `latest`: most recent image produced in CI for the `main` branch
  - Images with this tag are updated for each MR merged into `main`
- `<major>.<minor>.<patch>-<sha>[-dev]`: precise NuRec version string which uniquely identifies the source tree used to build an image
  - Images for different source trees live side-by-side in the registry with different tags
  - A `-dev` suffix denotes an image coming from a feature branch, merge request pipeline etc. In other words, anything not present in the history of `main` or release branches
  - A tag of this form is applied by every CI job which pushes images. The version string is displayed near the top of each CI job log as `NuRec version string: <version>>`

_Tips for local use_

- The `latest` tag, also Docker's default if the user passes no tag, can refer to both an image in the registry and an image loaded from a local build
  - `docker run <image>:latest` automatically pulls an image from the registry only if `<image>:latest` does not exist locally
  - If you wish to refresh your local copy with the registry's `latest` image, run an explicit `docker pull`. Or use `load_*`targets from the above section to refresh from a local build.
- You can push your locally built images to the registry with commands of the form `bazel run //:push_run_image_oci [-- --tag ${TAG}]`
  - Without parameters, the image will be accessible by its digest, printed during command execution. Use `@sha256:<digest>` in place of a tag to reference it
  - You can assign a tag with the optional `--tag` parameter (or several of them by repeating the parameter). In this case, please make sure not to use `latest`, and also refrain from using NuRec version strings to avoid clashes with CI images

## Development

### Merge Request workflow

#### Code Ownership and MR approval

The repository uses a `CODEOWNERS` file to define code ownership and require approval from designated groups for different parts of the codebase.

- All MRs require at least one approval from `@nrs_owners/general_approvers`
  - This group shouldn't become a bottleneck, anyone actively developing NuRec can ask their functional area lead to be added
- Some paths require additional approval from specialized groups:
  - The `CODEOWNERS` file maps paths to owner groups and explains why each approval is needed
  - Required approvals and eligible approvers are visible in the GitLab MR UI
  - If an engineer belongs to multiple owner groups, their single approval satisfies all of them
- Avoid overloading code owners: first request a detailed review from engineers most familiar with your MR, even if they can't provide final approval

**Note:** GitLab does not automatically notify code owners about new MRs.

- **As an author:** Set _Assignees_ (people whose action you need) and _Reviewers_ (people who may be interested).
- **As an assignee:** Please respond within one business day.
  - Bookmark your assignments:
    `https://gitlab-master.nvidia.com/nrs/nre/-/merge_requests?scope=all&state=opened&assignee_username=<myusername>&not[author_username]=<myusername>`

##### Adding new CODEOWNERS groups and members

1. **Create a group** (if it doesn't exist): Create a new group under [NRS Owners](https://gitlab-master.nvidia.com/groups/nrs_owners/). If you don't have permissions, ask for help from [existing members](https://gitlab-master.nvidia.com/groups/nrs_owners/-/group_members).

2. **Add members**: Add the desired code owners to your group at `https://gitlab-master.nvidia.com/groups/nrs_owners/<groupname>/-/group_members`.

   - Important: Only direct members of the group receive the right to approve MRs, inherited membership is not sufficient.
   - Assign the _Developer_ role to people who only need to approve MRs, and the _Owner_ role to those who also need to invite new members.

3. **Invite the group**: If the group isn't already a member of NRE/NuRec, invite it as _Developer_ [here](https://gitlab-master.nvidia.com/nrs/nre/-/project_members?tab=groups) (needs to be done by a member of the group with _Maintainer_ or _Owner_ NuRec rights.).

4. **Edit CODEOWNERS**: Add the group to the `CODEOWNERS` file:

   ```
   # Explanation stating why this group is needed as additional approver
   [Example-section] @nrs_owners/<groupname>
   /path/to/code/owned
   ```

#### Merging MRs

Once an MR has all required approvals, the author is responsible for merging it via the _Merge_ button. GitLab then runs a final CI validation through the Merge Train before the code lands on the target branch.

As the person merging, the author is responsible for:

- Performing appropriate testing based on the MR's scope (e.g., local validation, optional CI jobs and Maglev workflows)
- Requesting re-review from approvers if significant changes were made after initial approval
- Monitoring post-merge CI jobs and reverting or fixing any breakages

##### Special case of release branches

Release branches are short-lived and only accept MRs critical to the release. The release manager tracks all changes and ensures they are either back-ported from `main` or integrated to `main` promptly.

To support this workflow, release branches use a special setup:

- Approval rules are the same as `main`
- Only the release manager can merge MRs
  - This is controlled via the `@nrs_owners/release_managers` group
  - Each release typically has one designated release manager, announced separately

### Backward Compatibility

To ensure that older artifacts remain compatible with newer versions of the codebase, the project includes an upgrade system that can automatically update an artifact's configuration and model state. For a detailed explanation of how this system works and how to add new upgrade functions, please refer to the [Artifact Upgrade System guide](./nre/utils/upgrade/README.md).

### Code structure

The goal of this codebase is to be as modular as possible, such that new applications can easily be built by only implementing a new `system`.

- **Models** All the building blocks of our codebase are found in the `models` folder. The different components are designed to be modular to ease experimentation with new configurations.

- **Systems** To train these models we package everything in Pytorch Lightning modules that we call `Systems`, these include everything you need during training and define the training strategy.

- **Datasets**:
  - `NCOREDataset`: directly reads data from [NRECore](https://gitlab-master.nvidia.com/nrs/ncore)'s V3/V4 shards. This is the preferred solution for internal AV data.

### Configuring the IDE

<details>
<summary>VSCode / Cursor</summary>

VSCode must be launched from the repository's root directory. First, build a Bazel Python target to generate the virtual environment:

```bash
bazel build //:run
```

Then, open a Python file in the IDE and check the status bar to see which Python environment is being used. Press `Ctrl-Shift-P` to open the command palette and type `Python: Select interpreter`. Select the interpreter from the Bazel-generated environment, typically located at `bazel-bin/run.runfiles/_main/run.venv/bin/python` (or similar paths for other targets). If it's not in the list, you can enter the path manually.

> **Note for Cursor users:** You may need to disable the "Python" extension by Anysphere for the Python interpreter selection to work correctly.

</details>

### Formatting code via `bazel`

To format all source code:

```bash
bazel run //:format
```

(Use `bazel run //:format.check` if you only want to check code-formatting violations without modifying the files)

Note a special case for Bazel: the commands above are used both for traditional formatting and for linter warnings. Bazel
attempts to fix some of the linter warnings automatically as part of the `//:format` target, but others will be left
untouched and need to be manually corrected by the user.

Formatting and linting of Bazel build files primarily uses Buildifier, but we have also integrated a few custom checks
written in Python at `bazel/format/starlark_custom_lint.py`. Feel free to extend these checks if needed.

#### Optional: Automated Code Formatting and Linting with `pre-commit` Hooks

The repository includes an optional pre-commit configuration that automatically runs before each commit.

<ins>_Setup (one-time):_</ins>

To enable `pre-commit`, install and configure the hooks, e.g. via

```bash
# Install pre-commit (if not already installed)
pip install pre-commit # or use any other package manager, if you don't like to install it system-wide

# Install the pre-commit hooks for this repository
pre-commit install
```

<ins>_How it works:_</ins>

- The `.pre-commit-config.yaml` configuration defines what is being executed before each commit. Hooks integrate the above Bazel formatting commands.
  - See YAML contents for a list of file formats that trigger each hook.
- Hook `bazel-format`
  - This modifies your files automatically to match NuRec formatting regulations.
  - In case formatting changes are made, the hook fails. It gives users a chance to review modifications before proceeding with the commit.
- Hook `bazel-format.check`
  - This runs after the format hook to report any left-over linter warnings, that the user will need to fix manually.

If necessary, users can bypass pre-commit hooks and merge their code as is with `git commit -n` (or `--no-verify`).

> **Note:** Pre-commit hooks are entirely optional. However, the CI pipeline will always check formatting regardless of whether you use local pre-commit hooks.

### Changing Python dependencies

Before adding new dependencies, you need to make sure there is proper NVIDIA SWIPAT (Software IP Audit) in place for the dependency.

To change python dependencies, edit the `.in` files in `deps/python`, and
regenerate the corresponding `.txt` files via the Bazel command below, then
submit the changed and regenerated files as a merge-request.
The
[deps/python/requirements_3_11_common.in](deps/python/requirements_3_11_common.in)
file contains non-arch-specific `any` python dependencies, while
[deps/python/requirements_3_11_x86_64.in](deps/python/requirements_3_11_x86_64.in)
and
[deps/python/requirements_3_11_aarch64.in](deps/python/requirements_3_11_aarch64.in)
are CPU architecture specific. If the new dependencies are _internal_ only, edit instead
[deps/python/requirements_3_11_internal_x86_64.in](deps/python/requirements_3_11_internal_x86_64.in) -
this "extends" the lockfile results of non-internal `requirements_3_11_x86_64`.
To list requirements specific to documentation builds, use file
[deps/python/requirements_3_11_docs.in](deps/python/requirements_3_11_docs.in).

Afterward, run the update command

```bash
bazel run //deps/python:update_all_requirements
```

This should finish error-free to update all resolved `.txt` requirements lock files and the top-level `uv.lock` (used for security scans).

### Debugging Python targets

This method allows to attach a remote debugger to a running interpreter and does not require any local installation, as it relies fully on the `debugpy` library that is available in all Python environments.

To enable, set `NRE_DEBUGPY_ENABLED=true`, e.g. `NRE_DEBUGPY_ENABLED=true bazel run //nre:run -- ...`.

> :warning: env vars are only propagated to `bazel run`. With `bazel test` you need `bazel test //... --test_env=NRE_DEBUGPY_ENABLED=true`.

Additional controls are via

- `NRE_DEBUGPY_PORT` to change the port debugpy is listening to
- `NRE_DEBUGPY_LOG_DIR` to change the log directory of debugpy
- `NRE_DEBUGPY_ALLOW_PORT_INCREMENT` to allow debugpy to try listening on next consecutive ports if `NRE_DEBUGPY_PORT` is busy
- `NRE_DEBUGPY_BREAK_ON_CONNECTION` to automatically insert a breakpoint immediately after establishing connection with debugpy

Additionally it's possible to use the tools in `nre.utils.debug` to attach multiple debuggers to the same running process (using different ports). This can be helpful when reaching breakpoints take a long time.

For details, please refer to [nre/utils/debug/README.md](nre/utils/debug/README.md).

### Large-file support

Please refer to [this wiki page](https://gitlab-master.nvidia.com/nrs/nre/-/wikis/Development/Large-file-support).

### Static code analysis

The repository makes use of `mypy` for static-code validation of the important components. These are executed as bazel test targets with `_mypy` suffixes, for example:

```bash
bazel test //:run_mypy
```

In order to speed up _local_ execution of `mypy` there are two options:

- Make persistent user-cache available to `mypy`: due to the way Bazel sandboxes are setup, `mypy` is not able to
  generate its cache in the Bazel output hierarchy for faster incremental analysis. We are relying on a separate local
  cache folder, which can be enabled by setting the following in `<repo-root>/.bazelrc.user`.

  ```
  # Make use of local mypy cache
  run --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.mypy_cache
  test --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.mypy_cache
  test --test_env=MYPY_CACHE_DIR=<ABSOLUTE-PATH-TO>/.mypy_cache
  ```

  Without these options there will be no caching of intermediate incremental mypy results (Bazel caching of final test
  states is not affected by this and is still active).

- Configure your IDE to run the `python` interpreter and `mypy` from the main Python environment and make sure the IDE passes the argument `--config-file=bazel/typing/mypy.ini` to `mypy`.
  For example, in VSCode, install the [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) plugin, make sure the `mypy` Python package is installed in your main environment, and add following settings (after replacing the `<>` parts) to the VSCode workspace settings in `<repo-root>/.vscode/settings.json`:

  ```json
  "mypy-type-checker.importStrategy": "fromEnvironment",
  "mypy-type-checker.interpreter": [
    "</path/to/your/main/python/env>/bin/python"
  ],
  "mypy-type-checker.path": [
    "</path/to/your/main/python/env>/bin/mypy"
  ],
  "mypy-type-checker.args": [
    "--config-file=bazel/typing/mypy.ini"
  ],
  ```

  A simple verification is to create a file `example.py` with the following content, and verify whether VSCode decorates the code with a Mypy error (matching the output of the command `mypy example.py`).

  ```python
  def func() -> str:
      i: int = 1
      return i
  ```

### Testing

Automated testing of the NRE codebase is implemented with bazel and pytest
framework, and configured for execution per commit in GitLab CI.

This test configuration is defined in [.gitlab-ci.yml](.gitlab-ci.yml) as
follows:

```bash
bazelisk run //:format.check
bazelisk build ...
bazelisk test ...
```

To run the same process locally, just execute the same commands directly with
bazel, for example:

```bash
bazel test ...
```

The bazel target can be specialised to run just specific tests, for example:

```bash
# Run all the tests inside the `nre/datamodules` directory
bazel test //nre/datamodules/...

# Run just the tests in the `nre/config/schema_test.py` file
bazel test //nre/config:schema_test
```

> **NOTE:** The bazel `--test_filter` option, which should allow more precise
> filtering of which tests to run, does not work with pytest.

#### Algorithm Probing and Test Data Generation

See internal/scripts/test_data/README.md for more details.

#### Disable Caching

By default bazel will cache test results to avoid re-running tests that are
known to pass. This can be disabled with the `--cache_test_results` parameter:

```bash
bazel test --cache_test_results=no //nre/datamodules:dataloader_test
```

#### Show logging

By default bazel will not show the logging from each test, unless a test fails.
This can be changed with the `--test_output` parameter. This will even work
with cached test results, and will show the logging from a previously cached
test run.

```bash
bazel test --test_output=all //nre/datamodules:dataloader_test
```

#### Running multiple iterations

Before merging a new test it's useful to run multiple iterations to check it
passes reliably. This can be done using the `--runs_per_test` parameter. This
will run each test iteration in parallel (up to a default amount). The
`--local_test_jobs` parameter can be used to avoid this and limit how many tests
can be run at the same time.

```bash
bazel test --runs_per_test=20 --local_test_jobs=1 //nre/datamodules:dataloader_test
```

#### Performance related tests

Some of the tests are aimed towards testing the efficiency of certain implementation and we don't want to run them in CI. For such tests please include a decorator

```python
@unittest.skipUnless(os.environ.get("RUN_PERF_TESTS") == "1", "Performance tests are skipped by default. Set RUN_PERF_TESTS=1 to run.")
```

To run the performance related tests locally you can then set the environment variable for example:

```bash
RUN_PERF_TESTS=1 bazel run //libs/gaussian_mcmc:gaussian_mcmc_test
```

#### Multi-GPU support

When running tests on a multi-GPU system, the test framework automatically
load-balances single-GPU tests across all available GPUs. This improves test
throughput and spreads memory usage across GPUs rather than overloading GPU 0.

The number of GPUs is auto-detected at Bazel fetch time using `nvidia-smi`.
No manual configuration is required.

This feature applies to tests using the `pytest_test` macro. A specific GPU is
exposed to each test via `CUDA_VISIBLE_DEVICES` based on a hash of the test
name. The selection is consistent across runs, but not necessarily optimal
when running a small set of tests.

The GPU selection is logged at the start of each test:

```text
[GPU Selection] Exposing GPU 1 (CUDA_VISIBLE_DEVICES=1), out of 2 total GPUs
```

**Multi-GPU tests**: Some tests can natively use multiple GPUs. These tests
should be tagged with `multi-gpu` in their `pytest_test` rule to leave all GPUs
exposed:

```python
pytest_test(
    name = "<my_multi_gpu_test>",
    tags = ["multi-gpu"],  # Expose all available GPUs
    deps = [...],
)
```

Multi-GPU tests will log:

```text
[GPU Selection] Multi-GPU test, exposing all available GPUs
```

**Forcing GPU count**: To override auto-detection (e.g., to test on a subset of
GPUs or simulate a different configuration), set `FORCE_GPU_COUNT` in
`.bazelrc.user`:

```bash
common --repo_env=FORCE_GPU_COUNT=2
```

#### Test sandboxes and writable cache paths

Certain software components require access to `$HOME/.cache`, which is read-only by default from within bazel during sandboxed test execution. To address this limitation, we configure a writable and persistent path that redirects cache operations away from read-only directories.

You can enable this by adding the following configuration to your `.bazelrc.user` file:

```bash
# Set up writable and persistent path for bazel tests
test --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.test_cache

# Cache directories for different software components which require one
test --test_env=ASSET_HARVESTER_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/asset_harvester
test --test_env=TORCH_HOME=<ABSOLUTE-PATH-TO>/.test_cache/torch
test --test_env=TRITON_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/triton
test --test_env=TCNN_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/tinycudann
test --test_env=HF_HOME=<ABSOLUTE-PATH-TO>/.test_cache/huggingface
```

The repo's `.gitignore` file allows to set `<ABSOLUTE-PATH-TO>` to the local checkout path, as it excludes `.test_cache/`.

##### Cache sharing

Cache folders in the above setup will only be visible to `bazel test`, while `bazel run` and direct execution of NuRec SW will continue to cache under `HOME`. It is possible to share the cache between all these use cases, with the caveat that custom cache paths will be global for the user and not limited to the NRE repo. Proceed as follows:

Define the paths in `~/.bashrc` and remember to source it again in open shells:

```bash
export ASSET_HARVESTER_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/asset_harvester
export TORCH_HOME=<ABSOLUTE-PATH-TO>/.test_cache/torch
export TRITON_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/triton
export TCNN_CACHE_DIR=<ABSOLUTE-PATH-TO>/.test_cache/tinycudann
export HF_HOME=<ABSOLUTE-PATH-TO>/.test_cache/huggingface
```

Merely expose these paths to Bazel test sandboxes through `.bazelrc.user` if you want to avoid duplication:

```bash
# Set up writable and persistent path for bazel tests
test --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.test_cache

# Cache directories for different software components which require one
test --test_env=ASSET_HARVESTER_CACHE_DIR
test --test_env=TORCH_HOME
test --test_env=TRITON_CACHE_DIR
test --test_env=TCNN_CACHE_DIR
test --test_env=HF_HOME
```

### Code coverage

Bazel can be used to gather code coverage measurements for the project's tests.
For now this support is present for the Python and C/C++ languages.

The following commands run a chosen set of tests with coverage measurement and
generate a combined HTML report for them in folder `coverage`, containing
information about line and function coverage.

```bash
bazel coverage <test_targets>
genhtml --output coverage bazel-out/_coverage/_coverage_report.dat
```

The following commands perform coverage measurements for the full set of tests
in the project (excluding static analysis etc.), and are integrated to CI as the
`coverage` job. For any CI pipeline, the coverage job can be triggered manually
and the HTML report observed through the "Browse" button in the job's detail
page.

```bash
bazel coverage \
  $(bazel query 'kind(".*_test rule", //...) except (attr("name", "^_|^requirements_|_mypy$", //...) + attr("tags", "manual", //...))')
genhtml --output coverage bazel-out/_coverage/_coverage_report.dat
```

Important notes:

- By default, these commands include the test code itself in coverage measurement.
  This is useful to notice any unused code in the test codebase, but the final
  metric as a coverage percentage needs to be interpreted accordingly.
  This behavior can be disabled with option `--noinstrument_test_targets`.

#### Unified Coverage (with CUDA & Slang)

Unified Coverage uses a two-stage approach for CUDA code:

- **Host Coverage** — Uses `gcov` (C++ & CUDA-host) and `Coverage.py` (Python) to measure line coverage of CPU-side code.
- **Device Coverage** — Uses NVIDIA Nsight Compute (NCU) profiling to infer line coverage of GPU kernel code.

Both stages are merged into a single HTML report via `lcov` and `genhtml`.

Run the following command to generate the unified report:

```bash
bazel run //internal/scripts/cuda_coverage:run_combined_coverage
```

This produces a main coverage report at `combined_coverage_html/index.html` and a
"test miss" report showing source files not touched by any test. Key options:

```bash
# Skip host or device coverage
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-host
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-device

# Run on a single test target
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --target //path/to:test

# Custom output directory
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --output-dir my_coverage
```

For full documentation of the CUDA & Slang coverage infrastructure, including script details, architecture, CI integration, and troubleshooting, see [`internal/scripts/cuda_coverage/README.md`](internal/scripts/cuda_coverage/README.md).

### Profiling

For effective profiling during development:

- **Reduce workloads for faster profiling**:

  - Limit iterations: `dataset.n_samples_per_epoch=${n_iterations}`
  - Shorten dataset duration: `dataset.seek_offset_sec=1.0 dataset.duration_sec=0.5`

- **Scoped Timer**:

  The scoped timer provides detailed timing information with low overhead for different components of the NRE pipeline, enabling precise performance analysis and optimization.

  **Configuration:**

  You can enable scoped timing by adding these parameters to your NRE launch command:

  ```bash
  scopedtimer.enabled=true scopedtimer.verbosity=BASIC scopedtimer.synchronize=true scopedtimer.profiling_backend=NONE
  ```

  **Step-gated emission (backend-agnostic):**

  - `scopedtimer.emit_start_step`: first step to emit backend ranges (default: disabled)
  - `scopedtimer.emit_num_steps`: number of steps to emit after `emit_start_step`
  - `scopedtimer.emit_repeat_interval`: repeat period in steps (0/None => single window)

  Example: emit for steps 1–50, 1001–1050, 2001–2050, ... with profiling ranges

  ```bash
  scopedtimer.enabled=true \
  scopedtimer.emit_start_step=1 \
  scopedtimer.emit_num_steps=50 \
  scopedtimer.emit_repeat_interval=1000
  ```

  | Parameter                          | Description                                      | Options                    |
  | ---------------------------------- | ------------------------------------------------ | -------------------------- |
  | `scopedtimer.enabled`              | Enables the scoped timer                         | `true`/`false`             |
  | `scopedtimer.verbosity`            | Sets output detail level                         | `NONE`, `BASIC`, `DETAILS` |
  | `scopedtimer.synchronize`          | Synchronizes GPU operations for accurate timing  | `true`/`false`             |
  | `scopedtimer.profiling_backend`    | Selects profiling backend                        | `NONE`, `TRACY`, `NVTX`    |
  | `scopedtimer.emit_start_step`      | First step to emit backend ranges                | int or null                |
  | `scopedtimer.emit_num_steps`       | Number of steps to emit after start              | int or null                |
  | `scopedtimer.emit_repeat_interval` | Repeat period in steps (0/None => single window) | int or null                |

  **Output Options:**

  - **Console Output:** Timing results are displayed in the terminal by default
  - **File Output:** You can save results for detailed analysis:
    ```bash
    scopedtimer.logfile=timing.log
    ```
    Creates `timing.log` in the job run directory with comprehensive timing data.

  **Custom Ranges:**

  You can instrument your own code using the `ScopedTimer` class:

  ```python
  # Context manager (recommended for most use cases)
  with ScopedTimer(name="data_loading", tag=TimingTag.DEFAULT):
      data = load_dataset()

  # Function decorator (ideal for profiling entire functions)
  @ScopedTimer("model_inference", tag=TimingTag.DEFAULT)
  def run_inference(model, input_data):
      return model(input_data)

  # Manual control (useful for complex timing scenarios)
  timer = ScopedTimer("custom_range", tag=TimingTag.DEFAULT)
  timer.start("preprocessing")
  # preprocessing code
  timer.stop("preprocessing")

  timer.start("computation")
  # computation code
  timer.stop("computation")
  ```

  **Profiling Backends:**

  - **`none`**: Text-only timing output to console/logfile
  - **`nvtx`**: NVTX markers for NVIDIA Nsight Systems profiling
  - **`tracy`**: Real-time profiling with Tracy GUI (requires `TRACY_ENABLE=1`)

  **Advanced Profiling:**

  For detailed visual profiling with specialized tools, see the dedicated profiling scripts:

  - **Tracy profiling**: `./internal/scripts/profilers/tracy/run_with_tracy.sh` - Real-time GUI profiling
  - **NSys profiling**: `./internal/scripts/profilers/nsys/run_with_nsys.sh` - File-based profiling

  See `internal/scripts/profilers/README.md` for complete profiling documentation.

  **Framework Integration:**

  The scoped timer integrates with PyTorch Lightning and can be passed as a profiler to the `Trainer` class for seamless profiling during training.

  > **Note:** PyTorch Lightning integration is actively being developed and behavior might be subject to changes.

- **Using NVIDIA Nsight Systems**:
  ```bash
  nsys profile --trace=cuda,nvtx bazel run //:run -- [parameters]
  ```
- **Enabling blocking mode (can be useful)**:

  ```bash
  CUDA_LAUNCH_BLOCKING=1 nsys profile --trace=cuda,nvtx bazel run //:run -- [parameters]
  ```

- **Enable PyTorch autograd NVTX profiling**:

  ```bash
  profiling.enabled=true profiling.params.start_step=0 profiling.params.emit_nvtx=true
  ```

- **Optional: Capture only ScopedTimer windows in Nsight Systems**:

  When using `scopedtimer.profiling_backend=NVTX` **with emit windows configured**, ScopedTimer calls `cudaProfilerStart()`/`cudaProfilerStop()` at emit window boundaries, allowing nsys to only collect data during active windows. This results in **much smaller trace files** for long training runs with `--capture-range=cudaProfilerApi`.

  ```bash
  nsys profile --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    bazel run //:run -- --config-name=... \
      scopedtimer.enabled=true \
      scopedtimer.profiling_backend=NVTX \
      scopedtimer.emit_start_step=0 \
      scopedtimer.emit_num_steps=50 \
      scopedtimer.emit_repeat_interval=1000
  ```

  This approach **actually stops data collection** outside emit windows, keeping trace files small even for multi-hour runs.

For other informations please refer to [this wiki page](https://gitlab-master.nvidia.com/nrs/nre/-/wikis/Development/Profiling).

#### Adjusting Training Parameters

When reducing the number of iterations for profiling, you may need to adjust related parameters:

```
dataset.n_samples_per_epoch=${n_iterations}
model.strategy.densify.start_iteration=${start_iteration}
model.strategy.densify.end_iteration=${n_iterations}
model.strategy.prune.start_iteration=${start_iteration}
model.strategy.prune.end_iteration=${n_iterations}
model.layers.dynamic_rigids.track_albedo.start_global_step=${start_iteration}
model.layers.dynamic_rigids.track_scale.start_global_step=${start_iteration}
model.layers.dynamic_rigids.tracks_calib.start_global_step=${start_iteration}
model.strategy.prune.end_iteration=${n_iterations}
model.layers.dynamic_deformables.track_albedo.start_global_step=${start_iteration}
model.layers.dynamic_deformables.track_scale.start_global_step=${start_iteration}
model.layers.dynamic_deformables.tracks_calib.start_global_step=${start_iteration}
```

### Benchmarking

The benchmarking framework is described in [this README](internal/workflows/cluster_toolbox/benchmark/README.md).

The benchmarking results can be found in the [nvidia-toronto/nre-benchmark](https://wandb.ai/nvidia-toronto/nre-benchmark) project on Weights & Biases: see [NRE Benchmarking Dashboard](https://wandb.ai/nvidia-toronto/nre-benchmark/reports/NRE-Quality-Benchmark-Dashboard--Vmlldzo3MzAyNzU2) or
[Benchmark Sweeps](https://wandb.ai/nvidia-toronto/nre-benchmark/sweeps).
You need to request a W&B license and be added to the `nvidia-toronto` group to access it (follow https://wandb.ai/site/join-your-team/nvidia).

## CI

### Bazel test logs

Bazel only prints the output of failing test cases to GitLab CI's main job log. For better visibility into test
behavior, jobs that run `bazel test` or `bazel coverage` also package test logs and outputs in the `testlogs/` folder
as job artifacts. This includes:

- Test stdout/stderr logs (`test.log`)
- Any files written to the test's undeclared outputs directory
- Per-test coverage output in LCOV format (`coverage.dat`), for `bazel coverage` only

These are generally exposed on the MR page as "Bazel test logs and outputs" and are available for both passing and
failing jobs.

### Bazel telemetry

CI jobs automatically capture Bazel telemetry for debugging and performance analysis. The following files are
generated in `bazel-telemetry/` for each Bazel invocation:

- **Build Event Protocol**: `bep_00_build.json`, `bep_01_test.json`, etc.
  - [Documentation](https://bazel.build/remote/bep) — structured build/test results, target info, timing data
- **Trace profiles**: `profile_00_build.json.gz`, `profile_01_test.json.gz`, etc.
  - [Documentation](https://bazel.build/advanced/performance/json-trace-profile) — visualize in `chrome://tracing` to
    analyze build/test performance
- **Execution logs**: `exec_log_00_build.binpb`, `exec_log_01_test.binpb`, etc. (build/test/run/coverage only)
  - [Documentation](https://bazel.build/remote/cache-remote#compare-logs) — detailed binary logs, e.g. to debug
    remote cache misses
- **Explain logs**: `explain_00_build.txt`, `explain_01_test.txt`, etc. (build/test/run/coverage only)
  - Human-readable explanation of why actions were (re)executed

These files are available as job artifacts.

### Analytics/Dashboard

A set of scripts used to gather and visualize data about CI jobs for the NRS group of projects is described in [this README](internal/scripts/ci/analytics/README.md).

The scripts are integrated into CI as the `ci_analytics` job, running on a schedule. The latest data is published on the project's GitLab pages.

## Container image analysis

Tooling and procedures to study and diff container images produced in CI or locally are documented in [this README](internal/scripts/ci/image_analysis/README.md).

### CI pipelines

More documentation about the CI pipelines can be found in [this README](.gitlab/ci/README.md).

## Usage

### Datasets

For a quick first test, there is a short clip [test_data_ncore](https://gitlab-master.nvidia.com/nrs/nre/-/packages?orderBy=created_at&sort=desc&search[]=test_data_ncore) (aux data included) in the GitLab Package Registry. Make sure to download its latest version. For more datasets, please refer to [this wiki page](https://gitlab-master.nvidia.com/nrs/nre/-/wikis/Datasets).

### Configuration

Command-line execution of the program requires to feed a top-level configuration YAML file via `--config-name=<path/to/some.yaml>`. This specifies what the program is supposed to execute, i.e. the system, models, lossess, logger etc. to use. A set of predefined configuration files are provided under [configs/apps/](configs/apps) and [configs/experimental/](configs/experimental). These represent the top level of a configuration hierarchy that maps to the directory tree under [configs/](configs) with the help of [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/en/latest/).

_Note: for convenience the [configs/personal/](configs/personal/) path can be used to store any personal configs not intended to be shared as the path is ignored by git._

All configuration parameters loaded from the hierarchy can be overridden via command line (see [syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)).

For example, if you pass `configs/apps/top.yaml` to the program, and it contains the line `/dir/subdir: config1`, it means that there exists a directory `configs/dir/subdir/config1.yaml`. If `config1.yaml` contains

```yaml
mystruct:
  myparam: 0
```

then you can override this parameter with an extra argument `dir.subdir.mystruct.myparam=42` from command line (notice the lack of the `--` prefix). Please refer to the [Hydra doc](https://hydra.cc/docs/intro/) for more info.

### Logging

The code supports Weights & Biases (W&B, default), TensorBoard and a Dummy logger for logging. You can select these via the command-line options `logger=wandb`, `logger=tensorboard`, or `logger=dummy`. The default parameters for each can be found in `configs/logger/<logger_name>.yaml` (see [wandb.yaml](configs/logger/wandb.yaml) or [tensorboard.yaml](configs/logger/tensorboard.yaml)), and can be overridden by extra command-line arguments `logger.<param>=<value>`.

By default, the program assumes a W&B online account and membership of the `nvidia-toronto` group on W&B, and would prompt for the corresponding credientials (unless the user issued `wandb disabled` or `wandb login <wandb_api_key>` with their valid [wandb API key](https://wandb.ai/authorize) in a Python environment with the `wandb` package pip-installed). The easiest to get away without a W&B account is by adding the extra argument `logger.offline=true` to the program launch.

Developers can request a W&B account and `nvidia-toronto` group membership from Zan Gojcic.

### Running the code

You can run our code in `train`, `val`, `trainval` or `test` mode, and by using one of the provided config files.
Note that you do not need a Weights & Biases (W&B) account to be able to run the program, and you can suppress the related prompt as described in [Logging](#logging).)
For example, one can train and validate a static 3D Gaussians scene as follows.

- via Bazel:

  ```bash
  bazel run //:run -- \
    --config-name=apps/AV/NV/3dgut_static.yaml \
    out_dir=<where-the-outputs-will-be-saved> \
    mode=trainval \
    dataset.path=<path-to-dataset>
  ```

- via Python:

  ```bash
  python run.py \
    --config-name=apps/AV/NV/3dgut_static.yaml \
    out_dir=<where-the-outputs-will-be-saved> \
    mode=trainval \
    dataset.path=<path-to-dataset>
  ```

Or for a dynamic reconstruction:

- via Bazel:

  ```bash
  bazel run //:run -- \
    --config-name=apps/AV/NV/3dgut_dynamic.yaml \
    out_dir=<where-the-outputs-will-be-saved> \
    mode=trainval \
    dataset.path=<path-to-dataset>
  ```

- via Python:

  ```bash
  python run.py \
    --config-name=apps/AV/NV/3dgut_dynamic.yaml \
    out_dir=<where-the-outputs-will-be-saved> \
    mode=trainval \
    dataset.path=<path-to-dataset>
  ```

These will export a `parsed.yaml` to the output directory, which is the entire parsed Hydra configuration tree (see [Configuration](#configuration)), saved in a single YAML file that is required by subsequent training/test/validation steps.

Once the model is trained, you can either resume training (see [Resuming a training run from a pretrained checkpoint](#resuming-a-training-run-from-a-pretrained-checkpoint)), or run the validation or test loop, using the model checkpoint and the parsed config in the output directory, as follows. Note that `out_dir` in validation or testing run defaults to the output directory of the training run, but can also be overridden.

- via Bazel:

  ```bash
  bazel run //:run -- \
    --config-name=<path-to-the-parsed-config-file> \
    resume=<filename-of-the-pretrained-checkpoint> \
    mode=<val/test>
  ```

- via Python:

  ```bash
  python run.py \
    --config-name=<path-to-the-PARSED-config-file> \
    resume=<path-to-the-pretrained-checkpoint> \
    mode=<val/test>
  ```

The mandatory `dataset.path` parameter is interpreted in a dataset-dependent way:

- For `ngp`-based datasets, the `path` parameter can either indicate a _single_ `.json` file
  with NGP input transformations, or a directory, from which all `.json`
  files with NGP input transformations are loaded from

- For `ncore`-based datasets, the `path` parameter has to point to a _single-sequence_ NCore meta-file
  (_multi-chunk_ datasets are deprecated). The sequence time range can be restricted with additional
  `seek_offset_sec` / `duration_sec` dataset configuration parameters

> Note: for supported model, one can use NRend (fast cuda renderer) in validation (only) or testing modes by enabling it through the setting `test.nrend.enabled=true`.

### Rendering via the `render` command

The `render` command (`render` CLI) can be used to render views from a trained model. It requires a USDZ file as input. It does not need the original input clip like `mode=val`, or a gRPC service to be launched for rendering frames.

The `render` command supports rendering from a training trajectory, or from a trajectory that is at a fixed rotation/translation away from each training pose. It does not yet support user-fed trajectory and intrinsics, although this is planned. It is limited to using the intrinsics of a training camera, which needs to be selected by name.
For example, to render from the original training views:

```bash
bazel run //:run -- render \
  --artifact-path "${USDZ_PATH}" \
  --output-dir "${RENDER_OUTPUT_DIR}" \
  --frame-step 1 \
  --image-scale 0.25 \
  --camera-id camera_front_wide_120fov \
  --camera-id camera_front_tele_30fov \
  --image-format png \
  --frame-naming frame-end-timestamp \
  --replicate-training-views
```

The `--replicate-training-views` option ensures that the trained per-frame ISP parameters are applied, otherwise the pixel intensities and colors may differ from those in the training images. Use `--no-replicate-training-views` if any transformation (see below) is to be applied, generating novel views, or `--rolling-shutter-duration` is overriden or `--enable-editing-actors` is set.

A fixed transformation can be applied in vehicle space (see [conventions](https://nrs.gitlab-master-pages.nvidia.com/ncore/data/conventions.html)) to every pose of the rig trajectory by appending the following options.

```bash
  --rig-rotation-offset <yaw> <-roll> <pitch>  --rig-translation-offset <tx> <ty> <tz>
```

The `--enable-editing-actors` flag allows modifications to dynamic actor poses and only renders the dynamic actors present within the current frame's start and end timestamp. The `--demo-actor-transform` flag is a demo that applies a rotation along the vertical axis to the editable (dynamic) actors in the rendered images. Note: `--demo-actor-transform` requires `--enable-editing-actors` to be set.

For further details of the `render` command, please refer to `bazel run //:run -- render --help`.

### Remote rendering via gRPC

With a trained model (see the section above) you can run a gRPC server which enables remote render queries. The implemented API is defined [here](nre/grpc/protos/) (as `sensorsim.proto`) and can be tested with a demo client [here](https://gitlab-master.nvidia.com/mtyszkiewicz/nre-gradio-client).

"Artifacts" refers to `.usdz` files produced by the training script. The glob pattern should be passed to the python script rather than expanded in shell, so in most cases you want to put it in quotes (to avoid premature expansion).

> Note: the gRPC API does **not** allow for overriding hydra config parameters (because it would be unclear to which of the many checkpoints it applies).

> :warning: Running with Bazel with clean repository state (use `git stash/git pop`) is _strongly_ recommended for everything but debugging - this is the only configuration which correctly logs NRE version and makes the results reproducible.

- via Bazel:

  ```
  bazel run //:run -- serve-grpc \
    --artifact-glob '<glob pattern>' \
    --host <hostname, defaults to localhost> \
    --port <defaults to 8080> \
    --health-port <optional; if set, health is served on that port instead of the main gRPC port>
  ```

- via Python:

  ```
  python run.py serve-grpc \
    --artifact-glob '<glob pattern>' \
    --host <hostname, defaults to localhost> \
    --port <defaults to 8080> \
    --health-port <optional; if set, health is served on that port instead of the main gRPC port>
  ```

> Note: for supported model, one can use NRend (fast cuda renderer) with gRPC by enabling it through the option `--enable-nrend`. Similarly, Difix render post-processing can be enabled via
> `--enable-difix`, with optional parameters specifying Difix model checkpoint `--difix-url=<str>` and local checkpoint storage `--difix-cache=</full/path/to/desired/dir>` (default value`~/.cache/NRE/difix/checkpoints`).

The `render-grpc` command is a demo gRPC client for a basic testing of the gRPC service by rendering training-view frames via the service and saving them to an output directory.
Usage example:

```bash
bazel run //:run -- render-grpc \
  --artifact-path=${USDZ_PATH} \
  --output-dir=${RENDER_OUTPUT_DIR}/${CAMERA_ID} \
  --image-format png \
  --port 8080 \
  --height 540 \
  --frame-step 1 \
  --camera-id ${CAMERA_ID}
```

For further details of the `render-grpc` command, please refer to `bazel run //:run -- render-grpc --help`.

### Exporting to USD

Following a reconstruction trained by the commands given above, USD files can be exported for visualization and other downstream tasks in Omniverse.

To export USD files to a certain directory, the `export-usdz-artifact` sub-command is used with the main `run` script.
This example command uses Bazel, but works equivalently with the Python environment by replacing `bazel run //:run --` with `python run.py`.

This command loads a checkpoint from a previous training run, given by a `parsed.yaml` file, and passed as the `config-name` to the script.

```bash
bazel run //:run -- export-usdz-artifact \
  --config-name='/path/to/out/config/parsed.yaml' \
  --checkpoint-name='/path/to/out/checkpoints/last.ckpt' \
  --output-dir='/path/to/usd_out'
```

With this command, several files, including the main USD file, by default called `default.usda`, will be stored in the `/path/to/usd_out` directory.

For a more detailed example, [this wiki page on USD export](https://gitlab-master.nvidia.com/nrs/nre/-/wikis/USD-Export) can be consulted.

### Exporting artifacts (mesh, rig trajectories and sequence tracks)

For certain downstream applications (ie. Alpamayo Simulation) it can be useful to export specific artifact (mesh, rig trajectories, sequence tracks). While NRE can be configured to store these artifacts during training, for convenience we also provide dedicated export commmands:

- via Bazel:

  ```bash
  bazel run //:run -- export-mesh \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

  ```bash
  bazel run //:run -- export-rig-trajectories \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

  ```bash
  bazel run //:run -- export-sequence-tracks \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

- via Python:

  ```bash
  python run.py export-mesh \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

  ```bash
  python run.py export-rig-trajectories \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

  ```bash
  python run.py export-sequence-tracks \
    --config-name '<path-to-config-with-dataset-specs>' \
    --output-dir '<path-to-export>'
  ```

### Resuming a training run from a pretrained checkpoint

Training runs can be resumed from a pretrained checkpoint by pointing to the parsed config file and selecting the desired checkpoint as:

- via Bazel:

```bash
# Run the training
bazel run //:run -- \
  --config-name=apps/Replica/ngp_ds_nerfacc_acc_fused_fv.yaml \
  out_dir=out/example/ \
  mode=trainval \
  dataset.path=<path-to-dataset>

# Assume that the above created a run with hash: B9paa282846FiWqRQj9NZJ
# Resume training with the latest checkpoint:
bazel run //:run -- \
  --config-name=out/example/B9paa282846FiWqRQj9NZJ/config/parsed.yaml \
  resume=last \
  mode=train

# Resume training from an intermediate checkpoint (assume that epoch=4.ckpt was saved) - again one of the two:
bazel run //:run -- \
  --config-name=out/example/B9paa282846FiWqRQj9NZJ/config/parsed.yaml \
  resume="epoch=4.ckpt" \
  mode=train
```

- via Python:

```bash
# Run the training
python run.py \
  --config-name=apps/Replica/ngp_ds_nerfacc_acc_fused_fv.yaml \
  out_dir=out/example/ \
  mode=trainval \
  dataset.path=<path-to-dataset>

# Assume that the above created a run with hash: B9paa282846FiWqRQj9NZJ
# Resume training with the latest checkpoint:
python run.py \
  --config-name=out/example/B9paa282846FiWqRQj9NZJ/config/parsed.yaml \
  resume=last \
  mode=train

# Resume training from an intermediate checkpoint (assume that epoch=4.ckpt was saved) - again one of the two:
python run.py \
  --config-name=out/example/B9paa282846FiWqRQj9NZJ/config/parsed.yaml \
  resume="epoch=4.ckpt" \
  mode=train
```

The log files and checkpoints of the resumed training runs will be stored in the same folder (changing `config.out_dir` is not allowed), and will be combined with previous runs for a nice continuous visualization of training curves (for both wandb and tensorboard). To get the data from an individual run you can check `<out_dir>/wandb` folder which will contain multiple `run-*` folders. Or in case of tensorboard, directly the `<out_dir>/` folder which will contain multiple `*.tfrecords` files.

### Multi-GPU Training

One can use multiple GPUs to speed up training. Simply specify the number of GPUs to use via the `trainer.world_size` parameter.

For example:

```bash
python run.py --config apps/prod/Hyperion-8.1/car2sim.yaml trainer.world_size=4
```

By default only one GPU is used. Make sure your machine has at least the specified number of GPUs available.

Note: Both the learning rate and training schedule (number of steps) are automatically scaled according to the number of GPUs to maintain consistent convergence behavior across different distributed training setups.

### Upgrading Artifacts

To maintain backward compatibility, older artifacts are automatically upgraded in memory whenever they are loaded by the application. This ensures that you can always use older `.usdz` files with newer versions of the codebase without manual intervention.

However, this automatic, in-memory upgrade can be slow if the same artifact is loaded repeatedly. For a more efficient workflow, a script is provided to perform a manual, persistent upgrade. This allows you to upgrade an artifact once and save the new version, which can then be reused without requiring further upgrades.

For detailed instructions on the upgrade process, please refer to the [artifact upgrade documentation](./nre/utils/upgrade/cli/README.md). For export commands, see the [export documentation](./nre/utils/io/export/README.md).

To run the manual upgrade script, you can use Bazel:

```bash
bazel run //:run -- upgrade-artifact --input <input_path.usdz> --output <output_path.usdz>
```

### Config Inspection and Debugging

To inspect and debug configuration files or export parsed configs from artifacts, you can use the `export-parsed-config` command:

```bash
# Export parsed config from YAML file
bazel run //:run -- export-parsed-config --config-name <config.yaml>

# Export parsed config from USDZ artifact
bazel run //:run -- export-parsed-config --input <artifact.usdz>

# Save parsed config to file with upgrade
bazel run //:run -- export-parsed-config --config-name <config.yaml> --upgrade --output parsed.yaml
```

This is particularly useful for comparing configuration structures between different NRE versions and identifying changes needed for upgrade functions.

### Environment Variables

This section documents important environment variables used by the NRE system.

#### NRE_ENV_RUN_ID

**Purpose**: Provides consistent run IDs across distributed training jobs and resource management systems such as SLURM.

**Usage**:

- Automatically set by the cluster toolbox when submitting jobs to ensure all workers in a distributed job share the same run ID
- Used by the cluster toolbox to ensure that the same run ID is used for the submitted job and the job's output directory
- Picked up automatically when `run_id: GENERATE` is specified in logger configuration
- If not set, the default behavior is to generate a random run ID

**Important**: This variable must be unset when running Weights & Biases (wandb) sweep agents to prevent all runs in a sweep from inheriting the same run ID. Each wandb agent should generate unique run IDs for individual sweep runs to avoid conflicts.

#### CUDA_SYNC_DEBUG

Enables PyTorch's CUDA synchronization debug mode to detect implicit GPU synchronizations that can hurt training performance.

**Usage**:

```bash
CUDA_SYNC_DEBUG=1 bazel run //:run -- --config-name=<config.yaml> ...
```

**Supported entry points**: NRE training (`nre/run/main.py`) and NRM training (`nre/nrm/run.py`).

**Reference**: [torch.cuda.set_sync_debug_mode](https://docs.pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html)

# Gaussian Statistics

You can generate a statistical report on the distribution of Gaussians by using the `gaussian-statistics` command:

```bash
# generate gaussians statistics report and heap maps
bazel run //:run -- gaussian-statistics --config-name <config.yaml> [--output-file <report.yaml>]
```
