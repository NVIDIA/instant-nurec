## Build Docker Image with cbuild

`cbuild` is a tool that makes it possible to build container images without having to set up Docker on local workstations or use the ADLR Docker build nodes. The tool leverages the nvpark k8s cluster and the Kaniko build tool to perform the container builds.

- [ADLR UTILS](https://confluence.nvidia.com/display/ADLR/ADLR+Utils)
- [CBUILD DOCS](https://confluence.nvidia.com/display/ADLR/cbuild)
- [CBUILD DEMO](https://gitlab-master.nvidia.com/ADLR/cbuild_demo)
- [CBUILD gitab ci template](https://gitlab-master.nvidia.com/ADLR/cbuild)
- [CBUILD overview presentation](https://docs.google.com/presentation/d/11UCF7XvUgcT54hgSulWnp4ewDagcghL-mtXeq5nNVrk/edit?usp=sharing)

#### Setup

First, set up `ADLR_UTILS` following the instructions [here](https://confluence.nvidia.com/display/ADLR/ADLR+Utils) on OCI login nodes. The setup process will require you to enter your GitLab token for accessing the GitLab Container Registries.
You can leave it empty if you don't use the GitLab Container Registries.

The setup script will then edit your `~/.netrc` and `~/.bashrc` files on the login node. The GitLab token you provided will also be stored as the `GITLAB_TOKEN` environment variable in your `~/.bashrc` file.
You can edit it manually if you want to change the token.

Then run the following commands on the OCI login node to allow `cbuild` to log in to GitLab and NGC. You might need to add the `--force` flag if you are overwriting the existing token. Note that the `<token>` in the commands should be the same as the one you specified in the setup process.

```bash
cbuild login --token=<token>  # login to GitLab
cbuild login --registry=nvcr.io --user \$oauthtoken --token=<token>  # login to NGC
```

Additionally, you might also need to edit the `~/.config/enroot/.credentials` file to add your `nvcr.io` tokens.

#### Build

Build the Docker image using the script `build_dev_bazel_docker.sh`. The script should be run on OCI login nodes where the `cbuild` utility is available. Theoretically, `cbuild` claims to be able to cross-build images for different
clusters, but in practice we have noticed that the build process is not always successful (e.g., incorrectly selecting the platform architecture). So we recommend building the image on the cluster you want to run the job on.

```bash
# On OCI login nodes (force the cbuild-based path)
USE_CBUILD=1 bash ./internal/scripts/cluster_toolbox/build_dev_bazel_docker.sh
```

The script will build the Docker image and then print a command to export that image to an enroot `.sqsh` filesystem image in your shared output directory. An enroot `.sqsh` file accelerates job startup time by avoiding the need to download the image from the registry every time. You can execute the printed command to perform the export. The output filename is `$SQUASH`, derived from `$IMAGE` and `$TAG` (for example, `${IMAGE}-${TAG}.sqsh`), and it lives under `$SHARE_OUTPUT/containers/`.

```bash
srun -A <ACCOUNT> --partition cpu --nodes 1 --pty --pty enroot import \
  --output $SHARE_OUTPUT/containers/$SQUASH \
  'docker://$oauthtoken@nvcr.io#'$IMAGE
```

**NOTE**: `SHARE_OUTPUT` is an environment variable defined by the `ADLR_UTILS` setup script that points to the user directory on `lustre`, typically it is `/lustre/fsw/portfolios/<PROJECT>/users/<USER>`.

**Note**: The `cbuild` utility actually performs the build in a Kubernetes cluster, so it does not have access to the local folder, and we need to avoid using relative paths in the Dockerfile,
including commands like `COPY`, `RUN --mount=...`, etc. Instead, we need to assume the Dockerfile can be built from arbitrary locations via commands like `docker build -f <arbitrary-path>/Dockerfile .`
