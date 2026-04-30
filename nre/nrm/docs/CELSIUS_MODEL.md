# Celsius Models

Celsius models (named after the NVIDIA Celsius Graphics Card Architecture) are a set of models that inherit the design from STORM and BTimer.
They are Vision Transformers that directly take multi-view images and the corresponding plucker rays as input, and output the 3DGS attributes that are pixel-aligned and ready for rendering at arbitrary timestamps and viewpoints.
For more information of the model, please refer to [This document](https://nvidia-my.sharepoint.com/:p:/p/jiahuih/Ed1tuc12Y-ZHrM01kUPH_3kBPYt2A_T_CwwmgxzvG_D8GQ?e=oXv8U1).

Note that this document might not be 100% up-to-date. Please adjust command line arguments accordingly or reach out to any document authors if you have any questions.

## Launch training jobs

### Running large-scale training on ORD

We mainly use ORD to run large-scale training jobs. Example command is provided below:

```bash
# YOUR_PPP_NAME is usually: nvr_torontoai_3dscenerecon
# YOUR_JOB_NAME will be the name of the folder created on lustre. Multiple runs can share the same job name
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- --cluster-name ord --config-name ord.yaml submit-job user=YOUR_USERNAME team=YOUR_PPP_NAME partition=PARTIION_NAME keep_workspace=true num_nodes=2 num_gpus=8 --job-name YOUR_JOB_NAME --command 'FULL_COMMAND'
```

For local debugging, you should use the original `//nre/nrm:run` target directly.

### Concrete commands to reproduce training

In all cases, you should append `logger.run_id=YOUR_RUN_ID` to your command, where `YOUR_RUN_ID` is a unique identifier for your run (as shown in wandb).

- `Celsius_2.0.0-1f4c`:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200_1f4c.yaml
```

- `Celsius-p16b16_2.0.0-6f3c`:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_6f3c.yaml
# Finetune dynamic model with the above checkpoint (below is an example pretrained checkpoint).
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_6f3c.yaml \
  nrm/apps/celsius_base=_dynamic_full nrm/dataset/trisplit=_mixed_motion_data \
  model.init_weights_path=/lustre/fsw/portfolios/nvr/users/jiahuih/workspace/nre/doc-ckpt/celsius200p16b16_6f3c_static/e_low.ckpt
```

- `Celsius_2.0.0-6f1c`:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200_6f1c.yaml
# Finetune dynamic model with the above checkpoint (below is an example pretrained checkpoint).
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200_6f1c.yaml \
  nrm/apps/celsius_base=_dynamic_full nrm/dataset/trisplit=_mixed_motion_data \
  model.init_weights_path=/lustre/fsw/portfolios/nvr/users/jiahuih/workspace/nre/doc-ckpt/celsius200_6f1c_static/e6.ckpt
```

- `Celsius-p16b16_2.0.0-prod6f6c`:

This is the model that we want to move forward with the new production 6-camera setting.

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_prod6f6c.yaml
# Finetune dynamic model from the above checkpoints
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_prod6f6c.yaml \
  nrm/apps/celsius_base=_dynamic_full \
  model.init_weights_path=s3@pdx-team-ncore://nrm-training-data/init-ckpt/celsius200p16b16_prod6f6c_static.ckpt
```

- `Celsius-p16b16_2.0.0-init5c`:

This is a low resolution but longer context model that is used for NuRec initialization. It takes 5 cameras as input.

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_prod6f6c.yaml \
  nrm/dataset/trisplit=_ncore_5cam_init_pdx
```

### Local debugging

For different developers to run on different machines, it is recommended to create a local folder `/nrm_debug` under the root directory of your machine so consistent paths can be used in the configs. If you do not have administrative rights to create this folder, you will have to adapt your path accordingly:

```bash
sudo mkdir /nrm_debug
sudo chown $USER /nrm_debug
# (Optional) conviniently add symlink to your home directory
ln -s /nrm_debug ~/nrm_debug
# (Optional) download gslrm pretrained ckpt for initialization
scp cs-oci-ord-dc-01.nvidia.com:/lustre/fsw/portfolios/nvr/users/jiahuih/workspace/nre/gslrm.ckpt /nrm_debug/
# (Optional) download mamba pretrained ckpt for initialization
scp cs-oci-ord-dc-01.nvidia.com:/lustre/fsw/portfolios/nvr/users/jiahuih/workspace/nre/model_7m1t.ckpt /nrm_debug/
```

#### Debug run using the `ncore` dataset

- First grab some example data from PBSS onto your local machine:

```bash
all_clipids=()
for SEQ_NAME in clipgt-5c1be3df-012f-4544-ab5f-833bb0314743 clipgt-6d426cbb-adfd-474e-bc9a-9aa5638ff61c; do
    mkdir -p /nrm_debug/data/celsius2_l3_55k_720p/
    aws --profile pdx-team-ncore s3 sync s3://nrm-training-data/celsius_6cam_720p_all_in_one/celsius2_gen3c_5k/${SEQ_NAME} /nrm_debug/data/celsius2_l3_55k_720p/${SEQ_NAME}
    all_clipids+=("${SEQ_NAME}/${SEQ_NAME}.json")
done
printf "%s\n" "${all_clipids[@]}" > /nrm_debug/data/celsius2_l3_55k_720p/all.lst
```

- Then launch the training job with:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_ncore.yaml
# Training output will be saved to /nrm_debug/out
```

#### Debug run using the `dataverse` dataset

- Grab toy data from the `dataverse` project:

```bash
# Download the test assets as specified in `WORKSPACE` in the root of this repo. At the time of writing, this is achieved by
wget -O /tmp/dataverse_test_assets.tar.gz https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/dataverse_test_assets/0.0.1/dataverse_test_assets.tar.gz
mkdir -p /nrm_debug/data/dataverse
tar -xvf /tmp/dataverse_test_assets.tar.gz -C /nrm_debug/data/dataverse
```

- Then launch the training job with:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_dataverse.yaml
# Training output will be saved to /nrm_debug/out
```

## Validate, test and predict trained models

**Method 1: via Checkpoint** (for developers or those needing to modify the config/weights)

```bash
bazel run //nre/nrm:run -- --config-name FULL_PATH_TO_CONFIG_YAML resume=FULL_PATH_TO_CHECKPOINT mode=MODE
```

**Method 2: via NGC Registry** (for users who only need to use the pretrained models)

```bash
bazel run //nre/nrm:run -- --config-name FULL_PATH_TO_PRETRAINED_MODEL_CONFIG_YAML mode=MODE
```

For accessing NGC registry, you need to acquire a NGC personal access token by first logging into NGC using the `nvstaging/nre` organization before navigating to [this page](https://org.ngc.nvidia.com/setup/api-keys). You need to set the token either as the `NGC_API_KEY` environment variable or updating `~/.netrc` accordingly with the following content:

```
machine api.ngc.nvidia.com
  login oauth2
  password <NGC_API_KEY>
```

Here, `MODE` can be chosen from one the following three:

- `val`: validate the model using the datasets specified in `dataset.val`. Visualizer will NOT block if new data is available. Trajectory videos will be logged according to `log_media_every_n_steps_val` settings.
- `test`: test the model using the datasets specified in `dataset.test`. Visualizer will block until user clicks the "Next scene" button. Trajectory videos will be logged unconditionally for all samples.
- `predict`: predict the model using the datasets specified in `dataset.predict`. Visualizer will NOT block if new data is available. One can typically use `+nrm/dataset/concrete@dataset.predict=ncore_local_low_res` to specify the dataset to use (this e.g. could be a dataset based on HTTP requests). Trajectory videos will not be logged.

In cases such as primitive merging during prediction or testing locally with single GPU, there might be so many additional overrides to append to the command line, we hence create a new hydra config groups under `configs/nrm/apps/options`, and you can simply add `+nrm/apps/options=xxx` in your command line to activate those.

The following overriding priority applies:
**Overriding Priority**: `override_config` in config file / command line overrides > other command line arguments > `--config-name` file.

### Pretrained checkpoints

We provide the following pretrained checkpoints hosted at NGC (for those who are interested in publishing their own models, please refer to [this](PUBLISH_MODEL.md)):

- `Celsius_1.0.0`:
  - Static model:
    - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c100_static.yaml`
    - via checkpoints: download with `ngc registry model download-version nvstaging/nre/nrm-celsius:static_1.0`.
  - Dynamic model:
    - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c100_dynamic.yaml`
    - via checkpoints: download with `ngc registry model download-version nvstaging/nre/nrm-celsius:dynamic_1.0`.
- `Celsius_2.0.0-6f1c`:
  - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c200_6f1c.yaml`
  - via checkpoints: download with `ngc registry model download-version nvstaging/nre/nrm-celsius:c200_6f1c`.
- `Celsius-p16b16_2.0.0-6f3c`:
  - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c200p16b16_6f3c.yaml`
  - via checkpoints: download with `ngc registry model download-version nvstaging/nre/nrm-celsius:c200p16b16_6f3c`.
- `Celsius_2.0.0-1f4c`
  - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c200_1f4c.yaml`
  - via checkpoints: download with `ngc registry model download-version nvstaging/nre/nrm-celsius:c200_1f4c`.
- `Celsius-p16b16_2.0.0-prod6f6c`:
  - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c200p16b16_prod6f6c.yaml`
- `Celsius-p16b16_2.0.0-init5c`:
  - via NRE model-registry: use `configs/nrm/apps/pretrained/ngc_c200p16b16_init5c.yaml`

Additionally, a model tracker is available at [here](https://docs.google.com/spreadsheets/d/1gh1tUwwu9HqIviaGKeTn0C3mtDZ_TQbGKZ_7GOi8bWw/edit?usp=sharing).

Example invocations:

```bash
# Run prediction with the April Celsius_1.0.0 Dynamic model and a customized dataset
bazel run //nre/nrm:run \
    -- \
    --config-name=configs/nrm/apps/pretrained/ngc_c100_dynamic.yaml \
    +nrm/dataset/concrete@dataset.predict=ncore_local_low_res \
    ++mode=predict \
    ++viewer.enabled=true

# Run prediction with the Celsius_2.0.0-6f1c model loaded from NGC registry with long primitive merging (with visualizer enabled)
bazel run //nre/nrm:run \
    -- \
    --config-name=configs/nrm/apps/pretrained/ngc_c200_6f1c.yaml \
    +nrm/apps/options=_long_primitive_merge \
    nrm/apps/options/sensors@dataset.predict=h81_front_960 \
    viewer.enabled=true

# Run prediction with the init5c model from NGC single chunk with visualizer enabled
bazel run //nre/nrm:run \
    -- \
    --config-name=configs/nrm/apps/pretrained/ngc_c200p16b16_init5c.yaml \
    +nrm/apps/options=_long_primitive_merge \
    nrm/apps/options/sensors@dataset.predict=h81_5cam_416 \
    dataset.predict.frame_batch_samplers.sequential.n_frames_per_sample=18 \
    shared_config.n_chunks=1 \
    viewer.enabled=true

# For other celsius2 models, you have to append some additional overrides to the command line:
#  - for Celsius-p16b16_2.0.0-6f3c:
#          nrm/apps/options/sensors@dataset.predict=h81_3cam_960
#  - for Celsius_2.0.0-1f4c:
#          dataset.predict.context_camera_ids=[camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov,camera_front_tele_30fov]
#          dataset.predict.frame_batch_samplers.sequential.n_frames_per_sample=1
#          shared_config.n_chunks=1
#  - for Celsius-p16b16_2.0.0-prod6f6c:
#          nrm/apps/options/sensors@dataset.predict=h81_6cam_800

# Run local test (together with **Difix** model) on a fresh run grabbed from ORD.
# Optionally mount data via `sshfs ORD:/lustre /lustre`.
bazel run //nre/nrm:run \
    -- \
    --config-name=FULL_PATH_TO_PARSED_YAML \
    resume=FULL_PATH_TO_CHECKPOINT \
    +difix@model.difix=cosmos_difix +nrm/apps/options=_single_gpu_test
```

**SIL Wheel Integration**: [SIL Wheel](http://sil-wheel.nvidia.com:8000/) maintains a websocket server that runs typically on port 7000, and it will broadcast the selected CLIP ID to all the websocket clients. The `ncore_websocket` dataset act as a compatible client. To use it, set the following additional flags:

```bash
BAZEL_COMMAND \
  nrm/dataset@dataset.predict=ncore_websocket \
  dataset.predict.ws_server_addr=sil-wheel.nvidia.com \
  dataset.predict.ncore_json_list_path=S3_LUSTRE_OR_LOCAL_PATH \
  dataset.predict.ncore_json_base_path=S3_LUSTRE_OR_LOCAL_PATH
```

## Additional notes

- `mamba_ssm` and `causal_conv1d` are internal dependencies that are made optional since they significantly slow down the bazel start-up time. If you want to use mamba in your target, you can add `--config=mamba` to the command line.
