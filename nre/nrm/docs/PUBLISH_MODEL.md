# Publishing Trained Models

Once a model is trained, we can use the following steps to export the checkpoint and publish it to NGC along with the corresponding config.
The steps can be fully automated using an agent. You can ask AI to follow the steps below to publish the model.

## Step 0: Meta information

Make sure you have the following meta information:

- Model architecture name (e.g. `nrm-celsius` or `nrm-kelvin-pa`)
- Model version (e.g. `1.0.0-front`)

## Step 1: Obtaining the checkpoint and config files

Most likely the checkpoint is from an existing training run in the cluster. Please ask the user to provide the `wandb` run id, or the path to the run directory, as well as the cluster name.
Locate the training base directory from the `cluster_toolbox`, and search for the checkpoint file given the run id.
For example, if the run id is `aa-bb-cc_01-01_123456`, then most likely the run output directory is `/path/to/nre_experiments/aa-bb-cc/[DATE_AND_HASH]/results/aa-bb-cc_01-01_123456`.

There might be multiple checkpoints (`ckpt` files) under `OUTPUT_DIR/checkpoints`. Please ask the user to provide the checkpoint file name. Config files is located at `OUTPUT_DIR/config/parsed.yaml`.
Suppose the checkpoint path is `PATH_TO_CHECKPOINT.ckpt` and the config path is `PATH_TO_CONFIG.yaml`.

The original checkpoint contains non-useful information about the training hyperparameters, optimizer state etc, which is not needed for inference. We can strip the checkpoint to keep only the state dict of the model using the following code:

```python
import torch

ckpt_data = torch.load("PATH_TO_CHECKPOINT.ckpt", weights_only=False, map_location="cpu")
new_ckpt_data = {"state_dict": ckpt_data["state_dict"]}
torch.save(new_ckpt_data, "PATH_TO_STRIPPED_CHECKPOINT.ckpt")
```

Store the processed checkpoint and config into a temporary directory, e.g. `TEMP_DIR/MODEL_ARCH_MODEL_VERSION`.

## Step 2: Publishing trained models to NGC

For a new model architecture release, go to https://registry.ngc.nvidia.com/models, log in via `nvstaging/nre`, and use the GUI to create a new model.
This is typically not needed if the model architecture already exists.

For new versions of a model, use the command line interface to deploy updates (in the form of model and config pairs) or remove versions:

```bash
# Login to NGC
ngc config set # use `nvstaging` ORG and `nre` team along with Personal NGC API key credentials

# Upload files for model version
ngc registry model upload-version nvstaging/nre/MODEL_ARCH:MODEL_VERSION --gpu-model 'A100' --source TEMP_DIR/MODEL_ARCH_MODEL_VERSION

# [Optional] Remove model versions
ngc registry model remove nvstaging/nre/MODEL_ARCH:MODEL_VERSION
```

Models deployed this way will have corresponding NGC API URLs of the form

```
https://api.ngc.nvidia.com/v2/org/nvstaging/team/nre/models/MODEL_ARCH/versions/MODEL_VERSION/files/MODEL_ARCH_MODEL_VERSION.yaml
https://api.ngc.nvidia.com/v2/org/nvstaging/team/nre/models/MODEL_ARCH/versions/MODEL_VERSION/files/MODEL_ARCH_MODEL_VERSION.ckpt
```

Confirm that they exist and are accessible. Report the URLs to the user.
