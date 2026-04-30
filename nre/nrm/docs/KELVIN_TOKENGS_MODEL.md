# Kelvin TokenGS Models

Kelvin TokenGS model is a model development that is in progress. It is mainly intended to reduce the number of Gaussians output.

## Local debugging

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_tokengs.yaml
```

## Loading a legacy TokenGS checkpoint

Kelvin can load and fine-tune checkpoints from the legacy TokenGS model. The config `configs/nrm/apps/local_debug/local_tokengs_pretrained.yaml` is already set up for legacy compatibility.

**Checkpoint location (rclone):**

- Remote path: `ncore:/scratch-jiaweir/checkpoints/tokengs/large_ps8_concat_emb.safetensors`
- List: `rclone ls ncore:/scratch-jiaweir/checkpoints/tokengs/large_ps8_concat_emb.safetensors`
- Copy locally or mount so the run can read the file (e.g. at `/nrm_debug/checkpoints/large_ps8_concat_emb.safetensors`).

**Training with the legacy checkpoint:**

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_tokengs_pretrained.yaml \
  +model.init_weights_path=/nrm_debug/checkpoints/large_ps8_concat_emb.safetensors \
  system.train_batch_size=1 system.optimizer.args.lr=1e-5
```

**Validation only:** add `mode=val` and `system.val_batch_size=1`. When using init weights without a checkpoint for val/test/predict, also add `+call_train_from_scratch_hook_for_validation=true` so the init-weights hook runs (it is off by default for eval to avoid train-only hook logic):

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_tokengs_pretrained.yaml \
  +model.init_weights_path=/nrm_debug/checkpoints/large_ps8_concat_emb.safetensors \
  +call_train_from_scratch_hook_for_validation=true mode=val system.val_batch_size=1
```
