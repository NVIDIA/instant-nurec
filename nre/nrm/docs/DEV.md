# Developer Notes

## Verifying Preemption and Mid-Epoch Resuming

In PL, **validation loop** is _within_ the **training loop**. This could be verified by checking when the hooks are called (https://lightning.ai/docs/pytorch/stable/common/hooks.html).
So validation epoch start/end is **completely included within** training epoch start/end.
When preempted, no xxx_epoch_end hook is further executed. When resuming, no xxx_epoch_start hook is executed.

PL maintains `batch_progress` for the loops. It contains 4 states:
Data is ready from the dataloader -> **ready** -> Data has gone through the pre-batch callbacks -> **started** -> Data has gone through the model -> **processed** -> Data has gone through the post-batch callbacks -> **completed**
The `batch_progress` is stored in the checkpoint state dicts.

The loop logics are here: `pytorch_lightning/loops/training_epoch_loop.py` and `pytorch_lightning/loops/evaluation_loop.py`. It detects how to resume from restarted checkpoints by checking the above 4 states.
As one can see, saving preempted checkpoints at both `train_batch_end` and `train_epoch_end` are properly supported in PL2.6. In PL2.4, one need to manually set up the `batch_progress` to the correct value when saving the preempted checkpoint.
However, saving preempted checkpoints during validation is only supported **during the model processing** (i.e. `restarted_mid_evaluation` in PL2.6), and per testing neither PL2.4 nor PL2.6 can handle `batch_idx` correctly in the hooks.

In conclusion, a recommended way to resume from preempted checkpoints during validation is to **completely rerun the full validation loop**.
Additionally, for `torchmetrics`, although it appears as a lightningmodule's submodule, all of its `states` are **non-persistent**. This means after resuming all previous states about a metric will be lost. This indicates you cannot resume the metrics during mid-epoch resuming. We need to be careful when logging epoch-based metrics during training. Validation is okay since we always rerun the full loop.

```bash
bazel run //nre/nrm:run -- --config-name=configs/nrm/apps/experimental/test_data_system logger.run_id=[SPECIFIED_RUN_ID]
# Use Ctrl+C to simulate preemption
# Check whether train-step/val-step data index matches, check whether val-epoch data index sum metrics matches.
```

## Verifying Cluster Multi-Node Training

In order to check the environment setup on cluster, you could run:

```bash
bazel run //internal/workflows/cluster_toolbox:run_cluster_toolbox -- \
    --cluster-name ord --config-name [YOUR_CLUSTER_CONFIG.yaml] submit-job \
    user=YOUR_USERNAME team=YOUR_PPP keep_workspace=true num_nodes=2 num_gpus=8 \
    --job-name nccl-probe --command 'bazel run //nre/nrm:probe_cluster_env'
```

Afterwards, ask Claude to analyze the output logs and check if the environment setup is correct.

## Benchmarking Timing and Memory Usage

You could easily use `nsys` to profile the timing and memory usage of the training loop.

```bash
nsys profile --trace=cuda,nvtx [YOUR_BAZEL_COMMAND] +nrm/apps/options=_profile_training
```

See log for dumped memory snapshots. Use https://heiwang1997.github.io/memory_viz/main.html to visualize them. Use nsys-ui to visualize the timing profile (saved at the current working directory).

## Benchmarking Data Streaming from S3

You could benchmark the data streaming from S3 by using the following command.

```bash
bazel run //nre/nrm/datasets:ncore_benchmark -- --config-name configs/nrm/apps/celsius_training/celsius200p16b16_prod6f6c.yaml --num-batches 32 --num-workers 8 --s3-block-size-mb 8 --s3-cache-type blockcache --log-s3-requests
```
