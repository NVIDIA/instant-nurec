# Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `PretrainedModelError: Could not download nvidia/instant-nurec/pth/instant_nurec_pa_front_1.1.0.pth` | No network / proxy blocks `huggingface.co` or `*.cloudfront.net` | Set `HTTPS_PROXY`, or copy `instant_nurec_pa_front_1.1.0.pth` locally and set `INSTANT_NUREC_FULL_PT` to its absolute path. |
| `ModelCheckpointError: ... is a legacy traced-model archive` | `INSTANT_NUREC_FULL_PT` points at the retired artifact | Download `pth/instant_nurec_pa_front_1.1.0.pth` and update the environment variable. |
| `PretrainedModelError: huggingface_hub is required` | Dependency missing | `uv sync --frozen`. |
| `ValueError: --ncore-path ...: not an existing JSON/LST file` | Path doesn't resolve | Check the resolved path; `.lst` entries resolve relative to the LST file's dir, not `$PWD`. |
| `ValueError: --ncore-path must end in .json or .lst` | Wrong suffix | Pass a single `.json` or a `.lst` manifest. |
| `InstantNuRecDataError: Context camera <id> not found in supervision cameras [...]` | `--camera-id` not in the ncorev4 sequence | Inspect the JSON's camera list and pass a matching id. |
| `RuntimeError: Found no NVIDIA driver` / `Torch not compiled with CUDA enabled` | No GPU, driver below the [NuRec hardware minimums](https://docs.nvidia.com/nurec/basics/hardware.html#hardware-setup-and-requirements), or non-CUDA torch wheel | `nvidia-smi` to confirm GPU + driver; `python -c "import torch; print(torch.cuda.is_available())"` must be `True`. No CPU fallback exists. |
| `torch.cuda.OutOfMemoryError` during `Predicting in chunks` | `--max-chunks` × per-chunk working set exceeds VRAM | Lower `--max-chunks`. |
| `WARNING ... Clip spans Xs ... chunk(s) will be silently dropped` | Clip longer than `--max-chunks × 13.5 s` (default 108 s) | Bump `--max-chunks` to the value the warning prints. |

For anything not listed, file a GitHub issue with the full traceback, `nvidia-smi`, `python --version`, and `pip list | grep -iE "torch|huggingface|instant"`.
