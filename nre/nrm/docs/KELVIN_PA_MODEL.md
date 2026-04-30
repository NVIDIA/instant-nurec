# Kelvin Pixel-Aligned Models

Kelvin Pixel-aligned model is the next-generation of the Celsius model, which provides much better geometry quality.
It is intended to be published as GA release this June. Please contact [Jiahui Huang](jiahuih@nvidia.com) for more details.

## Running Inference

```bash
# Specify the NGC API key (see README.md for more details)
NGC_API_KEY=<your_ngc_api_key>

# Running the front camera HD model with visualizer enabled
bazel run //nre/nrm:run \
    -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_list_path=/nrm_debug/data/parking-demo/all.lst \
    dataset.predict.ncore_json_base_path=/nrm_debug/data/parking-demo/ \
    viewer.enabled=true

# Running with 3 cameras with visualizer
bazel run //nre/nrm:run \
    -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_varying.yaml \
    +nrm/apps/options=_kelvin_predict \
    nrm/apps/options/sensors@dataset.predict=h81_3cam_504 \
    dataset.predict.ncore_json_list_path=/nrm_debug/data/parking-demo/all.lst \
    dataset.predict.ncore_json_base_path=/nrm_debug/data/parking-demo/ \
    viewer.enabled=true

# (Other potential datasets)
# ------------
# ncore_json_list_path: /nrm_debug/data/lidarfree/all.lst
# ncore_json_base_path: /nrm_debug/data/lidarfree/
# ------------
# ncore_json_list_path: /nrm_debug/data/celsius1_protoplus_1k/motion.lst
# ncore_json_base_path: /nrm_debug/data/celsius1_protoplus_1k/
```

## Local debugging

Context training phase:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_dav3_context.yaml
```

GS training phase:

```bash
bazel run //nre/nrm:run -- --config-name configs/nrm/apps/local_debug/local_dav3_render.yaml
# Note that one may need to provide model.init_weights_paths.full via CLI override
```
