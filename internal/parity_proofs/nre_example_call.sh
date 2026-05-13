#!/bin/bash -ex

OUT_DIR=$1

# NRE/NRM's Kelvin - no merging
cd /storage/projects/nre
git checkout a54a6af0a177beabd01fe37e398c45be165a270f
bazel run //nre/nrm:run -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_base_path=/storage/data/nurec/ncorev4/ \
    dataset.predict.ncore_json_list_path=/storage/data/nurec/ncorev4/debug.lst \
    dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
    out_dir="${OUT_DIR}/no_merge" \
    predict.primitive_merge.enabled=false \
    predict.render_video.enabled=false

# NRE/NRM's Kelvin - frustum-ownership
bazel run //nre/nrm:run -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_base_path=/storage/data/nurec/ncorev4/ \
    dataset.predict.ncore_json_list_path=/storage/data/nurec/ncorev4/debug.lst \
    dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
    out_dir="${OUT_DIR}/merge" \
    predict.render_video.enabled=false

git checkout -
cd -
