#!/bin/bash -ex

# Generate the three canonical NRE@a54a6af reference outputs for a single
# ncorev4 clip — one per instant-nurec CLI mode:
#   1. no_merge                — per-chunk PLYs (instant: default).
#   2. merge_voxel_<size_1>    — frustum-ownership merge + KL-optimal voxelization
#                                (instant: --merge=frustum-ownership --voxel-size <size_1>).
#   3. merge_voxel_<size_2>    — same, with a second voxel size.
#
# Usage:
#   nre_example_call.sh OUT_DIR [VOXEL_SIZE_1 [VOXEL_SIZE_2]]
#
# Defaults: VOXEL_SIZE_1=0.1, VOXEL_SIZE_2=0.2. Both voxelization runs pin
# `voxel_fusion_mode=kl_optimal` to match instant-nurec's only fusion
# strategy; without the override, nre defaults to "average" and
# rotation/scale attribute parity becomes meaningless. instant-nurec's
# --merge bundles voxelization unconditionally, so there is no parity
# counterpart for nre's plain merge (no-voxel) case.

OUT_DIR=$1
VOXEL_SIZE_1=${2:-0.1}
VOXEL_SIZE_2=${3:-0.2}

NRE_REPO=/storage/projects/nre
NRE_COMMIT=a54a6af0a177beabd01fe37e398c45be165a270f

cd "$NRE_REPO"
git checkout "$NRE_COMMIT"

COMMON_OVERRIDES=(
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml
    +nrm/apps/options=_kelvin_predict
    dataset.predict.ncore_json_base_path=/storage/data/nurec/ncorev4/
    dataset.predict.ncore_json_list_path=/storage/data/nurec/ncorev4/debug.lst
    dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov
    predict.render_video.enabled=false
)

# 1. no_merge
bazel run //nre/nrm:run -- \
    "${COMMON_OVERRIDES[@]}" \
    out_dir="${OUT_DIR}/no_merge" \
    predict.primitive_merge.enabled=false

# 2. merge + kl-optimal voxelization @ VOXEL_SIZE_1
bazel run //nre/nrm:run -- \
    "${COMMON_OVERRIDES[@]}" \
    out_dir="${OUT_DIR}/merge_voxel_${VOXEL_SIZE_1}" \
    predict.primitive_merge.enable_voxelization=true \
    predict.primitive_merge.voxel_size="${VOXEL_SIZE_1}" \
    predict.primitive_merge.voxel_fusion_mode=kl_optimal

# 3. merge + kl-optimal voxelization @ VOXEL_SIZE_2
bazel run //nre/nrm:run -- \
    "${COMMON_OVERRIDES[@]}" \
    out_dir="${OUT_DIR}/merge_voxel_${VOXEL_SIZE_2}" \
    predict.primitive_merge.enable_voxelization=true \
    predict.primitive_merge.voxel_size="${VOXEL_SIZE_2}" \
    predict.primitive_merge.voxel_fusion_mode=kl_optimal

git checkout -
cd -
