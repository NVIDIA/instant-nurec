#!/bin/bash
# Run the NRE-copy predict pipeline (Phase 0 step 2.1 / Phase 1+2 baseline)
# from this repository, instead of cd-ing into /storage/projects/nre. Mirrors
# nre_example_call.sh but runs against the local copy of the source so that
# subsequent Phase 1 strips can be verified against the same bazel toolchain.
#
# Usage:
#   scripts/run_nre_predict_local.sh <out_dir>
#
# Produces:
#   <out_dir>/no_merge/...   (predict.primitive_merge.enabled=false)
#   <out_dir>/merge/...      (predict.primitive_merge.enabled=true,
#                             overlap_strategy=frustum_ownership default)
#
# Both write PLY files; render_video is hard-disabled.

set -e

OUT_DIR="${1:?usage: $0 <out_dir>}"
NCORE_BASE="${NCORE_BASE:-/storage/data/nurec/ncorev4/}"
NCORE_LIST="${NCORE_LIST:-/storage/data/nurec/ncorev4/debug.lst}"
LIDAR_ID="${LIDAR_ID:-lidar_top_360fov}"

# Bazel needs writable output_user_root + output_base; default ~/.cache/bazel
# is read-only inside our sandbox. Override only if BAZEL_OUTPUT_ROOT is set.
BAZEL_FLAGS=""
if [ -n "${BAZEL_OUTPUT_ROOT:-}" ]; then
    BAZEL_FLAGS="--output_user_root=${BAZEL_OUTPUT_ROOT}/user --output_base=${BAZEL_OUTPUT_ROOT}/out"
fi

mkdir -p "${OUT_DIR}/no_merge" "${OUT_DIR}/merge"

# no-merge mode
bazel ${BAZEL_FLAGS} run //nre/nrm:run -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_base_path="${NCORE_BASE}" \
    dataset.predict.ncore_json_list_path="${NCORE_LIST}" \
    dataset.predict.cuboid_tracks_params.lidar_id="${LIDAR_ID}" \
    out_dir="${OUT_DIR}/no_merge" \
    predict.primitive_merge.enabled=false \
    predict.render_video.enabled=false

# merge / frustum-ownership mode
bazel ${BAZEL_FLAGS} run //nre/nrm:run -- \
    --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
    +nrm/apps/options=_kelvin_predict \
    dataset.predict.ncore_json_base_path="${NCORE_BASE}" \
    dataset.predict.ncore_json_list_path="${NCORE_LIST}" \
    dataset.predict.cuboid_tracks_params.lidar_id="${LIDAR_ID}" \
    out_dir="${OUT_DIR}/merge" \
    predict.render_video.enabled=false
