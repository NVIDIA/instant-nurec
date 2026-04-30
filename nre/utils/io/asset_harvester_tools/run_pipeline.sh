#!/usr/bin/env bash
set -euo pipefail

# Unified pipeline: plan -> harvest -> generate overrides -> (optional) train -> (optional) render (with offsets)
# Skips each step if its expected output already exists.

usage() {
  cat <<'USAGE'
Usage: run_pipeline.sh \
  --dataset-json /abs/path/to/sequence.json \
  [--usdz /abs/path/to/last.usdz] \
  [--base-config configs/apps/prod/Hyperion-8.1/car2sim.yaml] \
  [--camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov] \
  [--train-camera-ids camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov] \
  [--radius-m 10.0] [--min-visible-frames 5] \
  [--offsets "0 3 -3"] [--render-camera-id camera_front_wide_120fov] [--height 1080] [--frame-step 1] \
  [--output-dir /abs/path/to/output_root] [--run-id ID] [--skip-training] [--skip-render] \
  [--harvest-output-dir /abs/path/to/reusable_harvest_dir] \
  [--template-yaml nre/utils/io/asset_harvester_tools/template_deformation_network_difix.yaml] \
  [--resume auto|/abs/path/to/checkpoint.ckpt]
USAGE
}

DATASET_JSON=""
USDZ_PATH=""
BASE_CONFIG="configs/apps/prod/Hyperion-8.1/car2sim.yaml"
CAMERA_IDS="camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov"
TRAIN_CAMERA_IDS="camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_front_tele_30fov"
RADIUS_M="10.0"
MIN_VISIBLE_FRAMES="5"
OFFSETS=(0 3 -3)
RENDER_CAMERA_ID="camera_front_wide_120fov"
HEIGHT=1080
FRAME_STEP=1
SKIP_RENDER=0
SKIP_TRAINING=0
OUTPUT_DIR=""
RUN_ID=""
HARVEST_OUTPUT_DIR=""
CLIP_ID=""
TEMPLATE_YAML="nre/utils/io/asset_harvester_tools/template_deformation_network_difix.yaml"
RESUME="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-json) DATASET_JSON="$2"; shift 2;;
    --usdz) USDZ_PATH="$2"; shift 2;;
    --base-config) BASE_CONFIG="$2"; shift 2;;
    --camera-ids) CAMERA_IDS="$2"; shift 2;;
    --train-camera-ids) TRAIN_CAMERA_IDS="$2"; shift 2;;
    --radius-m) RADIUS_M="$2"; shift 2;;
    --min-visible-frames) MIN_VISIBLE_FRAMES="$2"; shift 2;;
    --offsets) IFS=' ' read -r -a OFFSETS <<< "$2"; shift 2;;
    --render-camera-id) RENDER_CAMERA_ID="$2"; shift 2;;
    --height) HEIGHT="$2"; shift 2;;
    --frame-step) FRAME_STEP="$2"; shift 2;;
    --skip-render) SKIP_RENDER=1; shift 1;;
    --skip-training) SKIP_TRAINING=1; shift 1;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --harvest-output-dir) HARVEST_OUTPUT_DIR="$2"; shift 2;;
    --template-yaml) TEMPLATE_YAML="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# If USDZ was provided explicitly, skip training automatically
if [[ -n "${USDZ_PATH}" ]]; then
  SKIP_TRAINING=1
fi

# Require dataset json unless user wants to skip everything
if [[ -z "${DATASET_JSON}" ]]; then
  echo "ERROR: --dataset-json is required" >&2
  usage; exit 1
fi

# If skipping training and rendering, nothing to do
if [[ ${SKIP_TRAINING} -eq 1 && ${SKIP_RENDER} -eq 1 ]]; then
  echo "Nothing to do (skip-training and skip-render are set)."; exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required" >&2
  exit 1
fi

# Generate RUN_ID if not provided
if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: --run-id is required" >&2
  usage; exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: --output-dir is required" >&2
  usage; exit 1
fi

if [[ "${RESUME}" != "auto" && ! -f "${RESUME}" ]]; then
  echo "ERROR: --resume must be set to auto or the path to a checkpoint" >&2
  usage; exit 1
fi

DATASET_DIR="$(dirname "${DATASET_JSON}")"
# Determine base output directory
if [[ -n "${OUTPUT_DIR}" ]]; then
  OUT_BASE="${OUTPUT_DIR}/${RUN_ID}"
fi
mkdir -p "${OUT_BASE}"
# Default harvest output directory to OUT_BASE/harvested_assets if not provided
if [[ -z "${HARVEST_OUTPUT_DIR}" ]]; then
  HARVEST_OUTPUT_DIR="${OUT_BASE}/harvested_assets"
fi
mkdir -p "${HARVEST_OUTPUT_DIR}"

# YAMLs live in the run dir (per-run)
TRAIN_OVERLAY_YAML="${OUT_BASE}/training_assets_overlay.yaml"

echo "[1/5] Plan tracks"
# Attempt to reuse existing plan if present, else run planner
# Construct expected plan filename based on dataset JSON basename
DATASET_BASENAME=$(basename "${DATASET_JSON}" .json)
EXPECTED_PLAN_JSON="${HARVEST_OUTPUT_DIR}/harvest_plan_${DATASET_BASENAME}.json"
PLAN_JSON=""

if [[ -f "${EXPECTED_PLAN_JSON}" ]]; then
  PLAN_JSON="${EXPECTED_PLAN_JSON}"
  echo "  Found existing plan: ${PLAN_JSON} (skipping plan generation)"
else
  bazel run //nre/utils/io/asset_harvester_tools:plan_tracks -- \
    --config-name "${BASE_CONFIG}" \
    --dataset-path "${DATASET_JSON}" \
    --camera-ids "${CAMERA_IDS}" \
    --train-camera-ids "${TRAIN_CAMERA_IDS}" \
    --radius-m "${RADIUS_M}" \
    --min-visible-frames "${MIN_VISIBLE_FRAMES}" \
    --label-source scene:obstacles:autolabels:v2 \
    --output-dir "${HARVEST_OUTPUT_DIR}"
  # Check if the expected plan was generated
  if [[ -f "${EXPECTED_PLAN_JSON}" ]]; then
    PLAN_JSON="${EXPECTED_PLAN_JSON}"
  else
    echo "ERROR: Expected plan JSON not found at ${EXPECTED_PLAN_JSON}" >&2
    exit 1
  fi
fi

# Derive identifiers from plan and construct harvest directory as "<clip_id>/<radius>"
SCENE_ID=$(jq -r '.[0].scene_id' "${PLAN_JSON}")
CLIP_ID=$(jq -r '.[0].clip_id // .[0].scene_id // "unknown_clip"' "${PLAN_JSON}")
TRACK_IDS=$(jq -r '.[0].tracks | map(.track_id) | join(",")' "${PLAN_JSON}")
echo "  Scene: ${SCENE_ID}"
echo "  Clip: ${CLIP_ID}"
echo "  Tracks: ${TRACK_IDS}"

# Build reusable harvest dir under the harvest root
HARVEST_DIR="${HARVEST_OUTPUT_DIR}/${CLIP_ID}/${RADIUS_M}"
mkdir -p "${HARVEST_DIR}"

# If plan is not already under the harvest dir, move it there and point PLAN_JSON to it
PLAN_BASENAME=$(basename "${PLAN_JSON}")
if [[ ! -f "${HARVEST_DIR}/${PLAN_BASENAME}" ]]; then
  mv "${PLAN_JSON}" "${HARVEST_DIR}/" || true
fi
PLAN_JSON="${HARVEST_DIR}/${PLAN_BASENAME}"

# Metadata lives in the clip/radius harvest dir (reusable)
METADATA_YAML="${HARVEST_DIR}/metadata.yaml"

echo "[2/5] Harvest assets"
if [[ -f "${METADATA_YAML}" ]]; then
  echo "  Found ${METADATA_YAML} (skipping harvest)"
else
  SHARD_PATH_PATTERN="${DATASET_JSON%.*}.zarr.itar"
  mkdir -p "${HARVEST_DIR}"
  bazel run //apps/asset_harvester:asset_harvester -- \
    --component-store="${SHARD_PATH_PATTERN}" \
    --output-dir="${HARVEST_DIR}" \
    --track-ids="${TRACK_IDS}" \
    ncore_parser.camera_ids="[\"camera_front_wide_120fov\",\"camera_cross_right_120fov\",\"camera_cross_left_120fov\",\"camera_front_tele_30fov\"]"
fi

echo "[3/5] Generate training/render overrides"
if [[ -f "${TRAIN_OVERLAY_YAML}" ]]; then
  echo "  Found ${TRAIN_OVERLAY_YAML} (skipping generation)"
else
  echo "Deleting ${PLAN_JSON} to force re-generation..."
  rm -f "${PLAN_JSON}"
  cp nre/utils/io/asset_harvester_tools/class_exemplars.example.yaml "${OUT_BASE}/class_exemplars.yaml"
  ASSET_BANK="${OUT_BASE}/asset_bank"
  # Copy example bank from runfiles location
  cp -r nre/utils/io/asset_harvester_tools/asset_bank "${ASSET_BANK}" || true
  bazel run //nre/utils/io/asset_harvester_tools:generate_asset_harvester_training_yaml -- \
    --metadata "${METADATA_YAML}" \
    --harvested-assets "${HARVEST_DIR}" \
    --base-config "${BASE_CONFIG}" \
    --template-yaml "${TEMPLATE_YAML}" \
    --output-yaml "${TRAIN_OVERLAY_YAML}" \
    --plan-json "${PLAN_JSON}" \
    --class-exemplars "${OUT_BASE}/class_exemplars.yaml" \
    --config-name "${BASE_CONFIG}" \
    --dataset-path "${DATASET_JSON}" \
    --camera-ids "${CAMERA_IDS}" \
    --train-camera-ids "${TRAIN_CAMERA_IDS}" \
    --radius-m "${RADIUS_M}" \
    --min-visible-frames "${MIN_VISIBLE_FRAMES}" \
    --label-source scene:obstacles:autolabels:v2 \
    --asset-bank "${ASSET_BANK}"
fi

# Optional training step
if [[ ${SKIP_TRAINING} -eq 0 ]]; then
  echo "[4/5] Train NRE (RUN_ID=${RUN_ID})"
  LIDAR_ID_LIST="[$(echo "$(jq '.shards[0].sensors | keys' "$DATASET_JSON" | grep -oP 'lidar_[a-zA-Z0-9_]+')" | sed 's/[[:space:]]/,/g')]"
  # LOGLEVEL=DEBUG 
  bazel run //:run -- --config-name="${TRAIN_OVERLAY_YAML}" \
    mode=trainval \
    out_dir="$(dirname ${OUT_BASE})" \
    logger=tensorboard \
    logger.run_id="${RUN_ID}" \
    resume="${RESUME}" \
    dataset.path="${DATASET_JSON}" \
    dataset.aux_data=True \
    dataset.lidar_ids=$LIDAR_ID_LIST \
    dataset.camera_ids="[${CAMERA_IDS//,/\,}]" \
    checkpoint.artifact.enabled=true \
    checkpoint.artifact.checkpoint.enabled=true \
    checkpoint.artifact.nrend.enabled=false \
    checkpoint.artifact.rig_trajectories.enabled=true \
    checkpoint.artifact.sequence_tracks.enabled=true \
    checkpoint.every_n_train_steps=5000 \
    checkpoint.save_top_k=-1 \
    system.test.save_extra_signals=true
else
  echo "[4/5] Training skipped"
fi

if [[ ${SKIP_RENDER} -eq 1 ]]; then
  echo "[5/5] Render skipped by flag"
  exit 0
else
  # Use artifact produced by training
  USDZ_PATH="${OUT_BASE}/artifacts/last.usdz"
  if [[ ! -f "${USDZ_PATH}" ]]; then
    echo "ERROR: Expected artifact not found at ${USDZ_PATH}" >&2
    exit 1
  fi
fi

# Render step
echo "[5/5] Render with offsets: ${OFFSETS[*]}"
# Where to place renders
PARENT_DIR="$(dirname "${USDZ_PATH}")"
FILENAME="$(basename "${USDZ_PATH}")"

for offset in "${OFFSETS[@]}"; do
  OUT_DIR="${PARENT_DIR}/render_${FILENAME}_${offset}m_shift"
  FINAL_MP4="${PARENT_DIR}/$(basename "$(dirname "${PARENT_DIR}")")-${FILENAME}_${offset}m_shift.mp4"

  if [[ -f "${FINAL_MP4}" ]]; then
    echo "  Found ${FINAL_MP4} (skipping render for offset ${offset})"
    continue
  fi

  bazel run //:run -- render \
    --artifact-path "${USDZ_PATH}" \
    --image-format jpg \
    --frame-step "${FRAME_STEP}" \
    --camera-id "${RENDER_CAMERA_ID}" \
    --height "${HEIGHT}" \
    --output-dir "${OUT_DIR}" \
    --no-replicate-training-views \
    --rig-translation-offset 0 "${offset}" 0

  bazel run //nre/utils:ffmpeg_wrap -- images-to-mp4 \
    --frames-dir ${OUT_DIR}/${RENDER_CAMERA_ID} \
    --pattern "%06d.jpg" \
    --out ${FINAL_MP4} \
    --framerate 30 --profile high --pix-fmt yuv420p --bitrate 8462k --preset medium
done

bazel run //nre/utils:ffmpeg_wrap -- grid3 \
  --top "${PARENT_DIR}/$(basename "$(dirname "${PARENT_DIR}")")-${FILENAME}_${OFFSETS[0]}m_shift.mp4" \
  --left "${PARENT_DIR}/$(basename "$(dirname "${PARENT_DIR}")")-${FILENAME}_${OFFSETS[1]}m_shift.mp4" \
  --right "${PARENT_DIR}/$(basename "$(dirname "${PARENT_DIR}")")-${FILENAME}_${OFFSETS[2]}m_shift.mp4" \
  --out "${PARENT_DIR}/$(basename "$(dirname "${PARENT_DIR}")")-${FILENAME}_render_concat.mp4"

echo "Done. Plan: ${PLAN_JSON} | Train overlay: ${TRAIN_OVERLAY_YAML} | USDZ: ${USDZ_PATH} (RUN_ID=${RUN_ID})"





