#!/bin/bash

VERSION=$1

# MODE
# 0 : train,
# 1 : nrend validation fullres
# 2 : nrend validation half res
# 3 : N/A
# 4 : nrend validation quarter res
# 5 : N/A
# 6 : nre validation fullres
# 7 : nre validation halfres
# 8 : N/A
# 9 : nre validation quarter res
MODE=$2

BENCHMARK_DATASETS=(
  clip_001/001.json
  clip_002/002.json
  clip_005/005.json
  clip_008/008.json
  clip_010/010.json
)

# LOCAL BENCHMARK DATASET DIRECTORY
# should contains the file listed in BENCHMARK_DATASETS
DATASET_DIR=$3

# LOCAL OUTPUT DIRECTORY
# will contains the run folders
OUT_DIR="$4/${VERSION}"
if [ ! -d "${OUT_DIR}" ]; then
  mkdir -p ${OUT_DIR}
fi

# CONFIGURATION FILES PATH RELATIVE TO ./configs/tests
NREND_CONFIGS=(
  nrend_test_nerf_default.yaml
  nrend_test_nerf_glo_sky.yaml
  nrend_test_dnsg_default.yaml
)

# COMMON TEST OPTIONS
NREND_OPTS="logger=dummy system.test.nrend.log_level=3"

# training phase
if [ ${MODE} = 0 ]; then
  for i in "${!NREND_CONFIGS[@]}"; do
    for j in "${!BENCHMARK_DATASETS[@]}"; do
      RUN_ID="tr_config${i}_data${j}"
      if [ ! -d "${OUT_DIR}/${RUN_ID}" ]; then
        CMD="bazel run //:run --  --config-name=configs/tests/${NREND_CONFIGS[$i]} out_dir=${OUT_DIR} mode=train dataset.path=${DATASET_DIR}/${BENCHMARK_DATASETS[$j]} ${NREND_OPTS} +run_id=${RUN_ID}"
        echo ${CMD}
        ${CMD}
      else
        echo "Cannot create run ${RUN_ID}, ${OUT_DIR}/${RUN_ID} exists!"
      fi
    done
  done
# validation phase : done for every folder in OUT_DIR
else
  DS_FACTOR=$(($MODE % 5))
  if (($MODE > 4)); then
    USE_NREND="false"
  else
    USE_NREND="true"
  fi
  LOG_FILE="nrend_log_${MODE}.csv"
  NREND_VAL_OPTS="${NREND_OPTS} dataset.n_val_image_subsample=${DS_FACTOR}  system.save_logger=false logger=wandb logger.project=NREND-TESTBENCH system.test.nrend.enabled=${USE_NREND} dataset.val_camera_frame_step=7"
  for i in "${!NREND_CONFIGS[@]}"; do
    for j in "${!BENCHMARK_DATASETS[@]}"; do
      TRAIN_RUN_ID="tr_config${i}_data${j}"
      VAL_RUN_ID="val_config${i}_mode${MODE}_data${j}"
      WDB_GROUP_ID="v${VERSION}_conf${i}_mode${MODE}"
      WDB_RUN_ID="${WDB_GROUP_ID}_data${j}"
      if [ -d "${OUT_DIR}/${TRAIN_RUN_ID}" ]; then
        if [ ! -d "${OUT_DIR}/${VAL_RUN_ID}" ]; then
          CMD="bazel run //:run -- --config-name=${OUT_DIR}/${TRAIN_RUN_ID}/config/parsed.yaml resume=last mode=val logger.offline=false logger.group=${WDB_GROUP_ID} logger.run_id=${WDB_RUN_ID} run_id=${VAL_RUN_ID} ${NREND_VAL_OPTS}"
          echo ${CMD}
          ${CMD}
        else
          echo "Cannot create run ${VAL_RUN_ID}, ${OUT_DIR}/${VAL_RUN_ID} exists!"
        fi
      else
        echo "Cannot create run ${VAL_RUN_ID}, ${OUT_DIR}/${TRAIN_RUN_ID} does not exists!"
      fi
    done
  done
fi
