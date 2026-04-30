#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.


# Empty mean use toy dataset
# DATASET_ARG=
DATASET_ARG="--dataset-path /data/h81/clipgt-37272bc7-7904-46e3-8d85-cddff3124ad0/clipgt-37272bc7-7904-46e3-8d85-cddff3124ad0.json"
EVERY_N_STEPS=1000
N_SAMPLES_PER_EPOCH=$(($EVERY_N_STEPS+1))
VERSION=0.1.5.5

###############################
GENERATE=0
LOCAL_TEST=0
TEST=0
UPLOAD=0

for ARG in "$@"; do
  if [ "$ARG" == "--generate" ]; then
    GENERATE=1
  elif [ "$ARG" == "--test" ]; then
    TEST=1
  elif [ "$ARG" == "--local-test" ]; then
    LOCAL_TEST=1
  elif [ "$ARG" == "--upload" ]; then
    UPLOAD=1
  fi
done

if [ "$GENERATE" == 0 ] && [ "$TEST" == 0 ] && [ "$LOCAL_TEST" == 0 ] && [ "$UPLOAD" == 0 ]; then
  echo "Usage: $0 [--generate] [--test] [--local-test] [--upload]"
  echo "     --generate      Generate test data"
  echo "     --local-test    Run tests using extracted test data from generated test data (test_data_prober_generated.tar.gz)"
  echo "     --test          Run tests using the test data configured in bazel build files"
  echo "     --upload        Upload generated test data to GitLab package registry"
  echo "At least one option must be specified."
  exit 1
fi

if [ "$GENERATE" == 1 ]; then
  rm -fR test_data
  mkdir -p test_data

  # Run with bilarf and ppisp
  bazel run //internal/scripts/test_data:generate_test_data -- \
    ${DATASET_ARG} \
    --test-data-dir "$(realpath test_data)" \
    --config-name apps/prod/Hyperion-8.1/car2sim.yaml \
    --n-samples-per-epoch ${N_SAMPLES_PER_EPOCH} \
    --every-n-steps ${EVERY_N_STEPS} \
    --additional-args "dataset.duration_sec=1" "model/post_processing@model.post_processing.b=ppisp" "model/post_processing@model.post_processing.c=bilateral_grid_per_camera" \
                      "model.post_processing.c.width=4" "model.post_processing.c.height=4" "model.post_processing.c.depth=2"
  if [ $? -ne 0 ]; then
    echo "Error: Prober generation failed"
    exit 1
  fi

  # Run with gsplat renderer to generate viewmat test data
  bazel run //internal/scripts/test_data:generate_test_data -- \
    ${DATASET_ARG} \
    --test-data-dir "$(realpath test_data)" \
    --config-name tests/gsplat/3dgut_gsplat.yaml \
    --n-samples-per-epoch ${N_SAMPLES_PER_EPOCH} \
    --every-n-steps ${EVERY_N_STEPS} \
    --additional-args "dataset.duration_sec=1"
  if [ $? -ne 0 ]; then
    echo "Error: GSplat prober generation failed"
    exit 1
  fi
fi

function run_tests {
  # Here are the tests that rely on the prober datasets
  bazel run //nre/models/post_processings/ppisp:ppisp_slang_test || exit 1
  bazel run //nre/models:ppisp_slang_test || exit 1
  bazel run //nre/utils:batch_test -- -k pixels_to_world_rays_shutter_pose || exit 1
  bazel run //nre/utils:batch_test -- -k elements_to_world_rays_shutter_pose || exit 1
  bazel run //nre/models/gaussians:gsplat_viewmat_test || exit 1
}

if [ "$LOCAL_TEST" == 1 ]; then
  tar xvzf test_data_prober_generated.tar.gz test_data
  NRE_PROBER_DIR="$(realpath test_data)" run_tests || exit 1
fi

if [ "$TEST" == 1 ]; then
  unset NRE_PROBER_DIR
  run_tests || exit 1
fi

if [ "$UPLOAD" == 1 ]; then
  GITLAB_TOKEN=$(awk '/machine.*gitlab/ {found=1} found && /login/ {login=$2} found && /password/ {print $2; exit}' ~/.netrc)
  URL="https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/test_data_prober_generated/${VERSION}/test_data_prober_generated.tar.gz"
  SHA256=$(sha256sum ./test_data_prober_generated.tar.gz | awk '{print $1}')
  curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
      --upload-file ./test_data_prober_generated.tar.gz \
      $URL && \
  echo "\nUploaded to $URL, don't forget to update MODULE.bazel:"
  echo "http_archive(
    name = \"test_data_prober_generated\",
    sha256 = \"${SHA256}\",
    urls = [\"https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/test_data_prober_generated/${VERSION}/test_data_prober_generated.tar.gz\"],
)"
fi