#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script is used in the security.gitlab-ci.yml file to scan secrets in docker images.

set -e

for IMAGE_SUFFIX in run tools obfuscated_run obfuscated_tools ; do
  export TRUFFLEHOG_IMAGE="${DEPLOY_REGISTRY}/${DEPLOY_BASE_REPOSITORY}_${IMAGE_SUFFIX}:${VERSION_STRING}"
  OUTPUT_FILE_NAME="${IMAGE_SUFFIX}-pulse-secret-scan-results"
  echo $OUTPUT_FILE_NAME
  echo $TRUFFLEHOG_IMAGE
  echo $TRUFFLE_HOG_EXTRA_OPTIONS
  EXIT_CODE=0
  pulse-secret-scanner docker --print-avg-detector-time --user-agent-suffix=pulse-secret-scanner-container-pulse-secret-scan --exclude-detectors=$EXCLUDE_DETECTORS --verifier gitlab=https://gitlab-master.nvidia.com $TRUFFLE_HOG_EXTRA_OPTIONS > ./.pulse-secret-scan/$OUTPUT_FILE_NAME.json || EXIT_CODE=$?
  echo "EXIT_CODE: $EXIT_CODE"
  if [ $EXIT_CODE -eq 183 ]; then
    echo "pulse-secret-scanner scan found secrets in your docker image."
    echo "Delete and rotate verified/live secret as soon as possible."
    echo "pulse-secret-scanner guide: https://confluence.nvidia.com/x/-xMlZQ"
    export SUBJECT="[Important]Verified secrets found in the pipeline"
    export BODY="Verified secrets found in the pipeline. These need to be removed from the image(s). Please check the pipeline ${CI_PIPELINE_URL} for more details."
    export TO_ADDR="nre-images-scanning-notify@nvidia.com"
    echo "Sending email to $TO_ADDR about the vulnerabilities found."
    ./internal/scripts/ci/smtp_send_email.sh
  elif [ $EXIT_CODE -eq 185 ]; then
    echo "All results found by pulse-secret-scanner scan are unverified, please take a look and allowlist if needed"
    echo "pulse-secret-scanner guide: https://confluence.nvidia.com/x/-xMlZQ"
  elif [ $EXIT_CODE -eq 189 ]; then
    echo "All results found by pulse-secret-scanner scan are allowlisted"   
  else
    echo "pulse-secret-scanner scan did not find any secrets"
  fi
done 
exit 0
