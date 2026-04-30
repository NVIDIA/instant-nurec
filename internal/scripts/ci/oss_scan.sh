#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script is used in the security.gitlab-ci.yml file to scan container images for OSS vulnerabilities.

set -e

# Create main artifacts directory
ARTIFACTS_DIR="oss_scan_artifacts"
mkdir -p "$ARTIFACTS_DIR"
# The general exit code that decides if the job fails 
DEFAULT_EXIT_CODE=0
# If email notifications should be sent out to notify developers about high/critical vulnerabilities. 
# Set to 1 if high/critical vulnerabilities found.
SHOULD_NOTIFY=0
POLICY_RESULT_FILE="policy_evaluation.json"

for IMAGE_SUFFIX in run tools obfuscated_run obfuscated_tools ; do
  export CONTAINER_IMAGE="${DEPLOY_REGISTRY}/${DEPLOY_BASE_REPOSITORY}_${IMAGE_SUFFIX}:${VERSION_STRING}"

  # Create subdirectory for this image suffix
  SCAN_DIR="$ARTIFACTS_DIR/${IMAGE_SUFFIX}"
  mkdir -p "$SCAN_DIR"
  echo "Scanning: $CONTAINER_IMAGE"
  echo "Output directory: $SCAN_DIR"

  EXIT_CODE=0
  # Change to scan directory before running pulse-cli so files are created there
  cd "$SCAN_DIR"
  pulse-cli -n $NSPECT_ID --ssa $SSA_TOKEN scan-image -i $CONTAINER_IMAGE --platform=$PULSE_CONTAINER_SCANNER_PLATFORM --sbom $SBOM_OUTPUT_FORMAT -p $CONTAINER_SCAN_POLICY_REF --govready=$GOVREADY --ignore-base-vulns $IGNORE_BASE_VULNS -o || EXIT_CODE=$?

  # Check for high or critical severity vulnerabilities and fail if found
  if grep -q '"final_action": "stop"' $POLICY_RESULT_FILE; then
    echo "ERROR: High or Critical severity vulnerabilities found in $CONTAINER_IMAGE, policy evaluation failed"
    EXIT_CODE=210
  fi

  # Return to repo root
  cd - > /dev/null
  if [ $EXIT_CODE -ne 0 ]; then
    DEFAULT_EXIT_CODE=1
    if [ $EXIT_CODE -eq 210 ]; then 
      SHOULD_NOTIFY=1
    fi
    echo "scanning failed for $CONTAINER_IMAGE with exit code $EXIT_CODE"
  else 
    echo "scanning successful for $CONTAINER_IMAGE"
  fi

done

# Send email notifications to notify developers about high/critical vulnerabilities if found.
if [ $SHOULD_NOTIFY -eq 1 ]; then
  echo "Sending email notifications to notify developers about high/critical vulnerabilities."
  export SUBJECT="[Important]High/Critical vulnerabilities found in the pipeline"
  export BODY="High/Critical vulnerabilities found in the pipeline. Please check the pipeline ${CI_PIPELINE_URL} for more details."
  export TO_ADDR="nre-images-scanning-notify@nvidia.com"
  "$CI_PROJECT_DIR/internal/scripts/ci/smtp_send_email.sh"
fi

exit $DEFAULT_EXIT_CODE