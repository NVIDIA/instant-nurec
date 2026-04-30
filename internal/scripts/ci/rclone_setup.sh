#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script is used in sqa.gitlab-ci.yml to set up rclone config for accessing Swift storage to download previous release artifacts for testing full forward rendering.
set -e

if [ -z "$PDX_TEAM_NCORE_KEY" ]; then
  echo "Error: PDX_TEAM_NCORE_KEY is not set, unable to configure rclone."
  exit 1
fi

# Create rclone config directory if it doesn't exist
mkdir -p ~/.config/rclone

# Start with an empty config file
> ~/.config/rclone/rclone.conf

cat >> ~/.config/rclone/rclone.conf << EOF
[pdx-team-ncore]
type = swift
env_auth = false
user = team-ncore
key = ${PDX_TEAM_NCORE_KEY}
auth = https://pdx.s8k.io/auth/v1.0
auth_version = 1

EOF

# Set appropriate permissions for the config file
chmod 600 ~/.config/rclone/rclone.conf
