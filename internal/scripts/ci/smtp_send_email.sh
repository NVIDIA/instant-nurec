#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script is used to send an email using msmtp.
# It is used in the security.gitlab-ci.yml file to send an email when vulnerabilities are found.
# Need to override the SUBJECT, BODY, and TO_ADDR variables in the security.gitlab-ci.yml file.

set -eu

have() { command -v "$1" >/dev/null 2>&1; }

install_msmtp_if_missing() {
  if have msmtp; then return; fi
  if have apk; then apk add --no-cache msmtp ca-certificates; 
  elif have apt-get; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      -o Dpkg::Options::=--force-confdef \
      -o Dpkg::Options::=--force-confold \
      msmtp ca-certificates
  elif have yum; then yum install -y msmtp ca-certificates
  elif have microdnf; then microdnf install -y msmtp ca-certificates
  else
    echo "FATAL: No known package manager to install msmtp." 
    exit 1
  fi
}

if install_msmtp_if_missing; then
  echo "msmtp installed"
else
  echo "msmtp not installed"
  exit 1
fi

# Defaults (can be overridden by env)
SMTP_HOST="hqmail.nvidia.com"
SMTP_PORT="587"
SMTP_USER="${SVCNRE_IMAGE_SCANS_ADDRESS}"
SMTP_PASS="${SVCNRE_IMAGE_SCANS_PASSWORD}"
TO_ADDR="${TO_ADDR}"
SUBJECT="${SUBJECT:-"CI Image Scan Report"}"
BODY="${BODY:-"Image scan finished."}"

# Basic validation
for v in SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS TO_ADDR; do
  if [[ -z "${!v}" ]]; then
    echo "FATAL: $v is required (set via env)." >&2
    exit 2
  fi
done

# Configure msmtp (umask 077 to protect credentials)
umask 077
cat > "${HOME}/.msmtprc" <<EOF
account default
host ${SMTP_HOST}
port ${SMTP_PORT}
auth login
user ${SMTP_USER}
password ${SMTP_PASS}
from ${SMTP_USER}
allow_from_override on
tls on
tls_starttls on
logfile -
tls_trust_file /etc/ssl/certs/ca-certificates.crt
EOF

# Build RFC-5322 message with CRLF line endings
tmpmsg="$(mktemp)"
trap 'rm -f "$tmpmsg"' EXIT

{
  printf 'From: %s\r\n' "$SMTP_USER"
  printf 'To: %s\r\n' "$TO_ADDR"
  printf 'Subject: %s\r\n' "$SUBJECT"
  printf 'Content-Type: text/plain; charset=UTF-8\r\n'
  printf '\r\n'
  printf '%s\r\n' "$BODY"
} > "$tmpmsg"

# Send
echo "Sending email to $TO_ADDR"
msmtp -t < "$tmpmsg"