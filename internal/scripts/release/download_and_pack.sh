#!/usr/bin/env bash
set -euo pipefail

# Tool for downloading all files from a list, e.g. copying the column
# "Location where component was downloaded from" from the google spreadsheet
# https://docs.google.com/spreadsheets/d/1UYCCAlf4bVlS_b6__b6A-6KjB58FbMNOtWCXn5hxUfs/edit?usp=sharing
# and then pasting it to a text file to used it as the first argument of this tool,
# and packing those files in a tar.gz file, e.g. 25.08.tar.gz,
# to be included in an OSRB bug request.

# Usage: ./download_and_pack.sh [list_file] [output_archive]
# Defaults:
LIST_FILE="${1:-list.txt}"
ARCHIVE="${2:-downloads.tar.gz}"
DEST_DIR="downloads"

mkdir -p "${DEST_DIR}"

echo "Reading URLs from ${LIST_FILE} (skipping header)..."
# Skip duplicates
declare -A seen
while IFS= read -r url; do
  [ -z "${url}" ] && continue
  if [[ -n "${seen[$url]:-}" ]]; then
    echo "Skipping duplicate ${url}"
    continue
  fi
  seen[$url]=1
  echo "Downloading ${url}..."
  wget -q --show-progress -P "${DEST_DIR}" "${url}"
done < <(tail -n +2 "${LIST_FILE}")

echo "Creating archive ${ARCHIVE}..."
tar -czvf "${ARCHIVE}" -C "${DEST_DIR}" .

echo "Done: Created ${ARCHIVE} containing all downloads."
