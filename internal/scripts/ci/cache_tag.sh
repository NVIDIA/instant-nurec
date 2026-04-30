#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

# Script to handle cache dir (re-)initialization based on a cache tag

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <cache_directory> <expected_tag>"
  exit 1
fi

# Assign arguments to variables
CACHE_DIR="$1"
EXPECTED_TAG="$2"
CACHE_TAG_FILE="$CACHE_DIR/cache_tag.txt" # File containing the cache tag

# Function to delete cache
delete_cache() {
  echo "Deleting outdated cache..."
  sudo rm -rf "$CACHE_DIR"
  echo "Cache deleted."
}

# Function to initialize cache
initialize_cache() {
  echo "Initializing new cache..."
  mkdir -p "$CACHE_DIR"
  echo "$EXPECTED_TAG" > "$CACHE_TAG_FILE"
  echo "Cache initialized with new tag: $EXPECTED_TAG"
}

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
  echo "Cache directory does not exist. Initializing new cache..."
  initialize_cache
  exit 0
fi

# Check if cache tag file exists
if [ ! -f "$CACHE_TAG_FILE" ]; then
  echo "Cache tag file does not exist. Re-initializing cache..."
  delete_cache
  initialize_cache
  exit 0
fi

# Read the current cache tag
CURRENT_TAG=$(cat "$CACHE_TAG_FILE")

# Compare the current tag with the expected tag
if [ "$CURRENT_TAG" != "$EXPECTED_TAG" ]; then
  echo "Cache tag is outdated or incorrect. Re-initializing cache..."
  delete_cache
  initialize_cache
else
  echo "Cache is up to date."
fi
