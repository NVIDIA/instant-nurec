#!/usr/bin/env bash
set -euo pipefail

# SQA script for Mask Annotator end-to-end testing

# 1) Non-GUI smoke tests: CLI interface
echo "[SQA] Running CLI smoke tests"
# Test help output
bazel run //apps/avmask_annotator:mask_annotator_cpu -- --help | grep -q "Options" && echo "--help OK"
# Test version output
bazel run //apps/avmask_annotator:mask_annotator_cpu -- --version | grep -q "Mask Annotator, version" && echo "--version OK"

# 2) GUI end-to-end tests using pytest-qt offscreen mode
echo "[SQA] Running GUI end-to-end tests"
bazel test //apps/avmask_annotator:mask_annotator_gui_test

echo "[SQA] All tests passed!" 