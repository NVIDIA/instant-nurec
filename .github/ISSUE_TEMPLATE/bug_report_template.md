---
name: Bug report
about: Create a bug report to help us improve InstantNuRec
title: "[BUG]"
labels: "bug"
assignees: 'daehyoungko'

---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

**Describe the bug**
Provide a clear and concise description of the problem.

**Steps or command to reproduce**
Provide a minimal reproducer, including the full InstantNuRec command and the
input shape (a single NCore V4 sequence `.json` or a directory of sequences).
Do not include proprietary data or credentials.

**Expected behavior**
Describe what you expected to happen.

**Environment (please complete the following information)**
- InstantNuRec version or commit:
- Model checkpoint filename and source:
- Installation method (`./setup.sh`, `uv sync`, or source):
- Python version:
- Operating system:
- GPU model and VRAM:
- NVIDIA driver, CUDA, and PyTorch versions:
- Inference mode (`--merge` or per-chunk):

**Logs and output**
Include the full traceback and relevant command output. Attach `nvidia-smi`
output when the problem may be GPU-, driver-, CUDA-, or memory-related.

**Additional context**
Add any other context that could help reproduce the problem.
