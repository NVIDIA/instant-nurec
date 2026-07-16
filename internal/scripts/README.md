<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# internal/scripts

One-off maintenance scripts. They are not part of the public API and are not
shipped in the wheel.

## dataset_load_scan.py

Loads candidate NCore sequences with the canonical public input dimensions and
frame count, then records whether preprocessing succeeds. Use it to audit a
large dataset before an inference sweep.
