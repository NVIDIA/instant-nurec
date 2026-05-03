#!/bin/bash -ex

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OUT_DIR=$1

# instant-nurec - no merging
/storage/projects/instant-nurec/.venv/bin/python run_inference.py \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir "${OUT_DIR}/no_merge" \
    --merge none \
    --log-level INFO

# instant-nurec - frustum-ownership
/storage/projects/instant-nurec/.venv/bin/python run_inference.py \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir "${OUT_DIR}/merge" \
    --merge frustum-ownership \
    --log-level INFO