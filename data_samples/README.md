<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Data samples

This directory will hold a small ncorev4 fixture (≤ 50 MB, 1 sequence) for
the README quickstart. Until that fixture is published, point
`run_inference.py --ncore-path` at your own ncorev4 dataset.

The HuggingFace mock at `instant_nurec/_hf_mock.py:get_sample_data_path()`
already references this directory by name (`ncorev4_sample/`) — when the
corp publishes the placeholder repo `nvidia/instant-nurec-kelvin`, the mock
will resolve the sample data here automatically.
