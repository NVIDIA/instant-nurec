<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary -->

<!--
NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# NDAS Workflow Tests

This directory contains tests for the `run_ndas_workflow.py` workflow generation system.

## Test Structure

```
tests/
├── BUILD.bazel                           # Bazel build targets
├── README.md                             # This file
├── fetch_sample_specs.py                 # Script to fetch sample workflow specs
├── run_ndas_workflow_integration_test.py # Integration tests
└── sample_test_data/                     # Cached workflow specs (generated)
    └── <workflow-name>.yaml              # Named after the workflow (e.g., josliu-car2sim-template.yaml)
```

## Running Tests

### Integration Tests

The integration tests validate that workflow specs are generated correctly for each
config flavor and pass maglev validation. Config files are automatically discovered
from `cluster_configs/ndas_workflows/`, so new configs are tested without code changes
(unless they should be excluded). Tests run in dry-run mode and do not submit actual workflows.

```bash
# Run all integration tests
bazel test //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test

# Run with verbose output
bazel test //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test --test_output=all

# Run a specific config test
bazel test //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test --test_arg="-k" --test_arg="test_config_ndas_workflow_base"
```

### Dumping Specs for Before/After Diffing

You can dump all generated specs to a directory and diff them:

```bash
# Before change
NDAS_WORKFLOW_TEST_OUTPUT_DIR=$PWD/before \
    bazel run //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test

# After change
NDAS_WORKFLOW_TEST_OUTPUT_DIR=$PWD/after \
    bazel run //internal/workflows/cluster_toolbox/ndas_workflows/tests:run_ndas_workflow_integration_test

# Compare the outputs
diff -r before after
```

Note: Use `bazel run` (not `bazel test`) to write files outside the sandbox.

### Requirements

- **Maglev CLI**: Must be authenticated. You can export `MAGLEV_API_KEY` for this.

## Updating Sample Test Data

The `sample_test_data/` directory contains cached workflow specs fetched from Maglev.
These are used as reference data and may need to be updated periodically when source
workflows change.

To refresh the sample specs:

```bash
# Run the fetch script
bazel run //internal/workflows/cluster_toolbox/ndas_workflows/tests:fetch_sample_specs

# Or run directly with Python (from repo root)
python internal/workflows/cluster_toolbox/ndas_workflows/tests/fetch_sample_specs.py
```

The script will:

1. Clear any existing files in `sample_test_data/`
2. Scan all config files in `cluster_configs/ndas_workflows/`
3. Extract unique `*source_wf` references (source_wf, shim_source_wf, etc.)
4. Fetch each workflow spec using `maglev workflows2 get --spec`
5. Save them to `sample_test_data/`, named after the workflow
   (e.g., `josliu-car2sim-template/latest` -> `josliu-car2sim-template.yaml`)
