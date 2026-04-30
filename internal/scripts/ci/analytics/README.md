<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# GitLab CI Job Analytics

This folder contains two Python scripts for collecting, analyzing, and visualizing GitLab CI job data for the NRS group. These tools help monitor CI pipeline performance, identify bottlenecks, and track trends over time.

Scripts are fully AI-generated and meant as a stop-gap solution until we can integrate with more mature infrastructure. Code quality may not be on par with usual Nvidia standards.

## Overview

The toolkit consists of two main components:

1. **ci_job_collector.py**: Collects CI job data from GitLab via the API
2. **ci_visualization.py**: Generates interactive HTML visualizations from the collected data

## ci_job_collector.py

This script queries the GitLab API to collect comprehensive CI job data from NRS projects and saves it as JSON. It gathers detailed information including:

- Job results (success, failure, canceled)
- Runner information
- Timing metrics (duration, queue time)
- User and project details
- Link to job details on GitLab

The collector incrementally updates the dataset, so you can run it regularly to keep data current.

When the project list includes the NRE project, the job collector also downloads the `serialized_tests` job's artifacts and extracts test runtimes from the Bazel BEP files. This data is also incrementally updated into a JSON file.

The collector also records compressed sizes of the main `:latest` container images for NRE and appends one snapshot per day into a JSON file.

### Usage

```bash
# Basic usage with .netrc authentication. Collects jobs from the past week by default,
# reruns incrementally update the dataset with latest jobs.
# Data is saved to ci_analytics/ci_jobs.json relative to repo root
./ci_job_collector.py

# Specify a different time range for original collection
./ci_job_collector.py --days 30  # Collect jobs from the past month
./ci_job_collector.py --days all  # Collect all available jobs
```

## ci_visualization.py

This script processes the JSON data created by ci_job_collector.py and generates interactive HTML visualizations of CI job metrics.

### Features

- Daily timeline views showing all jobs by runner, with monthly overview pages
- Interactive timeline filtering by project and branch
- Statistical analysis and trend charts
- Pareto chart and history of test runtimes
- Container image sizes: comparison chart and per-image size history
- Responsive design for easy navigation
- Visualizations are static HTML files that can be hosted on any web server
- All dates and times are treated as UTC

### Usage

```bash
# Basic usage (reads from ci_analytics/ci_jobs.json and outputs to ci_analytics relative to repo root)
./ci_visualization.py
```

## Getting Started

### Prerequisites

- Python 3.6+
- GitLab access with appropriate permissions
- Required Python packages:
  - python-dateutil

### Workflow

1. Run `ci_job_collector.py` to gather CI job data (saves to `ci_analytics/ci_jobs.json`, `ci_analytics/ci_test_runtimes.json` and `ci_analytics/container_image_sizes.json`)
2. Run `ci_visualization.py` to generate visualizations (outputs to `ci_analytics` by default)
3. Open the generated HTML files in a web browser to explore the data

## Example Output

The visualization creates several HTML files in the output folder:

- `index.html`: Main entry point with links to monthly data and statistics
- `statistics.html`: Overall statistics and trends
- `container_image_sizes.html`: Image size comparison and history
- `month_YYYY-MM.html`: Detailed daily timelines for each month
