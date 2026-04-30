<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# CI/CD Pipeline Documentation

## Security Stage: SonarQube SAST

We have integrated SonarQube Static Application Security Testing (SAST) to analyze code quality and security vulnerabilities.

**Dashboard**: <https://sonar-sw.nvidia.com/dashboard?id=GPUSW_Pixels_NuRec_NuRec>

### Triggering the Scan

The `sonarqube_scan` job is configured to run nightly for now to gather data and identify issues. Once the issues are resolved, we will add other triggers. For example, post-merge scans on main/release branches.

### Configuration Details

- **Pipeline Stage**: `security`
- **Job Definition**: `.gitlab/ci/sonarqube.gitlab-ci.yml`
- **Scanner Config**: `sonar-project.properties` (Root directory)
- **Build Integration**: Generates a compilation database (`compile_commands.json`) at the root of the repo for accurate C++ analysis.
  - **Note**: Tools like clangd can also benefit from it. It can be generated locally for development by running `bazel run @hedron_compile_commands//:refresh_all`.
