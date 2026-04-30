<!-- Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved. -->

# Unified Coverage Tools (CUDA & Slang)

Two-tier code coverage for CUDA & Slang: **host coverage** (gcov) and **device coverage** (NCU profiling of GPU kernels). Both produce standard LCOV reports, merged and rendered as HTML via `genhtml`.

## Architecture

```
                      run_combined_coverage.sh  (orchestrator)
                         ┌──────┴──────┐
                    Host (gcov)    Device (NCU)
                         │             │
                         │        ncu_coverage_wrapper.sh
                         │          └─ infer_ncu_coverage.py
                         ▼             ▼
                   host_coverage.dat  device_coverage.dat
                         └──────┬──────┘
                           lcov merge → postprocess_coverage.py → genhtml
                                                                → generate_baseline_coverage.py (test miss report)
```

## Prerequisites

- **NVIDIA Nsight Compute (ncu)** from the CUDA Toolkit
- **GPU profiling enabled**: `grep RmProfilingAdminOnly /proc/driver/nvidia/params` must show `0`

## Quick Start

```bash
bazel run //internal/scripts/cuda_coverage:run_combined_coverage                    # full (host + device)
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-device    # host only
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-host      # device only
bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --target //p:t   # single target
```

HTML report: `combined_coverage_html/index.html` (default). For standard Python/C++ coverage without CUDA, see the [Code coverage](../../README.md#code-coverage) section in the main README.

## Scripts

| Script                          | Purpose                                                                                                                                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_combined_coverage.sh`      | Orchestrator: runs host/device coverage, merges with `lcov`, post-processes, generates HTML. Flags: `--skip-host`, `--skip-device`, `--output-dir DIR`, `--target TARGET`, `--verbose`, `--run-long-tests` |
| `ncu_coverage_wrapper.sh`       | Bazel `--run_under` wrapper. Runs test under `ncu`, converts `.ncu-rep` → LCOV via `infer_ncu_coverage.py`. Not called directly.                                                                           |
| `infer_ncu_coverage.py`         | Converts NCU `.ncu-rep` reports to LCOV. Auto-detects best available metric, processes kernels in batches, filters out library kernels.                                                                    |
| `postprocess_coverage.py`       | Cleans LCOV data: resolves sandbox paths, removes non-executable lines, infers coverage within code blocks, detects function boundaries.                                                                   |
| `generate_baseline_coverage.py` | Generates zero-count LCOV for all source files (`.py`, `.cpp`, `.cu`, `.cuh`, `.slang`, `.c`, `.h`, `.hpp`). Untested files appear as 0% in the merged report.                                             |
| `line_classifier.py`            | Shared library classifying source lines (executable, blank, comment, preprocessor, brace-only, declaration, attribute, namespace) for C++/CUDA, Slang, and Python.                                         |
| `skipcov_testlist.txt`          | Test targets to exclude from device coverage (one per line). Host coverage still runs all tests.                                                                                                           |

## Bazel Configuration

Two configs in `.bazelrc`:

**`cuda_coverage_host`** — Compiles in debug mode with gcov instrumentation to measure host-side line coverage. Handles Bazel sandbox path remapping.

**`cuda_coverage_dev`** — Compiles with minimal optimization and full line-info for accurate source correlation. Runs each test under the NCU profiler wrapper. Extends test timeouts and limits parallelism to account for profiling overhead. JIT env vars ensure runtime-compiled code (CUDA RTC, Slang) is also built unoptimized.

## CI Integration

Two jobs in `.gitlab/ci/coverage.gitlab-ci.yml`:

- **`coverage`** — Standard Python/C++ coverage. Auto on `main`/`release/*`, manual elsewhere. Artifacts: `coverage/`.
- **`combined_coverage`** — Full host + device CUDA & Slang coverage. Weekly (auto), manual anytime. Checks `RmProfilingAdminOnly` at start. Artifacts: `combined_coverage/`, `test_miss_report_html/`, `host_coverage.dat`, `device_coverage.dat`.

## Output Files

| File                      | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `host_coverage.dat`       | Host (gcov) LCOV data                                 |
| `device_coverage.dat`     | Device (NCU) LCOV data                                |
| `combined_coverage.dat`   | Merged LCOV data                                      |
| `processed_coverage.dat`  | Post-processed LCOV (cleaned paths, classified lines) |
| `combined_coverage_html/` | HTML coverage report                                  |
| `test_miss_report_html/`  | HTML report of untested source files                  |

## Troubleshooting

- **`RmProfilingAdminOnly` not 0** — NCU needs profiling access. Contact your sysadmin to set this driver parameter.
- **NCU not found** — Verify with `which ncu`. Ensure the CUDA Toolkit is installed with Nsight Compute.
- **`ncu_report` import errors** — NCU Python module missing or version mismatch. The wrapper handles path setup automatically; check `ncu_coverage_wrapper.sh` if issues persist.
- **Sandbox permission errors** — The orchestrator sets up writable caches for dependencies that need them. Check `run_combined_coverage.sh` if issues persist.
- **Tests timing out** — NCU profiling adds significant overhead. Use `--target` to run a single target, add slow tests to `skipcov_testlist.txt`, or use `--skip-device`.
