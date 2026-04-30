#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
#
# Unified CUDA Coverage Script
# Runs host and device coverage, merges reports with lcov, and generates HTML output.
#
# Usage:
#   bazel run //internal/scripts/cuda_coverage:run_combined_coverage
#   bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-host --verbose
#   bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --output-dir my_coverage

set -euo pipefail

###############################################################################
# Helper Functions
###############################################################################

# Detect which bazel command to use (bazelisk for CI, bazel for local dev)
detect_bazel_command() {
    for cmd in bazelisk bazel; do
        if command -v "$cmd" &> /dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    echo "ERROR: Neither 'bazel' nor 'bazelisk' found in PATH" >&2
    exit 1
}

# Find the workspace root directory
find_workspace_root() {
    # Method 1: Use BUILD_WORKSPACE_DIRECTORY (set by bazel run)
    if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
        echo "${BUILD_WORKSPACE_DIRECTORY}"
        return 0
    fi

    # Method 2: Walk up from script location looking for MODULE.bazel
    local dir
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/MODULE.bazel" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done

    # Method 3: Fallback to current directory
    echo "$PWD"
}

# Filter out tests listed in skipcov_testlist.txt
# File format: one Bazel target per line, # comments, blank lines ignored
filter_skipped_tests() {
    local -n targets_ref=$1
    local skiplist="${WORKSPACE_ROOT}/internal/scripts/cuda_coverage/skipcov_testlist.txt"

    if [[ ! -f "$skiplist" ]]; then
        return
    fi

    # Read skip targets (strip comments and blank lines)
    local -a skip_targets=()
    while IFS= read -r line; do
        line="${line%%#*}"       # strip inline comments
        line="${line// /}"       # strip whitespace
        [[ -z "$line" ]] && continue
        skip_targets+=("$line")
    done < "$skiplist"

    if [[ ${#skip_targets[@]} -eq 0 ]]; then
        return
    fi

    # Filter targets array
    local -a filtered=()
    for target in "${targets_ref[@]}"; do
        local skip=0
        for st in "${skip_targets[@]}"; do
            if [[ "$target" == "$st" ]]; then
                skip=1
                echo "Skipping test (in skipcov_testlist.txt): $target"
                break
            fi
        done
        [[ "$skip" == "0" ]] && filtered+=("$target")
    done
    targets_ref=("${filtered[@]}")
}

BAZEL_CMD=$(detect_bazel_command)
echo "Using bazel command: ${BAZEL_CMD}"

###############################################################################
# Argument Parsing
###############################################################################

# Defaults
SKIP_HOST=0
SKIP_DEVICE=0
OUTPUT_DIR="combined_coverage_html"
VERBOSE=0
RUN_LONG_TESTS=0
TARGET=""

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified CUDA Coverage Script - runs host and device coverage, merges reports,
post-processes, and generates HTML output.

Options:
    --skip-host         Skip host coverage collection
    --skip-device       Skip device coverage collection
    --output-dir DIR    Set HTML output directory (default: combined_coverage_html)
    --target TARGET     Run coverage on a single test target (e.g., //path/to:test)
    --verbose, -v       Stream test output (--test_output=streamed)
    --run-long-tests    Include tests with timeout=long or timeout=eternal (skipped by default)
    --help, -h          Show this help message

Examples:
    # Run both host and device coverage (long/eternal tests skipped by default)
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage

    # Skip host coverage, run device only
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-host

    # Skip device coverage, run host only with verbose output
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --skip-device --verbose

    # Custom output directory
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --output-dir my_coverage

    # Include long/eternal timeout tests
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --run-long-tests

    # Run coverage on a single test target
    bazel run //internal/scripts/cuda_coverage:run_combined_coverage -- --target //path/to:my_test
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-host)
            SKIP_HOST=1
            shift
            ;;
        --skip-device)
            SKIP_DEVICE=1
            shift
            ;;
        --output-dir)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "ERROR: --output-dir requires a directory path argument."
                echo "Use --help for usage information."
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --run-long-tests)
            RUN_LONG_TESTS=1
            shift
            ;;
        --target)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "ERROR: --target requires a target path argument."
                echo "Use --help for usage information."
                exit 1
            fi
            TARGET="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Build test output flag based on VERBOSE setting
TEST_OUTPUT_FLAG=""
if [[ "${VERBOSE}" == "1" ]]; then
    TEST_OUTPUT_FLAG="--test_output=streamed"
fi

###############################################################################
# Change to Workspace Root
###############################################################################
# When run via 'bazel run', the script executes from the runfiles directory
# (inside bazel-out/). We must cd to the workspace root before running bazel
# commands to avoid "bazel should not be called from a bazel output directory".

WORKSPACE_ROOT=$(find_workspace_root)
echo "Changing to workspace root: ${WORKSPACE_ROOT}"
cd "${WORKSPACE_ROOT}"

###############################################################################
# Cache Directory Setup (for sandboxed test execution)
###############################################################################
# Bazel sandboxes make $HOME read-only. PyTorch/Triton need writable caches.
# This respects CI's persistent cache if set, otherwise uses workspace-local defaults.

# Default cache directory (used if CI hasn't configured persistent cache)
DEFAULT_CACHE_DIR="${WORKSPACE_ROOT}/.test_cache"

# Use existing env vars if set (respects CI's persistent cache), otherwise use defaults
TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_DIR}/torch}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEFAULT_CACHE_DIR}/triton}"
ASSET_HARVESTER_CACHE_DIR="${ASSET_HARVESTER_CACHE_DIR:-${DEFAULT_CACHE_DIR}/asset_harvester}"

# Determine the sandbox writable path - use parent of TORCH_HOME if it exists,
# otherwise use the default cache dir
if [[ -n "${TORCH_HOME:-}" && "${TORCH_HOME}" != "${DEFAULT_CACHE_DIR}/torch" ]]; then
    # CI case: TORCH_HOME was pre-set, derive sandbox path from its parent
    SANDBOX_WRITABLE_PATH="$(dirname "${TORCH_HOME}")"
else
    # Standalone case: use default cache directory
    SANDBOX_WRITABLE_PATH="${DEFAULT_CACHE_DIR}"
fi

# Ensure the cache directories exist
mkdir -p "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${ASSET_HARVESTER_CACHE_DIR}"

# Build cache flags for bazel coverage commands
CACHE_FLAGS=(
    "--sandbox_writable_path=${SANDBOX_WRITABLE_PATH}"
    "--test_env=TORCH_HOME=${TORCH_HOME}"
    "--test_env=TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    "--test_env=ASSET_HARVESTER_CACHE_DIR=${ASSET_HARVESTER_CACHE_DIR}"
)

###############################################################################
# GPU Architecture Detection (for faster device coverage builds)
###############################################################################
# Detect the GPU's compute capability and limit compilation to that architecture.
# This avoids compiling for multiple architectures when we only need 1, giving compilation speedups.

detect_gpu_arch() {
    # Get compute capability from nvidia-smi (e.g., "8.9" for L40)
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')

    if [[ -z "$compute_cap" ]]; then
        echo ""
        return 1
    fi

    # Convert "8.9" to "sm_89"
    local sm_version="sm_${compute_cap//./}"
    echo "$sm_version"
}

GPU_ARCH=$(detect_gpu_arch)
if [[ -n "$GPU_ARCH" ]]; then
    echo "Detected GPU architecture: ${GPU_ARCH}"
    GPU_ARCH_FLAG="--@rules_cuda//cuda:archs=${GPU_ARCH}"
else
    echo "WARNING: Could not detect GPU architecture, using default (all architectures)"
    GPU_ARCH_FLAG=""
fi

# Intermediate files
HOST_COVERAGE_DAT="host_coverage.dat"
DEVICE_COVERAGE_DAT="device_coverage.dat"
COMBINED_COVERAGE_DAT="combined_coverage.dat"
PROCESSED_COVERAGE_DAT="processed_coverage.dat"
TEST_MISS_COVERAGE_DAT="test_miss_coverage.dat"
TEST_MISS_OUTPUT_DIR="test_miss_report_html"

echo "=== NRE Unified Coverage Script ==="
echo "  --skip-host:   ${SKIP_HOST}"
echo "  --skip-device: ${SKIP_DEVICE}"
echo "  --output-dir:  ${OUTPUT_DIR}"
echo "  --target:      ${TARGET:-<all tests>}"
echo "  --verbose:     ${VERBOSE}"
echo "  --run-long-tests: ${RUN_LONG_TESTS}"
echo "  GPU architecture: ${GPU_ARCH:-<all>}"
echo "  Sandbox writable path: ${SANDBOX_WRITABLE_PATH}"
echo "  TORCH_HOME: ${TORCH_HOME}"
echo "  TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo ""

# Clean up previous intermediate files
rm -f "${HOST_COVERAGE_DAT}" "${DEVICE_COVERAGE_DAT}" "${COMBINED_COVERAGE_DAT}" "${PROCESSED_COVERAGE_DAT}" "${TEST_MISS_COVERAGE_DAT}"

###############################################################################
# Timing Infrastructure
###############################################################################
SCRIPT_START_TIME=$(date +%s)
HOST_COVERAGE_TIME=0
DEVICE_COVERAGE_TIME=0
MERGE_TIME=0
TEST_MISS_TIME=0
POSTPROCESS_TIME=0
HTML_GEN_TIME=0

# Helper function to format seconds as HH:MM:SS
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Helper function to safely calculate percentage (avoids division by zero)
safe_pct() {
    local value=$1
    local total=$2
    if [[ "$total" -eq 0 ]]; then
        echo 0
    else
        echo $((value * 100 / total))
    fi
}

###############################################################################
# 1. Run Host Coverage (if SKIP_HOST != 1)
###############################################################################
if [[ "${SKIP_HOST}" != "1" ]]; then
    echo "=== Running Host Coverage ==="
    HOST_START=$(date +%s)

    # Calculate dynamic GCOV_PREFIX_STRIP for sandbox paths
    EXECROOT=$(${BAZEL_CMD} info execution_root)
    STRIP_COUNT=$(($(echo "${EXECROOT}" | tr '/' '\n' | grep -c .) + 3))
    echo "Using GCOV_PREFIX_STRIP=${STRIP_COUNT}"

    # Determine test targets
    if [[ -n "${TARGET}" ]]; then
        # Single target specified
        TARGETS=("${TARGET}")
        echo "Running coverage on single target: ${TARGET}"
    elif [[ "${RUN_LONG_TESTS}" == "1" ]]; then
        # Include all tests (excluding internal/manual/no-coverage targets)
        mapfile -t TARGETS < <(${BAZEL_CMD} query 'kind(".*_test rule", //...) except (attr("name", "^_|^requirements_|_mypy$", //...) + attr("tags", "manual|no-coverage", //...))')
    else
        # Default: exclude long/eternal timeout tests
        echo "Filtering out tests with timeout=long or timeout=eternal (use --run-long-tests to include)..."
        mapfile -t TARGETS < <(${BAZEL_CMD} query 'kind(".*_test rule", //...) except (attr("name", "^_|^requirements_|_mypy$", //...) + attr("tags", "manual|no-coverage", //...) + attr("timeout", "long", //...) + attr("timeout", "eternal", //...))')
    fi
    echo "Running ${#TARGETS[@]} host test targets"

    # Run CUDA host coverage with cuda_coverage_host config
    # Use || true to continue even if some tests fail (coverage is still generated)
    ${BAZEL_CMD} coverage --config=cuda_coverage_host \
        --keep_going \
        --flaky_test_attempts=3 \
        --test_env=GCOV_PREFIX_STRIP=${STRIP_COUNT} \
        ${GPU_ARCH_FLAG} \
        "${CACHE_FLAGS[@]}" \
        "${TARGETS[@]}" || true

    # Save host coverage report
    if [[ -f bazel-out/_coverage/_coverage_report.dat ]]; then
        cp bazel-out/_coverage/_coverage_report.dat "${HOST_COVERAGE_DAT}"
        echo "Host coverage saved to ${HOST_COVERAGE_DAT}"
    else
        echo "WARNING: Host coverage report not found"
    fi

    HOST_END=$(date +%s)
    HOST_COVERAGE_TIME=$((HOST_END - HOST_START))
    echo "Host coverage completed in $(format_time $HOST_COVERAGE_TIME)"
else
    echo "=== Skipping Host Coverage (--skip-host) ==="
fi

echo ""

###############################################################################
# 2. Run Device Coverage (if SKIP_DEVICE != 1)
###############################################################################
: # No-op to ensure clean parsing after previous block
if [[ "${SKIP_DEVICE}" != "1" ]]; then
    echo "=== Running Device Coverage ==="
    DEVICE_START=$(date +%s)

    # Determine device test targets
    if [[ -n "${TARGET}" ]]; then
        # Single target specified
        DEV_TARGETS=("${TARGET}")
        echo "Running device coverage on single target: ${TARGET}"
    elif [[ "${RUN_LONG_TESTS}" == "1" ]]; then
        # Include all device test targets (py_test/pytest_test only, excluding internal/manual/no-coverage)
        mapfile -t DEV_TARGETS < <(${BAZEL_CMD} query 'kind("py_test|pytest_test", //...) except (attr("name", "^_|^requirements_|_mypy|bazel_targets_test$", //...) + attr("tags", "manual|no-coverage", //...))')
    else
        # Default: exclude long/eternal timeout tests
        echo "Filtering out tests with timeout=long or timeout=eternal (use --run-long-tests to include)..."
        mapfile -t DEV_TARGETS < <(${BAZEL_CMD} query 'kind("py_test|pytest_test", //...) except (attr("name", "^_|^requirements_|_mypy|bazel_targets_test$", //...) + attr("tags", "manual|no-coverage", //...) + attr("timeout", "long", //...) + attr("timeout", "eternal", //...))')
    fi
    filter_skipped_tests DEV_TARGETS
    echo "Running ${#DEV_TARGETS[@]} device test targets"

    # Run CUDA device coverage with cuda_coverage_dev config
    # Use || true to continue even if some tests fail (coverage is still generated)
    ${BAZEL_CMD} coverage --config=cuda_coverage_dev \
        --keep_going \
        --flaky_test_attempts=3 \
        ${GPU_ARCH_FLAG} \
        ${TEST_OUTPUT_FLAG} \
        "${CACHE_FLAGS[@]}" \
        "${DEV_TARGETS[@]}" || true

    # Save device coverage report
    if [[ -f bazel-out/_coverage/_coverage_report.dat ]]; then
        cp bazel-out/_coverage/_coverage_report.dat "${DEVICE_COVERAGE_DAT}"
        echo "Device coverage saved to ${DEVICE_COVERAGE_DAT}"
    else
        echo "WARNING: Device coverage report not found"
    fi

    DEVICE_END=$(date +%s)
    DEVICE_COVERAGE_TIME=$((DEVICE_END - DEVICE_START))
    echo "Device coverage completed in $(format_time $DEVICE_COVERAGE_TIME)"
else
    echo "=== Skipping Device Coverage (--skip-device) ==="
fi

echo ""

###############################################################################
# 3. Merge Coverage Reports with lcov
###############################################################################
: # No-op to ensure clean parsing after previous block
echo "=== Merging Coverage Reports ==="
MERGE_START=$(date +%s)

TRACEFILES=()
[[ -f "${HOST_COVERAGE_DAT}" ]] && TRACEFILES+=(--add-tracefile "${HOST_COVERAGE_DAT}")
[[ -f "${DEVICE_COVERAGE_DAT}" ]] && TRACEFILES+=(--add-tracefile "${DEVICE_COVERAGE_DAT}")

if [[ ${#TRACEFILES[@]} -eq 0 ]]; then
    echo "ERROR: No coverage reports found to merge"
    exit 1
fi

lcov --quiet "${TRACEFILES[@]}" --output-file "${COMBINED_COVERAGE_DAT}"
echo "Combined coverage saved to ${COMBINED_COVERAGE_DAT}"

MERGE_END=$(date +%s)
MERGE_TIME=$((MERGE_END - MERGE_START))
echo "Merge completed in $(format_time $MERGE_TIME)"

echo ""

###############################################################################
# 4. Post-process Coverage Data
###############################################################################
echo "=== Post-processing Coverage Data ==="
POSTPROCESS_START=$(date +%s)

POSTPROCESS_ARGS=(
    "${PWD}/${COMBINED_COVERAGE_DAT}"
    -o "${PWD}/${PROCESSED_COVERAGE_DAT}"
    -s "${PWD}"
)
[[ "${VERBOSE}" == "1" ]] && POSTPROCESS_ARGS+=(-v)

${BAZEL_CMD} run //internal/scripts/cuda_coverage:postprocess_coverage -- "${POSTPROCESS_ARGS[@]}"

echo "Processed coverage saved to ${PROCESSED_COVERAGE_DAT}"

POSTPROCESS_END=$(date +%s)
POSTPROCESS_TIME=$((POSTPROCESS_END - POSTPROCESS_START))
echo "Post-processing completed in $(format_time $POSTPROCESS_TIME)"

echo ""

###############################################################################
# 5. Generate HTML Report
###############################################################################
echo "=== Generating HTML Report ==="
HTML_GEN_START=$(date +%s)

genhtml \
    --title "NRE Combined Coverage Report" \
    --output "${OUTPUT_DIR}" \
    "${PROCESSED_COVERAGE_DAT}" \
    --ignore-errors source \
    --quiet

HTML_GEN_END=$(date +%s)
HTML_GEN_TIME=$((HTML_GEN_END - HTML_GEN_START))
echo "HTML generation completed in $(format_time $HTML_GEN_TIME)"

echo ""
echo "=== Coverage Report Complete ==="
echo "HTML report generated at: ${OUTPUT_DIR}/index.html"

echo ""

###############################################################################
# 6. Generate Test Miss Files Report (files not touched by any test)
###############################################################################
echo "=== Generating Test Miss Files Report ==="
TEST_MISS_START=$(date +%s)

# Generate DA:line,0 entries for source files NOT present in actual coverage.
# This produces a separate report showing files with zero test coverage.
${BAZEL_CMD} run //internal/scripts/cuda_coverage:generate_baseline_coverage -- \
    -s "${PWD}" \
    -o "${PWD}/${TEST_MISS_COVERAGE_DAT}" \
    --exclude-from "${PWD}/${COMBINED_COVERAGE_DAT}"

if [[ -f "${TEST_MISS_COVERAGE_DAT}" ]]; then
    genhtml \
        --title "NRE Test Miss Files Report (Untested Source Files)" \
        --output "${TEST_MISS_OUTPUT_DIR}" \
        "${TEST_MISS_COVERAGE_DAT}" \
        --ignore-errors source \
        --quiet
    echo "Test miss report generated at: ${TEST_MISS_OUTPUT_DIR}/index.html"
else
    echo "WARNING: Test miss coverage data not generated"
fi

TEST_MISS_END=$(date +%s)
TEST_MISS_TIME=$((TEST_MISS_END - TEST_MISS_START))
echo "Test miss report completed in $(format_time $TEST_MISS_TIME)"

###############################################################################
# Timing Summary
###############################################################################
SCRIPT_END_TIME=$(date +%s)
TOTAL_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo ""
echo "========================================"
echo "         TIMING SUMMARY"
echo "========================================"
if [[ "${SKIP_HOST}" != "1" ]]; then
    printf "Host Coverage:        %s (%3d%%)\n" "$(format_time $HOST_COVERAGE_TIME)" "$(safe_pct $HOST_COVERAGE_TIME $TOTAL_TIME)"
fi
if [[ "${SKIP_DEVICE}" != "1" ]]; then
    printf "Device Coverage:      %s (%3d%%)\n" "$(format_time $DEVICE_COVERAGE_TIME)" "$(safe_pct $DEVICE_COVERAGE_TIME $TOTAL_TIME)"
fi
printf "Merge Reports:        %s (%3d%%)\n" "$(format_time $MERGE_TIME)" "$(safe_pct $MERGE_TIME $TOTAL_TIME)"
printf "Post-processing:      %s (%3d%%)\n" "$(format_time $POSTPROCESS_TIME)" "$(safe_pct $POSTPROCESS_TIME $TOTAL_TIME)"
printf "HTML Generation:      %s (%3d%%)\n" "$(format_time $HTML_GEN_TIME)" "$(safe_pct $HTML_GEN_TIME $TOTAL_TIME)"
printf "Test Miss Report:     %s (%3d%%)\n" "$(format_time $TEST_MISS_TIME)" "$(safe_pct $TEST_MISS_TIME $TOTAL_TIME)"
echo "----------------------------------------"
printf "TOTAL TIME:           %s\n" "$(format_time $TOTAL_TIME)"
echo "========================================"

echo ""
echo "========================================"
echo "         REPORT LOCATIONS"
echo "========================================"
echo "Main Coverage Report:     ${OUTPUT_DIR}/index.html"
if [[ -f "${TEST_MISS_COVERAGE_DAT}" ]]; then
    echo "Test Miss Report:         ${TEST_MISS_OUTPUT_DIR}/index.html"
fi
echo "========================================"

