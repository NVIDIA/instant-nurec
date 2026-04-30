#!/bin/bash
# NCU Coverage Wrapper for bazel coverage --run_under
# Usage: bazel coverage --config=cuda_device_coverage //target:test

set -euo pipefail

# The test binary is $1, remaining args go to the test
TEST_BINARY="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output paths
NCU_REP="${TEST_UNDECLARED_OUTPUTS_DIR:-/tmp}/coverage.ncu-rep"
LCOV_OUT="${COVERAGE_OUTPUT_FILE:-${TEST_UNDECLARED_OUTPUTS_DIR:-/tmp}/coverage.dat}"

# Find repo root - resolve symlinks from runfiles to find real source tree
find_repo_root() {
    # Method 1: Resolve symlinks from the script itself (runfiles are symlinks to source)
    local real_script
    real_script="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "")"
    if [[ -n "$real_script" ]]; then
        local dir="$(dirname "$real_script")"
        while [[ ! -f "$dir/MODULE.bazel" && "$dir" != "/" ]]; do
            dir="$(dirname "$dir")"
        done
        if [[ -f "$dir/MODULE.bazel" ]]; then
            echo "$dir"
            return 0
        fi
    fi
    
    # Method 2: Fallback to BUILD_WORKSPACE_DIRECTORY or BUILD_WORKING_DIRECTORY
    if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
        echo "${BUILD_WORKSPACE_DIRECTORY}"
        return 0
    fi
    if [[ -n "${BUILD_WORKING_DIRECTORY:-}" ]]; then
        echo "${BUILD_WORKING_DIRECTORY}"
        return 0
    fi
    
    echo "$PWD"
}
SOURCE_ROOT="$(find_repo_root)"

# Find NCU installation directory that has BOTH ncu binary AND Python module
# Returns the installation directory (not the binary path)
find_ncu_installation() {
    local candidates=()
    
    # Collect candidate installation directories
    # 1. /opt/nvidia/nsight-compute/<version>/ (Docker/apt package)
    if [[ -d "/opt/nvidia/nsight-compute" ]]; then
        for dir in /opt/nvidia/nsight-compute/*/; do
            [[ -d "$dir" ]] && candidates+=("${dir%/}")
        done
    fi
    
    # 2. /opt/nvidia/nsight-compute-<version>/ (alternative layout)
    for dir in /opt/nvidia/nsight-compute-*/; do
        [[ -d "$dir" ]] && candidates+=("${dir%/}")
    done
    
    # 3. /usr/local/cuda/nsight-compute-<version>/
    for dir in /usr/local/cuda/nsight-compute-*/; do
        [[ -d "$dir" ]] && candidates+=("${dir%/}")
    done
    
    # Check each candidate for BOTH ncu binary AND Python module
    for install_dir in "${candidates[@]}"; do
        local ncu_bin=""
        local python_path="${install_dir}/extras/python"
        
        # Find ncu binary (could be at root or in bin/ or target/)
        if [[ -x "${install_dir}/ncu" ]]; then
            ncu_bin="${install_dir}/ncu"
        elif [[ -x "${install_dir}/bin/ncu" ]]; then
            ncu_bin="${install_dir}/bin/ncu"
        fi
        
        # Only return if BOTH exist
        if [[ -n "$ncu_bin" && -f "${python_path}/ncu_report.py" ]]; then
            echo "$install_dir"
            return 0
        fi
    done
    
    return 1
}

# Find valid NCU installation
NCU_INSTALL=$(find_ncu_installation)
if [[ -z "$NCU_INSTALL" ]]; then
    echo "[ncu_coverage] ERROR: No NCU installation found with both binary and Python module." >&2
    echo "[ncu_coverage] Searched: /opt/nvidia/nsight-compute/*/, /opt/nvidia/nsight-compute-*/, /usr/local/cuda/nsight-compute-*/" >&2
    exit 1
fi

# Derive paths from installation
if [[ -x "${NCU_INSTALL}/ncu" ]]; then
    NCU_BIN="${NCU_INSTALL}/ncu"
else
    NCU_BIN="${NCU_INSTALL}/bin/ncu"
fi
export NCU_PYTHON_PATH="${NCU_INSTALL}/extras/python"

echo "[ncu_coverage] NCU installation: $NCU_INSTALL"
echo "[ncu_coverage] NCU binary: $NCU_BIN"
echo "[ncu_coverage] NCU Python: $NCU_PYTHON_PATH"

INFER_SCRIPT="${SCRIPT_DIR}/infer_ncu_coverage.py"
if [[ ! -f "$INFER_SCRIPT" && -n "${RUNFILES_DIR:-}" ]]; then
    INFER_SCRIPT="${RUNFILES_DIR}/_main/internal/scripts/cuda_coverage/infer_ncu_coverage.py"
fi
if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "[ncu_coverage] ERROR: Cannot find infer_ncu_coverage.py" >&2
    exit 1
fi

echo "[ncu_coverage] Test: $TEST_BINARY"
echo "[ncu_coverage] Source root: $SOURCE_ROOT"
echo "[ncu_coverage] Running test under NCU profiler..."
echo "[ncu_coverage] NCU report: $NCU_REP"
echo "[ncu_coverage] LCOV output: $LCOV_OUT"

METRIC="thread_inst_executed_true"

# Run test under NCU
# collect only single targetted metric
## --replay-mode application is empirically faster for single pass
NCU_START=$(date +%s)
env -u COVERAGE_DIR -u COVERAGE_PROCESS_START \
    "$NCU_BIN" \
    --target-processes-filter python3 \
    --replay-mode application \
    --disable-extra-suffixes \
    --metrics $METRIC \
    --import-source no \
    --clock-control none \
    --cache-control none \
    --kernel-id ::regex:'^(?!.*(nccl|Device[A-Z]|fillSequence|_elementwise|elementwise_|reduce_kernel|cat_kernel|index_select|fill_kernel|copy_kernel|batch_norm|layer_norm|softmax_warp|implicit_convolve|winograd|gemm|gemv|void at_cuda|mlp_convert|jit_layout|FullyFusedMLP|activation_kernel|CUtensorMap|grid_stride_kernel|distribution_|radixSort|bitonicSort|mergeSort|segmentedSort|CatArray|Batch|roll_cuda_kernel|indexing_backward_kernel_stride|multi_tensor_apply_kernel|grid_sampler_3d_kernel|indexing_backward_kernel|grid_sampler_3d_backward_kernel))':'^([0-9]|1[0-5]|4[5-9]|5[0-9]|60)$' \
    -o "$NCU_REP" \
    -f \
    "$TEST_BINARY" "$@"

TEST_EXIT=$?
NCU_END=$(date +%s)
NCU_ELAPSED=$((NCU_END - NCU_START))
echo "[ncu_coverage] NCU profiling completed in ${NCU_ELAPSED}s"

# Convert NCU report to LCOV format
if [[ -f "$NCU_REP" ]]; then
    echo "[ncu_coverage] Converting NCU report to LCOV..."
    CONVERT_START=$(date +%s)

    # Find NCU's bundled libstdc++ (needed when system GCC is older than NCU's build)
    # We use it unconditionally to avoid the overhead of a failed first attempt
    NCU_LIBSTDCXX=$(find "${NCU_INSTALL}" -name "libstdc++.so*" -type f 2>/dev/null | head -1)
    if [[ -n "$NCU_LIBSTDCXX" ]]; then
        LD_PRELOAD="$NCU_LIBSTDCXX" python3 "$INFER_SCRIPT" "$NCU_REP" --lcov "$LCOV_OUT" -m $METRIC --source "$SOURCE_ROOT" || echo "[ncu_coverage] Warning: LCOV conversion failed"
    else
        # Fallback: try without LD_PRELOAD if bundled libstdc++ not found
        python3 "$INFER_SCRIPT" "$NCU_REP" --lcov "$LCOV_OUT" -m $METRIC --source "$SOURCE_ROOT" || echo "[ncu_coverage] Warning: LCOV conversion failed"
    fi

    CONVERT_END=$(date +%s)
    CONVERT_ELAPSED=$((CONVERT_END - CONVERT_START))
    echo "[ncu_coverage] LCOV conversion completed in ${CONVERT_ELAPSED}s"

    TOTAL_ELAPSED=$((CONVERT_END - NCU_START))
    echo "[ncu_coverage] Total test runtime: ${TOTAL_ELAPSED}s (NCU: ${NCU_ELAPSED}s, conversion: ${CONVERT_ELAPSED}s)"
else
    echo "[ncu_coverage] Warning: No NCU report generated"
    echo -e "TN:cuda_device_coverage\nend_of_record" > "$LCOV_OUT"
fi

exit $TEST_EXIT
