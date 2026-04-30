# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Bazelisk wrapper that automatically captures telemetry files for each invocation.
# This file is sourced via BASH_ENV in CI jobs.
#
# Output files are placed in $CI_PROJECT_DIR/bazel-telemetry/:
#   - bep_00_build.json, bep_01_test.json, ... (all commands)
#   - profile_00_build.json.gz, profile_01_test.json.gz, ... (all commands)
#   - exec_log_00_build.binpb, exec_log_01_test.binpb, ... (build/test/run/coverage only)
#   - explain_00_build.txt, explain_01_test.txt, ... (build/test/run/coverage only)

bazelisk() {
  local cmd="${1:-}"
  local telemetry_flags=()

  case "$cmd" in
    # Commands that support telemetry, pass additional flags
    build|test|run|coverage|query|cquery|aquery|fetch|sync)
      local telemetry_dir="${CI_PROJECT_DIR:-$(pwd)}/bazel-telemetry"
      local counter_file="$telemetry_dir/.call_counter"

      mkdir -p "$telemetry_dir"
      if [ ! -f "$counter_file" ]; then
        echo "0" > "$counter_file"
      fi

      local count
      count=$(cat "$counter_file")
      echo $((count + 1)) > "$counter_file"

      local seq
      seq=$(printf "%02d" "$count")

      # BEP and profile: supported by all telemetry-enabled commands
      telemetry_flags+=(
        "--build_event_json_file=$telemetry_dir/bep_${seq}_${cmd}.json"
        "--profile=$telemetry_dir/profile_${seq}_${cmd}.json.gz"
      )

      # Execution log and explain: only for execution commands
      case "$cmd" in
        build|test|run|coverage)
          telemetry_flags+=(
            "--execution_log_compact_file=$telemetry_dir/exec_log_${seq}_${cmd}.binpb"
            "--explain=$telemetry_dir/explain_${seq}_${cmd}.txt"
          )
          ;;
      esac

      command bazelisk "$1" "${telemetry_flags[@]}" "${@:2}"
      ;;
    *)
      # Pass through without telemetry (version, help, info, clean, etc.)
      command bazelisk "$@"
      ;;
  esac
}

# Export function so it's available in subshells and called scripts
export -f bazelisk
