#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pick N random uuids that passed the dataset-load smoke and run
# parity_one_dataset.sh against each. Each per-clip parity report
# lands at <output_dir>/pai_<UUID>.json; PLYs are deleted by
# parity_one_dataset.sh's own trap.
#
# Usage:
#   parity_subset.sh <clips_dir> <dataset_load_report.json> <output_dir> [N] [SEED]
#
# Defaults: N=50, SEED=42.

set -uo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "usage: $0 <clips_dir> <dataset_load_report.json> <output_dir> [N=50] [SEED=42]" >&2
    exit 64
fi

CLIPS_DIR=$1
REPORT_JSON=$2
OUT_DIR=$3
N=${4:-50}
SEED=${5:-42}

if [[ ! -d "$CLIPS_DIR" ]]; then
    echo "error: clips_dir '$CLIPS_DIR' is not a directory" >&2
    exit 65
fi
if [[ ! -f "$REPORT_JSON" ]]; then
    echo "error: report '$REPORT_JSON' does not exist" >&2
    exit 65
fi

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER=$REPO_DIR/internal/parity_proofs/parity_one_dataset.sh
if [[ ! -x "$RUNNER" ]]; then
    echo "error: runner '$RUNNER' missing or not executable" >&2
    exit 66
fi

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
SUBSET_FILE=$OUT_DIR/subset_uuids.txt

# Pick N random uuids from the report's passed_uuids[] list, with a fixed seed.
VENV_PY=$REPO_DIR/.venv/bin/python
"$VENV_PY" - "$REPORT_JSON" "$N" "$SEED" "$SUBSET_FILE" <<'PY'
import json, random, sys
report_path, n_arg, seed_arg, out_path = sys.argv[1:5]
with open(report_path) as f:
    report = json.load(f)
passed = report.get("passed_uuids", [])
n = int(n_arg)
if n > len(passed):
    print(f"warning: requested {n} > available {len(passed)} passed uuids; "
          f"using all {len(passed)}", file=sys.stderr)
    n = len(passed)
rng = random.Random(int(seed_arg))
chosen = rng.sample(passed, n)
with open(out_path, "w") as f:
    for uuid in chosen:
        f.write(uuid + "\n")
print(f"picked {len(chosen)} uuids (seed={seed_arg})")
PY

mapfile -t UUIDS < "$SUBSET_FILE"
TOTAL=${#UUIDS[@]}
echo "running parity_one_dataset.sh on $TOTAL clip(s) -> $OUT_DIR"

PASSED=0
FAILED=0
FAILED_UUIDS=()
for i in "${!UUIDS[@]}"; do
    uuid=${UUIDS[$i]}
    json_path=$CLIPS_DIR/$uuid/pai_$uuid.json
    if [[ ! -f "$json_path" ]]; then
        echo "[$((i+1))/$TOTAL] $uuid  MISSING_JSON  $json_path" >&2
        FAILED=$((FAILED + 1))
        FAILED_UUIDS+=("$uuid")
        continue
    fi
    echo "[$((i+1))/$TOTAL] $uuid  start"
    if "$RUNNER" "$json_path" "$OUT_DIR"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_UUIDS+=("$uuid")
    fi
done

echo
echo "=== parity_subset summary ==="
echo "passed: $PASSED / $TOTAL"
echo "failed: $FAILED / $TOTAL"
if [[ $FAILED -gt 0 ]]; then
    echo "failed uuids:"
    printf '  %s\n' "${FAILED_UUIDS[@]}"
fi

# Roll up the per-clip JSON reports into one summary.
"$VENV_PY" - "$OUT_DIR" "$OUT_DIR/parity_subset_summary.json" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
per_clip = []
for p in sorted(out_dir.glob("pai_*.json")):
    try:
        per_clip.append(json.loads(p.read_text()))
    except json.JSONDecodeError as exc:
        per_clip.append({"_path": str(p), "_decode_error": str(exc)})
overall_pass = sum(1 for c in per_clip if c.get("overall") == "PASS")
overall_fail = sum(1 for c in per_clip if c.get("overall") == "FAIL")
summary = {
    "n_clips": len(per_clip),
    "n_overall_pass": overall_pass,
    "n_overall_fail": overall_fail,
    "per_clip": per_clip,
}
summary_path.write_text(json.dumps(summary, indent=2))
print(f"summary -> {summary_path}  (PASS={overall_pass}, FAIL={overall_fail})")
PY

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
