#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run NRE@a54a6af + standalone instant-nurec on a single ncorev4 sequence
# JSON, compare the produced PLYs with validate_parity.py, write a JSON
# report to <out>/parity.json, delete all PLYs.
#
# Usage:
#   parity_one_dataset.sh <ncorev4_pai_<uuid>.json> <output_dir>
#
# Output (named after the dataset's basename, e.g. pai_<UUID>.json):
#   <output_dir>/<basename>.json        — always (success and failure).
#   <output_dir>/<basename>.errors.log  — only when at least one subprocess failed.

set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <dataset.json> <output_dir>" >&2
    exit 64
fi

JSON_PATH=$1
OUT_DIR=$2

NRE_REPO=/storage/projects/nre
NRE_COMMIT=a54a6af0a177beabd01fe37e398c45be165a270f
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VALIDATE=$REPO_DIR/internal/benchmark/validate_parity.py
VENV_PY=$REPO_DIR/.venv/bin/python

if [[ ! -f "$JSON_PATH" || "$JSON_PATH" != *.json ]]; then
    echo "error: '$JSON_PATH' is not an existing .json file" >&2
    exit 65
fi
if [[ ! -f "$REPO_DIR/run_inference.py" ]]; then
    echo "error: $REPO_DIR/run_inference.py missing" >&2
    exit 66
fi
if [[ ! -x "$VENV_PY" ]]; then
    echo "error: $VENV_PY missing — run ./setup.sh first" >&2
    exit 66
fi

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
JSON_PATH=$(realpath "$JSON_PATH")
PARITY_JSON=$OUT_DIR/$(basename "$JSON_PATH" .json).json
ERR_LOG=$OUT_DIR/$(basename "$JSON_PATH" .json).errors.log
: > "$ERR_LOG"

SCRATCH=$(mktemp -d "$OUT_DIR/scratch.XXXXXX")
NRE_OUT=$SCRATCH/nre
INR_OUT=$SCRATCH/instant
mkdir -p "$NRE_OUT/no_merge" "$NRE_OUT/merge" "$INR_OUT/no_merge" "$INR_OUT/merge"

NRE_PREV_HEAD=""
cleanup() {
    if [[ -n "$NRE_PREV_HEAD" ]]; then
        git -C "$NRE_REPO" checkout --quiet "$NRE_PREV_HEAD" 2>/dev/null || true
    fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

# If the NRE repo is already at $NRE_COMMIT, skip the checkout/restore entirely
# — the script is then non-mutating on the NRE side, which matters when running
# this driver many times in a row (subset parity sweeps, CI, etc).
NRE_ALREADY_AT_COMMIT=0
NRE_CUR_SHA=$(git -C "$NRE_REPO" rev-parse HEAD 2>/dev/null || echo "")
NRE_TARGET_SHA=$(git -C "$NRE_REPO" rev-parse "$NRE_COMMIT" 2>/dev/null || echo "")
if [[ -n "$NRE_CUR_SHA" && "$NRE_CUR_SHA" == "$NRE_TARGET_SHA" ]]; then
    NRE_ALREADY_AT_COMMIT=1
fi

# Derive base_path + LST entry. ncorev4 layout is <base>/clips/<UUID>/pai_<UUID>.json
# so base_path is the grandparent of the JSON's directory.
JSON_DIR=$(dirname "$JSON_PATH")
BASE_PATH=$(dirname "$(dirname "$JSON_DIR")")
LST_ENTRY=$(realpath --relative-to="$BASE_PATH" "$JSON_PATH")
LST_FILE=$SCRATCH/single.lst
echo "$LST_ENTRY" > "$LST_FILE"

NRE_FAIL=0
INR_FAIL=0

# 1) NRE bazel reference run -----------------------------------------------------
# Two bazel runs; each call's exit status is captured explicitly via ||.
# The surrounding ``{ } >> log 2>&1`` only redirects output — its own exit
# status would be that of the last command (popd in the old version), which
# masked real failures.
{
    echo "===== $(date -Is) NRE bazel run ($NRE_COMMIT) =====" >&2
    if [[ $NRE_ALREADY_AT_COMMIT -eq 0 ]]; then
        NRE_PREV_HEAD=$(git -C "$NRE_REPO" rev-parse --abbrev-ref HEAD)
        [[ "$NRE_PREV_HEAD" == "HEAD" ]] && NRE_PREV_HEAD=$(git -C "$NRE_REPO" rev-parse HEAD)
        git -C "$NRE_REPO" checkout --quiet "$NRE_COMMIT" || NRE_FAIL=1
    fi

    if [[ $NRE_FAIL -eq 0 ]]; then
        pushd "$NRE_REPO" >/dev/null
        bazel run //nre/nrm:run -- \
            --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
            +nrm/apps/options=_kelvin_predict \
            dataset.predict.ncore_json_base_path="$BASE_PATH" \
            dataset.predict.ncore_json_list_path="$LST_FILE" \
            dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
            out_dir="$NRE_OUT/no_merge" \
            predict.primitive_merge.enabled=false \
            predict.render_video.enabled=false || NRE_FAIL=1
        if [[ $NRE_FAIL -eq 0 ]]; then
            bazel run //nre/nrm:run -- \
                --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
                +nrm/apps/options=_kelvin_predict \
                dataset.predict.ncore_json_base_path="$BASE_PATH" \
                dataset.predict.ncore_json_list_path="$LST_FILE" \
                dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
                out_dir="$NRE_OUT/merge" \
                predict.render_video.enabled=false || NRE_FAIL=1
        fi
        popd >/dev/null
    fi
} >> "$ERR_LOG" 2>&1

# 2) Standalone instant-nurec run ------------------------------------------------
# Inherits INSTANT_NUREC_FULL_PT from the parent env if set — needed when
# the HF auto-download path can't reach huggingface.co (proxy/auth issues).
if [[ $NRE_FAIL -eq 0 ]]; then
    {
        echo "===== $(date -Is) instant-nurec run =====" >&2
        pushd "$REPO_DIR" >/dev/null
        # Call run_inference.py via $VENV_PY directly — run.sh resolves
        # bare ``python`` from $PATH, which on this host picks up conda's
        # python (missing shortuuid). Equivalent direct-CLI form per README.
        "$VENV_PY" run_inference.py --ncore-path "$JSON_PATH" --output-dir "$INR_OUT/no_merge" --merge none || INR_FAIL=1
        if [[ $INR_FAIL -eq 0 ]]; then
            "$VENV_PY" run_inference.py --ncore-path "$JSON_PATH" --output-dir "$INR_OUT/merge" --merge frustum-ownership || INR_FAIL=1
        fi
        popd >/dev/null
    } >> "$ERR_LOG" 2>&1
fi

# 3) Locate the PLYs and run validate_parity -------------------------------------
MERGE_RC=-1
NM_RC=-1
MERGE_OUT_FILE=$SCRATCH/merge_validate.txt
NM_OUT_FILE=$SCRATCH/no_merge_validate.txt
: > "$MERGE_OUT_FILE"
: > "$NM_OUT_FILE"

if [[ $NRE_FAIL -eq 0 && $INR_FAIL -eq 0 ]]; then
    NRE_M_PLY=$(find "$NRE_OUT/merge"    -name '*.ply' | head -1)
    NRE_NM_DIR=$(find "$NRE_OUT/no_merge" -type d -name 'pai_*' | head -1)
    INR_M_PLY=$(find "$INR_OUT/merge"    -name '*.ply' | head -1)
    INR_NM_DIR=$(find "$INR_OUT/no_merge" -type d -name 'pai_*' | head -1)

    if [[ -z "$NRE_M_PLY" || -z "$INR_M_PLY" || -z "$NRE_NM_DIR" || -z "$INR_NM_DIR" ]]; then
        {
            echo "error: could not locate PLYs"
            echo "  NRE_M_PLY=$NRE_M_PLY"
            echo "  NRE_NM_DIR=$NRE_NM_DIR"
            echo "  INR_M_PLY=$INR_M_PLY"
            echo "  INR_NM_DIR=$INR_NM_DIR"
        } >> "$ERR_LOG"
    else
        "$VENV_PY" "$VALIDATE" merge    "$NRE_M_PLY"  "$INR_M_PLY"  >"$MERGE_OUT_FILE" 2>&1
        MERGE_RC=$?
        "$VENV_PY" "$VALIDATE" no_merge "$NRE_NM_DIR" "$INR_NM_DIR" >"$NM_OUT_FILE"    2>&1
        NM_RC=$?
    fi
fi

# 4) Aggregate the JSON report ---------------------------------------------------
"$VENV_PY" - <<PY > "$PARITY_JSON"
import datetime, json, pathlib

merge_out  = pathlib.Path("$MERGE_OUT_FILE").read_text() if pathlib.Path("$MERGE_OUT_FILE").exists() else ""
nm_out     = pathlib.Path("$NM_OUT_FILE").read_text()    if pathlib.Path("$NM_OUT_FILE").exists()    else ""

nre_fail   = bool(int("$NRE_FAIL"))
inr_fail   = bool(int("$INR_FAIL"))
merge_rc   = int("$MERGE_RC")
nm_rc      = int("$NM_RC")

def status(rc, ran):
    if not ran:                return "SKIPPED"
    if rc == 0:                return "PASS"
    return "FAIL"

ran = (not nre_fail) and (not inr_fail) and merge_rc != -1 and nm_rc != -1

report = {
    "dataset_json": "$JSON_PATH",
    "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
    "nre_commit":   "$NRE_COMMIT",
    "stages": {
        "nre_bazel":      "FAIL" if nre_fail else "PASS",
        "instant_nurec":  "SKIPPED" if nre_fail else ("FAIL" if inr_fail else "PASS"),
    },
    "modes": {
        "merge": {
            "status":    status(merge_rc, ran),
            "exit_code": merge_rc,
            "details":   merge_out.strip(),
        },
        "no_merge": {
            "status":    status(nm_rc, ran),
            "exit_code": nm_rc,
            "details":   nm_out.strip(),
        },
    },
    "overall": "PASS" if ran and merge_rc == 0 and nm_rc == 0 else "FAIL",
}
print(json.dumps(report, indent=2))
PY

# Keep errors.log only if it has content.
if [[ ! -s "$ERR_LOG" ]]; then
    rm -f "$ERR_LOG"
fi

echo "parity report: $PARITY_JSON"
[[ $(grep -c '"overall": "PASS"' "$PARITY_JSON") -ge 1 ]] && exit 0 || exit 1
