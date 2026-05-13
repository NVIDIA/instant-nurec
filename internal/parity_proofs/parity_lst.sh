#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pick N random uuids that passed the dataset-load smoke (Stage B), write
# them into a single ncorev4 LST, then run:
#   1. NRE @ a54a6af bazel once on that LST  (full 50-clip predict pass)
#   2. instant-nurec once on the same LST    (full 50-clip predict pass)
#   3. validate_parity.py per (chunk0, chunk1, merge) pair across all
#      50 sequences against baselines/original_baseline.
#
# Versus parity_subset.sh's one-clip-at-a-time loop, this collapses
# 50 bazel daemon startups + 50 model loads + 50 dataset constructions
# into ~1 each, saving the bulk of the per-clip overhead.
#
# Usage:
#   parity_lst.sh <clips_dir> <dataset_load_report.json> <output_dir> [N=50] [SEED=42]

set -uo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "usage: $0 <clips_dir> <dataset_load_report.json> <output_dir> [N=50] [SEED=42]" >&2
    exit 64
fi

CLIPS_DIR=$(cd "$1" && pwd)
REPORT_JSON=$2
OUT_DIR=$3
N=${4:-50}
SEED=${5:-42}

NRE_REPO=/storage/projects/nre
NRE_COMMIT=a54a6af0a177beabd01fe37e398c45be165a270f
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV_PY=$REPO_DIR/.venv/bin/python
VALIDATE=$REPO_DIR/internal/benchmark/validate_parity.py

if [[ ! -d "$CLIPS_DIR" ]]; then
    echo "error: clips_dir '$CLIPS_DIR' is not a directory" >&2; exit 65
fi
if [[ ! -f "$REPORT_JSON" ]]; then
    echo "error: report '$REPORT_JSON' does not exist" >&2; exit 65
fi
if [[ ! -x "$VENV_PY" ]]; then
    echo "error: $VENV_PY missing — run ./setup.sh first" >&2; exit 66
fi

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
LST_FILE=$OUT_DIR/subset.lst
NRE_OUT=$OUT_DIR/nre
INSTANT_OUT=$OUT_DIR/instant
ERR_LOG=$OUT_DIR/errors.log
SUMMARY_JSON=$OUT_DIR/parity_results.json
mkdir -p "$NRE_OUT" "$INSTANT_OUT"
: > "$ERR_LOG"

echo "===== $(date -Is) Phase 0: pick $N random uuids (seed=$SEED) =====" | tee -a "$ERR_LOG"
"$VENV_PY" - "$REPORT_JSON" "$CLIPS_DIR" "$N" "$SEED" "$LST_FILE" <<'PY'
import json, random, sys
from pathlib import Path
report_path, clips_dir, n_arg, seed_arg, out_path = sys.argv[1:6]
n = int(n_arg)
with open(report_path) as f:
    report = json.load(f)
passed = report.get("passed_uuids", [])
if not passed:
    sys.exit("error: report has empty passed_uuids[]")
if n > len(passed):
    print(f"warning: requested {n} > available {len(passed)}; using all",
          file=sys.stderr)
    n = len(passed)
chosen = random.Random(int(seed_arg)).sample(passed, n)
clips_dir = Path(clips_dir)
with open(out_path, "w") as f:
    for uuid in chosen:
        f.write(str(clips_dir / uuid / f"pai_{uuid}.json") + "\n")
print(f"wrote {len(chosen)} uuids -> {out_path}")
PY
if [[ ! -s "$LST_FILE" ]]; then
    echo "error: LST file empty" >&2; exit 1
fi
LINE_COUNT=$(wc -l < "$LST_FILE")
echo "LST has $LINE_COUNT entries (absolute paths)" | tee -a "$ERR_LOG"

# Common base for NRE (which prepends base_path even when entries are
# absolute — os.path.join treats the absolute entry as final).
NRE_BASE=$(dirname "$CLIPS_DIR")/

# Decide which NRE checkout to use. If the repo is already at $NRE_COMMIT,
# skip the destructive checkout entirely.
NRE_ALREADY_AT_COMMIT=0
NRE_CUR_SHA=$(git -C "$NRE_REPO" rev-parse HEAD 2>/dev/null || echo "")
NRE_TARGET_SHA=$(git -C "$NRE_REPO" rev-parse "$NRE_COMMIT" 2>/dev/null || echo "")
if [[ -n "$NRE_CUR_SHA" && "$NRE_CUR_SHA" == "$NRE_TARGET_SHA" ]]; then
    NRE_ALREADY_AT_COMMIT=1
fi
NRE_PREV_HEAD=""
cleanup() {
    if [[ -n "$NRE_PREV_HEAD" ]]; then
        git -C "$NRE_REPO" checkout --quiet "$NRE_PREV_HEAD" 2>/dev/null || true
    fi
}
trap cleanup EXIT

NRE_FAIL=0
INSTANT_FAIL=0

echo "===== $(date -Is) Phase 1: NRE bazel run on $LINE_COUNT clips =====" | tee -a "$ERR_LOG"
{
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
            dataset.predict.ncore_json_base_path="$NRE_BASE" \
            dataset.predict.ncore_json_list_path="$LST_FILE" \
            dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
            out_dir="$NRE_OUT/no_merge" \
            predict.primitive_merge.enabled=false \
            predict.render_video.enabled=false || NRE_FAIL=1
        if [[ $NRE_FAIL -eq 0 ]]; then
            bazel run //nre/nrm:run -- \
                --config-name=configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml \
                +nrm/apps/options=_kelvin_predict \
                dataset.predict.ncore_json_base_path="$NRE_BASE" \
                dataset.predict.ncore_json_list_path="$LST_FILE" \
                dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov \
                out_dir="$NRE_OUT/merge" \
                predict.render_video.enabled=false || NRE_FAIL=1
        fi
        popd >/dev/null
    fi
} >> "$ERR_LOG" 2>&1
echo "NRE phase done (fail=$NRE_FAIL)" | tee -a "$ERR_LOG"

if [[ $NRE_FAIL -eq 0 ]]; then
    echo "===== $(date -Is) Phase 2: instant-nurec run on $LINE_COUNT clips =====" | tee -a "$ERR_LOG"
    {
        pushd "$REPO_DIR" >/dev/null
        "$VENV_PY" run_inference.py --ncore-path "$LST_FILE" --output-dir "$INSTANT_OUT/no_merge" --merge none || INSTANT_FAIL=1
        if [[ $INSTANT_FAIL -eq 0 ]]; then
            "$VENV_PY" run_inference.py --ncore-path "$LST_FILE" --output-dir "$INSTANT_OUT/merge" --merge frustum-ownership || INSTANT_FAIL=1
        fi
        popd >/dev/null
    } >> "$ERR_LOG" 2>&1
    echo "instant-nurec phase done (fail=$INSTANT_FAIL)" | tee -a "$ERR_LOG"
fi

if [[ $NRE_FAIL -eq 0 && $INSTANT_FAIL -eq 0 ]]; then
    echo "===== $(date -Is) Phase 3: validate_parity across all pairs =====" | tee -a "$ERR_LOG"
    "$VENV_PY" - "$NRE_OUT" "$INSTANT_OUT" "$LST_FILE" "$VALIDATE" "$SUMMARY_JSON" <<'PY' 2>&1 | tee -a "$ERR_LOG"
import json, subprocess, sys
from pathlib import Path

nre_out, instant_out, lst_file, validate_script, summary_path = sys.argv[1:6]

def sequence_id_from_uuid(uuid: str) -> str:
    return f"pai_{uuid}"

# Pull the uuid list from the LST. Each line is an absolute path
# .../clips/<UUID>/pai_<UUID>.json — derive UUID from the basename.
uuids = []
for line in Path(lst_file).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = Path(line).stem  # pai_<UUID>
    uuids.append(name.removeprefix("pai_"))

# Find NRE no_merge dir, instant no_merge dir, NRE merge dir, instant merge dir.
def first_run_dir(parent: Path, kind: str) -> Path | None:
    candidate = parent / kind
    if not candidate.is_dir():
        return None
    # exactly one run subdir is expected; pick the most recently modified
    subs = [p for p in candidate.iterdir() if p.is_dir()]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)

nre_nm_run = first_run_dir(Path(nre_out), "no_merge")
nre_m_run = first_run_dir(Path(nre_out), "merge")
inr_nm_run = first_run_dir(Path(instant_out), "no_merge")
inr_m_run = first_run_dir(Path(instant_out), "merge")
if not all([nre_nm_run, nre_m_run, inr_nm_run, inr_m_run]):
    print("error: could not locate run dirs", file=sys.stderr)
    print({"nre_nm": nre_nm_run, "nre_m": nre_m_run,
           "inr_nm": inr_nm_run, "inr_m": inr_m_run}, file=sys.stderr)
    sys.exit(1)

per_clip = []
n_pass = n_fail = n_missing = 0
for uuid in uuids:
    seq = sequence_id_from_uuid(uuid)
    nre_nm_dir = nre_nm_run / "ply" / seq
    inr_nm_dir = inr_nm_run / "ply" / seq
    nre_m_ply = nre_m_run / "ply" / seq / f"{seq}.ply"
    inr_m_ply = inr_m_run / "ply" / seq / f"{seq}.ply"

    record = {"uuid": uuid, "modes": {}}
    for mode, args in (
        ("no_merge", ["no_merge", str(nre_nm_dir), str(inr_nm_dir)]),
        ("merge",    ["merge",    str(nre_m_ply),  str(inr_m_ply)]),
    ):
        if mode == "no_merge" and not (nre_nm_dir.is_dir() and inr_nm_dir.is_dir()):
            record["modes"][mode] = {"status": "MISSING", "exit_code": -1, "details": ""}
            n_missing += 1
            continue
        if mode == "merge" and not (nre_m_ply.is_file() and inr_m_ply.is_file()):
            record["modes"][mode] = {"status": "MISSING", "exit_code": -1, "details": ""}
            n_missing += 1
            continue
        proc = subprocess.run(
            [sys.executable, validate_script, *args],
            capture_output=True, text=True,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        record["modes"][mode] = {
            "status": status, "exit_code": proc.returncode,
            "details": (proc.stdout + proc.stderr).strip()[:600],
        }
        if status == "PASS": n_pass += 1
        else: n_fail += 1
    overall = "PASS" if all(m["status"] == "PASS" for m in record["modes"].values()) else "FAIL"
    record["overall"] = overall
    per_clip.append(record)

summary = {
    "n_clips": len(uuids),
    "n_pair_pass": n_pass,
    "n_pair_fail": n_fail,
    "n_pair_missing": n_missing,
    "n_clip_overall_pass": sum(1 for r in per_clip if r["overall"] == "PASS"),
    "n_clip_overall_fail": sum(1 for r in per_clip if r["overall"] != "PASS"),
    "per_clip": per_clip,
}
Path(summary_path).write_text(json.dumps(summary, indent=2))
print(f"summary -> {summary_path}")
print(f"  pairs PASS={n_pass} FAIL={n_fail} MISSING={n_missing}")
print(f"  clip overall PASS={summary['n_clip_overall_pass']} "
      f"FAIL={summary['n_clip_overall_fail']}")
PY
fi

echo "===== $(date -Is) DONE =====" | tee -a "$ERR_LOG"
[[ ! -s "$ERR_LOG" ]] && rm -f "$ERR_LOG" || true
echo "outputs in $OUT_DIR"
[[ $NRE_FAIL -eq 0 && $INSTANT_FAIL -eq 0 ]] && exit 0 || exit 1
