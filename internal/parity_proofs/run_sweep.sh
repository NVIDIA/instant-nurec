#!/bin/bash
# Per-commit quality + runtime sweep.
#
# Outputs go under /storage/projects/instant-nurec/.scratch/sweep/.

set -uo pipefail
REPO=/storage/projects/instant-nurec
SCRATCH=/storage/projects/instant-nurec/.scratch/sweep
BASE_NM="$REPO/baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb"
BASE_M="$REPO/baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.ply"
COMPARE="$REPO/benchmark/compare_clouds.py"
CU128="$REPO/.venv-c"

mkdir -p "$SCRATCH"

# Commits in chronological order; phase indicates how to run them.
COMMITS=(
    "2b48686:bazel:A.4_packed_ops"
    "fc23075:bazel:A.1_se3pose"
    "6b32da4:bazel:A.5_pose_calib"
    "037ed34:bazel:A.2_A.3_A.6_bundle"
    "efa4cc9:bazel:A.7_vren"
    "882c0d0:bazel:A.8_drop_libs"
    "c144996:bazel:B_lietorch_shim"
    "e177e4c:bazel:C_basic_flatten"
    "b33892f:bazel:B_torchvision_drop"
    "07c8b20:cu128:B.2_drop_bazel"
    "7ea7a65:cu128:single_pt_artifact"
    "HEAD:cu128:HEAD"
)

run_one() {
    local hash="$1" phase="$2" label="$3"
    local outdir="$SCRATCH/$label"
    local wt="$SCRATCH/wt_${label}"
    mkdir -p "$outdir"

    echo
    echo "##### $label  ($hash)  [$phase]"

    if [[ ! -d "$wt" ]]; then
        git -C "$REPO" worktree add "$wt" "$hash" 2>&1 | tail -1
    fi
    cd "$wt"

    local nm="$outdir/no_merge"
    local m="$outdir/merge"
    mkdir -p "$nm" "$m"
    local pt="$outdir/kelvin_full.pt"

    if [[ "$phase" == "bazel" ]]; then
        local t0=$(date +%s.%N)
        INSTANT_NUREC_FULL_PT="$pt" bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 --output-dir "$nm" --merge none --log-level INFO 2>&1 | tail -3 > "$outdir/no_merge.log"
        local t1=$(date +%s.%N)
        INSTANT_NUREC_FULL_PT="$pt" bazel run //instant_nurec:run -- --ncore-path /storage/data/nurec/ncorev4 --output-dir "$m" --merge frustum-ownership --log-level INFO 2>&1 | tail -3 > "$outdir/merge.log"
        local t2=$(date +%s.%N)
    else
        "$CU128/bin/pip" uninstall -y -q instant_nurec 2>&1 | tail >/dev/null
        "$CU128/bin/pip" install --quiet -e . 2>&1 | tail -3 >> "$outdir/install.log"
        local t0=$(date +%s.%N)
        INSTANT_NUREC_FULL_PT="$pt" "$CU128/bin/python" run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir "$nm" --merge none 2>&1 | tail -3 > "$outdir/no_merge.log"
        local t1=$(date +%s.%N)
        INSTANT_NUREC_FULL_PT="$pt" "$CU128/bin/python" run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir "$m" --merge frustum-ownership 2>&1 | tail -3 > "$outdir/merge.log"
        local t2=$(date +%s.%N)
    fi
    python3 -c "print(f'no_merge_wall_s={$t1 - $t0:.2f}\nmerge_wall_s={$t2 - $t1:.2f}')" > "$outdir/runtime.txt"

    local nm_proposed=$(echo "$nm"/*/ply/pai_*)
    local m_proposed=$(echo "$m"/*/ply/*/pai_*.ply)

    "$CU128/bin/python" "$COMPARE" --no-merge "$BASE_NM" "$nm_proposed" > "$outdir/metrics_no_merge.txt" 2>&1
    "$CU128/bin/python" "$COMPARE" "$BASE_M" "$m_proposed" > "$outdir/metrics_merge.txt" 2>&1

    cat "$outdir/runtime.txt"
    grep -E "vertex counts|Chamfer \(½ sum|F-score @ τ:|τ=0.01:|τ=0.10:" "$outdir/metrics_no_merge.txt" | head -8
    echo "---"
    grep -E "vertex counts|Chamfer \(½ sum|F-score @ τ:|τ=0.01:|τ=0.10:" "$outdir/metrics_merge.txt" | head -8

    # Reclaim disk: drop the .pt and the heavy proposed PLYs after metrics.
    rm -f "$pt"
    rm -rf "$nm" "$m"
}

for spec in "${COMMITS[@]}"; do
    IFS=":" read -r hash phase label <<< "$spec"
    run_one "$hash" "$phase" "$label" || echo "  ERROR running $label"
done
