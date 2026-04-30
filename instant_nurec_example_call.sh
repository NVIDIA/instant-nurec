#!/bin/bash -ex

OUT_DIR=$1

# instant-nurec - no merging
/storage/projects/instant-nurec/.venv/bin/python run_inference.py \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir "${OUT_DIR}/no_merge" \
    --merge none \
    --log-level INFO

# instant-nurec - frustum-ownership
/storage/projects/instant-nurec/.venv/bin/python run_inference.py \
    --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir "${OUT_DIR}/merge" \
    --merge frustum-ownership \
    --log-level INFO