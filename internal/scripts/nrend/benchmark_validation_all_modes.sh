#!/bin/bash

# ALL MODES
# ALL_MODES=(0 1 2 4 6 7 9)
ALL_MODES=(0 2 7) # HD MODES ONLY

SCRIPT_DIR=$(dirname "$0")
for i in "${ALL_MODES[@]}"; do
  CMD="${SCRIPT_DIR}/benchmark_validation.sh $1 $i $2 $3"
  echo ${CMD}
  ${CMD}
done
