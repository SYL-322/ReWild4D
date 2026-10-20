#!/bin/bash
# DAVIS animal sequences (8), config: configs/animal.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
data_name="DAVIS_processed"
scene_list=$(python ../evaluation/protocols/datasets.py DAVIS)
mapfile -t SEQUENCES <<< "$scene_list"

for seq in "${SEQUENCES[@]}"; do
    bash scripts/run_scene.sh configs/animal.yaml "${data_name}" "${seq}"
done

if [ "${EVAL:-1}" = "1" ]; then
    python ../evaluation/summarize.py real --output_root "${OUTPUT_ROOT:-./output}" --data_name "${data_name}" --iteration 20000 --scenes "${SEQUENCES[@]}"
fi
