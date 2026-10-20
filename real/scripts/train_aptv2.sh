#!/bin/bash
# APTv2 clips (68), config: configs/aptv2.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
data_name="APTv2"
scene_list=$(python ../evaluation/protocols/datasets.py APTv2)
mapfile -t SEQUENCES <<< "$scene_list"

for seq in "${SEQUENCES[@]}"; do
    bash scripts/run_scene.sh configs/aptv2.yaml "${data_name}" "${seq}"
done

if [ "${EVAL:-1}" = "1" ]; then
    python ../evaluation/summarize.py real --output_root "${OUTPUT_ROOT:-./output}" --data_name "${data_name}" --iteration 10000 --scenes "${SEQUENCES[@]}"
fi
