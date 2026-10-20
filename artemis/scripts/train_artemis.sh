#!/bin/bash
# Artemis sequences (24), config: configs/artemis.yaml
# data: Artemis_processed/train/; evaluation: Artemis_processed/eval/{views,geometry}/, see README
set -euo pipefail
cd "$(dirname "$0")/.."
data_name="Artemis_processed"
scene_list=$(python ../evaluation/protocols/datasets.py Artemis)
mapfile -t SEQUENCES <<< "$scene_list"

for seq in "${SEQUENCES[@]}"; do
    bash scripts/run_scene.sh configs/artemis.yaml "${data_name}" "${seq}"
done

if [ "${EVAL:-1}" = "1" ]; then
    python ../evaluation/summarize.py artemis --output_root "${OUTPUT_ROOT:-./output}" --data_name "${data_name}" --iteration 20000 --scenes "${SEQUENCES[@]}"
fi
