#!/bin/bash
# Optimize and evaluate one scene.
# usage: bash scripts/run_scene.sh <config.yaml> <data_name> <scene> [extra train.py args]
# env:   DATA_ROOT (default ./data), OUTPUT_ROOT (default ./output), EVAL (default 1), CHAMFER_ROOT (optional)
set -eo pipefail
config=$1; data_name=$2; seq=$3; shift 3

DATA_ROOT=${DATA_ROOT:-./data}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output}
EVAL=${EVAL:-1}
iterations=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['iterations'])" "${config}")
# an --iterations override among the extra arguments wins
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --iterations) iterations=${args[i+1]} ;;
        --iterations=*) iterations=${args[i]#--iterations=} ;;
    esac
done

source_path=${DATA_ROOT}/${data_name}/train/${seq}/cropped
model_path=${OUTPUT_ROOT}/${data_name}/${seq}
mkdir -p "${model_path}"

python train.py --config "${config}" -s "${source_path}" -m "${model_path}" "$@" 2>&1 | tee "${model_path}/log.txt"

if [ "${EVAL}" = "1" ]; then
    # novel views: PSNR / SSIM / LPIPS, geometry: chamfer / F-score
    eval_args=()
    if [ -n "${CHAMFER_ROOT:-}" ]; then
        eval_args+=(--chamfer_root "${CHAMFER_ROOT}")
    fi
    python render_eval.py -m "${model_path}" --iteration "${iterations}"
    python ../evaluation/evaluate.py artemis -m "${model_path}" --iteration "${iterations}" "${eval_args[@]}"
fi
