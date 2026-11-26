#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Step 1 / prompt-sensitivity: official OSDFace inference. Default: --split val, variant $PROMPT_VARIANT (normal|shuffled|zero_code|zero_embed).
# Run inside an already-allocated session:  bash jobs/infer_official.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=4:00:00
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/infer_official_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
if [ "$#" -eq 0 ]; then set -- --split val --prompt_variant "${PROMPT_VARIANT:-normal}"; fi
python -m hqprompt.infer_official "$@"
