#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Step 4/6: fine-tune one row. Env: PROMPT (required: lq|null|hq_true|hq_pred), SEED (default 0). Add --hq_pred_run DIR for hq_pred.
# Run inside an already-allocated session:  bash jobs/finetune.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/finetune_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
python -m hqprompt.finetune --prompt "${PROMPT:?set PROMPT=lq|null|hq_true|hq_pred}" --seed "${SEED:-0}" --lambda_3d 0 "$@"
