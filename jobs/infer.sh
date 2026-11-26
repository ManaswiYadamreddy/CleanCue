#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Inference for a fine-tuned run, e.g.:  bash jobs/infer.sh --run_dir $HQP_ROOT/runs/lq_l3d0_s0 --codes lq --split val
# Run inside an already-allocated session:  bash jobs/infer.sh [ARGS]   (ARGS are passed through)
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
exec > >(tee -a "$HQP_ROOT/logs/infer_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
python -m hqprompt.infer "$@"
