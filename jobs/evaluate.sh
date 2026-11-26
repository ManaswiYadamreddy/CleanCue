#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Score a restored-image directory, e.g.:  bash jobs/evaluate.sh score --restored_dir D --split val --metrics lpips dists deg lmd musiq --out_dir O  (also compare|diff|table).
# Run inside an already-allocated session:  bash jobs/evaluate.sh [ARGS]   (ARGS are passed through)
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
exec > >(tee -a "$HQP_ROOT/logs/evaluate_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
python -m hqprompt.evaluate "$@"
