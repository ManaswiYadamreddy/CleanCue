#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Step 2: build the code/latent/teacher cache. Defaults: --split all --stage all; HQ_INPUT_MODE (single|double) must come from checks/c1.json.
# Run inside an already-allocated session:  bash jobs/build_cache.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/build_cache_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
if [ "$#" -eq 0 ]; then set -- --split "${SPLIT:-all}" --stage "${STAGE:-all}" --hq_input_mode "${HQ_INPUT_MODE:?set HQ_INPUT_MODE to recommended_hq_input_mode from c1.json}"; fi
python -m hqprompt.build_cache "$@"
