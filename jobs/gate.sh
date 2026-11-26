#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Step 4 gate (CPU, bootstrap only). Env ROW4/ROW2/ROW3 = score output dirs (default $HQP_ROOT/eval/row{4,2,3}).
# Run inside an already-allocated session:  bash jobs/gate.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=1:00:00
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/gate_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
E="$HQP_ROOT/eval"
python -m hqprompt.evaluate gate --row4 "${ROW4:-$E/row4}" --row2 "${ROW2:-$E/row2}" --row3 "${ROW3:-$E/row3}" \
    --n_boot 10000 --seed 0 --out "$E/gate.json" --md "$E/gate.md" "$@"
