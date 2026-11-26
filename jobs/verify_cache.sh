#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Step 2b: re-compute 100 random cache rows and compare (run in the same GPU session type as the build so index ties resolve identically; exit 1 on mismatch).
# Run inside an already-allocated session:  bash jobs/verify_cache.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=4:00:00
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/verify_cache_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
if [ "$#" -eq 0 ]; then set -- --verify --n 100 --seed 1; fi
python -m hqprompt.build_cache "$@"
