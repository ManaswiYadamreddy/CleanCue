#!/bin/bash
# USER APPROVAL REQUIRED: do not run this script until the user has approved it.
# Download pyiqa, facexlib ArcFace and face_alignment weights into $HQP_ROOT/model_cache (needs internet on this node).
# MUST run inside a GPU session: facexlib builds the recognition model on cuda.
# Run inside an already-allocated session:  bash jobs/prefetch_models.sh [ARGS]   (ARGS are passed through)
# Optional qsub headers below are inert under bash. No GPU type is named: add
# '-l gpu_type=...' at submit time yourself. Under qsub export HQP_CODE first, because
# BASH_SOURCE points at the spool directory.
#$ -P cs599bg
#$ -l h_rt=1:00:00
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
set -euo pipefail
HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HQP_CODE
source "$HQP_CODE/jobs/env.sh"
exec > >(tee -a "$HQP_ROOT/logs/prefetch_models_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$HQP_CODE"
nvidia-smi
python - <<PY
import torch
from hqprompt.paths import get_paths
paths = get_paths()
import pyiqa
for m in ("lpips", "dists", "musiq", "niqe", "clipiqa", "maniqa", "fid"):
    pyiqa.create_metric(m, device=torch.device("cpu"))
    print("ok pyiqa", m, flush=True)
from facexlib.recognition import init_recognition_model
init_recognition_model("arcface", device=torch.device("cpu"), model_rootpath=paths.facexlib_dir)
print("ok facexlib arcface", flush=True)
import face_alignment
face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False, compile=False)
print("ok face_alignment", flush=True)
PY
