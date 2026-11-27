#!/bin/bash
# =============================================================================
# USER APPROVAL REQUIRED: do not run until the user has approved this script.
# setup_env.sh — conda env for hqprompt (Proposal 1) on BU SCC.
# Ported from CRAFT/setup2.sh + requirements.txt (same pins).
# Only change: the env lives inside the hqprompt write area.
#
# Usage (any node, GPU not needed):
#   bash jobs/setup_env.sh
# Afterwards:
#   source jobs/env.sh
# =============================================================================
set -euo pipefail

ENV_DIR="/projectnb/cs599bg/manaswi/CleanCue/env"
PYTHON_VERSION="3.10"
BASE="/projectnb/cs599bg/manaswi/CleanCue/work"
# Keep pip / conda caches out of $HOME (writes are only allowed under ${BASE})
mkdir -p "${BASE}/model_cache/pip" "${BASE}/model_cache/conda_pkgs" "${BASE}/logs"
export PIP_CACHE_DIR="${BASE}/model_cache/pip"
export CONDA_PKGS_DIRS="${BASE}/model_cache/conda_pkgs"

if [ -e "${ENV_DIR}" ]; then
    echo "ERROR: ${ENV_DIR} already exists. Remove it yourself if you want a fresh env." >&2
    exit 1
fi

# `module` is a shell function that non-login shells (qsub, bash script.sh) may lack
type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
module load miniconda
# conda's activate scripts reference unset vars; relax -u around them
set +u
eval "$(conda shell.bash hook)"

echo "[1/5] Creating conda env at ${ENV_DIR}"
conda create --prefix "${ENV_DIR}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_DIR}"
set -u
echo "Active env: ${CONDA_PREFIX}"

echo "[2/5] torch 2.4.0+cu121"
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/5] remaining dependencies"
# torch/torchvision listed again so the resolver keeps 2.4.0 instead of pulling a newer CUDA build
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "setuptools<70" pillow opencv-python imageio "numpy<2" scipy \
    pyiqa==0.1.12 facexlib lpips face-alignment==1.5.0 scikit-image pytest \
    omegaconf einops ftfy sentencepiece PyYAML tqdm tensorboard

echo "[4/5] pinned core (installed after pyiqa so its deps cannot move them)"
pip install \
    diffusers==0.27.2 \
    transformers==4.37.2 \
    accelerate==0.31.0 \
    peft==0.12.0 \
    tokenizers==0.15.2 \
    huggingface_hub==0.24.7 \
    safetensors

# pyiqa / openai-clip import pkg_resources, which setuptools>=81 removed
pip install "setuptools<70"

# Re-pin torch in case anything above upgraded it
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

echo "[5/5] verify"
python - <<'EOF'
import sys
expected = {"torch": "2.4.0", "diffusers": "0.27.2", "transformers": "4.37.2",
            "peft": "0.12.0", "accelerate": "0.31.0", "huggingface_hub": "0.24.7"}
ok = True
for name in ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
             "huggingface_hub", "safetensors", "pyiqa", "facexlib", "face_alignment", "lpips",
             "skimage", "pytest", "cv2", "scipy"]:
    try:
        v = getattr(__import__(name), "__version__", "ok")
        bad = name in expected and not v.startswith(expected[name])
        ok &= not bad
        print(f"  {'✗' if bad else '✓'}  {name:<16} {v}" + (f"  (expected {expected[name]})" if bad else ""))
    except Exception as e:
        ok = False
        print(f"  ✗  {name:<16} FAILED — {e}")
import math; math.prod  # SCC default python lacked this
import pkg_resources  # needed by pyiqa; fails if setuptools is too new
print("All imports OK" if ok else "Some checks FAILED"); sys.exit(0 if ok else 1)
EOF

echo "Done. Activate with: source jobs/env.sh"
