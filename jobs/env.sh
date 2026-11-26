# Source this (do not execute): source jobs/env.sh
# conda's activate scripts reference unset variables, so relax `set -u` around them.
_hqp_had_u=0; case $- in *u*) _hqp_had_u=1; set +u;; esac
# `module` is a shell function that non-login shells (qsub, bash script.sh) may lack
type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
module load miniconda
eval "$(conda shell.bash hook)"
conda activate /projectnb/cs599bg/manaswi/craft/hqprompt/env
[ "$_hqp_had_u" = 1 ] && set -u
unset _hqp_had_u

export HQP_ROOT="${HQP_ROOT:-/projectnb/cs599bg/manaswi/craft/hqprompt}"
# Code checkout = parent of jobs/ unless the caller already set it.
export HQP_CODE="${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$HQP_CODE${PYTHONPATH:+:$PYTHONPATH}"

# Dataset / checkpoint locations: same defaults as hqprompt/paths.py
_hqp_scc="/projectnb/cs599bg/manaswi/craft"
export HQP_PRETRAINED="${HQP_PRETRAINED:-$_hqp_scc/ivc_checkpoints/osdface_official/pretrained}"
export HQP_FFHQ_HQ="${HQP_FFHQ_HQ:-$_hqp_scc/data/train/images512x512}"
export HQP_FFHQ_LQ="${HQP_FFHQ_LQ:-$_hqp_scc/data/train/LQ_images_512x512}"
export HQP_CELEBA_HQ="${HQP_CELEBA_HQ:-$_hqp_scc/data/test/CelebA/CelebA_Validation/celeba_512_validation}"
export HQP_CELEBA_LQ="${HQP_CELEBA_LQ:-$_hqp_scc/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2}"
export HQP_LFW_LQ="${HQP_LFW_LQ:-$_hqp_scc/data/test/LFW/cropped_faces}"
unset _hqp_scc

# Model downloads / caches stay inside the only write area (never $HOME)
export HQP_MODEL_CACHE="${HQP_MODEL_CACHE:-$HQP_ROOT/model_cache}"
export HF_HOME="$HQP_MODEL_CACHE/hf"
export TORCH_HOME="$HQP_MODEL_CACHE/torch"
export XDG_CACHE_HOME="$HQP_MODEL_CACHE/xdg"
export MPLCONFIGDIR="$HQP_MODEL_CACHE/mpl"
export PIP_CACHE_DIR="$HQP_MODEL_CACHE/pip"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$HQP_MODEL_CACHE/nv"
mkdir -p "$CUDA_CACHE_PATH" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$HQP_ROOT/logs"
