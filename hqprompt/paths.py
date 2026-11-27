"""Single source of truth for every filesystem path, plus write/load guards.

Precedence for each path: explicit override (non-None) > environment variable > SCC default.
Every file write in hqprompt must go through `assert_writable`; every .ckpt/.pth load through
`safe_torch_load` (plan.md section 2, rules 1-2 and 7).
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, fields

import torch

log = logging.getLogger(__name__)

# Read-only inputs (weights, datasets) live in the old CRAFT area.
SCC_BASE = "/projectnb/cs599bg/manaswi/craft"
# The only write area: inside the CleanCue clone, git-ignored.
HQP_ROOT_DEFAULT = "/projectnb/cs599bg/manaswi/CleanCue/work"

# field name -> (env var, default; "{out_root}" / "{SCC_BASE}" are substituted)
_SPEC = {
    "out_root":    ("HQP_ROOT",        HQP_ROOT_DEFAULT),
    "pretrained":  ("HQP_PRETRAINED",  "{SCC_BASE}/ivc_checkpoints/osdface_official/pretrained"),
    "ffhq_hq":     ("HQP_FFHQ_HQ",     "{SCC_BASE}/data/train/images512x512"),
    "ffhq_lq":     ("HQP_FFHQ_LQ",     "{SCC_BASE}/data/train/LQ_images_512x512"),
    "celeba_hq":   ("HQP_CELEBA_HQ",   "{SCC_BASE}/data/test/CelebA/CelebA_Validation/celeba_512_validation"),
    # SELF-GENERATED LQ set, not the official DAEFR CelebA-Test LQ. Same filenames as celeba_hq.
    "celeba_lq":   ("HQP_CELEBA_LQ",   "{SCC_BASE}/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2"),
    "lfw_lq":      ("HQP_LFW_LQ",      "{SCC_BASE}/data/test/LFW/cropped_faces"),
    "model_cache": ("HQP_MODEL_CACHE", "{out_root}/model_cache"),
}


@dataclass(frozen=True)
class Paths:
    out_root: str
    pretrained: str
    ffhq_hq: str
    ffhq_lq: str
    celeba_hq: str
    celeba_lq: str
    lfw_lq: str
    model_cache: str

    # ---- weights (read-only) ----
    @property
    def associate_ckpt(self) -> str:
        return f"{self.pretrained}/associate_2.ckpt"

    @property
    def lora(self) -> str:
        return f"{self.pretrained}/pytorch_lora_weights.safetensors"

    @property
    def embedding_change(self) -> str:
        return f"{self.pretrained}/embedding_change_weights.pth"

    @property
    def sd21(self) -> str:
        return f"{self.pretrained}/sd21"

    # ---- outputs (writable, under out_root) ----
    @property
    def splits_json(self) -> str:
        return f"{self.out_root}/splits.json"

    @property
    def cache_dir(self) -> str:
        return f"{self.out_root}/cache"

    @property
    def runs_dir(self) -> str:
        return f"{self.out_root}/runs"

    @property
    def predictor_dir(self) -> str:
        return f"{self.out_root}/predictor"

    @property
    def outputs_dir(self) -> str:
        return f"{self.out_root}/outputs"

    @property
    def checks_dir(self) -> str:
        return f"{self.out_root}/checks"

    @property
    def eval_dir(self) -> str:
        return f"{self.out_root}/eval"

    @property
    def logs_dir(self) -> str:
        return f"{self.out_root}/logs"

    @property
    def facexlib_dir(self) -> str:
        return f"{self.model_cache}/facexlib"


def get_paths(**overrides) -> Paths:
    unknown = set(overrides) - set(_SPEC)
    if unknown:
        raise TypeError(f"unknown path overrides: {sorted(unknown)}")
    vals: dict[str, str] = {}
    # out_root first, because model_cache's default depends on it
    for name in ["out_root"] + [n for n in _SPEC if n != "out_root"]:
        env, default = _SPEC[name]
        v = overrides.get(name)
        if v is None:
            v = os.environ.get(env)
        if v is None:
            v = default.format(SCC_BASE=SCC_BASE, out_root=vals.get("out_root", ""))
        vals[name] = os.path.abspath(os.path.expanduser(v))
    return Paths(**vals)


def add_path_args(parser) -> None:
    g = parser.add_argument_group("paths (default: env var, then SCC default)")
    for f in fields(Paths):
        g.add_argument(f"--{f.name}", default=None, help=f"env {_SPEC[f.name][0]}")


def paths_from_args(args) -> Paths:
    return get_paths(**{f.name: getattr(args, f.name, None) for f in fields(Paths)})


def _under(child: str, parent: str) -> bool:
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def assert_writable(path: str, paths: Paths) -> str:
    """Return realpath(path) if it is inside paths.out_root, else raise PermissionError."""
    rp = os.path.realpath(path)
    if not _under(rp, paths.out_root):
        raise PermissionError(f"refusing to write {path!r} (resolves to {rp!r}); "
                              f"only {os.path.realpath(paths.out_root)!r} is writable")
    return rp


def safe_torch_load(path: str, paths: Paths, map_location="cpu"):
    """torch.load restricted to the two allowlisted pretrained files (plan rule 7)."""
    rp = os.path.realpath(path)
    allowed = {os.path.realpath(paths.associate_ckpt), os.path.realpath(paths.embedding_change)}
    if rp not in allowed:
        raise PermissionError(f"torch.load of {path!r} not allowlisted; allowed: {sorted(allowed)}")
    try:
        return torch.load(rp, map_location=map_location, weights_only=True)
    except Exception as e:  # Lightning 1.0.8 ckpts may contain non-tensor pickles
        log.warning("weights_only=True failed for %s (%s: %s); retrying weights_only=False "
                    "(file is allowlisted)", rp, type(e).__name__, str(e)[:200])
        return torch.load(rp, map_location=map_location, weights_only=False)


def file_fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {"path": os.path.realpath(path), "size": st.st_size, "mtime": st.st_mtime}


def weights_manifest(paths: Paths) -> dict[str, dict]:
    out = {
        "associate_2.ckpt": file_fingerprint(paths.associate_ckpt),
        "lora": file_fingerprint(paths.lora),
        "embedding_change": file_fingerprint(paths.embedding_change),
        "scheduler": file_fingerprint(f"{paths.sd21}/scheduler/scheduler_config.json"),
    }
    for sub in ("unet", "vae"):
        files = sorted(glob.glob(f"{paths.sd21}/{sub}/*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors in {paths.sd21}/{sub}")
        for i, f in enumerate(files):
            out[sub if i == 0 else f"{sub}_{i}"] = file_fingerprint(f)
    return out
