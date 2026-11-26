"""Small shared helpers: seeding, numerics, atomic JSON, hashing, provenance."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import random
import tempfile

import numpy as np
import torch

from hqprompt.paths import Paths, assert_writable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_numerics(deterministic: bool = True, tf32: bool = False) -> None:
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def require_bf16(device) -> None:
    """bf16 autocast on CUDA needs an Ampere+ GPU (compute capability >= 8); no-op on CPU."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("bf16 requested but CUDA is not available")
    if torch.cuda.get_device_capability(dev)[0] < 8:
        raise RuntimeError(f"bf16 needs an Ampere+ GPU (A100/A40/L40S/H100); got "
                           f"{torch.cuda.get_device_name(dev)}; request a different GPU or pass --amp none")


def get_device(dry_run: bool) -> torch.device:
    if dry_run:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run inside a GPU session, or pass --dry_run.")
    return torch.device("cuda")


def atomic_write_json(path: str, obj, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(rp), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, rp)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_list(xs) -> str:
    return sha256_bytes(json.dumps(xs, sort_keys=True).encode())


def code_fingerprint() -> dict:
    """sha256 of every hqprompt/**/*.py (provenance in place of a git hash)."""
    root = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        with open(f, "rb") as fh:
            out[os.path.relpath(f, root)] = sha256_bytes(fh.read())
    out["_all"] = sha256_list(out)
    return out


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "host": platform.node()}
    for mod in ("diffusers", "peft", "transformers", "accelerate", "pyiqa", "facexlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
