"""Setup shared by the Step 0-2 scripts: output paths, splits, and the tiny stand-ins used by --dry_run.

A dry run swaps only model and data construction (random VRE, tiny diffusers models, synthetic FFHQ);
all downstream logic is the real code path.
"""
from __future__ import annotations

import dataclasses
import os

import torch

from hqprompt import data, osdface
from hqprompt.paths import Paths, assert_writable, paths_from_args

DRY_SIZES = {"pred_train": 8, "ft_train": 4, "val": 2, "spare": 2}


def run_paths(args) -> Paths:
    """Paths from CLI args. A dry run must name its --out_root so it can never touch the real one."""
    if args.dry_run and args.out_root is None:
        raise SystemExit("--dry_run needs an explicit --out_root (a scratch directory)")
    paths = paths_from_args(args)
    os.makedirs(assert_writable(paths.out_root, paths), exist_ok=True)
    return paths


def setup_data(paths: Paths, dry_run: bool) -> tuple[Paths, dict]:
    """Return (paths, splits). Real: create splits.json from the FFHQ dirs if missing. Dry: synthetic FFHQ."""
    if dry_run:
        root = assert_writable(f"{paths.out_root}/dry_data", paths)
        hq, lq = data.write_synthetic_ffhq(root, n=sum(DRY_SIZES.values()))
        paths = dataclasses.replace(paths, ffhq_hq=hq, ffhq_lq=lq)
        splits = data.make_splits(n_total=sum(DRY_SIZES.values()), sizes=DRY_SIZES)
    elif os.path.exists(paths.splits_json):
        return paths, data.load_splits(paths.splits_json)
    else:
        splits = data.make_splits()
    data.write_splits(paths, splits, paths.ffhq_hq, paths.ffhq_lq)
    return paths, data.load_splits(paths.splits_json)


def tiny_merged_models(seed: int = 0):
    """Tiny UNet/VAE with a synthetic LoRA merged through the real merge function (dry runs)."""
    models = osdface.tiny_models(seed)
    lora = osdface.tiny_lora_state_dict(models.unet, seed)
    merged, report = osdface.merge_lora_state_dict(models.unet.state_dict(), lora, 1.0)
    models.unet.load_state_dict(merged, strict=True)
    return models, report


def load_batch(paths: Paths, ids, hq: bool = True) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Stack FFHQ images for `ids` as fp32 (B,3,512,512) in [0,1]: (hq or None, lq)."""
    hq_t = torch.stack([data.load_image_01(data.ffhq_hq_path(paths, i)) for i in ids]) if hq else None
    lq_t = torch.stack([data.load_image_01(data.ffhq_lq_path(paths, i)) for i in ids])
    return hq_t, lq_t
