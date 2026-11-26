"""Inference for a fine-tuned run (rows 2-5, and row 4's model with predicted codes).

Rebuilds the merged UNet + new LoRA + PromptEncoder exactly as in training, loads `trainable.safetensors`
strictly, and runs the same one-step path. Refuses codes from a different codebook than the trained mode.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torchvision.transforms import ToPILImage

from hqprompt import data, finetune, osdface, prompt as prompt_lib, vre
from hqprompt.common import (atomic_write_json, code_fingerprint, env_info, get_device, require_bf16, set_numerics,
                             sha256_list)
from hqprompt.paths import Paths, add_path_args, assert_writable, paths_from_args

log = logging.getLogger(__name__)
CODES = ("lq", "hq_true", "hq_pred", "null")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: the latest)")
    ap.add_argument("--codes", required=True, choices=CODES)
    ap.add_argument("--hq_pred_run", default=None, help="predictor run dir (required for --codes hq_pred)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=["val"])
    src.add_argument("--input_dir")
    ap.add_argument("--dataset_name", default=None, help="required with --input_dir")
    ap.add_argument("--hq_dir", default=None, help="clean images for --codes hq_true with --input_dir")
    ap.add_argument("--amp", choices=["bf16", "none"], default=None, help="default: as in training")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default=None, help="default {run_dir}/outputs/step_{step}/{codes}/{dataset}")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.input_dir and not args.dataset_name:
        ap.error("--input_dir needs --dataset_name")
    if args.codes == "hq_pred" and not args.hq_pred_run:
        ap.error("--codes hq_pred needs --hq_pred_run")
    if args.codes == "hq_true" and args.input_dir and not args.hq_dir:
        ap.error("--codes hq_true with --input_dir needs --hq_dir")
    return args


def check_codes_compatible(trained_mode: str, codes: str) -> None:
    ck_trained, ck_codes = prompt_lib.CODEBOOK_FOR_MODE[trained_mode], prompt_lib.CODEBOOK_FOR_MODE[codes]
    if ck_trained != ck_codes:
        raise ValueError(f"run was trained with {trained_mode!r} prompts (codebook {ck_trained}); refusing "
                         f"--codes {codes!r} (codebook {ck_codes})")


def pick_step(run_dir: str, step: int | None) -> int:
    steps = sorted(int(d.split("_")[1]) for d in os.listdir(f"{run_dir}/ckpt")
                   if d.startswith("step_") and not d.endswith(".tmp"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints in {run_dir}/ckpt")
    if step is None:
        return steps[-1]
    if step not in steps:
        raise FileNotFoundError(f"no checkpoint for step {step} in {run_dir}/ckpt (have {steps})")
    return step


def _cache_settings(paths: Paths) -> dict:
    with open(f"{paths.cache_dir}/manifest.json") as f:
        return json.load(f)["settings"]


def _pred_array(pred_run: str, stem: str, expect: list) -> np.ndarray:
    ids = json.load(open(f"{pred_run}/pred/{stem}.ids.json" if stem == "val" else f"{pred_run}/pred/{stem}.names.json"))
    if ids != expect:
        raise ValueError(f"{pred_run}/pred/{stem}: ids/names differ from the images to restore")
    return np.load(f"{pred_run}/pred/{stem}.npy").astype(np.int64)


def _val_inputs(args, paths: Paths, splits: dict):
    """-> (names, lat_lq (N,4,64,64) fp32 memmap-backed, idx (N,L) int64 | None)."""
    src = {"lq": "lq_idx", "hq_true": "hq_idx"}.get(args.codes)
    rd = data.CacheReader(paths, "val", ["lat_lq"] + ([src] if src else []))
    ids = data.split_ids(splits, "val")
    if not np.array_equal(rd.ids, ids):
        raise ValueError("cache val ids differ from splits.json")
    idx = None
    if src:
        idx = np.asarray(rd.array(src), dtype=np.int64)
    elif args.codes == "hq_pred":
        idx = _pred_array(args.hq_pred_run, "val", [int(i) for i in ids])
    return [f"{int(i):05d}" for i in ids], rd.array("lat_lq"), idx


def _hq_files(hq_dir: str, names: list[str]) -> list[str]:
    by_stem = {data.norm_stem(n): n for n in data.list_images(hq_dir)}
    missing = [n for n in names if data.norm_stem(n) not in by_stem]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images have no clean counterpart in {hq_dir}, e.g. {missing[:5]}")
    return [os.path.join(hq_dir, by_stem[data.norm_stem(n)]) for n in names]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = paths_from_args(args)
    with open(f"{args.run_dir}/config.json") as f:
        cfg = json.load(f)
    targs = cfg["args"]
    check_codes_compatible(targs["prompt"], args.codes)
    if bool(targs["dry_run"]) != args.dry_run:
        raise ValueError(f"run dry_run={targs['dry_run']} but infer --dry_run={args.dry_run}")
    step = pick_step(args.run_dir, args.step)
    device = get_device(args.dry_run)
    set_numerics(deterministic=True, tf32=False)
    amp = args.amp or targs["amp"]
    amp_dtype = torch.bfloat16 if amp == "bf16" and not args.dry_run else None
    if amp_dtype is not None:
        require_bf16(device)
    if args.dry_run:
        paths = finetune.prepare_dry_run(paths)

    dataset = "val" if args.split else args.dataset_name
    out_dir = assert_writable(args.out_dir or f"{args.run_dir}/outputs/step_{step}/{args.codes}/{dataset}", paths)
    os.makedirs(f"{out_dir}/images", exist_ok=True)

    models, report = finetune.load_models(paths, device, args.dry_run)
    n_mod = finetune.add_lora(models.unet, targs["lora_rank"], targs["lora_alpha"], args.dry_run)
    pe = finetune.build_prompt_encoder(targs["prompt"], paths, args.dry_run, device)
    ckpt = f"{args.run_dir}/ckpt/step_{step:06d}"
    finetune.load_trainable(ckpt, models.unet, pe)
    models.unet.eval()

    settings = _cache_settings(paths)
    with torch.no_grad():
        if args.split:
            splits = data.load_splits(paths.splits_json)
            names, lat_src, idx_all = _val_inputs(args, paths, splits)
            batches = ((names[s:s + args.batch_size],
                        torch.from_numpy(np.array(lat_src[s:s + args.batch_size], dtype=np.float32)),
                        None if idx_all is None else torch.from_numpy(idx_all[s:s + args.batch_size]))
                       for s in range(0, len(names), args.batch_size))
            stems = [n for n in names]
        else:
            files = data.list_images(args.input_dir)
            stems = [os.path.splitext(n)[0] for n in files]
            v = vre.random_vre(seed=0) if args.dry_run else None
            if args.codes in ("lq", "hq_true") and v is None:
                v = vre.load_vre(paths, branches=("lq",) if args.codes == "lq" else ("hq",), device=str(device))
            if v is not None:
                v = v.to(device).eval()
            hq_files = _hq_files(args.hq_dir, files) if args.codes == "hq_true" else None
            hq_mode = settings.get("hq_input_mode", "single" if args.dry_run else None)
            if args.codes == "hq_true" and hq_mode not in ("single", "double"):
                raise ValueError(f"cache manifest has no valid hq_input_mode: {hq_mode!r}")
            pred = _pred_array(args.hq_pred_run, f"test_{dataset}", files) if args.codes == "hq_pred" else None
            ds = data.ImageDataset(paths, files, source=args.input_dir, resize="official")

            def gen():
                for s in range(0, len(files), args.batch_size):
                    rows = range(s, min(s + args.batch_size, len(files)))
                    lq01 = torch.stack([ds[k]["lq01"] for k in rows]).to(device)
                    lat = osdface.vae_encode(models.vae, lq01 * 2 - 1, "mean")
                    if args.codes == "lq":
                        idx = v.lq_codes(lq01)[0]
                    elif args.codes == "hq_true":
                        hq01 = torch.stack([data.load_image_01(hq_files[k], resize="official") for k in rows]).to(device)
                        idx = v.hq_codes(hq01, hq_mode)
                    elif args.codes == "hq_pred":
                        idx = torch.from_numpy(pred[s:s + len(rows)])
                    else:
                        idx = None
                    yield [files[k] for k in rows], lat.cpu(), None if idx is None else idx.cpu()
            batches = gen()

        to_pil, n_done = ToPILImage(), 0
        for bnames, lat, idx in batches:
            lat = lat.to(device)
            prompt = pe(batch_size=lat.shape[0]) if pe.mode == "null" else pe(idx.to(device))
            x0 = osdface.one_step(models.unet, lat, prompt, models.alphas_cumprod, amp_dtype=amp_dtype)
            out = osdface.vae_decode(models.vae, x0, amp_dtype=amp_dtype, clamp=True) * 0.5 + 0.5
            for name, img in zip(bnames, out.clamp(0, 1).cpu()):
                to_pil(img).save(f"{out_dir}/images/{os.path.splitext(name)[0]}.png")
                n_done += 1

    meta = {"args": vars(args), "trained_prompt": targs["prompt"], "codes": args.codes, "step": step,
            "checkpoint": ckpt, "dataset": dataset, "n_images": n_done, "names_sha256": sha256_list(stems),
            "amp": amp if amp_dtype else "none", "n_lora_modules": n_mod, "merge_report": report,
            "cache_settings": settings, "train_code_sha": cfg["code"]["_all"],
            "env": env_info(), "code": code_fingerprint()}
    atomic_write_json(f"{out_dir}/meta.json", meta, paths)
    log.info("wrote %d images to %s/images", n_done, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
