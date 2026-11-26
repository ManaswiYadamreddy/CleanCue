"""Step 1 and the prompt-sensitivity check: official OSDFace inference, reproduced from OSDFace/infer.py.

Per image: official bilinear resize, official 77-token LQ prompt (double-normalised encoder input), VAE latent
(`sample()` by default, seeded per image), one UNet step at t=399, x0 in float64, VAE decode, ToPILImage.
Differences from the official script: fp32 prompt/lookup path, per-image seeds, optional batching.
`--prompt_variant` perturbs only the prompt: shuffled (another image's codes), zero_code, zero_embed.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np
import torch
from torchvision.transforms import ToPILImage

from hqprompt import _steps, data, osdface, vre
from hqprompt.common import (atomic_write_json, code_fingerprint, env_info, get_device, seed_everything,
                             require_bf16, set_numerics, sha256_list)
from hqprompt.paths import add_path_args, assert_writable, weights_manifest
from hqprompt.prompt import PromptEncoder, load_embedding_change, random_embedding_change_sd

VARIANTS = ("normal", "shuffled", "zero_code", "zero_embed")
LQ_SET_NOTE = ("The default CelebA LQ set (HQP_CELEBA_LQ) is SELF-GENERATED, not the official DAEFR CelebA-Test "
               "LQ set, so paper numbers need not be reproduced even with correct code.")


def derangement(n: int, seed: int) -> np.ndarray:
    """src with src[i] != i for all i: image i takes the codes of image src[i]."""
    assert n >= 2, "the shuffled variant needs at least 2 images"
    perm = np.random.default_rng(seed).permutation(n)
    src = np.empty(n, dtype=np.int64)
    src[perm] = perm[(np.arange(n) + 1) % n]
    return src


def _inputs(args, paths, splits) -> tuple[str, list[str], list[str], str]:
    """(dataset name, stems, file paths, input dir)."""
    if args.split:
        ids = data.split_ids(splits, args.split).tolist()
        return (f"ffhq_{args.split}", [f"{i:05d}" for i in ids], [data.ffhq_lq_path(paths, i) for i in ids],
                paths.ffhq_lq)
    names = data.list_images(args.input_dir)
    return (args.dataset_name, [os.path.splitext(n)[0] for n in names],
            [os.path.join(args.input_dir, n) for n in names], args.input_dir)


def _load(files, device) -> torch.Tensor:
    return torch.stack([data.load_image_01(f, resize="official") for f in files]).to(device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=["val"], help="FFHQ LQ val faces")
    src.add_argument("--input_dir")
    ap.add_argument("--dataset_name", help="output folder name for --input_dir")
    ap.add_argument("--prompt_variant", choices=VARIANTS, default="normal")
    ap.add_argument("--vae_mode", choices=["sample", "mean"], default="sample")
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast around UNet/VAE")
    ap.add_argument("--tf32", choices=["off", "on"], default="off")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="allow a non-empty output image folder")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.input_dir and not args.dataset_name:
        ap.error("--input_dir needs --dataset_name")

    paths = _steps.run_paths(args)
    device = get_device(args.dry_run)
    seed_everything(args.seed)
    set_numerics(deterministic=True, tf32=args.tf32 == "on")
    amp = torch.bfloat16 if args.dtype == "bf16" else None
    if amp is not None:
        require_bf16(device)
    paths, splits = _steps.setup_data(paths, args.dry_run) if args.split or args.dry_run else (paths, None)

    dataset, stems, files, input_dir = _inputs(args, paths, splits)
    if args.limit:
        stems, files = stems[:args.limit], files[:args.limit]
    n = len(files)
    assert n > 0, f"no images in {input_dir}"

    out_dir = f"{paths.outputs_dir}/official/{dataset}_{args.prompt_variant}"
    img_dir = assert_writable(f"{out_dir}/images", paths)
    if os.path.isdir(img_dir) and os.listdir(img_dir) and not args.overwrite:
        raise FileExistsError(f"{img_dir} is not empty; pass --overwrite to reuse it")
    os.makedirs(img_dir, exist_ok=True)

    if args.dry_run:
        lq_vre = vre.random_vre().to(device)
        models, report = _steps.tiny_merged_models()
        ec_sd, manifest = random_embedding_change_sd(), {}
    else:
        lq_vre = vre.load_vre(paths, branches=("lq",), device=device)
        models, report = osdface.load_osdface(paths, device, merge_lora=True)
        ec_sd, manifest = load_embedding_change(paths), weights_manifest(paths)
    unet, vae, alphas = models.unet.to(device), models.vae.to(device), models.alphas_cumprod.to(device)
    prompt_enc = PromptEncoder("official", lq_vre.lq.codebook.cpu(), ec_sd).to(device)

    # pass 1: official 77-token codes for every image (the shuffled variant needs all of them)
    idx77 = []
    with torch.no_grad():
        for k in range(0, n, args.batch_size):
            idx77.append(lq_vre.lq_official_prompt(_load(files[k:k + args.batch_size], device)).cpu())
    idx77 = torch.cat(idx77)
    if args.prompt_variant == "shuffled":
        shuffle_src = derangement(n, args.seed)
        used = idx77[torch.from_numpy(shuffle_src)]
        assert (shuffle_src != np.arange(n)).all()
    else:
        shuffle_src = None
        used = idx77 if args.prompt_variant == "normal" else torch.full_like(idx77, -1)

    # pass 2: generate
    to_pil = ToPILImage()
    with torch.no_grad():
        for k in range(0, n, args.batch_size):
            sl = slice(k, min(k + args.batch_size, n))
            x01 = _load(files[sl], device)
            if args.prompt_variant == "zero_embed":
                prompt = torch.zeros(len(x01), 77, 1024, device=device)
            else:
                prompt = prompt_enc(used[sl].to(device))
            z = torch.cat([
                osdface.vae_encode(vae, x[None] * 2 - 1, args.vae_mode, amp_dtype=amp, generator=(
                    torch.Generator(device).manual_seed(args.seed + zlib.crc32(stem.encode()))
                    if args.vae_mode == "sample" else None))
                for x, stem in zip(x01, stems[sl])])
            x0 = osdface.one_step(unet, z, prompt, alphas, amp_dtype=amp)
            out = osdface.vae_decode(vae, x0, amp_dtype=amp, clamp=True) * 0.5 + 0.5
            for img, stem in zip(out.clamp(0, 1).cpu(), stems[sl]):
                to_pil(img).save(assert_writable(f"{img_dir}/{stem}.png", paths))
            print(f"{sl.stop}/{n}", flush=True)

    np.save(assert_writable(f"{out_dir}/prompt_idx.npy", paths), used.numpy().astype(np.int16))
    meta = {"args": vars(args), "dataset": dataset, "input_dir": input_dir, "n_files": n,
            "names_sha256": sha256_list(stems), "weights": manifest, "merge_report": report,
            "shuffle_src": None if shuffle_src is None else shuffle_src.tolist(),
            "prompt_idx_note": "indices actually used; -1 everywhere for zero_code and zero_embed",
            "lq_set_note": LQ_SET_NOTE, "env": env_info(), "code": code_fingerprint()}
    atomic_write_json(f"{out_dir}/meta.json", meta, paths)
    print(f"wrote {n} images to {img_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
