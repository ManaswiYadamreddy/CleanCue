"""Steps 3, 4, 6: fine-tune a new LoRA (plus the prompt projection / positions / null tokens) on top of the
merged official OSDFace UNet, one step at t=399, with gradients through the frozen VAE decoder. No GAN.

Everything the model sees as a prompt is built by `prompt.PromptEncoder`; this file only wires it up.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from hqprompt import data, losses, osdface, prompt as prompt_lib, vre
from hqprompt.common import (atomic_write_json, code_fingerprint, env_info, get_device, seed_everything,
                             set_numerics)
from hqprompt.paths import Paths, add_path_args, assert_writable, paths_from_args, weights_manifest

log = logging.getLogger(__name__)

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
EXPECTED_LORA_MODULES = 128          # 16 transformer blocks x 2 attentions x 4 targets (real SD-2.1 UNet)
DRY_SIZES = {"pred_train": 4, "ft_train": 8, "val": 4, "spare": 2}
SMOKE_MAX_DIFF, SMOKE_MEAN_DIFF = 1e-2, 1e-3
PROBE_N = 4


# --------------------------------------------------------------------------- model construction
def dry_models() -> osdface.OSDFaceModels:
    """Tiny deterministic stand-in for the merged OSDFace models (a deterministic tiny LoRA is merged in)."""
    m = osdface.tiny_models(seed=0)
    merged, _ = osdface.merge_lora_state_dict(m.unet.state_dict(), osdface.tiny_lora_state_dict(m.unet, seed=0), 1.0)
    m.unet.load_state_dict(merged, strict=True)
    return m


def load_models(paths: Paths, device, dry_run: bool) -> tuple[osdface.OSDFaceModels, dict]:
    if dry_run:
        m = dry_models()
        m.unet.to(device), m.vae.to(device)
        return m, {"dry_run": True}
    return osdface.load_osdface(paths, device, merge_lora=True)


def load_codebook(kind: str, paths: Paths, dry_run: bool) -> torch.Tensor:
    if dry_run:
        return getattr(vre.random_vre(seed=0), kind).codebook.detach().clone()
    return vre.load_codebook(paths.associate_ckpt, kind, paths)


def load_ec(paths: Paths, dry_run: bool) -> dict:
    return prompt_lib.random_embedding_change_sd(0) if dry_run else prompt_lib.load_embedding_change(paths)


def build_prompt_encoder(mode: str, paths: Paths, dry_run: bool, device) -> prompt_lib.PromptEncoder:
    kind = prompt_lib.CODEBOOK_FOR_MODE[mode]
    cb = load_codebook(kind, paths, dry_run) if kind else None
    return prompt_lib.PromptEncoder(mode, cb, load_ec(paths, dry_run), trainable=True).to(device)


def count_attention(unet) -> int:
    from diffusers.models.attention_processor import Attention
    return sum(isinstance(m, Attention) for m in unet.modules())


def lora_param_names(unet) -> list[str]:
    return [n for n, _ in unet.named_parameters() if "lora_" in n]


def add_lora(unet, rank: int, alpha: int, dry_run: bool) -> int:
    """Add the new trainable LoRA; returns the number of wrapped modules (checked against the architecture)."""
    from peft import LoraConfig
    from peft.tuners.lora import LoraLayer

    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, init_lora_weights=True,
                                target_modules=LORA_TARGETS))
    n_mod = sum(isinstance(m, LoraLayer) for m in unet.modules())
    expected = count_attention(unet) * len(LORA_TARGETS)
    assert n_mod == expected, f"{n_mod} LoRA modules, expected {expected} (= Attention modules x 4)"
    if not dry_run:
        assert n_mod == EXPECTED_LORA_MODULES, f"{n_mod} LoRA modules, expected {EXPECTED_LORA_MODULES}"
    names = lora_param_names(unet)
    assert len(names) == 2 * n_mod, f"{len(names)} LoRA tensors for {n_mod} modules"
    for n, p in unet.named_parameters():
        p.requires_grad_("lora_" in n)
        assert not ("lora_" in n) or p.dtype == torch.float32, f"{n} is {p.dtype}"
    return n_mod


def lora_state_dict(unet) -> dict[str, torch.Tensor]:
    from peft.utils import get_peft_model_state_dict
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(unet).items()}


def load_lora_state_dict(unet, sd: dict) -> None:
    """Strict load: every LoRA tensor present, none unexpected, and the result reads back identical."""
    from peft.utils import set_peft_model_state_dict
    expected = set(lora_state_dict(unet))
    if set(sd) != expected:
        raise KeyError(f"LoRA keys differ: missing {sorted(expected - set(sd))[:5]}, "
                       f"unexpected {sorted(set(sd) - expected)[:5]}")
    set_peft_model_state_dict(unet, {k: v.clone() for k, v in sd.items()})
    back = lora_state_dict(unet)
    bad = [k for k in sd if not torch.equal(back[k], sd[k].to(back[k].dtype))]
    if bad:
        raise RuntimeError(f"LoRA load did not round-trip for {len(bad)} tensors, e.g. {bad[:3]}")


def load_trainable(ckpt_dir: str, unet, prompt_enc) -> int:
    """Load trainable.safetensors strictly (both prefixes). Returns the step stored in state.json."""
    tensors = load_file(f"{ckpt_dir}/trainable.safetensors")
    split = {p: {k[len(p) + 1:]: v for k, v in tensors.items() if k.startswith(p + ".")} for p in ("lora", "prompt")}
    stray = [k for k in tensors if k.split(".", 1)[0] not in split]
    if stray:
        raise KeyError(f"unexpected keys in {ckpt_dir}/trainable.safetensors: {stray[:5]}")
    load_lora_state_dict(unet, split["lora"])
    prompt_enc.load_trainable_state_dict(split["prompt"], strict=True)
    with open(f"{ckpt_dir}/state.json") as f:
        return int(json.load(f)["step"])


# --------------------------------------------------------------------------- losses
def compute_losses(out_pm1, x0, batch, loss_modules, weights, structure_fn: Callable | None = None) -> dict:
    """out_pm1: decoded image (unclamped, [-1,1] scale), x0: scaled latent. `structure_fn(out01, hq01, batch)`
    is the hook for Proposal 2 (weight weights["lambda_3d"]); finetune passes None for now."""
    hq_pm1 = batch["hq_pm1"]
    o01, h01 = (out_pm1.clamp(-1, 1) + 1) / 2, (hq_pm1 + 1) / 2
    parts = {"mse": nn.functional.mse_loss(out_pm1, hq_pm1),
             "dists": loss_modules["ea_dists"](o01, h01),
             "id": loss_modules["id"](o01, h01),
             "anchor": nn.functional.mse_loss(x0, batch["lat_teacher"])}
    total = (weights["w_mse"] * parts["mse"] + weights["w_dists"] * parts["dists"]
             + weights["w_id"] * parts["id"] + weights["w_anchor"] * parts["anchor"])
    if structure_fn is not None:
        parts["structure"] = structure_fn(o01, h01, batch)
        total = total + weights["lambda_3d"] * parts["structure"]
    parts["total"] = total
    return parts


@dataclass
class Setup:
    models: osdface.OSDFaceModels
    prompt_enc: prompt_lib.PromptEncoder
    loss_modules: dict
    weights: dict
    amp_dtype: torch.dtype | None
    device: torch.device
    opt: torch.optim.Optimizer
    sched: object
    groups: dict[str, list[tuple[str, torch.Tensor]]]
    n_lora_modules: int
    merge_report: dict

    @property
    def trainable(self) -> list[torch.Tensor]:
        return [p for g in ("lora", "proj", "pos") for _, p in self.groups[g]]


def forward_losses(s: Setup, batch: dict, structure_fn=None) -> dict:
    B = batch["lat_lq"].shape[0]
    pe = s.prompt_enc
    prompt = pe(batch_size=B) if pe.mode == "null" else pe(batch["idx"])          # fp32, outside autocast
    x0 = osdface.one_step(s.models.unet, batch["lat_lq"], prompt, s.models.alphas_cumprod, amp_dtype=s.amp_dtype)
    out = osdface.vae_decode(s.models.vae, x0, amp_dtype=s.amp_dtype, clamp=False)  # grads flow through the decoder
    if torch.is_grad_enabled():
        assert out.requires_grad, "decoded image has no grad: the decoder path is detached (trap 3)"
    return compute_losses(out, x0, batch, s.loss_modules, s.weights, structure_fn)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def grad_norms(s: Setup) -> dict:
    def total(ps):
        gs = [p.grad.detach().float().norm() ** 2 for _, p in ps if p.grad is not None]
        return float(torch.stack(gs).sum().sqrt()) if gs else 0.0
    return {"grad_norm_lora": total(s.groups["lora"]), "grad_norm_proj": total(s.groups["proj"]),
            ("grad_norm_tokens" if s.prompt_enc.mode == "null" else "grad_norm_pos"): total(s.groups["pos"])}


def per_tensor_grad_norms(s: Setup) -> dict[str, float]:
    return {n: float(p.grad.detach().float().norm()) if p.grad is not None else 0.0
            for g in ("lora", "proj", "pos") for n, p in s.groups[g]}


def hash_frozen(models: osdface.OSDFaceModels, prompt_enc) -> dict[str, str]:
    """sha256 of every non-trainable tensor (UNet base weights, VAE, codebook, ...)."""
    out = {}
    for prefix, mod in (("unet", models.unet), ("vae", models.vae), ("prompt", prompt_enc)):
        for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
            if not t.requires_grad:
                out[f"{prefix}.{n}"] = hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    out["alphas_cumprod"] = hashlib.sha256(models.alphas_cumprod.cpu().numpy().tobytes()).hexdigest()
    return out


def assert_no_dropout(*mods) -> None:
    for m in mods:
        for sub in m.modules():
            if isinstance(sub, nn.Dropout) and sub.p != 0:
                raise ValueError(f"cannot use train mode for checkpointing: dropout p={sub.p} in {type(sub).__name__}")


def build_setup(args, paths: Paths, device) -> Setup:
    models, report = load_models(paths, device, args.dry_run)
    if not args.dry_run:
        assert (report["n_pairs"], report["n_conv"]) == (278, 64), report
    n_mod = add_lora(models.unet, args.lora_rank, args.lora_alpha, args.dry_run)
    pe = build_prompt_encoder(args.prompt, paths, args.dry_run, device)
    loss_modules = losses.build_losses(paths, device, args.dry_run)

    if args.grad_ckpt_unet:
        assert_no_dropout(models.unet)
        models.unet.enable_gradient_checkpointing()
        models.unet.train()                  # diffusers only checkpoints in train mode
    if args.grad_ckpt_vae:
        assert_no_dropout(models.vae)
        models.vae.enable_gradient_checkpointing()
        models.vae.decoder.train()

    groups = {"lora": [(n, p) for n, p in models.unet.named_parameters() if p.requires_grad],
              "proj": [(n, p) for n, p in pe.named_parameters() if p.requires_grad and n.startswith("proj.")],
              "pos": [(n, p) for n, p in pe.named_parameters() if p.requires_grad and not n.startswith("proj.")]}
    assert all("lora_" in n for n, _ in groups["lora"]) and len(groups["lora"]) == 2 * n_mod
    expected_prompt = {"lq": {"proj", "pos"}, "hq_true": {"proj", "pos"}, "hq_pred": {"proj", "pos"},
                       "null": {"pos"}}[args.prompt]
    assert {g for g in ("proj", "pos") if groups[g]} == expected_prompt, {g: len(v) for g, v in groups.items()}
    trainable = [p for g in groups.values() for _, p in g]
    assert all(p.dtype == torch.float32 for p in trainable), "trainable params must be fp32 (trap 5)"

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    all_req = {id(p) for mod in (models.unet, models.vae, pe, *loss_modules.values())
               for p in mod.parameters() if p.requires_grad}
    assert {id(p) for p in trainable} == all_req == {id(p) for g in opt.param_groups for p in g["params"]}, \
        "optimizer params differ from the set of requires_grad params"
    from diffusers.optimization import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.max_steps)

    amp_dtype = torch.bfloat16 if args.amp == "bf16" and not args.dry_run else None
    if amp_dtype is not None:
        from hqprompt.common import require_bf16
        require_bf16(device)
    weights = {"w_mse": args.w_mse, "w_dists": args.w_dists, "w_id": args.w_id, "w_anchor": args.w_anchor,
               "lambda_3d": args.lambda_3d}
    return Setup(models, pe, loss_modules, weights, amp_dtype, device, opt, sched, groups, n_mod, report)


# --------------------------------------------------------------------------- dry-run data
def prepare_dry_run(paths: Paths) -> Paths:
    """Synthetic images, splits and cache under out_root. lat_teacher is the real teacher of the tiny merged
    UNet (one_step with the official prompt from lq_prompt77), so the smoke iteration-0 check is meaningful.
    Idempotent. Returns paths whose ffhq dirs point at the synthetic images."""
    root = f"{paths.out_root}/dry_run"
    hq_dir, lq_dir = f"{root}/imgs/hq", f"{root}/imgs/lq"
    if not os.path.exists(paths.splits_json):
        splits = data.make_splits(n_total=sum(DRY_SIZES.values()), sizes=DRY_SIZES)
        data.write_synthetic_ffhq(f"{root}/imgs", n=splits["n_total"])
        data.write_splits(paths, splits, hq_dir, lq_dir, check_files=False)
    splits = data.load_splits(paths.splits_json)
    if not os.path.exists(f"{paths.cache_dir}/manifest.json"):
        data.write_synthetic_cache(paths, splits)
        m, pe = dry_models(), build_prompt_encoder("official", paths, True, "cpu")
        for split in ("ft_train", "val"):
            rd = data.CacheReader(paths, split, ["lat_lq", "lq_prompt77"])
            lat = torch.from_numpy(np.array(rd.array("lat_lq"), dtype=np.float32))
            idx = torch.from_numpy(np.array(rd.array("lq_prompt77"), dtype=np.int64))
            with torch.no_grad():
                x0 = osdface.one_step(m.unet, lat, pe(idx), m.alphas_cumprod)
            w = data.CacheWriter(paths, split, rd.ids, ["lat_teacher"], {"synthetic": True}, {}, splits["sha256"])
            w.write("lat_teacher", np.arange(len(rd.ids)), x0.numpy().astype(np.float16))
            w.finalize()
    return dataclasses.replace(paths, ffhq_hq=hq_dir, ffhq_lq=lq_dir)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(run_dir: str, step: int, s: Setup, paths: Paths) -> str:
    final = assert_writable(f"{run_dir}/ckpt/step_{step:06d}", paths)
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    tensors = {f"lora.{k}": v.contiguous().cpu() for k, v in lora_state_dict(s.models.unet).items()}
    tensors.update({f"prompt.{k}": v.contiguous().cpu() for k, v in s.prompt_enc.trainable_state_dict().items()})
    save_file(tensors, f"{tmp}/trainable.safetensors")
    torch.save(s.opt.state_dict(), f"{tmp}/optimizer.pt")
    torch.save(s.sched.state_dict(), f"{tmp}/scheduler.pt")
    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, f"{tmp}/rng.pt")
    with open(f"{tmp}/state.json", "w") as f:
        json.dump({"step": step}, f)
    os.replace(tmp, final)
    return final


def resume_checkpoint(ckpt_dir: str, s: Setup) -> int:
    step = load_trainable(ckpt_dir, s.models.unet, s.prompt_enc)
    s.opt.load_state_dict(torch.load(f"{ckpt_dir}/optimizer.pt", map_location=s.device, weights_only=True))
    s.sched.load_state_dict(torch.load(f"{ckpt_dir}/scheduler.pt", weights_only=True))
    rng = torch.load(f"{ckpt_dir}/rng.pt", weights_only=True)
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return step


# --------------------------------------------------------------------------- smoke checks
@torch.no_grad()
def smoke_iteration0(s: Setup, paths: Paths, args) -> dict:
    """Adapters off + official prompt + fp32 must reproduce the cached teacher latent."""
    ds = data.FinetuneDataset(paths, "val", "official")
    batch = to_device(torch.utils.data.default_collate([ds[k] for k in range(min(PROBE_N, len(ds)))]), s.device)
    pe = prompt_lib.PromptEncoder("official", load_codebook("lq", paths, args.dry_run), load_ec(paths, args.dry_run)).to(s.device)
    s.models.unet.disable_adapters()
    try:
        prompt = pe(batch["idx"])
        x0 = osdface.one_step(s.models.unet, batch["lat_lq"], prompt, s.models.alphas_cumprod)
        x0_amp = (osdface.one_step(s.models.unet, batch["lat_lq"], prompt, s.models.alphas_cumprod, amp_dtype=s.amp_dtype)
                  if s.amp_dtype is not None else None)
    finally:
        s.models.unet.enable_adapters()
    d = (x0 - batch["lat_teacher"]).abs()
    res = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
           "max_allowed": SMOKE_MAX_DIFF, "mean_allowed": SMOKE_MEAN_DIFF}
    if x0_amp is not None:
        da = (x0_amp - batch["lat_teacher"]).abs()
        res["bf16_max_abs_diff"], res["bf16_mean_abs_diff"] = float(da.max()), float(da.mean())
    res["pass"] = res["max_abs_diff"] <= SMOKE_MAX_DIFF and res["mean_abs_diff"] <= SMOKE_MEAN_DIFF
    return res


@torch.no_grad()
def probe_loss(s: Setup, batch: dict) -> float:
    return float(forward_losses(s, batch)["total"])


def check_grads(grads0: dict, grads10: dict) -> dict:
    zero0 = [n for n, v in grads0.items() if ("lora_B" in n or "lora_" not in n) and not v > 0]
    zero10 = [n for n, v in grads10.items() if "lora_A" in n and not v > 0]
    return {"step0_zero_grad_tensors": zero0[:10], "step10_zero_lora_A_tensors": zero10[:10],
            "n_step0_tensors": sum(1 for n in grads0 if "lora_B" in n or "lora_" not in n),
            "n_step10_lora_A": sum(1 for n in grads10 if "lora_A" in n),
            "pass": not zero0 and not zero10}


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True, choices=["lq", "null", "hq_true", "hq_pred"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_3d", type=float, default=0.0)
    ap.add_argument("--hq_pred_run", default=None, help="predictor run dir (required for hq_pred)")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="default 10000 (200 with --smoke)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=None, help="default 200 (20 with --smoke so the lr actually moves)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_dists", type=float, default=1.0)
    ap.add_argument("--w_id", type=float, default=0.1)
    ap.add_argument("--w_anchor", type=float, default=1.0)
    ap.add_argument("--amp", choices=["bf16", "none"], default=None, help="default bf16 (none for --dry_run)")
    ap.add_argument("--grad_ckpt_unet", action="store_true")
    ap.add_argument("--grad_ckpt_vae", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None, help="checkpoint dir inside this run's ckpt/")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.lambda_3d != 0:
        raise NotImplementedError("structure loss (Proposal 2) not implemented")
    if args.prompt == "hq_pred" and not args.hq_pred_run:
        ap.error("--hq_pred_run is required for --prompt hq_pred")
    if args.prompt != "hq_pred" and args.hq_pred_run:
        ap.error("--hq_pred_run only applies to --prompt hq_pred")
    args.max_steps = args.max_steps or (200 if args.smoke else 10000)
    args.warmup = args.warmup if args.warmup is not None else (20 if args.smoke else 200)
    args.amp = args.amp or ("none" if args.dry_run else "bf16")
    args.run_name = args.run_name or (f"smoke_{args.prompt}_s{args.seed}" if args.smoke
                                      else f"{args.prompt}_l3d{args.lambda_3d:g}_s{args.seed}")
    if args.dry_run:
        args.num_workers = 0
    if args.smoke and args.max_steps <= 10:
        ap.error("--smoke needs --max_steps > 10 (lora_A gradients are checked at step 10)")
    return args


def _hq_pred_file(args, splits: dict) -> str:
    meta = json.load(open(f"{args.hq_pred_run}/meta.json"))
    if meta.get("train_split") != "pred_train":
        raise ValueError(f"{args.hq_pred_run}/meta.json train_split={meta.get('train_split')!r}, expected 'pred_train'")
    ids = json.load(open(f"{args.hq_pred_run}/pred/ft_train.ids.json"))
    if ids != splits["splits"]["ft_train"]:
        raise ValueError("predictor ft_train ids differ from splits.json ft_train")
    return f"{args.hq_pred_run}/pred/ft_train.npy"


def _check_run_dir(args, run_dir: str, config: dict) -> None:
    exists = os.path.isdir(run_dir) and bool(os.listdir(run_dir))
    if not args.resume:
        if exists:
            raise FileExistsError(f"{run_dir} is not empty; pick another --run_name or pass --resume CKPT_DIR")
        return
    ckpt = os.path.realpath(args.resume)
    if os.path.dirname(ckpt) != os.path.realpath(f"{run_dir}/ckpt"):
        raise ValueError(f"--resume {args.resume} is not inside {run_dir}/ckpt")
    with open(f"{run_dir}/config.json") as f:
        old = json.load(f)["args"]
    new = {k: v for k, v in config["args"].items() if k != "resume"}
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if k != "resume" and old.get(k) != new.get(k)}
    if diff:
        raise ValueError(f"config differs from the original run (old, new): {diff}")


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = paths_from_args(args)
    device = get_device(args.dry_run)
    run_dir = f"{paths.runs_dir}/{args.run_name}"
    assert_writable(run_dir, paths)
    if args.dry_run:
        paths = prepare_dry_run(paths)
    seed_everything(args.seed)
    set_numerics(deterministic=True, tf32=False)

    splits = data.load_splits(paths.splits_json)
    if splits["source"] != "ffhq":
        raise ValueError(f"finetune only trains on FFHQ splits, got source={splits['source']!r}")
    ds = data.FinetuneDataset(paths, "ft_train", args.prompt,
                              _hq_pred_file(args, splits) if args.prompt == "hq_pred" else None)
    if not np.array_equal(ds.ids, data.split_ids(splits, "ft_train")):
        raise ValueError("cache ft_train ids differ from splits.json")

    s = build_setup(args, paths, device)
    config = {"args": {k: v for k, v in vars(args).items()}, "n_lora_modules": s.n_lora_modules,
              "merge_report": s.merge_report, "splits_sha256": splits["sha256"],
              "cache_settings": ds.reader.settings, "env": env_info(), "code": code_fingerprint(),
              "weights": {} if args.dry_run else weights_manifest(paths)}
    _check_run_dir(args, run_dir, config)
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)

    start = 0
    if args.resume:
        start = resume_checkpoint(args.resume, s)
        log_path = f"{run_dir}/train_log.jsonl"
        if os.path.exists(log_path):  # drop lines from the crashed tail after the checkpoint
            keep = [l for l in open(log_path) if json.loads(l)["step"] < start]
            with open(log_path, "w") as f:
                f.writelines(keep)
        log.info("resumed from %s at step %d", args.resume, start)
    else:
        atomic_write_json(f"{run_dir}/config.json", config, paths)

    smoke = {}
    probe = None
    if args.smoke:
        smoke["iteration0"] = smoke_iteration0(s, paths, args)
        probe = to_device(torch.utils.data.default_collate([ds[k] for k in range(min(PROBE_N, len(ds)))]), device)
        smoke["probe_loss_start"] = probe_loss(s, probe)
        hashes_before = hash_frozen(s.models, s.prompt_enc)

    order = data.step_order(len(ds), args.max_steps, args.batch_size, args.seed)
    loader = DataLoader(ds, batch_sampler=[list(map(int, r)) for r in order[start:]],
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log_f = open(f"{run_dir}/train_log.jsonl", "a")
    step_losses, grads0, grads10 = [], {}, {}
    t0, t_steps = time.time(), 0
    for step, batch in enumerate(loader, start=start):
        batch = to_device(batch, device)
        parts = forward_losses(s, batch)
        parts["total"].backward()
        if args.smoke and step in (0, 10):
            (grads0 if step == 0 else grads10).update(per_tensor_grad_norms(s))
        lr = s.sched.get_last_lr()[0]
        log_now = step % args.log_every == 0
        norms = grad_norms(s) if log_now else {}   # pre-clip
        gn_total = float(torch.nn.utils.clip_grad_norm_(s.trainable, args.grad_clip))
        s.opt.step()
        s.sched.step()
        s.opt.zero_grad(set_to_none=True)
        t_steps += 1
        total = float(parts["total"])
        if not np.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in parts.items()} }")
        step_losses.append(total)
        if log_now:
            rec = {"step": step, "lr": lr, "loss": total, **{k: float(parts[k]) for k in ("mse", "dists", "id", "anchor")},
                   "grad_norm_total": gn_total, **norms, "it_s": t_steps / (time.time() - t0),
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            log.info("%s", rec)
        done = step + 1
        if done % args.save_every == 0 or done == args.max_steps:
            save_checkpoint(run_dir, done, s, paths)
    log_f.close()
    with open(f"{run_dir}/DONE", "w") as f:
        f.write(f"{args.max_steps}\n")

    if not args.smoke:
        return 0
    smoke["gradients"] = check_grads(grads0, grads10)
    after = hash_frozen(s.models, s.prompt_enc)
    changed = sorted(k for k in hashes_before if hashes_before[k] != after.get(k)) + sorted(set(after) - set(hashes_before))
    smoke["frozen_weights"] = {"n_tensors": len(hashes_before), "changed": changed[:10], "pass": not changed}
    end = probe_loss(s, probe)
    w = min(50, args.max_steps // 4)
    smoke["loss"] = {"probe_start": smoke["probe_loss_start"], "probe_end": end,
                     "mean_first": float(np.mean(step_losses[:w])), "mean_last": float(np.mean(step_losses[-w:])),
                     "window": w, "pass": end < smoke["probe_loss_start"]}
    del smoke["probe_loss_start"]
    smoke["speed"] = {"it_s": t_steps / (time.time() - t0),
                      "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
    smoke["passed"] = all(smoke[k]["pass"] for k in ("iteration0", "gradients", "frozen_weights", "loss"))
    atomic_write_json(f"{run_dir}/smoke_report.json", smoke, paths)
    log.info("smoke report: %s", json.dumps(smoke, indent=1))
    if not smoke["passed"]:
        log.error("SMOKE FAILED: %s", [k for k in ("iteration0", "gradients", "frozen_weights", "loss") if not smoke[k]["pass"]])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
