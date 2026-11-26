"""Step 5: predict the 256 HQ code indices per face from the LQ VQ features z_lq.

Trains on pred_train only, selects on val, writes predictions for ft_train, val and test dirs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from hqprompt import data
from hqprompt.common import (atomic_write_json, get_device, require_bf16, seed_everything, set_numerics,
                             sha256_list)
from hqprompt.paths import Paths, add_path_args, assert_writable, paths_from_args

N_CODES, N_PATCH, DIM = 1024, 256, 512
BUCKETS = (("top10", 0, 10), ("11-50", 10, 50), ("51-200", 50, 200), ("rest", 200, N_CODES))
DRY_SIZES = {"pred_train": 96, "ft_train": 16, "val": 16, "spare": 16}


# --------------------------------------------------------------------------- models
class _Base(nn.Module):
    def __init__(self, seen: torch.Tensor):
        super().__init__()
        self.register_buffer("seen", seen.bool().clone())  # (1024,) codes seen in pred_train

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B,256,512) -> (B,256,1024)
        raise NotImplementedError

    def forward(self, z: torch.Tensor, amp: bool = False) -> torch.Tensor:
        """z (B,512,16,16) fp32 -> masked fp32 logits (B,256,1024)."""
        x = z.flatten(2).transpose(1, 2)
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = self.features(x)
        return logits.float().masked_fill(~self.seen, float("-inf"))


class LinearPredictor(_Base):
    """Shared Linear(512,1024) plus a learned per-position bias (256,1024)."""

    def __init__(self, seen: torch.Tensor):
        super().__init__(seen)
        self.proj = nn.Linear(DIM, N_CODES)
        self.pos_bias = nn.Parameter(torch.zeros(N_PATCH, N_CODES))

    def features(self, x):
        return self.proj(x) + self.pos_bias


class TransformerPredictor(_Base):
    def __init__(self, seen: torch.Tensor, layers: int = 6, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__(seen)
        self.inp = nn.Linear(DIM, d_model)
        self.row_emb, self.col_emb = nn.Embedding(16, d_model), nn.Embedding(16, d_model)
        self.register_buffer("rows", torch.arange(N_PATCH) // 16, persistent=False)  # raster order
        self.register_buffer("cols", torch.arange(N_PATCH) % 16, persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_CODES)

    def features(self, x):
        h = self.inp(x) + self.row_emb(self.rows) + self.col_emb(self.cols)
        return self.head(self.norm(self.encoder(h)))


def build_model(cfg: dict, seen: torch.Tensor) -> _Base:
    if cfg["model"] == "linear":
        return LinearPredictor(seen)
    return TransformerPredictor(seen, cfg["layers"], cfg["d_model"], cfg["nhead"])


# --------------------------------------------------------------------------- helpers
def check_split_ids(train_ids, splits: dict) -> None:
    """Training ids must be exactly pred_train and disjoint from every other split."""
    if splits.get("source") != "ffhq":
        raise ValueError(f"splits source {splits.get('source')!r} is not ffhq")
    train = set(int(i) for i in train_ids)
    if train != set(splits["splits"]["pred_train"]):
        raise ValueError("training ids are not exactly splits['pred_train']")
    for other in ("ft_train", "val", "spare"):
        if train & set(splits["splits"][other]):
            raise ValueError(f"pred_train overlaps {other}")


def _write_json(path: str, obj, paths: Paths) -> None:
    atomic_write_json(path, obj, paths)


def _append_jsonl(path: str, obj, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _save_model(path: str, model: nn.Module, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, rp)


def _save_npy(path: str, arr: np.ndarray, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    np.save(rp, arr)


def _hq_codebook(paths: Paths, dry_run: bool) -> torch.Tensor:
    if dry_run:
        return torch.randn(N_CODES, DIM, generator=torch.Generator().manual_seed(123))
    from hqprompt import vre
    return vre.load_codebook(paths.associate_ckpt, "hq", paths).float()


@torch.no_grad()
def predict(model: _Base, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred (N,256) int64, top5 (N,256,5) int64) over loader batches (needs 'z')."""
    model.eval()
    p1, p5 = [], []
    for b in loader:
        top = model(b["z"].to(device), amp).topk(5, dim=-1).indices
        p5.append(top.cpu())
        p1.append(top[..., 0].cpu())
    return torch.cat(p1).numpy(), torch.cat(p5).numpy()


def evaluate_split(model, loader, device, amp, codebook, counts) -> dict:
    pred, top5 = predict(model, loader, device, amp)
    y = np.concatenate([b["y"].numpy() for b in loader])
    hit1, hit5 = pred == y, (top5 == y[..., None]).any(-1)
    rank = np.empty(N_CODES, dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(N_CODES)
    per_bucket = {}
    for name, lo, hi in BUCKETS:
        m = (rank[y] >= lo) & (rank[y] < hi)
        per_bucket[name] = {"top1": float(hit1[m].mean()) if m.any() else None, "n": int(m.sum())}
    cb = codebook.cpu()
    l2 = (cb[torch.from_numpy(pred)] - cb[torch.from_numpy(y)]).norm(dim=-1).mean().item()
    return {"top1": float(hit1.mean()), "top5": float(hit5.mean()), "per_bucket_top1": per_bucket,
            "mean_l2_pred_vs_true": l2, "n_images": int(len(y)),
            "frac_true_code_unseen": float((counts[y] == 0).mean())}


def baselines(loader, codebook, counts, per_pos_counts, seen, device) -> dict:
    """No-training baselines with the same seen-code mask."""
    from hqprompt import vre
    seen_idx = torch.nonzero(seen).squeeze(1)
    glob_code = int(np.argmax(counts))
    pos_code = per_pos_counts.argmax(1)  # (256,)
    ys, nearest = [], []
    with torch.no_grad():
        for b in loader:
            ys.append(b["y"].numpy())
            idx, _ = vre.nearest_codes(b["z"].float().to(device), codebook[seen_idx].to(device))
            nearest.append(seen_idx[idx.cpu()].numpy())
    y, nearest = np.concatenate(ys), np.concatenate(nearest)
    return {"most_frequent_global": float((y == glob_code).mean()),
            "most_frequent_per_position": float((y == pos_code[None]).mean()),
            "nearest_hq_code_to_lq_feature": float((nearest == y).mean())}


# --------------------------------------------------------------------------- data setup
def _prepare_dry_run(paths: Paths) -> None:
    if not os.path.exists(paths.splits_json):
        splits = data.make_splits(n_total=sum(DRY_SIZES.values()), sizes=DRY_SIZES)
        data.write_splits(paths, splits, "synthetic", "synthetic", check_files=False)
    if not os.path.exists(f"{paths.cache_dir}/manifest.json"):
        data.write_synthetic_cache(paths, data.load_splits(paths.splits_json))


def _preload(*datasets) -> None:
    """Copy each dataset's z_lq memmap into RAM (fp16, as stored) once; y stays memmapped (small)."""
    need = sum(int(np.prod(d.z.shape)) * 2 for d in datasets)
    print(f"preload: z_lq of pred_train+val needs ~{need / 1e9:.1f} GB RAM (expected ~16 GB)", flush=True)
    for d in datasets:
        d.z = np.asarray(d.z, dtype=np.float16).copy()


def _loader(ds, args, shuffle=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, generator=g if shuffle else None,
                      num_workers=args.num_workers)


def _predict_dir(model, args, paths, device, amp, name, d, out_dir) -> None:
    from hqprompt import vre
    v = vre.random_vre(seed=0) if args.dry_run else vre.load_vre(paths, branches=("lq",), device=str(device))
    v = v.to(device).eval()
    names = data.list_images(d)
    ds = data.ImageDataset(paths, names, source=d, resize="official")
    zs = []
    with torch.no_grad():
        for s in range(0, len(names), args.batch_size):
            x = torch.stack([ds[k]["lq01"] for k in range(s, min(s + args.batch_size, len(names)))]).to(device)
            zs.append(v.lq_features(x).float().cpu())
    z = torch.cat(zs)
    loader = DataLoader([{"z": zi} for zi in z], batch_size=args.batch_size, collate_fn=lambda bs: {
        "z": torch.stack([b["z"] for b in bs])})
    pred, _ = predict(model, loader, device, amp)
    _save_npy(f"{out_dir}/pred/test_{name}.npy", pred.astype(np.int16), paths)
    _write_json(f"{out_dir}/pred/test_{name}.names.json", names, paths)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["linear", "transformer"], default="transformer")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--amp", choices=["bf16", "none"], default=None, help="default bf16 (none for --dry_run)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="default 4 with --no_preload, else 0 (preloaded data needs no workers)")
    ap.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                    help="load pred_train/val z_lq fully into RAM (~16 GB fp16) before training")
    ap.add_argument("--predict_dir", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--predict_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    device = get_device(args.dry_run)
    amp = (args.amp or ("none" if args.dry_run else "bf16")) == "bf16"
    if amp:
        require_bf16(device)
    if args.num_workers is None:
        args.num_workers = 0 if args.preload else 4
    run_dir = f"{paths.predictor_dir}/{args.run_name}"
    test_dirs = []
    for s in args.predict_dir:
        name, sep, d = s.partition("=")
        if not sep or not name or not d:
            raise ValueError(f"--predict_dir expects NAME=DIR, got {s!r}")
        test_dirs.append((name, d))

    seed_everything(args.seed)
    set_numerics(deterministic=True, tf32=False)
    if args.dry_run:
        _prepare_dry_run(paths)
    splits = data.load_splits(paths.splits_json)
    train_ds = data.PredictorDataset(paths, "pred_train")
    val_ds = data.PredictorDataset(paths, "val")
    if args.preload:
        _preload(train_ds, val_ds)
    check_split_ids(train_ds.ids, splits)
    if not np.array_equal(val_ds.ids, data.split_ids(splits, "val")):
        raise ValueError("cache val ids differ from splits['val']")

    counts = np.bincount(np.asarray(train_ds.y).ravel().astype(np.int64), minlength=N_CODES)
    seen = torch.from_numpy(counts > 0)
    cfg = {"model": args.model, "layers": args.layers, "d_model": args.d_model, "nhead": args.nhead}
    model = build_model(cfg, seen).to(device)
    codebook = _hq_codebook(paths, args.dry_run)

    if args.predict_only:
        model.load_state_dict(load_file(f"{run_dir}/best.safetensors"))
    else:
        if os.path.exists(f"{run_dir}/best.safetensors"):
            raise FileExistsError(f"{run_dir} already has a trained model; use --predict_only or a new --run_name")
        _write_json(f"{run_dir}/config.json", {**vars(args), **cfg, "amp": amp,
                                               "splits_sha256": splits["sha256"]}, paths)
        _train(model, args, paths, device, amp, train_ds, val_ds, run_dir)
        model.load_state_dict(load_file(f"{run_dir}/best.safetensors"))
        _final_metrics(model, args, paths, device, amp, train_ds, val_ds, counts, codebook, seen, run_dir)
        _write_json(f"{run_dir}/meta.json", {
            "train_split": "pred_train", "train_ids_sha": sha256_list(train_ds.ids.tolist()),
            "codebook": "hq", "mask_n_seen": int(seen.sum())}, paths)

    for split in ("ft_train", "val"):
        ds = data.PredictorDataset(paths, split)
        pred, _ = predict(model, _loader(ds, args), device, amp)
        assert seen.numpy()[pred].all(), "predicted a masked code"
        _save_npy(f"{run_dir}/pred/{split}.npy", pred.astype(np.int16), paths)
        _write_json(f"{run_dir}/pred/{split}.ids.json", [int(i) for i in ds.ids], paths)
    for name, d in test_dirs:
        _predict_dir(model, args, paths, device, amp, name, d, run_dir)
    return 0


def _train(model, args, paths, device, amp, train_ds, val_ds, run_dir) -> None:
    loader = _loader(train_ds, args, shuffle=True, seed=args.seed)
    val_loader = _loader(val_ds, args)
    total = args.epochs * math.ceil(len(train_ds) / args.batch_size)
    warm = min(args.warmup, max(1, total // 10))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for b in loader:
            logits = model(b["z"].to(device), amp)
            loss = F.cross_entropy(logits.reshape(-1, N_CODES), b["y"].to(device).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
        pred, _ = predict(model, val_loader, device, amp)
        y = np.concatenate([b["y"].numpy() for b in val_loader])
        acc = float((pred == y).mean())
        _append_jsonl(f"{run_dir}/train_log.jsonl", {
            "epoch": epoch + 1, "step": step, "lr": sched.get_last_lr()[0], "train_loss": float(np.mean(losses)),
            "val_top1": acc, "elapsed_s": time.time() - t0}, paths)
        if acc > best:
            best = acc
            _save_model(f"{run_dir}/best.safetensors", model, paths)
    _save_model(f"{run_dir}/last.safetensors", model, paths)


def _final_metrics(model, args, paths, device, amp, train_ds, val_ds, counts, codebook, seen, run_dir) -> None:
    n_sub = min(2000, len(train_ds))
    sub = np.sort(np.random.default_rng(0).choice(len(train_ds), n_sub, replace=False))
    loaders = {"val": _loader(val_ds, args), "pred_train_subset": _loader(Subset(train_ds, sub.tolist()), args)}
    y_all = np.asarray(train_ds.y).astype(np.int64)
    per_pos = np.stack([np.bincount(y_all[:, p], minlength=N_CODES) for p in range(N_PATCH)])
    out = {"model": {k: evaluate_split(model, l, device, amp, codebook, counts) for k, l in loaders.items()},
           "baselines": {k: baselines(l, codebook, counts, per_pos, seen, device) for k, l in loaders.items()},
           "n_seen_codes": int(seen.sum())}
    _write_json(f"{run_dir}/metrics.json", out, paths)


if __name__ == "__main__":
    raise SystemExit(main())
