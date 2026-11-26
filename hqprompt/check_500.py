"""Step 0: sanity checks C1-C3 on faces from pred_train.

C1  Which branch is really the HQ one? Reconstruct clean and LQ faces through both branches (both input
    ranges, both output mappings). Fails (exit 2) unless the hq branch beats the lq branch on clean faces.
C2  Code usage statistics on all selected faces.
C3  Linear predictor of the HQ code from the LQ feature, against no-training baselines.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hqprompt import _steps, data, vre
from hqprompt.common import atomic_write_json, code_fingerprint, env_info, get_device, seed_everything, set_numerics
from hqprompt.evaluate import paired_bootstrap
from hqprompt.paths import Paths, add_path_args, assert_writable, weights_manifest
from hqprompt.train_predictor import LinearPredictor

MODES = ("single", "double")
MAPPINGS = {"pm1": (1.0, 2.0), "double": (3.0, 4.0)}  # out01 = (out + a) / b
GRID_FACES = 8


# --------------------------------------------------------------------------- helpers
def _save_png(arr: np.ndarray, path: str, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    Image.fromarray(arr).save(rp)


def _to_u8(x: torch.Tensor) -> np.ndarray:  # (3,H,W) in [0,1] -> (H,W,3)
    return (x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _lpips_metric(device):
    """(metric or None, reason). LPIPS is optional in C1, so only a missing pyiqa is tolerated."""
    try:
        pyiqa = importlib.import_module("pyiqa")
    except ImportError as e:
        return None, f"pyiqa not importable: {e}"
    return pyiqa.create_metric("lpips", device=device), None


def c1_decision(psnr_hq: np.ndarray, psnr_lq: np.ndarray, n_boot: int = 10000) -> dict:
    """Pass iff the 95% paired-bootstrap CI of mean(hq - lq) clean->clean PSNR lies above 0."""
    boot = paired_bootstrap(psnr_hq, psnr_lq, n_boot=n_boot, seed=0)
    return {"pass": bool(boot["lo"] > 0), **boot}


# --------------------------------------------------------------------------- C1
@torch.no_grad()
def run_c1(model: vre.VRE, ids, paths: Paths, batch_size: int, device, lpips) -> tuple[dict, dict]:
    """Returns (report, raw grid outputs). Stats keyed by (image type, branch, mode, mapping)."""
    stats: dict[tuple, dict[str, list]] = {}
    grid: dict[tuple, list] = {}
    inputs: dict[str, list] = {"clean": [], "lq": []}
    for chunk in _chunks(list(ids), batch_size):
        hq01, lq01 = _steps.load_batch(paths, chunk)
        for img_type, x in (("clean", hq01), ("lq", lq01)):
            x = x.to(device)
            if sum(len(t) for t in inputs[img_type]) < GRID_FACES:
                inputs[img_type].append(x[:GRID_FACES].cpu())
            for mode in MODES:
                x_enc = vre.prepare_input(x, mode)
                for name in ("hq", "lq"):
                    branch = getattr(model, name)
                    idx, _ = vre.nearest_codes(branch.encode_features(x_enc), branch.codebook)
                    out = branch.decode_indices(idx)
                    assert torch.isfinite(out).all(), f"non-finite decoder output ({img_type}, {name}, {mode})"
                    got = sum(len(t) for t in grid.get((img_type, name, mode), []))
                    if got < GRID_FACES:
                        grid.setdefault((img_type, name, mode), []).append(out[:GRID_FACES - got].cpu())
                    for mapping, (a, b) in MAPPINGS.items():
                        y = ((out + a) / b).clamp(0, 1)
                        s = stats.setdefault((img_type, name, mode, mapping), {"psnr": [], "l1": [], "lpips": []})
                        mse = (y - x).pow(2).mean((1, 2, 3)).clamp_min(1e-10)
                        s["psnr"] += (-10 * torch.log10(mse)).tolist()
                        s["l1"] += (y - x).abs().mean((1, 2, 3)).tolist()
                        if lpips is not None:
                            s["lpips"] += lpips(y, x).flatten().tolist()

    table = [{"image_type": t, "branch": b, "mode": m, "mapping": mp,
              "psnr": float(np.mean(s["psnr"])), "l1": float(np.mean(s["l1"])),
              "lpips": float(np.mean(s["lpips"])) if s["lpips"] else None}
             for (t, b, m, mp), s in sorted(stats.items())]

    clean_hq = {(m, mp): np.mean(stats[("clean", "hq", m, mp)]["psnr"]) for m in MODES for mp in MAPPINGS}
    mode, mapping = max(clean_hq, key=clean_hq.get)
    P = lambda t, b: np.asarray(stats[(t, b, mode, mapping)]["psnr"])  # noqa: E731
    primary = c1_decision(P("clean", "hq"), P("clean", "lq"))
    secondary = paired_bootstrap(P("lq", "lq"), P("lq", "hq"), n_boot=10000, seed=0)  # lq branch minus hq branch
    report = {
        "c1_pass": primary["pass"], "recommended_hq_input_mode": mode, "recommended_output_mapping": mapping,
        "primary_hq_minus_lq_clean_psnr": primary,
        "secondary_lq_minus_hq_on_lq_inputs_psnr": {**secondary, "lq_beats_hq": bool(secondary["lo"] > 0)},
        "table": table, "n": len(P("clean", "hq")),
    }
    return report, {"grid": grid, "inputs": inputs, "mode": mode, "mapping": mapping}


def write_c1_grids(raw: dict, paths: Paths) -> None:
    a, b = MAPPINGS[raw["mapping"]]
    for img_type in ("clean", "lq"):
        x = torch.cat(raw["inputs"][img_type])[:GRID_FACES]
        cols = [x]
        for mode in MODES:
            for name in ("hq", "lq"):
                cols.append(((torch.cat(raw["grid"][(img_type, name, mode)]) + a) / b).clamp(0, 1))
        # columns: input, hq-single, lq-single, hq-double, lq-double (2x downsampled)
        rows = [torch.cat([F.avg_pool2d(c[i:i + 1], 2)[0] for c in cols], dim=2) for i in range(len(x))]
        _save_png(_to_u8(torch.cat(rows, dim=1)), f"{paths.checks_dir}/c1_grid_{img_type}.png", paths)


# --------------------------------------------------------------------------- C2 / C3 data
@torch.no_grad()
def collect(model: vre.VRE, ids, paths: Paths, batch_size: int, device, hq_mode: str) -> dict:
    parts = {"z": [], "hq": [], "lq": [], "p77": []}
    for chunk in _chunks(list(ids), batch_size):
        hq01, lq01 = _steps.load_batch(paths, chunk)
        hq01, lq01 = hq01.to(device), lq01.to(device)
        z = model.lq_features(lq01)
        lq_idx, dmin = vre.nearest_codes(z, model.lq.codebook)
        parts["z"].append(z.cpu())
        parts["lq"].append(lq_idx.cpu())
        parts["p77"].append(vre.official_prompt_indices(lq_idx, dmin).cpu())
        parts["hq"].append(model.hq_codes(hq01, hq_mode).cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def _usage(hist: np.ndarray, codebook: torch.Tensor, shared_rows: np.ndarray) -> dict:
    norms = codebook.norm(dim=1).numpy()
    picks = hist.sum()
    return {"n_distinct": int((hist > 0).sum()), "n_picks": int(picks),
            "frac_picks_norm_lt_0.1": float(hist[norms < 0.1].sum() / picks),
            "frac_picks_norm_ge_1": float(hist[norms >= 1].sum() / picks),
            "n_used_codes_in_shared_rows": int(((hist > 0) & shared_rows).sum()),
            "frac_picks_in_shared_rows": float(hist[shared_rows].sum() / picks)}


def _hist_png(hist: np.ndarray, path: str, paths: Paths) -> None:
    h = 200
    bars = np.round(hist / max(hist.max(), 1) * (h - 1)).astype(int)
    img = np.full((h, len(hist), 3), 255, np.uint8)
    for x, v in enumerate(bars):
        if v > 0:
            img[h - v:, x] = (40, 80, 200)
    _save_png(img, path, paths)


def run_c2(model: vre.VRE, c: dict, paths: Paths) -> dict:
    cb = {"hq": model.hq.codebook.cpu(), "lq": model.lq.codebook.cpu()}
    hq_rows = {r.tobytes() for r in cb["hq"].numpy()}
    lq_rows = {r.tobytes() for r in cb["lq"].numpy()}
    shared = {n: np.array([r.tobytes() in (lq_rows if n == "hq" else hq_rows) for r in cb[n].numpy()])
              for n in cb}  # rows bit-identical to a row of the other codebook
    out = {}
    for n in ("hq", "lq"):
        hist = np.bincount(c[n].numpy().ravel(), minlength=1024)
        os.makedirs(paths.checks_dir, exist_ok=True)
        np.save(assert_writable(f"{paths.checks_dir}/c2_hist_{n}.npy", paths), hist)
        _hist_png(hist, f"{paths.checks_dir}/c2_hist_{n}.png", paths)
        out[n] = {**_usage(hist, cb[n], shared[n]), "n_rows_shared_with_other_codebook": int(shared[n].sum())}
    uniq = (c["p77"] >= 0).sum(1).numpy()
    out["official_prompt"] = {"unique_codes_min": int(uniq.min()), "unique_codes_mean": float(uniq.mean()),
                              "unique_codes_max": int(uniq.max()),
                              "frac_images_needing_padding": float((uniq < 77).mean()),
                              "histogram": {str(k): int(v) for k, v in zip(*np.unique(uniq, return_counts=True))}}
    return out


def run_c3(model: vre.VRE, c: dict, device, n_fit: int, epochs: int = 50, lr: float = 1e-3,
           batch_images: int = 16) -> dict:
    """Linear predictor z_lq -> hq_idx on the first n_fit faces, evaluated on the rest (fp32).
    Batches hold 16 faces = 4096 patches."""
    z, y = c["z"], c["hq"]
    z_fit, y_fit, z_te, y_te = z[:n_fit], y[:n_fit], z[n_fit:], y[n_fit:]
    assert len(z_te) > 0, "C3 needs at least one held-out face"
    seen = torch.bincount(y_fit.ravel(), minlength=1024) > 0
    net = LinearPredictor(seen).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for sel in torch.randperm(n_fit, generator=g).split(batch_images):
            logits = net(z_fit[sel].to(device))
            loss = F.cross_entropy(logits.reshape(-1, 1024), y_fit[sel].to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        top5 = torch.cat([net(zb.to(device)).topk(5, dim=-1).indices.cpu() for zb in z_te.split(batch_images)])
        top1 = top5[..., 0]
        nearest_hq = torch.cat([vre.nearest_codes(zb.to(device), model.hq.codebook)[0].cpu()
                                for zb in z_te.split(batch_images)])
    counts = torch.bincount(y_fit.ravel(), minlength=1024)
    pos_counts = torch.zeros(256, 1024, dtype=torch.long)
    pos_counts.scatter_add_(1, y_fit.T, torch.ones_like(y_fit.T))
    top5_global = counts.topk(5).indices
    acc = lambda pred: float((pred == y_te).float().mean())  # noqa: E731
    return {
        "n_fit_faces": int(n_fit), "n_test_faces": int(len(z_te)), "final_train_loss": float(loss),
        "linear": {"top1": acc(top1), "top5": float((top5 == y_te[..., None]).any(-1).float().mean())},
        "baseline_global_mode": {"top1": acc(counts.argmax().expand_as(y_te)),
                                 "top5": float(torch.isin(y_te, top5_global).float().mean())},
        "baseline_per_position_mode": {"top1": acc(pos_counts.argmax(1)[None].expand_as(y_te))},
        "baseline_nearest_hq_code_to_lq_feature": {"top1": acc(nearest_hq)},
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--continue_on_c1_fail", action="store_true",
                    help="run C2/C3 anyway; their reports are labelled valid=false")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = _steps.run_paths(args)
    device = get_device(args.dry_run)
    seed_everything(0)
    set_numerics(deterministic=True, tf32=False)
    paths, splits = _steps.setup_data(paths, args.dry_run)

    pred_ids = data.split_ids(splits, "pred_train")
    assert args.n <= len(pred_ids), f"--n {args.n} > pred_train size {len(pred_ids)}"
    ids = np.sort(np.random.default_rng(0).choice(pred_ids, args.n, replace=False)).astype(int).tolist()
    n_c1 = min(100, args.n)

    if args.dry_run:
        model = vre.random_vre(with_decoder=True).to(device)
        weights = {}
    else:
        model = vre.load_vre(paths, with_decoder=True, device=device)
        weights = weights_manifest(paths)
    lpips, lpips_reason = _lpips_metric(device)
    prov = {"dry_run": args.dry_run, "weights": weights, "env": env_info(), "code": code_fingerprint(),
            "splits_sha256": splits["sha256"], "ids": ids}

    c1, raw = run_c1(model, ids[:n_c1], paths, args.batch_size, device, lpips)
    c1.update(prov, c1_ids_n=n_c1, lpips_available=lpips is not None, lpips_unavailable_reason=lpips_reason)
    write_c1_grids(raw, paths)
    atomic_write_json(f"{paths.checks_dir}/c1.json", c1, paths)
    print(f"C1 {'PASS' if c1['c1_pass'] else 'FAIL'}: recommended hq input mode {c1['recommended_hq_input_mode']!r}, "
          f"mapping {c1['recommended_output_mapping']!r}; hq-lq clean PSNR "
          f"{c1['primary_hq_minus_lq_clean_psnr']['mean_diff']:.3f} dB "
          f"[{c1['primary_hq_minus_lq_clean_psnr']['lo']:.3f}, {c1['primary_hq_minus_lq_clean_psnr']['hi']:.3f}]")
    if not c1["c1_pass"] and not args.continue_on_c1_fail:
        print("C1 FAILED: vqvae.* is not established as the HQ branch. Stopping; nothing else is valid.",
              file=sys.stderr)
        return 2

    c = collect(model, ids, paths, args.batch_size, device, c1["recommended_hq_input_mode"])
    meta = {"valid": c1["c1_pass"], "n": len(ids), "hq_input_mode": c1["recommended_hq_input_mode"]}
    atomic_write_json(f"{paths.checks_dir}/c2.json", {**meta, **run_c2(model, c, paths)}, paths)
    c3 = run_c3(model, c, device, n_fit=int(0.8 * len(ids)))
    atomic_write_json(f"{paths.checks_dir}/c3.json", {**meta, **c3}, paths)
    print(json.dumps({"c3": {k: v for k, v in c3.items() if isinstance(v, dict)}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
