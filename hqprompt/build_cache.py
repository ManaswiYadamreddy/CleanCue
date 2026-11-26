"""Step 2: build the resumable code / latent / teacher cache, and verify it.

    python -m hqprompt.build_cache --split {pred_train,ft_train,val,all} --stage {codes,latents,teacher,all}
                                   --hq_input_mode {single,double}
    python -m hqprompt.build_cache --verify --n 100 --seed 1

Stages (items written, per data.SPLIT_ITEMS):
  codes    hq_idx, lq_idx, lq_prompt77, z_lq   (indices from fp32 features, before any fp16 cast)
  latents  lat_lq       VAE posterior mean of the LQ image
  teacher  lat_teacher  official one-step x0 from the posterior-mean latent and the official 77-token prompt
All data goes through data.CacheWriter, so a rerun only fills missing rows and refuses changed settings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hqprompt import _steps, data, osdface, vre
from hqprompt.common import (atomic_write_json, code_fingerprint, env_info, get_device, seed_everything,
                             require_bf16, set_numerics)
from hqprompt.paths import Paths, add_path_args, weights_manifest
from hqprompt.prompt import PromptEncoder, load_embedding_change, random_embedding_change_sd

STAGE_ITEMS = {"codes": ["hq_idx", "lq_idx", "lq_prompt77", "z_lq"], "latents": ["lat_lq"], "teacher": ["lat_teacher"]}
SPLITS = tuple(data.SPLIT_ITEMS)
FLOAT_ITEMS = ("z_lq", "lat_lq", "lat_teacher")


def settings_for(args) -> dict:
    return {"lq_input_mode": vre.LQ_INPUT_MODE, "hq_input_mode": args.hq_input_mode,
            "code_lookup": "fp32_raw_l2_argmin", "vae_latent": "mean", "scaling_factor": osdface.SCALING_FACTOR,
            "t": osdface.T_FIXED, "teacher_prompt": "official77", "teacher_unet_dtype": args.teacher_dtype,
            "tf32": False, "patch_order": "raster_row_major"}


def check_c1(paths: Paths, hq_input_mode: str) -> None:
    """Refuse to build unless C1 passed and --hq_input_mode is the mode it recommended."""
    try:
        with open(f"{paths.checks_dir}/c1.json") as f:
            c1 = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"{paths.checks_dir}/c1.json missing: run Step 0 (check_500) first") from None
    if not c1["c1_pass"]:
        raise RuntimeError("C1 did not pass: refusing to build a cache with unestablished HQ branch")
    if c1["recommended_hq_input_mode"] != hq_input_mode:
        raise RuntimeError(f"--hq_input_mode {hq_input_mode!r} != c1.json recommendation "
                           f"{c1['recommended_hq_input_mode']!r}")


class _Rows(Dataset):
    """FFHQ images for given ids: dict with lq01 (and hq01 if wanted)."""

    def __init__(self, paths: Paths, ids, hq: bool):
        self.paths, self.ids, self.hq = paths, [int(i) for i in ids], hq

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        out = {"lq01": data.load_image_01(data.ffhq_lq_path(self.paths, i))}
        if self.hq:
            out["hq01"] = data.load_image_01(data.ffhq_hq_path(self.paths, i))
        return out


def _loader(paths, ids, batch_size, hq, workers):
    return DataLoader(_Rows(paths, ids, hq), batch_size=batch_size, shuffle=False, num_workers=workers)


class Models:
    """Lazy model loading, so each stage loads only what it needs."""

    def __init__(self, paths: Paths, device, dry_run: bool):
        self.paths, self.device, self.dry_run = paths, device, dry_run
        self._vre = self._osd = self._ec = None

    @property
    def enc(self) -> vre.VRE:
        if self._vre is None:
            self._vre = (vre.random_vre() if self.dry_run else vre.load_vre(self.paths, device=self.device)).to(self.device)
        return self._vre

    @property
    def osd(self):
        if self._osd is None:
            models, self.report = (_steps.tiny_merged_models() if self.dry_run
                                   else osdface.load_osdface(self.paths, self.device, merge_lora=True))
            models.unet.to(self.device), models.vae.to(self.device)
            self._osd = models
        return self._osd

    @property
    def ec_sd(self) -> dict:
        if self._ec is None:
            self._ec = random_embedding_change_sd() if self.dry_run else load_embedding_change(self.paths)
        return self._ec

    def teacher_prompt(self) -> PromptEncoder:
        return PromptEncoder("official", self.enc.lq.codebook.cpu(), self.ec_sd).to(self.device)


# --------------------------------------------------------------------------- computations (shared by build and verify)
@torch.no_grad()
def compute_codes(model: vre.VRE, hq01, lq01, hq_input_mode: str) -> dict:
    """Everything the codes stage stores, plus the fp32 distances needed to explain mismatches."""
    z = model.lq_features(lq01)
    lq_idx, dmin = vre.nearest_codes(z, model.lq.codebook)
    hq_z = model.hq.encode_features(vre.prepare_input(hq01, hq_input_mode))
    hq_idx, _ = vre.nearest_codes(hq_z, model.hq.codebook)
    return {"z": z, "hq_z": hq_z, "hq_idx": hq_idx, "lq_idx": lq_idx, "dmin": dmin,
            "lq_prompt77": vre.official_prompt_indices(lq_idx, dmin)}


@torch.no_grad()
def compute_latents(osd, lq01) -> torch.Tensor:
    return osdface.vae_encode(osd.vae, lq01 * 2 - 1, "mean")


@torch.no_grad()
def compute_teacher(osd, prompt_enc, lat_lq, prompt77, teacher_dtype: str) -> torch.Tensor:
    amp = torch.bfloat16 if teacher_dtype == "bf16" else None
    return osdface.one_step(osd.unet, lat_lq, prompt_enc(prompt77), osd.alphas_cumprod, amp_dtype=amp)


def _np(x: torch.Tensor, item: str) -> np.ndarray:
    a = x.cpu().numpy().astype(data.ITEM_SPECS[item][1])
    if item in FLOAT_ITEMS:
        assert np.isfinite(a).all(), f"non-finite values in {item}"
    return a


# --------------------------------------------------------------------------- build
def _disk_check(paths: Paths, splits: dict, todo: list[tuple[str, str]]) -> None:
    need = 0
    for split, stage in todo:
        for item in STAGE_ITEMS[stage]:
            if item in data.SPLIT_ITEMS[split] and not os.path.exists(f"{paths.cache_dir}/{split}/{item}.npy"):
                tail, dtype = data.ITEM_SPECS[item]
                need += len(splits["splits"][split]) * int(np.prod(tail)) * np.dtype(dtype).itemsize
    free = shutil.disk_usage(paths.out_root).free
    print(f"cache needs {need / 1e9:.2f} GB new; {free / 1e9:.2f} GB free under {paths.out_root} "
          f"(check the shared quota manually: disk_usage may not reflect it)")
    if free < 1.2 * need:
        raise RuntimeError(f"not enough disk: need 1.2 x {need / 1e9:.2f} GB, have {free / 1e9:.2f} GB")


def build_split_stage(args, paths, splits, models: Models, split: str, stage: str, settings: dict,
                      weights: dict) -> None:
    items = [i for i in STAGE_ITEMS[stage] if i in data.SPLIT_ITEMS[split]]
    if not items:
        return
    ids = data.split_ids(splits, split)
    writer = data.CacheWriter(paths, split, ids, items, settings, weights, splits["sha256"])
    pend = {i: writer.pending(i) for i in items}
    rows = np.unique(np.concatenate(list(pend.values())))
    bs = args.teacher_batch_size if stage == "teacher" else args.batch_size
    chunks = [rows[k:k + bs] for k in range(0, len(rows), bs)]
    if args.limit_batches is not None:
        chunks = chunks[:args.limit_batches]
    print(f"[{split}/{stage}] {len(rows)} rows pending of {len(ids)}; {len(chunks)} batches", flush=True)

    reader = data.CacheReader(paths, split, ["lq_prompt77"]) if stage == "teacher" else None
    prompt_enc = models.teacher_prompt() if stage == "teacher" else None
    loader = _loader(paths, ids[np.concatenate(chunks)] if chunks else [], bs, stage == "codes", args.num_workers)
    for done, (chunk, batch) in enumerate(zip(chunks, loader), 1):
        lq01 = batch["lq01"].to(models.device)
        if stage == "codes":
            c = compute_codes(models.enc, batch["hq01"].to(models.device), lq01, args.hq_input_mode)
            vals = {"hq_idx": c["hq_idx"], "lq_idx": c["lq_idx"], "lq_prompt77": c["lq_prompt77"], "z_lq": c["z"]}
        elif stage == "latents":
            vals = {"lat_lq": compute_latents(models.osd, lq01)}
        else:
            p77 = torch.from_numpy(np.asarray(reader.array("lq_prompt77")[chunk], dtype=np.int64))
            lat = compute_latents(models.osd, lq01)
            vals = {"lat_teacher": compute_teacher(models.osd, prompt_enc, lat, p77.to(models.device),
                                                   args.teacher_dtype)}
        for item in items:
            keep = np.isin(chunk, pend[item])
            if keep.any():
                writer.write(item, chunk[keep], _np(vals[item], item)[keep])
        if done % 20 == 0 or done == len(chunks):
            print(f"  {done}/{len(chunks)} batches", flush=True)
    writer.finalize()


def _clear_verification(paths: Paths, build_info: dict | None = None) -> None:
    mp = f"{paths.cache_dir}/manifest.json"
    with open(mp) as f:
        m = json.load(f)
    m["verification"] = None  # the cache changed; any earlier verification is stale
    if build_info:  # batch shapes used for the build (outside "settings": resume checks are unchanged)
        old = m.get("build") or {}
        new = {**old, **build_info}
        if any(k in old and old[k] != v for k, v in build_info.items()):
            new["mixed_batch_sizes"] = True  # resumed with another batch size: verify cannot match every row
        m["build"] = new
    atomic_write_json(mp, m, paths)


def build(args, paths, splits, models: Models) -> None:
    splits_todo = SPLITS if args.split == "all" else (args.split,)
    stages = tuple(STAGE_ITEMS) if args.stage == "all" else (args.stage,)
    settings = settings_for(args)
    weights = {} if args.dry_run else weights_manifest(paths)
    _disk_check(paths, splits, [(sp, st) for st in stages for sp in splits_todo])
    for stage in stages:
        for split in splits_todo:
            build_split_stage(args, paths, splits, models, split, stage, settings, weights)
    info = {}
    if set(stages) & {"codes", "latents"}:
        info["batch_size"] = args.batch_size
    if "teacher" in stages:
        info["teacher_batch_size"] = args.teacher_batch_size
    _clear_verification(paths, info)


# --------------------------------------------------------------------------- verify
def _code_gap(z_row: torch.Tensor, codebook: torch.Tensor, a: int, b: int) -> float:
    """|d(z, code a) - d(z, code b)| in float64, with raw squared L2."""
    z = z_row.double()
    return float(abs((z - codebook[a].double()).pow(2).sum() - (z - codebook[b].double()).pow(2).sum()))


def verify(args, paths, splits, models: Models) -> int:
    with open(f"{paths.cache_dir}/manifest.json") as f:
        manifest = json.load(f)
    settings = manifest["settings"]
    readers = {s: data.CacheReader(paths, s) for s in manifest["items"]}  # all rows must be done
    pool = [(s, r) for s, rd in readers.items() for r in range(len(rd.ids))]
    pick = np.random.default_rng(args.seed).choice(len(pool), min(args.n, len(pool)), replace=False)
    wanted: dict[str, list[int]] = {}
    for k in sorted(pick):
        wanted.setdefault(pool[k][0], []).append(pool[k][1])

    if settings.get("teacher_unet_dtype") == "bf16":
        require_bf16(models.device)
    build_info = manifest.get("build") or {}
    if build_info.get("mixed_batch_sizes"):
        print("WARNING: cache was resumed with different batch sizes; float items may mismatch at batch edges")
    bs_codes = int(build_info.get("batch_size") or args.batch_size)
    bs_teacher = int(build_info.get("teacher_batch_size") or args.teacher_batch_size)
    print(f"verify: recomputing with the build batch sizes: codes/latents {bs_codes}, teacher {bs_teacher}")

    prompt_enc = models.teacher_prompt()
    mismatches, max_err = [], {}

    def compare(rd, split, item, new, rows, off):
        """new: array for the full block starting at row `off`; compare only the sampled `rows`."""
        ids = rd.ids
        stored = np.asarray(rd.array(item)[np.asarray(rows)])
        for j, row in enumerate(rows):
            n_ = new[row - off]
            where = {"split": split, "row": int(row), "id": int(ids[row]), "item": item}
            if item in FLOAT_ITEMS:
                rec32, st32 = n_.astype(np.float32), stored[j].astype(np.float32)
                err = np.abs(rec32 - st32)
                max_err[item] = max(max_err.get(item, 0.0), float(err.max()))
                if (err > 1e-3 + 2.0 ** -10 * np.abs(rec32)).any():
                    mismatches.append({**where, "max_abs_err": float(err.max())})
            elif not np.array_equal(n_.astype(stored.dtype), stored[j]):
                bad = np.flatnonzero(n_.astype(stored.dtype) != stored[j])
                entry = {**where, "n_mismatch": int(len(bad)), "positions": bad[:20].tolist()}
                extra = extras.get(item)
                if extra is not None:
                    entry.update(extra(row - off, bad, stored[j], n_))
                mismatches.append(entry)

    for split, wrows in wanted.items():
        rd = readers[split]
        ids = rd.ids
        has = set(rd.items)
        n_rows = len(ids)
        code_items = has & {"hq_idx", "lq_idx", "lq_prompt77", "z_lq", "lat_lq"}
        for bs, group in ((bs_codes, code_items), (bs_teacher, has & {"lat_teacher"})):
            if not group:
                continue
            for blk in sorted({r // bs for r in wrows}):
                off, end = blk * bs, min((blk + 1) * bs, n_rows)
                rows = [r for r in wrows if off <= r < end]
                # the full original build batch containing the sampled rows
                batch = next(iter(_loader(paths, ids[off:end], end - off, True, 0)))
                hq01, lq01 = batch["hq01"].to(models.device), batch["lq01"].to(models.device)
                out, extras = {}, {}
                if group & {"hq_idx", "lq_idx", "lq_prompt77", "z_lq"}:
                    rec = compute_codes(models.enc, hq01, lq01, settings["hq_input_mode"])
                    codebooks = {"hq_idx": (rec["hq_z"], models.enc.hq.codebook),
                                 "lq_idx": (rec["z"], models.enc.lq.codebook)}
                    out.update({"hq_idx": rec["hq_idx"], "lq_idx": rec["lq_idx"],
                                "lq_prompt77": rec["lq_prompt77"], "z_lq": rec["z"]})

                    def code_gap(item):
                        def f(j, bad, st, nw):
                            z, cb = codebooks[item]
                            return {"distance_gaps": [_code_gap(z[j].flatten(1).T[p].cpu(), cb.cpu(),
                                                                int(st[p]), int(nw[p])) for p in bad[:20]]}
                        return f
                    extras = {k: code_gap(k) for k in codebooks}

                    def p77_gap(j, bad, st, nw):
                        d = torch.sort(rec["dmin"][j].flatten().double().cpu()).values
                        return {"dmin_gaps": {"min_adjacent_sorted_dmin_gap": float((d[1:] - d[:-1]).min())}}
                    extras["lq_prompt77"] = p77_gap
                if "lat_lq" in group:
                    out["lat_lq"] = compute_latents(models.osd, lq01)
                if "lat_teacher" in group:
                    lat = compute_latents(models.osd, lq01)
                    p77 = torch.from_numpy(np.asarray(rd.array("lq_prompt77")[off:end], dtype=np.int64))
                    out["lat_teacher"] = compute_teacher(models.osd, prompt_enc, lat, p77.to(models.device),
                                                         settings["teacher_unet_dtype"])
                for item in group & set(out):
                    compare(rd, split, item, out[item].cpu().numpy(), rows, off)
    report = {"passed": not mismatches, "n_checked": len(pick), "seed": args.seed, "settings": settings,
              "max_abs_err_float_items": max_err, "mismatches": mismatches[:100], "n_mismatches": len(mismatches),
              "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "code": code_fingerprint(), "env": env_info()}
    atomic_write_json(f"{paths.cache_dir}/verify_report.json", report, paths)
    manifest["verification"] = {k: report[k] for k in ("passed", "n_checked", "seed", "time")}
    atomic_write_json(f"{paths.cache_dir}/manifest.json", manifest, paths)
    print(f"verify: {len(pick)} rows checked, {len(mismatches)} mismatches -> "
          f"{'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=[*SPLITS, "all"])
    ap.add_argument("--stage", choices=[*STAGE_ITEMS, "all"])
    ap.add_argument("--hq_input_mode", choices=["single", "double"], help="no default: must equal c1.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--teacher_batch_size", type=int, default=4)
    ap.add_argument("--teacher_dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit_batches", type=int, default=None, help="stop each split/stage early (tests)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="--verify: rows to recompute")
    ap.add_argument("--seed", type=int, default=1, help="--verify: sampling seed")
    ap.add_argument("--dry_run", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if not args.verify and None in (args.split, args.stage, args.hq_input_mode):
        ap.error("--split, --stage and --hq_input_mode are required (unless --verify)")
    if args.dry_run:
        args.num_workers = 0

    paths = _steps.run_paths(args)
    if not args.verify and not args.dry_run:
        check_c1(paths, args.hq_input_mode)  # fail before touching a GPU or any model
    device = get_device(args.dry_run)
    seed_everything(0)
    set_numerics(deterministic=True, tf32=False)
    paths, splits = _steps.setup_data(paths, args.dry_run)
    models = Models(paths, device, args.dry_run)
    if args.teacher_dtype == "bf16" and not args.verify:
        require_bf16(device)
    if args.verify:
        return verify(args, paths, splits, models)
    build(args, paths, splits, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
