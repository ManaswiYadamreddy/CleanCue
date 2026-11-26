"""Evaluation CLI: score / compare / gate / diff / table, plus paired-bootstrap helpers.

Everything except `score` and the LPIPS part of `diff` needs only numpy; pyiqa, facexlib and
face_alignment are imported lazily by the metric classes. Metric failures raise. The only NaN is
LMD where face detection fails, with the reason recorded in `lmd_fail`.

NOTE: the default CelebA LQ set is self-generated (not the official DAEFR set); say so in reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

from hqprompt.common import atomic_write_json, env_info
from hqprompt.paths import Paths, add_path_args, assert_writable, paths_from_args

DIRECTION = {"lpips": -1, "dists": -1, "deg": -1, "lmd": -1, "niqe": -1, "fid": -1,
             "musiq": +1, "clipiqa": +1, "maniqa": +1, "psnr": +1, "ssim": +1}
FULL_REF = ("lpips", "dists", "psnr", "ssim")
NO_REF = ("musiq", "niqe", "clipiqa", "maniqa")
IMG_EXT = (".png", ".jpg", ".jpeg")
SUFFIXES = ("_LQ", "_lq", "_bicubic", "_deg")
GATE_LOWER = ("lpips", "deg", "lmd")   # lower is better; the decision metrics besides MUSIQ


# ----------------------------------------------------------------------------- statistics
def read_per_image(out_dir: str) -> dict[str, dict[str, float]]:
    """O/per_image.csv -> {name: {metric: float}} (the string column lmd_fail is skipped)."""
    out = {}
    with open(os.path.join(out_dir, "per_image.csv"), newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {k: float(v) for k, v in row.items()
                                if k not in ("name", "lmd_fail") and v != ""}
    return out


def read_lmd_fail(out_dir: str) -> dict[str, str]:
    with open(os.path.join(out_dir, "per_image.csv"), newline="") as f:
        return {r["name"]: r.get("lmd_fail", "") for r in csv.DictReader(f)}


def common_names(tables: list[dict], metric: str) -> list[str]:
    """Sorted names where `metric` is present and finite in every table."""
    sets = [{n for n, r in t.items() if metric in r and math.isfinite(r[metric])} for t in tables]
    return sorted(set.intersection(*sets)) if sets else []


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        raise ValueError("paired_bootstrap: empty input")
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


def _boot_on(tables, metric, names, n_boot, seed):
    """Bootstrap of mean(tables[0] - tables[1]) on `names`."""
    a = [tables[0][n][metric] for n in names]
    b = [tables[1][n][metric] for n in names]
    return paired_bootstrap(a, b, n_boot, seed)


def gate_decision(row4: dict, row2: dict, row3: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Step 4 gate on per-image tables {name: {metric: value}}.

    Strict rule (decides): row4 must have CI_hi(mean(row4 - c)) < 0 against BOTH comparators on at
    least two of lpips/deg/lmd, AND MUSIQ must not be significantly lower than either comparator.
    The lenient variant (at least two wins against each comparator separately) is reported only.
    """
    tables = {"row4": row4, "row2": row2, "row3": row3}
    name_sets = [set(t) for t in tables.values()]
    if not (name_sets[0] == name_sets[1] == name_sets[2]):
        raise ValueError("gate: the three runs do not cover identical image names")
    all_names = sorted(name_sets[0])
    main_metrics = ("lpips", "deg", "musiq", "dists")
    ci: dict = {}
    for c in ("row2", "row3"):
        ci[c] = {}
        for m in main_metrics:
            names = common_names(list(tables.values()), m)
            if names != all_names:
                raise ValueError(f"gate: metric {m!r} missing or non-finite for some of the "
                                 f"{len(all_names)} images in at least one run")
            ci[c][m] = _boot_on((row4, tables[c]), m, names, n_boot, seed)
        lmd_names = common_names(list(tables.values()), "lmd")
        if not lmd_names:
            raise ValueError("gate: no image has a finite LMD in all three runs")
        ci[c]["lmd"] = _boot_on((row4, tables[c]), "lmd", lmd_names, n_boot, seed)
    win = {c: {m: ci[c][m]["hi"] < 0 for m in GATE_LOWER} for c in ci}
    musiq_worse = {c: ci[c]["musiq"]["hi"] < 0 for c in ci}
    n_joint = sum(win["row2"][m] and win["row3"][m] for m in GATE_LOWER)
    strict_go = n_joint >= 2 and not any(musiq_worse.values())
    lenient_go = all(sum(win[c].values()) >= 2 for c in win) and not any(musiq_worse.values())
    return {
        "go": bool(strict_go), "strict_go": bool(strict_go), "lenient_go": bool(lenient_go),
        "n_joint_wins": int(n_joint), "win": win, "musiq_clearly_worse": musiq_worse, "ci": ci,
        "n_images": len(all_names), "lmd_common_n": len(lmd_names),
        "lmd_nan_per_run": {k: sum(not math.isfinite(r.get("lmd", float("nan"))) for r in t.values())
                            for k, t in tables.items()},
        "n_boot": n_boot, "seed": seed,
        "rule": "strict: >=2 of lpips/deg/lmd with CI_hi(row4-c)<0 against BOTH row2 and row3, "
                "and MUSIQ not significantly lower (CI_hi<0) than either. Lower is better for "
                "lpips/deg/lmd/dists; dists is informational.",
    }


def gate_markdown(g: dict) -> str:
    lines = [f"# Gate: {'GO' if g['go'] else 'NO-GO'} (strict); lenient: {'go' if g['lenient_go'] else 'no-go'}",
             "", f"n images = {g['n_images']}, LMD common n = {g['lmd_common_n']}, "
                 f"bootstrap {g['n_boot']} (seed {g['seed']}). Difference = row4 - comparator; "
                 "a win needs the 95% CI upper bound below 0 (lower is better).", "",
             "| comparator | metric | mean diff | 95% CI | n | win |", "|---|---|---|---|---|---|"]
    for c, ms in g["ci"].items():
        for m, r in ms.items():
            win = g["win"][c].get(m, "")
            if m == "musiq":
                win = f"clearly worse: {g['musiq_clearly_worse'][c]}"
            lines.append(f"| {c} | {m} | {r['mean_diff']:.4f} | [{r['lo']:.4f}, {r['hi']:.4f}] | {r['n']} | {win} |")
    lines += ["", f"Joint wins (lpips/deg/lmd against both): {g['n_joint_wins']}",
              f"LMD NaN per run: {g['lmd_nan_per_run']}",
              "Note: the CelebA LQ set used elsewhere is self-generated."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- io helpers
def _norm_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    for s in SUFFIXES:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem


def _image_dir(d: str) -> str:
    sub = os.path.join(d, "images")
    return sub if os.path.isdir(sub) else d


def _index_images(d: str) -> dict[str, str]:
    d = _image_dir(d)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    out = {}
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(IMG_EXT):
            n = _norm_stem(f)
            if n in out:
                raise ValueError(f"duplicate normalized name {n!r} in {d}")
            out[n] = os.path.join(d, f)
    return out


def _load01(path: str):
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _batches(names, size):
    for i in range(0, len(names), size):
        yield names[i:i + size]


def _write_text(path: str, text: str, paths: Paths) -> None:
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        f.write(text)


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _device(arg):
    import torch
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- score
def cmd_score(args, paths: Paths) -> int:
    restored = _index_images(args.restored_dir)
    metrics = list(args.metrics)
    unknown = set(metrics) - set(DIRECTION) - {"fid"}
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}")
    if "fid" in metrics:
        metrics.remove("fid")
        if not args.fid_ref:
            raise ValueError("metric 'fid' needs --fid_ref")
    needs_hq = [m for m in metrics if m in FULL_REF + ("deg", "lmd")]
    hq_dir = args.hq_dir
    if args.split:
        if args.split != "val":
            raise ValueError("only --split val is supported")
        with open(paths.splits_json) as f:
            ids = json.load(f)["splits"]["val"]
        expected = {f"{int(i):05d}" for i in ids}
        got = set(restored)
        if got != expected:
            raise ValueError(f"restored names != val ids: missing {sorted(expected - got)[:10]}, "
                             f"extra {sorted(got - expected)[:10]}")
        hq_dir = hq_dir or paths.ffhq_hq
    elif not args.names_from_dir:
        raise ValueError("give --split val or --names_from_dir")
    names = sorted(restored)
    hq = _index_images(hq_dir) if needs_hq else {}
    if needs_hq:
        if not hq_dir:
            raise ValueError(f"metrics {needs_hq} need --hq_dir")
        missing = [n for n in names if n not in hq]
        if missing:
            raise FileNotFoundError(f"{len(missing)} HQ images missing, e.g. {missing[:5]}")

    dev = _device(args.device)
    iqa_names = [m for m in metrics if m in FULL_REF + NO_REF]
    iqa = ident = lmdm = None
    if iqa_names:
        from hqprompt.metrics.iqa import IQAMetrics
        iqa = IQAMetrics(iqa_names, dev)
    if "deg" in metrics:
        from hqprompt.metrics.identity import IdentityDistance
        ident = IdentityDistance(paths, dev)
    if "lmd" in metrics:
        from hqprompt.metrics.landmark import LandmarkDistance
        lmdm = LandmarkDistance(dev)

    import torch
    rows: dict[str, dict] = {n: {} for n in names}
    for chunk in _batches(names, args.batch_size):
        x = torch.stack([_load01(restored[n]) for n in chunk])
        h = torch.stack([_load01(hq[n]) for n in chunk]) if needs_hq else None
        if h is not None and h.shape != x.shape:
            raise ValueError(f"restored {tuple(x.shape)} and HQ {tuple(h.shape)} shapes differ")
        res = iqa.compute(x, h) if iqa else {}
        if ident:
            res["deg"] = ident.compute(x, h)["deg"]
        for m, v in res.items():
            for n, val in zip(chunk, v):
                rows[n][m] = float(val)
        if lmdm:
            for n, r in zip(chunk, lmdm.compute(x, h)):
                rows[n].update(r)

    cols = [m for m in metrics]
    out_dir = args.out_dir
    csv_cols = ["name"] + cols + (["lmd_fail"] if "lmd" in cols else [])
    lines = []
    for n in names:
        lines.append(",".join([n] + [repr(rows[n][m]) if m != "lmd_fail" else rows[n][m]
                                     for m in csv_cols[1:]]))
    _write_text(os.path.join(out_dir, "per_image.csv"), ",".join(csv_cols) + "\n" + "\n".join(lines) + "\n", paths)

    summary: dict = {"metrics": {}, "inputs": {"restored_dir": os.path.abspath(args.restored_dir),
                                               "hq_dir": hq_dir, "n_images": len(names),
                                               "split": args.split, "argv": sys.argv[1:]},
                     "env_info": env_info()}
    for m in cols:
        v = np.array([rows[n][m] for n in names], dtype=np.float64)
        v = v[np.isfinite(v)]
        summary["metrics"][m] = {"mean": float(v.mean()) if v.size else float("nan"),
                                 "std": float(v.std()) if v.size else float("nan"), "n": int(v.size)}
    if "deg" in cols:
        from hqprompt.metrics.identity import DEG_PREPROCESS
        summary["deg_preprocess"] = DEG_PREPROCESS
    if "lmd" in cols:
        fails = [rows[n]["lmd_fail"] for n in names]
        summary["lmd"] = {"n_valid": fails.count("") , "n_fail_restored": fails.count("restored"),
                          "n_fail_hq": fails.count("hq"), "n_fail_both": fails.count("both")}
    if args.fid_ref:
        from hqprompt.metrics.fid import FIDScorer
        summary["fid"] = {"value": FIDScorer(dev, paths).compute(_image_dir(args.restored_dir), args.fid_ref),
                          "reference": args.fid_ref}
    atomic_write_json(os.path.join(out_dir, "summary.json"), summary, paths)
    print(json.dumps({k: summary[k] for k in summary if k in ("metrics", "lmd", "fid")}, indent=2))
    return 0


# ----------------------------------------------------------------------------- compare
def cmd_compare(args, paths: Paths) -> int:
    ta, tb = read_per_image(args.a), read_per_image(args.b)
    extra = [read_per_image(d) for d in (args.lmd_common or [])]
    metrics = args.metrics or sorted(set.intersection(*[{m for r in t.values() for m in r}
                                                       for t in (ta, tb)]))
    out = {"a": args.a, "b": args.b, "lmd_common": args.lmd_common or [], "n_boot": args.n_boot,
           "seed": args.seed, "metrics": {}}
    for m in metrics:
        tables = [ta, tb] + (extra if m == "lmd" else [])
        names = common_names(tables, m)
        if not names:
            raise ValueError(f"compare: no common finite values for {m!r}")
        r = _boot_on((ta, tb), m, names, args.n_boot, args.seed)
        r["direction"] = DIRECTION.get(m)
        # a significantly better than b: CI excludes 0 on the favourable side
        r["a_significantly_better"] = (None if m not in DIRECTION else
                                       (r["hi"] < 0 if DIRECTION[m] < 0 else r["lo"] > 0))
        out["metrics"][m] = r
    atomic_write_json(args.out, out, paths)
    print(json.dumps(out["metrics"], indent=2))
    return 0


# ----------------------------------------------------------------------------- gate
def cmd_gate(args, paths: Paths) -> int:
    g = gate_decision(read_per_image(args.row4), read_per_image(args.row2), read_per_image(args.row3),
                      args.n_boot, args.seed)
    g["inputs"] = {"row4": args.row4, "row2": args.row2, "row3": args.row3}
    atomic_write_json(args.out, g, paths)
    if args.md:
        _write_text(args.md, gate_markdown(g), paths)
    print(gate_markdown(g))
    return 0


# ----------------------------------------------------------------------------- diff
def _quantiles(v: np.ndarray) -> dict:
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(v.mean()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4])}


def cmd_diff(args, paths: Paths) -> int:
    import torch
    ia, ib = _index_images(args.a), _index_images(args.b)
    if set(ia) != set(ib):
        raise ValueError(f"diff: name sets differ ({len(set(ia) ^ set(ib))} names)")
    names = sorted(ia)
    from hqprompt.metrics.iqa import IQAMetrics
    lp = IQAMetrics(["lpips"], _device(args.device))
    mad, lpips = [], []
    for chunk in _batches(names, args.batch_size):
        xa = torch.stack([_load01(ia[n]) for n in chunk])
        xb = torch.stack([_load01(ib[n]) for n in chunk])
        if xa.shape != xb.shape:
            raise ValueError("diff: image shapes differ")
        mad += (xa - xb).abs().mean(dim=(1, 2, 3)).double().tolist()
        lpips += lp.compute(xa, xb)["lpips"].tolist()
    out = {"a": args.a, "b": args.b, "n": len(names),
           "mean_abs_diff_01": _quantiles(np.array(mad)), "lpips": _quantiles(np.array(lpips)),
           "per_image": {n: {"mean_abs": m, "lpips": l} for n, m, l in zip(names, mad, lpips)}}
    atomic_write_json(args.out, out, paths)
    print(json.dumps({k: out[k] for k in ("n", "mean_abs_diff_01", "lpips")}, indent=2))
    return 0


# ----------------------------------------------------------------------------- table
def cmd_table(args, paths: Paths) -> int:
    runs = []
    for spec in args.run:
        name, _, d = spec.partition("=")
        if not d:
            raise ValueError(f"--run expects NAME=DIR, got {spec!r}")
        runs.append((name, d))
    tables = [read_per_image(d) for _, d in runs]
    fails = [read_lmd_fail(d) for _, d in runs]
    metrics = [m for m in DIRECTION if all(any(m in r for r in t.values()) for t in tables)]
    lmd_common = set(common_names(tables, "lmd")) if "lmd" in metrics else set()
    header = ["run", "n"] + metrics
    if "lmd" in metrics:
        header += ["lmd_common_n", "lmd_valid", "fail_restored", "fail_hq", "fail_both"]
    header.append("fid")
    rows = []
    for (name, d), t, fl in zip(runs, tables, fails):
        row = {"run": name, "n": len(t)}
        for m in metrics:
            names = sorted(lmd_common) if m == "lmd" else [n for n, r in t.items() if math.isfinite(r.get(m, float("nan")))]
            row[m] = float(np.mean([t[n][m] for n in names])) if names else float("nan")
        if "lmd" in metrics:
            vals = list(fl.values())
            row.update(lmd_common_n=len(lmd_common), lmd_valid=vals.count(""),
                       fail_restored=vals.count("restored"), fail_hq=vals.count("hq"),
                       fail_both=vals.count("both"))
        sj = os.path.join(d, "summary.json")
        row["fid"] = json.load(open(sj)).get("fid", {}).get("value", "") if os.path.exists(sj) else ""
        rows.append(row)
    base = os.path.splitext(args.out)[0]
    csv_text = ",".join(header) + "\n" + "\n".join(",".join(str(r.get(h, "")) for h in header) for r in rows) + "\n"
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(_fmt(r.get(h, "")) for h in header) + " |" for r in rows]
    md.append("")
    md.append("LMD is the mean over the images detected in every listed run. "
              "The CelebA LQ set is self-generated, not the official one.")
    _write_text(base + ".csv", csv_text, paths)
    _write_text(base + ".md", "\n".join(md) + "\n", paths)
    print("\n".join(md))
    return 0


# ----------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hqprompt.evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name):
        sp = sub.add_parser(name)
        add_path_args(sp)
        return sp

    s = add("score")
    s.add_argument("--restored_dir", required=True)
    s.add_argument("--hq_dir")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--split")
    g.add_argument("--names_from_dir", action="store_true")
    s.add_argument("--metrics", nargs="+", required=True)
    s.add_argument("--fid_ref")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--batch_size", type=int, default=8)
    s.add_argument("--device")

    c = add("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--lmd_common", nargs="*")
    c.add_argument("--metrics", nargs="*")
    c.add_argument("--n_boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)

    g_ = add("gate")
    for r in ("row4", "row2", "row3"):
        g_.add_argument(f"--{r}", required=True)
    g_.add_argument("--n_boot", type=int, default=10000)
    g_.add_argument("--seed", type=int, default=0)
    g_.add_argument("--out", required=True)
    g_.add_argument("--md")

    d = add("diff")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--batch_size", type=int, default=8)
    d.add_argument("--device")

    t = add("table")
    t.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    t.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = paths_from_args(args)
    return {"score": cmd_score, "compare": cmd_compare, "gate": cmd_gate,
            "diff": cmd_diff, "table": cmd_table}[args.cmd](args, paths)


if __name__ == "__main__":
    sys.exit(main())
