import csv
import json
import os

import numpy as np
import pytest

from hqprompt import evaluate as ev


def make_table(n, **offsets):
    """Per-image table: lpips/deg/lmd/musiq/dists = base noise + offset."""
    rng = np.random.default_rng(0)
    base = {m: rng.normal(1.0, 0.1, n) for m in ("lpips", "deg", "lmd", "musiq", "dists")}
    return base


def table(base, delta, n=60, nan_lmd=()):
    t = {}
    for i in range(n):
        r = {m: float(base[m][i] + delta.get(m, 0.0)) for m in base}
        if i in nan_lmd:
            r["lmd"] = float("nan")
        t[f"{i:05d}"] = r
    return t


@pytest.fixture
def base():
    return make_table(60)


def test_gate_all_better_is_go(base):
    r4 = table(base, {"lpips": -0.2, "deg": -0.2, "lmd": -0.2})
    g = ev.gate_decision(r4, table(base, {}), table(base, {}), n_boot=500)
    assert g["go"] and g["n_joint_wins"] == 3 and g["lenient_go"]


def test_gate_one_metric_is_no_go(base):
    r4 = table(base, {"lpips": -0.2})
    g = ev.gate_decision(r4, table(base, {}), table(base, {}), n_boot=500)
    assert not g["go"] and not g["lenient_go"]


def test_gate_musiq_worse_is_no_go(base):
    r4 = table(base, {"lpips": -0.2, "deg": -0.2, "lmd": -0.2, "musiq": -0.3})
    g = ev.gate_decision(r4, table(base, {}), table(base, {}), n_boot=500)
    assert g["musiq_clearly_worse"]["row2"] and not g["go"]


def test_gate_strict_vs_lenient(base):
    # row4 beats row2 on lpips+deg, row3 on deg+lmd: >=2 against each, but only deg is joint
    r4 = table(base, {"lpips": -0.2, "deg": -0.2, "lmd": -0.2})
    r2 = table(base, {"lmd": -0.2})   # row4 ties row2 on lmd
    r3 = table(base, {"lpips": -0.2})  # row4 ties row3 on lpips
    g = ev.gate_decision(r4, r2, r3, n_boot=500)
    assert g["n_joint_wins"] == 1 and not g["go"] and g["lenient_go"]


def test_gate_nan_lmd_excluded_consistently(base):
    r4 = table(base, {"lpips": -0.2, "deg": -0.2, "lmd": -0.2}, nan_lmd=(0, 1))
    r2 = table(base, {}, nan_lmd=(1, 2))
    r3 = table(base, {})
    g = ev.gate_decision(r4, r2, r3, n_boot=200)
    assert g["lmd_common_n"] == 60 - 3
    assert g["ci"]["row2"]["lmd"]["n"] == g["ci"]["row3"]["lmd"]["n"] == 57
    assert g["ci"]["row2"]["lpips"]["n"] == 60
    assert g["lmd_nan_per_run"] == {"row4": 2, "row2": 2, "row3": 0}


def test_gate_requires_same_names(base):
    r4 = table(base, {})
    r2 = table(base, {})
    del r2["00003"]
    with pytest.raises(ValueError):
        ev.gate_decision(r4, r2, table(base, {}), n_boot=50)


def test_bootstrap_deterministic_and_collapses():
    a = np.random.default_rng(1).normal(size=40)
    b = np.random.default_rng(2).normal(size=40)
    assert ev.paired_bootstrap(a, b, 300, seed=3) == ev.paired_bootstrap(a, b, 300, seed=3)
    r = ev.paired_bootstrap(a, a, 300)
    assert r["mean_diff"] == r["lo"] == r["hi"] == 0.0 and r["n"] == 40


def write_run(d, tbl, fails=None):
    os.makedirs(d, exist_ok=True)
    cols = list(next(iter(tbl.values())))
    with open(os.path.join(d, "per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + cols + ["lmd_fail"])
        for n, r in tbl.items():
            w.writerow([n] + [repr(r[c]) for c in cols] + [(fails or {}).get(n, "")])


def test_cli_gate_compare_table(hqp_root, base):
    root = hqp_root.out_root
    dirs = {}
    for k, delta, nan in (("r4", {"lpips": -0.2, "deg": -0.2, "lmd": -0.2}, (0,)), ("r2", {}, (1,)), ("r3", {}, ())):
        dirs[k] = os.path.join(root, k)
        write_run(dirs[k], table(base, delta, nan_lmd=nan),
                  {f"{i:05d}": "restored" for i in nan})
    out = os.path.join(root, "gate.json")
    assert ev.main(["gate", "--row4", dirs["r4"], "--row2", dirs["r2"], "--row3", dirs["r3"],
                    "--n_boot", "200", "--out", out, "--md", os.path.join(root, "gate.md")]) == 0
    assert json.load(open(out))["go"] is True
    assert "GO" in open(os.path.join(root, "gate.md")).read()
    cmp_out = os.path.join(root, "cmp.json")
    ev.main(["compare", "--a", dirs["r4"], "--b", dirs["r2"], "--lmd_common", dirs["r3"],
             "--n_boot", "200", "--out", cmp_out])
    assert json.load(open(cmp_out))["metrics"]["lmd"]["n"] == 58
    ev.main(["table", "--run", f"a={dirs['r4']}", "--run", f"b={dirs['r2']}",
             "--out", os.path.join(root, "results.csv")])
    rows = list(csv.DictReader(open(os.path.join(root, "results.csv"))))
    assert [r["lmd_common_n"] for r in rows] == ["58", "58"]
    assert rows[0]["fail_restored"] == "1"


def test_outside_out_root_refused(hqp_root, base, tmp_path_factory):
    d = os.path.join(hqp_root.out_root, "r")
    write_run(d, table(base, {}))
    outside = str(tmp_path_factory.mktemp("elsewhere") / "x.json")
    with pytest.raises(PermissionError):
        ev.main(["compare", "--a", d, "--b", d, "--n_boot", "10", "--out", outside])


def test_score_pyiqa_two_images(hqp_root, tmp_path):
    pytest.importorskip("pyiqa")
    from PIL import Image
    rng = np.random.default_rng(0)
    for sub in ("rest", "hq"):
        os.makedirs(tmp_path / sub)
        for n in ("a", "b"):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / sub / f"{n}.png")
    out = os.path.join(hqp_root.out_root, "s")
    ev.main(["score", "--restored_dir", str(tmp_path / "rest"), "--hq_dir", str(tmp_path / "hq"),
             "--names_from_dir", "--metrics", "psnr", "ssim", "--out_dir", out, "--device", "cpu"])
    assert len(ev.read_per_image(out)) == 2


def test_fid_ref_stats_cache(hqp_root, tmp_path, monkeypatch):
    from hqprompt.metrics import fid
    ref = tmp_path / "ref"
    ref.mkdir()
    for i in range(3):
        (ref / f"{i}.png").write_bytes(b"x")
    calls = []

    def feats(d):
        calls.append(d)
        return np.random.default_rng(0).normal(size=(20, 4))

    mu, sig, hit = fid.load_or_compute_ref_stats(str(ref), hqp_root, feats)
    assert not hit and len(calls) == 1 and sig.shape == (4, 4)
    path = fid.ref_stats_path(hqp_root.eval_dir, str(ref))
    assert os.path.exists(path) and path.startswith(os.path.realpath(hqp_root.eval_dir))
    mu2, sig2, hit2 = fid.load_or_compute_ref_stats(str(ref), hqp_root, feats)
    assert hit2 and len(calls) == 1 and np.allclose(mu, mu2) and np.allclose(sig, sig2)
    (ref / "3.png").write_bytes(b"x")  # file count changes the key
    assert fid.ref_stats_path(hqp_root.eval_dir, str(ref)) != path
    _, _, hit3 = fid.load_or_compute_ref_stats(str(ref), hqp_root, feats)
    assert not hit3 and len(calls) == 2
    with pytest.raises(FileNotFoundError):
        fid.load_or_compute_ref_stats(str(tmp_path / "nope"), hqp_root, feats)
