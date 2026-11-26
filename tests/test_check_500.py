import json
import os

import numpy as np
import pytest

from hqprompt import check_500


def _run(hqp_root, *extra):
    return check_500.main(["--dry_run", "--n", "8", "--batch_size", "4", "--out_root", hqp_root.out_root, *extra])


def test_c1_decision():
    rng = np.random.default_rng(0)
    base = rng.normal(20, 2, 100)
    assert check_500.c1_decision(base + 3, base)["pass"]
    assert not check_500.c1_decision(base, base + 3)["pass"]
    assert not check_500.c1_decision(base, base)["pass"]  # CI collapses to 0: not above 0


def test_dry_run_writes_everything(hqp_root):
    assert _run(hqp_root, "--continue_on_c1_fail") == 0
    c = hqp_root.checks_dir
    for f in ("c1.json", "c2.json", "c3.json", "c1_grid_clean.png", "c1_grid_lq.png",
              "c2_hist_hq.npy", "c2_hist_lq.npy", "c2_hist_hq.png", "c2_hist_lq.png"):
        assert os.path.exists(f"{c}/{f}"), f
    c1 = json.load(open(f"{c}/c1.json"))
    assert c1["recommended_hq_input_mode"] in ("single", "double")
    assert len(c1["table"]) == 2 * 2 * 2 * 2 and len(c1["ids"]) == 8 and c1["lpips_available"] is False
    assert json.load(open(f"{c}/c2.json"))["valid"] == c1["c1_pass"]
    c3 = json.load(open(f"{c}/c3.json"))
    assert c3["n_fit_faces"] == 6 and c3["n_test_faces"] == 2 and "top5" in c3["linear"]
    assert np.load(f"{c}/c2_hist_hq.npy").sum() == 8 * 256


def test_c1_failure_exits_2_and_stops(hqp_root):
    rc = check_500.main(["--dry_run", "--n", "4", "--batch_size", "4", "--out_root", hqp_root.out_root])
    c1 = json.load(open(f"{hqp_root.checks_dir}/c1.json"))
    assert rc == (0 if c1["c1_pass"] else 2)
    assert os.path.exists(f"{hqp_root.checks_dir}/c2.json") == c1["c1_pass"]


def test_dry_run_needs_explicit_out_root():
    with pytest.raises(SystemExit):
        check_500.main(["--dry_run"])
