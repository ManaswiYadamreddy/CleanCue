import json
import os

import numpy as np
import pytest

from hqprompt import build_cache as bc, data


def _args(root, *extra, hq="double"):
    return ["--dry_run", "--split", "all", "--stage", "all", "--hq_input_mode", hq, "--batch_size", "2",
            "--teacher_batch_size", "2", "--out_root", str(root), *extra]


def _arrays(root):
    out = {}
    for split in ("pred_train", "ft_train", "val"):
        d = f"{root}/cache/{split}"
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy") and ".done" not in f:
                out[f"{split}/{f}"] = np.load(f"{d}/{f}")
    return out


def test_full_build_and_verify(hqp_root):
    assert bc.main(_args(hqp_root.out_root)) == 0
    arrs = _arrays(hqp_root.out_root)
    assert set(arrs) == {f"{s}/{i}.npy" for s, items in data.SPLIT_ITEMS.items() for i in items} | \
        {f"{s}/ids.npy" for s in data.SPLIT_ITEMS}
    m = json.load(open(f"{hqp_root.cache_dir}/manifest.json"))
    assert m["settings"]["hq_input_mode"] == "double" and m["verification"] is None
    assert m["items"]["ft_train"]["arrays"]["lat_teacher"]["n_done"] == 4
    assert m["build"] == {"batch_size": 2, "teacher_batch_size": 2} and "build" not in m["settings"]
    for split in data.SPLIT_ITEMS:  # readers require every row done
        data.CacheReader(hqp_root, split)
    assert arrs["val/lat_teacher.npy"].dtype == np.float16 and np.abs(arrs["val/lat_teacher.npy"]).max() > 0
    p77 = arrs["ft_train/lq_prompt77.npy"]
    assert p77.shape == (4, 77) and p77.min() >= -1 and (p77[:, 0] >= 0).all()
    # each valid prompt entry is a distinct code
    assert all(len(set(r[r >= 0])) == (r >= 0).sum() for r in p77)

    assert bc.main(["--verify", "--dry_run", "--n", "100", "--seed", "1", "--out_root", hqp_root.out_root]) == 0
    rep = json.load(open(f"{hqp_root.cache_dir}/verify_report.json"))
    assert rep["passed"] and rep["n_checked"] == 14 and rep["n_mismatches"] == 0
    assert json.load(open(f"{hqp_root.cache_dir}/manifest.json"))["verification"]["passed"] is True


def test_verify_detects_tampering(hqp_root):
    assert bc.main(_args(hqp_root.out_root)) == 0
    f = f"{hqp_root.cache_dir}/ft_train/hq_idx.npy"
    a = np.load(f, mmap_mode="r+")
    a[:, 0] = (a[:, 0] + 1) % 1024
    a.flush()
    z = np.load(f"{hqp_root.cache_dir}/val/z_lq.npy", mmap_mode="r+")
    z[:] += np.float16(0.5)
    z.flush()
    rc = bc.main(["--verify", "--dry_run", "--n", "100", "--out_root", hqp_root.out_root])
    assert rc == 1
    rep = json.load(open(f"{hqp_root.cache_dir}/verify_report.json"))
    assert not rep["passed"]
    kinds = {(x["split"], x["item"]) for x in rep["mismatches"]}
    assert ("ft_train", "hq_idx") in kinds and ("val", "z_lq") in kinds
    assert all("distance_gaps" in x for x in rep["mismatches"] if x["item"] == "hq_idx")
    assert json.load(open(f"{hqp_root.cache_dir}/manifest.json"))["verification"]["passed"] is False


def test_kill_and_resume_equals_fresh(hqp_root, tmp_path):
    root = hqp_root.out_root
    part = ["--dry_run", "--split", "all", "--stage", "codes", "--hq_input_mode", "double", "--batch_size", "2",
            "--limit_batches", "1", "--out_root", root]
    assert bc.main(part) == 0
    m = json.load(open(f"{root}/cache/manifest.json"))
    assert m["items"]["pred_train"]["arrays"]["hq_idx"]["n_done"] == 2  # killed early
    with pytest.raises(data.CacheMismatchError):  # incomplete cache is unreadable
        data.CacheReader(hqp_root, "pred_train")
    assert bc.main(_args(root)) == 0

    fresh = str(tmp_path / "fresh")
    assert bc.main(_args(fresh)) == 0
    a, b = _arrays(root), _arrays(fresh)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_changed_settings_on_resume_raise(hqp_root):
    assert bc.main(_args(hqp_root.out_root, "--stage", "codes")) == 0
    with pytest.raises(data.CacheMismatchError):
        bc.main(_args(hqp_root.out_root, "--stage", "codes", hq="single"))
    with pytest.raises(data.CacheMismatchError):
        bc.main(_args(hqp_root.out_root, "--stage", "latents", "--teacher_dtype", "bf16"))


@pytest.mark.parametrize("c1", [None, {"c1_pass": False, "recommended_hq_input_mode": "double"},
                                {"c1_pass": True, "recommended_hq_input_mode": "single"}])
def test_real_run_refuses_without_valid_c1(hqp_root, c1):
    if c1 is not None:
        os.makedirs(hqp_root.checks_dir)
        json.dump(c1, open(f"{hqp_root.checks_dir}/c1.json", "w"))
    with pytest.raises(RuntimeError, match="c1|C1"):
        bc.main(["--split", "all", "--stage", "codes", "--hq_input_mode", "double", "--out_root", hqp_root.out_root])
    assert not os.path.exists(hqp_root.cache_dir)


def test_requires_flags(hqp_root):
    with pytest.raises(SystemExit):
        bc.main(["--dry_run", "--split", "all", "--out_root", hqp_root.out_root])
