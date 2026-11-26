import json

import numpy as np
import pytest
import torch

from hqprompt import data
from hqprompt.data import CacheMismatchError, CacheReader, CacheWriter

SIZES = {"pred_train": 8, "ft_train": 4, "val": 2, "spare": 2}


def test_make_splits_deterministic_disjoint():
    a = data.make_splits()
    b = data.make_splits()
    assert a == b
    sp = a["splits"]
    assert {k: len(v) for k, v in sp.items()} == data.SPLIT_SIZES
    allids = [i for v in sp.values() for i in v]
    assert sorted(allids) == list(range(data.N_FFHQ))
    assert all(v == sorted(v) for v in sp.values())
    assert data.make_splits(seed=1)["splits"] != sp


def test_load_splits_rejects_tampering(hqp_root):
    s = data.make_splits(16, sizes=SIZES)
    data.write_splits(hqp_root, s, "h", "l", check_files=False)
    assert data.load_splits(hqp_root.splits_json)["sha256"] == s["sha256"]
    d = json.load(open(hqp_root.splits_json))
    d["splits"]["val"][0], d["splits"]["spare"][0] = d["splits"]["spare"][0], d["splits"]["val"][0]
    d["splits"]["val"].sort(); d["splits"]["spare"].sort()
    json.dump(d, open(hqp_root.splits_json, "w"))
    with pytest.raises(AssertionError):
        data.load_splits(hqp_root.splits_json)


def test_write_splits_overwrite_and_missing(hqp_root, tmp_path):
    s = data.make_splits(16, sizes=SIZES)
    data.write_splits(hqp_root, s, "h", "l", check_files=False)
    data.write_splits(hqp_root, s, "h", "l", check_files=False)  # identical: ok
    with pytest.raises(FileExistsError):
        data.write_splits(hqp_root, data.make_splits(16, seed=3, sizes=SIZES), "h", "l", check_files=False)
    with pytest.raises(FileNotFoundError):
        data.write_splits(hqp_root, s, str(tmp_path / "nohq"), str(tmp_path / "nolq"), check_files=True)


def test_norm_stem_and_list_images(tmp_path):
    assert data.norm_stem("00001_LQ.png") == "00001"
    assert data.norm_stem("a/b/x_bicubic.jpg") == "x"
    assert data.norm_stem("x_deg.png") == "x"
    assert data.norm_stem("plain.png") == "plain"
    (tmp_path / "b.png").write_bytes(b""); (tmp_path / "a.jpg").write_bytes(b""); (tmp_path / "c.txt").write_bytes(b"")
    assert data.list_images(str(tmp_path)) == ["a.jpg", "b.png"]


def test_load_image_01_resize(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((64, 64, 3), 255, np.uint8)).save(tmp_path / "s.png")
    Image.fromarray(np.zeros((64, 32, 3), np.uint8)).save(tmp_path / "r.png")
    with pytest.raises(ValueError):
        data.load_image_01(str(tmp_path / "s.png"))
    x = data.load_image_01(str(tmp_path / "s.png"), resize="official")
    assert x.shape == (3, 512, 512) and x.dtype == torch.float32 and torch.allclose(x, torch.ones_like(x))
    with pytest.raises(ValueError):
        data.load_image_01(str(tmp_path / "r.png"), resize="official")


def test_step_order():
    a = data.step_order(10, 7, 4, seed=3)
    assert a.shape == (7, 4) and np.array_equal(a, data.step_order(10, 7, 4, seed=3))
    assert not np.array_equal(a, data.step_order(10, 7, 4, seed=4))
    flat = data.step_order(10, 5, 4, seed=0).ravel()  # 20 = 2 epochs
    assert sorted(flat[:10]) == list(range(10)) and sorted(flat[10:]) == list(range(10))


def _writer(paths, ids=None, settings=None, sha="s"):
    ids = np.arange(6, dtype=np.int32) if ids is None else ids
    return CacheWriter(paths, "val", ids, ["hq_idx", "lq_prompt77", "lat_lq"],
                       settings or {"a": 1}, {}, sha)


def test_cache_writer_resume_and_reader(hqp_root):
    w = _writer(hqp_root)
    rows = np.arange(3)
    w.write("hq_idx", rows, np.arange(3 * 256).reshape(3, 256).astype(np.int16) % 1024)
    assert list(w.pending("hq_idx")) == [3, 4, 5] and len(w.pending("lat_lq")) == 6
    w.finalize()
    w2 = _writer(hqp_root)  # reopen
    assert list(w2.pending("hq_idx")) == [3, 4, 5]
    w2.write("hq_idx", np.arange(3, 6), np.zeros((3, 256), np.int16))
    w2.write("lq_prompt77", np.arange(6), np.full((6, 77), -1, np.int16))
    w2.write("lat_lq", np.arange(6), np.zeros((6, 4, 64, 64), np.float16))
    w2.finalize()
    r = CacheReader(hqp_root, "val")
    assert r.array("hq_idx").shape == (6, 256) and r.row_of(4) == 4 and r.settings == {"a": 1}
    assert r.array("hq_idx")[1, 5] == 261
    assert not r.array("hq_idx").flags.writeable
    m = json.load(open(f"{hqp_root.cache_dir}/manifest.json"))
    assert m["items"]["val"]["arrays"]["hq_idx"]["n_done"] == 6


def test_cache_reader_partial_raises(hqp_root):
    w = _writer(hqp_root)
    w.write("hq_idx", np.arange(2), np.zeros((2, 256), np.int16))
    w.finalize()
    with pytest.raises(CacheMismatchError):
        CacheReader(hqp_root, "val", ["hq_idx"])
    CacheReader(hqp_root, "val", ["hq_idx"], require_complete=False)


def test_cache_mismatch_and_bad_values(hqp_root):
    w = _writer(hqp_root)
    w.finalize()
    with pytest.raises(CacheMismatchError):
        _writer(hqp_root, settings={"a": 2})
    with pytest.raises(CacheMismatchError):
        _writer(hqp_root, sha="other")
    with pytest.raises(CacheMismatchError):
        _writer(hqp_root, ids=np.arange(6, dtype=np.int32)[::-1].copy())
    ok = np.zeros((2, 256), np.int16)
    with pytest.raises(TypeError):
        w.write("hq_idx", np.arange(2), ok.astype(np.int32))
    with pytest.raises(ValueError):
        w.write("hq_idx", np.arange(3), ok)
    with pytest.raises(ValueError):
        w.write("hq_idx", np.arange(2), ok - 1)
    bad = ok.copy(); bad[0, 0] = 1024
    with pytest.raises(ValueError):
        w.write("hq_idx", np.arange(2), bad)
    w.write("lq_prompt77", np.arange(2), np.full((2, 77), -1, np.int16))  # -1 allowed here
    assert len(w.pending("hq_idx")) == 6  # failed writes marked nothing


@pytest.fixture
def synth(hqp_root, tmp_path):
    hq, lq = data.write_synthetic_ffhq(str(tmp_path / "ffhq"), n=16)
    import dataclasses
    paths = dataclasses.replace(hqp_root, ffhq_hq=hq, ffhq_lq=lq)
    splits = data.make_splits(16, sizes=SIZES)
    data.write_splits(paths, splits, hq, lq)
    data.write_synthetic_cache(paths, splits)
    return paths, splits


def test_synthetic_ffhq_pairs(synth):
    paths, _ = synth
    ds = data.ImageDataset(paths, [0, 1])
    it = ds[0]
    assert it["hq01"].shape == it["lq01"].shape == (3, 512, 512)
    assert not torch.equal(it["hq01"], it["lq01"])


@pytest.mark.parametrize("prompt,shape", [("lq", 256), ("hq_true", 256), ("official", 77), ("null", None),
                                           ("hq_pred", 256)])
def test_finetune_dataset_modes(synth, tmp_path, prompt, shape):
    paths, splits = synth
    pred_file = None
    if prompt == "hq_pred":
        ids = data.split_ids(splits, "val")
        pred_file = str(tmp_path / "val.npy")
        np.save(pred_file, np.zeros((len(ids), 256), np.int16))
        json.dump([int(i) for i in ids], open(str(tmp_path / "val.ids.json"), "w"))
    ds = data.FinetuneDataset(paths, "val", prompt, pred_file)
    it = ds[1]
    assert len(ds) == 2
    assert it["hq_pm1"].shape == (3, 512, 512) and it["hq_pm1"].min() >= -1 and it["hq_pm1"].max() <= 1
    assert it["lat_lq"].dtype == it["lat_teacher"].dtype == torch.float32
    assert it["lat_lq"].shape == (4, 64, 64) and it["id"].dtype == torch.int64
    if shape is None:
        assert "idx" not in it
    else:
        assert it["idx"].shape == (shape,) and it["idx"].dtype == torch.int64
    if prompt == "official":
        assert (it["idx"] == -1).any()


def test_hq_pred_ids_mismatch_raises(synth, tmp_path):
    paths, _ = synth
    np.save(tmp_path / "p.npy", np.zeros((2, 256), np.int16))
    json.dump([98, 99], open(tmp_path / "p.ids.json", "w"))
    with pytest.raises(ValueError):
        data.FinetuneDataset(paths, "val", "hq_pred", str(tmp_path / "p.npy"))


def test_predictor_dataset(synth):
    paths, _ = synth
    it = data.PredictorDataset(paths, "pred_train")[0]
    assert it["z"].shape == (512, 16, 16) and it["z"].dtype == torch.float32
    assert it["y"].shape == (256,) and it["y"].dtype == torch.int64
