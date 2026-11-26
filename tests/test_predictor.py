import json

import numpy as np
import pytest
import torch

from hqprompt import data, train_predictor as tp


def test_models_mask_and_shapes():
    seen = torch.zeros(1024, dtype=torch.bool)
    seen[:10] = True
    z = torch.randn(2, 512, 16, 16)
    for model in (tp.LinearPredictor(seen), tp.TransformerPredictor(seen, layers=1, d_model=64, nhead=4)):
        out = model(z)
        assert out.shape == (2, 256, 1024) and out.dtype == torch.float32
        assert torch.isinf(out[..., 10:]).all() and torch.isfinite(out[..., :10]).all()
        assert model(z).argmax(-1).max() < 10
        assert "seen" in model.state_dict()


def test_linear_param_count():
    m = tp.LinearPredictor(torch.ones(1024, dtype=torch.bool))
    assert sum(p.numel() for p in m.parameters()) == 512 * 1024 + 1024 + 256 * 1024


def test_check_split_ids_rejects_overlap():
    s = data.make_splits(16, sizes={"pred_train": 8, "ft_train": 4, "val": 2, "spare": 2})
    tp.check_split_ids(s["splits"]["pred_train"], s)
    with pytest.raises(ValueError):
        tp.check_split_ids(s["splits"]["pred_train"] + s["splits"]["ft_train"][:1], s)
    with pytest.raises(ValueError):  # right size but not pred_train (overlaps ft_train)
        tp.check_split_ids(s["splits"]["pred_train"][:-1] + s["splits"]["ft_train"][:1], s)
    with pytest.raises(ValueError):
        tp.check_split_ids(s["splits"]["pred_train"], dict(s, source="celeba"))


def _run(hqp_root, name, extra=()):
    return tp.main(["--run_name", name, "--dry_run", "--epochs", "2", "--batch_size", "16",
                    "--out_root", hqp_root.out_root, *extra])


def test_dry_run_linear(hqp_root):
    pytest.importorskip("hqprompt.vre")  # baseline (b) uses vre.nearest_codes
    assert _run(hqp_root, "lin", ["--model", "linear"]) == 0
    _check_outputs(hqp_root, "lin")


def test_dry_run_transformer_and_predict_dir(hqp_root, tmp_path):
    pytest.importorskip("hqprompt.vre")
    hq, lq = data.write_synthetic_ffhq(str(tmp_path / "imgs"), n=3)
    assert _run(hqp_root, "tf", ["--model", "transformer", "--layers", "1", "--d_model", "64", "--nhead", "4",
                                 "--predict_dir", f"toy={lq}"]) == 0
    run = f"{hqp_root.predictor_dir}/tf"
    _check_outputs(hqp_root, "tf")
    pred = np.load(f"{run}/pred/test_toy.npy")
    names = json.load(open(f"{run}/pred/test_toy.names.json"))
    assert pred.shape == (3, 256) and pred.dtype == np.int16 and len(names) == 3
    with pytest.raises(FileExistsError):  # no silent overwrite of a trained run
        _run(hqp_root, "tf", ["--layers", "1", "--d_model", "64", "--nhead", "4"])
    assert tp.main(["--run_name", "tf", "--dry_run", "--predict_only", "--layers", "1", "--d_model", "64",
                    "--nhead", "4", "--out_root", hqp_root.out_root]) == 0


def _check_outputs(hqp_root, name):
    run = f"{hqp_root.predictor_dir}/{name}"
    for f in ("config.json", "best.safetensors", "last.safetensors", "train_log.jsonl", "metrics.json", "meta.json"):
        assert json.load(open(f"{run}/{f}")) if f.endswith(".json") else True
    log = [json.loads(l) for l in open(f"{run}/train_log.jsonl")]
    assert [r["epoch"] for r in log] == [1, 2]
    m = json.load(open(f"{run}/metrics.json"))
    for split in ("val", "pred_train_subset"):
        assert set(m["model"][split]["per_bucket_top1"]) == {"top10", "11-50", "51-200", "rest"}
        assert 0 <= m["model"][split]["top1"] <= m["model"][split]["top5"] <= 1
        assert set(m["baselines"][split]) == {"most_frequent_global", "most_frequent_per_position",
                                              "nearest_hq_code_to_lq_feature"}
    meta = json.load(open(f"{run}/meta.json"))
    assert meta["train_split"] == "pred_train" and meta["codebook"] == "hq" and meta["mask_n_seen"] > 0
    seen_n = meta["mask_n_seen"]
    for split, n in (("ft_train", 16), ("val", 16)):
        p = np.load(f"{run}/pred/{split}.npy")
        assert p.shape == (n, 256) and p.dtype == np.int16 and p.min() >= 0 and p.max() < 1024
        assert len(np.unique(p)) <= seen_n  # only seen codes
        assert len(json.load(open(f"{run}/pred/{split}.ids.json"))) == n
