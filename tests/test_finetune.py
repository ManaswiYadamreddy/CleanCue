import json
import os
import shutil

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from hqprompt import data, finetune, infer, osdface  # noqa: E402

FAST = ["--dry_run", "--max_steps", "20", "--save_every", "10", "--warmup", "2", "--lr", "1e-3"]


def _ft(hqp_root, prompt, extra=(), name=None):
    argv = ["--prompt", prompt, *FAST, "--out_root", hqp_root.out_root, *extra]
    if name:
        argv += ["--run_name", name]
    return finetune.main(argv)


def _infer(hqp_root, run, codes, *extra):
    return infer.main(["--run_dir", run, "--codes", codes, "--dry_run", "--out_root", hqp_root.out_root, *extra])


def _fake_pred_run(hqp_root, test_names=None, test_name="toy"):
    """Stand-in for a predictor run (codes are random; only the file contract matters)."""
    paths = finetune.prepare_dry_run(hqp_root)
    splits = data.load_splits(paths.splits_json)
    run = f"{hqp_root.out_root}/pred_run"
    os.makedirs(f"{run}/pred", exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("ft_train", "val"):
        ids = [int(i) for i in splits["splits"][split]]
        np.save(f"{run}/pred/{split}.npy", rng.integers(0, 1024, (len(ids), 256)).astype(np.int16))
        json.dump(ids, open(f"{run}/pred/{split}.ids.json", "w"))
    if test_names:
        np.save(f"{run}/pred/test_{test_name}.npy", rng.integers(0, 1024, (len(test_names), 256)).astype(np.int16))
        json.dump(test_names, open(f"{run}/pred/test_{test_name}.names.json", "w"))
    json.dump({"train_split": "pred_train", "codebook": "hq"}, open(f"{run}/meta.json", "w"))
    return run


def _prompt_args(hqp_root, prompt):
    return ["--hq_pred_run", _fake_pred_run(hqp_root)] if prompt == "hq_pred" else []


@pytest.mark.parametrize("prompt", ["lq", "null", "hq_true", "hq_pred"])
def test_dry_run_smoke_passes(hqp_root, prompt):
    assert _ft(hqp_root, prompt, ["--smoke", *_prompt_args(hqp_root, prompt)]) == 0
    run = f"{hqp_root.runs_dir}/smoke_{prompt}_s0"
    rep = json.load(open(f"{run}/smoke_report.json"))
    assert rep["passed"]
    assert rep["iteration0"]["max_abs_diff"] <= 1e-2 and rep["iteration0"]["mean_abs_diff"] <= 1e-3
    assert rep["gradients"]["n_step0_tensors"] > 0 and rep["gradients"]["n_step10_lora_A"] > 0
    assert rep["frozen_weights"]["n_tensors"] > 100 and rep["frozen_weights"]["changed"] == []
    assert rep["loss"]["probe_end"] < rep["loss"]["probe_start"]
    assert os.path.exists(f"{run}/DONE") and os.path.isdir(f"{run}/ckpt/step_000020")
    log = [json.loads(l) for l in open(f"{run}/train_log.jsonl")]
    assert [r["step"] for r in log] == [0, 10]
    gkey = "grad_norm_tokens" if prompt == "null" else "grad_norm_pos"
    assert {"loss", "mse", "dists", "id", "anchor", "grad_norm_total", "grad_norm_lora", "grad_norm_proj" if prompt != "null" else gkey,
            gkey, "it_s", "max_mem_gb", "lr"} <= set(log[0])
    if prompt == "null":  # no projection is trained for the null prompt
        assert log[0]["grad_norm_tokens"] > 0
    keys = load_file(f"{run}/ckpt/step_000020/trainable.safetensors")
    assert any(k.startswith("lora.") for k in keys) and any(k.startswith("prompt.") for k in keys)


def test_smoke_fails_when_teacher_does_not_match(hqp_root):
    """The iteration-0 check must be real: a cache with a random teacher exits 1."""
    paths = finetune.prepare_dry_run(hqp_root)  # correct cache first, then corrupt the teacher
    rd = data.CacheReader(paths, "val", ["lat_teacher"])
    w = data.CacheWriter(paths, "val", rd.ids, ["lat_teacher"], {"synthetic": True}, {}, data.load_splits(paths.splits_json)["sha256"])
    w.write("lat_teacher", np.arange(len(rd.ids)), np.random.default_rng(1).normal(size=(len(rd.ids), 4, 64, 64)).astype(np.float16))
    assert _ft(hqp_root, "lq", ["--smoke"]) == 1
    rep = json.load(open(f"{hqp_root.runs_dir}/smoke_lq_s0/smoke_report.json"))
    assert not rep["passed"] and not rep["iteration0"]["pass"]


def test_resume_matches_uninterrupted_and_infer(hqp_root):
    assert _ft(hqp_root, "lq", name="a") == 0
    run = f"{hqp_root.runs_dir}/a"
    final = load_file(f"{run}/ckpt/step_000020/trainable.safetensors")

    # simulate a crash after step 10: drop the last checkpoint, then resume
    shutil.rmtree(f"{run}/ckpt/step_000020")
    os.remove(f"{run}/DONE")
    with pytest.raises(FileExistsError):  # no silent auto-resume
        _ft(hqp_root, "lq", name="a")
    assert _ft(hqp_root, "lq", ["--resume", f"{run}/ckpt/step_000010"], name="a") == 0
    resumed = load_file(f"{run}/ckpt/step_000020/trainable.safetensors")
    assert set(resumed) == set(final)
    for k in final:
        torch.testing.assert_close(resumed[k], final[k], rtol=1e-4, atol=1e-6)
    steps = [json.loads(l)["step"] for l in open(f"{run}/train_log.jsonl")]
    assert steps == [0, 10]

    with pytest.raises(ValueError):  # resume with a different config
        _ft(hqp_root, "lq", ["--resume", f"{run}/ckpt/step_000010", "--w_id", "0.5"], name="a")
    with pytest.raises(ValueError):  # resume from outside this run
        _ft(hqp_root, "lq", ["--resume", f"{hqp_root.out_root}", ], name="a")

    # infer: val split from the cache, default output location
    assert _infer(hqp_root, run, "lq", "--split", "val") == 0
    out = f"{run}/outputs/step_20/lq/val"
    meta = json.load(open(f"{out}/meta.json"))
    ids = data.split_ids(data.load_splits(hqp_root.splits_json), "val")
    assert sorted(os.listdir(f"{out}/images")) == [f"{i:05d}.png" for i in ids] and meta["n_images"] == len(ids)
    # infer is deterministic and uses the trained weights (step 10 differs from step 20)
    assert _infer(hqp_root, run, "lq", "--split", "val", "--step", "10") == 0
    from PIL import Image
    a = np.asarray(Image.open(f"{out}/images/{ids[0]:05d}.png"))
    b = np.asarray(Image.open(f"{run}/outputs/step_10/lq/val/images/{ids[0]:05d}.png"))
    assert a.shape == (512, 512, 3) and not np.array_equal(a, b)
    # arbitrary directory of LQ images
    lq_dir = f"{hqp_root.out_root}/dry_run/imgs/lq"
    assert _infer(hqp_root, run, "lq", "--input_dir", lq_dir, "--dataset_name", "toy", "--batch_size", "3") == 0
    assert len(os.listdir(f"{run}/outputs/step_20/lq/toy/images")) == len(os.listdir(lq_dir))


def test_infer_refuses_wrong_codebook_and_supports_hq_modes(hqp_root):
    assert _ft(hqp_root, "hq_true", name="hq") == 0
    run = f"{hqp_root.runs_dir}/hq"
    for codes in ("lq", "null"):
        with pytest.raises(ValueError, match="refusing"):
            _infer(hqp_root, run, codes, "--split", "val")
    assert _infer(hqp_root, run, "hq_true", "--split", "val") == 0
    lq_dir = f"{hqp_root.out_root}/dry_run/imgs/lq"
    names = sorted(os.listdir(lq_dir))
    pred = _fake_pred_run(hqp_root, names)
    # row 4's model with predicted codes (same codebook)
    assert _infer(hqp_root, run, "hq_pred", "--hq_pred_run", pred, "--split", "val") == 0
    assert _infer(hqp_root, run, "hq_pred", "--hq_pred_run", pred, "--input_dir", lq_dir, "--dataset_name", "toy") == 0
    # hq_true on a directory needs the clean images; names are matched through norm_stem
    hq_dir = f"{hqp_root.out_root}/dry_run/imgs/hq"
    assert _infer(hqp_root, run, "hq_true", "--input_dir", lq_dir, "--dataset_name", "toy2", "--hq_dir", hq_dir) == 0
    with pytest.raises(SystemExit):  # --hq_dir is mandatory there
        _infer(hqp_root, run, "hq_true", "--input_dir", lq_dir, "--dataset_name", "toy3")

    assert _ft(hqp_root, "null", name="null") == 0
    with pytest.raises(ValueError, match="refusing"):
        _infer(hqp_root, f"{hqp_root.runs_dir}/null", "lq", "--split", "val")
    assert _infer(hqp_root, f"{hqp_root.runs_dir}/null", "null", "--split", "val") == 0


def test_lambda_3d_not_implemented(hqp_root):
    with pytest.raises(NotImplementedError):
        _ft(hqp_root, "lq", ["--lambda_3d", "0.1"])
    assert not os.path.exists(hqp_root.runs_dir)  # raised before anything was created


def test_nonempty_run_dir_refused_and_hq_pred_needs_run(hqp_root):
    run = f"{hqp_root.runs_dir}/x"
    os.makedirs(run)
    open(f"{run}/stale", "w").close()
    with pytest.raises(FileExistsError):
        _ft(hqp_root, "lq", name="x")
    with pytest.raises(SystemExit):
        _ft(hqp_root, "hq_pred")


def test_hq_pred_rejects_wrong_predictor_meta(hqp_root):
    pred = _fake_pred_run(hqp_root)
    json.dump({"train_split": "ft_train"}, open(f"{pred}/meta.json", "w"))
    with pytest.raises(ValueError, match="pred_train"):
        _ft(hqp_root, "hq_pred", ["--hq_pred_run", pred])


def test_grad_checkpointing_runs(hqp_root):
    assert _ft(hqp_root, "lq", ["--grad_ckpt_unet", "--grad_ckpt_vae", "--max_steps", "3", "--save_every", "3"]) == 0
    assert os.path.exists(f"{hqp_root.runs_dir}/lq_l3d0_s0/DONE")


def test_lora_module_count_and_state_roundtrip():
    m = osdface.tiny_models(seed=1)
    n = finetune.add_lora(m.unet, 4, 4, dry_run=True)
    assert n == finetune.count_attention(m.unet) * 4 > 0
    sd = finetune.lora_state_dict(m.unet)
    assert len(sd) == 2 * n and all(v.dtype == torch.float32 for v in sd.values())
    new = {k: torch.randn_like(v) for k, v in sd.items()}
    finetune.load_lora_state_dict(m.unet, new)
    back = finetune.lora_state_dict(m.unet)
    assert all(torch.equal(back[k], new[k]) for k in new)
    with pytest.raises(KeyError):
        finetune.load_lora_state_dict(m.unet, dict(list(new.items())[:-1]))
    # disable/enable adapters: with B != 0 the adapter changes the output, and disabling restores the base
    z = torch.randn(1, 4, 64, 64)
    p = torch.randn(1, 8, 1024)
    on = osdface.one_step(m.unet, z, p, m.alphas_cumprod)
    m.unet.disable_adapters()
    off = osdface.one_step(m.unet, z, p, m.alphas_cumprod)
    m.unet.enable_adapters()
    assert not torch.allclose(on, off) and torch.equal(on, osdface.one_step(m.unet, z, p, m.alphas_cumprod))


def test_compute_losses_hook_and_weights():
    from hqprompt import losses
    mods = {"ea_dists": losses.StubDists(), "id": losses.StubID()}
    out = torch.randn(1, 3, 64, 64, requires_grad=True)
    x0 = torch.randn(1, 4, 8, 8, requires_grad=True)
    batch = {"hq_pm1": torch.rand(1, 3, 64, 64) * 2 - 1, "lat_teacher": torch.randn(1, 4, 8, 8)}
    w = {"w_mse": 1.0, "w_dists": 1.0, "w_id": 0.1, "w_anchor": 1.0, "lambda_3d": 0.5}
    base = finetune.compute_losses(out, x0, batch, mods, w)
    assert "structure" not in base
    ref = base["mse"] + base["dists"] + 0.1 * base["id"] + base["anchor"]
    torch.testing.assert_close(base["total"], ref)
    hooked = finetune.compute_losses(out, x0, batch, mods, w, structure_fn=lambda o, h, b: o.mean() * 0 + 2.0)
    torch.testing.assert_close(hooked["total"], base["total"] + 1.0)
    base["total"].backward()
    assert out.grad is not None and x0.grad is not None
