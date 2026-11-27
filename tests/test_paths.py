import os

import pytest
import torch

from hqprompt.paths import assert_writable, get_paths, safe_torch_load


def test_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("HQP_ROOT", str(tmp_path / "env"))
    assert get_paths().out_root == str(tmp_path / "env")
    assert get_paths(out_root=str(tmp_path / "ovr")).out_root == str(tmp_path / "ovr")
    assert get_paths().model_cache == str(tmp_path / "env" / "model_cache")


def test_assert_writable(hqp_root, tmp_path):
    assert assert_writable(os.path.join(hqp_root.out_root, "a/b.json"), hqp_root)
    for bad in ["/projectnb/cs599bg/manaswi/craft/data/x.png",
                os.path.join(hqp_root.out_root, "../escape.json"),
                str(tmp_path) + "_sibling/x"]:
        with pytest.raises(PermissionError):
            assert_writable(bad, hqp_root)


def test_safe_torch_load(tmp_path):
    p = get_paths(out_root=str(tmp_path / "out"), pretrained=str(tmp_path))
    torch.save({"state_dict": {"w": torch.ones(2)}}, p.associate_ckpt)
    torch.save({"w": torch.ones(1)}, tmp_path / "other.pth")
    assert safe_torch_load(p.associate_ckpt, p)["state_dict"]["w"].sum() == 2
    with pytest.raises(PermissionError):
        safe_torch_load(str(tmp_path / "other.pth"), p)


def test_safe_torch_load_stubs_lightning_objects(tmp_path):
    """associate_2.ckpt pickles a pytorch_lightning ModelCheckpoint; load must not need lightning."""
    import sys
    import types

    import torch

    mod = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        def __init__(self):
            self.best = 1.0

    ModelCheckpoint.__module__ = mod.__name__
    ModelCheckpoint.__qualname__ = "ModelCheckpoint"
    mod.ModelCheckpoint = ModelCheckpoint
    names = ["pytorch_lightning", "pytorch_lightning.callbacks", mod.__name__]
    for n in names[:2]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules[mod.__name__] = mod
    p = get_paths(out_root=str(tmp_path / "out"), pretrained=str(tmp_path))
    try:
        torch.save({"state_dict": {"w": torch.ones(2)}, "callbacks": {"cb": ModelCheckpoint()}},
                   p.associate_ckpt)
    finally:
        for n in names:
            del sys.modules[n]
    out = safe_torch_load(p.associate_ckpt, p)
    assert out["state_dict"]["w"].sum() == 2
    assert out["callbacks"]["cb"].best == 1.0
