import importlib.util
import os
import sys

import pytest
import torch

from hqprompt import prompt as P

CRAFT = os.environ.get("HQP_CRAFT_REPO", "/Users/manaswiyadamreddy/Documents/CRAFT")


def make(mode, null_init="pad", trainable=True, seed=0):
    cb = None if mode == "null" else torch.randn(1024, 512, generator=torch.Generator().manual_seed(7))
    return P.PromptEncoder(mode, cb, P.random_embedding_change_sd(seed), trainable=trainable, null_init=null_init)


def idx_for(mode, B=2):
    n = P.N_TOKENS[mode]
    return torch.randint(0, 1024, (B, n))


@pytest.mark.parametrize("mode", P.MODES)
def test_shapes_dtypes(mode):
    enc = make(mode)
    out = enc(batch_size=3) if mode == "null" else enc(idx_for(mode, 3))
    assert out.shape == (3, P.N_TOKENS[mode], 1024) and out.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in enc.parameters())
    assert enc.codebook_kind == P.CODEBOOK_FOR_MODE[mode]


def test_projection_parity_with_craft():
    path = os.path.join(CRAFT, "lq_embed.py")
    if not os.path.exists(path):
        pytest.skip(f"CRAFT repo not found at {CRAFT}")
    vq = importlib.util.spec_from_file_location("craft_vqvae", os.path.join(CRAFT, "vqvae.py"))
    vqm = importlib.util.module_from_spec(vq)
    vq.loader.exec_module(vqm)
    saved = sys.modules.get("vqvae")
    sys.modules["vqvae"] = vqm  # lq_embed does `from vqvae import ...`
    try:
        spec = importlib.util.spec_from_file_location("craft_lq_embed", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        if saved is None:
            sys.modules.pop("vqvae", None)
        else:
            sys.modules["vqvae"] = saved
    ref = m.TwoLayerConv1x1(512, 1024)
    ours = P.TokenProjection()
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 77, 512)
    assert torch.equal(ours(x), ref(x))
    assert ours(torch.randn(2, 256, 512)).shape == (2, 256, 1024)  # no 77 assert


def test_official_all_pad_is_proj_of_zeros():
    enc = make("official")
    out = enc(torch.full((2, 77), -1))
    assert torch.equal(out, enc.proj(torch.zeros(2, 77, 512)))  # same shape: bit-exact
    ref = enc.proj(torch.zeros(1, 1, 512))[0, 0]  # different shape: kernels differ at ~1e-8
    assert (out - ref).abs().max() < 1e-6 and (out[0, 0] == out[1, 76]).all()
    assert not any(p.requires_grad for p in enc.parameters())
    assert make("official", trainable=True).trainable_state_dict() == {}


def test_lq_zero_pos_at_init():
    enc = make("lq")
    assert (enc.pos == 0).all()
    idx = idx_for("lq")
    assert torch.equal(enc(idx), enc.proj(enc.codebook[idx]))


@pytest.mark.parametrize("mode", ["lq", "hq_true", "hq_pred"])
def test_gradients_code_modes(mode):
    enc = make(mode)
    enc(idx_for(mode)).pow(2).sum().backward()
    for name in ("proj.conv1.weight", "proj.conv2.bias", "pos"):
        g = dict(enc.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
    assert enc.codebook.grad is None and not enc.codebook.requires_grad
    assert "codebook" not in enc.state_dict()  # non-persistent buffer


def test_gradients_null_and_init():
    enc = make("null")
    enc(batch_size=2).pow(2).sum().backward()
    assert enc.tokens.grad.abs().sum() > 0
    assert set(enc.trainable_state_dict()) == {"tokens"}
    # pad init == the official padding token, in every row
    ec = P.random_embedding_change_sd(0)
    proj = P.TokenProjection()
    proj.load_state_dict(ec)
    pad = proj(torch.zeros(1, 1, 512))[0, 0]
    assert torch.allclose(enc.tokens, pad.expand(256, 1024), atol=1e-6)
    assert (make("null", null_init="zeros").tokens == 0).all()


def test_bad_idx_raises():
    enc = make("lq")
    with pytest.raises(ValueError):
        enc(torch.full((1, 256), 1024))
    with pytest.raises(ValueError):
        enc(torch.full((1, 256), -1))  # -1 pad only allowed in official mode
    with pytest.raises(TypeError):
        enc(torch.zeros(1, 256, dtype=torch.int32))
    with pytest.raises(ValueError):
        enc(torch.zeros(1, 77, dtype=torch.int64))
    off = make("official")
    with pytest.raises(ValueError):
        off(torch.full((1, 77), -2))
    with pytest.raises(ValueError):
        make("null")(torch.zeros(1, 256, dtype=torch.int64), batch_size=1)
    with pytest.raises(ValueError):
        P.PromptEncoder("bogus", None, None)


def test_bad_embedding_change_raises():
    sd = P.random_embedding_change_sd()
    del sd["conv1.bias"]
    with pytest.raises(KeyError):
        P.PromptEncoder("lq", torch.zeros(1024, 512), sd)


def test_trainable_state_dict_roundtrip():
    a = make("hq_true", seed=1)
    with torch.no_grad():
        a.pos.normal_()
        a.proj.conv1.weight.add_(1.0)
    sd = a.trainable_state_dict()
    assert set(sd) == {"proj.conv1.weight", "proj.conv1.bias", "proj.conv2.weight", "proj.conv2.bias", "pos"}
    b = make("hq_true", seed=2)
    b.load_trainable_state_dict(sd)
    idx = idx_for("hq_true")
    assert torch.equal(a(idx), b(idx))
    with pytest.raises(KeyError):
        b.load_trainable_state_dict({"pos": sd["pos"]})
    bad = dict(sd)
    bad["pos"] = torch.zeros(10, 1024)
    with pytest.raises(ValueError):
        b.load_trainable_state_dict(bad)
