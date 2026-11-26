import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from hqprompt import vre
from hqprompt.paths import get_paths

CRAFT = os.environ.get("HQP_CRAFT_REPO", "/Users/manaswiyadamreddy/Documents/CRAFT")


def load_craft_module(name):
    """Import CRAFT/<name>.py by file path; 'vqvae' is registered in sys.modules only temporarily."""
    path = os.path.join(CRAFT, f"{name}.py")
    if not os.path.exists(path):
        pytest.skip(f"CRAFT repo not found at {CRAFT}")
    saved = sys.modules.get("vqvae")
    try:
        if name != "vqvae":
            sys.modules["vqvae"] = load_craft_module("vqvae")
        spec = importlib.util.spec_from_file_location(f"craft_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            sys.modules.pop("vqvae", None)
        else:
            sys.modules["vqvae"] = saved


def test_prepare_input_ranges():
    x = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3).expand(1, 3, 2, 3).contiguous()
    s = vre.prepare_input(x, "single")
    d = vre.prepare_input(x, "double")
    assert s.min() == -1 and s.max() == 1
    assert torch.allclose(d[0, 0, 0], torch.tensor([-3.0, -1.0, 1.0]))
    with pytest.raises(AssertionError):
        vre.prepare_input(x + 0.1, "single")
    with pytest.raises(AssertionError):
        vre.prepare_input(x - 0.1, "double")
    with pytest.raises(AssertionError):
        vre.prepare_input(x.double(), "single")
    with pytest.raises(ValueError):
        vre.prepare_input(x, "triple")


def test_nearest_codes_dtype_and_autocast():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, 512, 16, 16, generator=g)
    cb = torch.randn(1024, 512, generator=g)
    for dt in (torch.bfloat16, torch.float16):
        with pytest.raises(TypeError):
            vre.nearest_codes(z.to(dt), cb)
        with pytest.raises(TypeError):
            vre.nearest_codes(z, cb.to(dt))
    idx0, d0 = vre.nearest_codes(z, cb)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        idx1, d1 = vre.nearest_codes(z, cb)
    assert d1.dtype == torch.float32 and idx1.dtype == torch.int64
    assert torch.equal(idx0, idx1) and torch.equal(d0, d1)


def test_nearest_codes_matches_cdist():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(3, 512, 16, 16, generator=g)
    cb = torch.randn(1024, 512, generator=g)
    idx, dmin = vre.nearest_codes(z, cb)
    assert idx.shape == (3, 256) and dmin.shape == (3, 256)
    assert idx.min() >= 0 and idx.max() <= 1023
    zf = z.permute(0, 2, 3, 1).reshape(3, 256, 512)
    ref = torch.cdist(zf.double(), cb.double()).argmin(-1)
    assert (idx == ref).float().mean() > 0.999  # differences only at (near) exact ties


def test_official_prompt_indices_crafted():
    # 8 patches, codes with distances; unique codes in distance order are 5,2,9,7
    idx = torch.tensor([[2, 5, 2, 9, 5, 7, 9, 2]])
    dmin = torch.tensor([[0.5, 0.1, 0.6, 0.3, 0.2, 0.9, 0.4, 0.7]])
    out = vre.official_prompt_indices(idx, dmin)
    assert out.shape == (1, 77) and out.dtype == torch.int64
    # ascending dmin: 0.1(5) 0.2(5) 0.3(9) 0.4(9) 0.5(2) 0.6(2) 0.7(2) 0.9(7)
    assert out[0, :4].tolist() == [5, 9, 2, 7]
    assert (out[0, 4:] == -1).all()


def test_official_prompt_indices_all_equal_and_cap():
    out = vre.official_prompt_indices(torch.full((2, 256), 17), torch.rand(2, 256))
    assert (out[:, 0] == 17).all() and (out[:, 1:] == -1).all()
    # many unique codes: capped at 77, all unique, sorted by distance
    idx = torch.arange(256).unsqueeze(0)
    dmin = torch.arange(256, dtype=torch.float32).flip(0).unsqueeze(0)
    out = vre.official_prompt_indices(idx, dmin)
    assert out[0].tolist() == list(range(255, 255 - 77, -1))


def test_codes_to_embeddings_zero_rows():
    cb = torch.randn(1024, 512)
    idx = torch.tensor([[3, -1, 1023, -1]])
    e = vre.codes_to_embeddings(idx, cb)
    assert e.dtype == torch.float32 and e.shape == (1, 4, 512)
    assert torch.equal(e[0, 0], cb[3]) and torch.equal(e[0, 2], cb[1023])
    assert (e[0, 1] == 0).all() and (e[0, 3] == 0).all()


def test_craft_parity():
    cv = load_craft_module("vqvae")
    args = SimpleNamespace(cat_prompt_embedding=False, use_att_pool=False,
                           use_pos_embedding=False, learnable_pos_emb=False)
    torch.manual_seed(0)
    ref = cv.VQVAEencoder(args=args, ch=32).eval()
    ours = vre.VQBranch(vre.VREConfig(ch=32), with_decoder=False)
    ours.load_state_dict(ref.state_dict(), strict=True)
    ours.eval()
    x = torch.rand(2, 3, 512, 512) * 2 - 1
    with torch.no_grad():
        hs, _ = ref.encoder(x)
        h_ref = ref.quant_conv(hs["out"])
    h = ours.encode_features(x)
    assert h.shape == (2, 512, 16, 16)
    assert torch.allclose(h, h_ref, atol=1e-5)

    # quantizer selection on random features with many distinct codes
    g = torch.Generator().manual_seed(3)
    ref.quantize.embedding.weight.data = torch.randn(1024, 512, generator=g)
    z = torch.randn(2, 512, 16, 16, generator=g) * 3
    cb = ref.quantize.embedding.weight.data
    want = ref.quantize.inference(z)
    idx, dmin = vre.nearest_codes(z, cb)
    got = vre.codes_to_embeddings(vre.official_prompt_indices(idx, dmin), cb)
    assert torch.equal(got, want)
    assert (vre.official_prompt_indices(idx, dmin) >= 0).all()  # prompt is full here
    # and on the encoder's own (few distinct codes, padded) output
    want = ref.quantize.inference(h_ref)
    idx, dmin = vre.nearest_codes(h_ref, cb)
    assert torch.equal(vre.codes_to_embeddings(vre.official_prompt_indices(idx, dmin), cb), want)


def test_decoder_runs():
    v = vre.random_vre(seed=0, with_decoder=True)
    idx = torch.randint(0, 1024, (2, 256))
    out = v.hq.decode_indices(idx)
    assert out.shape == (2, 3, 512, 512) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_vre_methods():
    v = vre.random_vre(seed=0)
    img = torch.rand(2, 3, 512, 512)
    assert v.lq_features(img).shape == (2, 512, 16, 16)
    idx, dmin = v.lq_codes(img)
    assert idx.shape == (2, 256) and dmin.dtype == torch.float32
    p = v.lq_official_prompt(img)
    assert p.shape == (2, 77) and p.min() >= -1
    assert v.hq_codes(img, "single").shape == (2, 256)
    norms = v.hq.codebook.norm(dim=1)
    assert (norms < 0.1).sum() == 400


def _synthetic_ckpt(tmp_path, with_decoder=False, drop=None, extra=None):
    """Save a ckpt with vqvae_LQ.* (random) and vqvae.* keys; returns (Paths, ref branch)."""
    cfg = vre.VREConfig(ch=32)
    torch.manual_seed(0)
    lq = vre.VQBranch(cfg, with_decoder=with_decoder)
    hq = vre.VQBranch(cfg, with_decoder=with_decoder)
    sd = {"vqvae_LQ." + k: v for k, v in lq.state_dict().items()}
    sd.update({"vqvae." + k: v for k, v in hq.state_dict().items()})
    if drop:
        del sd[drop]
    if extra:
        sd[extra] = torch.zeros(1)
    path = tmp_path / "pretrained" / "associate_2.ckpt"
    path.parent.mkdir()
    torch.save({"state_dict": sd}, path)
    return get_paths(pretrained=str(path.parent), out_root=str(tmp_path / "out")), lq


def test_load_branch_strict(tmp_path):
    paths, lq = _synthetic_ckpt(tmp_path)
    cfg = vre.VREConfig(ch=32)
    b = vre.load_branch(paths.associate_ckpt, "lq", paths, cfg=cfg)
    assert not b.with_decoder and not any(p.requires_grad for p in b.parameters())
    assert torch.equal(b.codebook, lq.codebook)
    assert torch.equal(vre.load_codebook(paths.associate_ckpt, "lq", paths), lq.codebook)
    # a ckpt that carries decoder keys loads with the decoder, and without it (keys are dropped)
    (tmp_path / "d").mkdir()
    paths2, _ = _synthetic_ckpt(tmp_path / "d", with_decoder=True)
    b2 = vre.load_branch(paths2.associate_ckpt, "hq", paths2, with_decoder=True, cfg=cfg)
    assert b2.decoder is not None
    b3 = vre.load_branch(paths2.associate_ckpt, "hq", paths2, with_decoder=False, cfg=cfg)
    assert not hasattr(b3, "decoder")


def test_load_branch_extra_and_missing(tmp_path):
    cfg = vre.VREConfig(ch=32)
    (tmp_path / "a").mkdir()
    paths, _ = _synthetic_ckpt(tmp_path / "a", extra="vqvae_LQ.bogus.weight")
    with pytest.raises(RuntimeError, match="bogus"):
        vre.load_branch(paths.associate_ckpt, "lq", paths, cfg=cfg)
    (tmp_path / "b").mkdir()
    paths, _ = _synthetic_ckpt(tmp_path / "b", drop="vqvae_LQ.quant_conv.bias")
    with pytest.raises(RuntimeError, match="quant_conv.bias"):
        vre.load_branch(paths.associate_ckpt, "lq", paths, cfg=cfg)
    # the hq branch is untouched by damage to the lq branch
    vre.load_branch(paths.associate_ckpt, "hq", paths, cfg=cfg)


def test_load_branch_rejects_other_files(tmp_path):
    paths, _ = _synthetic_ckpt(tmp_path)
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"x")
    with pytest.raises(PermissionError):
        vre.load_branch(str(other), "lq", paths)
