import numpy as np
import pytest
import torch

from hqprompt import osdface
from hqprompt.osdface import T_FIXED, get_x0_from_noise, merge_lora_state_dict


def _handmade(seed=0, r=3):
    g = torch.Generator().manual_seed(seed)
    r_ = lambda *s: torch.randn(*s, generator=g)
    unet = {"a.to_q.weight": r_(8, 6), "b.to_k.weight": r_(4, 5),
            "c.conv1.weight": r_(8, 6, 3, 3), "d.conv2.weight": r_(4, 5, 3, 3), "other.weight": r_(2, 2)}
    lora = {}
    for k, w in list(unet.items())[:4]:
        stem = "unet." + k[:-7]
        if w.ndim == 2:
            lora[stem + ".lora_A.weight"], lora[stem + ".lora_B.weight"] = r_(r, w.shape[1]), r_(w.shape[0], r)
        else:
            lora[stem + ".lora_A.weight"], lora[stem + ".lora_B.weight"] = r_(r, *w.shape[1:]), r_(w.shape[0], r, 1, 1)
    return unet, lora


def test_merge_matches_reference():
    unet, lora = _handmade()
    merged, rep = merge_lora_state_dict(unet, lora, 0.5)
    assert rep == {"n_pairs": 4, "n_conv": 2, "n_linear": 2, "rank": 3, "unprocessed": []}
    for k in ("a.to_q.weight", "b.to_k.weight"):
        A, B = lora["unet." + k[:-7] + ".lora_A.weight"], lora["unet." + k[:-7] + ".lora_B.weight"]
        torch.testing.assert_close(merged[k], unet[k] + 0.5 * B @ A)
    for k in ("c.conv1.weight", "d.conv2.weight"):
        A, B = lora["unet." + k[:-7] + ".lora_A.weight"], lora["unet." + k[:-7] + ".lora_B.weight"]
        ref = unet[k] + 0.5 * torch.einsum("or,rikl->oikl", B[:, :, 0, 0], A)
        torch.testing.assert_close(merged[k], ref)
    assert torch.equal(merged["other.weight"], unet["other.weight"])


def test_merge_legacy_format_linear_only():
    unet, _ = _handmade()
    up, down = torch.randn(8, 2), torch.randn(2, 6)
    merged, rep = merge_lora_state_dict(unet, {"unet.a.to_q.lora.up.weight": up, "unet.a.to_q.lora.down.weight": down}, 1.0)
    torch.testing.assert_close(merged["a.to_q.weight"], unet["a.to_q.weight"] + up @ down)
    assert rep["n_pairs"] == 1
    with pytest.raises(ValueError):
        merge_lora_state_dict(unet, {"unet.c.conv1.lora.up.weight": torch.randn(8, 2, 1, 1),
                                     "unet.c.conv1.lora.down.weight": torch.randn(2, 6, 3, 3)}, 1.0)


def test_merge_failures():
    unet, lora = _handmade()
    with pytest.raises(ValueError):
        merge_lora_state_dict(unet, {**lora, "unet.zzz.extra": torch.zeros(1)}, 1.0)
    with pytest.raises(KeyError):  # missing base key
        merge_lora_state_dict(unet, {**lora, "unet.nope.lora_A.weight": torch.zeros(3, 2),
                                     "unet.nope.lora_B.weight": torch.zeros(2, 3)}, 1.0)
    bad = dict(lora)
    bad["unet.a.to_q.lora_B.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError):
        merge_lora_state_dict(unet, bad, 1.0)
    with pytest.raises(ValueError):
        merge_lora_state_dict(unet, {**lora, "unet.a.to_q.alpha": torch.tensor(1.0)}, 1.0)
    del_b = {k: v for k, v in lora.items() if k != "unet.a.to_q.lora_B.weight"}
    with pytest.raises(KeyError):
        merge_lora_state_dict(unet, del_b, 1.0)


def test_unet_prefix_only_stripped_at_start():
    unet = {"down.unet.x.weight": torch.zeros(4, 4)}
    lora = {"unet.down.unet.x.lora_A.weight": torch.ones(1, 4), "unet.down.unet.x.lora_B.weight": torch.ones(4, 1)}
    merged, _ = merge_lora_state_dict(unet, lora, 1.0)
    assert torch.all(merged["down.unet.x.weight"] == 1)


def test_get_x0_float64_reference():
    g = torch.Generator().manual_seed(0)
    z, e = torch.randn(2, 4, 8, 8, generator=g), torch.randn(2, 4, 8, 8, generator=g)
    al = osdface._tiny_alphas()
    a = np.float64(al.numpy()[T_FIXED])
    ref = (z.numpy().astype(np.float64) - np.sqrt(1 - a) * e.numpy().astype(np.float64)) / np.sqrt(a)
    out = get_x0_from_noise(z, e, al)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)
    assert T_FIXED == 399


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("diffusers")
    return osdface.tiny_models(0)


class TestWithDiffusers:
    def test_deterministic(self, tiny):
        other = osdface.tiny_models(0)
        for (k, a), (_, b) in zip(tiny.unet.state_dict().items(), other.unet.state_dict().items()):
            assert torch.equal(a, b), k
        for (k, a), (_, b) in zip(tiny.vae.state_dict().items(), other.vae.state_dict().items()):
            assert torch.equal(a, b), k
        assert not torch.equal(osdface.tiny_models(1).unet.state_dict()["conv_in.weight"], tiny.unet.state_dict()["conv_in.weight"])

    def test_one_step_and_vae(self, tiny):
        z = torch.randn(2, 4, 16, 16)
        prompt = torch.randn(2, 77, 1024)
        x0 = osdface.one_step(tiny.unet, z, prompt, tiny.alphas_cumprod)
        assert x0.shape == z.shape and x0.dtype == torch.float32 and torch.isfinite(x0).all()
        x0b = osdface.one_step(tiny.unet, z, prompt, tiny.alphas_cumprod, amp_dtype=torch.bfloat16)
        assert x0b.dtype == torch.float32

    def test_vae_roundtrip_and_grad(self, tiny):
        img = torch.rand(1, 3, 64, 64) * 2 - 1
        l1 = osdface.vae_encode(tiny.vae, img, "mean")
        assert l1.shape == (1, 4, 8, 8) and torch.equal(l1, osdface.vae_encode(tiny.vae, img, "mean"))
        s1 = osdface.vae_encode(tiny.vae, img, "sample", torch.Generator().manual_seed(1))
        s2 = osdface.vae_encode(tiny.vae, img, "sample", torch.Generator().manual_seed(1))
        assert torch.equal(s1, s2)
        lat = l1.clone().requires_grad_(True)
        out = osdface.vae_decode(tiny.vae, lat, clamp=False)
        assert out.shape == (1, 3, 64, 64) and out.requires_grad
        out.sum().backward()
        assert lat.grad is not None and lat.grad.abs().sum() > 0
        assert osdface.vae_decode(tiny.vae, l1).abs().max() <= 1

    def test_merge_changes_only_targets(self, tiny):
        lora = osdface.tiny_lora_state_dict(tiny.unet)
        sd = tiny.unet.state_dict()
        merged, rep = merge_lora_state_dict(sd, lora, 1.0)
        assert rep["n_pairs"] == 4 and rep["n_conv"] == 2 and rep["n_linear"] == 2
        targets = {k[len("unet."):-len(".lora_A.weight")] + ".weight" for k in lora if k.endswith("lora_A.weight")}
        changed = {k for k in sd if not torch.equal(sd[k], merged[k])}
        assert changed == targets
        tiny.unet.load_state_dict(merged, strict=True)  # restore-able strictly
        tiny.unet.load_state_dict(sd, strict=True)

    def test_tiny_alphas_match_ddim(self, tiny):
        from diffusers import DDIMScheduler
        sch = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000)
        torch.testing.assert_close(tiny.alphas_cumprod, sch.alphas_cumprod, rtol=1e-5, atol=1e-7)

    def test_adapter_api_present(self, tiny):
        for name in ("add_adapter", "disable_adapters", "enable_adapters"):
            assert hasattr(tiny.unet, name)
