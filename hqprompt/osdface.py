"""Merged OSDFace UNet, frozen VAE, and the one-step x0 forward.

Traps handled here: t is fixed at 399, LoRA conv deltas are never built with torch.mm,
x0 is computed in float64, decode keeps gradients, bf16 autocast wraps only UNet/VAE calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hqprompt.paths import Paths

T_FIXED = 399
SCALING_FACTOR = 0.18215


@dataclass
class OSDFaceModels:
    unet: "UNet2DConditionModel"  # noqa: F821
    vae: "AutoencoderKL"  # noqa: F821
    alphas_cumprod: torch.Tensor  # (1000,) float32 as the scheduler gives it
    scaling_factor: float


def merge_lora_state_dict(unet_sd: dict, lora_sd: dict, scale: float) -> tuple[dict, dict]:
    """Return (merged copy of unet_sd, report). W' = W + scale * delta, computed in fp32."""
    if any("alpha" in k for k in lora_sd):
        raise ValueError(f"alpha keys not supported: {[k for k in lora_sd if 'alpha' in k]}")
    merged = dict(unet_sd)
    done: set[str] = set()
    n_pairs = n_conv = n_linear = 0
    ranks: set[int] = set()
    for key in lora_sd:
        # lora_A / lora.down hold the "A" matrix; lora_B / lora.up hold "B".
        if key.endswith(".lora_A.weight"):
            stem, a_key, b_key = key[: -len(".lora_A.weight")], key, key[: -len("lora_A.weight")] + "lora_B.weight"
        elif key.endswith(".lora.down.weight"):
            stem, a_key, b_key = key[: -len(".lora.down.weight")], key, key[: -len("down.weight")] + "up.weight"
        else:
            continue
        legacy = key.endswith(".lora.down.weight")
        base = stem[len("unet."):] if stem.startswith("unet.") else stem
        base += ".weight"
        if b_key not in lora_sd:
            raise KeyError(f"missing partner {b_key!r} for {a_key!r}")
        if base not in unet_sd:
            raise KeyError(f"LoRA target {base!r} (from {a_key!r}) not in UNet state dict")
        A, B = lora_sd[a_key].float(), lora_sd[b_key].float()
        W = unet_sd[base].float()
        if W.ndim == 4:
            if legacy:
                raise ValueError(f"legacy lora.up/down format only supports 2-D weights: {a_key}")
            out_c, in_c, kh, kw = W.shape
            r = A.shape[0]
            if A.shape != (r, in_c, kh, kw) or B.shape != (out_c, r, 1, 1):
                raise ValueError(f"conv LoRA shape mismatch for {base}: W{tuple(W.shape)} A{tuple(A.shape)} B{tuple(B.shape)}")
            delta = (B.reshape(out_c, r) @ A.reshape(r, -1)).reshape_as(W)
            n_conv += 1
        elif W.ndim == 2:
            r = A.shape[0]
            if A.shape != (r, W.shape[1]) or B.shape != (W.shape[0], r):
                raise ValueError(f"linear LoRA shape mismatch for {base}: W{tuple(W.shape)} A{tuple(A.shape)} B{tuple(B.shape)}")
            delta = B @ A
            n_linear += 1
        else:
            raise ValueError(f"unsupported weight ndim {W.ndim} for {base}")
        merged[base] = (W + scale * delta).to(unet_sd[base].dtype)
        ranks.add(r)
        done.update((a_key, b_key))
        n_pairs += 1
    unprocessed = sorted(set(lora_sd) - done)
    if unprocessed:
        raise ValueError(f"unprocessed LoRA keys ({len(unprocessed)}): {unprocessed[:10]}")
    if len(ranks) > 1:
        raise ValueError(f"mixed LoRA ranks: {sorted(ranks)}")
    report = {"n_pairs": n_pairs, "n_conv": n_conv, "n_linear": n_linear,
              "rank": next(iter(ranks)) if ranks else None, "unprocessed": unprocessed}
    return merged, report


def load_merged_unet(paths: Paths, expected_pairs=278, expected_conv=64, expected_rank=16, scale=1.0):
    from diffusers import UNet2DConditionModel
    from safetensors import safe_open

    unet = UNet2DConditionModel.from_pretrained(paths.sd21, subfolder="unet", torch_dtype=torch.float32)
    with safe_open(paths.lora, framework="pt") as f:
        lora_sd = {k: f.get_tensor(k) for k in f.keys()}
    merged, report = merge_lora_state_dict(unet.state_dict(), lora_sd, scale)
    if (report["n_pairs"], report["n_conv"], report["rank"]) != (expected_pairs, expected_conv, expected_rank):
        raise AssertionError(f"LoRA merge counts {report} != expected pairs={expected_pairs} "
                             f"conv={expected_conv} rank={expected_rank}")
    unet.load_state_dict(merged, strict=True)
    return unet.requires_grad_(False).eval(), report


def load_vae(paths: Paths):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(paths.sd21, subfolder="vae", torch_dtype=torch.float32)
    assert abs(vae.config.scaling_factor - SCALING_FACTOR) < 1e-9, vae.config.scaling_factor
    return vae.requires_grad_(False).eval()


def load_alphas(paths: Paths) -> torch.Tensor:
    from diffusers import DDIMScheduler

    sch = DDIMScheduler.from_pretrained(paths.sd21, subfolder="scheduler")
    assert sch.config.prediction_type == "epsilon", sch.config.prediction_type
    assert sch.config.num_train_timesteps == 1000, sch.config.num_train_timesteps
    return sch.alphas_cumprod.clone()


def load_osdface(paths: Paths, device, merge_lora: bool = True) -> tuple[OSDFaceModels, dict]:
    if merge_lora:
        unet, report = load_merged_unet(paths)
    else:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(paths.sd21, subfolder="unet", torch_dtype=torch.float32)
        unet.requires_grad_(False).eval()
        report = {}
    vae = load_vae(paths)
    models = OSDFaceModels(unet.to(device), vae.to(device), load_alphas(paths), vae.config.scaling_factor)
    return models, report


def _tiny_alphas() -> torch.Tensor:
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def tiny_models(seed: int = 0, device="cpu") -> OSDFaceModels:
    from diffusers import AutoencoderKL, UNet2DConditionModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        unet = UNet2DConditionModel(
            sample_size=64, in_channels=4, out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=1024, attention_head_dim=8, norm_num_groups=32)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, block_out_channels=(32, 32, 64, 64),
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            latent_channels=4, norm_num_groups=32, sample_size=64, scaling_factor=SCALING_FACTOR)
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    return OSDFaceModels(unet.to(device), vae.to(device), _tiny_alphas(), SCALING_FACTOR)


def tiny_lora_state_dict(unet, seed: int = 0, rank: int = 4) -> dict:
    """Synthetic LoRA for a few linear (attention to_q/to_k) and conv (resnet conv1) weights."""
    g = torch.Generator().manual_seed(seed)
    sd = unet.state_dict()
    linear = [k for k, v in sd.items() if v.ndim == 2 and (".attn1.to_q.weight" in k or ".attn1.to_k.weight" in k)][:2]
    conv = [k for k, v in sd.items() if v.ndim == 4 and k.endswith("resnets.0.conv1.weight")][:2]
    assert linear and conv, "tiny unet has no suitable LoRA targets"
    out = {}
    for k in linear + conv:
        W = sd[k]
        stem = "unet." + k[: -len(".weight")]
        if W.ndim == 2:
            a_shape, b_shape = (rank, W.shape[1]), (W.shape[0], rank)
        else:
            a_shape, b_shape = (rank, *W.shape[1:]), (W.shape[0], rank, 1, 1)
        out[f"{stem}.lora_A.weight"] = torch.randn(a_shape, generator=g) * 0.05
        out[f"{stem}.lora_B.weight"] = torch.randn(b_shape, generator=g) * 0.05
    return out


def vae_encode(vae, img_pm1, mode: Literal["mean", "sample"], generator=None, amp_dtype=None) -> torch.Tensor:
    with torch.autocast(img_pm1.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        dist = vae.encode(img_pm1).latent_dist
        lat = dist.mean if mode == "mean" else dist.sample(generator)
    return lat.float() * vae.config.scaling_factor


def vae_decode(vae, lat_scaled, amp_dtype=None, clamp: bool = True) -> torch.Tensor:
    """Decode with gradients (no no_grad: trap 3). Returns fp32 (B,3,H,W)."""
    with torch.autocast(lat_scaled.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = vae.decode(lat_scaled / vae.config.scaling_factor).sample
    out = out.float()
    return out.clamp(-1, 1) if clamp else out


def get_x0_from_noise(z_t, eps, alphas_cumprod, t: int = T_FIXED) -> torch.Tensor:
    a = alphas_cumprod.to(z_t.device).double()[t]
    x0 = (z_t.double() - (1 - a).sqrt() * eps.double()) / a.sqrt()
    return x0.float()


def one_step(unet, z_lq, prompt_embeds, alphas_cumprod, t: int = T_FIXED, amp_dtype=None) -> torch.Tensor:
    ts = torch.full((z_lq.shape[0],), t, device=z_lq.device, dtype=torch.long)
    with torch.autocast(z_lq.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        eps = unet(z_lq, ts, encoder_hidden_states=prompt_embeds).sample
    return get_x0_from_noise(z_lq, eps.float(), alphas_cumprod, t)
