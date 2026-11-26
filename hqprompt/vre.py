"""HQ and LQ VQ branches of the OSDFace "VRE" (visual representation encoder), with fp32 code lookup.

Ported from CRAFT `vqvae.py` (itself from the official OSDFace code). Parameter and attribute
names are kept identical so `associate_2.ckpt` keys load with strict=True.

Differences from the source, all deliberate:
  * `MultiHeadDecoder.forward` unpacks the `(h, attn)` tuple returned by the attention block
    (the original crashes), and its `print` is gone.
  * `MultiHeadAttnBlock` no longer deep-copies the attention map (output unchanged) and does not
    use an in-place `mul_` on `q`.
  * Code lookup never touches the model dtype: it is fp32 raw-L2 argmin (plan trap 1).
"""
from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from hqprompt.paths import Paths, safe_torch_load

log = logging.getLogger(__name__)

BRANCH_PREFIX = {"hq": "vqvae.", "lq": "vqvae_LQ."}  # "hq" = vqvae.* is an ASSUMPTION until check C1 passes
LQ_INPUT_MODE = "double"  # official path (infer.py + lq_embed/inference.py): trap 6


@dataclass(frozen=True)
class VREConfig:
    n_embed: int = 1024
    embed_dim: int = 512
    ch: int = 64
    ch_mult: tuple = (1, 2, 2, 2, 4, 8)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (16,)
    z_channels: int = 512
    resolution: int = 512
    head_size: int = 1
    dropout: float = 0.0


# --------------------------------------------------------------------------- building blocks
def nonlinearity(x):
    return x * torch.sigmoid(x)  # swish


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert in_channels % head_size == 0, "The size of head should be divided by the number of channels."

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.num = 0

    def forward(self, x, y=None):
        h_ = self.norm1(x)
        y = h_ if y is None else self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)  # b, hw, head, att
        k = k.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)
        v = v.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * (int(self.att_size) ** (-0.5))
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)
        attn = w_  # no deepcopy needed: nothing below modifies w_ in place

        w_ = w_.matmul(v)
        w_ = w_.transpose(1, 2).contiguous()  # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1).permute(0, 3, 1, 2)
        w_ = self.proj_out(w_)
        return x + w_, attn


class MultiHeadEncoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, double_z=True, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                           temb_channels=self.temb_ch, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                           temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                        kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Returns (hs, attn_weight) like the source; the feature map is hs['out']."""
        hs = {}
        temb = None
        h = self.conv_in(x)
        hs["in"] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h, _ = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                hs["block_" + str(i_level)] = h
                h = self.down[i_level].downsample(h)
        atten_weight = None
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs["block_" + str(i_level) + "_atten"] = h
            h, atten_weight = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs["mid_atten"] = h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        hs["out"] = h
        return hs, atten_weight


class MultiHeadDecoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                           temb_channels=self.temb_ch, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                           temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h, _ = self.mid.attn_1(h)  # bug fix: attention block returns (x + w_, attn)
            h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h, _ = self.up[i_level].attn[i_block](h)  # bug fix, as above
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


# --------------------------------------------------------------------------- branch
class Codebook(nn.Module):
    """Holds only `embedding`, so the state-dict key is `quantize.embedding.weight`."""

    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)


class VQBranch(nn.Module):
    def __init__(self, cfg: VREConfig = VREConfig(), with_decoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultiHeadEncoder(
            ch=cfg.ch, out_ch=3, ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=cfg.attn_resolutions, dropout=cfg.dropout, in_channels=3,
            resolution=cfg.resolution, z_channels=cfg.z_channels, double_z=False,
            enable_mid=True, head_size=cfg.head_size)
        self.quant_conv = nn.Conv2d(cfg.z_channels, cfg.embed_dim, 1)
        self.quantize = Codebook(cfg.n_embed, cfg.embed_dim)
        self.with_decoder = with_decoder
        if with_decoder:
            self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.z_channels, 1)
            self.decoder = MultiHeadDecoder(
                ch=cfg.ch, out_ch=3, ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
                attn_resolutions=cfg.attn_resolutions, dropout=cfg.dropout, in_channels=3,
                resolution=cfg.resolution, z_channels=cfg.z_channels, enable_mid=True,
                head_size=cfg.head_size)

    @torch.no_grad()
    def encode_features(self, x_enc: torch.Tensor) -> torch.Tensor:
        """x_enc (B,3,512,512) fp32 from `prepare_input`; returns (B,512,16,16) fp32."""
        if x_enc.dtype != torch.float32:
            raise TypeError(f"encode_features needs fp32 input, got {x_enc.dtype}")
        with _autocast_off():
            hs, _ = self.encoder(x_enc)
            return self.quant_conv(hs["out"])

    @property
    def codebook(self) -> torch.Tensor:
        return self.quantize.embedding.weight

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """idx (B,256) -> raw decoder output (B,3,512,512) fp32 (no output mapping applied)."""
        assert self.with_decoder, "branch was built without a decoder"
        assert idx.dtype == torch.int64 and idx.ndim == 2, (idx.dtype, idx.shape)
        B, N = idx.shape
        w = int(round(N ** 0.5))
        assert w * w == N, f"{N} codes is not a square grid"
        with _autocast_off():
            rows = self.codebook[idx]  # (B,N,C)
            z = rows.permute(0, 2, 1).reshape(B, rows.shape[-1], w, w)  # raster order: p = row*w + col
            return self.decoder(self.post_quant_conv(z))


# --------------------------------------------------------------------------- numerics helpers
def prepare_input(img01: torch.Tensor, mode: Literal["single", "double"]) -> torch.Tensor:
    """Map an image in [0,1] to the encoder input range.

    "single": 2x-1 in [-1,1].  "double": (2x-1-0.5)/0.5 = 4x-3 in [-3,1], which is what the official
    LQ path actually feeds the encoder (trap 6).
    """
    assert img01.dtype == torch.float32, f"img01 must be fp32, got {img01.dtype}"
    assert img01.ndim == 4 and img01.shape[1] == 3, img01.shape
    lo, hi = float(img01.min()), float(img01.max())
    assert lo >= -1e-6 and hi <= 1 + 1e-6, f"img01 outside [0,1]: [{lo}, {hi}]"
    x = img01 * 2 - 1
    if mode == "single":
        return x
    if mode == "double":
        return (x - 0.5) / 0.5
    raise ValueError(f"unknown input mode {mode!r}")


@contextmanager
def _autocast_off():
    with ExitStack() as st:
        st.enter_context(torch.autocast("cpu", enabled=False))
        if torch.cuda.is_available():
            st.enter_context(torch.autocast("cuda", enabled=False))
        yield


@contextmanager
def fp32_lookup():
    """Autocast off (cpu and cuda) and TF32 matmul off, so the distance matmul is true fp32."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with _autocast_off():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def nearest_codes(z: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest codebook row per patch, official raw-L2 formula, strictly fp32.

    z (B,512,16,16), codebook (1024,512). Returns idx (B,256) int64 and dmin (B,256) fp32.
    """
    if z.dtype != torch.float32 or codebook.dtype != torch.float32:
        raise TypeError(f"nearest_codes needs fp32 inputs, got z={z.dtype}, codebook={codebook.dtype}")
    with fp32_lookup():
        B, C = z.shape[0], z.shape[1]
        zf = z.permute(0, 2, 3, 1).reshape(B, -1, C)  # raster order
        d = zf.pow(2).sum(-1, keepdim=True) + codebook.pow(2).sum(-1) - 2 * zf @ codebook.T
        dmin, idx = d.min(dim=-1)
    return idx, dmin


def official_prompt_indices(idx: torch.Tensor, dmin: torch.Tensor, n_min: int = 77) -> torch.Tensor:
    """Official 77-token prompt selection, per sample (port of CRAFT VectorQuantizer.inference).

    Patches are ordered by their min distance (ascending); the first n_min distinct codes are kept.
    Returns (B,n_min) int64, padded with -1 when fewer are available.
    """
    B = idx.shape[0]
    out = torch.full((B, n_min), -1, dtype=torch.int64)
    for i in range(B):
        order = torch.sort(dmin[i], dim=0).indices  # default stable=False, as in the official code
        seen, picked = set(), []
        for c in idx[i][order].tolist():
            if c not in seen:
                seen.add(c)
                picked.append(c)
                if len(picked) >= n_min:
                    break
        out[i, :len(picked)] = torch.tensor(picked, dtype=torch.int64)
    return out.to(idx.device)


def codes_to_embeddings(idx: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """idx (B,N) int64 (-1 allowed) -> (B,N,512) fp32; rows with idx == -1 are exact zeros."""
    assert idx.dtype == torch.int64, idx.dtype
    assert codebook.dtype == torch.float32, codebook.dtype
    valid = idx >= 0
    return codebook[idx.clamp(min=0)] * valid.unsqueeze(-1).to(codebook.dtype)


# --------------------------------------------------------------------------- loading
def _branch_state(ckpt_path: str, branch: str, paths: Paths) -> dict:
    prefix = BRANCH_PREFIX[branch]
    sd = safe_torch_load(ckpt_path, paths)["state_dict"]
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no keys with prefix {prefix!r} in {ckpt_path}")
    return out


def load_codebook(ckpt_path: str, branch: str, paths: Paths) -> torch.Tensor:
    """(1024,512) fp32 codebook of one branch, without building the model."""
    sd = _branch_state(ckpt_path, branch, paths)
    key = "quantize.embedding.weight"
    if key not in sd:
        raise KeyError(f"{BRANCH_PREFIX[branch]}{key} missing from {ckpt_path}")
    return sd[key].detach().float().clone()


def load_branch(ckpt_path: str, branch: Literal["hq", "lq"], paths: Paths, with_decoder: bool = False,
                cfg: VREConfig = VREConfig(), device="cpu") -> VQBranch:
    sd = _branch_state(ckpt_path, branch, paths)
    n_branch = len(sd)
    if not with_decoder:
        sd = {k: v for k, v in sd.items() if not k.startswith(("decoder.", "post_quant_conv."))}
    model = VQBranch(cfg, with_decoder=with_decoder)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"strict load of branch {branch!r} ({BRANCH_PREFIX[branch]}*) failed: {e}") from e
    if with_decoder:
        n_model = len(model.state_dict())
        log.info("branch %s: %d checkpoint tensors, %d module tensors (plan expects 405)", branch, n_branch, n_model)
        assert n_branch == n_model, f"branch {branch}: {n_branch} ckpt tensors vs {n_model} module tensors"
    model = model.float().to(device).eval().requires_grad_(False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model


class VRE(nn.Module):
    def __init__(self, hq: VQBranch | None, lq: VQBranch | None):
        super().__init__()
        self.hq = hq
        self.lq = lq

    def lq_features(self, img01: torch.Tensor) -> torch.Tensor:
        return self.lq.encode_features(prepare_input(img01, LQ_INPUT_MODE))

    def lq_codes(self, img01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return nearest_codes(self.lq_features(img01), self.lq.codebook)

    def lq_official_prompt(self, img01: torch.Tensor) -> torch.Tensor:
        idx, dmin = self.lq_codes(img01)
        return official_prompt_indices(idx, dmin)

    def hq_codes(self, img01: torch.Tensor, input_mode: str) -> torch.Tensor:
        """input_mode has no default on purpose: it must come from check C1."""
        z = self.hq.encode_features(prepare_input(img01, input_mode))
        return nearest_codes(z, self.hq.codebook)[0]


def load_vre(paths: Paths, branches=("hq", "lq"), with_decoder: bool = False, device="cpu") -> VRE:
    br = {b: load_branch(paths.associate_ckpt, b, paths, with_decoder=with_decoder, device=device)
          if b in branches else None for b in ("hq", "lq")}
    return VRE(br["hq"], br["lq"]).to(device)


def random_vre(seed: int = 0, ch: int = 32, with_decoder: bool = False, near_zero_rows: int = 400) -> VRE:
    """Random-weight VRE for tests. ch=32 because GroupNorm uses 32 groups."""
    cfg = VREConfig(ch=ch)
    branches = []
    with torch.random.fork_rng(devices=[]):
        for k in range(2):
            torch.manual_seed(seed * 2 + k)
            b = VQBranch(cfg, with_decoder=with_decoder)
            with torch.no_grad():
                w = b.quantize.embedding.weight
                w.copy_(torch.randn_like(w) * 0.1)
                w[:near_zero_rows] = torch.randn(near_zero_rows, w.shape[1]) * 1e-3  # norm ~0.02
            branches.append(b.eval().requires_grad_(False))
    return VRE(hq=branches[0], lq=branches[1])
