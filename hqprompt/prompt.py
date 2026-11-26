"""The single prompt-construction contract (plan section 4). Every prompt in the project is built here.

Path: idx (int64) -> codebook[idx] (fp32) -> TokenProjection -> (+ learned positions) -> UNet
cross-attention. Everything here runs in fp32 with autocast disabled.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from hqprompt.paths import Paths, safe_torch_load
from hqprompt.vre import _autocast_off, codes_to_embeddings

MODES = ("official", "lq", "hq_true", "hq_pred", "null")
CODEBOOK_FOR_MODE = {"official": "lq", "lq": "lq", "hq_true": "hq", "hq_pred": "hq", "null": None}
N_TOKENS = {"official": 77, "lq": 256, "hq_true": 256, "hq_pred": 256, "null": 256}

_EC_SHAPES = {"conv1.weight": (256, 512, 1), "conv1.bias": (256,),
              "conv2.weight": (1024, 256, 1), "conv2.bias": (1024,)}


class TokenProjection(nn.Module):
    """OSDFace `TwoLayerConv1x1` without its `assert N == 77` (trap 7). Same parameter names."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv1d(256, 1024, kernel_size=1)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,N,512) -> (B,N,1024)
        x = x.permute(0, 2, 1)
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        return x.permute(0, 2, 1)


def _check_ec(sd: dict) -> dict:
    if set(sd) != set(_EC_SHAPES):
        raise KeyError(f"embedding_change keys {sorted(sd)} != {sorted(_EC_SHAPES)}")
    for k, shape in _EC_SHAPES.items():
        if tuple(sd[k].shape) != shape:
            raise ValueError(f"embedding_change[{k}] has shape {tuple(sd[k].shape)}, expected {shape}")
    return {k: v.detach().float().clone() for k, v in sd.items()}


def load_embedding_change(paths: Paths) -> dict[str, torch.Tensor]:
    return _check_ec(safe_torch_load(paths.embedding_change, paths))


def random_embedding_change_sd(seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights with the right keys/shapes (tests only)."""
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(*s, generator=g) * 0.05 for k, s in _EC_SHAPES.items()}


class PromptEncoder(nn.Module):
    def __init__(self, mode: str, codebook: torch.Tensor | None, embedding_change_sd: dict | None,
                 trainable: bool = True, null_init: Literal["pad", "zeros"] = "pad"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown prompt mode {mode!r}; expected one of {MODES}")
        if null_init not in ("pad", "zeros"):
            raise ValueError(f"unknown null_init {null_init!r}")
        self.mode = mode
        self.n_tokens = N_TOKENS[mode]

        if mode == "null":
            if codebook is not None:
                raise ValueError("null mode takes no codebook")
            init = torch.zeros(256, 1024)
            if null_init == "pad":
                # exactly the official prompt's padding token: proj(zero code)
                if embedding_change_sd is None:
                    raise ValueError("null_init='pad' needs embedding_change_sd")
                proj = TokenProjection()
                proj.load_state_dict(_check_ec(embedding_change_sd), strict=True)
                with torch.no_grad(), _autocast_off():
                    init = proj(torch.zeros(1, 256, 512))[0]
            self.tokens = nn.Parameter(init.clone().float())
            self.tokens.requires_grad_(trainable)
        else:
            if codebook is None or embedding_change_sd is None:
                raise ValueError(f"mode {mode!r} needs a codebook and embedding_change_sd")
            if codebook.dtype != torch.float32 or tuple(codebook.shape) != (1024, 512):
                raise TypeError(f"codebook must be fp32 (1024,512), got {codebook.dtype} {tuple(codebook.shape)}")
            self.register_buffer("codebook", codebook.detach().clone(), persistent=False)
            self.proj = TokenProjection()
            self.proj.load_state_dict(_check_ec(embedding_change_sd), strict=True)
            if mode == "official":
                self.proj.requires_grad_(False)  # official prompt path is frozen, whatever `trainable` says
            else:
                self.proj.requires_grad_(trainable)
                self.pos = nn.Parameter(torch.zeros(256, 1024))
                self.pos.requires_grad_(trainable)

        assert all(p.dtype == torch.float32 for p in self.parameters())
        assert all(b.dtype == torch.float32 for b in self.buffers())

    @property
    def codebook_kind(self) -> str | None:
        return CODEBOOK_FOR_MODE[self.mode]

    def forward(self, idx: torch.Tensor | None = None, batch_size: int | None = None) -> torch.Tensor:
        with _autocast_off():
            if self.mode == "null":
                if idx is not None:
                    raise ValueError("null mode takes no idx")
                if batch_size is None:
                    raise ValueError("null mode needs batch_size")
                return self.tokens[None].expand(batch_size, -1, -1)

            if idx is None:
                raise ValueError(f"mode {self.mode!r} needs idx")
            if idx.dtype != torch.int64:
                raise TypeError(f"idx must be int64, got {idx.dtype}")
            if idx.ndim != 2 or idx.shape[1] != self.n_tokens:
                raise ValueError(f"idx must be (B,{self.n_tokens}), got {tuple(idx.shape)}")
            lo = -1 if self.mode == "official" else 0  # only the official prompt is -1 padded
            if int(idx.min()) < lo or int(idx.max()) > 1023:
                raise ValueError(f"idx values must lie in [{lo},1023], got [{int(idx.min())},{int(idx.max())}]")
            emb = codes_to_embeddings(idx, self.codebook)
            out = self.proj(emb)
            if self.mode != "official":
                out = out + self.pos[None]
            return out

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, sd: dict, strict: bool = True) -> None:
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad}
        if strict and set(sd) != expected:
            raise KeyError(f"trainable keys mismatch: missing {sorted(expected - set(sd))}, "
                           f"unexpected {sorted(set(sd) - expected)}")
        with torch.no_grad():
            for n, v in sd.items():
                if n not in expected:
                    continue
                if tuple(v.shape) != tuple(params[n].shape):
                    raise ValueError(f"{n}: shape {tuple(v.shape)} != {tuple(params[n].shape)}")
                params[n].copy_(v.float())
