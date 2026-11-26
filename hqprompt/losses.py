"""EA-DISTS and ArcFace identity losses (fp32, frozen), plus cheap stubs for tests and dry runs."""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from hqprompt.paths import Paths


def _freeze(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


class Sobel(nn.Module):
    """3x3 Sobel magnitude per channel, min-max scaled to [0, 1] per image."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kernel", torch.cat([kx, kx.transpose(-1, -2).contiguous()], dim=0), persistent=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = F.conv2d(x.reshape(B * C, 1, H, W), self.kernel, padding=1)
        mag = torch.sqrt(g.pow(2).sum(1, keepdim=True) + self.eps).reshape(B, -1)
        mn = mag.min(1, keepdim=True).values
        mx = mag.max(1, keepdim=True).values
        return ((mag - mn) / (mx - mn).clamp_min(self.eps)).reshape(B, C, H, W)


class EADists(nn.Module):
    """DISTS(pred, target) + DISTS(Sobel(pred), Sobel(target)); inputs in [0,1]."""

    def __init__(self, device="cpu"):
        super().__init__()
        import pyiqa
        self.dists = _freeze(pyiqa.create_metric("dists", as_loss=True, device=device))
        self.sobel = Sobel()

    def train(self, mode: bool = True):
        super().train(mode)
        self.dists.eval()
        return self

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        with torch.autocast(pred01.device.type, enabled=False):
            p, t = pred01.float(), target01.float()
            return self.dists(p, t).mean() + self.dists(self.sobel(p), self.sobel(t)).mean()


def _patch_flatten_view(net: nn.Module) -> None:
    """facexlib's Flatten uses .view, which fails on non-contiguous tensors in torch 2.x."""
    def fwd(self, x):
        return x.reshape(x.size(0), -1)
    for m in net.modules():
        cls = m.__class__
        if cls.__name__ == "Flatten" and cls.__module__.endswith("arcface_arch"):
            m.forward = types.MethodType(fwd, m)


class ArcFaceID(nn.Module):
    """mean(1 - cos(F(pred), F(target))) with facexlib ArcFace; no alignment (faces are aligned)."""

    _MEAN = 127.5 / 255.0
    _STD = 128.0 / 255.0

    def __init__(self, paths: Paths, device="cpu", input_size: int = 112):
        super().__init__()
        from facexlib.recognition import init_recognition_model
        self.input_size = input_size
        self.arcface = _freeze(init_recognition_model("arcface", half=False, device=device,
                                                      model_rootpath=paths.facexlib_dir))
        _patch_flatten_view(self.arcface)

    def train(self, mode: bool = True):
        super().train(mode)
        self.arcface.eval()
        return self

    def embed(self, img01: torch.Tensor) -> torch.Tensor:
        with torch.autocast(img01.device.type, enabled=False):
            x = F.interpolate(img01.float(), size=(self.input_size,) * 2, mode="bicubic",
                              align_corners=False, antialias=True)
            x = (x - self._MEAN) / self._STD
            return F.normalize(self.arcface(x), dim=-1)

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            et = self.embed(target01)
        return (1.0 - (self.embed(pred01) * et).sum(-1)).mean()


class StubDists(nn.Module):
    """Differentiable stand-in: L1 of 4x avg-pooled images + L1 of their Sobel maps."""

    def __init__(self):
        super().__init__()
        self.sobel = Sobel()

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        p, t = pred01.float(), target01.float()
        return (F.l1_loss(F.avg_pool2d(p, 4), F.avg_pool2d(t, 4))
                + F.l1_loss(self.sobel(p), self.sobel(t)))


class StubID(nn.Module):
    """Fixed seeded random conv net -> GAP -> normalize; 1 - cos."""

    def __init__(self, seed: int = 0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.Tanh(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1))
        _freeze(self.net)

    def embed(self, img01: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(img01.float()).mean((2, 3)), dim=-1)

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        return (1.0 - (self.embed(pred01) * self.embed(target01)).sum(-1)).mean()


def build_losses(paths: Paths, device, dry_run: bool) -> dict[str, nn.Module]:
    if dry_run:
        mods = {"ea_dists": StubDists(), "id": StubID()}
    else:
        mods = {"ea_dists": EADists(device=device), "id": ArcFaceID(paths, device=device)}
    return {k: _freeze(m.to(device)) for k, m in mods.items()}
