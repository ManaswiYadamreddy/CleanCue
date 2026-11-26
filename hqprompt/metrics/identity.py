"""'Deg.': angular distance (degrees) between ArcFace embeddings of restored and HQ faces."""
from __future__ import annotations

import math
import types

import numpy as np
import torch
import torch.nn.functional as F

from hqprompt.paths import Paths


DEG_PREPROCESS = "resize to 112x112: bicubic, antialias=True, align_corners=False (as losses.py ArcFace)"


class IdentityDistance:
    def __init__(self, paths: Paths, device="cuda"):
        from facexlib.recognition import init_recognition_model

        self.device = torch.device(device)
        self.net = init_recognition_model("arcface", device=self.device,
                                          model_rootpath=paths.facexlib_dir)
        self.net.eval()
        self._patch_flatten_view(self.net)

    @staticmethod
    def _patch_flatten_view(net: torch.nn.Module) -> None:
        """facexlib's Flatten uses .view, which fails on non-contiguous input; use .reshape."""
        def fwd(self, x):
            return x.reshape(x.size(0), -1)

        for m in net.modules():
            cls = m.__class__
            if cls.__name__ == "Flatten" and cls.__module__.endswith("arcface_arch"):
                m.forward = types.MethodType(fwd, m)

    @torch.no_grad()
    def _embed(self, imgs_01: torch.Tensor) -> torch.Tensor:
        x = imgs_01.to(self.device).float().clamp(0, 1)
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode="bicubic", align_corners=False,
                              antialias=True)
        return F.normalize(self.net(x * 2.0 - 1.0), p=2, dim=1)

    @torch.no_grad()
    def compute(self, restored_01: torch.Tensor, hq_01: torch.Tensor) -> dict[str, np.ndarray]:
        cos = (self._embed(restored_01) * self._embed(hq_01)).sum(1).clamp(-1 + 1e-7, 1 - 1e-7)
        deg = torch.acos(cos) * (180.0 / math.pi)
        return {"deg": deg.double().cpu().numpy(), "cos_sim": cos.double().cpu().numpy()}
