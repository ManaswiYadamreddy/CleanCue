"""Per-image IQA metrics via pyiqa. Any load or compute failure raises (no NaN fallback)."""
from __future__ import annotations

import numpy as np
import torch

# metric name -> pyiqa id (unchanged from the old Evaluation code)
FULL_REF = {"lpips": "lpips", "dists": "dists", "psnr": "psnr", "ssim": "ssim"}
NO_REF = {"musiq": "musiq", "niqe": "niqe", "clipiqa": "clipiqa", "maniqa": "maniqa"}


class IQAMetrics:
    def __init__(self, metrics: list[str], device="cuda"):
        import pyiqa

        self.device = torch.device(device)
        self.metrics: dict[str, object] = {}
        for name in metrics:
            pid = FULL_REF.get(name) or NO_REF.get(name)
            if pid is None:
                raise ValueError(f"unknown IQA metric {name!r}")
            m = pyiqa.create_metric(pid, device=self.device, as_loss=False)
            m.eval()
            self.metrics[name] = m

    @torch.no_grad()
    def compute(self, restored_01: torch.Tensor, hq_01: torch.Tensor | None = None) -> dict[str, np.ndarray]:
        """(B,3,H,W) in [0,1] -> {metric: float64 array (B,)}."""
        x = restored_01.to(self.device).clamp(0, 1).float()
        ref = None if hq_01 is None else hq_01.to(self.device).clamp(0, 1).float()
        out = {}
        for name, m in self.metrics.items():
            if name in FULL_REF:
                if ref is None:
                    raise ValueError(f"{name} needs the HQ reference")
                v = m(x, ref)
            else:
                v = m(x)
            v = v.detach().double().cpu().reshape(-1).numpy()
            if v.shape[0] != x.shape[0]:
                raise RuntimeError(f"{name}: expected {x.shape[0]} values, got {v.shape}")
            out[name] = v
        return out
