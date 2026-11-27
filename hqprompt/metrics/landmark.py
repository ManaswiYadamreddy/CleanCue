"""LMD: mean 68-point FAN landmark distance in pixels. Per-image detection failure is recorded
as NaN plus a reason in `lmd_fail`; every other error raises."""
from __future__ import annotations

import numpy as np
import torch


class LandmarkDistance:
    def __init__(self, device="cuda"):
        import face_alignment

        self.device = torch.device(device)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device=str(self.device), flip_input=False,
            compile=False)  # torch.compile fails with SCC g++ (no C++17 attributes)

    def _landmarks(self, img_uint8_hwc: np.ndarray):
        """(68,2) float32 of the largest detected face, or None."""
        lms = self.fa.get_landmarks_from_image(img_uint8_hwc)
        if lms is None or len(lms) == 0:
            return None
        if len(lms) > 1:
            areas = [np.ptp(lm[:, 0]) * np.ptp(lm[:, 1]) for lm in lms]
            lms = [lms[int(np.argmax(areas))]]
        return lms[0].astype(np.float32)

    @staticmethod
    def _to_uint8_hwc(img_01: torch.Tensor) -> np.ndarray:
        return (img_01.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    @torch.no_grad()
    def compute(self, restored_01: torch.Tensor, hq_01: torch.Tensor) -> list[dict]:
        """Per image: {"lmd": float|nan, "lmd_fail": ""|"restored"|"hq"|"both"}."""
        out = []
        for i in range(restored_01.shape[0]):
            lm_r = self._landmarks(self._to_uint8_hwc(restored_01[i]))
            lm_h = self._landmarks(self._to_uint8_hwc(hq_01[i]))
            if lm_r is None or lm_h is None:
                fail = "both" if lm_r is None and lm_h is None else ("restored" if lm_r is None else "hq")
                out.append({"lmd": float("nan"), "lmd_fail": fail})
            else:
                out.append({"lmd": float(np.linalg.norm(lm_r - lm_h, axis=1).mean()), "lmd_fail": ""})
        return out
