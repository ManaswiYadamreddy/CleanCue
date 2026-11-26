"""Dataset-level FID from pyiqa's Inception-V3 features, with the reference statistics cached.

The reference set (mu, sigma) is computed once and stored at
{eval_dir}/fid_ref_stats/<sha256(realpath(ref_dir) + file count)>.npz. Missing dirs, missing pyiqa internals
or any failure raise (no silent fallback to pyiqa's uncached path).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from hqprompt.paths import Paths, assert_writable

_IMG_EXT = (".png", ".jpg", ".jpeg")


def ref_stats_key(ref_dir: str) -> str:
    """sha256 of realpath(ref_dir) and the number of image files in it."""
    n = sum(1 for f in os.listdir(ref_dir) if f.lower().endswith(_IMG_EXT))
    return hashlib.sha256(f"{os.path.realpath(ref_dir)}|{n}".encode()).hexdigest()


def ref_stats_path(eval_dir: str, ref_dir: str) -> str:
    return os.path.join(eval_dir, "fid_ref_stats", ref_stats_key(ref_dir) + ".npz")


def stats_from_features(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def load_or_compute_ref_stats(ref_dir: str, paths: Paths, features_fn) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (mu, sigma, cache_hit). features_fn(dir) -> (N, D) array; only called on a cache miss."""
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"FID directory missing: {ref_dir!r}")
    path = ref_stats_path(paths.eval_dir, ref_dir)
    if os.path.exists(path):
        with np.load(path) as z:
            return z["mu"], z["sigma"], True
    mu, sigma = stats_from_features(features_fn(ref_dir))
    rp = assert_writable(path, paths)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    tmp = rp + ".tmp.npz"
    np.savez(tmp, mu=mu, sigma=sigma, ref_dir=os.path.realpath(ref_dir))
    os.replace(tmp, rp)
    return mu, sigma, False


class FIDScorer:
    def __init__(self, device="cuda", paths: Paths | None = None):
        import pyiqa
        try:
            from pyiqa.archs import fid_arch
            self._get_features = fid_arch.get_folder_features
            self._frechet = fid_arch.frechet_distance
        except (ImportError, AttributeError) as e:
            raise RuntimeError("pyiqa.archs.fid_arch lacks get_folder_features / frechet_distance; the cached FID "
                               "needs those helpers (check the pinned pyiqa version)") from e
        if paths is None:
            raise ValueError("FIDScorer needs paths (for the reference-stats cache)")
        self.paths = paths
        self.device = torch.device(device)
        self.metric = pyiqa.create_metric("fid", device=self.device)
        self.cache_hit: bool | None = None

    def _features(self, d: str) -> np.ndarray:
        model = None
        for attr in ("feature_extractor", "inception", "model", "net"):
            if isinstance(getattr(self.metric, attr, None), torch.nn.Module):
                model = getattr(self.metric, attr)
                break
        if model is None:
            raise RuntimeError("could not find the Inception feature extractor on the pyiqa fid metric")
        return np.asarray(self._get_features(d, model, device=self.device))

    @torch.no_grad()
    def compute(self, restored_dir: str, reference_dir: str) -> float:
        for d in (restored_dir, reference_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"FID directory missing: {d!r}")
        mu_r, sig_r, self.cache_hit = load_or_compute_ref_stats(reference_dir, self.paths, self._features)
        mu, sig = stats_from_features(self._features(restored_dir))
        return float(self._frechet(mu, sig, mu_r, sig_r))
