"""Splits, image IO, the on-disk cache format and the datasets that read it.

Cache layout (written by `CacheWriter`, read by `CacheReader`):
    cache/manifest.json                    shared by all splits
    cache/{split}/ids.npy                  int32 (N,), the sorted split ids
    cache/{split}/{item}.npy               np.lib.format.open_memmap array (N, *tail)
    cache/{split}/{item}.done.npy          bool (N,), True once the row is fully written
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from hqprompt.common import atomic_write_json, env_info, sha256_list
from hqprompt.paths import Paths, assert_writable

SPLIT_SIZES = {"pred_train": 59000, "ft_train": 10000, "val": 500, "spare": 500}
N_FFHQ = 70000
ITEM_SPECS = {  # tail shape, numpy dtype
    "hq_idx":      ((256,), "int16"),
    "lq_idx":      ((256,), "int16"),
    "lq_prompt77": ((77,), "int16"),  # official 77-token LQ prompt indices, -1 padded
    "z_lq":        ((512, 16, 16), "float16"),
    "lat_lq":      ((4, 64, 64), "float16"),
    "lat_teacher": ((4, 64, 64), "float16"),
}
SPLIT_ITEMS = {
    "pred_train": ["hq_idx", "z_lq"],
    "ft_train":   ["hq_idx", "z_lq", "lq_idx", "lq_prompt77", "lat_lq", "lat_teacher"],
    "val":        ["hq_idx", "z_lq", "lq_idx", "lq_prompt77", "lat_lq", "lat_teacher"],
}
_INDEX_ITEMS = ("hq_idx", "lq_idx", "lq_prompt77")
_IMG_EXT = (".png", ".jpg", ".jpeg")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- splits
def make_splits(n_total: int = N_FFHQ, seed: int = 0, sizes: dict = SPLIT_SIZES) -> dict:
    assert sum(sizes.values()) == n_total, f"split sizes sum to {sum(sizes.values())}, not {n_total}"
    perm = np.random.default_rng(seed).permutation(n_total)
    splits, start = {}, 0
    for name in ("pred_train", "ft_train", "val", "spare"):
        splits[name] = sorted(int(i) for i in perm[start:start + sizes[name]])
        start += sizes[name]
    return {
        "version": 1, "seed": seed, "n_total": n_total, "source": "ffhq",
        "hq_pattern": "{id:05d}.png", "lq_pattern": "{id:05d}_LQ.png",
        "method": "np.random.default_rng(seed).permutation(n_total); contiguous slices "
                  "pred_train,ft_train,val,spare; each sorted",
        "sizes": dict(sizes), "splits": splits, "sha256": sha256_list(splits),
    }


def ffhq_hq_path(paths: Paths, i: int) -> str:
    return f"{paths.ffhq_hq}/{int(i):05d}.png"


def ffhq_lq_path(paths: Paths, i: int) -> str:
    return f"{paths.ffhq_lq}/{int(i):05d}_LQ.png"


def write_splits(paths: Paths, splits: dict, hq_dir: str, lq_dir: str, check_files: bool = True) -> None:
    if check_files:
        missing = []
        for i in sorted(i for ids in splits["splits"].values() for i in ids):
            for p in (f"{hq_dir}/{i:05d}.png", f"{lq_dir}/{i:05d}_LQ.png"):
                if not os.path.exists(p):
                    missing.append(p)
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing files, first 20: {missing[:20]}")
    out = dict(splits, hq_dir=hq_dir, lq_dir=lq_dir)
    target = assert_writable(paths.splits_json, paths)
    if os.path.exists(target):
        with open(target) as f:
            old = json.load(f)
        strip = lambda d: {k: v for k, v in d.items() if k != "created"}  # noqa: E731
        if strip(old) != strip(out):
            raise FileExistsError(f"{target} exists with different content; refusing to overwrite")
        return
    out["created"] = _now()
    atomic_write_json(target, out, paths)


def load_splits(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    sp = d["splits"]
    assert set(sp) == set(d["sizes"]), f"split names {sorted(sp)} != sizes {sorted(d['sizes'])}"
    for name, ids in sp.items():
        assert len(ids) == d["sizes"][name], f"split {name}: {len(ids)} ids, expected {d['sizes'][name]}"
        assert ids == sorted(ids), f"split {name} is not sorted"
    allids = [i for ids in sp.values() for i in ids]
    assert len(set(allids)) == len(allids), "splits overlap"
    assert sorted(allids) == list(range(d["n_total"])), "splits do not cover range(n_total)"
    assert sha256_list(sp) == d["sha256"], "splits.json sha256 mismatch (file was modified)"
    return d


def split_ids(splits: dict, name: str) -> np.ndarray:
    return np.asarray(splits["splits"][name], dtype=np.int32)


# --------------------------------------------------------------------------- image IO
def load_image_01(path: str, size: int = 512, resize: Literal["error", "official"] = "error") -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    h, w = x.shape[1:]
    if (h, w) != (size, size):
        if resize == "error":
            raise ValueError(f"{path}: size {(h, w)} != {(size, size)}")
        if h != w:
            raise ValueError(f"{path}: official resize needs a square image, got {(h, w)}")
        x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=True)[0]
    return x


def list_images(dir: str) -> list[str]:
    return sorted(n for n in os.listdir(dir) if n.lower().endswith(_IMG_EXT))


def norm_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    for suf in ("_LQ", "_lq", "_bicubic", "_deg"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


class ImageDataset(Dataset):
    """source "ffhq": `ids` are ints, HQ and LQ from paths. Otherwise `source` is an LQ directory,
    `ids` are file names and hq01 is None (do not use the default collate in that case)."""

    def __init__(self, paths: Paths, ids, source: str = "ffhq", resize: str = "error"):
        self.paths, self.ids, self.source, self.resize = paths, list(ids), source, resize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        if self.source == "ffhq":
            return {"id": int(i), "hq01": load_image_01(ffhq_hq_path(self.paths, i)),
                    "lq01": load_image_01(ffhq_lq_path(self.paths, i))}
        return {"id": i, "hq01": None,
                "lq01": load_image_01(os.path.join(self.source, i), resize=self.resize)}


# --------------------------------------------------------------------------- cache
class CacheMismatchError(RuntimeError):
    pass


def _manifest_path(paths: Paths) -> str:
    return f"{paths.cache_dir}/manifest.json"


def _weights_key(weights: dict) -> dict:
    return {k: (v.get("path"), v.get("size")) for k, v in weights.items()}


def _check_value(item: str, rows: np.ndarray, values: np.ndarray) -> None:
    tail, dtype = ITEM_SPECS[item]
    if values.dtype != np.dtype(dtype):
        raise TypeError(f"{item}: dtype {values.dtype}, expected {dtype}")
    if values.shape != (len(rows), *tail):
        raise ValueError(f"{item}: shape {values.shape}, expected {(len(rows), *tail)}")
    if item in _INDEX_ITEMS:
        lo = -1 if item == "lq_prompt77" else 0
        if values.size and (values.min() < lo or values.max() > 1023):
            raise ValueError(f"{item}: values outside [{lo}, 1023]")


class CacheWriter:
    def __init__(self, paths: Paths, split: str, ids: np.ndarray, items: list[str], settings: dict,
                 weights: dict, splits_sha: str):
        self.paths, self.split, self.items = paths, split, list(items)
        self.ids = np.asarray(ids, dtype=np.int32)
        self.dir = assert_writable(f"{paths.cache_dir}/{split}", paths)
        os.makedirs(self.dir, exist_ok=True)
        self._manifest = self._open_manifest(settings, weights, splits_sha)

        ids_path = f"{self.dir}/ids.npy"
        if os.path.exists(ids_path):
            if not np.array_equal(np.load(ids_path), self.ids):
                raise CacheMismatchError(f"{split}: ids differ from the existing cache")
        else:
            np.save(ids_path, self.ids)

        n = len(self.ids)
        self._data, self._done = {}, {}
        for item in self.items:
            tail, dtype = ITEM_SPECS[item]
            f, fd = f"{self.dir}/{item}.npy", f"{self.dir}/{item}.done.npy"
            if os.path.exists(f) and os.path.exists(fd):
                self._data[item] = np.lib.format.open_memmap(f, mode="r+")
                self._done[item] = np.lib.format.open_memmap(fd, mode="r+")
                if self._data[item].shape != (n, *tail) or self._data[item].dtype != np.dtype(dtype) \
                        or self._done[item].shape != (n,):
                    raise CacheMismatchError(f"{split}/{item}: existing array has a different shape or dtype")
            else:
                self._data[item] = np.lib.format.open_memmap(f, mode="w+", dtype=dtype, shape=(n, *tail))
                self._done[item] = np.lib.format.open_memmap(fd, mode="w+", dtype=bool, shape=(n,))
                self._done[item][:] = False
                self._done[item].flush()

    def _open_manifest(self, settings: dict, weights: dict, splits_sha: str) -> dict:
        mp = assert_writable(_manifest_path(self.paths), self.paths)
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.load(f)
            if m["settings"] != json.loads(json.dumps(settings)):
                raise CacheMismatchError(f"cache settings differ.\nexisting: {m['settings']}\nnew: {settings}")
            if m["splits"]["sha256"] != splits_sha:
                raise CacheMismatchError("cache was built from different splits")
            if _weights_key(m["weights"]) != _weights_key(weights):
                raise CacheMismatchError("cache was built with different weight files")
            m["weights"], m["env"] = weights, env_info()
        else:
            m = {"version": 1, "created": _now(), "updated": _now(),
                 "splits": {"path": self.paths.splits_json, "sha256": splits_sha},
                 "weights": weights, "settings": settings, "env": env_info(),
                 "items": {}, "verification": None}
        return m

    def pending(self, item: str) -> np.ndarray:
        return np.flatnonzero(~np.asarray(self._done[item]))

    def write(self, item: str, rows: np.ndarray, values: np.ndarray) -> None:
        rows = np.asarray(rows)
        _check_value(item, rows, values)
        self._data[item][rows] = values
        self._data[item].flush()
        self._done[item][rows] = True
        self._done[item].flush()

    def finalize(self) -> None:
        arrays = {}
        for item in self.items:
            arrays[item] = {"file": f"{self.split}/{item}.npy", "shape": list(self._data[item].shape),
                            "dtype": str(self._data[item].dtype), "n_done": int(np.sum(self._done[item]))}
        m = self._manifest
        m["items"].setdefault(self.split, {"arrays": {}})
        m["items"][self.split].update(n=len(self.ids), ids_sha256=sha256_list(self.ids.tolist()))
        m["items"][self.split]["arrays"].update(arrays)
        m["updated"] = _now()
        atomic_write_json(_manifest_path(self.paths), m, self.paths)


class CacheReader:
    def __init__(self, paths: Paths, split: str, items: list[str] | None = None, require_complete: bool = True):
        with open(_manifest_path(paths)) as f:
            self.manifest = json.load(f)
        if split not in self.manifest["items"]:
            raise KeyError(f"split {split!r} not in cache manifest")
        entry = self.manifest["items"][split]
        self.dir = f"{paths.cache_dir}/{split}"
        self.items = list(items) if items is not None else list(entry["arrays"])
        self.settings = self.manifest["settings"]
        self.ids = np.load(f"{self.dir}/ids.npy")
        self._row = {int(i): r for r, i in enumerate(self.ids)}
        self._arr = {}
        for item in self.items:
            if item not in entry["arrays"]:
                raise KeyError(f"{split}/{item} not in cache manifest")
            if require_complete:
                done = np.load(f"{self.dir}/{item}.done.npy")
                if not done.all():
                    raise CacheMismatchError(f"{split}/{item} is incomplete: {int(done.sum())}/{len(done)} rows")
            self._arr[item] = np.load(f"{self.dir}/{item}.npy", mmap_mode="r")

    def array(self, item: str) -> np.memmap:
        return self._arr[item]

    def row_of(self, id) -> int:
        return self._row[int(id)]


# --------------------------------------------------------------------------- synthetic data (tests, dry runs)
def write_synthetic_ffhq(root: str, n: int = 16, seed: int = 0) -> tuple[str, str]:
    rng = np.random.default_rng(seed)
    hq_dir, lq_dir = os.path.join(root, "hq"), os.path.join(root, "lq")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)
    for i in range(n):
        low = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        hq = Image.fromarray(low).resize((512, 512), Image.BICUBIC)
        hq.save(f"{hq_dir}/{i:05d}.png")
        lq = hq.filter(ImageFilter.GaussianBlur(3))
        noisy = np.asarray(lq, dtype=np.float32) + rng.normal(0, 5, (512, 512, 3))
        buf = io.BytesIO()
        Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=30)
        Image.open(buf).convert("RGB").save(f"{lq_dir}/{i:05d}_LQ.png")
    return hq_dir, lq_dir


def write_synthetic_cache(paths: Paths, splits: dict, seed: int = 0) -> None:
    """Random but valid cache. z_lq is a noisy lookup of a fixed random table at hq_idx, so it is learnable."""
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(1024, 512)).astype(np.float32)
    for split, items in SPLIT_ITEMS.items():
        ids = split_ids(splits, split)
        n = len(ids)
        w = CacheWriter(paths, split, ids, items, {"synthetic": True}, {}, splits["sha256"])
        hq = rng.integers(0, 1024, (n, 256))
        vals = {
            "hq_idx": hq.astype(np.int16),
            "lq_idx": rng.integers(0, 1024, (n, 256)).astype(np.int16),
            "z_lq": (table[hq].transpose(0, 2, 1).reshape(n, 512, 16, 16)
                     + 0.5 * rng.normal(size=(n, 512, 16, 16))).astype(np.float16),
            "lat_lq": rng.normal(size=(n, 4, 64, 64)).astype(np.float16),
            "lat_teacher": rng.normal(size=(n, 4, 64, 64)).astype(np.float16),
        }
        p77 = rng.integers(0, 1024, (n, 77))
        p77[:, 60:] = -1
        vals["lq_prompt77"] = p77.astype(np.int16)
        for item in items:
            w.write(item, np.arange(n), vals[item])
        w.finalize()


# --------------------------------------------------------------------------- datasets
class FinetuneDataset(Dataset):
    def __init__(self, paths: Paths, split: str, prompt: Literal["lq", "hq_true", "hq_pred", "null", "official"],
                 hq_pred_file: str | None = None):
        if split not in ("ft_train", "val"):
            raise ValueError(f"FinetuneDataset needs ft_train or val, got {split!r}")
        self.paths, self.prompt = paths, prompt
        src = {"lq": "lq_idx", "hq_true": "hq_idx", "official": "lq_prompt77"}.get(prompt)
        self.reader = CacheReader(paths, split, ["lat_lq", "lat_teacher"] + ([src] if src else []))
        self.ids = self.reader.ids
        self.idx = None
        if src:
            self.idx = self.reader.array(src)
        elif prompt == "hq_pred":
            if hq_pred_file is None:
                raise ValueError("hq_pred needs hq_pred_file")
            with open(hq_pred_file.replace(".npy", ".ids.json")) as f:
                pred_ids = np.asarray(json.load(f), dtype=np.int32)
            if not np.array_equal(pred_ids, self.ids):
                raise ValueError(f"{hq_pred_file}: ids differ from cache split {split}")
            self.idx = np.load(hq_pred_file, mmap_mode="r")
            assert self.idx.shape == (len(self.ids), 256), self.idx.shape
        elif prompt != "null":
            raise ValueError(f"unknown prompt {prompt!r}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = int(self.ids[k])
        item = {"id": torch.tensor(i, dtype=torch.int64),
                "hq_pm1": load_image_01(ffhq_hq_path(self.paths, i)) * 2 - 1,
                "lat_lq": torch.from_numpy(np.array(self.reader.array("lat_lq")[k], dtype=np.float32)),
                "lat_teacher": torch.from_numpy(np.array(self.reader.array("lat_teacher")[k], dtype=np.float32))}
        if self.idx is not None:
            item["idx"] = torch.from_numpy(np.array(self.idx[k], dtype=np.int64))
        return item


class PredictorDataset(Dataset):
    def __init__(self, paths: Paths, split: str):
        self.reader = CacheReader(paths, split, ["hq_idx", "z_lq"])
        self.ids = self.reader.ids
        self.z, self.y = self.reader.array("z_lq"), self.reader.array("hq_idx")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        return {"z": torch.from_numpy(np.array(self.z[k], dtype=np.float32)),
                "y": torch.from_numpy(np.array(self.y[k], dtype=np.int64))}


def step_order(n_items: int, n_steps: int, batch_size: int, seed: int) -> np.ndarray:
    """(n_steps, batch_size) int64 item indices; epochs are seeded permutations, so resume is exact."""
    need = n_steps * batch_size
    chunks, total, epoch = [], 0, 0
    while total < need:
        g = torch.Generator().manual_seed(seed + epoch)
        chunks.append(torch.randperm(n_items, generator=g).numpy())
        total += n_items
        epoch += 1
    return np.concatenate(chunks)[:need].reshape(n_steps, batch_size).astype(np.int64)
