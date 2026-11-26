import json

import numpy as np
import pytest

from hqprompt import data, infer_official as io_


def _run(hqp_root, variant, *extra):
    return io_.main(["--dry_run", "--split", "val", "--prompt_variant", variant, "--out_root", hqp_root.out_root, *extra])


def _out(hqp_root, variant):
    return f"{hqp_root.outputs_dir}/official/ffhq_val_{variant}"


def test_derangement():
    for n in (2, 3, 7, 50):
        src = io_.derangement(n, seed=1)
        assert sorted(src) == list(range(n)) and (src != np.arange(n)).all()
    with pytest.raises(AssertionError):
        io_.derangement(1, 0)


def test_all_variants_dry_run(hqp_root):
    outs = {}
    for v in io_.VARIANTS:
        assert _run(hqp_root, v) == 0
        d = _out(hqp_root, v)
        meta = json.load(open(f"{d}/meta.json"))
        assert meta["n_files"] == 2 and "SELF-GENERATED" in meta["lq_set_note"]
        assert meta["merge_report"]["n_pairs"] == 4
        outs[v] = (np.load(f"{d}/prompt_idx.npy"), meta)
        import os
        assert sorted(os.listdir(f"{d}/images")) == [f"{i:05d}.png" for i in
                                                     data.load_splits(hqp_root.splits_json)["splits"]["val"]]
    normal = outs["normal"][0]
    assert normal.shape == (2, 77) and normal.dtype == np.int16 and (normal >= -1).all()
    shuffled, meta = outs["shuffled"]
    src = np.array(meta["shuffle_src"])
    assert (src != np.arange(2)).all() and np.array_equal(shuffled, normal[src])
    assert not any(np.array_equal(shuffled[i], normal[i]) for i in range(2))  # nobody keeps its own codes
    assert (outs["zero_code"][0] == -1).all() and (outs["zero_embed"][0] == -1).all()


def test_input_dir_batched_matches_unbatched(hqp_root, tmp_path):
    _, lq = data.write_synthetic_ffhq(str(tmp_path / "imgs"), n=3)
    from PIL import Image
    common = ["--dry_run", "--input_dir", lq, "--vae_mode", "mean", "--out_root", hqp_root.out_root]
    assert io_.main([*common, "--dataset_name", "b1", "--batch_size", "1"]) == 0
    assert io_.main([*common, "--dataset_name", "b3", "--batch_size", "3"]) == 0
    for stem in ("00000_LQ", "00001_LQ", "00002_LQ"):
        a = np.asarray(Image.open(f"{hqp_root.outputs_dir}/official/b1_normal/images/{stem}.png"), dtype=int)
        b = np.asarray(Image.open(f"{hqp_root.outputs_dir}/official/b3_normal/images/{stem}.png"), dtype=int)
        assert np.abs(a - b).max() <= 2  # batched conv kernels may differ in the last bit before uint8 rounding


def test_refuses_non_empty_output(hqp_root):
    assert _run(hqp_root, "normal") == 0
    with pytest.raises(FileExistsError):
        _run(hqp_root, "normal")
    assert _run(hqp_root, "normal", "--overwrite") == 0


def test_per_image_seed_is_reproducible(hqp_root):
    from PIL import Image
    assert _run(hqp_root, "normal", "--limit", "1") == 0
    p = f"{_out(hqp_root, 'normal')}/images/"
    import os
    f = os.listdir(p)[0]
    first = np.asarray(Image.open(p + f))
    assert _run(hqp_root, "normal", "--limit", "1", "--overwrite") == 0
    assert np.array_equal(first, np.asarray(Image.open(p + f)))
