# Coordinator amendments (override anything below)

1. **No git operations of any kind** (no branch, commit, stash, checkout, push). Only create/edit files in the working tree.
2. **Already written by the coordinator, frozen — import, do not edit:** `hqprompt/__init__.py`, `hqprompt/paths.py`, `hqprompt/common.py`, `tests/conftest.py`, `tests/test_paths.py`. Interfaces match sections 1 / common.py below.
3. **Layout:** `jobs/`, `tests/`, `docs/`, `requirements.txt` are at the repo root (`CleanCue/`), next to the `hqprompt/` package. `jobs/setup_env.sh` and `jobs/env.sh` already exist as a first port; task D extends them.
4. **Local test env:** a Python 3.10 env with the SCC pins (torch 2.4.0 CPU, diffusers 0.27.2, transformers 4.37.2, peft 0.12.0, accelerate 0.31.0, huggingface_hub 0.24.7, pytest, sklearn) lives at
   `LOCALENV=/private/tmp/claude-501/-Users-manaswiyadamreddy-Documents-CleanCue/291032e1-e840-4476-ae2b-9986af5f0dc1/scratchpad/localenv`.
   Run tests from the repo root with `$LOCALENV/bin/python -m pytest -q tests/...`. So diffusers/peft tests must NOT be skipped; only pyiqa/facexlib/face_alignment tests use `importorskip`. If LOCALENV/bin/python does not exist yet, it is still installing: write code first and run tests at the end (retry; fall back to `python3` for torch-only tests).
5. Never write outside the repo or the pytest tmp dir. Never touch `/Users/manaswiyadamreddy/Documents/CRAFT` or the official OSDFace folder except to read.
6. `xformers` is not used; leave it out of `setup_env.sh`.

---

# HQ-prompt (Proposal 1): implementation spec for parallel coders

This spec is built on `/Users/manaswiyadamreddy/Documents/CleanCue/plan.md` plus the code I read. The coordinator sent updates mid-task and they override plan.md where the two conflict:
- **No git operations at all.** No branches, commits or git hashes in metadata. Code is edited in place in `/Users/manaswiyadamreddy/Documents/CleanCue/`.
- **Python env:** a conda prefix env at `/projectnb/cs599bg/manaswi/CleanCue/env`, built by `jobs/setup_env.sh`, which is a port of CRAFT `setup2.sh`.
- **How jobs run:** every step must work as plain `bash jobs/<step>.sh` inside a GPU session the user already allocated (qrsh or OnDemand). qsub headers are optional and never name a GPU type.
- **Data paths on SCC** are now verified (section 1).

Labels: **[V]** I verified it by reading code. **[U]** unverified, must be checked on SCC. **[D]** a design decision made here (the user can override it).

---

## 0. Global rules every coder must follow

1. **Python and imports.** Target Python 3.10. Use only absolute package imports (`from hqprompt import vre`). Every script runs as `python -m hqprompt.<module>`.
2. **Write guard.** Every file write goes through `paths.assert_writable(path)`. It raises unless the real path is under `Paths.out_root`. Tests set the out root to a tmp dir.
3. **Checkpoint loading.**
   - `.ckpt`/`.pth` files are loaded only through `paths.safe_torch_load`, which allowlists exactly `associate_2.ckpt` and `embedding_change_weights.pth`.
   - Our own checkpoints are saved as safetensors plus JSON. Optimizer and scheduler state is loaded with `torch.load(weights_only=True)`.
4. **Failures.**
   - Failed checks raise, or exit with a nonzero code, after writing their report.
   - Never wrap a metric or model load in `try/except` that turns it into NaN. This is plan rule 8; the old `Evaluation/metrics/iqa.py` breaks it.
5. **Precision.** Code lookup, prompt projection and all trainable parameters stay fp32. bf16 autocast wraps only the UNet forward and the VAE decode/encode. `x0` is computed in float64, as the official code does [V `OSDFace/infer.py`; plan says fp32, and float64 is the stricter choice].
6. **Local testing.** The local machine has torch 2.12 CPU, numpy, pytest and sklearn. It has no diffusers, peft, transformers, pyiqa, facexlib or face_alignment [V].
   - Tests that need those packages start with `pytest.importorskip(...)`.
   - Core tests (paths, vre, prompt, data, cache, predictor, bootstrap/gate) must run with torch and numpy only.
7. **Patch order.** Raster order everywhere: patch `p = row*16 + col`, matching `permute(0,2,3,1).view(B,-1,C)` [V vqvae.py].
8. **Dry runs.** Every CLI script has `--dry_run`. It uses tiny random models and synthetic data under `--out_root` (a tmp dir), runs on CPU, and finishes in under about 2 minutes.

---

## 1. Package layout and `hqprompt/paths.py`

```
CleanCue/
  hqprompt/
    __init__.py          (A)  empty
    paths.py             (A)
    common.py            (A)  seeding, numerics, json/hash helpers
    vre.py               (A)
    prompt.py            (A)
    osdface.py           (B)
    losses.py            (B)
    data.py              (C)
    train_predictor.py   (C)
    metrics/__init__.py, iqa.py, identity.py, landmark.py, fid.py   (D)
    evaluate.py          (D)
    check_500.py         (E)
    infer_official.py    (E)
    build_cache.py       (E)
    finetune.py          (F)
    infer.py             (F)  inference for fine-tuned runs (rows 2-5 and row 4 with predicted codes)
  jobs/                  (D)  setup_env.sh, env.sh, one script per step
  tests/                 conftest.py (A); test_<module>.py owned by that module's task
  requirements.txt       (D)
```

`structure_loss.py` is out of scope. `finetune.py` keeps a hook for it (section 5).

### `paths.py`: frozen interface

```python
SCC_BASE = "/projectnb/cs599bg/manaswi/craft"

@dataclass(frozen=True)
class Paths:
    out_root: str        # env HQP_ROOT        default f"{SCC_BASE}/hqprompt"
    pretrained: str      # env HQP_PRETRAINED  default f"{SCC_BASE}/ivc_checkpoints/osdface_official/pretrained"
    ffhq_hq: str         # env HQP_FFHQ_HQ     default f"{SCC_BASE}/data/train/images512x512"      # {id:05d}.png, 70000 [V by coordinator]
    ffhq_lq: str         # env HQP_FFHQ_LQ     default f"{SCC_BASE}/data/train/LQ_images_512x512"  # {id:05d}_LQ.png
    celeba_hq: str       # env HQP_CELEBA_HQ   default f"{SCC_BASE}/data/test/CelebA/CelebA_Validation/celeba_512_validation"     # {00000000..00002999}.png
    celeba_lq: str       # env HQP_CELEBA_LQ   default f"{SCC_BASE}/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2" # SELF-GENERATED, not the official DAEFR set; same filenames
    lfw_lq: str          # env HQP_LFW_LQ      default f"{SCC_BASE}/data/test/LFW/cropped_faces"   # 1711 *.png, no GT
    model_cache: str     # env HQP_MODEL_CACHE default f"{out_root}/model_cache"
    # derived properties (not settable):
    associate_ckpt -> f"{pretrained}/associate_2.ckpt"
    lora           -> f"{pretrained}/pytorch_lora_weights.safetensors"
    embedding_change -> f"{pretrained}/embedding_change_weights.pth"
    sd21           -> f"{pretrained}/sd21"   # unet/, vae/, scheduler/
    splits_json    -> f"{out_root}/splits.json"
    cache_dir      -> f"{out_root}/cache"
    runs_dir       -> f"{out_root}/runs"
    predictor_dir  -> f"{out_root}/predictor"
    outputs_dir    -> f"{out_root}/outputs"
    checks_dir     -> f"{out_root}/checks"
    eval_dir       -> f"{out_root}/eval"
    logs_dir       -> f"{out_root}/logs"
    facexlib_dir   -> f"{model_cache}/facexlib"

def get_paths(**overrides) -> Paths            # precedence: overrides (non-None) > env var > default
def add_path_args(parser) -> None              # adds --out_root --pretrained --ffhq_hq --ffhq_lq --celeba_hq --celeba_lq --lfw_lq --model_cache (all default None)
def paths_from_args(args) -> Paths
def assert_writable(path: str, paths: Paths) -> str   # realpath must be under realpath(paths.out_root); returns realpath; raises PermissionError
def safe_torch_load(path: str, paths: Paths, map_location="cpu"):
    # realpath must equal associate_ckpt or embedding_change, else PermissionError.
    # Try weights_only=True; on failure, log WARNING and retry weights_only=False (allowlisted files only).
def file_fingerprint(path) -> {"path": str, "size": int, "mtime": float}
def weights_manifest(paths) -> dict[str, dict]   # fingerprints of associate, lora, embedding_change, sd21/unet/*.safetensors, sd21/vae/*.safetensors, sd21/scheduler/scheduler_config.json
```

### `common.py` (A)

```python
def seed_everything(seed: int) -> None
def set_numerics(deterministic: bool = True, tf32: bool = False) -> None
    # sets cudnn.benchmark = not deterministic, cudnn.deterministic, cuda.matmul.allow_tf32, cudnn.allow_tf32
def get_device(dry_run: bool) -> torch.device   # dry_run -> cpu; otherwise cuda, failing if unavailable
def atomic_write_json(path, obj, paths) -> None # tmp file + os.replace, via assert_writable
def sha256_bytes(b) -> str
def sha256_list(xs) -> str                      # sha256 of json.dumps(list)
def code_fingerprint() -> dict                  # sha256 of every hqprompt/*.py (replaces git hash)
def env_info() -> dict                          # python, torch, cuda, diffusers/peft/transformers versions if importable, GPU name
```

`tests/conftest.py` (A) provides one fixture, `hqp_root(tmp_path, monkeypatch)`. It sets `HQP_ROOT=tmp_path` and returns `get_paths()`. No other shared fixtures.

---

## 2. Modules

### 2.1 `vre.py` (A): HQ and LQ VQ branches, fp32 code lookup

**What to port.**
- From CRAFT `vqvae.py`: `nonlinearity`, `Normalize`, `Upsample`, `Downsample`, `ResnetBlock`, `MultiHeadAttnBlock`, `MultiHeadEncoder`, `MultiHeadDecoder`. Keep the parameter and attribute names byte-identical so checkpoint keys load strictly.
- Required changes during the port:
  - `MultiHeadAttnBlock.forward`: drop `copy.deepcopy(w_)` (output is unchanged) but keep returning `(x + w_, attn)`.
  - **Bug fix [V]:** `MultiHeadDecoder.forward` does `h = self.mid.attn_1(h)` and `h = self.up[i].attn[j](h)`, but the attention block returns a tuple. The decoder as written crashes. The port must unpack: `h, _ = ...`.
  - `MultiHeadDecoder.__init__`: remove its `print`.
- Do not port `VectorQuantizer.inference`'s dtype behaviour. It uses `.to(z)` on indices, which is trap 1. Reimplement the selection logic below in fp32.

**Interface:**

```python
@dataclass(frozen=True)
class VREConfig:
    n_embed: int = 1024; embed_dim: int = 512; ch: int = 64; ch_mult: tuple = (1,2,2,2,4,8)
    num_res_blocks: int = 2; attn_resolutions: tuple = (16,); z_channels: int = 512
    resolution: int = 512; head_size: int = 1; dropout: float = 0.0

BRANCH_PREFIX = {"hq": "vqvae.", "lq": "vqvae_LQ."}   # "hq" = vqvae.* is an ASSUMPTION until C1 passes
LQ_INPUT_MODE = "double"                              # official path [V infer.py + lq_embed/inference.py]

class Codebook(nn.Module):          # holds .embedding = nn.Embedding(1024, 512) (key: quantize.embedding.weight)
class VQBranch(nn.Module):
    # attributes: encoder (MultiHeadEncoder, double_z=False), quant_conv Conv2d(512,512,1),
    #             quantize (Codebook), and if with_decoder: post_quant_conv Conv2d(512,512,1), decoder (MultiHeadDecoder)
    def __init__(self, cfg: VREConfig, with_decoder: bool)
    def encode_features(self, x_enc: Tensor) -> Tensor
        # x_enc (B,3,512,512) fp32, already normalized by prepare_input.
        # Returns (B,512,16,16) fp32 = quant_conv(encoder(x)['out']). Runs under no_grad with autocast disabled.
    @property
    def codebook(self) -> Tensor    # (1024,512) fp32 view of quantize.embedding.weight
    def decode_indices(self, idx: LongTensor) -> Tensor
        # idx (B,256) -> rows -> (B,512,16,16) -> post_quant_conv -> decoder -> (B,3,512,512) raw fp32

def prepare_input(img01: Tensor, mode: Literal["single","double"]) -> Tensor
    # img01 must be fp32 (B,3,512,512) within [-1e-6, 1+1e-6] (assert).
    # "single": img01*2-1, range [-1,1].
    # "double": (img01*2-1 - 0.5)/0.5 = 4*img01 - 3, range [-3,1]. This is the official LQ path (trap 6).

@contextmanager
def fp32_lookup():   # torch.autocast(device_type, enabled=False) for cpu and cuda; temporarily sets cuda.matmul.allow_tf32=False

def nearest_codes(z: Tensor, codebook: Tensor) -> tuple[LongTensor, Tensor]
    # z (B,512,16,16) fp32, codebook (1024,512) fp32. Assert both fp32 (raise TypeError otherwise).
    # Inside fp32_lookup(): zf = z.permute(0,2,3,1).reshape(B,256,512)
    #   d = zf.pow(2).sum(-1,keepdim=True) + codebook.pow(2).sum(-1) - 2*zf@codebook.T   # official formula, raw L2
    #   dmin, idx = d.min(dim=-1)
    # Returns idx (B,256) int64 in [0,1023] and dmin (B,256) fp32.

def official_prompt_indices(idx: LongTensor, dmin: Tensor, n_min: int = 77) -> LongTensor
    # Per sample i (port of CRAFT VectorQuantizer.inference, per-sample fix of the official batch bug):
    #   order = torch.sort(dmin[i], dim=0).indices        (default stable=False, same as official)
    #   walk idx[i][order], keep first occurrences until n_min unique.
    # Returns (B,77) int64, padded with -1 when fewer than 77 are unique.

def codes_to_embeddings(idx: LongTensor, codebook: Tensor) -> Tensor
    # idx (B,N) int64 with -1 allowed. Returns (B,N,512) fp32: codebook[idx] where idx >= 0, exact zeros where idx == -1.

def load_codebook(ckpt_path, branch: str, paths) -> Tensor   # (1024,512) fp32 only (no model build)

def load_branch(ckpt_path, branch: Literal["hq","lq"], paths, with_decoder=False, cfg=VREConfig(), device="cpu") -> VQBranch
    # sd = safe_torch_load(...)["state_dict"]; keep keys k.startswith(prefix), strip prefix with k[len(prefix):] (not str.replace).
    # If not with_decoder, drop keys starting with "decoder." or "post_quant_conv.".
    # load_state_dict(strict=True): any missing or unexpected key is an error that lists the keys. Do not loosen.
    # If with_decoder: assert the branch's tensor count equals len(module.state_dict()), and log both (plan says 405).
    # Return eval(), requires_grad_(False), fp32; assert every param is fp32.

class VRE(nn.Module):
    def __init__(self, hq: VQBranch | None, lq: VQBranch | None)
    def lq_features(self, img01) -> Tensor            # (B,512,16,16) fp32, LQ branch, LQ_INPUT_MODE
    def lq_codes(self, img01) -> tuple[LongTensor, Tensor]       # nearest LQ code per patch (B,256), plus dmin
    def lq_official_prompt(self, img01) -> LongTensor            # (B,77) with -1 padding
    def hq_codes(self, img01, input_mode: str) -> LongTensor     # (B,256). input_mode has NO default; it comes from C1.

def load_vre(paths, branches=("hq","lq"), with_decoder=False, device="cpu") -> VRE
def random_vre(seed=0, ch=32, with_decoder=False, near_zero_rows=400) -> VRE
    # Tests only. Random weights; codebooks scaled so `near_zero_rows` rows have norm < 0.1.
    # ch=32 because GroupNorm uses 32 groups.
```

**Tests (`tests/test_vre.py`), must pass locally:**
- `prepare_input` ranges: `single` gives [-1,1]; `double` maps 0 to -3 and 1 to 1. It asserts on out-of-range input.
- `nearest_codes` rejects bf16 and fp16 input. Called inside `torch.autocast("cpu", dtype=torch.bfloat16)`, it returns results identical to the call outside autocast.
- `nearest_codes` matches a brute-force `torch.cdist` argmin on random data, except at exact ties.
- `official_prompt_indices`:
  - On a crafted example (few unique codes), the result is sorted by distance, has unique codes and is padded with -1.
  - With all codes equal, it gives one code and 76 pads.
- `codes_to_embeddings` produces exact zero rows for -1.
- **Parity with CRAFT** (skipped if `HQP_CRAFT_REPO`, default `/Users/manaswiyadamreddy/Documents/CRAFT`, is missing): import `vqvae.py` by file path, read-only.
  - Build `VQVAEencoder(args=SimpleNamespace(cat_prompt_embedding=False, use_att_pool=False, use_pos_embedding=False, learnable_pos_emb=False), ch=32)`.
  - Copy its random state into our `VQBranch(with_decoder=False)` with `strict=True`.
  - Run batch 2. Features must be equal (`allclose` atol 1e-5). `codes_to_embeddings(official_prompt_indices(...))` must equal `quantize.inference(h)` exactly.
- The decoder runs: with `with_decoder=True` and a random VRE, `decode_indices` returns (B,3,512,512).
- `load_branch` on a synthetic ckpt (saved with `torch.save` to tmp, with the allowlist overridden via `get_paths(pretrained=tmp)`):
  - Loads the `vqvae_LQ.` keys strictly.
  - An extra unexpected key raises.
  - A missing key raises.

### 2.2 `prompt.py` (A): the single prompt-construction contract

This is the only place where prompts are built. `finetune`, `infer`, `infer_official`, `build_cache` (teacher) and `check_500` all import it.

```python
MODES = ("official", "lq", "hq_true", "hq_pred", "null")
CODEBOOK_FOR_MODE = {"official": "lq", "lq": "lq", "hq_true": "hq", "hq_pred": "hq", "null": None}
N_TOKENS = {"official": 77, "lq": 256, "hq_true": 256, "hq_pred": 256, "null": 256}

class TokenProjection(nn.Module):
    # TwoLayerConv1x1 without the 77 assert (trap 7).
    # Params: conv1 = Conv1d(512,256,1), conv2 = Conv1d(256,1024,1), same names, so embedding_change loads strictly.
    def forward(self, x):   # (B,N,512) -> permute -> conv1 -> SiLU -> conv2 -> SiLU -> permute -> (B,N,1024)

def load_embedding_change(paths) -> dict[str, Tensor]
    # safe_torch_load. Assert keys == {conv1.weight (256,512,1), conv1.bias (256,), conv2.weight (1024,256,1), conv2.bias (1024,)}.
    # Cast to fp32.
def random_embedding_change_sd(seed=0) -> dict   # tests

class PromptEncoder(nn.Module):
    def __init__(self, mode: str, codebook: Tensor | None, embedding_change_sd: dict | None,
                 trainable: bool = True, null_init: Literal["pad","zeros"] = "pad")
        # official : self.codebook buffer (persistent=False); self.proj TokenProjection loaded strictly from embedding_change, FROZEN
        #            (requires_grad False regardless of `trainable`); no positions.
        # lq/hq_*  : codebook buffer (persistent=False, frozen); proj loaded from embedding_change (trainable if trainable);
        #            self.pos = nn.Parameter(torch.zeros(256,1024)).
        # null     : self.tokens = nn.Parameter((256,1024)).
        #            null_init="pad" [D]: tokens = proj(zeros(1,256,512))[0], i.e. exactly the official prompt's padding
        #            token, using embedding_change. "zeros" -> zeros. No codebook, no proj.
        # All parameters and buffers are fp32 (assert).
    def forward(self, idx: LongTensor | None = None, batch_size: int | None = None) -> Tensor
        # Runs with autocast disabled.
        # official: idx (B,77) int64, values in [-1,1023]           -> proj(codes_to_embeddings(idx))           -> (B,77,1024) fp32
        # lq/hq_* : idx (B,256) int64, values in [0,1023] (assert)  -> proj(codebook[idx]) + pos[None]          -> (B,256,1024) fp32
        # null    : idx must be None; returns tokens[None].expand(batch_size,-1,-1)                             -> (B,256,1024) fp32
    def trainable_state_dict(self) -> dict[str, Tensor]   # names: "proj.conv1.weight", ..., "pos" | "tokens"; only requires_grad params
    def load_trainable_state_dict(self, sd, strict=True) -> None
    @property
    def codebook_kind(self) -> str | None
```

**Codebook per mode:** official and lq use the LQ codebook (`vqvae_LQ.quantize`). hq_true and hq_pred use the HQ codebook (`vqvae.quantize`). null uses none.

Callers obtain the codebook with `vre.load_codebook(paths.associate_ckpt, CODEBOOK_FOR_MODE[mode], paths)`.

`embedding_change` was trained on LQ codebook rows only. Applying it to HQ rows is a distribution shift, and that is why it is trainable.

**Tests (`tests/test_prompt.py`):**
- Shapes and dtypes for every mode.
- **Parity:** `TokenProjection` equals CRAFT `lq_embed.TwoLayerConv1x1` on N=77 (skip if the CRAFT repo is missing). It also runs on N=256.
- The official mode with all -1 input gives `proj(zeros)` in every row.
- `pos` is zero at init, so at init the lq-mode output equals `proj(codebook[idx])`.
- Gradients reach `proj` and `pos` (lq) and `tokens` (null). The codebook gets no gradient. Official mode has no trainable params.
- An out-of-range idx raises. A non-int64 idx raises.
- `trainable_state_dict` round-trips.

### 2.3 `osdface.py` (B): merged UNet, VAE, one-step forward

**What to port.**
- CRAFT `train_concat.py:merge_lora_into_unet`, rewritten as a pure function with asserts. Handle both the `lora_A/lora_B` and the `lora.up/lora.down` formats.
- `get_x0_from_noise` (official `utils/others.py`), in float64.
- **Do NOT use `infer_concat.py: merge_unet`** (`torch.mm` fails on conv layers, trap 2).
- **Do NOT copy the CLIP empty-text concatenation** from `infer_concat.py` or `benchmark_original_osdface.py`. Official uses only the 77 visual tokens [V `OSDFace/infer.py`].

```python
T_FIXED = 399                     # trap 4
@dataclass
class OSDFaceModels:
    unet: "UNet2DConditionModel"; vae: "AutoencoderKL"
    alphas_cumprod: Tensor        # (1000,) float32 exactly as the scheduler gives; cast to float64 at use
    scaling_factor: float

def merge_lora_state_dict(unet_sd: dict, lora_sd: dict, scale: float) -> tuple[dict, dict]
    # For each lora_A key: base = key with ".lora_A.weight" -> ".weight", and a leading "unet." removed only as a prefix.
    # Assert lora_B and base exist. Cast A, B and W to fp32.
    # 4-D: assert A (r,in,kh,kw) and B (out,r,1,1); delta = (B.view(out,r) @ A.view(r,-1)).view_as(W).
    # 2-D: delta = B @ A.
    # Old "lora.up/down" format: handle 2-D only; anything else raises.
    # Raise if any LoRA key is left unprocessed, or if any key contains "alpha".
    # Report: {"n_pairs", "n_conv", "n_linear", "rank", "unprocessed": []}.
def load_merged_unet(paths, expected_pairs=278, expected_conv=64, scale=1.0) -> tuple[UNet, dict]
    # UNet2DConditionModel.from_pretrained(paths.sd21, subfolder="unet"), fp32; merge via safetensors safe_open.
    # Assert report["n_pairs"] == 278, n_conv == 64, rank == 16.
    # load_state_dict(strict=True); requires_grad_(False); eval().
def load_vae(paths) -> AutoencoderKL           # fp32, frozen, eval; assert config.scaling_factor == 0.18215
def load_alphas(paths) -> Tensor
    # DDIMScheduler.from_pretrained(sd21, subfolder="scheduler").alphas_cumprod
    # Assert config.prediction_type == "epsilon" and num_train_timesteps == 1000.
def load_osdface(paths, device, merge_lora=True) -> tuple[OSDFaceModels, dict]   # dict = merge report
def tiny_models(seed=0, device="cpu") -> OSDFaceModels
    # Deterministic.
    # UNet2DConditionModel(sample_size=64, in/out 4, block_out_channels=(32,64), layers_per_block=1,
    #   down=("CrossAttnDownBlock2D","DownBlock2D"), up=("UpBlock2D","CrossAttnUpBlock2D"),
    #   cross_attention_dim=1024, attention_head_dim=8, norm_num_groups=32).
    # AutoencoderKL(block_out_channels=(32,32,64,64), 4 Down/UpEncoderBlock2D, latent_channels=4, norm_num_groups=32)  -> factor 8.
    # alphas: scaled_linear betas, linspace(0.00085**.5, 0.012**.5, 1000, fp32)**2, cumprod in fp32.
    # scaling_factor 0.18215.
def tiny_lora_state_dict(unet, seed=0, rank=4) -> dict   # synthetic "unet.<...>.lora_A.weight"/lora_B for some linear and some conv weights

def vae_encode(vae, img_pm1, mode: Literal["mean","sample"], generator=None, amp_dtype=None) -> Tensor
    # Returns latent_dist.mean (or .sample(generator)) * scaling_factor, as fp32 (B,4,64,64).
def vae_decode(vae, lat_scaled, amp_dtype=None, clamp=True) -> Tensor   # vae.decode(lat/sf).sample -> fp32 (B,3,512,512) in [-1,1] if clamp
def get_x0_from_noise(z_t, eps, alphas_cumprod, t=T_FIXED) -> Tensor
    # float64: a = alphas_cumprod.double()[t]; x0 = (z_t.double() - (1-a).sqrt()*eps.double()) / a.sqrt(); returns .float()
def one_step(unet, z_lq, prompt_embeds, alphas_cumprod, t=T_FIXED, amp_dtype=None) -> Tensor
    # eps = unet(z_lq, t, encoder_hidden_states=prompt_embeds).sample   (under autocast(amp_dtype) if given; prompt passed fp32)
    # Returns x0 latent, fp32 (B,4,64,64).
```

**Tests (`tests/test_osdface.py`):**
- **Without diffusers:**
  - `merge_lora_state_dict` on hand-made dicts (2 linear, 2 conv 3x3) equals W + B@A.
  - A leftover key raises. A missing base key raises. A shape mismatch raises. An alpha key raises. The `unet.` prefix is stripped only at the start.
  - `get_x0_from_noise` against a numpy float64 reference.
- **With diffusers** (`importorskip`):
  - `tiny_models` is deterministic across two calls.
  - `one_step` returns the right shape.
  - Merge on `tiny_models` plus `tiny_lora_state_dict` changes exactly the targeted weights.
  - `vae_encode` "mean" is deterministic.
  - The tiny alphas equal `DDIMScheduler(beta_schedule="scaled_linear", ...)`.

### 2.4 `losses.py` (B): EA-DISTS and ArcFace ID

**What to port.**
- CRAFT `losses/dists.py` (`Sobel`, `EADists`) verbatim.
- CRAFT `losses/arcface.py` (`ArcFaceID`) with these changes:
  - Pass `model_rootpath=paths.facexlib_dir` to `init_recognition_model` [U: signature in the installed facexlib].
  - Apply the same `Flatten` view-to-reshape patch as `Evaluation/metrics/identity.py`.
  - Build eagerly in `__init__(device)` rather than lazily, so download failures show at startup.

Both losses take `(pred01, target01)` (B,3,H,W) fp32 and return an fp32 scalar. Both run with autocast disabled (`.float()` inputs) and have frozen parameters.

```python
class EADists(nn.Module): forward(pred01, target01) -> Tensor   # DISTS(x,y) + DISTS(sobel x, sobel y)
class ArcFaceID(nn.Module): __init__(paths, device); embed(img01) -> (B,512); forward(pred01, target01) -> mean(1 - cos)
class StubDists(nn.Module)   # tests/dry_run: L1 of 4x avg-pooled images + L1 of Sobel, differentiable
class StubID(nn.Module)      # tests/dry_run: fixed random conv net (seeded) -> GAP -> normalize; 1 - cos
def build_losses(paths, device, dry_run: bool) -> dict[str, nn.Module]   # keys "ea_dists", "id"
```

**Tests:**
- Stubs give gradients to the inputs, and loss(x, x) ≈ 0.
- `Sobel` output lies in [0,1].
- The real ones run under `importorskip("pyiqa")` and `importorskip("facexlib")`.

### 2.5 `data.py` (C): splits, image IO, cache format, datasets

```python
SPLIT_SIZES = {"pred_train": 59000, "ft_train": 10000, "val": 500, "spare": 500}
N_FFHQ = 70000
ITEM_SPECS = {   # tail shape, numpy dtype
  "hq_idx":      ((256,), "int16"),
  "lq_idx":      ((256,), "int16"),
  "lq_prompt77": ((77,),  "int16"),     # official 77-token LQ prompt indices, -1 padded (used by teacher, smoke check)
  "z_lq":        ((512,16,16), "float16"),
  "lat_lq":      ((4,64,64), "float16"),
  "lat_teacher": ((4,64,64), "float16"),
}
SPLIT_ITEMS = {
  "pred_train": ["hq_idx", "z_lq"],
  "ft_train":   ["hq_idx", "z_lq", "lq_idx", "lq_prompt77", "lat_lq", "lat_teacher"],
  "val":        ["hq_idx", "z_lq", "lq_idx", "lq_prompt77", "lat_lq", "lat_teacher"],
}

def make_splits(n_total=N_FFHQ, seed=0, sizes=SPLIT_SIZES) -> dict
    # perm = np.random.default_rng(seed).permutation(n_total)
    # Contiguous slices in the order pred_train, ft_train, val, spare; each slice sorted ascending.
    # Assert sum(sizes) == n_total.
def write_splits(paths, splits: dict, hq_dir, lq_dir, check_files=True) -> None
    # If check_files, assert every hq/lq file exists (and fail listing up to 20 missing).
    # Refuse to overwrite an existing splits.json with different content.
def load_splits(path) -> dict
    # Validate: disjoint, sizes match, union == range(n_total), sha matches.
def ffhq_hq_path(paths, i) -> str  # f"{ffhq_hq}/{i:05d}.png"
def ffhq_lq_path(paths, i) -> str  # f"{ffhq_lq}/{i:05d}_LQ.png"
def load_image_01(path, size=512, resize: Literal["error","official"]="error") -> Tensor
    # PIL RGB -> float32 (3,H,W)/255.
    # If H,W != size: "error" raises. "official" requires a square image, then F.interpolate bilinear align_corners=True (as in official infer).
def list_images(dir) -> list[str]   # sorted *.png/*.jpg/*.jpeg basenames
def norm_stem(name) -> str          # strip extension and the suffixes _LQ/_lq/_bicubic/_deg (port of Evaluation/evaluate.py:_norm_stem)

class ImageDataset(Dataset):   # (paths, ids|names, source: "ffhq"|dir) -> {"id": int|str, "hq01": (3,512,512)|None, "lq01": (3,512,512)}

class CacheMismatchError(RuntimeError): ...
class CacheWriter:
    def __init__(self, paths, split: str, ids: np.ndarray, items: list[str], settings: dict, weights: dict, splits_sha: str)
    def pending(self, item) -> np.ndarray     # row indices whose done flag is False, ascending
    def write(self, item, rows: np.ndarray, values: np.ndarray) -> None
        # Assert values.dtype == ITEM_SPECS dtype and values.shape == (len(rows), *tail).
        # Index items: assert 0 <= v <= 1023 (lq_prompt77: -1 <= v).
        # Order: write data, flush, set done[rows] = True, flush done.
    def finalize(self) -> None                # updates manifest n_done
class CacheReader:
    def __init__(self, paths, split, items: list[str] | None = None, require_complete=True)
    ids: np.ndarray                           # int32 (N,)
    settings: dict
    def array(self, item) -> np.memmap        # read-only
    def row_of(self, id) -> int

def write_synthetic_ffhq(root, n=16, seed=0) -> tuple[str, str]
    # Writes 512x512 HQ PNGs (smooth random fields) and LQ PNGs (blur + noise + JPEG q=30 round-trip).
    # Returns (hq_dir, lq_dir).
def write_synthetic_cache(paths, splits, seed=0) -> None
    # Random but valid cache in the exact format (tests for C/F only; E builds the real one).

class FinetuneDataset(Dataset):
    def __init__(self, paths, split: str, prompt: Literal["lq","hq_true","hq_pred","null","official"], hq_pred_file: str | None = None)
    # __getitem__(k) -> {"id": int64, "hq_pm1": fp32 (3,512,512) from PNG, "lat_lq": fp32 (4,64,64),
    #                    "lat_teacher": fp32 (4,64,64), "idx": int64 (256,) | (77,) official | absent for null}
    # Index source: lq -> lq_idx; hq_true -> hq_idx; hq_pred -> hq_pred_file rows aligned by ids (assert ids equal);
    #               official -> lq_prompt77.
class PredictorDataset(Dataset):   # (paths, split) -> {"z": fp32 (512,16,16), "y": int64 (256,)}

def step_order(n_items, n_steps, batch_size, seed) -> np.ndarray
    # (n_steps, batch_size) int64. Concatenates torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
    # over epochs. Deterministic, so resume is exact.
```

**Tests:**
- `make_splits` is deterministic, disjoint and has the right sizes. `load_splits` rejects tampering.
- `CacheWriter` resumability: write half, reopen, `pending` returns only the rest. The settings mismatch raises `CacheMismatchError`. Wrong dtype or shape raises. Out-of-range indices raise.
- `CacheReader(require_complete=True)` raises on a partial cache.
- `FinetuneDataset` works for every prompt mode on a synthetic cache.
- `step_order` is reproducible and covers every item once per epoch.
- `norm_stem`.

### 2.6 `train_predictor.py` (C): Step 5

**Models** (`--model {linear,transformer}`):
- `linear` (the plan's "logistic regression") [D, the plan is ambiguous]: a shared `Linear(512,1024)` plus a learned per-position bias `(256,1024)`, so 0.5M params. A separate linear per patch would be 134M params.
- `transformer`:
  - Input: `z (B,512,16,16)` to `(B,256,512)`, then `Linear(512,512)`.
  - Learned 2-D positions: `row_emb(16,512)[r] + col_emb(16,512)[c]`.
  - `nn.TransformerEncoder(6 layers, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)`, then `LayerNorm`, then `Linear(512,1024)`.
- **Mask:** `logits[..., ~seen] = -inf`, where `seen = bincount(hq_idx[pred_train], minlength=1024) > 0`. It is stored as a buffer and saved with the model.

**Training:**
- CE loss.
- AdamW, lr 3e-4, wd 0.05, batch 64, `--epochs 15`, cosine schedule with 1000 warmup steps [D].
- bf16 autocast on the trunk (`--amp bf16|none`). Logits go to fp32 before the mask and CE.
- Evaluate on val every epoch and keep the best by val top-1.
- Seed with `--seed 0`.
- Assert the training ids are exactly `splits["pred_train"]` and do not intersect ft_train or val.

**Reported in `metrics.json`:**
- top-1 and top-5 accuracy, on val and on a 2000-image held-in subset of pred_train.
- Accuracy per frequency bucket. Codes are ranked by pred_train frequency into buckets `top10`, `11-50`, `51-200`, `rest`.
- Mean L2 distance between the predicted and true HQ codebook rows.
- Baselines, with the same mask applied:
  - (a) most frequent code, global and per-position;
  - (b) nearest HQ code to the LQ feature, `nearest_codes(z_lq.float(), hq_codebook)`.

**Outputs:**
- `{predictor_dir}/{run_name}/` with `config.json`, `best.safetensors`, `last.safetensors`, `train_log.jsonl`, `metrics.json`.
- `pred/{split}.npy`: int16 (N,256) for ft_train and val, plus `pred/{split}.ids.json`.
- `pred/test_{name}.npy` for each `--predict_dir name=DIR`. These use z_lq computed on the fly with `vre.load_vre(branches=("lq",)).lq_features`, `resize="official"`, and `pred/test_{name}.names.json`.
- `meta.json`: `{"train_split": "pred_train", "train_ids_sha": ..., "codebook": "hq", "mask_n_seen": int}`.

**CLI:**

```
python -m hqprompt.train_predictor --model transformer --run_name tf_s0 [--epochs 15] [--batch_size 64] [--lr 3e-4] [--seed 0]
  [--predict_dir celeba=$HQP_CELEBA_LQ --predict_dir lfw=$HQP_LFW_LQ] [--predict_only] [--dry_run] + path args
```

**Tests:** a dry run on a synthetic cache trains 2 epochs on a tiny transformer (`--layers 1 --d_model 64` flags allowed) and writes every output. Masked codes are never predicted. Overlap between the train ids and ft_train raises.

### 2.7 `metrics/` and `evaluate.py` (D)

**Metrics port** (`git show origin/eval:Evaluation/metrics/*.py`). Changes:
- Load failures raise. Per-metric `try/except` returning NaN is not allowed.
- `compute` returns **per-image** values, not batch means.
- `LandmarkDistance.compute` returns per image `{"lmd": float|nan, "lmd_fail": ""|"restored"|"hq"|"both"}`.
- `IdentityDistance` passes `model_rootpath=paths.facexlib_dir`.
- Keep the pyiqa ids exactly as in the old code: lpips, dists, psnr, ssim, musiq, niqe, clipiqa, maniqa, fid.
- **Never copy the old YAML paths.** They point to `/projectnb/cs585`, which is forbidden.

**`evaluate.py` subcommands:**

```
score   --restored_dir D [--hq_dir H] (--split val | --names_from_dir) --metrics lpips dists deg lmd musiq [psnr ssim niqe clipiqa maniqa]
        [--fid_ref DIR] --out_dir O
        -> O/per_image.csv (name + one column per metric + lmd_fail), O/summary.json
           summary: per metric {mean, std, n}, lmd {n_valid, n_fail_restored, n_fail_hq, n_fail_both}, fid {value}, inputs, env_info
        With --split val: the expected names are f"{id:05d}" for the val ids. Assert all are present, with no extras.
compare --a O_a --b O_b [--lmd_common O_c ...] [--metrics ...] --n_boot 10000 --seed 0 --out X.json
gate    --row4 O4 --row2 O2 --row3 O3 --n_boot 10000 --seed 0 --out gate.json --md gate.md
diff    --a dirA --b dirB --out diff.json      # prompt sensitivity: per-image mean |a-b| (0-1 scale) and LPIPS(a,b); mean and quantiles
table   --run NAME=O ... --out results(.csv/.md)   # one row per run; LMD on the common detected set across all listed runs, plus failure counts
```

```python
DIRECTION = {"lpips": -1, "dists": -1, "deg": -1, "lmd": -1, "niqe": -1, "fid": -1,
             "musiq": +1, "clipiqa": +1, "maniqa": +1, "psnr": +1, "ssim": +1}
def read_per_image(out_dir) -> dict[str, dict[str, float]]          # csv module, no pandas
def common_names(tables: list[dict], metric: str) -> list[str]      # names where the metric is finite in every table
def paired_bootstrap(a, b, n_boot=10000, seed=0, ci=0.95) -> dict
    # d = a - b; idx = default_rng(seed).integers(0, n, (n_boot, n)); boot = d[idx].mean(1)
    # Returns {"mean_diff", "lo", "hi", "n"} with percentile CI.
def gate_decision(row4, row2, row3, n_boot=10000, seed=0) -> dict
```

**Gate rule (plan Step 4), as implemented [D on the ambiguous parts, flagged in section 8]:**
- **Common sets:**
  - For LPIPS, Deg. and MUSIQ, use the 500 val names, which must be present in all three runs.
  - For LMD, use the names whose LMD is finite in **all three** runs.
- **Per comparator** c ∈ {row2, row3} and metric m ∈ {lpips, deg, lmd} (lower is better):
  - `win[c][m] = (CI_hi of mean(row4 - c) < 0)`.
- **MUSIQ:** `musiq_clearly_worse[c] = (CI_hi of mean(row4 - c) < 0)`, meaning row4's MUSIQ is significantly lower.
- **Strict rule (decides):**
  - `n_joint = #{m : win[row2][m] and win[row3][m]}`.
  - `go = n_joint >= 2 and not any(musiq_clearly_worse.values())`.
- **Also reported:**
  - The lenient variant: ≥2 wins against each comparator separately.
  - All CIs, the LMD common n and the failure counts.
  - DISTS CIs, for information only.
- **Outputs:** `gate.json` and a markdown table.

**Tests (synthetic CSVs):**
- row4 clearly better on all three metrics gives go.
- Better on only one metric gives no-go.
- MUSIQ significantly worse gives no-go.
- Metrics that win only against row2 give no-go strict and go lenient.
- NaN LMD rows are excluded consistently.
- The bootstrap is deterministic for a given seed.
- A CI for an identical-arrays diff collapses to 0.
- `score` with `importorskip("pyiqa")` on 2 tiny images.

### 2.8 `check_500.py` (E): Step 0

**Images:** 500 ids drawn with `default_rng(0).choice(pred_train, 500, replace=False)` and sorted. The ids are saved in the report. If splits.json is missing, call `data.make_splits` and `write_splits` first.

**C1 (100 clean and 100 LQ faces, the first 100 ids):**
- Load both branches with decoders.
- For each branch b ∈ {hq, lq}, input mode ∈ {single, double} and image type ∈ {clean, lq}:
  - `idx = nearest_codes(encode_features(prepare_input(x, mode)), codebook)`
  - `out = decode_indices(idx)`
  - Score against the input image x01 with two output mappings: `(out+1)/2` and `(out+3)/4`. Clamp to [0,1].
  - Metrics: PSNR and L1, plus LPIPS if pyiqa is importable.
- **Pass rule [D]:** pick the mapping and mode where the hq branch scores best on clean images. Then require that the hq branch beats the lq branch on clean→clean PSNR, with a paired bootstrap 95% CI (`evaluate.paired_bootstrap`) above 0.
- Secondary result, reported only: lq beats hq on LQ→LQ.
- **Outputs:**
  - `checks/c1.json` with `{"c1_pass", "recommended_hq_input_mode", "table"}`.
  - `checks/c1_grid_{clean,lq}.png`: 8 faces × columns [input, hq-single, lq-single, hq-double, lq-double].
- If C1 fails: write the report, then `sys.exit(2)`, unless `--continue_on_c1_fail` is given (C2/C3 are then labelled `"valid": false`).

**C2 (all 500):**
- HQ codes come from clean faces through the hq branch with the recommended mode. LQ codes come from LQ faces through the lq branch with the double mode.
- Report:
  - distinct codes used and the frequency histogram (`c2_hist_{hq,lq}.npy`/`.png`);
  - the fraction of picks with codebook-row norm < 0.1, and with norm ≥ 1;
  - the overlap of used codes with the bit-identical rows shared by the two codebooks;
  - the distribution of unique-code counts in the official 77 prompt, and the fraction of images needing padding.
- Output: `checks/c2.json`.

**C3:**
- Features are z_lq fp32 from 500 LQ faces. Targets are hq_idx.
- Fit on 400 faces and test on 100, using the `linear` model from 2.6: Adam lr 1e-3, 50 epochs, batch 4096 patches, fp32.
- Report top-1 and top-5 against the baselines (global mode, per-position mode, nearest-HQ-code-to-LQ-feature).
- Output: `checks/c3.json`.

**CLI:** `python -m hqprompt.check_500 [--n 500] [--batch_size 8] [--continue_on_c1_fail] [--dry_run]` + path args.

**Test:** a dry run (random VRE with decoder, synthetic FFHQ n=16, `--n 8`) completes and writes all JSONs and PNGs. Exit code 2 is allowed with random weights, so the test passes `--continue_on_c1_fail`.

### 2.9 `infer_official.py` (E): Step 1 and the prompt-sensitivity check

This reproduces `OSDFace/infer.py` exactly except for documented flags.

**Per image:**
1. `x01 = load_image_01(resize="official")`. This applies the official bilinear `align_corners=True` resize when the image is square and not 512.
2. `x_pm1 = x01*2-1`.
3. `idx77 = vre.lq_official_prompt(x01)`. This applies the double normalization, matching official.
4. `prompt = PromptEncoder("official", lq_codebook, ec_sd)(idx77)`.
5. `z = vae_encode(vae, x_pm1, mode=--vae_mode)`. The default is `sample`, as in official. The generator is per image: `torch.Generator(device).manual_seed(seed + zlib.crc32(name))`, with seed default 114.
6. `x0 = one_step(unet_merged, z, prompt, alphas)`. The UNet dtype follows `--dtype fp32` (the default).
7. `out = vae_decode(x0)`, then `*0.5+0.5`, then `clamp(0,1)`.
8. Save with `torchvision.transforms.ToPILImage()`. Its truncation matches official.

**Batching:** `--batch_size` defaults to 1. Batching is valid because the port handles each sample separately.

**Prompt variants** (`--prompt_variant {normal,shuffled,zero_code,zero_embed}`):
- `shuffled`: precompute idx77 for all images, then image i uses `idx[src[i]]`. `src` is a derangement: `perm = rng.permutation(N)`, `src[perm[k]] = perm[(k+1)%N]`.
- `zero_code`: all entries -1.
- `zero_embed`: `prompt = zeros(B,77,1024)`.

**Input:** `--split val` (FFHQ LQ val ids) or `--input_dir DIR --dataset_name NAME`. Also `--limit N` and `--tf32 off|on` (default off; official ran with cudnn TF32 on, which is a minor difference).

**Outputs** go to `{outputs_dir}/official/{dataset}_{variant}/`:
- `images/{stem}.png`
- `prompt_idx.npy` (N,77) int16 with -1 padding
- `meta.json`: args, weights_manifest, merge report, input dir, file count, sha of names, `"lq_set_note"` (the default CelebA LQ is self-generated) and env_info.

**Test:** a dry run with tiny models, a random VRE and synthetic images, for every variant. In the shuffled variant, no image uses its own codes.

### 2.10 `build_cache.py` (E): Step 2

**CLI:**

```
python -m hqprompt.build_cache --split {pred_train,ft_train,val,all} --stage {codes,latents,teacher,all} --hq_input_mode {single,double}
  [--batch_size 16] [--teacher_batch_size 4] [--teacher_dtype fp32] [--num_workers 4] [--dry_run]
python -m hqprompt.build_cache --verify --n 100 --seed 1
```

`--hq_input_mode` has **no default** and must equal `checks/c1.json` `recommended_hq_input_mode` (assert). If c1.json is missing or `c1_pass` is false, refuse, unless in dry_run.

**Stages:**
- **codes:** run once per image batch.
  - `z = lq_features` (fp32). `lq_idx`, `lq_prompt77` and `hq_idx` are computed from fp32 features before any fp16 cast.
  - Store `z_lq = z.half()`.
  - Only the items listed in `SPLIT_ITEMS[split]` are written.
- **latents:** `lat_lq = vae_encode(mode="mean")` in fp32, stored fp16.
- **teacher:**
  - `x0 = one_step(merged_unet, lat_lq_fp32_recomputed, PromptEncoder("official")(lq_prompt77), alphas)` with the UNet in `--teacher_dtype` (default fp32, no autocast). Store fp16.
  - The teacher uses the **posterior mean** latent, the same input the student sees. It is not the official `sample()` [D, flagged].

**Numerics:** `common.set_numerics(deterministic=True, tf32=False)`.

**Disk check:** before allocating, `shutil.disk_usage(out_root)` must be at least 1.2 × the bytes needed. Print the bytes needed. Remind the user to check the shared quota manually [U: whether `disk_usage` reflects the GPFS fileset quota].

**Manifest:** `cache/manifest.json`, written atomically. The settings must match on resume; on mismatch raise `CacheMismatchError` (no silent rebuild).

```json
{"version": 1, "created": "...", "updated": "...",
 "splits": {"path": ".../splits.json", "sha256": "..."},
 "weights": {"associate_2.ckpt": {"path","size","mtime"}, "lora": {...}, "embedding_change": {...}, "unet": {...}, "vae": {...}, "scheduler": {...}},
 "settings": {"lq_input_mode": "double", "hq_input_mode": "<from C1>", "code_lookup": "fp32_raw_l2_argmin",
              "vae_latent": "mean", "scaling_factor": 0.18215, "t": 399, "teacher_prompt": "official77",
              "teacher_unet_dtype": "fp32", "tf32": false, "patch_order": "raster_row_major"},
 "env": {...env_info...},
 "items": {"<split>": {"n": 10000, "ids_sha256": "...", "arrays": {
     "hq_idx": {"file": "ft_train/hq_idx.npy", "shape": [10000,256], "dtype": "int16", "n_done": 10000}, "...": {}}}},
 "verification": null}
```

**Files:** `cache/{split}/ids.npy` (int32 (N,), the sorted split ids), `cache/{split}/{item}.npy` (an `np.lib.format.open_memmap` file, so `np.load(mmap_mode="r")` works) and `cache/{split}/{item}.done.npy` (bool (N,)).

**Resumability:** each stage processes only `writer.pending(item)` in chunks of `batch_size`. Data is flushed before the done flags.

**Expected sizes:**
- z_lq: 69,500 × 512 × 256 × 2 B ≈ 18.2 GB.
- lat_lq and lat_teacher: 10,500 × 32 KB ≈ 0.34 GB each.
- Index arrays: under 40 MB.

**Verify:**
- Sample 100 random (split, row) pairs over all cached splits, seed 1, and recompute every item for them.
- Indices must match exactly.
- Features and latents must satisfy `|rec_fp32 - stored| <= 1e-3 + 2**-10*|rec_fp32|`.
- For every index mismatch, report the distance gap between the two candidate codes. Do not loosen the check.
- Write `cache/verify_report.json` and `manifest.verification`. Exit 1 on any mismatch.

**Test:** a dry run on synthetic FFHQ n=16 with a dry-run split (`sizes={pred_train:8, ft_train:4, val:2, spare:2}`):
- Builds all stages.
- Kill-and-resume: run `--stage codes` with `--limit_batches 1`, then the full build. The result equals a fresh build.
- `--verify` passes.
- A changed `--hq_input_mode` on resume raises.

### 2.11 `finetune.py` and `infer.py` (F): section 5 below

---

## 3. Cache format and `splits.json`

The cache format is covered in 2.5 and 2.10. Schema of `splits.json`:

```json
{"version": 1, "seed": 0, "n_total": 70000, "source": "ffhq",
 "hq_dir": "...images512x512", "lq_dir": "...LQ_images_512x512",
 "hq_pattern": "{id:05d}.png", "lq_pattern": "{id:05d}_LQ.png",
 "method": "np.random.default_rng(seed).permutation(n_total); contiguous slices pred_train,ft_train,val,spare; each sorted",
 "sizes": {"pred_train": 59000, "ft_train": 10000, "val": 500, "spare": 500},
 "splits": {"pred_train": [ ...ints... ], "ft_train": [...], "val": [...], "spare": [...]},
 "sha256": "<sha256_list of json.dumps(splits, sort_keys=True)>",
 "created": "ISO-8601"}
```

The spare split is never read by any script except `load_splits` validation. CelebA-Test is never used for tuning: `finetune` and `train_predictor` reject any split that is not a FFHQ split.

---

## 4. Prompt contract (summary; it lives only in `prompt.py`)

| mode | codes in | codebook | tokens | projection | positions | trainable |
|---|---|---|---|---|---|---|
| official (row 1, teacher, smoke check) | `lq_prompt77` (B,77) int64, -1 pad: 77 unique codes sorted by patch min-distance | LQ | 77 | embedding_change, frozen | none | nothing |
| lq (row 2) | `lq_idx` (B,256) int64, raster | LQ | 256 | TokenProjection initialized from embedding_change, fp32 | `pos` (256,1024), zero init | proj + pos |
| hq_true (row 4) | `hq_idx` (B,256) | HQ | 256 | same | same | proj + pos |
| hq_pred (row 5) | predictor `pred/{split}.npy` (B,256) | HQ | 256 | same | same | proj + pos |
| null (row 3) | none | none | 256 | none | none | `tokens` (256,1024), init = projected zero code [D] |

**Path:** idx int64, then `codebook[idx]` fp32 (B,N,512), then TokenProjection (conv1 → SiLU → conv2 → SiLU) to fp32 (B,N,1024), then `+ pos`, then UNet `encoder_hidden_states`. Cross-attention takes N=256 with no other change.

**Code source versus model:** `infer.py` asserts `CODEBOOK_FOR_MODE[codes] == CODEBOOK_FOR_MODE[trained_mode]`. So evaluating the row-4 model with hq_pred codes is allowed, but lq codes on an hq model are refused.

---

## 5. `finetune.py` (F)

**CLI:**

```
python -m hqprompt.finetune --prompt {lq,null,hq_true,hq_pred} --seed 0 --lambda_3d 0
  [--hq_pred_run DIR]        # required iff hq_pred; uses DIR/pred/ft_train.npy, asserts DIR/meta.json train_split == "pred_train" and ids == ft_train ids
  [--run_name NAME]          # default f"{prompt}_l3d{lambda_3d:g}_s{seed}"; run dir = {runs_dir}/{run_name}
  [--max_steps 10000] [--batch_size 2] [--lr 5e-5] [--warmup 200] [--weight_decay 0.0] [--grad_clip 1.0]
  [--save_every 2500] [--log_every 10] [--lora_rank 16] [--lora_alpha 16]
  [--w_mse 1.0] [--w_dists 1.0] [--w_id 0.1] [--w_anchor 1.0]
  [--amp bf16|none] [--grad_ckpt_unet] [--grad_ckpt_vae] [--num_workers 4]
  [--smoke] [--resume CKPT_DIR] [--dry_run] + path args
```

`--lambda_3d`: if it is not 0, `raise NotImplementedError("structure loss (Proposal 2) not implemented")` before any model loads.

**Hook:** `compute_losses(out_pm1, x0, batch, loss_modules, weights, structure_fn: Callable | None = None) -> dict[str, Tensor]`. It adds `lambda_3d * structure_fn(out01, hq01, batch)` when `structure_fn` is given. `finetune` always passes None for now.

**Model setup:**
1. `models, report = osdface.load_osdface(paths, device, merge_lora=True)`. The base UNet, VAE and alphas are fp32 and frozen. Assert 278 pairs and 64 conv layers.
2. Add the new LoRA:

   ```python
   unet.add_adapter(LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, init_lora_weights=True,
                               target_modules=["to_q","to_k","to_v","to_out.0"]))
   ```

   - Uses diffusers 0.27 `PeftAdapterMixin` [U: confirm `add_adapter`, `disable_adapters` and `enable_adapters` exist in 0.27.2].
   - Assert 128 LoRA modules (16 transformer blocks × 2 attention modules × 4 targets [V by architecture count; U: confirm]).
   - LoRA parameters are fp32 and trainable. Everything else in the UNet has `requires_grad False`.
   - Do not use CRAFT's `get_peft_model` with dropout 0.1, because the plan says 0.
3. `prompt_enc = PromptEncoder(args.prompt, codebook, ec_sd, trainable=True)`.
4. Optimizer: `AdamW(lora_params + prompt_enc params, lr, betas=(0.9,0.999), weight_decay=args.weight_decay)` and `diffusers.optimization.get_cosine_schedule_with_warmup(opt, 200, max_steps)`.
   - Assert the optimizer's param set equals the set of `requires_grad` params.
   - Assert every trainable param is fp32 (trap 5).
5. Loss modules come from `losses.build_losses`.
6. `--amp bf16`: assert `torch.cuda.is_bf16_supported()`. Both A100 and A40 support it.

**Step loop** (data from `FinetuneDataset(split="ft_train")` read via `step_order(N, max_steps, 2, seed)`, no augmentation):

```
idx/null -> prompt = prompt_enc(idx or batch_size=B)                       # fp32, outside autocast
x0  = osdface.one_step(unet, lat_lq, prompt, alphas, amp_dtype=bf16)       # UNet under bf16 autocast; x0 float64 -> fp32
out = osdface.vae_decode(vae, x0, amp_dtype=bf16, clamp=False).float()     # gradients flow through the frozen decoder (no no_grad, trap 3); assert out.requires_grad
L_mse    = mse(out, hq_pm1)                                                # pixel space [-1,1] [D]
o01, h01 = ((out.clamp(-1,1)+1)/2), ((hq_pm1+1)/2)
L_dists  = ea_dists(o01, h01); L_id = arcface(o01, h01)                    # both fp32, autocast disabled
L_anchor = mse(x0, lat_teacher)                                            # scaled latent space
L = w_mse*L_mse + w_dists*L_dists + w_id*L_id + w_anchor*L_anchor (+ lambda_3d*structure; hook only)
L.backward(); clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
```

No GAN. `--grad_ckpt_vae` calls `vae.enable_gradient_checkpointing()`. diffusers only checkpoints in train mode, so it also sets `vae.decoder.train()` after asserting every Dropout has p == 0 [U: memory at batch 2 without checkpointing].

**Run directory** (`{runs_dir}/{run_name}/`):
- `config.json`: args, weights_manifest, merge report, code_fingerprint, env_info, `splits.sha256`, `cache manifest.settings`.
- `train_log.jsonl`: one line every `log_every` steps with `{"step", "lr", "loss", "mse", "dists", "id", "anchor", "grad_norm_total", "grad_norm_lora", "grad_norm_proj", "grad_norm_pos"|"grad_norm_tokens", "it_s", "max_mem_gb"}`.
- `ckpt/step_{002500,005000,007500,010000}/`, each with:
  - `trainable.safetensors`: keys `lora.<peft name>` from `peft.utils.get_peft_model_state_dict(unet)` and `prompt.<name>` from `prompt_enc.trainable_state_dict()`.
  - `optimizer.pt`, `scheduler.pt`, `rng.pt` (only torch CPU and CUDA RNG tensors).
  - `state.json` `{"step"}`.
- `DONE`: written after the final save.

**Startup rules:**
- If the run dir exists and is non-empty, refuse to start unless `--resume` is given.
- `--resume` must point into this run's `ckpt/`, and `config.json` must match the new args except for `resume`.
- There is no auto-resume.

**`--smoke`** (200 steps; writes `smoke_report.json` and exits 1 on any failure):
1. **Iteration-0 match:** before any training, for 4 val items, `x0 = one_step(...)` with `PromptEncoder("official")(lq_prompt77)`, `unet.disable_adapters()` and no autocast in fp32, compared with `lat_teacher`. Require `max|diff| <= 1e-2` and `mean|diff| <= 1e-3`. Also report the bf16-autocast diff. Then `enable_adapters()`.
2. **Gradients:** at step 0, every `lora_B` gradient norm and the proj/pos (or tokens) gradient norms are > 0. At step 10, `lora_A` gradient norms are > 0 (A's gradient is zero while B = 0).
3. **Frozen weights:** per-tensor sha256 of every non-trainable UNet, VAE and codebook tensor before and after 200 steps must be identical.
4. **Loss falls:**
   - Primary check: total loss on a fixed 4-image probe batch from ft_train (no_grad) is lower at step 200 than at step 0.
   - Also report the mean loss over steps 0-49 versus 150-199.
5. Record it/s and peak memory.

### `infer.py` (F)

```
python -m hqprompt.infer --run_dir R [--step 10000] --codes {lq,hq_true,hq_pred,null} [--hq_pred_run P]
   (--split val | --input_dir DIR --dataset_name NAME [--hq_dir H]) [--amp bf16] [--batch_size 4] [--out_dir]
```

- **Default output dir:** `{run_dir}/outputs/step_{step}/{codes}/{dataset}/images/{stem}.png`, plus `meta.json`.
- **Model:** loads the merged UNet, adds the adapter with the same config, loads `trainable.safetensors` strictly (both prefixes) and builds `PromptEncoder(trained_mode)`.
- **`--split val`:** uses the cache (`lat_lq`, plus `lq_idx`/`hq_idx`, or `P/pred/val.npy`).
- **`--input_dir`:** loads images with `resize="official"` and computes `lat_lq = vae_encode(mode="mean")`.
  - lq codes come from `vre.lq_codes`.
  - hq_pred comes from `P/pred/test_{name}.npy` (names asserted equal).
  - hq_true needs `--hq_dir` and `vre.hq_codes` with the manifest's `hq_input_mode`.
- **Generation:** `x0 = one_step` (bf16 autocast as in training), then `vae_decode(clamp=True)`, then `*0.5+0.5`, then `ToPILImage`.

**Tests (F):**
- With diffusers and peft present (`importorskip`): `finetune --dry_run --smoke` for each of the 4 modes on `tiny_models` with stub losses and a synthetic cache whose `lat_teacher` comes from `osdface.one_step(tiny merged unet, official prompt)`.
  - All smoke checks pass, using `--max_steps 20` in dry-run smoke mode (`lora_A` checked at step 10).
  - Then checkpoint save, `--resume` and `infer.py --split val`.
- `--lambda_3d 0.1` raises.
- A non-empty run dir without `--resume` refuses to start.
- `infer --codes lq` on an `hq_true` run raises.

---

## 6. Evaluation and gate procedure

**Gate (Step 4):**
1. Fine-tune rows 2, 3 and 4 (`--prompt lq`, `null`, `hq_true`; seed 0).
2. `infer.py --split val` with the codes matching each row's training mode.
3. `evaluate score --split val --metrics lpips dists deg lmd musiq` for each row (no FID).
4. `evaluate gate --row4 --row2 --row3`, with the rule from 2.7.

**Prompt sensitivity:** `infer_official --split val --prompt_variant {normal,shuffled,zero_code,zero_embed}`, then `evaluate diff` of normal against each other variant, plus `score` on each.

**Step 7:** `score` with all metrics and `--fid_ref $HQP_FFHQ_HQ` on CelebA-Test (`--hq_dir celeba_hq`) and LFW (no-reference metrics only). Then `table` over the runs, with LMD on the common detected set and failure counts per row.

All reports must say the CelebA LQ set is self-generated.

---

## 7. Work split (6 tasks)

Interfaces in sections 1, 2 and 4 are frozen. Coders must not change signatures. If one must change, stop and report.

**Wave 1 (parallel):**
- **A: core.**
  - Files: `__init__.py`, `paths.py`, `common.py`, `vre.py`, `prompt.py`, `tests/conftest.py`, `tests/test_paths.py`, `tests/test_vre.py`, `tests/test_prompt.py`.
  - Acceptance: all tests in 2.1 and 2.2 pass locally, including the CRAFT parity tests. `assert_writable` rejects `/projectnb/.../data` and `../` escapes. `safe_torch_load` rejects non-allowlisted paths.
- **B: diffusion and losses.**
  - Files: `osdface.py`, `losses.py`, `tests/test_osdface.py`, `tests/test_losses.py`.
  - Acceptance: tests in 2.3 and 2.4 pass. The diffusers parts pass in an env with the pins.
- **C: data and predictor.**
  - Files: `data.py`, `train_predictor.py`, `tests/test_data.py`, `tests/test_predictor.py`.
  - Acceptance: tests in 2.5 and 2.6 pass. The predictor `--predict_dir` path is tested once A lands (lazy import of vre).
- **D: evaluation and jobs.**
  - Files: `metrics/*`, `evaluate.py`, `jobs/*`, `requirements.txt`, `tests/test_evaluate.py`.
  - Acceptance: gate and bootstrap tests pass locally. `bash -n` passes on every job script. No job script contains a GPU type or a cs585 path.

**Wave 2 (after A, B, C):**
- **E: Step 0-2 scripts.**
  - Files: `check_500.py`, `infer_official.py`, `build_cache.py`, and their tests.
  - Acceptance: the dry runs in 2.8, 2.9 and 2.10, including resume and verify.
- **F: training and inference.**
  - Files: `finetune.py`, `infer.py`, `tests/test_finetune.py`.
  - Acceptance: section 5 tests. The dry-run smoke passes for all 4 modes.

**Order on SCC:** `setup_env` → `check_500` (stop if C1 fails) → `infer_official` (Step 1) → `build_cache` + `--verify` → `finetune --smoke` → gate runs → `train_predictor` → row 5.

**`jobs/` spec (D):**
- **`setup_env.sh`**: port of CRAFT `setup2.sh` and `requirements.txt`. Changes from the original:
  - `ENV_DIR=/projectnb/cs599bg/manaswi/CleanCue/env`.
  - Starts with `module load miniconda`, then `eval "$(conda shell.bash hook)"` before `conda activate "$ENV_DIR"`. A bare `conda activate` fails in a non-interactive bash.
  - Exports `PIP_CACHE_DIR` and `CONDA_PKGS_DIRS` under `hqprompt/` so nothing is written to `$HOME`.
  - `python=3.10`; `torch==2.4.0+cu121` and `torchvision==0.19.0+cu121` from the cu121 index.
  - `diffusers==0.27.2`, `transformers==4.37.2`, `peft==0.12.0`, `accelerate==0.31.0`, `tokenizers==0.15.2`, and **`huggingface_hub==0.24.7`**. `setup2.sh` leaves it unpinned, but `requirements.txt` explains diffusers 0.27.2 needs `cached_download`.
  - Adds `face-alignment`, `lpips`, `scikit-image`, `pytest` and `safetensors`.
  - Keeps the xformers pin and the torch re-pin step from `setup2.sh`.
  - Keeps its import-verification block, plus face_alignment and pytest.
  - Refuses to run if `$ENV_DIR` already exists.
- **`env.sh`**:
  - `module load miniconda`, the conda hook, and `conda activate /projectnb/cs599bg/manaswi/CleanCue/env`.
  - `export HQP_ROOT=/projectnb/cs599bg/manaswi/CleanCue/work`.
  - Caches under `$HQP_ROOT/model_cache`: `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `PIP_CACHE_DIR`.
  - `HQP_CODE` defaults to the script's parent directory.
- **Step scripts:** `check_500.sh`, `infer_official.sh`, `build_cache.sh`, `verify_cache.sh`, `smoke.sh`, `finetune.sh`, `infer.sh`, `evaluate.sh`, `gate.sh`, `train_predictor.sh`, `prefetch_models.sh`.
  - Each runs as `bash jobs/X.sh [ARGS]` inside an allocated session and passes "$@" through. Variables like `PROMPT`/`SEED` can also be set via env.
  - Each starts with `set -euo pipefail`; `source "${HQP_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/jobs/env.sh"`; `nvidia-smi` (except the CPU-only ones); tee to `$HQP_ROOT/logs/<step>_<date>.log`.
  - Optional qsub header lines: `#$ -P cs599bg`, `#$ -l h_rt=…`, `#$ -l gpus=1`, `#$ -pe omp 4`, `#$ -j y`. They are inert under bash and have **no GPU type**. A comment says to add `-l gpu_type=…` at submit time, and that under qsub `HQP_CODE` must be exported because `BASH_SOURCE` points at the spool directory.
- **`prefetch_models.sh`:** downloads the pyiqa (lpips, dists, musiq, niqe, clipiqa, maniqa, fid), facexlib arcface and face_alignment weights into model_cache [U: internet on compute nodes].
- Every script starts with a comment saying the user must approve it before running (plan rule 5).

---

## 8. Open questions and risks

1. **[U] HQ/LQ branch identity.** `vqvae.*` being the HQ branch is only inferred from naming. C1 gates everything.
2. **[V] Decoder bug.** CRAFT and official `MultiHeadDecoder` cannot run: `MultiHeadAttnBlock` returns a tuple, which the decoder does not unpack. Official never uses the decoder (`VQVAEencoder` has none). C1 needs the fixed port.
   - **[U]** The decoder config (ch_mult, z_channels, head_size) is inferred. A wrong `head_size` loads strictly (same shapes) but gives a silently wrong result; bad C1 reconstructions on both branches would point to this.
3. **[V] Official loader uses `strict=False`**, which hides key mismatches. Ours is strict. **[U]** Whether the checkpoint contains extra keys such as `quantize.pos_embedding`. If the strict load fails, report the listed keys to the user instead of loosening it.
4. **[U] `associate_2.ckpt` (Lightning 1.0.8) may need `weights_only=False`.** The fallback is allowlisted and logged.
5. **[U] LoRA key format.** Naming (`unet.` prefix, `lora_A`/`lora_B` vs `lora.up`/`lora.down`) is unverified. The plan verified 278 pairs and 64 conv layers. The merge asserts these counts and that no keys are left over.
6. **[V] Official batch bug.** `lq_embed/vqvae.py:inference` uses `torch.min(d, dim=2)` over the whole batch, so it only works at batch size 1. CRAFT's `d[i]` fix equals official at batch size 1. Our per-sample port is equivalent.
7. **[V] Double normalization.** Official feeds the LQ encoder `4*x01-3`. Ours keeps this for the LQ path. The HQ mode comes from C1.
8. **[D] Teacher and VAE latent.** The teacher and the student both use the posterior mean latent; official `infer` uses `sample()`. So `lat_teacher` is not bit-identical to official outputs, and the smoke iteration-0 check compares against the cache, not against Step 1 images. Step 1's paper-number check is what anchors "matches official".
9. **[D] Plan ambiguities resolved here; the user should confirm:**
   - the pixel MSE space ([-1,1]);
   - null-token init (the projected zero code);
   - weight decay 0;
   - "one linear layer per patch" read as a shared linear plus per-position bias;
   - gate "beats both on at least two metrics" read strictly as the same two metrics against both (the lenient variant is also reported);
   - "MUSIQ not clearly worse" read as the CI not entirely below 0;
   - the C1 pass criterion;
   - smoke thresholds (1e-2 max, 1e-3 mean).
10. **[V] Old evaluation code problems.**
    - It turns metric failures into NaN, which breaks rule 8.
    - Its configs point to `/projectnb/cs585`, which is forbidden.
    - Its LPIPS comment says VGG, but pyiqa `"lpips"` may be the AlexNet variant [U]; which one the paper used is also unverified.
    - Deg. uses facexlib ArcFace, which may differ from the paper's network [U], so compare to paper numbers loosely.
    - The ID loss and Deg. share the ArcFace family, as plan caveat says.
11. **[V] CelebA LQ is self-generated**, so Step 1 numbers may not match the paper. Every report must state the LQ set used.
12. **[U] Environment.**
    - basicsr (pulled in by `setup2.sh`) imports `torchvision.transforms.functional_tensor`, which torchvision 0.19 removed. Importing pyiqa or facexlib may break. `setup_env.sh` must run its import check and report a failure rather than patch site-packages silently.
    - Whether pyiqa and facexlib can download on compute nodes (prefetch them).
    - facexlib's `model_rootpath` argument.
    - diffusers 0.27.2 on local torch 2.12 or Python 3.13, which is why local tests skip diffusers.
13. **[U] Memory.** Backprop through the VAE decoder at 512, plus DISTS and ArcFace at batch 2, may exceed A40 48 GB without `--grad_ckpt_vae` and `--grad_ckpt_unet`. Measure in the smoke run. The plan's "1-1.5 h" estimate came from SD-1.5 without decoder backprop.
14. **[U] Cache sizes and timing.** The 18.2 GB z_lq goes on a shared quota (check it manually). Random reads of 59K z_lq rows from GPFS during predictor training may be I/O bound; consider raising `num_workers`.
15. **[U] Strict cache verification.** Index equality can fail on near-ties if cuDNN picks a different algorithm. We force deterministic algorithms without TF32. Mismatches are reported with their distance gap, and the check is not loosened.
16. **Plan rule 6 (git branch and commits) is overridden by the coordinator:** no git operations. Provenance comes from `common.code_fingerprint()` hashes.

### Critical files for implementation
- `/Users/manaswiyadamreddy/Documents/CleanCue/plan.md`
- `/Users/manaswiyadamreddy/Documents/CRAFT/vqvae.py` (port encoder and decoder; fix the decoder tuple bug)
- `/Users/manaswiyadamreddy/Documents/CRAFT/lq_embed.py` (`vqvae_encoder` loading and normalization, `TwoLayerConv1x1`)
- `/Users/manaswiyadamreddy/Documents/CRAFT/train_concat.py` (`merge_lora_into_unet`)
- `/Users/manaswiyadamreddy/Documents/BU/Spring 26/IVC/Project/OSDFace/infer.py` (official inference path, `merge_Unet`) and `.../OSDFace/utils/others.py:get_x0_from_noise`
- `/Users/manaswiyadamreddy/Documents/CRAFT/losses/dists.py`, `/Users/manaswiyadamreddy/Documents/CRAFT/losses/arcface.py`
- `git -C /Users/manaswiyadamreddy/Documents/CRAFT show origin/eval:Evaluation/{evaluate.py,metrics/iqa.py,metrics/identity.py,metrics/landmark.py,metrics/fid.py}`
- `/Users/manaswiyadamreddy/Documents/CRAFT/setup2.sh`, `/Users/manaswiyadamreddy/Documents/CRAFT/requirements.txt` (`jobs/setup_env.sh` port)
