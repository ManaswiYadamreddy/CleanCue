# Plan: HQ-code prompts and a 3D structure loss for one-step face restoration

Handoff document for a coding session. Written 2026-11-25. Course deadline: **2026-12-12**.

Labels used below:
- **[verified]**: checked directly by reading the files or code
- **[notes]**: from project notes, not re-checked. Confirm before relying on it
- **[estimate]**: a guess, not a measurement

---

## 0. Weight paths on SCC (the only ones we need)

```
# HQ + LQ encoders and codebooks
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/associate_2.ckpt

# Official OSDFace LoRA
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/pytorch_lora_weights.safetensors

# Projection 512 -> 1024 (feeds codes into the UNet)
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/embedding_change_weights.pth

# SD-2.1-base UNet
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/sd21/unet/config.json
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/sd21/unet/diffusion_pytorch_model.safetensors

# SD-2.1-base VAE
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/sd21/vae/config.json
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/sd21/vae/diffusion_pytorch_model.safetensors

# Noise schedule
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/sd21/scheduler/scheduler_config.json
```

Read-only. Details and status of each file are in section 3.

---

## 1. What we are building

OSDFace (CVPR 2025) is a one-step diffusion face restorer. It prompts SD-2.1 with codes looked up from a codebook learned on **degraded (LQ)** faces.

- **Proposal 1 (main):** prompt it with codes from the **clean (HQ)** codebook. At test time there is no clean image, so a small classifier predicts the HQ codes from the LQ face.
- **Proposal 2 (add-on):** add a 3D facial-structure loss (D3DFR) to the same fine-tune.

**Frozen:** HQ encoder, LQ encoder, both codebooks, SD-2.1 UNet with the official LoRA merged, VAE.
**Trained:** the code predictor, a new small LoRA, and the prompt projection.

**Out of scope. Do not build unless asked:** training any codebook, region-aware codebooks, face-parser masks, GAN/discriminator, SD-1.5, text prompts, a second backbone, vertex loss, K-sample uncertainty maps.

---

## 2. Safety rules (must follow)

1. On SCC, **write only inside** `/projectnb/cs599bg/manaswi/CleanCue/work/` (CleanCue clone is `/projectnb/cs599bg/manaswi/CleanCue`; `craft/` is read-only).
2. Treat everything else as **read-only**, including `ivc_checkpoints/`, `data/` and `code/`.
3. **Never touch other people's folders** or anything under `/projectnb/cs585`.
4. Never run GPU or heavy CPU work on the login node. Use batch jobs.
5. **Show the user each script and ask before submitting any job.** Check disk quota before writing the cache (~20 GB). The quota is shared.
6. Locally, work on a new git branch. Commit or push only when asked.
7. Load `.ckpt`/`.pth` files only from the paths in section 3. `torch.load(weights_only=False)` can execute code.
8. Report failed checks as failures. Do not loosen a check to make it pass.

SSH: the alias is `scc`. The user opens the connection with `ssh -fN scc`. Use `ssh -o BatchMode=yes scc '...'`.

---

## 3. Weights: the only ones we need

Base folder:

```
/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained/
```

| File (relative to base) | Used for | Status |
|---|---|---|
| `associate_2.ckpt` | HQ and LQ encoders + codebooks | [verified] structure |
| `pytorch_lora_weights.safetensors` | official OSDFace LoRA | [verified] structure |
| `embedding_change_weights.pth` | projection 512 -> 1024 | [verified] structure |
| `sd21/unet/` (`config.json`, `diffusion_pytorch_model.safetensors`) | UNet | [verified] file size matches header |
| `sd21/vae/` (`config.json`, `diffusion_pytorch_model.safetensors`) | VAE | [verified] file size matches header |
| `sd21/scheduler/scheduler_config.json` | noise schedule | [verified] |

Full paths of every file are in section 0.

**Not needed:** SD-1.5, the SD-2.1 text encoder/tokenizer, `79999_iter.pth` (face parser), and everything in `checkpoints*/`.

**Missing:**
- **D3DFR + Basel Face Model** (Proposal 2 only). Needs a license request by the user.
- **ArcFace and DISTS** (losses/metrics). Used before via `facexlib` and `pyiqa`, which download on first use. Not checked on the new account setup.

None of these models has been run yet. Only file structure and codebook values were checked.

### Inside `associate_2.ckpt` [verified]

Lightning 1.0.8 checkpoint. Weights are under `["state_dict"]`.

| Prefix | Meaning |
|---|---|
| `vqvae.*` | **HQ branch.** Inferred from naming, must be confirmed (check C1) |
| `vqvae_LQ.*` | LQ branch |
| `loss.*`, `loss_LQ.*`, `logit_scale` | ignore |

Each branch has `encoder`, `quant_conv` (512->512), `quantize.embedding.weight` (1024 x 512), `post_quant_conv`, `decoder`. The two branches have matching tensor names and shapes (405 tensors).

- **Many codes look untrained:**

| Codebook | norm >= 1 | norm < 0.1 |
|---|---|---|
| HQ | 530 | 489 |
| LQ | 214 | 806 |

  218 rows are bit-identical in the two codebooks. Which codes real faces use has not been measured.
- Lookup is **raw L2 distance** (no normalization), unlike our own cosine codebooks. Do not mix them.

### Official LoRA [verified]

Rank 16, 278 A/B pairs, of which **64 are conv (4-D)**. Alpha is assumed to be 16 (scale 1.0). Reproducing official results confirms it.

---

## 4. Data

| What | Path | Status |
|---|---|---|
| FFHQ HQ, 70K | `/projectnb/cs599bg/manaswi/craft/data/train/images512x512/00000.png` | [notes] |
| FFHQ LQ, fixed on disk | `/projectnb/cs599bg/manaswi/craft/data/train/LQ_images_512x512/00000_LQ.png` | [notes] |
| CelebA, LFW test | under `/projectnb/cs599bg/manaswi/craft/data/test/` | [notes], list it to find the exact folders |

Splits (fixed, save the index lists to JSON):

| Split | Size | Use |
|---|---|---|
| `pred_train` | 59,000 | train the code predictor |
| `ft_train` | 10,000 | fine-tune the LoRA |
| `val` | 500 | gate and hyperparameters |
| `spare` | 500 | untouched |

The predictor must **not** train on `ft_train`. Otherwise its codes there are unrealistically good and the LoRA learns from a mismatch. Official OSDFace trained on FFHQ, so `val` is not unseen for it. That affects all rows alike.

Never tune on CelebA-Test. It is for final numbers only.

**No flips or other augmentation**, so the caches are exact.

---

## 5. Existing code

Local repo: `/Users/manaswiyadamreddy/Documents/CRAFT` (branch `main`). SCC copy: `/projectnb/cs599bg/manaswi/craft/code/` [notes]; its branch is unknown, so compare before reusing.

| File | Reuse | Notes |
|---|---|---|
| `vqvae.py`, `lq_embed.py` | encoder classes | official OSDFace style |
| `train_concat.py: merge_lora_into_unet` | LoRA merge | handles the conv layers. Assert all 278 pairs applied and no unprocessed keys |
| `losses/dists.py`, `losses/arcface.py` | EA-DISTS, ID loss | from our Stage 2 |
| `origin/eval:Evaluation/` | metrics | LPIPS, DISTS, Deg., LMD, MUSIQ, NIQE, CLIPIQA, MANIQA, FID |
| `origin/code_swap:benchmark_original_osdface.py` | reference only | shows the official inference path |

Put new code in `hqprompt/`. Official OSDFace repo, for reference: https://github.com/jkwang28/OSDFace

### Traps already found

1. **Run the code lookup in fp32.** `vqvae.py: inference()` casts code indices to the model dtype. In a synthetic test bf16 kept only 24 of 77 correct codes, and fp16 72 of 77.
2. **Do not use `infer_concat.py: merge_unet`.** It uses `torch.mm`, which fails on the 64 conv layers.
3. **Do not copy the pixel loss in `train_concat.py`.** It decodes under `no_grad`, so it gives no gradient.
4. **Timestep is 399, not 999.** Our old Stage 2 used 999, which amplifies UNet error about 14.6x instead of 1.16x.
5. **Keep trainable weights in fp32.** Use bf16 autocast for the UNet and VAE only.
6. **Check the encoder input range.** `lq_embed.py` applies `(img - 0.5) / 0.5`, while `infer_concat.py` already passes images in [-1, 1], so they are normalized twice. Compare with the official `infer.py` and use whatever it does for official OSDFace. For the HQ encoder, find the right range with check C1.
7. **`TwoLayerConv1x1` asserts 77 tokens.** Write a version without the assert. Its forward is conv1 -> SiLU -> conv2 -> SiLU.
8. The official prompt is **up to 77 unique codes sorted by distance, with no positions**, zero-padded if fewer.
9. Old codebook entropy terms have no gradient. Irrelevant here, but do not reuse them.

### Environment

SCC's default `python3` is too old (no `math.prod`). Ask the user which module or conda env to use. Do **not** use another person's env. Pins from `requirements.txt`: `diffusers==0.27.2`, `transformers==4.37.2`, `peft==0.12.0`, `accelerate==0.31.0`, `huggingface_hub==0.24.7`, tested with torch 2.4.0.

---

## 6. Results table we are producing

| # | Prompt | Fine-tuned | 3D loss | Purpose |
|---|---|---|---|---|
| 1 | LQ codes (official 77 tokens) | no | no | official OSDFace |
| 2 | LQ codes | yes | no | **fair baseline** |
| 3 | none (learned constant tokens) | yes | no | does the prompt matter? |
| 4 | true HQ codes (from the clean image) | yes | no | upper bound, **not** a method |
| 5 | **predicted HQ codes** | yes | no | **Proposal 1** |
| 6 | LQ codes | yes | yes | **Proposal 2** |
| 7 | predicted HQ codes | yes | yes | both |

Comparisons: **5 vs 2** (Proposal 1), **6 vs 2** (Proposal 2), **7 vs 5** (does the loss help on top). Rows 2-7 must share the same data, iterations, LR, LoRA, token count and seed. Repeat rows 2 and 5 with a second seed.

Rows 2-7 all use **256 tokens (16x16) with learned positions**, so only the codes differ.

Also evaluate row 4's model with predicted codes (no extra training).

---

## 7. Steps

### Step 0. Sanity checks (~30 min GPU [estimate], ~500 faces)

Script: `hqprompt/check_500.py`

- **C1. Is `vqvae.*` really the HQ branch?** Reconstruct 100 clean faces and 100 LQ faces through each branch. `vqvae` should do better on clean faces. Try both input ranges. Save an image grid.
- **C2. Code usage.** On the 500 faces, count the distinct codes used, their frequency, and how often near-zero-norm codes are picked, for both codebooks.
- **C3. Logistic regression.** Predict the HQ code from the LQ feature per patch (fit on 400 faces, test on 100). Compare with two no-training baselines: the most frequent code, and the nearest HQ code to the LQ feature.

**Stop and report if C1 fails.** Nothing else is valid without it.

### Step 1. Reproduce official OSDFace (~1 h [estimate])

Script: `hqprompt/infer_official.py`

- fp32 prompt path, t=399, official 77-token prompt, merged LoRA.
- x0 = (z_L - sqrt(1-ab) * eps) / sqrt(ab), in fp32.
- Paper, CelebA-Test: LPIPS 0.337, MUSIQ 75.64, Deg. 60.07, LMD 5.29, FID 45.42.
- **Caution:** our CelebA LQ images may be self-generated (an old folder was named `Self_CelebA_Validation_v2`), so the numbers may not match even if the code is right. Say which LQ set was used. Inspect images as well.

### Step 2. Cache (~1 h once [estimate])

Script: `hqprompt/build_cache.py`. Output: `/projectnb/cs599bg/manaswi/CleanCue/work/cache/`

| Item | Images | Size [estimate] |
|---|---|---|
| `hq_idx`: true HQ code per patch (256, int16) | 69K | ~35 MB |
| `z_lq`: LQ features after `quant_conv` (512x16x16, fp16) | 69K | ~18 GB |
| `lq_idx`: nearest LQ code per patch (256, int16) | 10K + val | small |
| `lat_lq`: VAE latent of LQ (4x64x64, fp16, posterior mean) | 10K + val | ~0.35 GB |
| `lat_teacher`: official OSDFace output latent | 10K + val | ~0.35 GB |

- 69K = `pred_train` + `ft_train`. Add `val` to every item.
- Compute every code index in **fp32** with raw L2, before any fp16 storage.
- Store as memory-mapped arrays, resumable, with a manifest (counts, dtypes, weight file size and mtime).
- Check: recompute 100 random items and compare exactly (indices) or within fp16 tolerance (features).

### Step 3. Fine-tuning script

Script: `hqprompt/finetune.py --prompt {lq,null,hq_true,hq_pred} --lambda_3d 0 --seed 0`

- **Model:** merged UNet + new LoRA (r=16, alpha=16, dropout 0, on `to_q,to_k,to_v,to_out.0`).
- **Prompt:** code embeddings (256 x 512) -> projection (initialized from `embedding_change`, trainable) -> + learned positions (zero init) -> cross-attention.
- **Null prompt:** 256 learned tokens of size 1024.
- **Loss:** MSE (1.0) + EA-DISTS (1.0) + ArcFace ID (0.1) + anchor (1.0: latent MSE to `lat_teacher`) + `lambda_3d` * structure. Gradients flow through the frozen VAE decoder. **No GAN.**
- **Optimizer:** AdamW, lr 5e-5, 200-step warmup, cosine, grad clip 1.0, batch 2, **10K iterations**, save every 2.5K.
- **Speed:** our old SD-1.5 Stage 2 logged 3.1 it/s at batch 2, so expect about 1-1.5 h [estimate].
- Use a separate output folder per run. No silent auto-resume across configs.

Smoke test first (200 iterations): the loss falls, LoRA **and** projection gradients are nonzero, frozen weights do not change, and iteration 0 with the official prompt path and the LoRA off matches official OSDFace.

### Step 4. GATE (3 runs in parallel, ~1.5 h each [estimate])

Run rows **2, 3, 4** and evaluate on `val` (500): LPIPS, DISTS, Deg., LMD, MUSIQ. No FID, because 500 images is too few.

- **Go** if row 4 beats **both** rows 2 and 3 on at least two of LPIPS, Deg., LMD, with paired-bootstrap 95% confidence intervals excluding zero, and MUSIQ is not clearly worse.
- **No-go** otherwise: report a clean negative result, skip rows 5 and 7, and lead with Proposal 2.

Also run a prompt-sensitivity check on official OSDFace (no training): normal, shuffled images' codes, zero prompt. Report how much the output changes.

### Step 5. Code predictor (~1 h [estimate])

Script: `hqprompt/train_predictor.py`

- **Baseline:** logistic regression (one linear layer per patch).
- **Main:** small transformer. 6 layers, 512-d, 8 heads, learned 2-D positions, input `z_lq`, output 1024 logits per patch. Mask logits of codes never seen in `pred_train`.
- **Loss:** cross-entropy. Train on `pred_train`, select on `val`.
- **Report:** top-1, top-5, accuracy per frequency bucket, distance between predicted and true code vectors, and both no-training baselines from C3.
- Write `hq_pred_idx` for `ft_train`, `val` and the test sets.

### Step 6. Proposal 1 and 2 runs

- Row 5: `--prompt hq_pred`.
- Row 6: `--prompt lq --lambda_3d L`. Row 7: `--prompt hq_pred --lambda_3d L`.
- Choose L on `val` using row 6 only (2-3 values), starting where the structure gradient is ~10-50% of the rest. Reuse it for row 7.

**Structure loss:**
- D3DFR (Deep3DFaceRecon_pytorch) on 224x224 crops. Compare shape (id) and expression coefficients of output and ground truth. Use one fixed distance (1 - cosine) and say so.
- **Align both images with the ground-truth 5-point transform**, using a differentiable warp. Aligning each separately hides the drift.
- Cache ground-truth coefficients and landmarks.
- Tests: loss(GT, GT) is about 0, gradients reach the LoRA, and artificially warping the jaw raises the loss.
- LAFR (arXiv 2505.23462) already uses a D3DFR structure loss in one-step restoration. Cite it and do not claim the loss itself is new.

**Fallback:** if the Basel Face Model is not available by **Dec 1**, use a 2-D landmark loss and call it that. Do not use FAN, because our LMD metric uses FAN. A candidate is `facexlib`'s alignment model; confirm it is differentiable.

**Caveat to report:** the ID loss and the Deg. metric use the same ArcFace family, so Deg. is partly trained on. This is the same for all fine-tuned rows.

### Step 7. Final evaluation (~1 h per model [estimate])

- CelebA-Test for all rows, LFW for rows 1, 2, 5, 6, 7.
- Metrics: LPIPS, DISTS, Deg., LMD, MUSIQ, NIQE, CLIPIQA, MANIQA, FID.
- Report LMD failures per row. The metric skips images where detection fails, so compare rows on the **common** set of detected images.
- Outputs: results table (CSV + markdown), qualitative grid, gate result, predictor table.

---

## 8. Schedule

| Dates | Work |
|---|---|
| Nov 25-26 | Steps 0-2. User requests the Basel Face Model |
| Nov 27 | Step 3 and the **gate** (step 4) |
| Nov 28-30 | Step 5 and row 5 |
| Dec 1-3 | Rows 6 and 7, second seeds |
| Dec 4-6 | Step 7 |
| Dec 7-12 | Report and slides |

Total compute: about 12-18 GPU hours [estimate], not counting queue time.

## 9. Files to create

```
hqprompt/
  vre.py              # load both branches from associate_2.ckpt; fp32 code lookup
  osdface.py          # merged UNet, VAE, one-step forward at t=399
  data.py             # splits, cache readers
  check_500.py        # step 0
  infer_official.py   # step 1
  build_cache.py      # step 2
  finetune.py         # steps 3, 4, 6
  train_predictor.py  # step 5
  structure_loss.py   # step 6
  evaluate.py         # wraps origin/eval Evaluation/, adds bootstrap CIs
  jobs/               # SCC batch scripts (ask before submitting)
```

## 10. Start here

1. Read this file, then `lq_embed.py`, `vqvae.py`, `train_concat.py`.
2. Ask the user which SCC Python environment, project flag and GPU type to use.
3. Write `vre.py` and `check_500.py` only. Show them to the user.
4. After approval, run step 0 and report C1-C3 before writing anything else.
