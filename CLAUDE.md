# CleanCue: HQ-code prompts for OSDFace (Proposal 1)

Spec: `plan.md` (research plan) and `docs/code_plan.md` (code design). Only **Proposal 1** is in scope.
The old repo `/Users/manaswiyadamreddy/Documents/CRAFT` is reference only.

## Rules (from the user; these override plan.md)
- **No git operations at all.** Do not branch, commit or push. The user does all git work. Edit the working tree only.
- **Workflow:** the user pushes from the Mac, then runs `git pull` in the SCC clone. Claude does not rsync code to SCC.
- **SCC writes:** write only inside `/projectnb/cs599bg/manaswi/CleanCue/work/` (plus `env/`). `craft/ivc_checkpoints` and `craft/data` are read-only. Never touch other users' folders or `/projectnb/cs585`.
- **No GPU or heavy work on the login node.** Show each job script and get approval before telling the user to run it. Report failed checks as failures and never loosen a check.
- **The user picks the GPU when requesting a session.** Do not hard-code GPU types in the scripts.

## SCC layout
| What | Path |
|---|---|
| Repo clone (HTTPS; SCC has no GitHub SSH key) | `/projectnb/cs599bg/manaswi/CleanCue` |
| Conda env (git-ignored) | `/projectnb/cs599bg/manaswi/CleanCue/env` |
| `HQP_ROOT` write area (git-ignored): cache, runs, outputs, checks, eval, logs, model_cache | `/projectnb/cs599bg/manaswi/CleanCue/work` |
| OSDFace weights (read-only) | `/projectnb/cs599bg/manaswi/craft/ivc_checkpoints/osdface_official/pretrained` |
| Datasets (read-only) | `/projectnb/cs599bg/manaswi/craft/data/{train,test}` |

Path defaults live in `hqprompt/paths.py` (`SCC_BASE`, `HQP_ROOT_DEFAULT`) and `jobs/env.sh`.
The old `craft/hqprompt/` folder is no longer used.

## Environment: built and working (2026-11-26)
- Built with `jobs/setup_env.sh`: Python 3.10, torch 2.4.0+cu121, diffusers 0.27.2, transformers 4.37.2, peft 0.12.0, accelerate 0.31.0, huggingface_hub 0.24.7, pyiqa 0.1.12, `numpy<2`, `setuptools<70`.
- Known pip warnings that are harmless: the `pkg_resources` DeprecationWarning, the timm `models.layers` FutureWarning, and the `.condarc` warning (the script sets `CONDA_PKGS_DIRS`).

**Problems already solved; do not re-diagnose:**
1. `git clone git@github.com:...` gave "Permission denied (publickey)". Fix: clone over HTTPS.
2. The `academic-ml/fall-2026` module env was active in the shell. Fix: run `conda deactivate; module purge` before sourcing `env.sh`.
3. `pyiqa` raised `No module named 'pkg_resources'` because setuptools 84 removed it. Fix: `pip install "setuptools<70"`. This is now pinned in `setup_env.sh`.
4. `basicsr` breaks with torchvision 0.19 and is unused, so it was removed. `xformers` is not used.
5. `associate_2.ckpt` pickles a `pytorch_lightning.callbacks.ModelCheckpoint`, so loading it raised `No module named 'pytorch_lightning'`. Fix: `paths.safe_torch_load` falls back to a pickle module that turns any `pytorch_lightning`/`lightning` class into a placeholder stub. Do NOT install pytorch_lightning, because it can move torch.
6. pyiqa, openai-clip and face_alignment hard-code `~/.cache`, so prefetch first wrote about 1 GB to `/usr3/graduate/manaswiy/.cache`. Fix: `env.sh` exports `HQP_HOME_CACHE=work/model_cache/home`, and `hqprompt/__init__.py` sets `HOME` to it inside Python processes only. The files were moved there.
7. face_alignment 1.5's `torch.compile` fails with the old SCC g++ (`[[unlikely]]` errors) and falls back to eager. Fix: `FaceAlignment(..., compile=False)`, with face-alignment pinned to 1.5.0.

## Every new shell or session on SCC
```bash
conda deactivate 2>/dev/null; module purge
cd /projectnb/cs599bg/manaswi/CleanCue
export HQP_CODE=/projectnb/cs599bg/manaswi/CleanCue
source /projectnb/cs599bg/manaswi/CleanCue/jobs/env.sh
```
- Batch jobs: `qsub -l gpus=1 -l gpu_c=8.0 -v HQP_CODE=/projectnb/cs599bg/manaswi/CleanCue,VAR=... jobs/X.sh`. `HQP_CODE` must be exported, because under qsub `BASH_SOURCE` points at the spool directory.
- Logs go to `work/logs/`.
- GPU: needs bf16 (Ampere or newer). The user has used an **L40S (46 GB)** session. If fine-tuning hits out-of-memory, add `--grad_ckpt_vae`.
- `verify_cache.sh` must run on the same GPU type as `build_cache.sh`.

## Run order and status
Steps marked *once* save their results in `work/` and are not rerun unless the related code changes.

| # | Command | Once? | Status |
|---|---|---|---|
| env | `bash jobs/setup_env.sh` | once | DONE |
| pre | `bash jobs/prefetch_models.sh` (GPU session) | once | DONE (rerun once after fixes 6–7 to confirm) |
| 0 | `bash jobs/check_500.sh` → `work/checks/c1.json`. Exit 2 means C1 failed: STOP. | once | DONE 2026-11-26: C1 PASS, **HQ_INPUT_MODE=single**, mapping pm1, HQ−LQ clean PSNR +1.45 dB [1.24, 1.67]. C3 linear top1 0.361 / top5 0.744 vs baselines 0.010 (global mode), 0.073 (per-position mode), 0.126 (nearest HQ code) |
| 1 | `bash jobs/infer_official.sh`, plus `PROMPT_VARIANT={shuffled,zero_code,zero_embed}` | once | pending |
| 2 | `HQ_INPUT_MODE=<c1.json recommended_hq_input_mode> bash jobs/build_cache.sh` (resumable, about 20 GB; run `pquota -u cs599bg` first), then `bash jobs/verify_cache.sh` (must exit 0) | once | pending |
| 3 | `PROMPT=lq bash jobs/smoke.sh` | once | pending |
| 4 | finetune `PROMPT=lq/null/hq_true` (rows 2/3/4, 10k steps, qsub) → `infer.sh --codes P --split val` → `evaluate.sh score ... --out_dir work/eval/row{2,3,4}` → `gate.sh` | per run | pending |
| 5 | `train_predictor.sh` (about 32 GB RAM) → finetune `PROMPT=hq_pred --hq_pred_run work/predictor/tf_s0` (row 5) | per run | pending |

Run directories are named `work/runs/{prompt}_l3d0_s{seed}`. Inference writes to `{run_dir}/outputs/step_N/{codes}/{dataset}/images`.

## Verified on SCC (2026-11-26)
- `associate_2.ckpt`: the hq and lq branches both strict-load with their decoders, and each codebook is (1024, 512).
- The official LoRA merge gives 278 pairs (64 conv, 214 linear) at rank 16, with nothing unprocessed.
- The pyiqa 0.1.12 functions `fid_arch.get_folder_features` and `frechet_distance` exist.

## Not yet verified
- C1 side observations: on LQ inputs the HQ branch also reconstructs better than the LQ branch (23.1 vs 21.9 dB, `lq_beats_hq=false`). Both branches reconstruct best with single normalization, but the official LQ prompt path keeps its double normalization, because embedding_change and the LoRA were trained with it.
- Whether the pyiqa FID function signatures match what `metrics/fid.py` calls. This is first exercised in Step 7.
- Memory use at batch size 2 on an L40S.

## Local testing (Mac)
- Env: `/private/tmp/claude-501/-Users-manaswiyadamreddy-Documents-CleanCue/<session>/scratchpad/localenv`. This lives in the scratchpad, so it may be gone; rebuild it from `requirements.txt` if needed.
- Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/`
- Last full run: 107 passed, 3 skipped (the pyiqa and facexlib tests), about 8 min.
