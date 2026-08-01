# HF Fine-Tuning Optimization

## Goal

Minimize the MAE% (mean absolute error as percentage of ground-truth mean) of the fine-tuned Qwen3-VL-2B model on the Nutrition5k validation set across four nutrients: calories, protein, fat, and carbs. The target is to match or beat the legacy MLX result (18.1% avg) — and critically, produce a model that survives export to GGUF without degradation, so it can run on-device via llama.cpp / RunAnywhere SDK.

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely.

1. **Read the codebase**: Read these files for full context:
   - `training/train.py` — HF/PEFT LoRA fine-tuning script. **This is the primary file you edit.**
   - `training/evaluate.py` — evaluation script that computes MAE/MAE% metrics.
   - `training/hf_utils.py` — custom LoRA implementation for vision tower + projector, PEFT helpers.
   - `data/nutrition5k_baseline_data.json` — Thames et al. CVPR 2021 baseline values.

2. **Check for existing progress**: Read `training/notes.md` and `training/results.tsv` to review what's been tried and what to try next.

3. **Verify the environment** by running a quick sanity check:
   ```bash
   cd /Users/jyan/src/my-own-plate && .venv/bin/python -c "import transformers, peft; print(f'transformers {transformers.__version__}, peft {peft.__version__}')"
   ```

## Pipeline overview

The HF pipeline replaces the legacy MLX pipeline to avoid the MLX-to-GGUF accuracy degradation that was observed previously. Key components:

- **Training** (`train.py`): HF transformers `Trainer` + PEFT LoRA for LLM + custom `LoRALinear` (in `hf_utils.py`) for vision tower and projector. Runs on Apple Silicon via MPS.
- **Evaluation** (`evaluate.py`): Loads base model + PEFT adapter + custom vision LoRA, runs inference on val/test set, reports per-nutrient MAE%.
- **Export** (`merge_and_export.py`): Merges all adapters into base weights, saves as HF safetensors, converts to GGUF via llama.cpp.
- **Data**: HuggingFace parquet chat dataset at `data/nutrition5k_hf_chat/` (produced by `training/convert_dataset.py`).

### Current best HF config (Exp 2, 27.8% avg)

LLM LoRA r64 alpha 64 on q/k/v/o/gate/up/down_proj. Vision tower LoRA r32 alpha 32 on all 24 blocks. Projector LoRA r64 alpha 64. Cosine LR 1e-5 to 1e-6. 12 epochs. Batch size 1. Image resize 384x384. No gradient checkpointing. No dropout.

### Legacy MLX best (Exp 44, 18.1% avg — target to match)

LLM LoRA r64 alpha 1.0 on q/k/v/o/gate/up/down_proj. Vision tower LoRA r32 on all 24 blocks. Projector r64. Cosine LR 1e-5 to 1e-6. 10 epochs. Image resize 384x384.

### N5k RGB baselines (Thames et al. CVPR 2021)

| Nutrient  | N5k Baseline |
|-----------|--------------|
| Calories  | 26.1%        |
| Protein   | 29.5%        |
| Fat       | 34.2%        |
| Carbs     | 31.9%        |
| **Avg**   | **30.4%**    |

## Success criteria

An experiment is **successful** if BOTH conditions are met:
1. The **average MAE%** across all four nutrients decreases (lower is better).
2. **No individual nutrient's MAE%** increases by more than 5 percentage points compared to the best-so-far values.

Parse failures also matter: if parse failures increase significantly (>2x), the experiment should be rejected regardless of MAE% improvement.

## Lessons learned (from MLX pipeline + HF experiments so far)

These are hard-won findings. Do NOT re-explore these:

### What works
- **Vision block selection was the biggest single win axis in MLX.** Going from no vision LoRA to all 24 blocks at r32 went from 59.6% to 18.9% avg over many experiments. The progression: top-2 → top-4 → top-6 → top-8 → top-12 → all 24 blocks, each step improving.
- **Cosine LR decay 1e-5 to 1e-6** consistently outperforms constant LR.
- **10 epochs is optimal.** 12 epochs shows mild overfitting; 15/20/40 epochs clearly overfit.
- **Full LLM LoRA targets (attn + MLP)**: q/k/v/o_proj + gate/up/down_proj. Removing MLP projections hurts significantly.
- **LLM rank 64** is the sweet spot. Rank 32 is too low.
- **No dropout.** Even 0.05 degrades results.
- **No weight decay.** Harmful on LoRA.
- **No gradient checkpointing** in HF pipeline — it blocks gradients to custom vision LoRA.

### What doesn't work
- 3-epoch screening: Produced 5 consecutive reverts in HF pipeline. Signal is too noisy at 3 epochs.
- Image size 448+: Too slow on M2 Pro, memory pressure kills training.
- Alpha != rank (for HF PEFT, where scaling = alpha/r): alpha=2*rank corrupts, alpha=0.5*rank under-adapts.
- Chain-of-thought prompts: Dilute the nutrient signal.
- Data augmentation (hflip): Didn't propagate correctly, hurt results.

### The gap between HF and MLX
The HF pipeline at comparable config (exp 2, 12 epochs) gets 27.8% avg. The MLX pipeline at comparable config (exp 30, 10 epochs) got 27.4%. The pipelines are nearly equivalent at baseline. The remaining 9pp gap was closed in MLX by:
1. Vision block subset selection experiments (exps 32-39): 27.4% → 18.9%
2. Cosine LR refinement (exp 43): → 18.5%
3. Final combination (exp 44): → 18.1%

These same strategies should be tried in the HF pipeline.

## Research methodology

- **One variable at a time**: Change one hyperparameter or technique per experiment. If a complex change improves results, ablate to find the essential ingredient.
- **Full runs only**: Run 10-epoch experiments. Do not use 3-epoch screens — they waste compute without reliable signal for this task.
- **Diminishing returns**: If the last 5 experiments were all minor tweaks with tiny deltas (<0.5pp average), change strategy entirely.
- **Long runs are fine**: Individual training runs take 8-12+ hours on M2 Pro. Evaluation takes 20-40+ min. Do not cut corners or reduce epochs to save time.
- **Background execution required**: Both training and evaluation MUST be run with `run_in_background=true` on the Bash tool, since they exceed the 10-minute Bash timeout. Wait for the background notification — do not poll or sleep.
- **Evaluation**: Always use `--mode val` (the default) — never evaluate on the test set during experimentation.

## The experiment loop

The experiment runs on `main`. All kept experiments are committed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `training/notes.md` and `training/results.tsv` to review what's been tried, what worked, and what to try next.

2. **Design the next experiment** based on insights from the notes. Check the "Ideas queue" section of `training/notes.md` first. Make changes to `training/train.py` (and if needed, `training/hf_utils.py` or `training/evaluate.py`).

3. `git add training/ && git commit -m "exp N: description of experiment"`

4. **Train the model** (runs in background — training takes 8-12+ hours on M2 Pro):
   ```bash
   # MUST use run_in_background=true — training exceeds the 10-minute Bash timeout.
   cd /Users/jyan/src/my-own-plate && .venv/bin/python training/train.py \
       --model training/cache/Qwen3-VL-2B-Instruct \
       --train-data ~/src/my-own-plate/data/nutrition5k_hf_chat \
       --output-dir ~/src/my-own-plate/training/output \
       [additional flags for this experiment]
   ```
   Use `run_in_background=true` on the Bash tool call. You will be notified when training completes. **Do NOT poll or sleep** — just wait for the background task notification.
   If training crashes or produces NaN losses, fix the issue or revert and try something else.

5. **Evaluate on validation set** (runs in background — eval takes 20-40+ min on M2 Pro):
   ```bash
   # MUST use run_in_background=true — evaluation exceeds the 10-minute Bash timeout.
   cd /Users/jyan/src/my-own-plate && .venv/bin/python training/evaluate.py --mode val
   ```
   Use `run_in_background=true` on the Bash tool call. Once notified of completion, read the summary JSON at `training/eval_results_hf/eval_summary_val.json` for precise numbers.

6. **Record results** in `training/results.tsv` (tab-separated):
   ```
   experiment	cal_mae_pct	protein_mae_pct	fat_mae_pct	carbs_mae_pct	avg_mae_pct	parse_failures	status	description
   ```
   The `status` column must be one of: `baseline`, `kept`, or `reverted`.

7. **Update `training/notes.md`**: Append an entry to the "Experiment log" with the hypothesis, result, and insights. Do this for every experiment — successes and failures both contain useful information.
   - **Mark kept/reverted clearly** in each entry heading (e.g., "### Exp 12: ... — KEPT" or "— REVERTED").
   - **Update the "Ideas queue"**: Add new ideas sparked by this experiment. Remove ideas you just tried.

8. **Keep/discard decision**:
   - If the experiment meets the success criteria (see above): keep the commit.
   - If the experiment does NOT meet the criteria: `git revert HEAD --no-edit` to create a revert commit, preserving history.

9. Go back to step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the evaluation output, analyze failure patterns, try combining near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

## What to try next

Priority order (based on MLX findings):

1. **Run 10 epochs with current defaults** — Exp 2 used 12 epochs and got 27.8%. Try 10 epochs to confirm the baseline matches MLX (~27.4%) and to verify 10 is indeed better than 12 for HF too.

2. **Vision block subset: top-12 blocks** — In MLX, adapting only the top-12 (deepest) vision blocks instead of all 24 was slightly better (21.3% vs 21.8% for top-8 vs all 24 at r16; then r32 on top-12 got 19.5%). This requires modifying `apply_vision_block_lora` in `hf_utils.py` to accept a `num_blocks` parameter. Apply LoRA only to the last N blocks of `vision.blocks`.

3. **Vision block subset: top-8 blocks** — Continue the exploration if top-12 helps.

4. **Vision rank sweep on selected blocks** — Once the best block count is found, try r64 (MLX found r64 on top-12 slightly better than r32: 19.0% vs 19.5%).

5. **Verify GGUF export quality** — After achieving a good val MAE%, run `merge_and_export.py` and test the GGUF with `eval_gguf_server.py` / `compare_hf_gguf.py` to confirm the HF→GGUF path doesn't degrade accuracy.

## Important notes

- **Ingredient cap**: Completions are capped at 5 ingredients (set in `data/prepare_nutrition5k.py`). If you change this, re-run the full data pipeline: `cd data && python prepare_nutrition5k.py && cd ../training && python convert_dataset.py`.
- Training changes go in `training/train.py` or `training/hf_utils.py`. Dataset format changes go in `data/prepare_nutrition5k.py` or `training/convert_dataset.py` (and require re-running the data pipeline).
- The PEFT adapter is saved to `training/output/adapter/`. The custom vision LoRA is saved to `training/output/vision_lora.pt`. Both are overwritten each training run.
- Do NOT evaluate on the test set (`--mode test`). Use only `--mode val` during experimentation.
- Training runs on Apple Silicon MPS. Keep `--batch-size 1` to avoid Metal GPU timeouts.
- The `autonomous_loop.py` script in `training/` is a batch runner from a previous approach. It is kept for reference but is not used — this `program.md` drives the experiment loop via Claude Code.
