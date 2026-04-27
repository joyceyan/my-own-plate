# Fine-Tuning Optimization

## Goal

Minimize the MAE% (mean absolute error as percentage of ground-truth mean) of the fine-tuned Qwen3-VL-2B model on the Nutrition5k validation set across four nutrients: calories, protein, fat, and carbs. The target is to approach the Thames et al. CVPR 2021 N5k RGB baselines (calories 26.1%, protein 29.5%, fat 34.2%, carbs 31.9%).

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely.

1. **Read the codebase**: Read these files for full context:
   - `training/train.py` — LoRA fine-tuning script. **This is the primary file you edit.**
   - `training/evaluate.py` — evaluation script that computes MAE/MAE% metrics.
   - `data/nutrition5k_baseline_data.json` — Thames et al. CVPR 2021 baseline values.

2. **Check for existing progress**: Read `training/notes.md` and `training/results.tsv` to review what's been tried and what to try next.

3. **Verify the environment** by running a quick sanity check:
   ```bash
   cd /Users/jyan/src/my-own-plate/training && python -c "import mlx_vlm; print('mlx-vlm OK')"
   ```

## Baseline

Current fine-tuned model (LoRA rank 16, alpha 1.0, lr 1e-5, 3 epochs, attention-only):

| Nutrient  | MAE%  | N5k Baseline |
|-----------|-------|--------------|
| Calories  | 59.6% | 26.1%        |
| Protein   | 54.5% | 29.5%        |
| Fat       | 74.3% | 34.2%        |
| Carbs     | 90.0% | 31.9%        |
| **Avg**   | **69.6%** | **30.4%** |

Parse failures: 6 / 349 samples.

## Success criteria

An experiment is **successful** if BOTH conditions are met:
1. The **average MAE%** across all four nutrients decreases (lower is better).
2. **No individual nutrient's MAE%** increases by more than 5 percentage points compared to the best-so-far values.

Example: if the best-so-far calories MAE% is 59.6%, a new experiment with calories MAE% of 64.5% (within 5pp) is acceptable if the average improves. But 64.7% (>5pp worse) means the experiment is rejected even if the average improved.

Parse failures also matter: if parse failures increase significantly (>2x), the experiment should be rejected regardless of MAE% improvement, as it indicates the model's output format is degrading.

## Research methodology

- **One variable at a time**: Change one hyperparameter or technique per experiment. If a complex change improves results, ablate to find the essential ingredient.
- **Diminishing returns**: If the last 5 experiments were all minor tweaks with tiny deltas (<0.5pp average), change strategy entirely.
- **Long runs are fine**: Individual training runs may take up to 24 hours on M2 Pro. Evaluation takes 20-40+ min. Longer experiments (more epochs, larger images, bigger models) are acceptable — the human expects this. Do not cut corners or reduce epochs to save time.
- **Background execution required**: Both training and evaluation MUST be run with `run_in_background=true` on the Bash tool, since they exceed the 10-minute Bash timeout. Wait for the background notification — do not poll or sleep.
- **Evaluation**: Always evaluate with `--no-base` flag to skip the slow base model comparison (we already have the base model numbers). Use `--mode val` (the default) — never evaluate on the test set during experimentation.

## The experiment loop

The experiment runs on `main`. All kept experiments are committed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `training/notes.md` and `training/results.tsv` to review what's been tried, what worked, and what to try next.

2. **Design the next experiment** based on insights from the notes. Check the "Ideas queue" section of `training/notes.md` first. Make changes to `training/train.py` (or rarely `training/evaluate.py` if the evaluation itself needs fixing).

3. `git add training/ && git commit -m "exp N: description of experiment"`

4. **Train the model** (runs in background — training takes 45-90+ min on M2 Pro):
   ```bash
   # MUST use run_in_background=true — training exceeds the 10-minute Bash timeout.
   cd /Users/jyan/src/my-own-plate/training && python train.py --train-data ~/src/my-own-plate/data/nutrition5k_hf
   ```
   Use `run_in_background=true` on the Bash tool call. You will be notified when training completes. **Do NOT poll or sleep** — just wait for the background task notification.
   If training crashes or produces NaN losses, fix the issue or revert and try something else.

5. **Evaluate on validation set** (runs in background — eval takes 20-40+ min on M2 Pro):
   ```bash
   # MUST use run_in_background=true — evaluation exceeds the 10-minute Bash timeout.
   cd /Users/jyan/src/my-own-plate/training && python evaluate.py --adapter-path ./output/adapters --no-base
   ```
   Use `run_in_background=true` on the Bash tool call. Once notified of completion, read the summary JSON at `training/eval_results/eval_summary.json` for precise numbers.

6. **Record results** in `training/results.tsv` (tab-separated):
   ```
   experiment	cal_mae_pct	protein_mae_pct	fat_mae_pct	carbs_mae_pct	avg_mae_pct	parse_failures	status	description
   ```
   The `status` column must be one of: `baseline`, `kept`, or `reverted`.

7. **Update `training/notes.md`**: Append an entry to the "Experiment log" with the hypothesis, result, and insights. Do this for every experiment — successes and failures both contain useful information.
   - **Mark kept/reverted clearly** in each entry heading (e.g., "### Exp 2: ... — KEPT" or "— REVERTED").
   - **Update the "Ideas queue"**: Add new ideas sparked by this experiment. Remove ideas you just tried.

8. **Keep/discard decision**:
   - If the experiment meets the success criteria (see above): keep the commit.
   - If the experiment does NOT meet the criteria: `git reset --hard HEAD~1` to revert, and also reset the adapter weights:
     ```bash
     git checkout -- training/output/adapters/
     ```
     (Or simply let the next training run overwrite them.)

9. Go back to step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the evaluation output, analyze failure patterns, try combining near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

## What to try

The model is currently far from the N5k baselines (69.6% avg vs 30.4%), so there is significant room for improvement. Possible directions:

- **Hyperparameters**: Learning rate schedules (warmup, cosine decay), different LR values, more epochs, LoRA rank/alpha tuning.
- **LoRA targets**: Expand LoRA to MLP layers (carefully — previous attempt caused overfitting), or try different layer selections.
- **Training strategy**: Curriculum learning, loss weighting, data augmentation via prompt variation.
- **Prompt engineering**: Modify the prompt template in the dataset to give the model better structure.
- **Image processing**: Different resize dimensions (current: 384x384), aspect ratio preservation.
- **Regularization**: Dropout in LoRA layers, weight decay, early stopping based on val loss.
- **Optimizer**: Try AdamW with weight decay, or different beta values.

## Important notes

- **Ingredient cap**: Completions are capped at 5 ingredients (set in `data/prepare_nutrition5k.py`). If you change this, re-run the full data pipeline: `cd data && python prepare_nutrition5k.py && cd ../training && python convert_dataset.py`.
- Training changes go in `training/train.py`. Dataset format changes go in `data/prepare_nutrition5k.py` or `training/convert_dataset.py` (and require re-running the data pipeline).
- The adapter checkpoint at `training/output/adapters/adapters.safetensors` is overwritten each training run.
- Do NOT evaluate on the test set (`--mode test`). Use only `--mode val` during experimentation.
- Training is Apple Silicon only (MLX). Do not attempt to use CUDA or PyTorch training.
- Keep `--batch-size 1` and `--grad-checkpoint` to avoid Metal GPU timeouts on laptops.
