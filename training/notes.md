# Lab Notebook — Fine-Tuning Optimization

## Setup

Model: Qwen3-VL-2B-Instruct-bf16 (mlx-community) fine-tuned with LoRA via mlx-vlm.
Dataset: Nutrition5k (80/10/10 train/val/test split by dish ID).
Task: Predict calories, protein, fat, carbs from food images.
Hardware: Apple Silicon (M2 Pro 32GB).

## Baseline

Exp 0: LoRA rank 16, alpha 1.0, lr 1e-5, 3 epochs, attention-only (q/k/v/o_proj), Adam optimizer, batch size 1, image resize 384x384, max seq length 2048.

| Nutrient  | MAE%  | MAE (abs) | N5k Target |
|-----------|-------|-----------|------------|
| Calories  | 59.6% | 138.2 kcal| 26.1%      |
| Protein   | 54.5% | 8.6 g     | 29.5%      |
| Fat       | 74.3% | 8.9 g     | 34.2%      |
| Carbs     | 90.0% | 15.7 g    | 31.9%      |
| **Avg**   | **69.6%** |       | **30.4%**  |

Parse failures: 6 / 349 (1.7%)

Key observations:
- The model is 2-3x worse than the N5k RGB baselines across all nutrients.
- Carbs and fat are the weakest (90% and 74% MAE%).
- The base model (no LoRA) scores very similarly (63.7%, 54.9%, 74.2%, 90.2% — avg 70.8%), meaning the current fine-tuning provides only marginal improvement (~1.2pp average).
- Parse failures are low (6/349), so the model is generating valid JSON most of the time.

## Ideas queue

**Priority 1 — address the 0.242 loss plateau (fundamental):**
- **Remove ingredients from completion** — the loss is ~50% spent learning ingredient names, diluting the gradient signal for nutrient values. Make the completion just `{"calories": X, "protein": Y, "fat": Z, "carbs": W}`. This concentrates 100% of the learning on the task we care about. Requires data pipeline change.
- AdamW with weight decay 0.01 — regularize to reduce the systematic overestimation bias seen in exp 4

**Priority 2 — combine what works:**
- Warmup + cosine decay at peak 1e-5 — use schedule from exp 4 (which fixed format) at safe LR
- Rank 64 + regularization — rank 64 improved calories (exp 5) but hurt format; add weight decay or lower alpha to compensate
- LoRA on MLP layers — attention-only may lack capacity for numerical regression

**Priority 3 — speculative:**
- Image resize to 512x512 — more visual detail
- Gradient accumulation (effective batch 4-8)
- Data augmentation

## Experiment log

### Exp 0: Baseline — BASELINE

See baseline section above. This is the starting point for all comparisons.

### Exp 1: Learning rate 1e-4 — REVERTED

**Hypothesis**: lr=1e-5 is too conservative (only ~1.2pp improvement over base). 10x increase to 1e-4 should allow meaningful learning.

**Result**: Complete failure. 20/20 parse failures — model gets stuck in repetitive ingredient loops (e.g., "chicken", "chicken", "chicken"... or "dill", "capers", "dill", "capers"...) and never reaches the nutrient values. The model learned to generate ingredient lists but the high LR destroyed its ability to produce structured JSON output.

**Training notes**: Loss dropped very fast from 1.68 to ~0.24 in first 20 steps, then plateaued. Final val loss 0.242. Training completed in 28 min.

**Insight**: 1e-4 is too aggressive for this model/task. The fast loss drop suggests the model is memorizing training patterns but losing generalization. Try 5e-5 as a middle ground, or combine moderate LR with warmup to avoid early destabilization.

### Exp 2: Learning rate 5e-5 — REVERTED

**Hypothesis**: 5e-5 is a middle ground between the too-conservative 1e-5 and the too-aggressive 1e-4.

**Result**: Same failure as exp 1 — 20/20 parse failures with identical repetitive ingredient loop patterns. Val loss converged to 0.242 (same as exp 1).

**Insight**: The LR sweet spot between "barely learning" (1e-5) and "degenerate output" (5e-5+) is extremely narrow, if it exists at all. The fundamental issue may be that at any LR high enough to learn, the LoRA adapter overwhelms the base model's generation structure. Next steps: try lr=2e-5 (minimal increase), or add LR warmup + cosine decay to prevent early destabilization, or try lower LoRA alpha (0.5) to reduce adapter strength.

### Exp 3: Learning rate 2e-5 constant — REVERTED

**Hypothesis**: 2e-5 is the minimum LR increase. With ingredient cap, should avoid degenerate loops.

**Result**: 20 parse failures in 120 samples (17% rate), eval aborted. Ingredient loops still occur.

**Insight**: Even 2e-5 constant LR causes format corruption from the early LR shock.

### Exp 4: Warmup + cosine decay, peak 5e-5, 5 epochs — REVERTED

**Hypothesis**: Warmup avoids early LR shock; cosine decay prevents late overfitting.

**Result**: Format fixed (4 parse failures) but predictions worse: 62.5/53.0/75.5/97.8 = **72.2% avg** (baseline 69.6%). Carbs +7.8pp. Model systematically overestimates and has compressed prediction ranges (correlation 0.44-0.71 with ground truth).

**Insight**: Warmup works for format stability. But 5e-5 peak is too aggressive for prediction quality. The 0.242 loss plateau is not breakable with LR changes.

### Exp 5: LoRA rank 64, lr=1e-5, 5 epochs — REVERTED

**Hypothesis**: The 0.242 loss plateau is a capacity limitation with rank 16.

**Result**: 54.9/54.5/73.5/95.3 = **69.6% avg** — same as baseline! Calories improved 4.7pp but carbs degraded 5.3pp. Parse failures increased to 24 (vs 6 baseline). Same 0.242 loss plateau.

**Insight**: The 0.242 plateau is NOT a capacity limitation — rank 64 (4x params) doesn't break it. The plateau likely reflects that ~50% of the completion tokens are ingredient names/JSON structure, diluting the gradient signal for nutrient values. The model efficiently learns the format tokens but has insufficient learning pressure on the actual numbers. **Next: remove ingredients from completion to focus 100% of loss on nutrient prediction.**

### Data change: Cap ingredients at 5 (pre-exp 3)

The repetitive ingredient loop failure in exps 1-2 motivated capping the ingredient list at 5 items in `prepare_nutrition5k.py`. Median dish has 4 ingredients; the long tail (up to 34) creates variable-length output that the model can get stuck looping on. Capping at 5 keeps the output short and focused. **Requires re-running the data pipeline** (`prepare_nutrition5k.py` then `convert_dataset.py`) before the next training run. The baseline numbers (exp 0) were trained on uncapped data, so the next experiment will establish a new baseline with the capped dataset.
