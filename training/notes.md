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

**Priority 1 — likely high impact:**
- LR warmup + cosine decay — constant LR at any value >1e-5 causes degenerate repetition. Warmup avoids early destabilization, cosine decay prevents late-stage overfitting. Try peak LR 5e-5 with 10% warmup.
- More epochs (5-10) at lr=1e-5 — model may simply need more gentle training time
- Lower LoRA alpha (0.5) — reduce adapter strength to allow higher LR without overwhelming base model

**Priority 2 — moderate expected impact:**
- LoRA rank increase (32 or 64) — more capacity for the adapter
- LoRA alpha tuning — try alpha=2.0 with rank=16 for stronger adapter signal
- Weight decay via AdamW (1e-2 or 1e-1)
- Image resize to 512x512 or native aspect ratio — more visual detail

**Priority 3 — speculative:**
- Expand LoRA to gate_proj/up_proj/down_proj (MLP) — previous attempt caused overfitting, but with proper regularization may help
- Prompt template changes — add explicit instructions about output format or nutrient ranges
- Data augmentation — random crops, color jitter during training
- Gradient accumulation to simulate larger effective batch size
- Mixed precision / different sequence lengths

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

### Data change: Cap ingredients at 5 (pre-exp 3)

The repetitive ingredient loop failure in exps 1-2 motivated capping the ingredient list at 5 items in `prepare_nutrition5k.py`. Median dish has 4 ingredients; the long tail (up to 34) creates variable-length output that the model can get stuck looping on. Capping at 5 keeps the output short and focused. **Requires re-running the data pipeline** (`prepare_nutrition5k.py` then `convert_dataset.py`) before the next training run. The baseline numbers (exp 0) were trained on uncapped data, so the next experiment will establish a new baseline with the capped dataset.
