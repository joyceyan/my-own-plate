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

**Current best: exp 23 — rank 32, alpha 1.0, LLM attn+MLP + VL merger, Adam lr=1e-5, 10 epochs = 59.6% avg.**

VL merger LoRA broke the exp 16 plateau. Full vision tower LoRA (exp 22) was too aggressive (prompt echoing), but targeted merger-only adaptation works.

### Key findings (settled — do not re-explore):
- **Modules**: attn+MLP (7 modules) >> attn-only (4 modules). MLP needed for numerical regression.
- **Rank**: 32 > 16. More capacity per module helps.
- **Alpha**: 1.0 is the only working value. 2.0 = corruption, 0.5 = under-adapts.
- **Dropout**: 0.0 only. 0.05 = corruption.
- **Epochs**: 10 for attn+MLP. 20+ overfits with MLP layers.
- **LR**: 1e-5 only. Higher = format corruption, lower (5e-6) = fat collapse.
- **Optimizer**: Adam only. AdamW/Muon both failed.
- **Data**: Float nutrients > integer. Nutrients-only > ingredients+nutrients.

### Next experiments:
- **Vision tower top 4 blocks only** — exp 22 (all 24 blocks) failed, but top 4 blocks might work. Fewer LoRA layers = less risk of prompt echoing.
- **Chain-of-thought completion** — add food description before numbers: `{"food": "grilled chicken with rice", "calories": 350, ...}`. Reasoning step may improve prediction.
- **Analyze worst predictions** — find systematic failure modes to target
- **Investigate train/eval image pipeline mismatch** — P1 blocker

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

### Exp 6: Nutrients-only completion, lr=1e-5, 5 epochs — KEPT

**Hypothesis**: The 0.242 loss plateau is because ~50% of completion tokens are ingredient names. Removing ingredients focuses 100% of the gradient signal on nutrient values.

**Data change**: Removed ingredients from completion format. New completion: `{"calories": X, "protein": Y, "fat": Z, "carbs": W}`. New prompt: "Estimate the nutritional content of this food image. Respond as JSON with keys: calories (kcal), protein (g), fat (g), carbs (g)." Re-ran full data pipeline.

**Result**: 62.5/64.1/68.1/71.8 = **66.6% avg** — best so far! Fat improved 6.2pp (74.3→68.1%), carbs improved **18.2pp** (90.0→71.8%). Zero parse failures (vs 6 in exp 0). But protein degraded 9.6pp (54.5→64.1%) and calories slightly worse (+2.9pp).

**Training notes**: Val loss still converges to 0.242 despite the shorter completion — the plateau is inherent to the optimization, not the format. Training took 45.9 min. Eval took only 10.2 min (shorter outputs).

**Insight**: Removing ingredients was the most impactful single change so far. Fat and carbs dramatically improved, suggesting the model was spending LoRA capacity on ingredient token prediction instead of nutrient regression. The protein degradation needs investigation — possibly the model traded protein accuracy for fat/carbs. Zero parse failures confirms the shorter format is much more robust. **This is the new baseline for all future experiments.**

### Exp 7: Warmup+cosine peak 5e-5, nutrients-only, 5 epochs — REVERTED

**Result**: 63.2/57.6/74.0/81.7 = 69.1% avg. Protein better (-6.5pp) but fat +5.9pp and carbs +9.9pp regressed.

**Insight**: Higher LR shifts learning focus between nutrients. 5 epochs is too short at any LR.

### Exp 8: 20 epochs, AdamW wd=0.01, warmup+cosine lr=1e-5 — REVERTED

**Hypothesis**: Longer training (20 epochs, ~3 hrs) + weight decay regularization should improve predictions.

**Result**: 68.6/60.0/74.5/76.5 = **69.9% avg** — worse than exp 6 (66.6%). Despite 4x more training.

**Insight**: **Weight decay on LoRA adapters is counterproductive.** LoRA weights initialize at zero (no adaptation). Weight decay pushes them back toward zero, actively undoing fine-tuning. Combined with cosine LR decay to near-zero, the model lost most of its adaptation by epoch 20. **Never use weight decay on LoRA.** Next: try 20 epochs with plain Adam, constant LR.

### Exp 9: 20 epochs, Adam, constant lr=1e-5, nutrients-only — KEPT

**Hypothesis**: Pure longer training (4x exp 6's 5 epochs) with the same Adam optimizer should help.

**Result**: 58.6/68.6/63.5/71.4 = **65.5% avg** — new best! Calories -3.9pp, fat -4.6pp, carbs -0.4pp improved vs exp 6. Protein +4.5pp worse (within 5pp threshold). Parse failures: 2 (minimal). Training took 179 min (~3 hrs).

**Insight**: Longer training definitively helps — 20 epochs is better than 5 at the same LR. The model continues to learn beyond the 0.242 val loss plateau (MAE% improves even though loss doesn't change). Protein is the persistent weak point; it trades off with fat and calories. Next: try even longer (40 epochs), or try Muon optimizer, or increase rank.

### Exp 10: 40 epochs, Adam, constant lr=1e-5 — REVERTED

**Result**: Catastrophic forgetting. CSS garbage output, 66% parse failure. 20 epochs is the ceiling for constant LR.

### Exp 11: Muon optimizer, lr=1e-3, 20 epochs — REVERTED

**Result**: UnicodeDecodeError — weights corrupted. Muon lr=1e-3 too aggressive for LoRA.

### Exp 12: P1 native resolution 640x480, 20 epochs — REVERTED

**Hypothesis**: 384x384 loses 52% of pixels and destroys 4:3 aspect ratio. Native 640x480 should give 2x more visual tokens.

**Result**: 66.5/78.9/99.6/89.0 = **83.5% avg** — dramatically worse than exp 9 (65.5%). 11 parse failures. Training speed and memory were IDENTICAL to 384x384, suggesting VisionDataset's image_resize_shape doesn't actually change what the vision encoder receives.

**Insight**: The image pipeline needs deeper investigation before resolution experiments. VisionDataset may pre-resize to 384x384, and mlx_vlm.generate() (used at eval) uses the processor's dynamic resolution. There may be a persistent train/eval resolution mismatch. P1 experiments are BLOCKED until the pipeline is understood. Moving to P2 (module selection).

### Exp 13: P2 attn+MLP LoRA, 20 epochs — REVERTED

**Result**: IndexError during eval (invalid token IDs) + 43% parse failure when errors caught. MLP layers overfit at 20 epochs.

### Exp 14: P2 attn+MLP LoRA, 10 epochs — KEPT

**Config**: LoRA on q/k/v/o_proj + gate/up/down_proj (7 modules), rank 16, alpha 1.0, Adam lr=1e-5, 10 epochs. ~1.75x more trainable params than attn-only.

**Result**: 59.1/58.6/72.2/62.0 = **63.0% avg** — new best! Protein improved 10pp (68.6→58.6%), carbs improved 9.4pp (71.4→62.0%). Fat regressed 8.7pp (63.5→72.2%). Zero parse failures. Training took 109 min.

**Parser note**: Model outputs `"kcal"` as key instead of `"calories"`. Added alias-aware parser (NUTRIENT_ALIASES dict) to handle this and other format variations (nested JSON, unit-suffixed keys). This parser improvement is critical for future experiments.

**Insight**: MLP layers dramatically improve protein and carbs prediction — confirms the hypothesis that attention-only LoRA lacks the computational capacity for numerical regression. Fat regression suggests a trade-off between nutrients with the current capacity (rank 16 may be too low for 7 modules). Next: try higher rank (32) with attn+MLP to give the model more capacity per module. Also need to address fat — possibly the MLP layers are learning a different feature space that's better for protein/carbs but worse for fat.

### Exp 15: 100 epochs cosine decay, attn+MLP — ABANDONED

Disk full at ~50 epochs. 277 checkpoints * 70MB = 18GB consumed all free space. Abandoned; epochs locked to ~10.

### Exp 16: rank 32 with attn+MLP, 10 epochs — KEPT

**Config**: LoRA rank 32 (2x exp 14), alpha 1.0, 7 modules (attn+MLP), Adam lr=1e-5, 10 epochs. Peak mem 4.84 GB, 113 min training.

**Result**: 55.8/59.4/68.4/56.4 = **60.0% avg** — new best! All nutrients improved vs exp 14: calories -3.3pp, fat -3.8pp, carbs -5.6pp. Protein +0.8pp (within threshold). Zero parse failures.

**Insight**: Rank 32 gives a clean 3pp average improvement. The extra capacity per module helps all nutrients, and partially recovers the fat regression from exp 14. Carbs is now the best nutrient at 56.4% (was 90% at exp 0!). Next: try rank 32 with dropout 0.05 for regularization, or try alpha tuning.

### Data change: Cap ingredients at 5 (pre-exp 3)

The repetitive ingredient loop failure in exps 1-2 motivated capping the ingredient list at 5 items in `prepare_nutrition5k.py`. Median dish has 4 ingredients; the long tail (up to 34) creates variable-length output that the model can get stuck looping on. Capping at 5 keeps the output short and focused.
