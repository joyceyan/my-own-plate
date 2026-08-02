# Lab Notebook — HF Fine-Tuning Optimization

## Setup

Model: Qwen3-VL-2B-Instruct fine-tuned with HuggingFace transformers + PEFT + manual PyTorch LoRA.
Dataset: Nutrition5k (80/10/10 train/val/test split by dish ID).
Task: Predict calories, protein, fat, carbs from food images.
Hardware: Apple Silicon (M2 Pro 32GB).

Legacy MLX experiments are preserved in `training/mlx/` (see `training/mlx/notes.md` and `training/mlx/results.tsv`).

## Baseline

Exp 0: LLM LoRA rank 64, alpha 64 on q/k/v/o/gate/up/down_proj; projector LoRA rank 64; all 24 vision blocks LoRA rank 32, alpha 32. Dropout 0.0. Learning rate 1e-5 → 1e-6 cosine. 10 epochs. Batch size 1. Image resize via `min_pixels=max_pixels=147456` (~520 merged patches for 4:3 images). Custom `LoRALinear` implementation for vision/projector.

| Nutrient  | MAE%  | N5k Target | MLX Best (18.1%) |
|-----------|-------|------------|------------------|
| Calories  | 35.4% | 26.1%      | 15.0%            |
| Protein   | 46.6% | 29.5%      | 16.9%            |
| Fat       | 86.5% | 34.2%      | 22.3%            |
| Carbs     | 55.0% | 31.9%      | 18.3%            |
| **Avg**   | **55.9%** | **30.4%** | **18.1%**     |

Parse failures: 1 / 349 (0.3%)

## Current best (Exp 14, 26.2% avg)

10 epochs, vision LoRA r64/alpha64 on all 24 blocks. LLM r64. Projector r64.

| Nutrient  | MAE%  |
|-----------|-------|
| Calories  | 17.6% |
| Protein   | 25.4% |
| Fat       | 39.6% |
| Carbs     | 22.2% |
| **Avg**   | **26.2%** |

## Strategic notes

**The HF and MLX pipelines are near-equivalent at baseline config.** Exp 2 (HF, 12ep) = 27.8% vs MLX exp 30 (10ep) = 27.4%. The 9.7pp gap to MLX best (18.1%) was closed entirely through config changes (vision block selection, cosine LR, rank tuning) — not pipeline differences. The same strategies should close the gap in HF.

**The autonomous_loop.py batch runner is deprecated.** It ran experiments from a static queue with no adaptive reasoning. Future experiments are driven by Claude Code via program.md.

## Ideas queue

### Settled (do NOT re-explore)

- ~~MLX-equivalent config in HF~~: Done as Exp 0; gradient checkpointing bug fixed in Exp 1.
- ~~LLM rank~~: 64 is optimal. Rank 32 worse (Exp 5).
- ~~Dropout~~: 0.0 only. 0.05 hurts (Exp 6).
- ~~LLM LoRA targets~~: Full attn+MLP required. Attention-only much worse (Exp 11).
- ~~Constant LR~~: Cosine 1e-5→1e-6 is better (Exp 10).
- ~~Vision rank 16~~: Too low (Exp 8). Default r32 is minimum.
- ~~3-epoch screens~~: Unreliable. 5 consecutive reverts (Exps 5,6,8,10,11). Use full 10-epoch runs only.
- ~~Image size 448+~~: Too slow / OOM on M2 Pro (Exp 4).
- ~~Gradient checkpointing~~: Must stay OFF — blocks custom vision LoRA gradients.

### Next experiments (priority order)

1. ~~**10-epoch baseline confirmation**~~: Done (Exp 12). 10 epochs = 27.2% avg, confirming MLX parity and slight improvement over 12 epochs.

2. ~~**Vision block subset: top-12 blocks**~~: Tried in Exp 13, significantly worse (+5.8pp avg). The HF pipeline benefits from all 24 blocks, unlike MLX.

3. ~~**Vision block subset: top-8 blocks**~~: Skipped — top-12 was already worse, top-8 would be worse still.

4. ~~**Vision rank 64 on all 24 blocks**~~: Done (Exp 14). Improved avg by 1.0pp (27.2% → 26.2%). Kept.

5. **Vision rank 128 on all 24 blocks**: If r64 helped, r128 may help further.

6. **Learning rate exploration**: Try higher peak LR (2e-5) or lower min LR (5e-7) with cosine decay.

7. **LLM rank 128**: If vision capacity helps, LLM capacity might too.

5. **Verify GGUF export**: After achieving good val MAE%, run `merge_and_export.py` and test GGUF with `compare_hf_gguf.py` to confirm no degradation.

### Deferred / speculative

- Image resolution / preprocessing alignment between train and deploy paths.
- Chat template verification (HF vs llama.cpp tokenization).
- Quantization sweep (F16 vs Q4_K_M vs Q8_0) — only after export path is verified.
- Gradient checkpointing fix for custom LoRA (would allow larger batch/resolution).

## Experiment log

### Exp 0: Baseline — BASELINE

See baseline section above. This is the starting point for all comparisons.

Config: LLM r64 (attn+MLP) + projector r64 + all 24 vision blocks r32, alpha=rank, dropout 0.0, lr 1e-5→1e-6 cosine, 10 epochs. Custom `LoRALinear` for vision/projector; PEFT for LLM.

Result: 35.4/46.6/86.5/55.0 = **55.9% avg**, 1 parse failure.

**Insight**: The HF pipeline trains but performs much worse than the legacy MLX result (18.1%). Two hypotheses are most likely: (1) the custom vision LoRA is not training correctly, and/or (2) the image preprocessing/chat-template path differs from MLX in a way that degrades learning. The next experiments should isolate these.

### Exp 1: Disable gradient checkpointing — KEPT

**Hypothesis**: Gradient checkpointing was preventing gradients from reaching the custom vision LoRA matrices, leaving the vision tower effectively frozen and forcing the LLM/projector LoRA to compensate.

**Change**: Set `--grad-checkpoint` default to `False` in `training/train.py`. All other hyperparameters identical to Exp 0.

**Result**: 19.7/30.3/50.0/26.1 = **31.5% avg**, 0 parse failures.

**Comparison vs Exp 0**:
- Calories: 35.4% → 19.7% (−15.7pp)
- Protein: 46.6% → 30.3% (−16.3pp)
- Fat: 86.5% → 50.0% (−36.5pp)
- Carbs: 55.0% → 26.1% (−28.9pp)
- Avg: 55.9% → 31.5% (−24.4pp)

**Insight**: Disabling gradient checkpointing was the critical fix. The custom vision LoRA clearly trained this time, and all nutrients improved dramatically. The result is now near the N5k RGB baseline (30.4%) but still 13.4pp behind the legacy MLX result (18.1%). Fat remains the hardest nutrient. Next experiments should focus on closing the remaining gap: longer training, vision/LLM rank, image resolution, and LoRA target selection.


### Exp 2: 12 epochs to test convergence — KEPT

**Params**: {"epochs": 12}

**Result**: 18.5/28.6/39.1/25.1 = **27.8% avg**, 0 parse failures.

**Comparison vs best-so-far**:
- calories: 19.7% → 18.5% (-1.2pp)
- protein: 30.3% → 28.6% (-1.7pp)
- fat: 50.0% → 39.1% (-10.9pp)
- carbs: 26.1% → 25.1% (-1.0pp)
- avg: 31.5% → 27.8% (-3.7pp)

**Insight**: TBD — update manually if needed.


### Exp 3: vision LoRA rank 64 (3 epoch screen) — KEPT

**Params**: {"epochs": 3, "lora_rank_vision": 64, "lora_alpha_vision": 64}

**Result**: 23.8/32.4/45.0/33.1 = **33.6% avg**, 0 parse failures.

**Insight**: Provisional 3-epoch fast baseline established by the loop. The initial automated evaluation was skipped due to a stale-summary bug in `autonomous_loop.py`, which was fixed; this result comes from a manual re-evaluation of the trained Exp 3 weights. Future 3-epoch screens will be compared against this baseline.

### Exp 4: image size 448 (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "image_size": 448}

**Result**: Aborted after ~7 hours (~40% through epoch 1.1). Training became orders of magnitude slower than the 384 default due to memory pressure / much larger vision-token sequences. System load was >16 and the process was frequently in uninterruptible sleep.

**Insight**: 448×448 image preprocessing is too expensive on this machine for the current pipeline. Larger resolution screens are deferred until memory headroom is available or the preprocessing is made more efficient.

### Exp 5: LLM LoRA rank 32 (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "lora_rank_llm": 32, "lora_alpha_llm": 32}

**Result**: 25.8/41.3/53.0/36.1 = **39.0% avg**, 0 parse failures.

**Comparison vs fast baseline (Exp 3)**:
- calories: 23.8% → 25.8% (+2.0pp)
- protein: 32.4% → 41.3% (+8.9pp)
- fat: 45.0% → 53.0% (+8.0pp)
- carbs: 33.1% → 36.1% (+3.0pp)
- avg: 33.6% → 39.0% (+5.4pp)

**Insight**: Lowering LLM LoRA rank from 64 to 32 hurt every nutrient, especially protein and fat. The default rank 64 appears important for this task.


### Exp 6: dropout 0.05 (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "lora_dropout": 0.05}

**Result**: 25.8/37.6/48.8/38.2 = **37.6% avg**, 0 parse failures.

**Comparison vs best-so-far**:
- calories: 23.8% → 25.8% (+2.0pp)
- protein: 32.4% → 37.6% (+5.2pp)
- fat: 45.0% → 48.8% (+3.8pp)
- carbs: 33.1% → 38.2% (+5.1pp)
- avg: 33.6% → 37.6% (+4.0pp)

**Insight**: Adding dropout 0.05 degraded every nutrient versus the fast baseline. For this small dataset and short 3-epoch run, regularization appears harmful; the zero-dropout default is preferred.

**Loop fix**: Removed `git reset --hard HEAD~1` from the revert path. That reset was rolling back `autonomous_status.json`/`experiment_queue.json`, causing reverted experiments to be re-queued and run repeatedly. Reverts now commit the reverted state directly so the loop advances to the next experiment.


### Exp 8: vision LoRA rank 16 (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "lora_rank_vision": 16, "lora_alpha_vision": 16}

**Result**: 24.7/34.7/47.2/33.9 = **35.1% avg**, 0 parse failures.

**Comparison vs best-so-far**:
- calories: 23.8% → 24.7% (+0.9pp)
- protein: 32.4% → 34.7% (+2.3pp)
- fat: 45.0% → 47.2% (+2.2pp)
- carbs: 33.1% → 33.9% (+0.8pp)
- avg: 33.6% → 35.1% (+1.5pp)

**Insight**: Reducing vision LoRA rank from 64 to 16 degraded every nutrient versus the fast baseline. Vision capacity clearly matters; rank 16 is not enough. We still have not tested a fair vision rank 32 screen in this pipeline (Exp 0's rank 32 was confounded by the gradient-checkpointing bug).


### Exp 10: constant LR 1e-5 (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "min_lr": 1e-05}

**Result**: 25.4/32.5/46.2/36.8 = **35.2% avg**, 0 parse failures.

**Comparison vs best-so-far**:
- calories: 23.8% → 25.4% (+1.6pp)
- protein: 32.4% → 32.5% (+0.1pp)
- fat: 45.0% → 46.2% (+1.2pp)
- carbs: 33.1% → 36.8% (+3.7pp)
- avg: 33.6% → 35.2% (+1.6pp)

**Insight**: A constant LR of 1e-5 performed worse than the cosine 1e-5→1e-6 schedule, especially on carbs. The lower minimum LR and cosine decay appear beneficial for this short run.


### Exp 11: LLM attention-only LoRA targets (3 epoch screen) — REVERTED

**Params**: {"epochs": 3, "llm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}

**Result**: 26.6/36.1/54.4/37.4 = **38.6% avg**, 0 parse failures.

**Comparison vs best-so-far**:
- calories: 23.8% → 26.6% (+2.8pp)
- protein: 32.4% → 36.1% (+3.7pp)
- fat: 45.0% → 54.4% (+9.4pp)
- carbs: 33.1% → 37.4% (+4.3pp)
- avg: 33.6% → 38.6% (+5.0pp)

**Insight**: Limiting LLM LoRA targets to attention projections only hurt every nutrient, with fat spiking badly. The MLP projections (gate/up/down) are important for this task; the full attention+MLP target set should be retained.


### Exp 12: 10-epoch baseline confirmation — KEPT

**Hypothesis**: 10 epochs (the MLX optimal) may be slightly better than 12 epochs in the HF pipeline as well, due to mild overfitting at 12 epochs.

**Change**: Run with default config (10 epochs). All other params identical to Exp 2.

**Result**: 19.3/26.9/37.4/25.0 = **27.2% avg**, 0 parse failures.

**Comparison vs Exp 2 (12 epochs, 27.8% avg)**:
- Calories: 18.5% → 19.3% (+0.8pp)
- Protein: 28.6% → 26.9% (-1.7pp)
- Fat: 39.1% → 37.4% (-1.7pp)
- Carbs: 25.1% → 25.0% (-0.1pp)
- Avg: 27.8% → 27.2% (-0.6pp)

**Insight**: 10 epochs is slightly better than 12 overall (-0.6pp avg), confirming the MLX finding. Protein and fat both improved; calories slightly worse but within noise. The HF baseline (27.2%) is now very close to the MLX baseline at comparable config (27.4% in MLX exp 30). This confirms the two pipelines are near-equivalent and the remaining gap to 18.1% should be closeable through vision block selection experiments.

**Also changed**: Increased default `--eval-steps` and `--save-steps` from 500 to 2000 to reduce checkpoint/eval overhead. This run took 22.5 hours due to the old 500-step defaults; future runs should be ~9-10 hours.


### Exp 13: Vision block subset — top-12 blocks (10 epochs) — REVERTED

**Hypothesis**: Adapting only the top-12 (deepest) vision blocks instead of all 24 will improve results, as it did in the MLX pipeline where vision block selection was the biggest win axis.

**Change**: Added `--vision-blocks 12` flag. LoRA applied to blocks 12–23 only (48 layers instead of 96). All other params identical to Exp 12.

**Result**: 22.1/30.5/50.5/28.9 = **33.0% avg**, 0 parse failures.

**Comparison vs Exp 12 (best, 27.2% avg)**:
- Calories: 19.3% → 22.1% (+2.8pp)
- Protein: 26.9% → 30.5% (+3.6pp)
- Fat: 37.4% → 50.5% (+13.1pp)
- Carbs: 25.0% → 28.9% (+3.9pp)
- Avg: 27.2% → 33.0% (+5.8pp)

**Insight**: Unlike MLX where top-12 was the biggest win, in the HF pipeline restricting to fewer vision blocks significantly degraded all nutrients — fat especially (+13.1pp). The HF custom LoRA implementation appears to need all 24 blocks for best results. This is a major divergence from the MLX pipeline behavior. The block-subset strategy that closed the gap in MLX does not transfer to HF. Next: try increasing vision rank to 64 on all 24 blocks (full 10 epochs).


### Exp 14: Vision LoRA rank 64 on all 24 blocks (10 epochs) — KEPT

**Hypothesis**: Higher vision LoRA rank (r64 vs r32) on all 24 blocks gives the vision tower more adaptation capacity and may improve results.

**Change**: `--lora-rank-vision 64 --lora-alpha-vision 64`. All other params identical to Exp 12.

**Result**: 17.6/25.4/39.6/22.2 = **26.2% avg**, 1 parse failure.

**Comparison vs Exp 12 (best, 27.2% avg)**:
- Calories: 19.3% → 17.6% (-1.7pp)
- Protein: 26.9% → 25.4% (-1.5pp)
- Fat: 37.4% → 39.6% (+2.2pp)
- Carbs: 25.0% → 22.2% (-2.8pp)
- Avg: 27.2% → 26.2% (-1.0pp)

**Insight**: Vision r64 improved the average by 1.0pp, with 3 of 4 nutrients improving. Fat regressed slightly (+2.2pp) but within tolerance. Doubling vision rank helped — more capacity in the vision tower lets the model learn better feature representations. The train loss was also slightly lower (0.800 vs 0.809). Next: try vision r128 to see if more capacity continues to help, and explore LR schedule changes.

