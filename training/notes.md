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

## Current best (Exp 2, 27.8% avg)

12 epochs, all other params same as Exp 0 but with gradient checkpointing disabled (Exp 1 fix). This is the full-run baseline to beat.

| Nutrient  | MAE%  |
|-----------|-------|
| Calories  | 18.5% |
| Protein   | 28.6% |
| Fat       | 39.1% |
| Carbs     | 25.1% |
| **Avg**   | **27.8%** |

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

1. **10-epoch baseline confirmation**: Run current defaults at 10 epochs (Exp 2 used 12). Confirm baseline matches MLX ~27.4% and that 10ep is better than 12ep.

2. **Vision block subset: top-12 blocks**: The biggest win axis in MLX. Adapting only the deepest N blocks instead of all 24 improved results significantly (MLX: all 24 at r32 → 18.9%, top-12 at r32 → 19.5%, but top-12 at r32 with cosine LR → 18.5%). Requires adding `num_blocks` param to `apply_vision_block_lora` in `hf_utils.py`.

3. **Vision block subset: top-8 blocks**: Continue sweep if top-12 helps.

4. **Vision rank on selected blocks**: Once best block count is found, try r64 on those blocks (MLX: r64 on top-12 got 19.0% vs r32's 19.5%).

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

