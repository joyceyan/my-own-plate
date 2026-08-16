# HF Pipeline + GGUF Quantization Comparison — MAE% on Test Set (349 samples)

| Model / Format | Calories | Protein | Fat | Carbs | Avg |
|----------------|----------|---------|-----|-------|-----|
| Qwen3-VL-2B base (no fine-tuning) | 50.7% | 53.5% | 68.8% | 83.9% | *64.2%* |
| HF adapters (Exp 17) | 22.8% | 39.1% | 53.3% | 25.4% | *35.1%* |
| GGUF F16 (v2) | 22.3% | 36.3% | 52.6% | 24.1% | *33.8%* |
| GGUF Q4_K_M (v2) | 26.6% | 41.4% | 57.1% | 45.4% | *42.6%* |
| N5k baseline | 26.1% | 29.5% | 34.2% | 31.9% | *30.4%* |

HF fine-tuned model: Exp 17 — LLM LoRA r128/α128, projector r128, vision LoRA r64/α64 on all 24 blocks, LR 2e-5 → 1e-6 cosine, 10 epochs. GGUF v2 variants used. --image-min-tokens 130 --image-max-tokens 130 required on llama-server.

Metric: fine-tuned models use per-sample MAE% (mean of |pred−true|/true). Base model uses MAE/mean(GT) to avoid inflation from low-calorie samples. N5k baseline is paper-reported MAE/mean on their own test split (Thames et al., CVPR 2021).

### Comparison with Validation Set

| Model / Format | Val Avg | Test Avg | Delta |
|----------------|---------|----------|-------|
| Qwen3-VL-2B base | *59.2%* | *64.2%* | +5.0 |
| HF adapters (Exp 17) | *24.4%* | *35.1%* | +10.7 |
| GGUF F16 | *23.7%* | *33.8%* | +10.1 |
| GGUF Q4_K_M | *28.3%* | *42.6%* | +14.3 |

# HF Pipeline + GGUF Quantization Comparison — MAE% on Validation Set (349 samples)

| Model / Format | Calories | Protein | Fat | Carbs | Avg |
|----------------|----------|---------|-----|-------|-----|
| Qwen3-VL-2B base (no fine-tuning) | 48.4% | 55.8% | 65.7% | 66.8% | **59.2%** |
| HF adapters (Exp 17) | 17.9% | 23.2% | 33.7% | 23.0% | **24.4%** |
| GGUF F16 | 19.4% | 22.0% | 30.4% | 23.1% | **23.7%** |
| GGUF Q4_K_M | 22.1% | 24.1% | 38.1% | 29.2% | **28.3%** |
| N5k baseline | 26.1% | 29.5% | 34.2% | 31.9% | **30.4%** |

HF fine-tuned model: Exp 17 — LLM LoRA r128/α128, projector r128, vision LoRA r64/α64 on all 24 blocks, LR 2e-5 → 1e-6 cosine, 10 epochs. GGUF export details and critical image-token matching fix are documented in `training/notes.md` under "GGUF Export Verification".

---

# Legacy MLX Model Comparison — MAE% on Validation Set (349 samples)

| Nutrient     | N5k Baseline (Thames et al.) | Qwen3-VL-2B Base | Qwen3-VL-2B Fine-tuned |
|--------------|------------------------------|-------------------|------------------------|
| Calories     | 26.1%                        | 48.4%             | **15.7%**              |
| Protein      | 29.5%                        | 55.8%             | **17.2%**              |
| Fat          | 34.2%                        | 65.7%             | **23.6%**              |
| Carbs        | 31.9%                        | 66.8%             | **18.6%**              |
| **Avg**      | **30.4%**                    | **59.2%**         | **18.8%**              |

Fine-tuned model: exp 44 — LLM LoRA r64, vision tower LoRA r32 (all 24 blocks), cosine LR decay, 10 epochs.

Note: Base model numbers from earlier eval run.

## Held-Out Test Set (349 samples)

| Nutrient     | N5k Baseline (Thames et al.) | Qwen3-VL-2B Base | Qwen3-VL-2B Fine-tuned |
|--------------|------------------------------|-------------------|------------------------|
| Calories     | 26.1%                        | 61.0%             | **15.0%**              |
| Protein      | 29.5%                        | 63.2%             | **17.2%**              |
| Fat          | 34.2%                        | 78.6%             | **23.8%**              |
| Carbs        | 31.9%                        | 89.2%             | **20.6%**              |
| **Avg**      | **30.4%**                    | **73.0%**         | **19.2%**              |

Fine-tuned: 4 parse failures. Base: 0 parse failures.
