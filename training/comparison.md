# HF Pipeline + GGUF Quantization Comparison — MAE/mean% (349 samples each)

Metric: MAE as % of mean ground truth per nutrient, matching Thames et al. CVPR 2021.

## Validation Set

| Model / Format | Calories | Protein | Fat | Carbs | Avg |
|----------------|----------|---------|-----|-------|-----|
| Qwen3-VL-2B base (no fine-tuning) | 48.4% | 47.0% | 63.8% | 78.0% | **59.3%** |
| HF adapters (Exp 17) | 13.7% | 14.9% | 19.6% | 15.4% | **15.9%** |
| GGUF F16 (v2) | 16.8% | 16.3% | 21.9% | 18.4% | **18.4%** |
| GGUF Q4_K_M (v2) | 15.6% | 17.6% | 21.0% | 18.3% | **18.2%** |
| N5k baseline | 26.1% | 29.5% | 34.2% | 31.9% | **30.4%** |

## Held-Out Test Set

| Model / Format | Calories | Protein | Fat | Carbs | Avg |
|----------------|----------|---------|-----|-------|-----|
| Qwen3-VL-2B base (no fine-tuning) | 50.7% | 53.5% | 68.8% | 83.9% | **64.2%** |
| HF adapters (Exp 17) | 13.6% | 15.2% | 19.9% | 16.9% | **16.4%** |
| GGUF F16 (v2) | 17.2% | 20.4% | 23.2% | 19.7% | **20.1%** |
| GGUF Q4_K_M (v2) | 15.7% | 18.2% | 23.5% | 19.7% | **19.3%** |
| N5k baseline | 26.1% | 29.5% | 34.2% | 31.9% | **30.4%** |

HF fine-tuned model: Exp 17 — LLM LoRA r128/α128, projector r128, vision LoRA r64/α64 on all 24 blocks, LR 2e-5 → 1e-6 cosine, 10 epochs. GGUF v2 variants used. `--image-min-tokens 130 --image-max-tokens 130` required on llama-server. N5k baseline is paper-reported MAE/mean on their own test split (Thames et al., CVPR 2021).

---

# Legacy MLX Model Comparison — MAE/mean% on Validation Set (349 samples)

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
