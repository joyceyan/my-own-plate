# Model Comparison — MAE% on Validation Set (349 samples)

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
