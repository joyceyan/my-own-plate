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
