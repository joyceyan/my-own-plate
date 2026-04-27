# Training Pipeline — My Own Plate

LoRA fine-tuning of Qwen3-VL-2B-Instruct on the Nutrition5k dataset using
**mlx-vlm** on Apple Silicon. Unlike mlx-lm (which strips the vision tower),
mlx-vlm keeps the full vision-language pipeline so the model actually sees
food images during training.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- Python 3.10+

## Install Dependencies

```bash
pip install -r requirements_mac.txt
```

## Step 1: Prepare Data

```bash
# Raw N5k CSVs → single JSONL (run once)
cd ../data && python prepare_nutrition5k.py

# Split into train/val/test + convert to HuggingFace parquet
cd ../training && python convert_dataset.py
```

`prepare_nutrition5k.py` reads the raw Nutrition5k CSVs and outputs
`data/nutrition5k_all.jsonl` with all valid samples (no splitting).

`convert_dataset.py` reads that JSONL and produces:
- `data/nutrition5k_hf/` — HuggingFace parquet (80/10/10 train/val/test)
- `data/nutrition5k_{train,validation,test}.jsonl` — per-split JSONL for readability

Split ratios are configurable: `--train-ratio 0.80 --val-ratio 0.10`.

## Step 2: Train

```bash
python train.py --train-data ~/src/my-own-plate/data/nutrition5k_hf
```

Key defaults (tuned for stable training on M2 Pro 32 GB):
- `--model mlx-community/Qwen3-VL-2B-Instruct-bf16`
- `--batch-size 1`
- `--learning-rate 2e-5`
- `--lora-rank 16`, `--lora-alpha 32`
- `--epochs 3`
- `--grad-checkpoint` on by default
- `--image-resize 384 384` (keeps memory manageable)
- `--steps-per-eval 200` (validation loss every 200 steps)
- `mx.compile` disabled by default (prevents Metal GPU timeout)

### Metal GPU timeout

On Apple Silicon laptops, macOS kills GPU commands that block the display
compositor ("Impacting Interactivity" error). If you hit this:

1. Close memory-heavy apps (browsers, Docker) — aim for <16 GB in use
2. Keep `--batch-size 1` and `--grad-checkpoint` (the defaults)
3. The script disables `mx.compile` and caps Metal memory by default

### What gets saved

| Artifact | Location |
|---|---|
| Adapter weights (every 500 steps) | `output/adapters/` |
| Final adapter weights | `output/adapters/adapters.safetensors` |

## Step 3: Evaluate

```bash
# Development: evaluate on validation set (default)
python evaluate.py --adapter-path ./output/adapters

# Final reporting: evaluate on held-out test set
python evaluate.py --adapter-path ./output/adapters --mode test
```

Use `--mode val` (default) during development and checkpoint selection.
Use `--mode test` once for final reporting — don't repeatedly evaluate on test.

Use `--no-base` to skip the base model comparison for a faster run.

Results are saved to `eval_results/per_sample_results.csv` and
`eval_results/eval_summary.json`.

## Step 4: Merge & Export (optional)

Merge the LoRA adapter into the base weights and convert to GGUF:

```bash
python merge_and_export.py --adapter-path ./output/adapters --output-dir ./output
```

This produces:
- `output/merged/` — fused model weights
- `output/myownplate-q4km.gguf` — Q4_K_M quantized GGUF

The GGUF conversion requires [llama.cpp](https://github.com/ggml-org/llama.cpp).
Set `--llama-cpp-dir` or `LLAMA_CPP_DIR` to point to your clone.
