"""
Evaluate a fine-tuned nutrition model against the NutriTest held-out set.

Supports two inference backends:
  - mlx: uses mlx-lm's generate() (text-only, vision tower stripped)
  - gguf: uses llama-cpp-python with Metal acceleration

Computes MAE and MAE% (Thames et al. CVPR 2021 style) for each macronutrient,
with optional side-by-side comparison against a baseline model.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUTRIENTS = ["calories", "protein", "fat", "carbs"]

# Default path to the Thames et al. baseline data
_DEFAULT_BASELINE_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "nutrition5k_baseline_data.json"
)


def load_n5k_baselines(path: Path = _DEFAULT_BASELINE_JSON):
    """
    Load Thames et al. CVPR 2021 baselines from nutrition5k_baseline_data.json.
    Returns dict keyed by nutrient with 'mae_absolute' and 'mae_pct_of_mean'.
    """
    if not path.exists():
        print(f"Warning: baseline file not found at {path}, using empty baselines")
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl_data(path: str):
    """Load evaluation data from a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                gt = json.loads(sample["completion"])
                sample["ground_truth"] = {k: float(gt[k]) for k in NUTRIENTS}
                data.append(sample)
    return data


def load_hf_split(dataset_dir: str, split: str = "validation"):
    """Load evaluation data from a HuggingFace dataset split."""
    from datasets import load_dataset
    ds = load_dataset(dataset_dir, split=split)
    data = []
    for row in ds:
        gt = json.loads(row["answer"])
        # Reconstruct the image_path from the HF Image object
        img = row.get("image")
        image_path = getattr(img, "filename", None) if img else None
        data.append({
            "image_path": image_path,
            "prompt": row["question"],
            "completion": row["answer"],
            "ground_truth": {k: float(gt[k]) for k in NUTRIENTS},
        })
    return data


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def load_mlx_model(model_path: str, adapter_path: str = None, lora_rank: int = 16, lora_alpha: float = 1.0):
    """
    Load a VLM via mlx-vlm (keeps vision tower intact).
    model_path: HF repo ID or local dir (base model)
    adapter_path: optional path to LoRA adapter directory
    Returns (model, processor, config).

    NOTE: We do NOT pass adapter_path to mlx_vlm.load() because its
    apply_lora_layers wraps ALL linear layers with LoRA (via
    find_all_linear_names), not just the ones we trained. This causes
    randomly-initialized MLP LoRA layers to corrupt the model. Instead
    we load the base model, apply LoRA only to attention projections,
    and load the adapter weights manually.
    """
    import mlx.core as mx
    from mlx_vlm.utils import load
    p = Path(model_path).expanduser()
    if p.exists():
        model_path = str(p.resolve())

    model, processor = load(
        model_path,
        processor_config={"trust_remote_code": True},
    )

    if adapter_path:
        ap = Path(adapter_path).expanduser()
        if ap.exists():
            adapter_path = str(ap.resolve())
        from mlx_vlm.trainer.utils import get_peft_model
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # mlx-vlm uses alpha as a raw multiplier (not alpha/rank)
        model = get_peft_model(model, attn_modules, rank=lora_rank, alpha=lora_alpha, dropout=0.0, verbose=False)
        adapter_file = Path(adapter_path) / "adapters.safetensors"
        model.load_weights(str(adapter_file), strict=False)

    config = model.config.__dict__
    return model, processor, config


def infer_mlx(model, processor, config, prompt: str, image_path: str = None,
              max_tokens: int = 512) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    num_images = 1 if (image_path and os.path.exists(image_path)) else 0
    formatted = apply_chat_template(
        processor, config, prompt, num_images=num_images
    )
    result = generate(
        model, processor, formatted,
        image=image_path if num_images else None,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return result.text if hasattr(result, "text") else str(result)


def load_gguf_model(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python is required for GGUF inference.")
        print("Install with: pip install llama-cpp-python")
        sys.exit(1)
    model = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    return model


def infer_gguf(model, prompt: str, max_tokens: int = 512) -> str:
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_completion(raw: str):
    """
    Parse model output as JSON. Falls back to regex extraction on failure.
    Returns (dict_with_nutrient_values, parse_failed_bool).
    """
    # Strip markdown code fences (```json ... ```)
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()

    # Try JSON parse first — look for the first { ... } block
    # Use a greedy match to handle nested structures (e.g. ingredients as array)
    try:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            json_str = match.group()
            # If truncated (no closing brace), try to fix
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'
            parsed = json.loads(json_str)
            result = {}
            for k in NUTRIENTS:
                if k in parsed:
                    result[k] = float(parsed[k])
            if len(result) == len(NUTRIENTS):
                return result, False
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback: extract "key": number patterns
    result = {}
    for nutrient in NUTRIENTS:
        pattern = rf'["\']?{nutrient}["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            result[nutrient] = float(m.group(1))

    if len(result) == len(NUTRIENTS):
        return result, True  # regex fallback succeeded but JSON failed
    return result, True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ground_truths, predictions):
    """
    Compute per-nutrient MAE, MAE%, and std of per-sample % errors.
    Returns dict keyed by nutrient name.
    """
    metrics = {}
    for nutrient in NUTRIENTS:
        gt = np.array([s[nutrient] for s in ground_truths])
        pred = np.array([s.get(nutrient, 0.0) for s in predictions])

        abs_errors = np.abs(gt - pred)
        mae = float(np.mean(abs_errors))

        gt_mean = float(np.mean(gt))
        mae_pct = (mae / gt_mean * 100) if gt_mean > 0 else 0.0

        # Per-sample % error relative to the dataset mean (Thames et al. style)
        pct_errors = abs_errors / gt_mean * 100 if gt_mean > 0 else abs_errors
        std_pct = float(np.std(pct_errors))

        metrics[nutrient] = {
            "mae": round(mae, 3),
            "mae_pct": round(mae_pct, 1),
            "std_pct": round(std_pct, 1),
            "gt_mean": round(gt_mean, 2),
        }
    return metrics


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results_table(ft_metrics, n5k_baselines, base_metrics=None,
                        ft_failures=0, base_failures=0, total=0):
    has_base = base_metrics is not None
    col_w = 20
    label_w = 20

    header = f"{'Metric':<{label_w}}| {'Fine-tuned':<{col_w}}"
    if has_base:
        header += f"| {'Base model':<{col_w}}"
    header += f"| {'N5k RGB baseline':<{col_w}}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    UNITS = {"calories": "kcal", "protein": "g", "fat": "g", "carbs": "g"}

    for nutrient in NUTRIENTS:
        ft = ft_metrics[nutrient]
        n5k = n5k_baselines.get(nutrient, {})
        unit = UNITS[nutrient]
        cap = nutrient.capitalize()

        # MAE% row
        ft_str = f"{ft['mae_pct']:.1f}% \u00b1 {ft['std_pct']:.1f}%"
        row = f"{cap + ' MAE%':<{label_w}}| {ft_str:<{col_w}}"
        if has_base:
            b = base_metrics[nutrient]
            b_str = f"{b['mae_pct']:.1f}% \u00b1 {b['std_pct']:.1f}%"
            row += f"| {b_str:<{col_w}}"
        n5k_pct = n5k.get("mae_pct_of_mean")
        row += f"| {n5k_pct:.1f}%" if n5k_pct is not None else f"| {'—':<{col_w}}"
        print(row)

        # Absolute MAE row
        ft_abs = f"{ft['mae']:.1f} {unit}"
        row = f"{cap + ' MAE':<{label_w}}| {ft_abs:<{col_w}}"
        if has_base:
            b = base_metrics[nutrient]
            b_abs = f"{b['mae']:.1f} {unit}"
            row += f"| {b_abs:<{col_w}}"
        n5k_abs = n5k.get("mae_absolute")
        n5k_unit = n5k.get("mae_absolute_unit", unit)
        row += f"| {n5k_abs:.1f} {n5k_unit}" if n5k_abs is not None else f"| {'—':<{col_w}}"
        print(row)

    # Parse failures row
    row = f"{'Parse failures':<{label_w}}| {str(ft_failures):<{col_w}}"
    if has_base:
        row += f"| {str(base_failures):<{col_w}}"
    row += f"| {'—'}"
    print(row)
    print(sep)
    print(f"Total test samples: {total}\n")


# ---------------------------------------------------------------------------
# Run evaluation for one model
# ---------------------------------------------------------------------------

class EvalAbortError(Exception):
    """Raised when parse failures exceed the threshold."""
    pass


def run_eval(samples, model_type, model_path, adapter_path=None,
             max_parse_failures=20, lora_rank=16, lora_alpha=1.0):
    """Run inference on all samples. Returns (predictions, parse_failures, failed_log).
    Raises EvalAbortError if parse failures exceed max_parse_failures."""
    if model_type == "mlx":
        model, processor, config = load_mlx_model(model_path, adapter_path=adapter_path,
                                                   lora_rank=lora_rank, lora_alpha=lora_alpha)
        infer_fn = lambda prompt, img: infer_mlx(model, processor, config, prompt, image_path=img)
    else:
        model = load_gguf_model(model_path)
        infer_fn = lambda prompt, img: infer_gguf(model, prompt)

    predictions = []
    parse_failures = 0
    failed_log = []

    for i, sample in enumerate(samples):
        raw_output = infer_fn(sample["prompt"], sample.get("image_path"))
        parsed, failed = parse_completion(raw_output)

        if failed:
            parse_failures += 1
            failed_log.append({
                "index": i,
                "image_path": sample["image_path"],
                "raw_output": raw_output,
                "parsed_partial": parsed,
            })
            if parse_failures >= max_parse_failures:
                raise EvalAbortError(
                    f"Aborting: {parse_failures} parse failures in {i + 1} samples. "
                    f"Model output is likely broken."
                )

        # Fill missing nutrients with 0 so metrics still compute
        pred = {k: parsed.get(k, 0.0) for k in NUTRIENTS}
        predictions.append(pred)

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{model_path}] {i + 1}/{len(samples)} samples "
                  f"({parse_failures} failures)", flush=True)

    return predictions, parse_failures, failed_log


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_per_sample_csv(path, samples, predictions, parse_failures_set):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "true_calories", "pred_calories", "cal_error",
            "true_protein", "pred_protein", "protein_error",
            "true_fat", "pred_fat", "fat_error",
            "true_carbs", "pred_carbs", "carbs_error",
            "parse_failed",
        ])
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            gt = sample["ground_truth"]
            writer.writerow([
                sample["image_path"],
                gt["calories"], pred["calories"],
                abs(gt["calories"] - pred["calories"]),
                gt["protein"], pred["protein"],
                abs(gt["protein"] - pred["protein"]),
                gt["fat"], pred["fat"],
                abs(gt["fat"] - pred["fat"]),
                gt["carbs"], pred["carbs"],
                abs(gt["carbs"] - pred["carbs"]),
                i in parse_failures_set,
            ])


def save_summary_json(path, metrics, model_path, test_data_path, total,
                      parse_failures, n5k_baselines, baseline_metrics=None,
                      baseline_failures=None):
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "test_data": test_data_path,
        "total_samples": total,
        "fine_tuned": {
            "metrics": metrics,
            "parse_failures": parse_failures,
        },
        "n5k_rgb_baselines": n5k_baselines,
    }
    if baseline_metrics is not None:
        summary["baseline"] = {
            "metrics": baseline_metrics,
            "parse_failures": baseline_failures,
        }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate nutrition model on NutriTest held-out set"
    )
    parser.add_argument(
        "--mode", type=str, choices=["val", "test"], default="val",
        help="Evaluation mode: 'val' for development/checkpoint selection, "
             "'test' for final reporting only (default: val)",
    )
    parser.add_argument(
        "--val-data", type=str,
        default="~/src/my-own-plate/data/nutrition5k_hf",
        help="Path to HF dataset dir containing validation.parquet (used with --mode val)",
    )
    parser.add_argument(
        "--test-data", type=str,
        default="~/src/my-own-plate/data/nutrition5k_test.jsonl",
        help="Path to test JSONL (used with --mode test). "
             "Generated by convert_dataset.py.",
    )
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3-VL-2B-Instruct-bf16",
        help="Base model HF ID or local path",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Path to LoRA adapter directory (contains adapters.safetensors). "
             "If provided, evaluates the fine-tuned model.",
    )
    parser.add_argument(
        "--model-type", type=str, choices=["mlx", "gguf"], default="mlx",
        help="Inference backend (default: mlx)",
    )
    parser.add_argument(
        "--no-base", action="store_true",
        help="Skip base model evaluation",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="~/src/my-own-plate/training/eval_results/",
        help="Directory for output files",
    )
    parser.add_argument(
        "--max-parse-failures", type=int, default=20,
        help="Abort eval if parse failures exceed this (default: 20). "
             "Set to 0 to disable.",
    )
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (must match training, default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=1.0,
                        help="LoRA alpha (must match training, default: 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.test_data = os.path.expanduser(args.test_data)
    args.val_data = os.path.expanduser(args.val_data)
    args.model = os.path.expanduser(args.model)
    args.output_dir = os.path.expanduser(args.output_dir)
    if args.adapter_path:
        args.adapter_path = os.path.expanduser(args.adapter_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load N5k baselines from JSON
    n5k_baselines = load_n5k_baselines()
    if n5k_baselines:
        print(f"Loaded N5k baselines: {', '.join(n5k_baselines.keys())}")

    # Load data based on mode
    if args.mode == "val":
        print(f"Mode: VALIDATION (for development — use --mode test for final reporting)")
        print(f"Loading validation data from: {args.val_data}")
        samples = load_hf_split(args.val_data, split="validation")
    else:
        print(f"Mode: TEST (final held-out evaluation)")
        print(f"Loading test data from: {args.test_data}")
        samples = load_jsonl_data(args.test_data)
    print(f"Loaded {len(samples)} samples")

    # Run fine-tuned model (base model + adapter)
    ft_label = f"{args.model}"
    if args.adapter_path:
        ft_label += f" + {args.adapter_path}"
    max_fail = args.max_parse_failures if args.max_parse_failures > 0 else float("inf")

    print(f"\nEvaluating fine-tuned model: {ft_label}")
    t0 = time.time()
    try:
        ft_preds, ft_failures, ft_failed_log = run_eval(
            samples, args.model_type, args.model,
            adapter_path=args.adapter_path, max_parse_failures=max_fail,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        )
    except EvalAbortError as e:
        print(f"\n  ABORTED: {e}")
        sys.exit(1)
    print(f"  Fine-tuned eval took {(time.time() - t0) / 60:.1f} min")
    gt_list = [s["ground_truth"] for s in samples]
    ft_metrics = compute_metrics(gt_list, ft_preds)

    # Log parse failures
    if ft_failed_log:
        fail_path = output_dir / "parse_failures_finetuned.jsonl"
        with open(fail_path, "w") as f:
            for entry in ft_failed_log:
                f.write(json.dumps(entry) + "\n")
        print(f"  Logged {len(ft_failed_log)} parse failures to {fail_path}")

    # Run base model for comparison (unless --no-base)
    base_metrics = None
    base_failures = 0
    if not args.no_base:
        print(f"\nEvaluating base model: {args.model}")
        t0 = time.time()
        try:
            base_preds, base_failures, base_failed_log = run_eval(
                samples, "mlx", args.model, max_parse_failures=max_fail,
            )
        except EvalAbortError as e:
            print(f"\n  ABORTED: {e}")
            sys.exit(1)
        print(f"  Base eval took {(time.time() - t0) / 60:.1f} min")
        base_metrics = compute_metrics(gt_list, base_preds)
        if base_failed_log:
            fail_path = output_dir / "parse_failures_baseline.jsonl"
            with open(fail_path, "w") as f:
                for entry in base_failed_log:
                    f.write(json.dumps(entry) + "\n")
            print(f"  Logged {len(base_failed_log)} parse failures to {fail_path}")

    # Print results table
    print_results_table(
        ft_metrics,
        n5k_baselines,
        base_metrics=base_metrics,
        ft_failures=ft_failures,
        base_failures=base_failures,
        total=len(samples),
    )

    # Save per-sample CSV
    ft_failure_indices = {e["index"] for e in ft_failed_log}
    csv_path = output_dir / "per_sample_results.csv"
    save_per_sample_csv(csv_path, samples, ft_preds, ft_failure_indices)
    print(f"Per-sample results saved to: {csv_path}")

    # Save summary JSON
    json_path = output_dir / "eval_summary.json"
    save_summary_json(
        json_path, ft_metrics, args.model, args.test_data, len(samples),
        ft_failures, n5k_baselines, baseline_metrics=base_metrics,
        baseline_failures=base_failures,
    )
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
