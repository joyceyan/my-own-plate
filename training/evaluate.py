"""
Evaluate the HF fine-tuned Qwen3-VL model on Nutrition5k validation/test set.

Loads the base model + PEFT adapter + custom vision LoRA and computes
per-nutrient MAE/MAE%. Outputs a summary table and saves per-sample CSV.

Usage:
    source .venv/bin/activate
    python evaluate_hf.py --mode val
    python evaluate_hf.py --mode test
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
import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from peft import PeftModel

from hf_utils import (
    apply_vision_block_lora,
    apply_projector_lora,
    load_custom_lora,
    merge_custom_lora,
)


NUTRIENTS = ["calories", "protein", "fat", "carbs"]
UNITS = {"calories": "kcal", "protein": "g", "fat": "g", "carbs": "g"}

DEFAULT_BASELINE_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "nutrition5k_baseline_data.json"
)


# ---------------------------------------------------------------------------
# Nutrient parsing (shared with evaluate.py)
# ---------------------------------------------------------------------------

NUTRIENT_ALIASES = {
    "calories": ["calories", "kcal", "cal", "cals", "energy"],
    "protein": ["protein", "protein_g"],
    "fat": ["fat", "fat_g", "total_fat"],
    "carbs": ["carbs", "carb", "carbohydrates", "carbs_g"],
}


def parse_completion(raw: str):
    """Parse model output as JSON. Falls back to regex extraction on failure."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            json_str = match.group()
            if json_str.count("{") > json_str.count("}"):
                json_str += "}"
            parsed = json.loads(json_str)
            flat = {}
            for k, v in parsed.items():
                if isinstance(v, dict):
                    flat.update(v)
                else:
                    flat[k] = v
            result = {}
            for k in NUTRIENTS:
                val = _find_nutrient_in_dict(flat, k)
                if val is not None:
                    result[k] = val
            if len(result) == len(NUTRIENTS):
                return result, False
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    result = {}
    for nutrient in NUTRIENTS:
        aliases = NUTRIENT_ALIASES.get(nutrient, [nutrient])
        for alias in aliases:
            pattern = rf'["\']?{alias}[^"\']*?["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                result[nutrient] = float(m.group(1))
                break

    if len(result) == len(NUTRIENTS):
        return result, True
    return result, True


def _find_nutrient_in_dict(flat: dict, nutrient: str):
    aliases = NUTRIENT_ALIASES.get(nutrient, [nutrient])
    for alias in aliases:
        if alias in flat:
            return float(flat[alias])
    for fk, fv in flat.items():
        fk_lower = fk.lower().strip()
        for alias in aliases:
            if fk_lower == alias or fk_lower.startswith(alias) or alias in fk_lower:
                return float(fv)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ground_truths, predictions, parse_failed_flags):
    """Compute per-sample absolute percentage errors, then average per nutrient."""
    metrics = {}
    for nutrient in NUTRIENTS:
        pct_errors = []
        for gt, pred, failed in zip(ground_truths, predictions, parse_failed_flags):
            if failed:
                continue
            val = gt[nutrient]
            if val > 0:
                err = abs(val - pred.get(nutrient, 0.0)) / val * 100
                pct_errors.append(err)
        if pct_errors:
            metrics[nutrient] = {
                "mae_pct": round(float(np.mean(pct_errors)), 1),
                "std_pct": round(float(np.std(pct_errors)), 1),
            }
        else:
            metrics[nutrient] = {"mae_pct": float("nan"), "std_pct": float("nan")}
    return metrics


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, adapter_dir: str, vision_lora_path: str,
               image_size: int = 384, vision_rank: int = 32, vision_alpha: int = 32,
               projector_rank: int = 64, projector_alpha: int = 64,
               no_adapter: bool = False):
    """Load base model, optionally apply LoRA adapters."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    pixels = image_size * image_size
    processor.image_processor.min_pixels = pixels
    processor.image_processor.max_pixels = pixels

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=device,
    )

    if not no_adapter:
        apply_vision_block_lora(model, r=vision_rank, alpha=vision_alpha, dropout=0.0)
        apply_projector_lora(model, r=projector_rank, alpha=projector_alpha, dropout=0.0)

        model = PeftModel.from_pretrained(model, adapter_dir)
        load_custom_lora(model, vision_lora_path)

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(model, processor, image, prompt: str, max_new_tokens: int = 128):
    if isinstance(image, str):
        image = PILImage.open(image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Only decode the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_n5k_baselines(path: Path = DEFAULT_BASELINE_JSON):
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {})


def load_hf_split(dataset_dir: str, split: str = "validation"):
    ds = load_dataset("parquet", data_dir=dataset_dir, split=split)
    data = []
    for row in ds:
        gt = json.loads(row["messages_json"])[-1]["content"]
        gt = json.loads(gt)
        data.append({
            "image": row["image"],
            "prompt": json.loads(row["messages_json"])[0]["content"][1]["text"],
            "completion": gt,
            "ground_truth": {k: float(gt[k]) for k in NUTRIENTS},
        })
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HF fine-tuned Qwen3-VL on Nutrition5k"
    )
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--model", type=str, default="training/cache/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="~/src/my-own-plate/training/output/adapter")
    parser.add_argument("--vision-lora", type=str, default="~/src/my-own-plate/training/output/vision_lora.pt")
    parser.add_argument("--dataset", type=str, default="~/src/my-own-plate/data/nutrition5k_hf_chat")
    parser.add_argument("--output-dir", type=str, default="~/src/my-own-plate/training/eval_results_hf")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lora-rank-vision", type=int, default=32)
    parser.add_argument("--lora-alpha-vision", type=int, default=32)
    parser.add_argument("--lora-rank-projector", type=int, default=64)
    parser.add_argument("--lora-alpha-projector", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-adapter", action="store_true",
                        help="Evaluate base model without any adapter/LoRA")
    return parser.parse_args()


def main():
    args = parse_args()
    args.adapter_dir = os.path.expanduser(args.adapter_dir)
    args.vision_lora = os.path.expanduser(args.vision_lora)
    args.dataset = os.path.expanduser(args.dataset)
    args.output_dir = os.path.expanduser(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    n5k_baselines = load_n5k_baselines()

    print(f"Mode: {args.mode}")
    samples = load_hf_split(args.dataset, split="validation" if args.mode == "val" else "test")
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples")

    print("\nLoading model...")
    model, processor = load_model(
        args.model, args.adapter_dir, args.vision_lora,
        image_size=args.image_size,
        vision_rank=args.lora_rank_vision,
        vision_alpha=args.lora_alpha_vision,
        projector_rank=args.lora_rank_projector,
        projector_alpha=args.lora_alpha_projector,
        no_adapter=args.no_adapter,
    )

    print("\nRunning inference...")
    predictions = []
    parse_failed_flags = []
    failed = 0
    failed_log = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        raw = infer(model, processor, sample["image"], sample["prompt"])
        pred, parse_failed = parse_completion(raw)
        parse_failed_flags.append(parse_failed)
        if parse_failed:
            failed += 1
            failed_log.append({"index": i, "raw": raw})
            if failed <= 5:
                print(f"  [{i+1}] PARSE FAIL: {raw[:120]}")
        predictions.append(pred)

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}] rate={rate:.2f} samples/s, eta={eta:.0f}s")

    gt_list = [s["ground_truth"] for s in samples]
    metrics = compute_metrics(gt_list, predictions, parse_failed_flags)

    # Print table
    print("\n" + "=" * 60)
    print(f"HF Fine-tuned Results — {args.mode} set ({len(samples)} samples)")
    print(f"Parse failures: {failed}/{len(samples)}")
    print(f"Time: {time.time() - t0:.0f}s")
    print("=" * 60)
    for n in NUTRIENTS:
        print(f"  {n:10s}: {metrics[n]['mae_pct']:.1f}% MAE%")
    avg = np.mean([metrics[n]["mae_pct"] for n in NUTRIENTS])
    print(f"  {'avg':10s}: {avg:.1f}%")

    if n5k_baselines:
        print("\nComparison:")
        print(f"  N5k baseline:         30.4%")
        print(f"  Qwen3-VL-2B base:     59.2%")
        print(f"  Previous MLX tuned:   18.1%")
        print(f"  HF fine-tuned:        {avg:.1f}%")

    # Save CSV
    csv_path = Path(args.output_dir) / f"per_sample_results_{args.mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "true_calories", "pred_calories", "true_protein", "pred_protein",
            "true_fat", "pred_fat", "true_carbs", "pred_carbs", "parse_failed",
        ])
        for s, p, pf in zip(samples, predictions, parse_failed_flags):
            gt = s["ground_truth"]
            writer.writerow([
                str(s["image"]), gt["calories"], p.get("calories", 0),
                gt["protein"], p.get("protein", 0), gt["fat"], p.get("fat", 0),
                gt["carbs"], p.get("carbs", 0), pf,
            ])
    print(f"\nPer-sample CSV saved to {csv_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "vision_lora": args.vision_lora,
        "total_samples": len(samples),
        "parse_failures": failed,
        "metrics": metrics,
        "avg_mae_pct": round(avg, 1),
    }
    json_path = Path(args.output_dir) / f"eval_summary_{args.mode}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
