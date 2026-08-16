"""
Evaluate a GGUF model via llama-server's OpenAI-compatible API using the
HF parquet dataset (images are loaded from the parquet, so no local
Nutrition5k imagery tree is required).

Start the server first, e.g.:
    ~/src/llama.cpp/build/bin/llama-server \
        -m training/output/gguf/myownplate-f16.gguf \
        --mmproj training/output/gguf/mmproj-myownplate-f16.gguf \
        --image-min-tokens 130 --image-max-tokens 130 \
        --port 8080 -ngl 99 -c 2048

Usage:
    python eval_gguf_parquet.py --mode test --server-url http://localhost:8080
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import NUTRIENTS, parse_completion


DEFAULT_DATASET = "~/src/my-own-plate/data/nutrition5k_hf_chat"
DEFAULT_OUTPUT_DIR = "~/src/my-own-plate/training/eval_results_gguf"


def image_to_base64(image: PILImage.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def infer(image: PILImage.Image, prompt: str, server_url: str, timeout: int = 120) -> str | None:
    b64 = image_to_base64(image)
    mime = "image/png" if image.format in (None, "PNG") else f"image/{image.format.lower()}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def load_split(dataset_dir: str, split: str = "test"):
    ds = load_dataset("parquet", data_dir=os.path.expanduser(dataset_dir), split=split)
    samples = []
    for row in ds:
        messages = json.loads(row["messages_json"])
        prompt = messages[0]["content"][1]["text"]
        gt = json.loads(messages[-1]["content"])
        samples.append({
            "image": row["image"],
            "prompt": prompt,
            "ground_truth": {k: float(gt[k]) for k in NUTRIENTS},
        })
    return samples


def check_server(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        return r.json().get("status") == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="test")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    args.dataset = os.path.expanduser(args.dataset)
    args.output_dir = os.path.expanduser(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    split = "validation" if args.mode == "val" else "test"
    if not check_server(args.server_url):
        print(f"ERROR: llama-server not running at {args.server_url}")
        sys.exit(1)

    samples = load_split(args.dataset, split=split)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Evaluating {len(samples)} {args.mode} samples via {args.server_url}")

    predictions = []
    parse_failed_flags = []
    failed = 0
    failed_log = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        raw = infer(sample["image"], sample["prompt"], args.server_url, timeout=args.timeout)
        pred, parse_failed = parse_completion(raw or "")
        parse_failed_flags.append(parse_failed)
        if parse_failed:
            failed += 1
            failed_log.append({"index": i, "raw": raw})
            if failed <= 5:
                print(f"  [{i+1}] PARSE FAIL: {str(raw)[:120]}")
        predictions.append(pred)

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}] rate={rate:.2f} samples/s, eta={eta:.0f}s")

    # Metrics (reuse evaluate.compute_metrics)
    from evaluate import compute_metrics
    gt_list = [s["ground_truth"] for s in samples]
    metrics = compute_metrics(gt_list, predictions, parse_failed_flags)

    print("\n" + "=" * 60)
    print(f"GGUF Results — {args.mode} set ({len(samples)} samples)")
    print(f"Parse failures: {failed}/{len(samples)}")
    print(f"Time: {time.time() - t0:.0f}s")
    print("=" * 60)
    for n in NUTRIENTS:
        print(f"  {n:10s}: {metrics[n]['mae_pct']:.1f}% MAE%")
    avg = np.mean([metrics[n]["mae_pct"] for n in NUTRIENTS])
    print(f"  {'avg':10s}: {avg:.1f}%")

    # Save CSV
    csv_path = Path(args.output_dir) / f"per_sample_results_{args.mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "true_calories", "pred_calories", "true_protein", "pred_protein",
            "true_fat", "pred_fat", "true_carbs", "pred_carbs", "parse_failed",
        ])
        for i, (s, p, pf) in enumerate(zip(samples, predictions, parse_failed_flags)):
            gt = s["ground_truth"]
            writer.writerow([
                i, gt["calories"], p.get("calories", 0),
                gt["protein"], p.get("protein", 0), gt["fat"], p.get("fat", 0),
                gt["carbs"], p.get("carbs", 0), pf,
            ])
    print(f"\nPer-sample CSV saved to {csv_path}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "server_url": args.server_url,
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
