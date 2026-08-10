"""
Evaluate GGUF model via llama-server's OpenAI-compatible API.

Prerequisites:
    Start the server first:
    ~/src/llama.cpp/build/bin/llama-server \
        -m training/output/gguf/myownplate-q4km.gguf \
        --mmproj training/output/gguf/mmproj-myownplate-f16.gguf \
        --port 8080 -ngl 99 -c 2048

Usage:
    python eval_gguf_server.py --mode val
    python eval_gguf_server.py --mode test
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

NUTRIENTS = ["calories", "protein", "fat", "carbs"]
PROMPT = "Estimate the nutritional content of this food image. Respond as JSON with keys: calories (kcal), protein (g), fat (g), carbs (g)."
SERVER_URL = "http://localhost:8080"


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def infer(image_path: str) -> str | None:
    """Send image + prompt to llama-server and return raw text output."""
    b64 = image_to_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def parse_nutrition(raw: str) -> dict | None:
    """Parse nutrition JSON from model output."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # Find JSON block
    match = re.search(r"\{[^}]*\"calories\"[^}]*\}", cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
            if all(n in parsed and parsed[n] is not None for n in NUTRIENTS):
                return {n: float(parsed[n]) for n in NUTRIENTS}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Regex fallback
    values = {}
    for n in NUTRIENTS:
        m = re.search(rf'["\']?{n}["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', raw, re.IGNORECASE)
        if m:
            values[n] = float(m.group(1))
    if len(values) == len(NUTRIENTS):
        return values
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to the JSONL file to evaluate (default: ../data/nutrition5k_{val|test}.jsonl)",
    )
    args = parser.parse_args()

    global SERVER_URL
    SERVER_URL = args.server_url

    # Check server health
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert r.json()["status"] == "ok"
    except Exception:
        print(f"ERROR: llama-server not running at {SERVER_URL}")
        print("Start it with:")
        print("  ~/src/llama.cpp/build/bin/llama-server \\")
        print("    -m training/output/gguf/myownplate-q4km.gguf \\")
        print("    --mmproj training/output/gguf/mmproj-myownplate-f16.gguf \\")
        print("    --port 8080 -ngl 99 -c 2048")
        sys.exit(1)

    # Load data
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = f"../data/nutrition5k_{'validation' if args.mode == 'val' else 'test'}.jsonl"
    with open(data_file) as f:
        samples = [json.loads(line) for line in f]

    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Evaluating {len(samples)} {args.mode} samples via llama-server...")
    print(f"Model: GGUF Q4_K_M + mmproj F16")
    print()

    errors = {n: [] for n in NUTRIENTS}
    parse_failures = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        img = os.path.expanduser(sample["image_path"])
        expected = json.loads(sample["completion"]) if isinstance(sample["completion"], str) else sample["completion"]

        raw = infer(img)
        if raw is None:
            parse_failures += 1
            continue

        pred = parse_nutrition(raw)
        if pred is None:
            parse_failures += 1
            if parse_failures <= 5:
                print(f"  [{i+1}] PARSE FAIL: {raw[:100]}")
            continue

        for n in NUTRIENTS:
            if expected[n] > 0:
                err = abs(pred[n] - expected[n]) / expected[n] * 100
                errors[n].append(err)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            avg = np.mean([np.mean(errors[n]) for n in NUTRIENTS if errors[n]])
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}] avg MAE%: {avg:.1f}% ({rate:.1f} samples/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"GGUF Q4_K_M Results — {args.mode} set ({len(samples)} samples)")
    print(f"Parse failures: {parse_failures}/{len(samples)}")
    print(f"Time: {elapsed:.0f}s ({len(samples)/elapsed:.1f} samples/s)")
    print(f"{'='*60}")

    total = []
    for n in NUTRIENTS:
        mae = np.mean(errors[n]) if errors[n] else float("nan")
        print(f"  {n:10s}: {mae:.1f}%")
        total.append(mae)
    avg = np.mean(total)
    print(f"  {'avg':10s}: {avg:.1f}%")

    print(f"\nComparison:")
    print(f"  N5k baseline (Thames):  30.4%")
    print(f"  Qwen3-VL-2B base:       59.2%")
    print(f"  Fine-tuned bf16:        18.8%")
    print(f"  Fine-tuned GGUF Q4_K_M: {avg:.1f}%")


if __name__ == "__main__":
    main()
