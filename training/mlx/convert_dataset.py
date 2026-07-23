"""
Convert Nutrition5k JSONL into a HuggingFace Dataset with train/val/test splits.

Reads nutrition5k_all.jsonl (produced by data/prepare_nutrition5k.py),
splits into train/validation/test, and saves:
  - HuggingFace parquet dataset (for mlx-vlm training)
  - Per-split JSONL files (for human readability and evaluate.py)

All split logic lives here — prepare_nutrition5k.py just cleans and outputs
a single JSONL with all valid samples.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Value


def load_jsonl(path: str):
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row["image_path"]
            if not os.path.exists(img_path):
                skipped += 1
                continue
            records.append({
                "image": img_path,
                "question": row["prompt"],
                "answer": row["completion"],
                # Keep original fields for JSONL re-export
                "_image_path": row["image_path"],
                "_prompt": row["prompt"],
                "_completion": row["completion"],
            })
    if skipped:
        print(f"  Skipped {skipped} samples with missing images")
    return records


def write_split_jsonl(records, path):
    """Write a split back to JSONL format for human readability."""
    with open(path, "w") as f:
        for r in records:
            entry = {
                "image_path": r["_image_path"],
                "prompt": r["_prompt"],
                "completion": r["_completion"],
            }
            f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split Nutrition5k JSONL into train/val/test HuggingFace datasets"
    )
    parser.add_argument(
        "--input", type=str,
        default="~/src/my-own-plate/data/nutrition5k_all.jsonl",
        help="Path to the complete JSONL (from prepare_nutrition5k.py)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="~/src/my-own-plate/data/nutrition5k_hf",
        help="Output directory for HuggingFace parquet dataset",
    )
    parser.add_argument(
        "--jsonl-dir", type=str,
        default="~/src/my-own-plate/data",
        help="Output directory for per-split JSONL files",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.80,
        help="Fraction for training (default: 0.80)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="Fraction for validation (default: 0.10)",
    )
    # test ratio is implicitly 1 - train - val
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.input = os.path.expanduser(args.input)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.jsonl_dir = os.path.expanduser(args.jsonl_dir)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print(f"Error: train_ratio + val_ratio > 1.0")
        return

    # Load all records
    print(f"Loading data from {args.input}")
    all_records = load_jsonl(args.input)
    print(f"  {len(all_records)} valid samples")

    # Group records by dish ID to prevent leakage across splits.
    # A dish may have multiple images (overhead, side angles, augmentations) —
    # all images for a given dish must land in the same split.
    dish_to_records = {}
    for r in all_records:
        # Extract dish ID from path like .../dish_1572029300/rgb.png
        match = re.search(r'dish_\d+', r["_image_path"])
        dish_id = match.group() if match else r["_image_path"]
        dish_to_records.setdefault(dish_id, []).append(r)

    dish_ids = list(dish_to_records.keys())
    print(f"  {len(dish_ids)} unique dishes")

    # Shuffle and split at the dish level
    np.random.seed(args.seed)
    dish_ids = [dish_ids[i] for i in np.random.permutation(len(dish_ids))]

    n_train = int(len(dish_ids) * args.train_ratio)
    n_val = int(len(dish_ids) * args.val_ratio)

    train_dishes = dish_ids[:n_train]
    val_dishes = dish_ids[n_train:n_train + n_val]
    test_dishes = dish_ids[n_train + n_val:]

    train_records = [r for d in train_dishes for r in dish_to_records[d]]
    val_records = [r for d in val_dishes for r in dish_to_records[d]]
    test_records = [r for d in test_dishes for r in dish_to_records[d]]

    print(f"  Split: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")

    # HuggingFace parquet features (exclude internal _fields)
    features = Features({
        "image": Image(),
        "question": Value("string"),
        "answer": Value("string"),
    })

    def to_hf_records(records):
        return [{"image": r["image"], "question": r["question"], "answer": r["answer"]}
                for r in records]

    # Save parquet
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Clean old files first
    for f in out.glob("*.parquet"):
        f.unlink()

    print(f"\nSaving parquet to {out}")
    for name, recs in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        Dataset.from_list(to_hf_records(recs), features=features).to_parquet(out / f"{name}.parquet")

    # Save per-split JSONL
    jsonl_dir = Path(args.jsonl_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving JSONL splits to {jsonl_dir}")
    for name, recs in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        write_split_jsonl(recs, jsonl_dir / f"nutrition5k_{name}.jsonl")

    print(f"\nDone.")
    print(f"  Parquet: {out}/{{train,validation,test}}.parquet")
    print(f"  JSONL:   {jsonl_dir}/nutrition5k_{{train,validation,test}}.jsonl")


if __name__ == "__main__":
    main()
