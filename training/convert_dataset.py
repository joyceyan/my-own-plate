"""
Convert Nutrition5k JSONL splits into a HuggingFace Dataset with chat messages.

Reads the train/validation/test JSONL files produced by convert_dataset.py and
writes data/nutrition5k_hf_chat/ as a HuggingFace parquet dataset with columns:
    - image (PIL)
    - messages (standard HF chat format, image as a placeholder dict)

The actual image object is passed to the processor separately; the message
content only contains a placeholder of type 'image' so the chat template renders
<|vision_start|><|image_pad|><|vision_end|>.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value


def load_jsonl(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row["image_path"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["prompt"]},
                    ],
                },
                {"role": "assistant", "content": row["completion"]},
            ]
            records.append({"image": img_path, "messages": messages})
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nutrition5k JSONL splits to HF chat-message dataset"
    )
    parser.add_argument(
        "--jsonl-dir",
        type=str,
        default="~/src/my-own-plate/data",
        help="Directory containing nutrition5k_{train,validation,test}.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/src/my-own-plate/data/nutrition5k_hf_chat",
        help="Output directory for HF chat dataset",
    )
    args = parser.parse_args()

    args.jsonl_dir = Path(args.jsonl_dir).expanduser()
    args.output_dir = Path(args.output_dir).expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = Features({
        "image": Image(),
        "messages": Sequence(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ),
    })

    # Content is normally a string or a list; HF datasets can't easily store
    # heterogeneous nested lists. We store messages as a JSON string instead and
    # parse them in the training script. This keeps the dataset simple and
    # avoids complex schema gymnastics.
    features = Features({
        "image": Image(),
        "messages_json": Value("string"),
    })

    for split in ["train", "validation", "test"]:
        path = args.jsonl_dir / f"nutrition5k_{split}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        records = load_jsonl(str(path))
        # Convert messages to JSON string for storage
        for r in records:
            r["messages_json"] = json.dumps(r.pop("messages"))
        ds = Dataset.from_list(records, features=features)
        ds.to_parquet(args.output_dir / f"{split}.parquet")
        print(f"{split}: {len(ds)} samples -> {args.output_dir / f'{split}.parquet'}")

    print(f"\nDone. Chat dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
