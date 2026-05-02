"""
Diagnostic script: compare the training image pipeline vs the eval image pipeline.

Processes a single image through both paths and reports:
  1. Whether pixel_values exist (or are None)
  2. pixel_values shape and dtype
  3. Number of image_pad tokens in input_ids
  4. Total input_ids length
  5. Whether the vision tower would be used

Usage:
    cd training && python diagnose_pipeline.py
"""

import os
import sys
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_token(input_ids, token_id):
    """Count occurrences of a token ID in input_ids (mx.array or list)."""
    if isinstance(input_ids, mx.array):
        arr = input_ids.flatten().tolist()
    elif isinstance(input_ids, list):
        arr = input_ids
    else:
        arr = list(input_ids)
    return arr.count(token_id)


def describe_tensor(name, t):
    """Print shape/dtype info for an mx.array or None."""
    if t is None:
        print(f"  {name}: None  <-- VISION TOWER WILL BE BYPASSED")
    elif isinstance(t, mx.array):
        print(f"  {name}: shape={t.shape}, dtype={t.dtype}")
    elif isinstance(t, np.ndarray):
        print(f"  {name}: shape={t.shape}, dtype={t.dtype} (numpy)")
    elif isinstance(t, list):
        print(f"  {name}: list of len={len(t)}")
    else:
        print(f"  {name}: type={type(t).__name__}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_dir = os.path.expanduser("~/src/my-own-plate/data/nutrition5k_hf")
    model_id = "mlx-community/Qwen3-VL-2B-Instruct-bf16"
    image_resize_shape = [384, 384]

    print("=" * 70)
    print("TRAIN/EVAL IMAGE PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model & processor (shared by both pipelines)
    # ------------------------------------------------------------------
    print("\n[1] Loading model and processor...")
    from mlx_vlm.utils import load
    model, processor = load(model_id, processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    model_type = getattr(model.config, "model_type", None)
    print(f"  Model type: {model_type}")
    print(f"  use_embedded_images: {model_type and model_type.startswith('qwen')}")

    # Get special token IDs
    image_token_id = getattr(model.config, "image_token_index", None) or \
                     getattr(model.config, "image_token_id", None) or \
                     config.get("image_token_index") or config.get("image_token_id")
    print(f"  image_token_id: {image_token_id}")

    # ------------------------------------------------------------------
    # Load one sample from the HF dataset
    # ------------------------------------------------------------------
    print("\n[2] Loading one sample from HF dataset...")
    ds = load_dataset(dataset_dir, split="validation")
    sample = ds[0]
    pil_image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]
    print(f"  Question: {question[:80]}...")
    print(f"  Answer: {answer[:80]}...")
    print(f"  Image size (PIL): {pil_image.size} (w x h)")

    # Also find the original image path from the JSONL
    jsonl_path = os.path.expanduser("~/src/my-own-plate/data/nutrition5k_validation.jsonl")
    image_path = None
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            first_line = json.loads(f.readline())
            image_path = first_line.get("image_path")
            if image_path:
                image_path = os.path.expanduser(image_path)
    print(f"  Image path (for eval): {image_path}")

    # ==================================================================
    # PATH A: BROKEN TRAINING PIPELINE (upstream VisionDataset)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PATH A: BROKEN TRAINING PIPELINE (upstream VisionDataset)")
    print("=" * 70)

    from mlx_vlm.lora import transform_dataset_to_messages
    from mlx_vlm.trainer.datasets import VisionDataset

    # Replicate what train.py used to do
    one_sample_ds = ds.select([0])
    one_sample_ds = transform_dataset_to_messages(one_sample_ds, model_type)

    broken_dataset = VisionDataset(
        one_sample_ds, config, processor, image_resize_shape=image_resize_shape,
    )

    print("\n  Processing sample through VisionDataset.__getitem__(0)...")
    broken_item = broken_dataset[0]

    print("\n  --- Broken training pipeline outputs ---")
    for key in sorted(broken_item.keys()):
        describe_tensor(key, broken_item[key])

    broken_input_ids = broken_item["input_ids"]
    if broken_input_ids is not None and image_token_id is not None:
        n_img_tokens_broken = count_token(broken_input_ids, image_token_id)
        print(f"\n  image_pad tokens in input_ids: {n_img_tokens_broken}")
        print(f"  total tokens in input_ids: {broken_input_ids.size}")

    broken_pv = broken_item.get("pixel_values")

    # ==================================================================
    # PATH A-FIXED: FIXED TRAINING PIPELINE (FixedVisionDataset)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PATH A-FIXED: FIXED TRAINING PIPELINE (FixedVisionDataset)")
    print("=" * 70)

    from train import FixedVisionDataset

    fixed_dataset = FixedVisionDataset(
        one_sample_ds, config, processor, image_resize_shape=image_resize_shape,
    )

    print("\n  Processing sample through FixedVisionDataset.__getitem__(0)...")
    fixed_item = fixed_dataset[0]

    print("\n  --- Fixed training pipeline outputs ---")
    for key in sorted(fixed_item.keys()):
        describe_tensor(key, fixed_item[key])

    train_input_ids = fixed_item["input_ids"]
    if train_input_ids is not None and image_token_id is not None:
        n_img_tokens = count_token(train_input_ids, image_token_id)
        print(f"\n  image_pad tokens in input_ids: {n_img_tokens}")
        print(f"  total tokens in input_ids: {train_input_ids.size}")

    train_pv = fixed_item.get("pixel_values")
    train_has_pixels = train_pv is not None

    # ==================================================================
    # PATH B: EVAL PIPELINE (apply_chat_template + generate)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PATH B: EVAL PIPELINE (apply_chat_template + generate)")
    print("=" * 70)

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    # Replicate what evaluate.py's infer_mlx does
    num_images = 1 if (image_path and os.path.exists(image_path)) else 0
    if num_images == 0:
        print("  WARNING: No image path found, using PIL image from dataset")
        # Save PIL image to temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        pil_image.save(tmp.name)
        image_path = tmp.name
        num_images = 1

    formatted = apply_chat_template(
        processor, config, question, num_images=num_images
    )
    print(f"\n  Formatted prompt (first 200 chars): {formatted[:200]}...")

    # Count image_pad tokens in the formatted text
    image_pad_text_count = formatted.count("<|image_pad|>")
    print(f"  <|image_pad|> occurrences in formatted text: {image_pad_text_count}")

    # Now call prepare_inputs the way stream_generate does (WITH images)
    from mlx_vlm.generate import normalize_resize_shape
    resize_shape = normalize_resize_shape(None)  # eval doesn't pass resize_shape
    eval_image_token_index = getattr(model.config, "image_token_index", None)

    eval_inputs = prepare_inputs(
        processor,
        images=[image_path],
        audio=None,
        prompts=formatted,
        image_token_index=eval_image_token_index,
        resize_shape=resize_shape,  # None — this is the eval default
    )

    print("\n  --- Eval pipeline outputs ---")
    for key in sorted(eval_inputs.keys()):
        describe_tensor(key, eval_inputs[key])

    eval_input_ids = eval_inputs.get("input_ids")
    if eval_input_ids is not None and image_token_id is not None:
        n_img_tokens_eval = count_token(eval_input_ids, image_token_id)
        total_tokens_eval = eval_input_ids.flatten().tolist().__len__() if isinstance(eval_input_ids, mx.array) else len(eval_input_ids)
        print(f"\n  image_pad tokens in input_ids: {n_img_tokens_eval}")
        print(f"  total tokens in input_ids: {total_tokens_eval}")

    eval_pv = eval_inputs.get("pixel_values")
    eval_has_pixels = eval_pv is not None

    # ==================================================================
    # PATH C: EVAL PIPELINE WITH 384x384 RESIZE (matching training param)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PATH C: EVAL PIPELINE WITH resize_shape=[384,384]")
    print("=" * 70)

    resize_shape_384 = normalize_resize_shape(image_resize_shape)
    eval_inputs_384 = prepare_inputs(
        processor,
        images=[image_path],
        audio=None,
        prompts=formatted,
        image_token_index=eval_image_token_index,
        resize_shape=resize_shape_384,
    )

    print("\n  --- Eval pipeline (384x384) outputs ---")
    for key in sorted(eval_inputs_384.keys()):
        describe_tensor(key, eval_inputs_384[key])

    eval_input_ids_384 = eval_inputs_384.get("input_ids")
    if eval_input_ids_384 is not None and image_token_id is not None:
        n_img_tokens_384 = count_token(eval_input_ids_384, image_token_id)
        total_tokens_384 = eval_input_ids_384.flatten().tolist().__len__() if isinstance(eval_input_ids_384, mx.array) else len(eval_input_ids_384)
        print(f"\n  image_pad tokens in input_ids: {n_img_tokens_384}")
        print(f"  total tokens in input_ids: {total_tokens_384}")

    eval_pv_384 = eval_inputs_384.get("pixel_values")

    # ==================================================================
    # COMPARISON SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    def pv_desc(pv):
        if pv is None:
            return "None"
        elif isinstance(pv, mx.array):
            return f"{list(pv.shape)}"
        return str(type(pv).__name__)

    def tok_count(ids):
        if ids is None:
            return "N/A"
        if isinstance(ids, mx.array):
            return str(ids.size)
        return str(len(ids))

    def grid_desc(g):
        if g is None:
            return "None"
        if isinstance(g, mx.array):
            return str(g.tolist())
        return str(g)

    def img_tok_count(ids):
        if ids is None:
            return "N/A"
        return str(count_token(ids, image_token_id))

    print(f"\n  {'Metric':<30} {'Broken':>15} {'FIXED':>15} {'Eval(native)':>15} {'Eval(384)':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'pixel_values':<30} {pv_desc(broken_pv):>15} {pv_desc(train_pv):>15} {pv_desc(eval_pv):>15} {pv_desc(eval_pv_384):>15}")
    print(f"  {'image_pad tokens':<30} {img_tok_count(broken_input_ids):>15} {img_tok_count(train_input_ids):>15} {img_tok_count(eval_input_ids):>15} {img_tok_count(eval_input_ids_384):>15}")
    print(f"  {'total tokens':<30} {tok_count(broken_input_ids):>15} {tok_count(train_input_ids):>15} {tok_count(eval_input_ids):>15} {tok_count(eval_input_ids_384):>15}")

    fixed_grid = fixed_item.get("image_grid_thw")
    broken_grid = broken_item.get("image_grid_thw")
    eval_grid = eval_inputs.get("image_grid_thw")
    eval_grid_384 = eval_inputs_384.get("image_grid_thw")
    print(f"  {'image_grid_thw':<30} {grid_desc(broken_grid):>15} {grid_desc(fixed_grid):>15} {grid_desc(eval_grid):>15} {grid_desc(eval_grid_384):>15}")
    print(f"  {'vision tower used':<30} {'NO':>15} {('YES' if train_has_pixels else 'NO'):>15} {'YES':>15} {'YES':>15}")

    # ------------------------------------------------------------------
    # DIAGNOSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Check broken pipeline
    print("\n  Broken (upstream VisionDataset):")
    if broken_pv is None:
        print("    CRITICAL: pixel_values=None — vision tower bypassed, text-only training")
    else:
        print("    pixel_values present")

    # Check fixed pipeline
    print("\n  Fixed (FixedVisionDataset):")
    if not train_has_pixels:
        print("    FAIL: pixel_values still None after fix!")
    else:
        print("    OK: pixel_values present — vision tower WILL be used")

        # Check alignment with eval pipeline (384x384)
        fixed_img_tok = count_token(train_input_ids, image_token_id)
        eval_img_tok_384 = count_token(eval_input_ids_384, image_token_id)

        if fixed_img_tok == eval_img_tok_384:
            print(f"    OK: image_pad token count matches eval (384x384): {fixed_img_tok}")
        else:
            print(f"    WARN: image_pad token count differs from eval (384x384): "
                  f"train={fixed_img_tok}, eval={eval_img_tok_384}")

        if train_pv is not None and eval_pv_384 is not None:
            if isinstance(train_pv, mx.array) and isinstance(eval_pv_384, mx.array):
                if train_pv.shape == eval_pv_384.shape:
                    print(f"    OK: pixel_values shape matches eval (384x384): {list(train_pv.shape)}")
                else:
                    print(f"    WARN: pixel_values shape differs: "
                          f"train={list(train_pv.shape)}, eval={list(eval_pv_384.shape)}")

    print()


if __name__ == "__main__":
    main()
