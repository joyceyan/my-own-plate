"""
LoRA fine-tuning for Qwen3-VL-2B-Instruct on the Nutrition5k dataset using mlx-vlm.

Unlike mlx-lm (which strips the vision tower), mlx-vlm keeps the full VL pipeline
so the model actually sees food images during training.

Calls mlx-vlm internals directly (rather than subprocess) to support validation
during training — mlx-vlm's CLI hardcodes val_dataset=None.

Usage:
    python train.py --train-data ~/src/my-own-plate/data/nutrition5k_hf
"""

import argparse
import logging
import os
import sys

import mlx.core as mx
import mlx.optimizers as optim
from datasets import load_dataset

from mlx_vlm.lora import transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import print_trainable_parameters
from mlx_vlm.utils import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3-VL on Nutrition5k (mlx-vlm)"
    )
    parser.add_argument(
        "--train-data", type=str,
        default="~/src/my-own-plate/data/nutrition5k_hf",
        help="Path to HuggingFace dataset directory (run convert_dataset.py first)",
    )
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3-VL-2B-Instruct-bf16",
        help="Model path (must be an mlx-community bf16 model for mlx-vlm)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/adapters/adapters.safetensors",
        help="Output path for adapter weights",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    # NOTE: mlx-vlm uses alpha as a raw multiplier (not alpha/rank like HF PEFT).
    # With rank=16, set alpha=1.0 for standard LoRA scaling.
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Use gradient checkpointing (default: on)")
    parser.add_argument("--image-resize", type=int, nargs=2, default=[384, 384],
                        help="Resize images to this shape (default: 384 384)")
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=500,
                        help="Run validation every N steps (default: 500)")
    parser.add_argument("--val-batches", type=int, default=25,
                        help="Number of validation batches per eval (default: 25)")
    parser.add_argument("--val-split", type=str, default="validation",
                        help="Dataset split to use for validation (default: validation)")
    parser.add_argument("--steps-per-save", type=int, default=100000,
                        help="Save checkpoint every N steps (default: 100000 — only final save)")
    parser.add_argument("--no-compile", action="store_true", default=True,
                        help="Disable mx.compile (default: on, prevents Metal timeout)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable mx.compile (faster but may crash on laptops)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.train_data = os.path.expanduser(args.train_data)
    args.output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Metal safety: disable compile and cap memory
    if args.no_compile and not args.compile:
        mx.disable_compile()
        print("mx.compile disabled (prevents Metal GPU timeout)")

    metal_mem_cap = int(mx.device_info()["memory_size"] * 0.5)
    mx.set_memory_limit(metal_mem_cap)
    mx.set_wired_limit(metal_mem_cap)
    print(f"Metal memory capped at {metal_mem_cap / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {args.model}")
    model, processor = load(
        args.model, processor_config={"trust_remote_code": True}
    )
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    config = model.config.__dict__

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    logger.info(f"Loading training data: {args.train_data} (split=train)")
    train_ds = load_dataset(args.train_data, split="train")

    if args.epochs is not None:
        iters = (len(train_ds) // args.batch_size) * args.epochs
    else:
        iters = len(train_ds)

    train_ds = train_ds.select(range(min(iters, len(train_ds))))
    train_ds = transform_dataset_to_messages(train_ds, model_type)
    train_dataset = VisionDataset(
        train_ds, config, processor, image_resize_shape=args.image_resize,
    )

    # Load validation dataset
    val_dataset = None
    try:
        logger.info(f"Loading validation data: {args.train_data} (split={args.val_split})")
        val_ds = load_dataset(args.train_data, split=args.val_split)
        val_ds = transform_dataset_to_messages(val_ds, model_type)
        val_dataset = VisionDataset(
            val_ds, config, processor, image_resize_shape=args.image_resize,
        )
        logger.info(f"Validation: {len(val_ds)} samples")
    except (ValueError, KeyError):
        logger.warning(f"No '{args.val_split}' split found — training without validation")

    # ------------------------------------------------------------------
    # Setup LoRA — attention + MLP projections
    # ------------------------------------------------------------------
    from mlx_vlm.trainer.utils import get_peft_model
    lora_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP (for numerical regression)
    ]
    model = get_peft_model(
        model,
        lora_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        verbose=False,
    )
    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        val_batches=args.val_batches,
        max_seq_length=args.max_seq_length,
        adapter_file=args.output_dir,
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
    )

    logger.info(f"Starting training: {iters} iters, batch_size={args.batch_size}")
    import time
    t0 = time.time()

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        train_on_completions=True,
    )

    elapsed = time.time() - t0
    mins = elapsed / 60
    print(f"\nTraining complete in {mins:.1f} minutes.")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
