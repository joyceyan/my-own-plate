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
from mlx_vlm.trainer.datasets import VisionDataset, get_prompt
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import print_trainable_parameters
from mlx_vlm.utils import load, prepare_inputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed VisionDataset — passes images through the vision pipeline
# ---------------------------------------------------------------------------

class FixedVisionDataset:
    """
    Drop-in replacement for mlx_vlm's VisionDataset that actually processes
    images through the vision pipeline for Qwen models.

    Bug: VisionDataset sets images=None for Qwen (use_embedded_images=True),
    which causes prepare_inputs to take the text-only path. Result: pixel_values
    is None, the vision tower is bypassed, and training is text-only.

    Fix: always pass images to prepare_inputs so pixel_values are computed.
    """

    def __init__(self, hf_dataset, config, processor, image_resize_shape=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = tuple(image_resize_shape) if image_resize_shape else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        conversations = item.get("messages", item.get("conversations"))
        model_type = self.config.get("model_type")
        prompt = get_prompt(model_type, self.processor, conversations)

        image_token_index = self.config.get("image_token_index") or \
                            self.config.get("image_token_id")

        inputs = prepare_inputs(
            processor=self.processor,
            images=images if images else None,
            prompts=[prompt],
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )

        # prepare_inputs returns tensors with a leading batch dim (1, N).
        # iterate_batches expects per-sample tensors — squeeze batch dim
        # from input_ids/attention_mask so len() returns the sequence length.
        result = {}
        for k, v in inputs.items():
            if isinstance(v, mx.array) and v.ndim >= 2 and v.shape[0] == 1:
                if k in ("input_ids", "attention_mask"):
                    v = v.squeeze(0)
            result[k] = v

        return result


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
    parser.add_argument("--lora-rank", type=int, default=64)
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
    train_dataset = FixedVisionDataset(
        train_ds, config, processor, image_resize_shape=args.image_resize,
    )

    # Load validation dataset
    val_dataset = None
    try:
        logger.info(f"Loading validation data: {args.train_data} (split={args.val_split})")
        val_ds = load_dataset(args.train_data, split=args.val_split)
        val_ds = transform_dataset_to_messages(val_ds, model_type)
        val_dataset = FixedVisionDataset(
            val_ds, config, processor, image_resize_shape=args.image_resize,
        )
        logger.info(f"Validation: {len(val_ds)} samples")
    except (ValueError, KeyError):
        logger.warning(f"No '{args.val_split}' split found — training without validation")

    # ------------------------------------------------------------------
    # Setup LoRA — LLM attn+MLP + VL merger (vision-language projector)
    # ------------------------------------------------------------------
    from mlx_vlm.trainer.utils import (
        get_peft_model, LoRaLayer, set_module_by_name, freeze_model
    )
    import mlx.nn as nn

    # 1) LLM layers
    llm_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        model, llm_modules,
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0, verbose=False,
    )

    # 2) VL merger + deepstack mergers (vision→language bridge, NOT vision blocks)
    merger_lora_count = 0
    for merger_module in [model.vision_tower.merger,
                          *model.vision_tower.deepstack_merger_list]:
        for name, module in merger_module.named_modules():
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                lora_layer = LoRaLayer(module, args.lora_rank, args.lora_alpha, 0.0)
                set_module_by_name(merger_module, name, lora_layer)
                merger_lora_count += 1
    print(f"Applied LoRA to {merger_lora_count} VL merger layers")

    # 3) Vision tower top-N blocks (lower rank to avoid disrupting pretrained features)
    vision_lora_rank = 16  # conservative rank for vision blocks
    vision_blocks = model.vision_tower.blocks
    num_vision_blocks = 8  # top 8 blocks
    vision_lora_count = 0
    for block in vision_blocks[-num_vision_blocks:]:
        for name, module in block.named_modules():
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                lora_layer = LoRaLayer(module, vision_lora_rank, args.lora_alpha, 0.0)
                set_module_by_name(block, name, lora_layer)
                vision_lora_count += 1
    print(f"Applied LoRA (r{vision_lora_rank}) to top {num_vision_blocks} vision blocks ({vision_lora_count} layers)")
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
