"""
LoRA fine-tune Qwen3-VL-2B-Instruct on Nutrition5k using HF transformers + PEFT.

This replaces the mlx-vlm training pipeline to avoid the MLX→GGUF accuracy
degradation. The vision tower is trained with manual LoRA (rank 32), while the
language model and vision-language projector use PEFT LoRA (rank 64).

Prerequisites:
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install -r training/requirements_hf.txt
    python training/convert_dataset_hf.py

Usage:
    source .venv/bin/activate
    python training/train.py --model training/cache/Qwen3-VL-2B-Instruct

The default paths assume the project is at ~/src/my-own-plate.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from peft import get_peft_model

from hf_utils import (
    apply_vision_block_lora,
    apply_projector_lora,
    get_peft_lora_config,
    print_trainable_parameters,
    save_custom_lora,
    LLM_LORA_TARGETS,
)


# ---------------------------------------------------------------------------
# Dataset / data collator
# ---------------------------------------------------------------------------

class NutritionDataset(Dataset):
    """Dataset that returns pre-tokenized tensors for one image+conversation."""

    def __init__(self, hf_dataset, processor, system_prompt=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        messages = json.loads(item["messages_json"])

        # Optionally prepend a system message
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # Full conversation (used for labels)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Prompt-only conversation (used to mask non-completion tokens)
        prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=False
        )

        # Process images and text together; keep the batch dimension
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        if "image_embed_seqlen" in inputs:
            result["image_embed_seqlen"] = inputs["image_embed_seqlen"].squeeze(0)
        return result


class VLDataCollator:
    """Pad sequences and stack image tensors for a batch of VL samples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            seq_len = len(b["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        # Stack pixel_values along the patch dimension (variable patch count is
        # allowed if images have different resolutions, but we force 384x384).
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
        if "image_embed_seqlen" in batch[0]:
            result["image_embed_seqlen"] = torch.stack([b["image_embed_seqlen"] for b in batch])

        return result


class NutritionTrainer(Trainer):
    """Trainer with cosine decay from peak LR down to a configurable minimum LR."""

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer or self.optimizer
            if optimizer is None:
                raise RuntimeError("optimizer is not set")
            self.lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=self.args.min_lr,
            )
            if self.args.lr_scheduler_type != "cosine":
                self.args.lr_scheduler_type = "cosine"
        return self.lr_scheduler


@dataclass
class NutritionTrainingArguments(TrainingArguments):
    """TrainingArguments with an explicit minimum LR for cosine decay."""

    min_lr: float = field(default=1e-6, metadata={"help": "Minimum LR for cosine decay"})

    def __post_init__(self):
        super().__post_init__()

def parse_args():
    parser = argparse.ArgumentParser(
        description="HF/PEFT LoRA fine-tune Qwen3-VL for Nutrition5k"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model HF ID or local directory (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="~/src/my-own-plate/data/nutrition5k_hf_chat",
        help="HF chat dataset directory (default: data/nutrition5k_hf_chat)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/src/my-own-plate/training/output",
        help="Directory for checkpoints and adapters",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Resize images to this square size (default: 384)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank-llm", type=int, default=64)
    parser.add_argument("--lora-alpha-llm", type=int, default=64)
    parser.add_argument(
        "--llm-lora-targets",
        nargs="+",
        default=LLM_LORA_TARGETS,
        help="Target modules for PEFT LLM LoRA (default: all linear projections)",
    )
    parser.add_argument("--lora-rank-vision", type=int, default=32)
    parser.add_argument("--lora-alpha-vision", type=int, default=32)
    parser.add_argument("--lora-rank-projector", type=int, default=None,
                        help="Projector LoRA rank (default: same as --lora-rank-llm)")
    parser.add_argument("--lora-alpha-projector", type=int, default=None,
                        help="Projector LoRA alpha (default: same as --lora-alpha-llm)")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--vision-lora",
        dest="vision_lora",
        action="store_true",
        default=True,
        help="Apply custom LoRA to the vision transformer blocks (default: on)",
    )
    parser.add_argument(
        "--no-vision-lora",
        dest="vision_lora",
        action="store_false",
        help="Disable custom LoRA on vision transformer blocks",
    )
    parser.add_argument(
        "--projector-lora",
        dest="projector_lora",
        action="store_true",
        default=True,
        help="Apply custom LoRA to the vision-language projector (default: on)",
    )
    parser.add_argument(
        "--no-projector-lora",
        dest="projector_lora",
        action="store_false",
        help="Disable custom LoRA on the vision-language projector",
    )
    parser.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (default: off for Exp 1)",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        dest="grad_checkpoint",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit train/val to N samples for smoke testing")
    parser.add_argument(
        "--resume-from-last-checkpoint",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in --output-dir if one exists",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.train_data = os.path.expanduser(args.train_data)
    args.output_dir = os.path.expanduser(args.output_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_checkpoint = None
    if args.resume_from_last_checkpoint:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"\nResuming training from {last_checkpoint}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load processor and model
    # ------------------------------------------------------------------
    print(f"Loading model and processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Force a single square resolution (384x384)
    pixels = args.image_size * args.image_size
    processor.image_processor.min_pixels = pixels
    processor.image_processor.max_pixels = pixels
    print(f"Image processor: min_pixels={pixels}, max_pixels={pixels}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=device,
    )
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    print("\nApplying LoRA adapters")
    if args.vision_lora:
        apply_vision_block_lora(
            model,
            r=args.lora_rank_vision,
            alpha=args.lora_alpha_vision,
            dropout=args.lora_dropout,
        )
    if args.projector_lora:
        proj_r = args.lora_rank_projector if args.lora_rank_projector is not None else args.lora_rank_llm
        proj_alpha = args.lora_alpha_projector if args.lora_alpha_projector is not None else args.lora_alpha_llm
        apply_projector_lora(
            model,
            r=proj_r,
            alpha=proj_alpha,
            dropout=args.lora_dropout,
        )
    lora_config = get_peft_lora_config(
        r=args.lora_rank_llm,
        alpha=args.lora_alpha_llm,
        dropout=args.lora_dropout,
        target_modules=args.llm_lora_targets,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print(f"\nLoading datasets from {args.train_data}")
    train_ds = load_dataset("parquet", data_dir=args.train_data, split="train")
    val_ds = load_dataset("parquet", data_dir=args.train_data, split="validation")
    if args.max_samples is not None:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))
    print(f"Train: {len(train_ds)} | Validation: {len(val_ds)}")

    train_dataset = NutritionDataset(train_ds, processor, system_prompt=args.system_prompt)
    val_dataset = NutritionDataset(val_ds, processor, system_prompt=args.system_prompt)
    collator = VLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    training_args = NutritionTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        min_lr=args.min_lr,
        warmup_ratio=0.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=False,
        bf16=False,
        fp16=torch.float16 == torch.float16 and device.type == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        weight_decay=0.0,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = NutritionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    print(f"\nStarting training: {args.epochs} epochs, lr={args.learning_rate} -> {args.min_lr}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ------------------------------------------------------------------
    # Save adapters
    # ------------------------------------------------------------------
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"\nPEFT adapter saved to {adapter_dir}")

    custom_lora_path = os.path.join(args.output_dir, "vision_lora.pt")
    save_custom_lora(model, custom_lora_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
