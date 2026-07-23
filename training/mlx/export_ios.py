"""
Export fine-tuned Qwen3-VL model for iOS: load base + LoRA, fuse, quantize (4-bit), save.

The vision tower is kept at full precision (bf16). Only the language model
is quantized to 4-bit with group_size=64, matching mlx-vlm conventions.
"""

import argparse
import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


def parse_args():
    parser = argparse.ArgumentParser(description="Export fine-tuned model for iOS")
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3-VL-2B-Instruct-bf16",
        help="Base model HF ID or local path",
    )
    parser.add_argument(
        "--adapter-path", type=str, default="./output/adapters",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/ios_model",
        help="Output directory for the iOS-ready model",
    )
    parser.add_argument(
        "--q-bits", type=int, default=4, help="Quantization bits",
    )
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base model + LoRA adapter
    # ------------------------------------------------------------------
    from mlx_vlm.utils import load

    print(f"[1/5] Loading model with adapter from {args.adapter_path}")
    model, processor = load(
        args.model,
        adapter_path=args.adapter_path,
        processor_config={"trust_remote_code": True},
    )

    # ------------------------------------------------------------------
    # 2. Fuse LoRA layers into base weights
    # ------------------------------------------------------------------
    print("[2/5] Fusing LoRA layers")
    from mlx_vlm.trainer.lora import LoRaLayer

    fused_updates = []
    for name, module in model.named_modules():
        if isinstance(module, LoRaLayer):
            W = module.original_layer.weight
            A = module.A  # (input_dims, rank)
            B = module.B  # (rank, output_dims)
            alpha = module.alpha
            W_fused = W + alpha * (B.T @ A.T).astype(W.dtype)

            new_linear = nn.Linear(W_fused.shape[1], W_fused.shape[0])
            new_linear.weight = W_fused
            if hasattr(module.original_layer, "bias") and module.original_layer.bias is not None:
                new_linear.bias = module.original_layer.bias

            fused_updates.append((name, new_linear))

    if fused_updates:
        model.update_modules(tree_unflatten(fused_updates))
        print(f"  Fused {len(fused_updates)} LoRA layers")
    else:
        print("  Warning: no LoRA layers found")

    # ------------------------------------------------------------------
    # 3. Quantize (language model only, skip vision tower)
    # ------------------------------------------------------------------
    print(f"[3/5] Quantizing language model to {args.q_bits}-bit (group_size={args.q_group_size})")

    from mlx_vlm.convert import skip_multimodal_module

    def quant_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        model_pred = getattr(model, "quant_predicate", None)
        if model_pred is not None:
            return model_pred(path, module)
        return True

    def class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % args.q_group_size != 0:
            return False
        return quant_predicate(path, module)

    nn.quantize(model, args.q_group_size, args.q_bits, class_predicate=class_predicate)

    # Compute bits per weight
    total_bits = 0
    total_params = 0
    for _, v in tree_flatten(model.parameters()):
        total_params += v.size
        total_bits += v.size * v.dtype.size * 8
    bpw = total_bits / total_params if total_params > 0 else 0
    print(f"  {bpw:.2f} bits per weight")

    # ------------------------------------------------------------------
    # 4. Save weights
    # ------------------------------------------------------------------
    print(f"[4/5] Saving to {output_dir}")
    from mlx_vlm.convert import save_weights
    save_weights(output_dir, model, donate_weights=True)

    # ------------------------------------------------------------------
    # 5. Copy config and tokenizer files from base model
    # ------------------------------------------------------------------
    print("[5/5] Copying config and tokenizer files")
    from huggingface_hub import snapshot_download
    base_path = Path(snapshot_download(args.model))

    # Build config with quantization info
    config_path = base_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy all other necessary files
    copy_patterns = ["*.json", "*.txt", "*.jinja", "*.model"]
    for pattern in copy_patterns:
        for src in base_path.glob(pattern):
            if src.name in ("config.json", "model.safetensors.index.json"):
                continue  # Already handled or generated
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {src.name}")

    # Report final size
    total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"\nDone! Output: {output_dir}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
