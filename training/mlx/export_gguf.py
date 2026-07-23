"""
Export fine-tuned Qwen3-VL-2B to GGUF via the HuggingFace/PyTorch path.

This avoids the MLX→HF key remapping issues by:
1. Loading the original HF PyTorch model
2. Converting MLX LoRA adapters to HF key naming
3. Merging LoRA weights into the base model in PyTorch
4. Saving in HF format (correct key names for llama.cpp)
5. Converting to GGUF using llama.cpp's convert_hf_to_gguf.py

Usage:
    python mlx/export_gguf.py --adapter-path ./mlx/output/adapters --output-dir ./mlx/output/gguf
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Key mapping: MLX adapter keys → HuggingFace model keys
# ---------------------------------------------------------------------------

def mlx_key_to_hf_key(mlx_key: str) -> str:
    """Convert an MLX model key to the equivalent HuggingFace key.

    MLX (mlx-vlm):          language_model.model.layers.0.self_attn.q_proj
    HuggingFace (PyTorch):  model.layers.0.self_attn.q_proj

    MLX:  vision_tower.blocks.0.attn.qkv
    HF:   visual.blocks.0.attn.qkv
    """
    k = mlx_key
    if k.startswith("language_model.model."):
        k = "model." + k[len("language_model.model."):]
    elif k.startswith("language_model."):
        k = k[len("language_model."):]
    elif k.startswith("vision_tower."):
        k = "visual." + k[len("vision_tower."):]
    return k


def hf_key_to_mlx_key(hf_key: str) -> str:
    """Inverse of mlx_key_to_hf_key."""
    k = hf_key
    if k.startswith("model."):
        k = "language_model.model." + k[len("model."):]
    elif k.startswith("visual."):
        k = "vision_tower." + k[len("visual."):]
    return k


# ---------------------------------------------------------------------------
# LoRA merge
# ---------------------------------------------------------------------------

def merge_lora_into_hf(base_weights: dict, adapter_weights: dict, lora_rank: int = 64) -> dict:
    """Merge MLX-format LoRA adapters into HuggingFace-format base weights.

    MLX LoRA stores A=[in_dim, rank] and B=[rank, out_dim].
    The merged weight is: W' = W + (A @ B).T
    Base weight W is [out_dim, in_dim] in PyTorch convention.
    """
    # Group adapter keys by prefix
    lora_prefixes = set()
    for k in adapter_weights:
        if k.endswith(".A"):
            lora_prefixes.add(k[:-2])

    print(f"  Merging {len(lora_prefixes)} LoRA pairs...")

    merged = dict(base_weights)  # copy
    applied = 0

    for mlx_prefix in sorted(lora_prefixes):
        A = adapter_weights[f"{mlx_prefix}.A"].float()  # [in, rank]
        B = adapter_weights[f"{mlx_prefix}.B"].float()  # [rank, out]

        # Map MLX key to HF key
        hf_key = mlx_key_to_hf_key(mlx_prefix) + ".weight"

        if hf_key not in merged:
            print(f"  WARNING: {hf_key} not found in base model, skipping {mlx_prefix}")
            continue

        base_w = merged[hf_key].float()  # [out, in]
        delta = (A @ B).T  # [out, in]

        if base_w.shape != delta.shape:
            print(f"  WARNING: shape mismatch for {hf_key}: base={base_w.shape} delta={delta.shape}")
            continue

        merged[hf_key] = (base_w + delta).to(merged[hf_key].dtype)
        applied += 1

    print(f"  Applied {applied}/{len(lora_prefixes)} LoRA pairs")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF")
    parser.add_argument("--adapter-path", type=str, default="~/src/my-own-plate/training/mlx/output/adapters")
    parser.add_argument("--output-dir", type=str, default="./output/gguf")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--llama-cpp-dir", type=str, default=os.path.expanduser("~/src/llama.cpp"))
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        help="Quantization type for language model (q4_k_m, q8_0, f16)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: Download base model -----
    print("Step 1: Downloading base model...")
    from huggingface_hub import snapshot_download
    base_path = Path(snapshot_download(args.base_model))
    print(f"  Base model at: {base_path}")

    # ----- Step 2: Load base weights -----
    print("Step 2: Loading base model weights...")
    base_weights = {}
    for sf_file in sorted(base_path.glob("*.safetensors")):
        w = load_file(str(sf_file))
        base_weights.update(w)
    print(f"  Loaded {len(base_weights)} base weight tensors")

    # ----- Step 3: Load and merge LoRA adapters -----
    print("Step 3: Loading LoRA adapters...")
    adapter_path = Path(args.adapter_path)
    adapter_weights = load_file(str(adapter_path / "adapters.safetensors"))
    print(f"  Loaded {len(adapter_weights)} adapter tensors")

    print("Step 4: Merging LoRA into base model...")
    merged = merge_lora_into_hf(base_weights, adapter_weights)

    # ----- Step 4: Save merged model in HF format -----
    print("Step 5: Saving merged model in HF format...")
    save_file(merged, str(merged_dir / "model.safetensors"))

    # Copy config and tokenizer files from base model
    for fname in base_path.iterdir():
        if fname.suffix in (".json", ".txt", ".model", ".jinja") or fname.name in ("merges.txt", "vocab.json"):
            shutil.copy2(fname, merged_dir / fname.name)
    print(f"  Saved to: {merged_dir}")

    # ----- Step 5: Convert to GGUF -----
    convert_script = Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: llama.cpp not found at {args.llama_cpp_dir}")
        sys.exit(1)

    # Language model (f16 first, then quantize)
    f16_gguf = output_dir / "myownplate-f16.gguf"
    final_gguf = output_dir / f"myownplate-{args.quantize}.gguf"

    print("Step 6: Converting language model to GGUF f16...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ], check=True)

    if args.quantize != "f16":
        quantize_bin = Path(args.llama_cpp_dir) / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            print(f"ERROR: llama-quantize not found. Build it first:")
            print(f"  cd {args.llama_cpp_dir} && cmake -B build -DGGML_METAL=ON && cmake --build build --target llama-quantize")
            sys.exit(1)

        print(f"Step 7: Quantizing to {args.quantize}...")
        subprocess.run([
            str(quantize_bin),
            str(f16_gguf),
            str(final_gguf),
            args.quantize.upper().replace("_", "_"),
        ], check=True)
        # Clean up f16 intermediate
        f16_gguf.unlink()
    else:
        final_gguf = f16_gguf

    # Vision projector (mmproj)
    mmproj_gguf = output_dir / "mmproj-myownplate-f16.gguf"
    print("Step 8: Converting vision projector to GGUF...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(mmproj_gguf),
        "--mmproj",
    ], check=True)

    # Clean up merged HF directory
    shutil.rmtree(merged_dir)

    # Report
    print(f"\n=== Export complete ===")
    print(f"  Language model: {final_gguf} ({final_gguf.stat().st_size / 1e9:.2f} GB)")
    print(f"  Vision proj:    {mmproj_gguf} ({mmproj_gguf.stat().st_size / 1e6:.0f} MB)")
    print(f"\nTo evaluate:")
    print(f"  ~/src/llama.cpp/build/bin/llama-mtmd-cli \\")
    print(f"    -m {final_gguf} \\")
    print(f"    --mmproj {mmproj_gguf} \\")
    print(f"    --image /path/to/food.jpg \\")
    print(f'    -p "Estimate the nutritional content..." \\')
    print(f"    --temp 0.0 -n 256")


if __name__ == "__main__":
    main()
