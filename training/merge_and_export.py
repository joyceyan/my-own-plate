"""
Merge HF PEFT + custom vision LoRA adapters into the base model and export to GGUF.

Steps:
  1. Load base HF model + PEFT adapter + custom vision LoRA weights
  2. Merge PEFT LoRA into base weights (merge_and_unload)
  3. Merge custom vision LoRA into base weights (manual fuse)
  4. Save as standard HF safetensors
  5. Convert text model to GGUF Q4_K_M via llama.cpp
  6. Export vision projector to GGUF F16 via llama.cpp --mmproj

Usage:
    source .venv/bin/activate
    python merge_and_export.py \
        --model training/cache/Qwen3-VL-2B-Instruct \
        --adapter-dir training/output/adapter \
        --vision-lora training/output/vision_lora.pt \
        --output-dir training/output/gguf
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

from hf_utils import (
    apply_vision_block_lora,
    apply_projector_lora,
    load_custom_lora,
    merge_custom_lora,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HF adapters and export to GGUF"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="training/cache/Qwen3-VL-2B-Instruct",
        help="Base HF model path or repo ID",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="~/src/my-own-plate/training/output/adapter",
        help="PEFT adapter directory",
    )
    parser.add_argument(
        "--vision-lora",
        type=str,
        default="~/src/my-own-plate/training/output/vision_lora.pt",
        help="Custom vision LoRA checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/src/my-own-plate/training/output/gguf",
        help="Output directory for GGUF files and merged HF model",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default="~/src/llama.cpp",
        help="Path to llama.cpp checkout",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Quantization type for the language model GGUF",
    )
    parser.add_argument("--vision-rank", type=int, default=64)
    parser.add_argument("--vision-alpha", type=int, default=64)
    parser.add_argument("--projector-rank", type=int, default=128)
    parser.add_argument("--projector-alpha", type=int, default=128)
    parser.add_argument(
        "--keep-merged-hf",
        action="store_true",
        help="Keep the merged HF model directory after GGUF conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.adapter_dir = os.path.expanduser(args.adapter_dir)
    args.vision_lora = os.path.expanduser(args.vision_lora)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.llama_cpp_dir = os.path.expanduser(args.llama_cpp_dir)

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged_hf"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load base model
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading base model from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=device,
    )

    # ------------------------------------------------------------------
    # 2. Load PEFT adapter (LLM only), then merge it into base weights
    # ------------------------------------------------------------------
    print("\n[2/6] Loading PEFT adapter and merging")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    # ------------------------------------------------------------------
    # 3. Apply custom vision LoRA scaffolding and load trained weights
    # ------------------------------------------------------------------
    print("\n[3/6] Loading custom vision LoRA")
    apply_vision_block_lora(model, r=args.vision_rank, alpha=args.vision_alpha, dropout=0.0)
    apply_projector_lora(model, r=args.projector_rank, alpha=args.projector_alpha, dropout=0.0)
    load_custom_lora(model, args.vision_lora)

    # ------------------------------------------------------------------
    # 4. Merge custom vision LoRA into base weights
    # ------------------------------------------------------------------
    print("\n[4/6] Merging custom vision LoRA")
    model = merge_custom_lora(model)
    print("All adapters merged into base weights")

    # ------------------------------------------------------------------
    # 5. Save merged HF model
    # ------------------------------------------------------------------
    print(f"\n[5/6] Saving merged HF model to {merged_dir}")
    # Move to CPU for stable saving; safetensors can handle either, but this
    # avoids any MPS-specific quirks during export.
    model = model.to("cpu")
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"Saved merged HF model ({sum(f.stat().st_size for f in merged_dir.rglob('*') if f.is_file()) / 1e9:.2f} GB)")

    # ------------------------------------------------------------------
    # 5. Convert to GGUF
    # ------------------------------------------------------------------
    convert_script = Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: llama.cpp not found at {args.llama_cpp_dir}")
        print("Clone it: git clone https://github.com/ggerganov/llama.cpp.git ~/src/llama.cpp")
        sys.exit(1)

    gguf_lm = output_dir / f"myownplate-{args.quantization}.gguf"
    gguf_f16 = output_dir / "myownplate-f16.gguf"
    gguf_mmproj = output_dir / "mmproj-myownplate-f16.gguf"

    print(f"\n[5/6] Converting language model to GGUF F16")
    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_f16),
        "--outtype", "f16",
    ], check=True)

    print(f"\n[6/6] Quantizing language model to {args.quantization.upper()}")
    quantize_bin = Path(args.llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(f"ERROR: {quantize_bin} not found. Build llama.cpp first.")
        sys.exit(1)
    subprocess.run([
        str(quantize_bin),
        str(gguf_f16),
        str(gguf_lm),
        args.quantization,
    ], check=True)

    print(f"\nQuantizing language model to Q8_0")
    gguf_q8 = output_dir / "myownplate-q8_0.gguf"
    subprocess.run([
        str(quantize_bin),
        str(gguf_f16),
        str(gguf_q8),
        "q8_0",
    ], check=True)

    print(f"\nExporting vision projector to GGUF F16 (--mmproj)")
    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_mmproj),
        "--mmproj",
    ], check=True)

    if not args.keep_merged_hf:
        print(f"\nCleaning up merged HF directory {merged_dir}")
        shutil.rmtree(merged_dir)

    print("\n" + "=" * 60)
    print("Export complete")
    print(f"  LM F16:          {gguf_f16} ({gguf_f16.stat().st_size / 1e9:.2f} GB)")
    print(f"  LM Q8_0:         {gguf_q8} ({gguf_q8.stat().st_size / 1e9:.2f} GB)")
    print(f"  LM {args.quantization.upper():13s} {gguf_lm} ({gguf_lm.stat().st_size / 1e9:.2f} GB)")
    print(f"  Vision projector: {gguf_mmproj} ({gguf_mmproj.stat().st_size / 1e6:.0f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
