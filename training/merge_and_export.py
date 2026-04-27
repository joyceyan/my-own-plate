"""
Merge LoRA adapters into the base model and export to GGUF (Q4_K_M).

Steps:
  1. Load base model + LoRA adapter weights via mlx-vlm
  2. Fuse LoRA layers into the base weights
  3. Save the merged model
  4. Convert to GGUF Q4_K_M via llama.cpp's convert_hf_to_gguf.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export to GGUF"
    )
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3-VL-2B-Instruct-bf16",
        help="Base model HF ID or local path",
    )
    parser.add_argument(
        "--adapter-path", type=str, default="./output/adapters",
        help="Path to directory containing adapters.safetensors",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Root output directory",
    )
    parser.add_argument(
        "--llama-cpp-dir", type=str, default=None,
        help="Path to llama.cpp repo (for GGUF conversion). "
             "If not set, searches common locations or $LLAMA_CPP_DIR.",
    )
    return parser.parse_args()


def find_llama_cpp(hint: str = None) -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py."""
    candidates = []
    if hint:
        candidates.append(Path(hint))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.home() / "src" / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]
    for d in candidates:
        script = d / "convert_hf_to_gguf.py"
        if script.is_file():
            return script
    return None


def main():
    args = parse_args()
    args.adapter_path = os.path.expanduser(args.adapter_path)
    args.output_dir = os.path.expanduser(args.output_dir)

    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = Path(args.output_dir) / "myownplate-q4km.gguf"

    # ------------------------------------------------------------------
    # 1. Load base model + adapter, fuse LoRA layers
    # ------------------------------------------------------------------
    from mlx_vlm.utils import load

    print(f"Loading model {args.model} with adapter from {args.adapter_path}")
    model, processor = load(
        args.model,
        adapter_path=args.adapter_path,
        processor_config={"trust_remote_code": True},
    )

    # Fuse any LoRA layers back into the base linear layers
    fused_linears = [
        (n, m.fuse())
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
        print(f"Fused {len(fused_linears)} LoRA layers into base weights")
    else:
        print("Warning: no LoRA layers found to fuse")

    # ------------------------------------------------------------------
    # 2. Save merged model
    # ------------------------------------------------------------------
    print(f"Saving merged model to: {merged_dir}")
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(merged_dir / "model.safetensors"), weights)

    # Copy tokenizer/processor files from the base model cache
    from huggingface_hub import snapshot_download
    base_path = snapshot_download(args.model)
    import shutil
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "preprocessor_config.json",
                  "chat_template.json", "generation_config.json"]:
        src = Path(base_path) / fname
        if src.exists():
            shutil.copy2(src, merged_dir / fname)

    print("Merged model saved.")

    # ------------------------------------------------------------------
    # 3. Convert to GGUF Q4_K_M
    # ------------------------------------------------------------------
    convert_script = find_llama_cpp(args.llama_cpp_dir)
    if convert_script is None:
        print(
            "\nllama.cpp not found. To produce the GGUF file, either:\n"
            "  - Set --llama-cpp-dir /path/to/llama.cpp\n"
            "  - Set env var LLAMA_CPP_DIR=/path/to/llama.cpp\n"
            "  - Clone llama.cpp to ~/src/llama.cpp\n"
            "\nThen re-run this script, or manually run:\n"
            f"  python convert_hf_to_gguf.py {merged_dir} "
            f"--outfile {gguf_path} --outtype q4_k_m"
        )
        sys.exit(1)

    print(f"Converting to GGUF Q4_K_M using: {convert_script}")
    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_path),
        "--outtype", "q4_k_m",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nGGUF conversion failed (exit code {result.returncode}).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Report final size
    # ------------------------------------------------------------------
    if gguf_path.exists():
        size_gb = gguf_path.stat().st_size / (1024 ** 3)
        print(f"\nGGUF file: {gguf_path}")
        print(f"File size: {size_gb:.2f} GB")
    else:
        print(f"\nWarning: expected GGUF at {gguf_path} but file not found.")


if __name__ == "__main__":
    main()
