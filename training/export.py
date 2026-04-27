"""Export a fine-tuned HuggingFace model to GGUF format."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to fine-tuned HF model")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="Quantization type")
    args = parser.parse_args()

    # TODO: implement GGUF conversion
    print(f"Export config: {vars(args)}")


if __name__ == "__main__":
    main()
