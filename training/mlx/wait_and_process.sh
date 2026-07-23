#!/bin/bash
# Wait for the model download to complete, then run merge+export+eval.
# Run: nohup bash training/wait_and_process.sh > training/output/process.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

DEST="$HOME/src/my-own-plate/training/mlx/output/merged_hf/model.safetensors"
EXPECTED_SIZE=4264218424
GGUF_DIR="$HOME/src/my-own-plate/training/mlx/output/gguf"
LLAMA_CPP="$HOME/src/llama.cpp"

echo "[$(date)] Waiting for model download to complete..."
echo "[$(date)] Expected: $EXPECTED_SIZE bytes at $DEST"

# Wait for download
while true; do
    if [ -f "$DEST" ]; then
        size=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
        if [ "$size" -ge "$EXPECTED_SIZE" ]; then
            echo "[$(date)] Download complete! Size: $size bytes"
            break
        fi
        echo "[$(date)] Still downloading: $((size / 1048576)) MB / $((EXPECTED_SIZE / 1048576)) MB"
    else
        echo "[$(date)] File not found yet..."
    fi
    sleep 60
done

# Wait a bit for any file system sync
sleep 5

# ===== Merge LoRA =====
echo ""
echo "[$(date)] === Merging LoRA adapters ==="

python3 << 'PYEOF'
import torch, json
from safetensors.torch import load_file, save_file
from pathlib import Path

WORK_DIR = Path.home() / "src/my-own-plate/training/output/merged_hf"
base = load_file(str(WORK_DIR / "model.safetensors"))
adapters = load_file(str(Path.home() / "src/my-own-plate/training/mlx/output/adapters/adapters.safetensors"))

scale = 1.0  # mlx-vlm uses alpha as raw multiplier

def mlx_to_hf(k):
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model."):]
    elif k.startswith("vision_tower."):
        return "model.visual." + k[len("vision_tower."):]
    return k

prefixes = set(k[:-2] for k in adapters if k.endswith(".A"))
applied = 0
for p in sorted(prefixes):
    A = adapters[f"{p}.A"].float()
    B = adapters[f"{p}.B"].float()
    hf_key = mlx_to_hf(p) + ".weight"
    if hf_key in base:
        w = base[hf_key].float()
        delta = scale * (A @ B).T
        if w.shape == delta.shape:
            base[hf_key] = (w + delta).to(base[hf_key].dtype)
            applied += 1

print(f"Applied {applied}/{len(prefixes)} LoRA pairs with scale={scale}")
save_file(base, str(WORK_DIR / "model.safetensors"))
print("Merged model saved.")
PYEOF

echo "[$(date)] LoRA merge complete."

# ===== Convert to GGUF =====
echo ""
echo "[$(date)] === Converting to GGUF ==="

rm -f training/mlx/output/merged_hf/model.safetensors.index.json
mkdir -p "$GGUF_DIR"

cd "$LLAMA_CPP" && python convert_hf_to_gguf.py \
    "$HOME/src/my-own-plate/training/output/merged_hf" \
    --outfile "$GGUF_DIR/myownplate-f16.gguf" \
    --outtype f16
echo "[$(date)] F16 GGUF created."

"$LLAMA_CPP/build/bin/llama-quantize" \
    "$GGUF_DIR/myownplate-f16.gguf" \
    "$GGUF_DIR/myownplate-q4km.gguf" \
    Q4_K_M
echo "[$(date)] Q4_K_M quantization complete."

python convert_hf_to_gguf.py \
    "$HOME/src/my-own-plate/training/output/merged_hf" \
    --outfile "$GGUF_DIR/mmproj-myownplate-f16.gguf" \
    --mmproj
echo "[$(date)] mmproj created."

rm -f "$GGUF_DIR/myownplate-f16.gguf"

echo "[$(date)] GGUF files:"
ls -lh "$GGUF_DIR/"

# ===== Benchmark =====
echo ""
echo "[$(date)] === Running benchmark ==="

cd "$HOME/src/my-own-plate"

"$LLAMA_CPP/build/bin/llama-server" \
    -m "$GGUF_DIR/myownplate-q4km.gguf" \
    --mmproj "$GGUF_DIR/mmproj-myownplate-f16.gguf" \
    --port 8080 -ngl 99 -c 2048 \
    2>/dev/null &
SERVER_PID=$!

for i in $(seq 1 30); do
    if curl -s http://localhost:8080/health 2>/dev/null | grep -q ok; then
        echo "[$(date)] Server ready."
        break
    fi
    sleep 2
done

cd training/mlx && python eval_gguf_server.py --mode val 2>&1 | tee output/eval_gguf_results.txt

kill $SERVER_PID 2>/dev/null

# ===== Update iOS bundle =====
echo ""
echo "[$(date)] === Updating iOS bundle ==="

cp "$GGUF_DIR/myownplate-q4km.gguf" "$HOME/src/my-own-plate/ios/MyOwnPlate/Resources/GGUFModels/myownplate-q4km.gguf"
cp "$GGUF_DIR/mmproj-myownplate-f16.gguf" "$HOME/src/my-own-plate/ios/MyOwnPlate/Resources/GGUFModels/mmproj-myownplate-f16.gguf"

echo ""
echo "=========================================="
echo "[$(date)] PIPELINE COMPLETE"
echo "=========================================="
echo "Results: training/mlx/output/eval_gguf_results.txt"
