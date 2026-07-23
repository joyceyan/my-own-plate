#!/bin/bash
# Download base model, merge LoRA, convert to GGUF, quantize, and benchmark.
# Run: nohup bash training/download_and_export.sh > training/output/pipeline.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

DEST="$HOME/src/my-own-plate/training/mlx/output/merged_hf/model.safetensors"
URL="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors"
EXPECTED_SIZE=4264218424
GGUF_DIR="$HOME/src/my-own-plate/training/mlx/output/gguf"
LLAMA_CPP="$HOME/src/llama.cpp"
MAX_ATTEMPTS=50

mkdir -p "$(dirname "$DEST")" "$GGUF_DIR"

check_complete() {
    if [ -f "$DEST" ]; then
        local size=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
        [ "$size" -ge "$EXPECTED_SIZE" ] && return 0
    fi
    return 1
}

# ===== PHASE 1: Download =====
echo "[$(date)] === PHASE 1: Download base model ==="

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    if check_complete; then
        echo "[$(date)] Download complete!"
        break
    fi

    echo "[$(date)] Download attempt $attempt..."

    # Try curl with resume first (fast if partially downloaded)
    # Exponential timeout: 1h, 2h, 4h, 8h, 16h
    timeout_secs=$((3600 * (1 << (attempt - 1 < 5 ? attempt - 1 : 4))))
    echo "[$(date)] Using timeout: ${timeout_secs}s ($((timeout_secs/3600))h)"
    curl -L -C - -o "$DEST" \
        --connect-timeout 120 \
        --max-time $timeout_secs \
        --retry 10 \
        --retry-delay 30 \
        --retry-max-time 1800 \
        "$URL" || true

    if check_complete; then
        echo "[$(date)] Download complete on attempt $attempt!"
        break
    fi

    local_size=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
    echo "[$(date)] Attempt $attempt: $local_size / $EXPECTED_SIZE bytes"

    # If curl consistently fails to resume, try fresh download with wget
    if [ "$local_size" -lt "$((EXPECTED_SIZE / 2))" ]; then
        echo "[$(date)] Progress too slow, trying wget..."
        rm -f "$DEST"
        wget -c --retry-connrefused --tries=10 --timeout=120 \
            --waitretry=15 --read-timeout=600 \
            -O "$DEST" "$URL" || true
    fi

    if check_complete; then break; fi

    echo "[$(date)] Sleeping 60s before retry..."
    sleep 60
done

if ! check_complete; then
    echo "[$(date)] FATAL: Could not download model after $MAX_ATTEMPTS attempts."
    exit 1
fi

echo "[$(date)] Download verified: $(ls -lh "$DEST" | awk '{print $5}')"

# ===== PHASE 2: Merge LoRA =====
echo ""
echo "[$(date)] === PHASE 2: Merge LoRA adapters ==="

python3 << 'PYEOF'
import torch, json
from safetensors.torch import load_file, save_file
from pathlib import Path

WORK_DIR = Path.home() / "src/my-own-plate/training/output/merged_hf"
base = load_file(str(WORK_DIR / "model.safetensors"))
adapters = load_file(str(Path.home() / "src/my-own-plate/training/mlx/output/adapters/adapters.safetensors"))

# mlx-vlm uses alpha as raw multiplier (NOT alpha/rank)
scale = 1.0

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

# ===== PHASE 3: Convert to GGUF =====
echo ""
echo "[$(date)] === PHASE 3: Convert to GGUF ==="

# Remove stale index file
rm -f training/mlx/output/merged_hf/model.safetensors.index.json

# Language model f16
cd "$LLAMA_CPP" && python convert_hf_to_gguf.py \
    "$HOME/src/my-own-plate/training/output/merged_hf" \
    --outfile "$GGUF_DIR/myownplate-f16.gguf" \
    --outtype f16

echo "[$(date)] F16 GGUF created."

# Quantize to Q4_K_M
"$LLAMA_CPP/build/bin/llama-quantize" \
    "$GGUF_DIR/myownplate-f16.gguf" \
    "$GGUF_DIR/myownplate-q4km.gguf" \
    Q4_K_M

echo "[$(date)] Q4_K_M quantization complete."

# mmproj
python convert_hf_to_gguf.py \
    "$HOME/src/my-own-plate/training/output/merged_hf" \
    --outfile "$GGUF_DIR/mmproj-myownplate-f16.gguf" \
    --mmproj

echo "[$(date)] mmproj created."

# Clean up intermediate
rm -f "$GGUF_DIR/myownplate-f16.gguf"

echo "[$(date)] GGUF files:"
ls -lh "$GGUF_DIR/"

# ===== PHASE 4: Benchmark =====
echo ""
echo "[$(date)] === PHASE 4: Benchmark ==="

cd "$HOME/src/my-own-plate"

# Start server
"$LLAMA_CPP/build/bin/llama-server" \
    -m "$GGUF_DIR/myownplate-q4km.gguf" \
    --mmproj "$GGUF_DIR/mmproj-myownplate-f16.gguf" \
    --port 8080 -ngl 99 -c 2048 \
    2>/dev/null &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 30); do
    if curl -s http://localhost:8080/health | grep -q ok; then
        echo "[$(date)] Server ready."
        break
    fi
    sleep 2
done

# Run eval on validation set
cd training/mlx && python eval_gguf_server.py --mode val 2>&1 | tee output/eval_gguf_results.txt

# Kill server
kill $SERVER_PID 2>/dev/null

# ===== PHASE 5: Update comparison.md =====
echo ""
echo "[$(date)] === PHASE 5: Update iOS bundle ==="

# Copy GGUFs to iOS bundle
cp "$GGUF_DIR/myownplate-q4km.gguf" "$HOME/src/my-own-plate/ios/MyOwnPlate/Resources/GGUFModels/myownplate-q4km.gguf"
cp "$GGUF_DIR/mmproj-myownplate-f16.gguf" "$HOME/src/my-own-plate/ios/MyOwnPlate/Resources/GGUFModels/mmproj-myownplate-f16.gguf"

echo "[$(date)] iOS bundle updated."

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "Eval results: training/mlx/output/eval_gguf_results.txt"
echo "GGUF model:   $GGUF_DIR/myownplate-q4km.gguf"
echo "mmproj:       $GGUF_DIR/mmproj-myownplate-f16.gguf"
echo "iOS bundle:   updated"
echo ""
echo "Next: rebuild iOS app"
echo "  cd ios && xcodebuild build -project MyOwnPlate.xcodeproj -scheme MyOwnPlate -destination 'id=F403B2DB-C13E-5085-81F9-EC4E25C4E038' -allowProvisioningUpdates"
