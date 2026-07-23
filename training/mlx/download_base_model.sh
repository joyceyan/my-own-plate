#!/bin/bash
# Persistent download script for Qwen/Qwen3-VL-2B-Instruct model.safetensors
# Tries multiple methods with retries until successful.
# Run: nohup bash training/download_base_model.sh > training/output/download.log 2>&1 &

set -euo pipefail

DEST="$HOME/src/my-own-plate/training/mlx/output/merged_hf/model.safetensors"
URL="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors"
EXPECTED_SIZE=4264218424  # ~4.26 GB
MAX_ATTEMPTS=100
mkdir -p "$(dirname "$DEST")"

check_complete() {
    if [ -f "$DEST" ]; then
        local size=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
        if [ "$size" -ge "$EXPECTED_SIZE" ]; then
            echo "[$(date)] SUCCESS: Download complete! Size: $size bytes"
            return 0
        fi
    fi
    return 1
}

# Check if already done
if check_complete; then
    exit 0
fi

echo "[$(date)] Starting persistent download of Qwen3-VL-2B-Instruct model.safetensors"
echo "[$(date)] Target: $DEST"
echo "[$(date)] Expected size: $EXPECTED_SIZE bytes (~4.26 GB)"
echo ""

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    if check_complete; then
        break
    fi

    METHOD=$((attempt % 5))
    echo "[$(date)] Attempt $attempt (method $METHOD)..."

    case $METHOD in
        0)
            # Method 0: curl with resume, long timeout
            echo "[$(date)] Using curl with resume..."
            curl -L -C - -o "$DEST" \
                --connect-timeout 30 \
                --max-time 1800 \
                --retry 5 \
                --retry-delay 10 \
                --retry-max-time 300 \
                "$URL" || true
            ;;
        1)
            # Method 1: wget with resume and retries
            echo "[$(date)] Using wget with resume..."
            wget -c --retry-connrefused --tries=5 --timeout=60 \
                --waitretry=10 --read-timeout=300 \
                -O "$DEST" "$URL" || true
            ;;
        2)
            # Method 2: curl with different DNS (use Google DNS via --resolve if needed)
            echo "[$(date)] Using curl fresh download (no resume)..."
            rm -f "$DEST"
            curl -L -o "$DEST" \
                --connect-timeout 60 \
                --max-time 3600 \
                --retry 3 \
                --retry-delay 30 \
                "$URL" || true
            ;;
        3)
            # Method 3: hf_hub_download with hf_transfer
            echo "[$(date)] Using hf_hub_download with hf_transfer..."
            rm -f "$DEST"
            rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/blobs/*.incomplete
            HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('Qwen/Qwen3-VL-2B-Instruct', 'model.safetensors', force_download=True)
shutil.copy2(path, '$DEST')
print('Done')
" || true
            ;;
        4)
            # Method 4: aria2c parallel download (if available)
            echo "[$(date)] Using aria2c (or falling back to curl)..."
            rm -f "$DEST"
            if command -v aria2c &>/dev/null; then
                aria2c -x 16 -s 16 --max-tries=5 --retry-wait=10 \
                    --connect-timeout=30 --timeout=300 \
                    -d "$(dirname "$DEST")" -o "$(basename "$DEST")" \
                    "$URL" || true
            else
                # Install aria2 and retry
                brew install aria2 2>/dev/null || true
                if command -v aria2c &>/dev/null; then
                    aria2c -x 16 -s 16 --max-tries=5 --retry-wait=10 \
                        -d "$(dirname "$DEST")" -o "$(basename "$DEST")" \
                        "$URL" || true
                else
                    echo "[$(date)] aria2c not available, using curl..."
                    curl -L -C - -o "$DEST" --max-time 3600 "$URL" || true
                fi
            fi
            ;;
    esac

    # Check if download succeeded
    if check_complete; then
        echo "[$(date)] Download verified complete on attempt $attempt!"
        break
    else
        local_size=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
        echo "[$(date)] Attempt $attempt failed. Current size: $local_size / $EXPECTED_SIZE bytes"
        echo "[$(date)] Sleeping 30s before retry..."
        sleep 30
    fi
done

if check_complete; then
    echo ""
    echo "=========================================="
    echo "DOWNLOAD COMPLETE"
    echo "=========================================="
    echo "File: $DEST"
    echo "Size: $(ls -lh "$DEST" | awk '{print $5}')"
    echo ""
    echo "Next steps:"
echo "  cd ~/src/my-own-plate/training/mlx"
echo "  python export_gguf.py --adapter-path ./output/adapters --output-dir ./output/gguf"
else
    echo ""
    echo "[$(date)] FAILED after $MAX_ATTEMPTS attempts."
    echo "Try downloading manually or on a different network."
fi
