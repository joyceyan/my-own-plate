#!/usr/bin/env bash
# Run the full HF pipeline + GGUF quantization benchmark on the held-out test split.
# Long operations are wrapped with `caffeinate` to prevent macOS sleep.
#
# Usage (from repo root):
#   tmux new-session -d -s mop-benchmark './training/run_test_benchmark.sh'
#   tmux detach -s mop-benchmark
#
# Follow progress:
#   tail -f training/logs/benchmark.log

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p training/logs

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found. Create it first: python3 -m venv .venv" | tee -a training/logs/benchmark.log
    exit 1
fi

caffeinate -i -s bash <<'EOF' 2>&1 | tee -a training/logs/benchmark.log
set -euo pipefail

cd /Users/joyce/src/my-own-plate
source .venv/bin/activate

PYTHON="/Users/joyce/src/my-own-plate/.venv/bin/python"
LLAMA_CPP="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"

# Avoid Xet backend compatibility issues on this machine
export HF_HUB_DISABLE_XET=1
BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_DIR="/Users/joyce/src/my-own-plate/training/output/adapter"
VISION_LORA="/Users/joyce/src/my-own-plate/training/output/vision_lora.pt"
GGUF_DIR="/Users/joyce/src/my-own-plate/training/output/gguf"
DATASET_DIR="/Users/joyce/src/my-own-plate/data/nutrition5k_hf_chat"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# --- 1. Ensure base model is cached -------------------------------------------
log 'Downloading/cache-checking base HF model...'
${PYTHON} - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-2B-Instruct')
PY

# --- 2. Build llama-server if missing -----------------------------------------
if [[ ! -f "${LLAMA_CPP}/build/bin/llama-server" ]]; then
    log 'Building llama-server...'
    cmake --build "${LLAMA_CPP}/build" --target llama-server -j
fi

# --- 3. Export fine-tuned model to GGUF ---------------------------------------
log 'Exporting fine-tuned model to GGUF (F16 + Q4_K_M + Q8_0)...'
rm -f "${GGUF_DIR}"/.myownplate-*.gguf.*
${PYTHON} training/merge_and_export.py \
    --model "${BASE_MODEL}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --vision-lora "${VISION_LORA}" \
    --output-dir "${GGUF_DIR}" \
    --llama-cpp-dir "${LLAMA_CPP}" \
    --vision-rank 64 --vision-alpha 64 \
    --projector-rank 128 --projector-alpha 128

# --- 4. Verify dataset images are reachable -----------------------------------
log 'Verifying Nutrition5k imagery is reachable from the test split...'
${PYTHON} - <<'PY' || {
    log 'ERROR: Cannot load test images. Place the Nutrition5k imagery tree at /Users/joyce/src/Nutrition5k (or update the paths in the dataset) and re-run.'
    exit 1
}
import sys
from datasets import load_dataset
try:
    ds = load_dataset('parquet', data_dir='/Users/joyce/src/my-own-plate/data/nutrition5k_hf_chat', split='test')
    _ = ds[0]['image']
    sys.exit(0)
except Exception as e:
    print(f'Image load check failed: {e}', file=sys.stderr)
    sys.exit(1)
PY

# --- 5. HF adapter evaluation on test split -----------------------------------
log 'Evaluating HF adapter model on test split...'
${PYTHON} training/evaluate.py \
    --mode test \
    --model "${BASE_MODEL}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --vision-lora "${VISION_LORA}" \
    --dataset "${DATASET_DIR}" \
    --output-dir training/eval_results_hf \
    --image-size 384 \
    --lora-rank-vision 64 --lora-alpha-vision 64 \
    --lora-rank-projector 128 --lora-alpha-projector 128 \
    2>&1 | tee -a training/logs/eval_hf_test.log

# --- 6. GGUF F16 evaluation ---------------------------------------------------
log 'Evaluating GGUF F16 model on test split...'
"${LLAMA_CPP}/build/bin/llama-server" \
    -m "${GGUF_DIR}/myownplate-f16.gguf" \
    --mmproj "${GGUF_DIR}/mmproj-myownplate-f16.gguf" \
    --image-min-tokens 130 --image-max-tokens 130 \
    --port 8081 -ngl 99 -c 2048 > training/logs/server_f16.log 2>&1 &
SERVER_PID=$!
for i in {1..60}; do
    if curl -sf http://localhost:8081/health >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -sf http://localhost:8081/health >/dev/null 2>&1; then
    log 'ERROR: llama-server (F16) failed to start'
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    exit 1
fi
${PYTHON} training/eval_gguf_parquet.py \
    --mode test \
    --server-url http://localhost:8081 \
    --dataset "${DATASET_DIR}" \
    --output-dir training/eval_results_gguf_f16 \
    2>&1 | tee -a training/logs/eval_gguf_f16_test.log
kill "${SERVER_PID}" >/dev/null 2>&1 || true
sleep 2

# --- 7. GGUF Q4_K_M evaluation ----------------------------------------------
log 'Evaluating GGUF Q4_K_M model on test split...'
"${LLAMA_CPP}/build/bin/llama-server" \
    -m "${GGUF_DIR}/myownplate-q4_k_m.gguf" \
    --mmproj "${GGUF_DIR}/mmproj-myownplate-f16.gguf" \
    --image-min-tokens 130 --image-max-tokens 130 \
    --port 8082 -ngl 99 -c 2048 > training/logs/server_q4km.log 2>&1 &
SERVER_PID=$!
for i in {1..60}; do
    if curl -sf http://localhost:8082/health >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -sf http://localhost:8082/health >/dev/null 2>&1; then
    log 'ERROR: llama-server (Q4_K_M) failed to start'
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    exit 1
fi
${PYTHON} training/eval_gguf_parquet.py \
    --mode test \
    --server-url http://localhost:8082 \
    --dataset "${DATASET_DIR}" \
    --output-dir training/eval_results_gguf_q4km \
    2>&1 | tee -a training/logs/eval_gguf_q4km_test.log
kill "${SERVER_PID}" >/dev/null 2>&1 || true

log 'Benchmark complete. Check training/eval_results_* for results.'
EOF
