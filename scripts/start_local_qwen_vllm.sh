#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507-FP8}"
HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-8000}"
GPU_MEM_UTIL="${QWEN_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "No python interpreter found in PATH." >&2
    exit 1
  fi
fi

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN"
