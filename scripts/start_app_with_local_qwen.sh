#!/usr/bin/env bash
set -euo pipefail

export QWEN_API_URL="${QWEN_API_URL:-http://127.0.0.1:8000/v1/chat/completions}"
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507-FP8}"

exec python3 app.py
