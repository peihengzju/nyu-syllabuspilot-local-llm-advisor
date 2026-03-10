#!/usr/bin/env bash
set -euo pipefail

API_BASE="${QWEN_API_BASE:-http://127.0.0.1:8000}"
curl -sS "${API_BASE}/v1/models"
