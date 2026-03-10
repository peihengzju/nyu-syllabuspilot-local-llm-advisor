#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
ASK_URL="$BASE_URL/ask"
DEBUG_URL="$BASE_URL/debug_retrieval"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-240}"

ASK_QUESTIONS=(
  "Hi, I’m a first-year ECE student interested in AI infra. What should I focus on this semester?"
  "For ECE-GY 6143, what is the grading breakdown?"
  "Does ECE-GY 6143 have a midterm and final exam?"
  "What are the prerequisites for ECE-GY 6483?"
  "Who is the instructor for ECE-GY 6913 and what are office hours?"
  "Compare ECE-GY 6143 vs ECE-GY 6483 for someone targeting embedded AI systems."
  "I like low-level systems and CPU architecture. Which course should I take?"
  "Is attendance required in ECE-GY 6483?"
  "What textbook or course materials are used in ECE-GY 6143?"
  "What weekly topics are covered in ECE-GY 6913?"
  "My goal is to become an AI systems engineer, and I prefer project-heavy courses."
  "Given what I just said, should I take ECE-GY 6143 or ECE-GY 6913 first?"
)

DEBUG_QUESTIONS=(
  "ECE-GY 6143 grading percentage for homework and exams"
  "ECE-GY 6483 class time and room"
  "ECE-GY 6913 final exam date"
)

json_escape() {
  python3 - <<'PY' "$1"
import json,sys
print(json.dumps(sys.argv[1]))
PY
}

call_api() {
  local url="$1"
  local q="$2"
  local body
  body="{\"question\":$(json_escape "$q")}"

  curl -sS -X POST "$url" \
    --max-time "$REQUEST_TIMEOUT_SEC" \
    -H 'Content-Type: application/json' \
    -d "$body" \
    -w '\n__HTTP_CODE__:%{http_code}\n'
}

pretty_print_ask() {
  python3 -c '
import json,sys
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception:
    print(raw)
    raise SystemExit(0)
print("request_id: {}".format(data.get("request_id", "N/A")))
ans = data.get("answer","")
print(ans if ans else raw)
'
}

pretty_print_debug() {
  python3 -c '
import json,sys
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception:
    print(raw)
    raise SystemExit(0)
print("request_id: {}".format(data.get("request_id", "N/A")))
print("refined: {}".format(data.get("refined", "")))
print("num_contexts: {}".format(data.get("num_contexts", "")))
dbg = data.get("debug",{})
print("mode: {}".format(dbg.get("mode", "")))
print("selected_course: {}".format(dbg.get("selected_course", "")))
print("route_source: {}".format(dbg.get("route_source", "")))
'
}

echo "== Smoke Test: /ask =="

echo "Preflight checks..."
if ! curl -sS --max-time 5 "$BASE_URL/" >/dev/null; then
  echo "[ERROR] Flask is not reachable at $BASE_URL"
  exit 1
fi
if ! curl -sS --max-time 5 "http://127.0.0.1:8000/v1/models" >/dev/null; then
  echo "[ERROR] Local Qwen API is not reachable at http://127.0.0.1:8000"
  exit 1
fi
echo "Preflight OK. timeout=${REQUEST_TIMEOUT_SEC}s"

echo "Warmup /ask ..."
if ! call_api "$ASK_URL" "hello" >/dev/null; then
  echo "[ERROR] Warmup request failed or timed out."
  exit 1
fi
echo "Warmup done."

for i in "${!ASK_QUESTIONS[@]}"; do
  q="${ASK_QUESTIONS[$i]}"
  echo
  echo "[$((i+1))] Q: $q"
  set +e
  resp="$(call_api "$ASK_URL" "$q" 2>&1)"
  code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    echo "[ERROR] curl failed on case $((i+1)) (exit=$code)"
    echo "$resp"
    exit 1
  fi
  http_code="$(printf '%s\n' "$resp" | awk -F: '/^__HTTP_CODE__/{print $2}' | tail -n1)"
  body="$(printf '%s\n' "$resp" | sed '/^__HTTP_CODE__:/d')"
  if [[ "$http_code" != "200" ]]; then
    echo "[ERROR] HTTP $http_code on case $((i+1))"
    echo "$body"
    exit 1
  fi
  printf '%s\n' "$body" | pretty_print_ask
  echo "------------------------------------------------------------"
done

echo
echo "== Retrieval Debug: /debug_retrieval =="
for i in "${!DEBUG_QUESTIONS[@]}"; do
  q="${DEBUG_QUESTIONS[$i]}"
  echo
  echo "[D$((i+1))] Q: $q"
  set +e
  resp="$(call_api "$DEBUG_URL" "$q" 2>&1)"
  code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    echo "[ERROR] curl failed on case D$((i+1)) (exit=$code)"
    echo "$resp"
    exit 1
  fi
  http_code="$(printf '%s\n' "$resp" | awk -F: '/^__HTTP_CODE__/{print $2}' | tail -n1)"
  body="$(printf '%s\n' "$resp" | sed '/^__HTTP_CODE__:/d')"
  if [[ "$http_code" != "200" ]]; then
    echo "[ERROR] HTTP $http_code on case D$((i+1))"
    echo "$body"
    exit 1
  fi
  printf '%s\n' "$body" | pretty_print_debug
  echo "------------------------------------------------------------"
done

echo
echo "Done. BASE_URL=$BASE_URL"
