#!/bin/bash
set -e
BASE_URL="${1:-http://localhost:8994}"
MODEL="${2:-ollama/glm-5-1-v2}"

echo "=== Test 1: Basic chat (should have thinking) ==="
RESP=$(curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50,\"temperature\":0}")
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]['message']; print('  content:', (c.get('content') or '')[:80]); print('  reasoning:', 'YES' if c.get('reasoning_content') else 'NO')"

echo ""
echo "=== Test 2: Structured output (may or may not think) ==="
RESP=$(curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}],\"response_format\":{\"type\":\"json_object\"},\"max_tokens\":50,\"temperature\":0}")
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]['message']; print('  content:', (c.get('content') or '')[:80]); print('  reasoning:', 'YES' if c.get('reasoning_content') else 'NO')"

echo ""
echo "=== Test 3: Streaming chat ==="
curl -sN "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20,\"temperature\":0,\"stream\":true}" \
  | head -5
echo "  (streaming OK)"

echo ""
echo "=== Test 4: Check think metric ==="
METRICS_PORT="${4:-9904}"
METRICS=$(curl -sf "http://localhost:${METRICS_PORT}/metrics" 2>/dev/null || echo "metrics unavailable")
echo "$METRICS" | grep "smg_response_thinking_total" || echo "  MISSING: smg_response_thinking_total"

echo ""
echo "=== Test 5: Check param log (requires --log-request-params) ==="
LOG_FILE="${3:-/home/connorli/smg_think_metrics_log.txt}"
if [ -f "$LOG_FILE" ]; then
  PARAM_LINES=$(grep "smg::request_params" "$LOG_FILE" | wc -l)
  echo "  request_params log lines: $PARAM_LINES"
  if [ "$PARAM_LINES" -gt 0 ]; then
    echo "  Sample:"
    grep "smg::request_params" "$LOG_FILE" | tail -1 | head -c 500
    echo ""
    if grep "smg::request_params" "$LOG_FILE" | grep -q "What is 2+2"; then
      echo "  FAIL: message content leaked into param log!"
    else
      echo "  OK: no message content in param log"
    fi
  fi
else
  echo "  Log file not found: $LOG_FILE"
fi

echo ""
echo "=== Test 6: Request with tools (param log should show tool names only) ==="
curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}}}}}],\"max_tokens\":50,\"temperature\":0}" > /dev/null 2>&1
if [ -f "$LOG_FILE" ]; then
  LAST_PARAM=$(grep "smg::request_params" "$LOG_FILE" | tail -1)
  if echo "$LAST_PARAM" | grep -q "get_weather"; then
    echo "  OK: tool name visible in param log"
  fi
  if echo "$LAST_PARAM" | grep -q "Get weather for a location"; then
    echo "  FAIL: tool description leaked into param log!"
  else
    echo "  OK: tool description NOT in param log"
  fi
fi

echo ""
echo "=== DONE ==="
