#!/usr/bin/env bash
# End-to-end demonstration of the streaming PII guardrail.
#
# Starts the mock provider and the gateway, streams a completion through the
# gateway with curl (unbuffered) so the redacted deltas are visible as they
# arrive, shows the non-streaming path and the Anthropic dialect, and prints the
# gateway's latency/redaction log lines.
#
#   PYTHON=/path/to/venv/bin/python bash task3-llm-guardrail/scripts/demo.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/task3-llm-guardrail/src${PYTHONPATH:+:$PYTHONPATH}"

PROVIDER_PORT="${PROVIDER_PORT:-9200}"
GATEWAY_PORT="${GATEWAY_PORT:-9300}"
PROVIDER="http://127.0.0.1:${PROVIDER_PORT}"
GATEWAY="http://127.0.0.1:${GATEWAY_PORT}"
LOG_DIR="$(mktemp -d)"

cleanup() {
  kill "${PROVIDER_PID:-}" "${GATEWAY_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
  echo
  echo "Logs: $LOG_DIR"
}
trap cleanup EXIT

wait_for() {
  for _ in $(seq 1 100); do
    if curl -fsS "$1" >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done
  echo "timed out waiting for $1" >&2
  exit 1
}

"$PYTHON" -m mock_llm_provider --port "$PROVIDER_PORT" >"$LOG_DIR/provider.log" 2>&1 &
PROVIDER_PID=$!
LLM_GATEWAY_UPSTREAM_BASE_URL="$PROVIDER" LLM_GATEWAY_UPSTREAM_API_KEY="sk-demo" \
LLM_GATEWAY_ALLOW_UNAUTHENTICATED=true \
"$PYTHON" -m llm_guardrail_gateway --port "$GATEWAY_PORT" >"$LOG_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
wait_for "$PROVIDER/healthz"
wait_for "$GATEWAY/healthz"

echo "## 1. What the provider emits (direct, no guardrail), assembled text"
curl -sN "$PROVIDER/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"hello"}],"mock":{"chunk_size":6}}' \
  | "$PYTHON" -c 'import json,sys; print("".join(json.loads(l[5:])["choices"][0]["delta"].get("content","") for l in sys.stdin if l.startswith("data:") and "[DONE]" not in l and json.loads(l[5:]).get("choices")))'

echo
echo "## 2. Through the gateway, raw SSE as it arrives (6-char deltas, 30 ms apart)"
curl -sN "$GATEWAY/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"hello"}],"mock":{"chunk_size":6,"delay_ms":30}}' \
  | tee "$LOG_DIR/stream.sse" | grep --line-buffered '^data:' | cut -c1-150
echo
echo "## 3. Assembled text received by the client"
"$PYTHON" -c 'import json,sys; print("".join(json.loads(l[5:])["choices"][0]["delta"].get("content","") for l in open(sys.argv[1]) if l.startswith("data:") and "[DONE]" not in l and json.loads(l[5:]).get("choices")))' "$LOG_DIR/stream.sse"

echo
echo "## 4. PII placed exactly on 2-character chunk boundaries"
curl -sN "$GATEWAY/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"echo:Card 4111-1111-1111-1111, mail a.b@c.io, SSN 536 22 8041 end"}],"mock":{"chunk_size":2}}' \
  | "$PYTHON" -c 'import json,sys; print("".join(json.loads(l[5:])["choices"][0]["delta"].get("content","") for l in sys.stdin if l.startswith("data:") and "[DONE]" not in l and json.loads(l[5:]).get("choices")))'

echo
echo "## 5. Non-streaming response"
curl -s "$GATEWAY/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mock","stream":false,"messages":[{"role":"user","content":"hello"}]}' \
  | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

echo
echo "## 6. Anthropic Messages dialect, streamed tool input"
curl -sN "$GATEWAY/v1/messages" -H "Content-Type: application/json" \
  -d '{"model":"mock","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hello"}],"mock":{"mode":"tool_call","chunk_size":3}}' \
  | "$PYTHON" -c 'import json,sys; print("".join((json.loads(l[5:]).get("delta") or {}).get("partial_json","") for l in sys.stdin if l.startswith("data:")))'

echo
echo "## 7. Gateway log: first-token latency and redaction counts per request"
cat > "$LOG_DIR/format_log.py" <<'PY'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f" - {r.get('dialect'):<9} ttft_ms={r.get('ttft_ms')!s:<7} upstream_ttfb_ms={r.get('upstream_ttfb_ms')!s:<7} duration_ms={r.get('duration_ms')!s:<8} redactions={r.get('redactions')}")
PY
grep '"stream.finished"\|"response.finished"' "$LOG_DIR/gateway.log" | "$PYTHON" "$LOG_DIR/format_log.py"
