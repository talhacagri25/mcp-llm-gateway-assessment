#!/usr/bin/env bash
# End-to-end demonstration of the rate-limiting and failover router.
#
# Starts two mock providers (primary answers 429 by default, secondary is
# healthy) and the router with the development tenants, then shows failover
# on 429, failover on a 3000 ms timeout, the standardised error payloads and
# the token-aware sliding window filling up for a small tenant.
#
#   PYTHON=/path/to/venv/bin/python bash task4-model-router/scripts/demo.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/task4-model-router/src${PYTHONPATH:+:$PYTHONPATH}"

PRIMARY_PORT="${PRIMARY_PORT:-9401}"
SECONDARY_PORT="${SECONDARY_PORT:-9402}"
ROUTER_PORT="${ROUTER_PORT:-9400}"
ROUTER="http://127.0.0.1:${ROUTER_PORT}"
LOG_DIR="$(mktemp -d)"
DB="$LOG_DIR/model_router.sqlite3"
TENANT_A="dev-only-tenant-a-key-not-for-production-0001"
TENANT_B="dev-only-tenant-b-key-not-for-production-0002"

cleanup() {
  kill "${P1:-}" "${P2:-}" "${P3:-}" 2>/dev/null || true
  wait 2>/dev/null || true
  echo
  echo "Logs and database: $LOG_DIR"
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

MOCK_PROVIDER_NAME=primary MOCK_PROVIDER_BEHAVIOR=rate_limited \
  "$PYTHON" -m mock_model_provider --port "$PRIMARY_PORT" >"$LOG_DIR/primary.log" 2>&1 &
P1=$!
MOCK_PROVIDER_NAME=secondary MOCK_PROVIDER_BEHAVIOR=ok \
  "$PYTHON" -m mock_model_provider --port "$SECONDARY_PORT" >"$LOG_DIR/secondary.log" 2>&1 &
P2=$!
MODEL_ROUTER_PRIMARY_URL="http://127.0.0.1:${PRIMARY_PORT}" \
MODEL_ROUTER_SECONDARY_URL="http://127.0.0.1:${SECONDARY_PORT}" \
MODEL_ROUTER_TENANTS_FILE="$ROOT/task4-model-router/config/tenants.dev.json" \
MODEL_ROUTER_DATABASE_PATH="$DB" \
MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS=true \
  "$PYTHON" -m model_router --port "$ROUTER_PORT" >"$LOG_DIR/router.log" 2>&1 &
P3=$!
wait_for "http://127.0.0.1:${PRIMARY_PORT}/healthz"
wait_for "http://127.0.0.1:${SECONDARY_PORT}/healthz"
wait_for "$ROUTER/healthz"

call() {
  local title="$1" token="$2" body="$3"
  echo
  echo "## $title"
  local start end
  start=$(date +%s%N)
  curl -sS -o "$LOG_DIR/body" -D "$LOG_DIR/headers" -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "$body"
  end=$(date +%s%N)
  echo "   $(head -n1 "$LOG_DIR/headers" | tr -d '\r')   ($(( (end - start) / 1000000 )) ms)"
  { grep -i '^x-model-router-provider\|^x-ratelimit\|^retry-after' "$LOG_DIR/headers" || true; } | sed 's/^/   /' | tr -d '\r'
  echo "   $(cut -c1-220 "$LOG_DIR/body")"
}

BODY='{"model":"demo-model","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'

call "1. Primary answers 429, request is served by the secondary" "$TENANT_A" "$BODY"

call "2. Primary hangs for 4 s: the 3000 ms deadline fires and the secondary serves it" "$TENANT_A" \
  '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"mock":{"primary":{"behavior":"slow","delay_ms":4000}}}'

call "3. Primary healthy for this request: served by the primary" "$TENANT_A" \
  '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"mock":{"primary":{"behavior":"ok"}}}'

call "4. Both providers failing: standardised 503, nothing internal leaks" "$TENANT_A" \
  '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"mock":{"behavior":"error"}}'

call "5. Provider rejects the request (400): relayed with a sanitised message, no failover" "$TENANT_A" \
  '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"mock":{"primary":{"behavior":"bad_request"}}}'

call "6. Invalid request never reaches a provider" "$TENANT_A" '{"model":"demo-model","stream":true,"messages":[{"role":"user","content":"hello"}]}'

echo
echo "## 7. Token-aware sliding window: tenant-b is limited to 2000 tokens per minute"
for i in 1 2 3; do
  call "   request $i for tenant-b (each one is charged about 950 tokens)" "$TENANT_B" \
    '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"max_tokens":950,"mock":{"primary":{"behavior":"ok"},"completion_tokens":940}}'
done

echo
echo "## 8. Current window usage for tenant-b"
curl -sS "$ROUTER/v1/usage" -H "Authorization: Bearer $TENANT_B"; echo

echo
echo "## 9. The on-disk ledger (SQLite) behind those decisions"
"$PYTHON" - "$DB" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
print("   token_ledger:")
for row in conn.execute("SELECT tenant_id, state, tokens, admitted_at_ms FROM token_ledger ORDER BY id"):
    print("    ", row)
print("   request_log outcomes:", conn.execute("SELECT outcome, COUNT(*) FROM request_log GROUP BY outcome").fetchall())
PY

echo
echo "## 10. Router log: provider attempts"
cat > "$LOG_DIR/format_attempts.py" <<'PY'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f" - {r['request_id'][:8]} {r['provider']:<9} {r['outcome']:<13} status={r.get('status')!s:<5} latency_ms={r.get('latency_ms')}")
PY
grep '"router.attempt"' "$LOG_DIR/router.log" | "$PYTHON" "$LOG_DIR/format_attempts.py"
