#!/usr/bin/env bash
# End-to-end demonstration of the MCP security gateway.
#
# Starts the mock downstream MCP server and the gateway on local ports, sends
# representative requests with curl, prints the HTTP status and JSON body of
# each, shows what the downstream actually received, and shuts both down.
#
#   PYTHON=/path/to/venv/bin/python bash task2-mcp-gateway/scripts/demo.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/task2-mcp-gateway/src${PYTHONPATH:+:$PYTHONPATH}"

DOWNSTREAM_PORT="${DOWNSTREAM_PORT:-9100}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"
GATEWAY="http://127.0.0.1:${GATEWAY_PORT}"
DOWNSTREAM="http://127.0.0.1:${DOWNSTREAM_PORT}"
ADMIN_TOKEN="dev-only-admin-token-not-for-production-0001"
VIEWER_TOKEN="dev-only-viewer-token-not-for-production-0002"
LOG_DIR="$(mktemp -d)"

cleanup() {
  kill "${MOCK_PID:-}" "${GW_PID:-}" 2>/dev/null || true
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

"$PYTHON" -m mock_mcp_server --port "$DOWNSTREAM_PORT" >"$LOG_DIR/downstream.log" 2>&1 &
MOCK_PID=$!
MCP_GATEWAY_DOWNSTREAM_URL="$DOWNSTREAM/mcp" \
MCP_GATEWAY_TOKENS_FILE="$ROOT/task2-mcp-gateway/config/tokens.dev.json" \
MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS=true \
"$PYTHON" -m mcp_gateway --port "$GATEWAY_PORT" >"$LOG_DIR/gateway.log" 2>&1 &
GW_PID=$!
wait_for "$DOWNSTREAM/healthz"
wait_for "$GATEWAY/healthz"

call() {
  local title="$1" token="$2" body="$3"
  echo
  echo "## $title"
  local auth=()
  if [[ -n "$token" ]]; then auth=(-H "Authorization: Bearer $token"); fi
  curl -sS -o "$LOG_DIR/body" -w "HTTP %{http_code}\n" -X POST "$GATEWAY/mcp" \
    -H "Content-Type: application/json" "${auth[@]}" -d "$body"
  cat "$LOG_DIR/body"; echo
}

call "No credentials: rejected before anything is parsed" "" \
  '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

call "Unknown token" "not-a-token" \
  '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

call "Viewer: tools/list is forwarded transparently" "$VIEWER_TOKEN" \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

call "Viewer: ordinary tool call is forwarded" "$VIEWER_TOKEN" \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Ankara"}}}'

call "Viewer: admin_reset_key is intercepted with -32001 (downstream never called)" "$VIEWER_TOKEN" \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"user_id":"u-1"}}}'

call "Viewer: casing does not bypass the prefix rule" "$VIEWER_TOKEN" \
  '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"Admin_Delete_User","arguments":{"user_id":"u-1"}}}'

call "Admin: admin_reset_key is forwarded" "$ADMIN_TOKEN" \
  '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"user_id":"u-1"}}}'

call "Identity propagation: downstream sees X-Gateway-* headers, never the bearer token" "$VIEWER_TOKEN" \
  '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"whoami"}}'

call "Fail closed: tools/call without a tool name" "$ADMIN_TOKEN" \
  '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{}}'

call "Batch containing an unauthorised call is rejected as a whole" "$VIEWER_TOKEN" \
  '[{"jsonrpc":"2.0","id":9,"method":"ping"},{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"admin_reset_key"}}]'

call "Malformed JSON" "$VIEWER_TOKEN" '{"jsonrpc": 2.0, '

echo
echo "## Tool calls that reached the downstream (via its debug endpoint)"
curl -sS "$DOWNSTREAM/__mock/calls" | "$PYTHON" -c 'import json,sys; [print(" -", c["tool"], "as", c["identity"]["x-gateway-subject"], "/", c["identity"]["x-gateway-role"], "| auth header present:", c["authorization_present"]) for c in json.load(sys.stdin)]'

echo
echo "## Gateway audit log (authz decisions)"
cat > "$LOG_DIR/format_audit.py" <<'PY'
import json, sys
for line in sys.stdin:
    record = json.loads(line)
    print(f" - {record['subject']:<6} {record['method']:<12} tool={str(record.get('tool')):<18} {record['verdict']:<7} {record['reason']}")
PY
grep '"authz.decision"' "$LOG_DIR/gateway.log" | "$PYTHON" "$LOG_DIR/format_audit.py"
