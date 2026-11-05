# Task 2: MCP Security Gateway Proxy

An HTTP/JSON-RPC reverse proxy between AI agent clients and a downstream MCP
server. It authenticates the bearer token, resolves the caller's role,
inspects the JSON-RPC payload, forwards everything except unauthorised
`admin_*` tool calls, and answers those itself with `-32001 Unauthorized Tool
Call`.

## Run it

```bash
make install                 # once, from the repository root
make run-task2-downstream    # terminal 1: mock downstream MCP server on :9100
make run-task2-gateway       # terminal 2: the gateway on :9000, development tokens
make demo-task2              # or: both processes plus a dozen representative requests
pytest task2-mcp-gateway     # 219 tests
```

The gateway needs two settings: `MCP_GATEWAY_DOWNSTREAM_URL` and a token
document (`MCP_GATEWAY_TOKENS_FILE` or `MCP_GATEWAY_TOKENS_JSON`). The token
document is a JSON object with a `tokens` map from each token to its
`subject` and `role`. The committed `config/tokens.dev.json` holds an admin
token and a viewer token and is accepted only with
`MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS=true`, which the run targets and
the demo set. Send requests to `POST /mcp` with `Authorization: Bearer <token>`.

## Answers to the brief

### Requirements

| The brief asks | Delivered | Where |
| --- | --- | --- |
| A lightweight HTTP/JSON-RPC reverse proxy between an agent client and a downstream mock MCP server | A Starlette application with one MCP route plus `/healthz`; the mock downstream is an MCP-shaped JSON-RPC server with `initialize`, `ping`, `tools/list`, `tools/call`, batches and notifications, and a call log that proves what did and did not reach it | `src/mcp_gateway/app.py`, `src/mock_mcp_server/` |
| Read the `Bearer <token>` header and extract the role | RFC 6750 bearer syntax, scheme case-insensitive. The token is compared in constant time against every configured token and resolved to a subject and a role. Missing, malformed or unknown tokens get `401` with a `WWW-Authenticate` challenge and JSON-RPC error `-32002` | `auth.py` |
| `tools/list` forwarded transparently | Forwarded with the original bytes, as is every method other than `tools/call`, including `GET` and `DELETE` on the MCP path used by the Streamable HTTP transport | `policy.py`, `app.py`, `proxy.py` |
| `tools/call` whose `params.name` starts with `admin_` requires the `admin` role, otherwise `-32001 Unauthorized Tool Call` without calling the downstream | Exactly that, with the request id preserved and the reason, tool and required roles in the error data. The protected prefixes and roles are settings (`admin_` and `admin` by default). A `tools/call` whose tool name cannot be determined is refused with `-32602` rather than forwarded | `policy.py`, `app.py` |

### Evaluation criteria

| Criterion | How it is met | Evidence |
| --- | --- | --- |
| Parsing JSON-RPC wire format structures | Single messages and batches; requests, notifications and client-to-server responses; string, finite number and null ids; `params` as object or array. Bodies must be UTF-8. Duplicate object members, NaN and Infinity tokens, oversized numbers, over-long or control-laden method names are refused with `-32700` or `-32600`, keeping the id where it can be determined | `tests/test_jsonrpc.py`, hostile-input cases in `tests/test_gateway_app.py` |
| Proxy middleware construction and HTTP forwarding | The body is relayed byte for byte and the downstream response is streamed back with its status and headers. Hop-by-hop headers, the caller's credentials and any client-supplied identity headers are stripped; `X-Gateway-Subject`, `X-Gateway-Role`, `X-Request-Id` and `X-Forwarded-For` are added; an optional service credential is presented to the downstream. Unreachable downstream is `502` with `-32003`, timeout `504` with `-32004`; a stream that breaks after its headers is logged and the connection closed rather than passed off as complete | `tests/test_gateway_app.py`; `tests/test_sdk_interop.py` puts a real MCP server built on the official SDK behind the gateway and drives it with the official client over sockets |
| Fine-grained authorisation and clean error handling | Prefix rules with configurable prefixes and roles; every decision is one audit line (subject, role, method, tool, verdict, reason, request id). Denied batches are refused as a whole with one error per request in them. Error payloads never echo credentials or arguments, downstream failures carry a correlation id only, and routing errors are JSON-RPC errors too | `tests/test_policy.py`, `tests/test_gateway_app.py` |

Capacity and robustness behaviour beyond the brief: a per-caller cap of 64
concurrent requests (`429` with `-32005`), a bounded downstream connection
pool (`503` with `-32005` when exhausted), bodies capped at 1 MiB and batches
at 100 messages, the concurrency slot released however a response ends,
method and tool names bounded at 256 characters, client headers relayed as
raw bytes, downstream `Server` and `Date` headers dropped, `/healthz` answering
`{"status": "ok"}` only.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_GATEWAY_DOWNSTREAM_URL` | required | Downstream MCP endpoint URL, without embedded credentials |
| `MCP_GATEWAY_TOKENS_FILE` or `MCP_GATEWAY_TOKENS_JSON` | one required | Token document |
| `MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS` | `false` | Accept a token document marked `development_only` |
| `MCP_GATEWAY_PATH` | `/mcp` | Path the gateway serves |
| `MCP_GATEWAY_ADMIN_TOOL_PREFIXES` | `admin_` | Protected tool name prefixes, comma separated |
| `MCP_GATEWAY_ADMIN_ROLES` | `admin` | Roles allowed to call protected tools, comma separated |
| `MCP_GATEWAY_DENIED_HTTP_STATUS` | `200` | `200` or `403` for denied tool calls |
| `MCP_GATEWAY_DOWNSTREAM_TOKEN` | unset | Service credential presented to the downstream |
| `MCP_GATEWAY_MAX_BODY_BYTES` | 1 MiB | Request body cap |
| `MCP_GATEWAY_MAX_BATCH_SIZE` | `100` | Messages per batch |
| `MCP_GATEWAY_MAX_INFLIGHT_PER_SUBJECT` | `64` | Concurrent requests per caller, `0` for no cap |
| `MCP_GATEWAY_MAX_CONNECTIONS` | `1000` | Downstream connection pool size |
| `MCP_GATEWAY_CONNECT_TIMEOUT_SECONDS`, `MCP_GATEWAY_READ_TIMEOUT_SECONDS` | `5`, `30` | Downstream connect timeout and maximum silence on a POST response |
| `MCP_GATEWAY_STREAM_READ_TIMEOUT_SECONDS` | `300` | Maximum silence on a GET event stream, `0` for none |
| `MCP_GATEWAY_LOG_LEVEL` | `INFO` | Log verbosity |

Server flags: `--host`, `--port`, `--limit-concurrency`,
`--timeout-keep-alive`, `--timeout-graceful-shutdown`, `--forwarded-allow-ips`
(trusted proxies for `X-Forwarded-*`; ignored otherwise), `--ssl-keyfile` and
`--ssl-certfile`. `uvicorn --factory mcp_gateway.app:app_from_env` serves the
same application.

## Deviations from the brief

* Prefix matching is case-insensitive and applied after Unicode
  normalisation with zero-width characters removed, so `ADMIN_reset` and
  look-alike spellings are protected too. The brief says starts with
  `admin_`. A legitimate tool that differs from a protected prefix only by
  case is treated as protected; that is the intended direction of failure.
* Denied tool calls are answered with HTTP 200 and the JSON-RPC error, so any
  JSON-RPC client surfaces `-32001` instead of a transport failure. HTTP 403
  is available through `MCP_GATEWAY_DENIED_HTTP_STATUS`. A denied
  notification, which has no reply channel, gets HTTP 403 with a null id.
* A batch containing one unauthorised call is rejected as a whole; nothing in
  it is executed.
* Tokens must be at least 32 characters from the bearer token alphabet, and
  the committed development tokens need an explicit opt-in.
* Method and tool names longer than 256 characters, or containing control
  characters or lone surrogates, are refused as invalid rather than compared.

## Limitations

* Roles are flat strings; there is no hierarchy.
* Only `tools/call` is subject to policy; every other method is forwarded for
  any authenticated caller.
* The gateway speaks plain HTTP unless the TLS flags are given or an ingress
  terminates TLS.
* Token fingerprints in the audit log are keyed per process: they correlate
  within one process lifetime, not across restarts.
