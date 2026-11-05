# Task 1: Customer MCP Server over stdio

An MCP server on the official Python SDK (`mcp`) that exposes
`get_customer_record` and `trigger_refund` over the stdio transport. Inputs
are validated by pydantic in strict mode, failures are reported with the
standard JSON-RPC error codes, stdout carries protocol frames only and all
logging goes to stderr as JSON lines.

## Run it

```bash
make install                                   # once, from the repository root
python -m customer_mcp_server                  # waits for JSON-RPC frames on stdin
make demo-task1                                # scripted conversation plus stdout purity audit
pytest task1-mcp-server                        # 106 tests
```

To use it from an MCP host, register the command `python -m customer_mcp_server`
(add `--log-level DEBUG` for verbose stderr logs, `--max-frame-bytes` to
change the 4 MiB frame limit). Five sample customers exist, `CUST-00001` to
`CUST-00005`, covering active, suspended and closed accounts.

## Answers to the brief

### Requirements

| The brief asks | Delivered | Where |
| --- | --- | --- |
| A runnable MCP server using the official SDK | Python package on `mcp` 2.x, started with `python -m customer_mcp_server`; also installed as the `customer-mcp-server` console script | `src/customer_mcp_server/__main__.py`, `server.py` |
| `get_customer_record` with `customer_id` as `CUST-XXXXX` | Implemented. The id must be the literal `CUST-` followed by exactly five ASCII digits. Returns the profile and refundable balance as text and as structured content with a matching output schema | `server.py`, `schemas.py`, `service.py` |
| `trigger_refund` with `customer_id`, a positive float `amount` and a `reason` of at least 10 characters | Implemented. The amount must be a positive finite number, the reason at least 10 characters. A valid refund returns a receipt and reduces the refundable balance. Unknown customers, suspended or closed accounts and amounts above the balance come back as tool results flagged `isError`, so the calling model can correct itself | `server.py`, `schemas.py`, `service.py` |
| Strict schema validation with Pydantic; invalid formats rejected with standard MCP JSON-RPC error codes | pydantic v2 strict mode: no type coercion, unknown fields refused, every failing field reported at once. Validation failures are `-32602 Invalid params`, an unknown tool is `-32602`, an undecodable frame is `-32700` or `-32600`, an unexpected exception is `-32603` with a generic message | `schemas.py`, `server.py`, `transport.py` |
| stdio transport; stdout reserved for JSON-RPC; logs exclusively on stderr | The SDK stdio transport owns stdout. Logging is configured to stderr before the transport starts, the lint configuration forbids `print`, the transport re-points file descriptor 1 at stderr for the session, and `sys.stdout` is redirected to stderr so nothing buffered can reach the wire at exit | `__main__.py`, `logging_config.py` |

### Evaluation criteria

| Criterion | How it is met | Evidence |
| --- | --- | --- |
| STDIO isolation | Five independent layers: stderr logging configured first, no `print` calls (enforced by lint), the SDK's descriptor swap, the `sys.stdout` redirect, and tests that read the real pipes of a subprocess | `tests/test_stdio_wire.py` runs a full conversation at `DEBUG` level and asserts every stdout line is a JSON-RPC frame; a second test injects a noisy handler that prints during the session |
| Protocol compliance | `initialize`, `tools/list` and `tools/call` through the SDK; the error mapping above; request ids preserved; frames with malformed ids answered with `-32600` instead of being dropped; business failures as `isError` results, never as protocol errors | `tests/test_sdk_client.py` drives the server with the official MCP client, in process and over stdio; `tests/test_stdio_wire.py` covers malformed frames and ids |
| Validation | Strict types, the `[0-9]` digit class (Unicode digits are not digits here), no surrounding whitespace, finite amounts (no NaN or Infinity, as float or as JSON token), trimmed reasons with at least one letter or digit and no control characters, unknown fields reported by name, at most 20 issues per response with a count of omitted ones, echoed values cut at 80 characters | `tests/test_schemas.py` covers every accepted and rejected shape; `tests/test_stdio_wire.py` covers oversized and hostile inputs |

Operational behaviour the brief does not ask for but a deployment needs:
frames above 4 MiB are discarded unparsed and answered with `-32600`;
argument objects with more than 32 members and reasons above 4096 raw
characters are refused before validation; an answer in flight survives the
client closing stdin; `SIGINT` and `SIGTERM` exit with 130 and 143 after
flushing the logs; end of input exits with 0; a broken pipe is one log line
and exit status 1.

## Deviations from the brief

* `amount` must have at most two decimal places and be at most 1,000,000.
  The brief says positive float; a value such as 10.005 or 2,000,000 is
  rejected with `-32602` and an explicit message. Both limits are single
  constants in `schemas.py`.
* `reason` length is counted after trimming surrounding whitespace, must
  contain at least one letter or digit, and may not exceed 500 characters or
  contain control characters. The brief only states the minimum of 10.
* An unknown tool name is answered with `-32602`, which is what the MCP
  specification prescribes, rather than `-32601`.

## Limitations

* Customer data is an in-memory sample set; there is no persistence.
* `trigger_refund` is not idempotent, and says so in its tool annotations. A
  host that retries an identical call after a timeout obtains a second
  refund; the brief fixes the tool inputs, so no idempotency key was added.
* A client that closes the server's stdout but keeps stdin open keeps the
  process alive until stdin closes too (SDK behaviour).
* The 130 and 143 exit codes on signals are a POSIX convention; the two
  tests for them are skipped on Windows.
