# MCP and LLM Gateway Engineering Assessment

Four independently runnable Python projects, one per task in
[`task.md`](task.md). Each task directory holds the service, a mock of the
system it talks to, a scripted demo and its tests. Each task README answers
the brief point by point and lists where the implementation deviates from it.

| Task | Directory | Delivered |
| --- | --- | --- |
| 1. MCP server with strict validation and transport handling | [`task1-mcp-server/`](task1-mcp-server/README.md) | Official `mcp` SDK server over stdio with `get_customer_record` and `trigger_refund`; pydantic strict validation; standard JSON-RPC error codes; stdout carries protocol frames only, logs go to stderr |
| 2. MCP security gateway proxy | [`task2-mcp-gateway/`](task2-mcp-gateway/README.md) | HTTP/JSON-RPC reverse proxy: bearer token resolved to a role, `tools/list` forwarded unchanged, `admin_*` tool calls by non-admins answered with `-32001 Unauthorized Tool Call` without reaching the downstream |
| 3. LLM gateway streaming guardrail | [`task3-llm-guardrail/`](task3-llm-guardrail/README.md) | Streaming proxy that replaces e-mails, SSNs and card numbers with `[REDACTED]` inside the token stream, with bounded memory and measured first-token overhead |
| 4. Rate-limiting and model fallback router | [`task4-model-router/`](task4-model-router/README.md) | Token-aware sliding window per tenant on an on-disk SQLite ledger, failover to the secondary on 429 or the 3000 ms deadline, standardised error payloads |

## Run it

Prerequisites: Python 3.11 or newer (verified on 3.12), Linux or macOS.
`make`, `bash` and `curl` are used by the convenience targets and the demos
only.

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pinned dependencies plus editable installs of the four tasks
make test           # 1001 tests
make check          # lint, strict type check and tests
make demo-task1     # likewise demo-task2, demo-task3, demo-task4
```

Without `make`: `pip install -r requirements.txt`, then
`pip install --no-deps -e <task directory>` for each task, then `pytest` from
the repository root.

What each demo shows, with the mock dependencies started and stopped for you:

| Demo | Shows |
| --- | --- |
| `make demo-task1` | A scripted stdio conversation: valid and invalid tool calls, the error codes, and an audit that stdout contained JSON-RPC frames only |
| `make demo-task2` | Missing and unknown tokens, a viewer calling an admin tool (intercepted), an admin calling it (forwarded), identity headers as seen by the downstream, the gateway's audit log |
| `make demo-task3` | A PII-laden answer streamed through the gateway for each provider dialect, with redaction counts and first-token latency |
| `make demo-task4` | Failover on 429, failover after the 3000 ms deadline, the standardised error payloads, a small tenant exhausting its window, the ledger contents |

`make help` lists the run targets that start each service and its mocks by
hand.

## Verification state

| Check | Result |
| --- | --- |
| Tests (task 1 / 2 / 3 / 4) | 1001 passed (106 / 219 / 485 / 191) |
| ruff (lint and format), mypy strict | clean |
| Four demos | pass |
| Fresh clone, new virtual environment, Python 3.12.3, Linux | install, tests, lint, type check and demos pass |

Three review passes were applied after the first delivery: an adversarial
edge-case review, a review for fine-tuned and self-hosted models, and a
security review. Everything they found is fixed and pinned with a test; the
task READMEs state the resulting behaviour directly.

## Deviations from the brief and limitations

Deliberate choices that differ from the literal text of `task.md`:

* **Task 1.** The refund `amount` must have at most two decimal places and
  be at most 1,000,000; the brief only says positive float. The `reason`
  length is counted after trimming surrounding whitespace, must contain a
  letter or digit, and is capped at 500 characters; the brief only gives the
  minimum of 10. An unknown tool is answered with `-32602` as the MCP
  specification requires, not `-32601`.
* **Task 2.** The `admin_` prefix is matched case-insensitively and after
  Unicode normalisation, so `ADMIN_reset` and look-alike spellings are also
  protected; the brief says starts with `admin_`. Denied tool calls are
  answered with HTTP 200 and the JSON-RPC error by default (HTTP 403 is a
  setting). A batch with one denied element is rejected as a whole.
* **Task 3.** Redaction covers more than the text deltas the brief names:
  every string in every event, error documents, and every 2xx response are
  redacted, token log probabilities are removed, and look-alike characters
  are folded before detection. Prompts are forwarded unchanged.
* **Task 4.** Limits are enforced per tenant; several API keys may share one
  tenant. Streaming completions are rejected with a 400. Client errors other
  than 408 and 429 are not failed over. A request larger than the whole window
  is answered 400 rather than a 429 it could never satisfy.
* **Credentials.** Gateway tokens and tenant keys must be at least 32
  characters. The committed development credential files are marked
  `development_only` and are refused unless
  `MCP_GATEWAY_ALLOW_DEVELOPMENT_CREDENTIALS=true` or
  `MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS=true` is set; the Makefile and
  the demos set it.

Limitations of the delivery and of its verification:

* **Platform.** Verified on Linux with Python 3.12. Every dependency supports
  3.11 and the tooling targets 3.11, but 3.11 itself was not executed.
  Windows was not executed either: the services use no Unix-only calls, five
  tests are POSIX-specific and are skipped there, and the convenience targets
  and demos need `make`, `bash` and `curl` (WSL works).
* **Timing assertions.** A few tests assert wall-clock bounds (the 3000 ms
  failover, first-token latency, adversarial input finishing within a
  second). They have wide margins but can fail on a heavily loaded machine;
  a rerun passes.
* **Demo ports.** The demos use ports 9000, 9100, 9200, 9300 and 9400 to
  9402; each script accepts port overrides through environment variables.
* **TLS.** All three HTTP services speak plain HTTP unless the `--ssl-keyfile`
  and `--ssl-certfile` flags are given or a TLS-terminating ingress sits in
  front.
* **Mocks.** The counterpart systems are mocks included in each task; the
  gateways work unchanged against real providers and MCP servers, which the
  demos do not exercise.

## Repository layout

| Path | Content |
| --- | --- |
| `Makefile` | install, test, lint, typecheck, check, audit, lock, run and demo targets |
| `pyproject.toml` | shared ruff, mypy and pytest configuration |
| `requirements.txt`, `requirements.lock` | direct pins, and the complete transitive closure for `make audit` |
| `task.md` | the assessment brief |
| `task1-mcp-server/` to `task4-model-router/` | one package per task: `src/`, `tests/`, `scripts/`, `README.md` |
