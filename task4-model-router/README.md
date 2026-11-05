# Task 4: Rate-Limiting and Model Fallback Router

A model-routing module for an LLM gateway. It accepts completion requests,
enforces a token-aware sliding window per tenant on an on-disk SQLite ledger,
sends each request to the primary provider under a 3000 ms deadline, fails
over to the secondary on 429 or timeout, and answers every failure with one
standardised error payload that carries no upstream internals.

## Run it

```bash
make install                 # once, from the repository root
make run-task4-primary       # terminal 1: mock provider on :9401 that answers 429
make run-task4-secondary     # terminal 2: healthy mock provider on :9402
make run-task4-router        # terminal 3: the router on :9400 with the development tenants
make demo-task4              # or: all three processes plus the full failover and rate-limit scenario
pytest task4-model-router    # 191 tests
```

The router needs the two provider URLs and a tenant document
(`MODEL_ROUTER_TENANTS_FILE` or `MODEL_ROUTER_TENANTS_JSON`): a JSON object
with a `tenants` map from each API key to its `tenant_id`, optional
`tokens_per_minute`, optional pinned `model` and optional `allowed_models`.
The committed `config/tenants.dev.json` is accepted only with
`MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS=true`, which the run targets and
the demo set. Clients send an OpenAI-style request to
`POST /v1/chat/completions` with `Authorization: Bearer <tenant key>`;
`GET /v1/usage` shows the caller's window.

## Answers to the brief

### Requirements

| The brief asks | Delivered | Where |
| --- | --- | --- |
| A token-aware sliding window rate limiter, for example 50,000 tokens per minute per tenant API key | A sliding window log per tenant, 50,000 tokens per 60 seconds by default and overridable per tenant. Admission reserves the estimated cost (prompt estimate times a learned calibration factor, plus the requested completion budget times `n`) in one atomic SQLite transaction; the reservation is settled to the provider's reported usage, or to an estimate of the generated text when none is reported, and released when no provider served the request | `src/model_router/rate_limiter.py`, `storage.py`, `token_estimator.py` |
| Failover to a secondary provider when the primary returns 429 or times out after 3000 ms | Each attempt runs under a 3000 ms deadline (configurable) that covers connect, send and the full body and cancels the in-flight request. 429, the deadline, transport failures, 408, 5xx, 529 and unparseable 2xx bodies fail over; a per-provider circuit breaker skips a repeatedly failing primary; a provider's `Retry-After` puts it on cooldown | `providers.py`, `router.py`, `circuit_breaker.py` |
| Standardised gateway error payloads without raw upstream stack traces or internal details | Every error is an object with `type`, `message` and `request_id` (plus `retry_after_seconds` for rate limits). Provider error text is relayed only when it is plain prose without URLs, paths, identifiers or key-shaped tokens; everything else becomes a generic sentence. Provider credential failures are reported as a 502 that does not blame the caller | `app.py`, `router.py` |
| On-disk SQLite for the database | `data/model_router.sqlite3` by default, WAL mode, write-locked admission so check-and-reserve is atomic across coroutines, threads and processes; file mode 0600; schema versioned; refused at start-up when it cannot be written | `storage.py` |

### Evaluation criteria

| Criterion | How it is met | Evidence |
| --- | --- | --- |
| Async concurrency handling and timeout race conditions | The deadline cancels a hanging primary so the fallback starts on time; a request cancelled before a verdict releases its reservation, one cancelled during admission is released by a follow-up task, and one cancelled after the provider answered is never released; a half-open circuit admits one trial and forgets a trial that never reports; per-provider concurrency caps fail over instead of queueing | `tests/test_router.py`, `tests/test_rate_limiter.py`; `tests/test_live_failover.py` proves over real sockets that a 4 second primary is abandoned at 3 seconds and the secondary answers before 3.9 |
| Accurate rate-limiter state eviction and token tracking | Rows older than the window are evicted inside admission and by a background task; a rejection computes the exact wait until enough tokens leave the window; settlement replaces estimates with reported usage; calibration learns the ratio of reported to estimated prompt tokens per tenant and provider; `n` multiplies the completion budget; the state survives restarts | `tests/test_storage.py` (window boundary, retry-after, eviction, 200 threaded admissions over two connections admitting exactly the limit), `tests/test_rate_limiter.py`, `tests/test_token_estimator.py` |
| Graceful fallback mechanics and standardised error sanitisation | The outcome table below; every response states which provider and model served it and whether that was a fallback; tests inject provider URLs, tracebacks and key-shaped strings upstream and assert none reaches a client | `tests/test_app.py`, `tests/test_router.py` |

| Situation | Status | `type` |
| --- | --- | --- |
| Missing or unknown tenant key | 401 | `authentication_error` |
| Malformed request (bad body, streaming requested, bad `max_tokens` or `n`, bad model name, too many messages) | 400 | `invalid_request` |
| Model outside the tenant's allow list, or missing when an allow list applies | 400 | `model_not_allowed`, `model_required` |
| Request shape no provider dialect can express | 400 | `unsupported_by_provider` |
| Body too large | 413 | `request_too_large` |
| Window exhausted | 429 with `Retry-After` | `rate_limit_exceeded` |
| Request larger than the whole window | 400 | `request_too_large` |
| Provider rejected the request (4xx other than 408 and 429) | 400, 404, 413 or 422 | `upstream_rejected` |
| Provider rejected the gateway's credentials | 502 | `upstream_unavailable` |
| All providers failed, all at capacity, or fallback disabled for the request | 503 with `Retry-After` | `upstream_unavailable` |
| Ledger unavailable | 503 with `Retry-After` | `service_unavailable` |
| Unknown path, wrong method, unexpected exception | 404, 405, 500 | `not_found`, `method_not_allowed`, `internal_error` |

Successful responses carry `X-Model-Router-Provider`, `X-Model-Router-Model`,
`X-Model-Router-Fallback`, the `X-RateLimit-*` headers and `X-Request-Id`.
`GET /healthz` is liveness only; `GET /readyz` probes the ledger with a real,
rolled-back write.

Behaviour beyond the brief: an Anthropic Messages dialect with strict
translation to and from the Chat Completions shape, pluggable token
estimators with persisted calibration, per-tenant model pins and allow lists,
per-request fallback control through `X-Model-Router-Allow-Fallback`,
per-provider concurrency caps and `Retry-After` cooldowns, the completion
budget sent to the provider when the client omitted one, provider bodies read
under a size cap, and request log rows only for requests that reached a
provider.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ROUTER_PRIMARY_URL`, `MODEL_ROUTER_SECONDARY_URL` | required | Provider base URLs, without embedded credentials |
| `MODEL_ROUTER_TENANTS_FILE` or `MODEL_ROUTER_TENANTS_JSON` | one required | Tenant document; keys are at least 32 printable ASCII characters |
| `MODEL_ROUTER_ALLOW_DEVELOPMENT_CREDENTIALS` | `false` | Accept a tenant document marked `development_only` |
| `MODEL_ROUTER_PRIMARY_API_KEY`, `..._SECONDARY_API_KEY` | unset | Credential presented to each provider |
| `MODEL_ROUTER_PRIMARY_DIALECT`, `..._SECONDARY_DIALECT` | `openai` | `openai` (Chat Completions compatible) or `anthropic` (Messages API, translated) |
| `MODEL_ROUTER_PRIMARY_MODEL`, `..._SECONDARY_MODEL` | unset | Model substituted per provider; a tenant pin takes precedence |
| `MODEL_ROUTER_PRIMARY_TIMEOUT_MS`, `..._SECONDARY_TIMEOUT_MS` | `3000` | Per-attempt deadline |
| `MODEL_ROUTER_PRIMARY_MAX_CONCURRENCY`, `..._SECONDARY_MAX_CONCURRENCY` | `0` | Requests in flight before failing over instead of queueing; `0` for no cap |
| `MODEL_ROUTER_PRIMARY_MAX_TOKENS_FIELD`, `..._SECONDARY_MAX_TOKENS_FIELD` | `max_tokens` | Field filled with the default completion budget when the client sent none |
| `MODEL_ROUTER_PRIMARY_NAME`, `..._SECONDARY_NAME` | `primary`, `secondary` | Labels in headers and logs |
| `MODEL_ROUTER_TOKENS_PER_MINUTE`, `MODEL_ROUTER_WINDOW_SECONDS` | `50000`, `60` | Default budget and window |
| `MODEL_ROUTER_DEFAULT_MAX_TOKENS` | `1024` | Completion budget reserved and sent when the request has none |
| `MODEL_ROUTER_TOKEN_ESTIMATOR` | `heuristic` | `heuristic[:chars_per_token]`, `tiktoken:<encoding>` or `module:factory` |
| `MODEL_ROUTER_ALLOW_FALLBACK_DEFAULT` | `true` | Whether requests without `X-Model-Router-Allow-Fallback` may use the secondary |
| `MODEL_ROUTER_FAILOVER_STATUSES` | `408,429,500,502,503,504,529` | Statuses that trigger failover |
| `MODEL_ROUTER_CIRCUIT_FAILURE_THRESHOLD`, `..._RECOVERY_SECONDS` | `5`, `30` | Circuit breaker |
| `MODEL_ROUTER_MAX_COOLDOWN_SECONDS` | `60` | Longest pause a provider's `Retry-After` can impose |
| `MODEL_ROUTER_DATABASE_PATH` | `data/model_router.sqlite3` | SQLite file |
| `MODEL_ROUTER_EVICTION_INTERVAL_SECONDS` | `30` | Background eviction period |
| `MODEL_ROUTER_MAX_BODY_BYTES`, `MODEL_ROUTER_MAX_RESPONSE_BYTES` | 1 MiB, 8 MiB | Request body cap, provider body cap |
| `MODEL_ROUTER_LOG_LEVEL` | `INFO` | Log verbosity |

Server flags: `--host`, `--port`, `--limit-concurrency`,
`--timeout-keep-alive`, `--timeout-graceful-shutdown`, `--forwarded-allow-ips`,
`--ssl-keyfile`, `--ssl-certfile`. `uvicorn --factory
model_router.app:app_from_env` serves the same application; the SQLite ledger
may be shared between worker processes, circuit breakers and concurrency
counters are per process.

## Deviations from the brief

* Limits are enforced per tenant, not per key: several API keys may map to
  one tenant and then share its window. With one key per tenant, which the
  development file uses, the two are the same.
* Streaming completions are rejected with a 400: failover after tokens have
  started flowing is ambiguous, and settlement needs the final usage.
* Client errors other than 408 and 429 are not failed over; retrying a request
  the provider found invalid would fail identically.
* A request whose cost exceeds the whole window is answered 400 rather than a
  429 it could never satisfy.
* Tenant keys must be at least 32 characters, model names follow a strict
  grammar of up to 256 characters, and messages and content parts are capped
  at 4096 each.

## Limitations

* The default token estimator is a heuristic; settlement corrects the charge
  afterwards, so the window can briefly exceed the limit by the estimation
  error until it slides.
* Circuit breakers and concurrency counters are per process; only the SQLite
  ledger is shared between processes, and SQLite needs a local disk.
* Providers speak the Chat Completions or the Anthropic Messages dialect;
  other native APIs need a further dialect.
* Calibration is learned per tenant and provider, not per model.
