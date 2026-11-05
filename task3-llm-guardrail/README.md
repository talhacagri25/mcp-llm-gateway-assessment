# Task 3: LLM Gateway Streaming Guardrail (PII Redaction)

An LLM gateway endpoint that forwards text generation requests to a provider
and streams the answer back while replacing e-mail addresses, US social
security numbers and payment card numbers with `[REDACTED]` inside the token
stream. The OpenAI Chat Completions, Anthropic Messages and Gemini
`generateContent` wire formats are supported, streaming and non-streaming.

## Run it

```bash
make install                 # once, from the repository root
make run-task3-provider      # terminal 1: mock provider on :9200 that streams a PII-laden answer
make run-task3-gateway       # terminal 2: the gateway on :9300
make demo-task3              # or: both processes, one streamed request per dialect, latency and redaction counts
pytest task3-llm-guardrail   # 485 tests
```

The gateway needs `LLM_GATEWAY_UPSTREAM_BASE_URL`. Point it at a real provider
(for example `https://api.openai.com` or `https://api.anthropic.com`), set
`LLM_GATEWAY_UPSTREAM_API_KEY` and `LLM_GATEWAY_API_KEYS`, and the SDK clients
of OpenAI, Anthropic and Gemini work unchanged against it. Clients send their
usual request to `/v1/chat/completions`, `/v1/messages` or the Gemini
`generateContent` paths and receive the usual response, redacted.

## Answers to the brief

### Requirements

| The brief asks | Delivered | Where |
| --- | --- | --- |
| A gateway endpoint that routes text generation requests to an LLM provider and streams the response back | One route per dialect. The request is forwarded with an allow-list of headers and the provider credential injected server side; the provider's SSE stream is relayed event by event as it arrives, and non-streaming responses are relayed after one pass | `src/llm_guardrail_gateway/app.py` |
| Intercept the returning chunk stream in real time | An incremental SSE parser hands each event to a dialect adapter, which routes every text channel (answer text, refusal, reasoning, tool arguments, per choice, block or candidate) through its own streaming redactor and re-emits the event at once | `sse.py`, `adapters/` |
| Parse the deltas, detect and redact e-mails, SSNs and card numbers, replacing them with `[REDACTED]` | Three detectors on by default: e-mail (RFC length bounds), SSN (allocation rules), card (Luhn, 13 to 19 digits, common separators). The replacement text is configurable. Presets for IBAN, US and E.164 phone numbers and compact SSNs, and a file of deployment specific identifiers, can be enabled by name | `redaction/patterns.py`, `redaction/custom.py` |
| Keep the stream responsive without accumulating the full response; minimise time to first token and overall latency | The redactor releases text as soon as no match can still grow across it and keeps only a bounded tail (at most one trailing word or digit group, never more than the longest possible match plus 256 characters). The SSE parser keeps one partial event. Nothing else about the response is retained | `redaction/streaming.py`, `sse.py` |

### Evaluation criteria

| Criterion | How it is met | Evidence |
| --- | --- | --- |
| Efficient asynchronous stream chunking and buffer state management | Each detector declares the characters a match can consist of and start with and its maximum length; after every chunk the redactor finds the earliest position a future match could still reach and releases everything before it, moving back only when a complete match would be split. The result is identical to redacting the whole text at once, whatever the chunk boundaries | `tests/test_streaming_redactor.py` checks every single split point, fixed chunk sizes from 1 to 64 and random multi-splits against the one-shot result |
| Performant pattern matching over partial text streams | Detection runs on a folded view (look-alike characters mapped to ASCII, zero-width characters dropped) with alignment-independent scanning; card detection is linear in the input. A 256 KiB adversarial payload of digits, pairs or dense card numbers is processed in well under a second | timing assertions in `tests/test_streaming_redactor.py` |
| Memory efficiency and low-latency proxying | Bounded tail per channel, one partial SSE event, at most 64 channels per response, stream budgets of 600 seconds and 64 MiB. First-token overhead measured over real sockets is within one provider delta of the direct call; the demo prints about one millisecond | `tests/test_streaming_latency.py` with real servers |

Behaviour beyond the brief: every SSE event is rebuilt from parsed fields,
unknown fields, numbers and keys are scanned, error documents and every 2xx
are redacted, token log probabilities are removed, unparseable provider
output is never forwarded, embedded JSON in prose stays parseable after
redaction, and credentials never appear in logs or in upstream URLs.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_GATEWAY_UPSTREAM_BASE_URL` | required | Provider base URL, without embedded credentials |
| `LLM_GATEWAY_UPSTREAM_API_KEY` | unset | Provider credential injected server side |
| `LLM_GATEWAY_API_KEYS` | unset | Keys callers must present (bearer, `x-api-key`, `x-goog-api-key` or `?key=`) |
| `LLM_GATEWAY_ALLOW_UNAUTHENTICATED` | `false` | Permit an open gateway that spends the upstream key for anonymous callers |
| `LLM_GATEWAY_PATTERNS` | `email,credit_card,ssn` plus declared names | Enabled detectors, including presets `iban`, `phone_us`, `phone_e164`, `ssn_compact` |
| `LLM_GATEWAY_PATTERNS_FILE` | unset | JSON file declaring deployment specific detectors: name, regex, alphabet, start characters, maximum length, optional Luhn or mod 97 check |
| `LLM_GATEWAY_REDACTION_TOKEN` | `[REDACTED]` | Replacement text |
| `LLM_GATEWAY_MAX_HOLDBACK_CHARS` | `256` | Holdback cap, raised automatically to the longest match |
| `LLM_GATEWAY_FORWARD_HEADERS` | `traceparent,tracestate` | Extra request headers relayed upstream |
| `LLM_GATEWAY_CONNECT_TIMEOUT_SECONDS`, `LLM_GATEWAY_READ_TIMEOUT_SECONDS` | `5`, `120` | Connect timeout and maximum silence between chunks |
| `LLM_GATEWAY_MAX_BODY_BYTES`, `LLM_GATEWAY_MAX_EVENT_BYTES`, `LLM_GATEWAY_MAX_RESPONSE_BYTES` | 1 MiB, 256 KiB, 8 MiB | Request body, one SSE event, buffered response |
| `LLM_GATEWAY_MAX_STREAM_SECONDS`, `LLM_GATEWAY_MAX_STREAM_BYTES` | `600`, 64 MiB | Budget of one streamed response |
| `LLM_GATEWAY_LOG_LEVEL` | `INFO` | Log verbosity |

Server flags: `--host`, `--port`, `--limit-concurrency`,
`--timeout-keep-alive`, `--timeout-graceful-shutdown`, `--forwarded-allow-ips`,
`--ssl-keyfile`, `--ssl-certfile`. `uvicorn --factory
llm_guardrail_gateway.app:app_from_env` serves the same application.

Errors use one shape with `type`, `message` and `request_id`: `401`
authentication, `413` body too large, `400` invalid request, `502` provider
unreachable or unparseable, `504` provider timeout, provider error statuses
relayed with their redacted document. A failure after the stream has started
is reported as an in-band `error` event; unredacted bytes are never relayed.

## Deviations from the brief

* Redaction is wider than the text deltas the brief names: every string in
  every event, comments and event names, error documents and all 2xx
  responses are redacted; token log probabilities are removed because they
  would reproduce redacted text token by token.
* Detection folds look-alike characters first (full-width digits, Cyrillic
  and Greek confusables, zero-width characters, every Unicode dash), so
  spellings meant to evade the patterns are still caught.
* A gateway configured with an upstream key but no gateway keys refuses to
  start unless `LLM_GATEWAY_ALLOW_UNAUTHENTICATED=true`, because it would spend
  the operator's credential for anyone.
* Prompts are forwarded unchanged; only the response is redacted, which is
  what the brief asks.

## Limitations

* Detection is pattern based. Internationalised e-mail addresses and PII
  written out in words are not detected. Phone numbers are off by default
  because they collide with order and reference numbers.
* A card number glued to further digits or to a hyphen is treated as part of
  a longer identifier and left alone; Luhn-valid numbers that are not cards
  are redacted.
* Redacting an Anthropic `thinking` block invalidates its signature, so such
  a block cannot be replayed to the provider.
* A provider that answers a streaming request with plain JSON is handled on
  the buffered path.
