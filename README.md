# llm-utils

![Tests](https://github.com/selizondo/llm-utils/actions/workflows/ci.yml/badge.svg)

Calling an LLM is three lines. Calling it reliably, with a separate judge model, structured output, rate-limit handling, and an observability hook that works with any backend, takes more. This shared library provides those primitives so each downstream project doesn't rebuild them.

**Stack:** Python · OpenAI SDK · instructor · Pydantic

## What It Provides

### Dual client: generation and judge are independent

Generation and judge each have their own model, API key, base URL, and rate-limit state. Pointing the judge at a cheaper model is a one-line env change with no code impact. The default: judge inherits generation values, so a single model works out of the box.

### Rate-limit handling that distinguishes throttle from quota exhaustion

Provider 429s carry a `retry-after` header. `client.py` parses both seconds and `Mm Ss` formats. Wait time under `TPD_THRESHOLD = 300s`: transient TPM throttle, sleep and retry. Wait over threshold: daily quota exhausted, raise `RuntimeError` immediately so the caller can checkpoint rather than sleep silently for hours.

### Structured outputs via instructor

`instructor_complete()` enforces a Pydantic response schema, retrying on parse failures. The caller gets a typed object, not a raw string. `judge_batch()` scores all criteria in a single call.

### Observability hook pattern

Every call site accepts an optional `obs_fn=` callable. Called with `(model, input_messages, output, duration_ms, error, extra_attributes)` on both success and error. Wire Langfuse, Logfire, or a plain logger without modifying call sites.

**Related projects:** Used by [rag-pipeline-from-scratch](https://github.com/selizondo/rag-pipeline-from-scratch), [llm-eval-harness](https://github.com/selizondo/llm-eval-harness), [synthetic-data-diy](https://github.com/selizondo/synthetic-data-diy), and other repos in this portfolio.

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
