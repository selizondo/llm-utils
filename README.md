# llm-utils

Shared LLM client library used by various projects. Wraps the OpenAI Python SDK with [instructor](https://github.com/jxnl/instructor) for structured outputs, rate-limit backoff, and an observability hook pattern.

*See [docs/tradeoffs.md](docs/tradeoffs.md) for design decisions and [docs/failures.md](docs/failures.md) for known failure modes.*

---

## Key Concepts

**Dual client** — generation and judge are separate `OpenAI` client instances, each with independent base URL, API key, model, and rate-limit delay. When unset, judge vars inherit generation values so the default is a single model. Pointing judge at a cheaper model is a one-line env change with no code impact.

**Rate-limit backoff vs daily quota** — provider 429s carry a `retry-after` header. `client.py` parses both seconds (`60`) and `Mm Ss` (`1m 30s`) formats. Wait ≤ `TPD_THRESHOLD = 300.0s` → transient TPM throttle → sleep and retry. Wait > threshold → daily token quota exhausted → raise `RuntimeError` immediately so the caller can checkpoint rather than sleep for hours.

**Structured outputs via instructor** — `instructor_complete()` wraps the OpenAI call with an `instructor` client that enforces a Pydantic response schema, retrying on parse failures up to the configured limit. The caller gets a typed object, not a raw string.

**Observability hook** — every call site accepts an optional `obs_fn=` callable. When provided it's called with `(model, input_messages, output, duration_ms, error, extra_attributes)` on both success and error. Wire Logfire, Langfuse, or a plain logger without modifying call sites.

---

## What it does

- **Dual client** — separate generation client and judge client, each independently configurable via environment variables. Pointing a judge at a cheaper/faster model without touching generation config is a one-line env change.
- **Rate-limit backoff** — parses provider-supplied `retry-after` headers (seconds and `Mm Ss` format). Raises `RuntimeError` immediately on daily quota exhaustion (`TPD_THRESHOLD = 300.0s`) instead of sleeping for hours.
- **Structured outputs** — `instructor_complete()` returns a typed Pydantic object; `judge_batch()` scores all criteria in a single call.
- **Observability hook** — every call site accepts `obs_fn=` which is called with `(model, input_messages, output, duration_ms, error, extra_attributes)` on both success and error. Wire Langfuse, Logfire, or a plain logger without touching call sites.

---

## Modules

| Module | Purpose |
|---|---|
| `config.py` | `Settings` dataclass + `get_settings()` — reads `.env`, cached via `lru_cache` |
| `client.py` | Cached OpenAI clients, backoff logic, `instructor_complete`, `chat_complete`, `judge_binary`, `judge_batch` |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Generation endpoint |
| `LLM_API_KEY` | *(required)* | Generation key |
| `LLM_MODEL` | `gpt-4o-mini` | Default generation model |
| `LLM_RATE_LIMIT_DELAY` | `0.5` | Inter-call sleep (seconds) |
| `LLM_JUDGE_BASE_URL` | inherits `LLM_BASE_URL` | Judge endpoint |
| `LLM_JUDGE_API_KEY` | inherits `LLM_API_KEY` | Judge key |
| `LLM_JUDGE_MODEL` | inherits `LLM_MODEL` | Judge model |
| `LLM_JUDGE_RATE_LIMIT_DELAY` | inherits `LLM_RATE_LIMIT_DELAY` | Judge inter-call sleep |

When judge vars are unset they inherit the generation values — generation and judge can be the same model or different ones with zero code change.

---

## Usage

```python
from llm_utils import instructor_complete, judge_binary, judge_batch
from pydantic import BaseModel

class QAPair(BaseModel):
    question: str
    answer: str

# Structured generation
pair = instructor_complete(
    messages=[{"role": "user", "content": "Generate a Q&A pair about plumbing."}],
    response_model=QAPair,
    model="gpt-4o-mini",
)

# Binary judge (0 or 1)
score = judge_binary(
    prompt="Does the answer mention turning off the water supply? Answer: ...",
    model="gpt-4o-mini",
)

# Multi-criteria judge (one call, structured Pydantic output)
class QualityScore(BaseModel):
    completeness: int
    safety: int

scores = judge_batch(
    prompt="Score the following answer on completeness and safety...",
    response_model=QualityScore,
    model="gpt-4o-mini",
)
```

Observability:
```python
import logfire

def obs_fn(model, input_messages, output, duration_ms, error=None, extra_attributes=None):
    logfire.info("llm_call", model=model, duration_ms=duration_ms, error=str(error))

pair = instructor_complete(..., obs_fn=obs_fn)
```

---

## Installation

```bash
pip install -e .
```

Or as a git dependency in another project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "llm-utils @ git+https://github.com/selizondo/llm-utils.git",
]
```

---

## Status

Shared internal library — no versioned releases. Pinned via git hash in downstream projects.
