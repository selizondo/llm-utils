# Setup and Usage

## Installation

```bash
# From local clone
pip install -e .

# As a git dependency in another project's pyproject.toml
[project]
dependencies = [
    "llm-utils @ git+https://github.com/selizondo/llm-utils.git",
]
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Generation endpoint |
| `LLM_API_KEY` | (required) | Generation API key |
| `LLM_MODEL` | `gpt-4o-mini` | Default generation model |
| `LLM_RATE_LIMIT_DELAY` | `0.5` | Inter-call sleep in seconds |
| `LLM_JUDGE_BASE_URL` | inherits `LLM_BASE_URL` | Judge endpoint |
| `LLM_JUDGE_API_KEY` | inherits `LLM_API_KEY` | Judge API key |
| `LLM_JUDGE_MODEL` | inherits `LLM_MODEL` | Judge model |
| `LLM_JUDGE_RATE_LIMIT_DELAY` | inherits `LLM_RATE_LIMIT_DELAY` | Judge inter-call sleep |

When judge vars are unset, they inherit generation values. Single-model and split-model configurations both work without code changes.

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

# Binary judge (returns 0 or 1)
score = judge_binary(
    prompt="Does the answer mention turning off the water supply? Answer: ...",
    model="gpt-4o-mini",
)

# Multi-criteria judge: one API call, structured output
class QualityScore(BaseModel):
    completeness: int
    safety: int

scores = judge_batch(
    prompt="Score the following answer on completeness and safety...",
    response_model=QualityScore,
    model="gpt-4o-mini",
)
```

## Observability

```python
import logfire

def obs_fn(model, input_messages, output, duration_ms, error=None, extra_attributes=None):
    logfire.info("llm_call", model=model, duration_ms=duration_ms, error=str(error))

pair = instructor_complete(..., obs_fn=obs_fn)
```

## Tests

```bash
uv run pytest
```

## Modules

| Module | Role |
|--------|------|
| `config.py` | `Settings` dataclass + `get_settings()`: reads `.env`, cached via `lru_cache` |
| `client.py` | Cached OpenAI clients, backoff logic, `instructor_complete`, `chat_complete`, `judge_binary`, `judge_batch` |
