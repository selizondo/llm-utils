# Design Decisions and Tradeoffs

## Dual cached client (generation + judge)

Generation and judge calls use independent `OpenAI` instances, each with their own base URL, API key, and rate-limit state. The alternative — a single client, model chosen per-call — would prevent pointing the judge at a cheaper endpoint (e.g., Groq for generation, local Ollama for judging) without refactoring call sites. The cost: two module-level globals and two lazy-init paths. Acceptable for a shared library where both clients are always needed.

## TPD_THRESHOLD = 300.0 seconds

When a provider `retry-after` exceeds 300s, the code raises `RuntimeError` immediately rather than sleeping. At 300s, the provider is signalling daily quota exhaustion (Groq rolling-window TPD limit sends `1m21s` → 81s for TPM; `>5m` → likely TPD). Sleeping would block the entire process for hours silently. The tradeoff: a transient provider outage returning a large `retry-after` would also raise, but that's preferable to a silent overnight hang.

## obs_fn hook pattern (not middleware)

Observability is injected at call time via an optional `obs_fn` callback rather than as a middleware layer or class decoration. This keeps the library dependency-free — callers choose Langfuse, Logfire, a plain logger, or nothing. The tradeoff: `obs_fn=None` means no default no-op is applied; if a caller forgets to pass it, the call is silently unobserved. A default `lambda **_: None` would make the contract explicit but adds noise to every call in tests.

## instructor.Mode.JSON for Ollama, TOOLS for cloud

Ollama's API does not support OpenAI function-calling format, so `instructor.Mode.TOOLS` silently produces malformed requests. `_instructor_mode()` inspects the base URL for `localhost` or `11434` and falls back to `Mode.JSON`. The tradeoff: URL sniffing is fragile — a cloud Ollama proxy on a non-standard port would get the wrong mode. An explicit `LLM_INSTRUCTOR_MODE` env var was considered and rejected as over-engineering for a dev/research context.

## lru_cache on get_settings()

`Settings` is cached after the first load so repeated calls in the same process don't re-read `.env`. This means `.env` changes require a process restart, not a reload. Acceptable because settings are infrastructure constants — they don't change between pipeline phases.

## judge_binary default_on_error = 0 (fail safe)

When the judge model returns an unparseable response, `judge_binary` returns 0 (fail) by default. The alternative — default 1 (pass) — would silently accept bad output as high-quality. A false negative (rejecting good output) is recoverable; a false positive (accepting bad output into the dataset) is not. Callers can override with `default_on_error=1` for contexts where a failed judge call should not block the pipeline.

## No importlib.metadata version export

The package version (`0.1.0` in `pyproject.toml`) is not exported as `llm_utils.__version__`. This was identified as a gap in the staff review. The fix is one line: `from importlib.metadata import version; __version__ = version("llm-utils")` in `__init__.py`. Left unresolved because downstream projects pin by git hash, not version string.
