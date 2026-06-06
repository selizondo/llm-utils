# Design and Tradeoffs

Decisions made during build, with the reasoning and the boundary where each would need revisiting.

---

## Dual Cached Client: Generation + Judge

Generation and judge use independent `OpenAI` instances, each with their own base URL, API key, and rate-limit state. A single client where model is chosen per-call would prevent pointing the judge at a cheaper endpoint (Groq for generation, local Ollama for judging) without refactoring call sites.

Cost: two module-level globals, two lazy-init paths. Acceptable for a shared library where both clients are always needed.

---

## TPD_THRESHOLD = 300.0 Seconds

When a provider `retry-after` exceeds 300 seconds, the code raises `RuntimeError` immediately instead of sleeping. At 300s, the provider is signaling daily quota exhaustion (Groq rolling-window TPD limit sends `1m21s` for TPM, >5 minutes for daily quota). Sleeping would block the entire process silently for hours.

Tradeoff: a transient provider outage returning a large `retry-after` would also raise immediately, but that outcome is preferable to a silent overnight hang.

---

## obs_fn Hook Pattern (not Middleware)

Observability is injected at call time via an optional `obs_fn` callback, not a middleware layer or class decoration. This keeps the library dependency-free: callers choose Langfuse, Logfire, a plain logger, or nothing.

Tradeoff: `obs_fn=None` means no default no-op. A caller that forgets to pass it gets no observability silently. A default `lambda **_: None` would make the contract explicit but adds noise to every call in tests.

---

## instructor.Mode.JSON for Ollama, TOOLS for Cloud

Ollama's API does not support OpenAI function-calling format. `instructor.Mode.TOOLS` silently produces malformed requests against Ollama. `_instructor_mode()` inspects the base URL for `localhost` or `11434` and falls back to `Mode.JSON`.

Tradeoff: URL sniffing is fragile. A cloud Ollama proxy on a non-standard port gets the wrong mode. An explicit `LLM_INSTRUCTOR_MODE` env var was considered and rejected as over-engineering for a dev/research context.

---

## lru_cache on get_settings()

`Settings` is cached after the first load so repeated calls in the same process do not re-read `.env`. `.env` changes require a process restart. Acceptable because settings are infrastructure constants that do not change between pipeline phases.

---

## judge_binary default_on_error = 0 (Fail Safe)

When the judge model returns an unparseable response, `judge_binary` returns 0 (fail) by default. Default 1 (pass) would silently accept bad output as high-quality. A false negative (rejecting good output) is recoverable; a false positive (accepting bad output into a dataset) is not. Callers can override with `default_on_error=1` for contexts where a failed judge call should not block the pipeline.

---

## importlib.metadata Version Export

`llm_utils.__version__` is populated from package metadata with a `PackageNotFoundError` fallback for uninstalled dev environments. Ensures the import never fails in a `pip install -e .` or bare-clone context.
