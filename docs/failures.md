# Failure Scenarios

Documented failure modes with detection mechanisms and fallback behavior.

---

## Failure 1: Daily Token Quota Exhaustion Mid-Batch

### What breaks
A generation batch hits the provider's daily TPM/TPD limit mid-run. Without protection, all remaining calls return 429 and the batch fails silently or raises after exhausting retries.

### Detection mechanism
`client.py` parses the `retry-after` header from the 429 response. If the wait time exceeds `TPD_THRESHOLD = 300.0s`, it's treated as a daily quota exhaustion (not a transient rate limit) and raises `RuntimeError` immediately with the message: `"Daily token quota exhausted: retry after UTC midnight"`. Under the threshold, it sleeps and retries.

### Fallback behavior
Raise immediately so the caller can checkpoint what was completed and resume tomorrow, rather than sleeping for hours or silently returning partial results.

---

## Failure 2: Judge Model Returns Malformed Structured Output

### What breaks
`judge_batch()` and `judge_binary()` use `instructor` to enforce a Pydantic response schema. If the judge model returns a response that can't be parsed into the expected schema, `instructor` retries up to its configured limit, then raises `InstructorRetryException`.

### Detection mechanism
`instructor` handles retry logic internally. The exception surfaces to the caller with the raw LLM response and the validation errors.

### Fallback behavior
The caller is responsible for catching `InstructorRetryException`. Pattern used by downstream projects: catch, log the raw response for debugging, and skip the pair rather than failing the entire batch.

---

## Failure 3: Judge and Generation Models Configured to the Same Endpoint at Different Rate Limits

### What breaks
`Settings` inherits judge vars from generation vars when unset. If both are pointed at the same model on the same account, rate-limit backoff from generation calls delays judge calls too, even if they're on separate clients.

### Detection mechanism
Not detected: this is a configuration responsibility. The `get_settings()` output logs both `LLM_BASE_URL` and `LLM_JUDGE_BASE_URL` at startup so the caller can verify they're independent when needed.

### Fallback behavior
Set `LLM_JUDGE_BASE_URL`, `LLM_JUDGE_API_KEY`, and `LLM_JUDGE_MODEL` explicitly to a separate endpoint or model to decouple the rate limits.
