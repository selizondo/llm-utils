"""
Unit tests for llm_utils.client — rate-limit backoff, TPD detection, and obs_fn hook.

All tests mock the OpenAI client and instructor — no real API calls.
"""

from unittest.mock import MagicMock, patch
import pytest


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_utils.client import (
    _parse_retry_after,
    DEFAULT_RETRY_WAIT,
    TPD_THRESHOLD,
    _call_obs,
    chat_complete,
    chat_complete_with_usage,
    judge_binary,
)


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_plain_seconds_format(self):
        exc = Exception("try again in 15.03s")
        wait = _parse_retry_after(exc)
        assert abs(wait - 15.03) < 0.001

    def test_minutes_seconds_format(self):
        exc = Exception("try again in 1m21.9s")
        wait = _parse_retry_after(exc)
        assert abs(wait - 81.9) < 0.001

    def test_no_match_returns_default(self):
        exc = Exception("something went wrong")
        wait = _parse_retry_after(exc)
        assert wait == DEFAULT_RETRY_WAIT

    def test_tpd_threshold_raises_immediately(self):
        # retry-after above TPD_THRESHOLD signals daily quota exhaustion
        exc = Exception(f"try again in {TPD_THRESHOLD + 1:.0f}s")
        with pytest.raises(RuntimeError, match="Daily token quota exhausted"):
            _parse_retry_after(exc)

    def test_exactly_at_tpd_threshold_does_not_raise(self):
        # guard is `wait > TPD_THRESHOLD` (strict), so exactly at the boundary
        # returns the wait value rather than raising
        exc = Exception(f"try again in {TPD_THRESHOLD:.0f}s")
        wait = _parse_retry_after(exc)
        assert wait == TPD_THRESHOLD

    def test_one_second_below_threshold_returns_wait(self):
        wait_val = TPD_THRESHOLD - 1
        exc = Exception(f"try again in {wait_val:.0f}s")
        wait = _parse_retry_after(exc)
        assert wait == wait_val

    def test_retry_after_alternative_prefix(self):
        exc = Exception("retry after 30.5s")
        wait = _parse_retry_after(exc)
        assert abs(wait - 30.5) < 0.001

    def test_case_insensitive_match(self):
        exc = Exception("Try Again In 45s")
        wait = _parse_retry_after(exc)
        assert abs(wait - 45.0) < 0.001

    def test_minutes_and_seconds_tpd_raises(self):
        # 5m30s = 330s — above TPD_THRESHOLD
        exc = Exception("try again in 5m30.0s")
        with pytest.raises(RuntimeError, match="Daily token quota exhausted"):
            _parse_retry_after(exc)


# ---------------------------------------------------------------------------
# _call_obs
# ---------------------------------------------------------------------------


class TestCallObs:
    def test_obs_fn_called_with_all_kwargs_on_success(self):
        obs = MagicMock()
        _call_obs(
            obs,
            model="gpt-4o-mini",
            input_messages=[{"role": "user", "content": "hi"}],
            output="result",
            duration_ms=123.4,
        )
        obs.assert_called_once_with(
            model="gpt-4o-mini",
            input_messages=[{"role": "user", "content": "hi"}],
            output="result",
            duration_ms=123.4,
            error=None,
            extra_attributes={},
        )

    def test_obs_fn_called_with_error_on_failure(self):
        obs = MagicMock()
        exc = ValueError("something broke")
        _call_obs(
            obs,
            model="gpt-4o-mini",
            input_messages=[],
            output=None,
            duration_ms=50.0,
            error=exc,
        )
        obs.assert_called_once()
        _, kwargs = obs.call_args
        assert kwargs["error"] is exc
        assert kwargs["output"] is None

    def test_obs_fn_none_does_not_raise(self):
        # no-op when obs_fn is None — callers should not need to guard
        _call_obs(
            None,
            model="m",
            input_messages=[],
            output="x",
            duration_ms=1.0,
        )

    def test_extra_attributes_forwarded(self):
        obs = MagicMock()
        _call_obs(
            obs,
            model="m",
            input_messages=[],
            output="x",
            duration_ms=10.0,
            extra_attributes={"trace_id": "abc"},
        )
        _, kwargs = obs.call_args
        assert kwargs["extra_attributes"] == {"trace_id": "abc"}


# ---------------------------------------------------------------------------
# judge_binary
# ---------------------------------------------------------------------------


class TestJudgeBinary:
    def _make_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_returns_1_on_valid_pass(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("1")
        mock_get_client.return_value = client

        result = judge_binary("Is this correct?", model="gpt-4o-mini")
        assert result == 1

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_returns_0_on_valid_fail(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("0")
        mock_get_client.return_value = client

        result = judge_binary("Is this correct?", model="gpt-4o-mini")
        assert result == 0

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_returns_default_on_empty_response(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("")
        mock_get_client.return_value = client

        result = judge_binary("prompt", model="m", default_on_error=0)
        assert result == 0

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_returns_default_on_unparseable_response(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("yes")
        mock_get_client.return_value = client

        result = judge_binary("prompt", model="m", default_on_error=1)
        assert result == 1

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_obs_fn_called_on_success(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("1")
        mock_get_client.return_value = client

        obs = MagicMock()
        judge_binary("prompt", model="m", obs_fn=obs)
        obs.assert_called_once()
        _, kwargs = obs.call_args
        assert kwargs["output"] == "1"
        assert kwargs["error"] is None

    @patch("llm_utils.client.get_judge_client")
    @patch("llm_utils.client._judge_delay", return_value=0.0)
    def test_obs_fn_called_on_exception(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.side_effect = ValueError("network error")
        mock_get_client.return_value = client

        obs = MagicMock()
        result = judge_binary("prompt", model="m", default_on_error=0, obs_fn=obs)
        assert result == 0
        obs.assert_called_once()
        _, kwargs = obs.call_args
        assert kwargs["output"] is None
        assert isinstance(kwargs["error"], ValueError)


# ---------------------------------------------------------------------------
# chat_complete / chat_complete_with_usage
# ---------------------------------------------------------------------------


class TestChatCompleteWithUsage:
    def _make_response(self, content: str, usage: dict | None):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        if usage is None:
            response.usage = None
        else:
            response.usage = MagicMock(**usage)
        return response

    @patch("llm_utils.client.get_client")
    @patch("llm_utils.client._gen_delay", return_value=0.0)
    def test_chat_complete_unchanged_returns_only_content(self, mock_delay, mock_get_client):
        """Existing chat_complete callers must keep getting a plain string."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response(
            "hello", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_get_client.return_value = client

        result = chat_complete([{"role": "user", "content": "hi"}], model="m")
        assert result == "hello"

    @patch("llm_utils.client.get_client")
    @patch("llm_utils.client._gen_delay", return_value=0.0)
    def test_with_usage_returns_content_and_token_counts(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response(
            "hello", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_get_client.return_value = client

        content, usage = chat_complete_with_usage([{"role": "user", "content": "hi"}], model="m")
        assert content == "hello"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @patch("llm_utils.client.get_client")
    @patch("llm_utils.client._gen_delay", return_value=0.0)
    def test_with_usage_returns_empty_dict_when_provider_omits_usage(self, mock_delay, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_response("hello", None)
        mock_get_client.return_value = client

        content, usage = chat_complete_with_usage([{"role": "user", "content": "hi"}], model="m")
        assert content == "hello"
        assert usage == {}
