"""Preserve OpenRouter reasoning without sending LiteLLM's SDK-only wrapper.

Friendli rejects ``provider_specific_fields`` in assistant message continuations.
The native transcript remains unchanged; only a deep copy for transport is
normalized to OpenRouter's message fields. Legacy dynamic caps preserve the
entire transport history. An explicitly enabled managed policy uses replayable
bounded complete-turn transport windows while retaining every native message.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import stat
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpcore
import httpx
import litellm
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openrouter.common_utils import OpenRouterException

from pydantic import Field
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

if __package__:
    from .mini_swe_friendli_context import (
        CONTEXT_WINDOW, OUTPUT_CEILING, SAFETY_TOKENS, IMAGE_RESERVE, CONTEXT_HISTORY_POLICY,
        TOOL_SCHEMA_UTF8_BYTES, TOOL_SCHEMA_SHA256,
        _encoded, _input_bound, _returned_assistant_bound, _text_excerpt, _context_turns,
        _clip_tool_observations, _context_memory, _context_transport_window,
        prepare_native_context_messages, validate_context_request_replay,
    )
else:
    from picbench_friendli_context import (
        CONTEXT_WINDOW, OUTPUT_CEILING, SAFETY_TOKENS, IMAGE_RESERVE, CONTEXT_HISTORY_POLICY,
        TOOL_SCHEMA_UTF8_BYTES, TOOL_SCHEMA_SHA256,
        _encoded, _input_bound, _returned_assistant_bound, _text_excerpt, _context_turns,
        _clip_tool_observations, _context_memory, _context_transport_window,
        prepare_native_context_messages, validate_context_request_replay,
    )
_BENCHMARK_STATUS = Path("/run/picbench/benchmark-status.json")
_TRUSTED_STATUS_UID = 0
_RETRY_BACKOFF_SECONDS = 1.0
_SHARED_POOL_BACKOFF_SECONDS = 60.0
_MAX_RETRY_AFTER_SECONDS = 120.0
_SDK_ERROR_BODY_BYTES = 65536


class FriendliElapsedDeadlineError(TimeoutError):
    """A model attempt exceeded its independent elapsed-time allowance."""


class FriendliAPIBudgetError(RuntimeError):
    """A separately configured diagnostic model budget cannot admit a call."""


class FriendliProviderResponseError(RuntimeError):
    """A returned response explicitly reports a provider-side failure."""

    def __init__(self, response: dict, *, code: int | None, error_type: str | None,
                 retryable: bool, partial_output: bool, partial_reasoning: bool,
                 usable_partial_output: bool):
        super().__init__(f"Friendli returned provider error {code}; partial_output={partial_output}")
        self.response = response
        self.response_error_code = code
        self.error_type = error_type
        self.retryable = retryable
        self.partial_output = partial_output
        self.partial_reasoning = partial_reasoning
        self.usable_partial_output = usable_partial_output


class _ElapsedDeadlineInterrupt(BaseException):
    # Bypass an SDK's broad Exception retry/normalization until HTTP unwinds.
    pass


@contextmanager
def _elapsed_api_deadline(seconds: float):
    """Interrupt synchronous inference without an orphan background worker.

    Unlike HTTPX inactivity timeouts, ITIMER_REAL covers continuously arriving
    body chunks. Existing alarm deadlines/periodic callbacks keep their timing.
    """
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL")):
        raise RuntimeError("Friendli elapsed deadlines require the POSIX main thread")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Friendli elapsed deadline must be positive and finite")
    started = time.monotonic()
    own_deadline = started + seconds
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    previous_deadline = started + previous_delay if previous_delay > 0 else None

    def arm():
        due = min(own_deadline, previous_deadline) if previous_deadline is not None else own_deadline
        signal.setitimer(signal.ITIMER_REAL, max(0.000001, due - time.monotonic()))

    def alarm(signum, frame):
        nonlocal previous_deadline
        now = time.monotonic()
        try:
            if previous_deadline is not None and now >= previous_deadline:
                if previous_interval > 0:
                    previous_deadline += (math.floor((now - previous_deadline) / previous_interval) + 1) * previous_interval
                else:
                    previous_deadline = None
                if callable(previous_handler):
                    previous_handler(signum, frame)
                elif previous_handler == signal.SIG_DFL:
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
                    signal.raise_signal(signal.SIGALRM)
        finally:
            if time.monotonic() >= own_deadline:
                raise _ElapsedDeadlineInterrupt()
            # An SDK may catch an earlier handler's Exception. The elapsed
            # alarm must remain armed even when that callback raises.
            arm()

    signal.signal(signal.SIGALRM, alarm)
    arm()
    try:
        yield
    except _ElapsedDeadlineInterrupt:
        raise FriendliElapsedDeadlineError("Friendli model attempt exceeded its elapsed deadline") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL,
                            max(0.000001, previous_deadline - time.monotonic()), previous_interval)
        else:
            signal.setitimer(signal.ITIMER_REAL, 0, previous_interval)


def _response_value(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _response_dict(response) -> dict:
    if isinstance(response, dict):
        return copy.deepcopy(response)
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        value = dump(mode="json")
        if isinstance(value, dict):
            return value
    raise ValueError("Friendli provider-error response cannot be retained losslessly")


def _provider_response_error(response) -> FriendliProviderResponseError | None:
    """Recognize returned errors without replaying actionable partial output.

    A committed HTTP200 may contain a provider502 error. LiteLLM can wrap that
    JSON in choice.message.content and normalize its outer finish reason to
    stop; native error flags and error metadata remain authoritative. Failed
    reasoning-only generations may be retried before any action parsing, while
    their exact private response and cost uncertainty remain in the proof.
    """
    error = _response_value(response, "error")
    reported = isinstance(error, dict)
    usable_partial = False
    partial_reasoning = False
    for choice in _response_value(response, "choices", []) or []:
        fields = _response_value(choice, "provider_specific_fields", {}) or {}
        choice_error = _response_value(fields, "error", _response_value(choice, "error"))
        if isinstance(choice_error, dict):
            reported = True
            if error is not None and error != choice_error:
                usable_partial = True
            error = choice_error
        flags = [_response_value(choice, key) for key in ("finish_reason", "native_finish_reason")]
        flags.append(_response_value(fields, "native_finish_reason"))
        reported |= "error" in flags
        message = _response_value(choice, "message", {}) or {}
        content = _response_value(message, "content")
        content_error = None
        if isinstance(content, str) and len(content.encode("utf-8")) <= 65536:
            try:
                decoded = json.loads(content)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, dict) and set(decoded) == {"error"} and isinstance(decoded["error"], dict):
                content_error = decoded["error"]
        if content_error is not None:
            if error is not None and error != content_error:
                usable_partial = True
            error = content_error
        elif content not in (None, "", []):
            usable_partial = True
        if any(_response_value(message, key) for key in (
                "tool_calls", "function_call", "refusal")):
            usable_partial = True
        if any(_response_value(message, key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
            partial_reasoning = True
        message_fields = _response_value(message, "provider_specific_fields", {}) or {}
        if _response_value(message_fields, "refusal"):
            usable_partial = True
        if any(_response_value(message_fields, key) for key in (
                "reasoning", "reasoning_content", "reasoning_details")):
            partial_reasoning = True
    if not reported:
        return None
    error = error if isinstance(error, dict) else {}
    code = error.get("code") if type(error.get("code")) is int else None
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    error_type = metadata.get("error_type", error.get("error_type"))
    error_type = error_type if isinstance(error_type, str) else None
    retryable = not usable_partial and (code, error_type) in {(502, "provider_unavailable"), (504, "timeout")}
    return FriendliProviderResponseError(_response_dict(response), code=code, error_type=error_type,
                                         retryable=retryable, partial_output=usable_partial or partial_reasoning,
                                         partial_reasoning=partial_reasoning, usable_partial_output=usable_partial)


def _record_response_cost(event: dict, response) -> None:
    """Keep reported customer cost distinct from upstream estimates."""
    usage = _response_value(response, "usage") or {}
    cost = _response_value(usage, "cost")
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        event.update(provider_cost_usd=cost, provider_billing_unknown=False)
    upstream = _response_value(_response_value(usage, "cost_details", {}) or {}, "upstream_inference_cost")
    if type(upstream) in (int, float) and math.isfinite(upstream) and upstream >= 0:
        event["provider_upstream_inference_cost_usd"] = upstream
        if cost == 0 and upstream > 0:
            event["provider_billing_unknown"] = True
            event["provider_cost_qualification"] = "zero_usage_cost_with_positive_upstream_estimate"


def _transient_transport_error(error: Exception) -> tuple[str | None, int | None]:
    """Classify only bounded elapsed and explicitly recognized provider failures.

    Concrete exception types and status codes keep arbitrary provider messages,
    policy/auth errors and local exceptions outside the retry policy.
    """
    if type(error) is FriendliElapsedDeadlineError:
        return "elapsed_api_deadline", None
    if type(error) is FriendliProviderResponseError:
        return ("provider_error_envelope" if error.retryable else None), None
    allowed = (litellm.Timeout, litellm.APIError, httpx.RemoteProtocolError, httpcore.RemoteProtocolError)
    if type(error) not in allowed:
        return None, None
    status_code = getattr(error, "status_code", None)
    if type(status_code) is not int:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    status_code = status_code if type(status_code) is int else None
    if status_code == 504 and type(error) in (litellm.Timeout, litellm.APIError):
        return "http504", status_code
    if status_code is not None and status_code != 500:
        return None, status_code
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (httpx.RemoteProtocolError, httpcore.RemoteProtocolError)):
            if "incomplete chunked read" in str(current).lower():
                return "incomplete_body", status_code
        pending.extend(item for item in (
            current.__cause__, current.__context__, getattr(current, "original_exception", None)
        ) if isinstance(item, BaseException))
    return None, status_code


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Bound SDK exception traversal; never classify an arbitrary message."""
    pending, result, seen = [error], [], set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend(value for value in (
            current.__cause__, current.__context__, getattr(current, "original_exception", None)
        ) if isinstance(value, BaseException))
    return result


def _bounded_utf8(value, maximum: int, *, allow_empty: bool = True) -> bool:
    if not isinstance(value, str):
        return False
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        return False
    return size <= maximum and (allow_empty or size > 0)


def _sdk_failure_evidence(error: Exception, *, model_name: str,
                          shared_pool_retry: bool, malformed_retry: bool) -> dict | None:
    """Recognize only the retained shared-pool and empty HTTP200 SDK failures.

    LiteLLM rebuilds some HTTP responses without their body. The pinned
    OpenRouter exception retains the original error JSON; its HTTP200 parser
    retains the original document in JSONDecodeError.doc. A malformed document
    containing any non-whitespace is deliberately not replayed: it might be a
    partially returned model/tool response.
    """
    if (getattr(error, "llm_provider", None) != "openrouter"
            or getattr(error, "model", None) not in {model_name, model_name.removeprefix("openrouter/")}):
        return None
    chain = _exception_chain(error)
    body = None
    extra = {}
    if shared_pool_retry and type(error) is litellm.RateLimitError and getattr(error, "status_code", None) == 429:
        candidates = [item for item in chain if type(item) is OpenRouterException
                      and getattr(item, "status_code", None) == 429]
        for candidate in candidates:
            text = getattr(candidate, "message", None)
            if not _bounded_utf8(text, _SDK_ERROR_BODY_BYTES):
                continue
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                continue
            envelope_only = (isinstance(value, dict) and "error" in value
                             and set(value) <= {"error", "user_id"}
                             and ("user_id" not in value
                                  or _bounded_utf8(value["user_id"], 256, allow_empty=False)))
            reported = value.get("error") if envelope_only else None
            metadata = reported.get("metadata") if isinstance(reported, dict) else None
            if (not isinstance(reported, dict) or type(reported.get("code")) is not int
                    or reported["code"] != 429 or not isinstance(metadata, dict)
                    or str(metadata.get("provider_name", "")).lower() != "friendli"
                    or metadata.get("limit_source") != "upstream_provider_shared_pool"):
                continue
            body = text
            extra = {"transient_kind": "shared_pool_rate_limit", "http_status": 429,
                     "rate_limit_scope": "upstream_provider_shared_pool", "rate_limit_provider": "Friendli",
                     "sdk_error_json_decode_error": False}
            break
    elif malformed_retry and type(error) is litellm.APIError and getattr(error, "status_code", None) == 200:
        parser_errors = [item for item in chain if type(item) is OpenAIError
                         and getattr(item, "status_code", None) == 200
                         and str(getattr(item, "message", "")).startswith("Unable to get json response - ")]
        decoders = [item for item in chain if type(item) is json.JSONDecodeError]
        if parser_errors and len(decoders) == 1:
            decoder = decoders[0]
            if (isinstance(decoder.doc, str) and not decoder.doc.strip()
                    and len(decoder.doc.encode("utf-8")) <= _SDK_ERROR_BODY_BYTES):
                body = decoder.doc
                extra = {"transient_kind": "malformed_json_response", "http_status": 200,
                         "sdk_error_json_decode_error": True, "sdk_error_body_whitespace_only": True,
                         "sdk_error_json_decode_pos": decoder.pos,
                         "sdk_error_json_decode_lineno": decoder.lineno,
                         "sdk_error_json_decode_colno": decoder.colno}
    if body is None:
        return None
    return {**extra, "sdk_error_response_status": extra["http_status"],
            "sdk_error_response_body": body,
            "sdk_error_response_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "sdk_error_response_body_utf8_bytes": len(body.encode("utf-8")),
            "sdk_error_chain_classes": [type(item).__module__ + "." + type(item).__name__ for item in chain],
            "partial_output": False, "partial_reasoning": False,
            "usable_partial_output": False, "actionable_output_absent": True}


def _retry_after(error: Exception, observed_at: datetime) -> tuple[str | None, float | None, str | None]:
    """Read only Retry-After, including headers preserved on SDK causes."""
    values = []
    for item in _exception_chain(error):
        response = getattr(item, "response", None)
        for headers in (getattr(item, "headers", None), getattr(item, "litellm_response_headers", None),
                        getattr(response, "headers", None)):
            if not hasattr(headers, "items"):
                continue
            for key, value in headers.items():
                if (str(key).lower() == "retry-after" and isinstance(value, str)
                        and len(value) <= 256 and value not in values):
                    values.append(value)
    parsed = []
    for value in values:
        text = value.strip()
        if text and all("0" <= char <= "9" for char in text):
            # Avoid accepting a hostile unbounded integer as a float.
            seconds = float(int(text)) if len(text) <= 12 else 1e12
            parsed.append((seconds, value, "retry_after_seconds"))
            continue
        try:
            date = parsedate_to_datetime(text)
            if date.tzinfo is not None:
                parsed.append((max(0.0, (date - observed_at).total_seconds()), value, "retry_after_http_date"))
        except (ValueError, TypeError, OverflowError):
            pass
    if not parsed:
        return (values[0] if values else None), None, None
    seconds, raw, source = max(parsed, key=lambda item: item[0])
    return raw, seconds, source


















class FriendliLitellmModelConfig(LitellmModelConfig):
    dynamic_output_cap: bool = Field(default=False, strict=True)
    context_management: bool = Field(default=False, strict=True)
    context_trigger_tokens: int = Field(default=524288, strict=True, ge=131072, le=786432)
    context_target_bytes: int = Field(default=262144, strict=True, ge=65536, le=393216)
    context_tool_output_bytes: int = Field(default=16384, strict=True, ge=1024, le=65536)
    context_memory_bytes: int = Field(default=32768, strict=True, ge=4096, le=65536)
    context_min_reply_tokens: int = Field(default=8192, strict=True, ge=1024, le=32768)
    practical_reply_tokens: int = Field(default=65536, strict=True, ge=8192, le=131072)
    transport_retry_limit: int = Field(default=0, strict=True, ge=0, le=1)
    hard_api_deadline_seconds: float | None = Field(default=None, strict=True, gt=0, le=600, allow_inf_nan=False)
    semantic_error_retry: bool = Field(default=False, strict=True)
    shared_pool_rate_limit_retry: bool = Field(default=False, strict=True)
    malformed_response_retry: bool = Field(default=False, strict=True)
    api_cost_budget_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    api_input_price_per_million_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    api_output_price_per_million_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)


class FriendliLitellmModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=FriendliLitellmModelConfig, **kwargs)
        if self.config.model_name != (
            "openrouter/z-ai/glm-5.3-flash@preset/"
            "picbench-glm-5-3-flash-friendli-only"
        ):
            raise ValueError("Friendli transport requires the exact Friendli-only GLM route")
        self._context_anchor: tuple[tuple[str, ...], int] | None = None
        self._context_output: tuple[str, int] | None = None
        self._context_requests: list[dict] = []
        self._context_compactions: list[dict] = []
        self._context_epoch = 0
        self._context_cut: int | None = None
        self._context_prefix = 0
        self._context_memory_message: dict | None = None
        self._context_source_anchor: tuple[str, ...] | None = None
        if self.config.context_management:
            if not self.config.dynamic_output_cap:
                raise ValueError("Context management requires dynamic output budgeting")
            if (len(_encoded([BASH_TOOL])) != TOOL_SCHEMA_UTF8_BYTES
                    or hashlib.sha256(_encoded([BASH_TOOL])).hexdigest() != TOOL_SCHEMA_SHA256):
                raise ValueError("Managed context requires the pinned Mini 2.4.6 bash tool declaration")
            if self.config.set_cache_control is not None:
                raise ValueError("Managed Friendli context requires no explicit cache markers")
            if self.config.context_min_reply_tokens > self.config.practical_reply_tokens:
                raise ValueError("Context minimum reply exceeds the practical reply ceiling")
            if self.config.context_target_bytes >= self.config.context_trigger_tokens:
                raise ValueError("Context target must be smaller than its proactive trigger")
        self._transport_attempts: list[dict] = []
        self._transport_request_count = 0
        self._retry_deadline_monotonic: float | None = None

    def _estimate_context(self, messages: list[dict]) -> dict:
        fingerprints = tuple(hashlib.sha256(_encoded(message)).hexdigest() for message in messages)
        anchor = self._context_anchor
        completion_bound = None
        if anchor is not None and fingerprints[:len(anchor[0])] == anchor[0]:
            base_tokens = anchor[1]
            appended = messages[len(anchor[0]):]
            output = self._context_output
            if appended and output is not None and fingerprints[len(anchor[0])] == output[0]:
                completion_bound = min(_input_bound(appended[:1]), output[1])
                growth_bound = completion_bound + _input_bound(appended[1:])
            else:
                growth_bound = _input_bound(appended)
            estimate = base_tokens + growth_bound
            method = ("provider_usage_plus_completion_and_appended_bytes" if completion_bound is not None
                      else "provider_usage_plus_appended_bytes")
        else:
            base_tokens = None
            growth_bound = _input_bound(messages) + len(_encoded([BASH_TOOL]))
            estimate = growth_bound
            method = "full_request_bytes"
        return {"fingerprints": fingerprints, "input_budget_tokens": estimate,
                "anchor_prompt_tokens": base_tokens, "new_input_byte_bound": growth_bound,
                "completion_budget_tokens": completion_bound, "estimator": method}

    def _manage_context(self, messages: list[dict], ceiling: int) -> tuple[list[dict], dict]:
        """Choose a bounded transport copy before output room becomes marginal."""
        source_fingerprints = tuple(hashlib.sha256(_encoded(message)).hexdigest() for message in messages)
        changed = (self._context_source_anchor is not None
                   and source_fingerprints[:len(self._context_source_anchor)] != self._context_source_anchor)
        if changed:
            self._context_cut = None
            self._context_memory_message = None
            self._context_anchor = None
            self._context_output = None
        prefix, turns = _context_turns(messages)
        self._context_prefix = prefix
        outgoing, elisions = _context_transport_window(
            messages, prefix, self._context_cut,
            self.config.context_tool_output_bytes, self.config.context_memory_bytes)
        estimate = self._estimate_context(outgoing)["input_budget_tokens"]
        available = CONTEXT_WINDOW - estimate - SAFETY_TOKENS
        minimum = min(ceiling, self.config.context_min_reply_tokens)
        trigger = (estimate >= self.config.context_trigger_tokens or available < minimum)
        compacted = False
        target_exceeded = False
        if trigger:
            hard_bound = CONTEXT_WINDOW - SAFETY_TOKENS - minimum
            best = None
            # Retain complete recent turns. Keep the latest atomic turn even
            # when a single large valid reasoning/tool call exceeds the target;
            # it must still fit the conservative hard bound and reply reserve.
            for begin, _end in reversed(turns):
                if begin <= prefix:
                    break
                candidate, candidate_elisions = _context_transport_window(
                    messages, prefix, begin,
                    self.config.context_tool_output_bytes, self.config.context_memory_bytes)
                bound = _input_bound(candidate) + len(_encoded([BASH_TOOL]))
                if best is None:
                    if bound > hard_bound:
                        break
                    best = (begin, candidate, candidate_elisions, bound)
                    if bound > self.config.context_target_bytes:
                        target_exceeded = True
                        break
                elif bound > self.config.context_target_bytes:
                    break
                else:
                    best = (begin, candidate, candidate_elisions, bound)
            if best is not None and (self._context_cut is None or best[0] > self._context_cut):
                cut, outgoing, elisions, _bound = best
                self._context_cut = cut
                self._context_memory_message = copy.deepcopy(outgoing[prefix])
                self._context_epoch += 1
                self._context_anchor = None
                self._context_output = None
                compacted = True
                self._context_compactions.append({
                    "epoch": self._context_epoch, "policy": CONTEXT_HISTORY_POLICY,
                    "source_message_count": len(messages),
                    "source_sha256": hashlib.sha256(_encoded(messages)).hexdigest(),
                    "prefix_count": prefix, "window_start": cut,
                    "omitted_prepared_range": [prefix, cut],
                    "omitted_sha256": hashlib.sha256(_encoded(messages[prefix:cut])).hexdigest(),
                    "memory_sha256": hashlib.sha256(_encoded(outgoing[prefix])).hexdigest(),
                    "effective_message_count": len(outgoing),
                    "effective_sha256": hashlib.sha256(_encoded(outgoing)).hexdigest(),
                    "trigger_input_budget_tokens": estimate,
                    "target_exceeded_for_atomic_latest_turn": target_exceeded,
                })
        self._context_source_anchor = source_fingerprints
        proof = {"context_epoch": self._context_epoch, "source_history_changed": changed,
                 "compacted_this_request": compacted, "source_message_count": len(messages),
                 "source_sha256": hashlib.sha256(_encoded(messages)).hexdigest(),
                 "effective_message_count": len(outgoing),
                 "effective_sha256": hashlib.sha256(_encoded(outgoing)).hexdigest(),
                 "effective_message_sha256": [hashlib.sha256(_encoded(message)).hexdigest() for message in outgoing],
                 "prefix_count": prefix, "window_start": self._context_cut,
                 "tool_observation_elisions": elisions}
        return outgoing, proof

    def _retry_remaining_seconds(self) -> float | None:
        """Read and pin the existing trusted phase deadline without extending it."""
        try:
            parent = _BENCHMARK_STATUS.parent.lstat()
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != _TRUSTED_STATUS_UID
                    or parent.st_mode & 0o022):
                return None
            descriptor = os.open(_BENCHMARK_STATUS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                metadata = os.fstat(handle.fileno())
                if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != _TRUSTED_STATUS_UID
                        or metadata.st_nlink != 1 or metadata.st_mode & 0o222
                        or not 0 < metadata.st_size <= 16384):
                    return None
                raw = handle.read(16385)
            if len(raw.encode("utf-8")) > 16384:
                return None
            status = json.loads(raw)
            if (not isinstance(status, dict) or type(status.get("schema_version")) is not int
                    or status.get("schema_version") != 1 or status.get("source") != "harbor_agent_phase"):
                return None
            timing = status.get("timing")
            deadline_text = timing.get("deadline_at_utc") if isinstance(timing, dict) else None
            if not isinstance(deadline_text, str):
                return None
            deadline = datetime.fromisoformat(deadline_text.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                return None
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            candidate = time.monotonic() + remaining
            if not math.isfinite(candidate):
                return None
            self._retry_deadline_monotonic = min(
                candidate, self._retry_deadline_monotonic if self._retry_deadline_monotonic is not None else candidate
            )
            return self._retry_deadline_monotonic - time.monotonic()
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            return None

    def _reserve_api_budget(self, messages: list[dict], parameters: dict, event: dict) -> dict | None:
        """Reserve an upper charge before each physical diagnostic attempt."""
        limit = self.config.api_cost_budget_usd
        if limit is None:
            return None
        input_price = self.config.api_input_price_per_million_usd
        output_price = self.config.api_output_price_per_million_usd
        output_limit = parameters.get("max_tokens")
        if input_price is None or output_price is None or type(output_limit) is not int or output_limit <= 0:
            raise ValueError("Friendli API budget requires explicit price ceilings and max_tokens")
        signature = (Decimal(str(limit)), Decimal(str(input_price)), Decimal(str(output_price)))
        state = getattr(self, "_api_budget_state", None)
        if state is None:
            state = self._api_budget_state = {
                "signature": signature, "reported": Decimal(0), "unknown": Decimal(0), "events": []}
        if state["signature"] != signature:
            raise ValueError("Friendli diagnostic price/budget bindings changed during a run")
        # The byte bound includes message framing and images, then the actual
        # fixed tool schema. It deliberately ignores cache discounts.
        input_bound = _input_bound(messages) + len(_encoded([BASH_TOOL]))
        quote = (Decimal(input_bound) * signature[1] + Decimal(output_limit) * signature[2]) / Decimal(1000000)
        record = {"request": event["request"], "attempt": event["attempt"],
                  "input_token_upper_bound": input_bound, "output_token_limit": output_limit,
                  "reservation_usd": quote, "provider_reported_charge_usd": None,
                  "accounted_usd": Decimal(0), "status": "blocked"}
        state["events"].append(record)
        event["api_budget_reservation_usd"] = float(quote)
        if state["reported"] + state["unknown"] + quote > signature[0]:
            event.update(status="failed", exception_type="FriendliAPIBudgetError", not_sent=True,
                         transient_kind=None, http_status=None, provider_cost_usd=0.0,
                         provider_billing_unknown=False, elapsed_seconds=0.0,
                         api_budget_status="blocked", api_budget_accounted_usd=0.0,
                         finished_at_utc=datetime.now(timezone.utc).isoformat())
            raise FriendliAPIBudgetError("Friendli diagnostic model budget cannot reserve the next attempt")
        state["unknown"] += quote
        record.update(accounted_usd=quote, status="reserved")
        event.update(api_budget_status="reserved", api_budget_accounted_usd=float(quote))
        return record

    def _settle_api_budget(self, record: dict | None, event: dict) -> None:
        if record is None:
            return
        state = self._api_budget_state
        if record["status"] != "reserved":
            raise ValueError("Friendli API reservation was settled more than once")
        quote = record["reservation_usd"]
        cost = event.get("provider_cost_usd")
        known = (event.get("provider_billing_unknown") is False
                 and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0)
        state["unknown"] -= quote
        if known:
            accounted = Decimal(str(cost))
            state["reported"] += accounted
            record.update(status="settled_known", provider_reported_charge_usd=accounted)
        else:
            # Retain the pre-call maximum for interrupted/unknown charges. A
            # larger returned upstream estimate can increase this conservative
            # budget charge, but never becomes fabricated billed usage.
            upstream = event.get("provider_upstream_inference_cost_usd")
            accounted = max(quote, Decimal(str(upstream))) if type(upstream) in (int, float) and math.isfinite(upstream) and upstream >= 0 else quote
            state["unknown"] += accounted
            record.update(status="retained_unknown")
        record["accounted_usd"] = accounted
        event.update(api_budget_status=record["status"], api_budget_accounted_usd=float(accounted))
        if state["reported"] + state["unknown"] > state["signature"][0]:
            event["api_budget_exceeded"] = True
            raise FriendliAPIBudgetError("Returned provider accounting exceeded the reserved diagnostic model budget")

    def _api_budget_proof(self) -> dict | None:
        state = getattr(self, "_api_budget_state", None)
        if state is None:
            return None
        limit, input_price, output_price = state["signature"]
        return {"schema_version": 1, "limit_usd": float(limit),
                "input_price_per_million_usd": float(input_price),
                "output_price_per_million_usd": float(output_price),
                "accounted_upper_usd": float(state["reported"] + state["unknown"]),
                "reported_charge_usd": float(state["reported"]),
                "unknown_charge_reserve_usd": float(state["unknown"]),
                "events": [{k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()}
                           for row in state["events"]]}

    def _transport_query(self, messages: list[dict], **kwargs):
        """Replay at most one complete unchanged request before any tool parsing."""
        limit = self.config.transport_retry_limit
        hard_limit = self.config.hard_api_deadline_seconds
        semantic_errors = self.config.semantic_error_retry
        new_recovery = self.config.shared_pool_rate_limit_retry or self.config.malformed_response_retry
        if not limit and hard_limit is None and not semantic_errors and not new_recovery and self.config.api_cost_budget_usd is None:
            return super()._query(messages, **kwargs)
        if semantic_errors and hard_limit is None:
            raise ValueError("Friendli semantic-error handling requires a hard API deadline")
        if new_recovery and hard_limit is None:
            raise ValueError("Friendli transient API recovery requires a hard API deadline")
        if self.config.api_cost_budget_usd is not None:
            if hard_limit is None or self.config.api_input_price_per_million_usd is None or self.config.api_output_price_per_million_usd is None:
                raise ValueError("Friendli API budget requires a hard deadline and explicit price ceilings")
        if hard_limit is not None and (threading.current_thread() is not threading.main_thread()
                                      or not hasattr(signal, "setitimer")):
            raise RuntimeError("Friendli elapsed deadlines require the POSIX main thread")
        combined = self.config.model_kwargs | kwargs
        timeout = combined.get("timeout")
        if (os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") != "1"
                or type(combined.get("num_retries")) is not int or combined["num_retries"] != 0):
            raise ValueError("Friendli retry requires Mini attempts1 and LiteLLM num_retries0")
        if ((combined.get("stream") is not None and combined.get("stream") is not False) or isinstance(timeout, bool)
                or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Friendli retry requires a bounded nonstreaming request")
        frozen_messages = copy.deepcopy(messages)
        frozen_parameters = copy.deepcopy(combined)
        fingerprint = hashlib.sha256(_encoded({
            "model": self.config.model_name, "messages": frozen_messages,
            "parameters": {key: value for key, value in frozen_parameters.items()
                           if key not in {"api_key", "headers", "extra_headers"}},
        })).hexdigest()
        self._transport_request_count += 1
        for attempt in range(limit + 1):
            # Pin the phase clock before inference. A missing/expired clock
            # remains visible in evidence and never permits an extra attempt.
            remaining_at_start = self._retry_remaining_seconds()
            event = {"request": self._transport_request_count, "attempt": attempt + 1,
                     "request_sha256": fingerprint, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "status": "started", "response_received": False,
                     "trusted_phase_seconds_remaining": remaining_at_start,
                     "provider_cost_usd": None, "provider_billing_unknown": True}
            self._transport_attempts.append(event)
            effective_limit = None
            if hard_limit is not None:
                if (remaining_at_start is None or not math.isfinite(remaining_at_start)
                        or remaining_at_start <= 0):
                    event.update(status="failed", exception_type="ValueError", not_sent=True,
                                 transient_kind=None, http_status=None, provider_cost_usd=0.0,
                                 provider_billing_unknown=False, elapsed_seconds=0.0,
                                 finished_at_utc=datetime.now(timezone.utc).isoformat())
                    raise ValueError("Friendli hard API deadline requires an available trusted phase clock")
                effective_limit = min(hard_limit, remaining_at_start)
                event.update(hard_api_deadline_seconds=hard_limit, hard_deadline_seconds=effective_limit)
            reservation = self._reserve_api_budget(frozen_messages, frozen_parameters, event)
            attempt_started = time.monotonic()
            try:
                if effective_limit is None:
                    response = super()._query(copy.deepcopy(frozen_messages), **copy.deepcopy(frozen_parameters))
                else:
                    with _elapsed_api_deadline(effective_limit):
                        response = super()._query(copy.deepcopy(frozen_messages), **copy.deepcopy(frozen_parameters))
                if semantic_errors:
                    response_error = _provider_response_error(response)
                    if response_error is not None:
                        raise response_error
            except Exception as error:
                kind, http_status = _transient_transport_error(error)
                sdk_failure = _sdk_failure_evidence(
                    error, model_name=self.config.model_name,
                    shared_pool_retry=self.config.shared_pool_rate_limit_retry,
                    malformed_retry=self.config.malformed_response_retry,
                ) if new_recovery else None
                if sdk_failure is not None:
                    kind, http_status = sdk_failure["transient_kind"], sdk_failure["http_status"]
                event.update(status="failed", exception_type=type(error).__name__,
                             transient_kind=kind, http_status=http_status,
                             elapsed_seconds=max(0.0, time.monotonic() - attempt_started),
                             finished_at_utc=datetime.now(timezone.utc).isoformat())
                if sdk_failure is not None:
                    event.update(sdk_failure)
                if type(error) is FriendliProviderResponseError:
                    returned = error.response
                    event.update(response_received=True, response_id=returned.get("id"),
                                 response=copy.deepcopy(returned),
                                 response_sha256=hashlib.sha256(_encoded(returned)).hexdigest(),
                                 response_error_code=error.response_error_code,
                                 error_type=error.error_type, partial_output=error.partial_output,
                                 partial_reasoning=error.partial_reasoning,
                                 usable_partial_output=error.usable_partial_output,
                                 actionable_output_absent=not error.usable_partial_output,
                                 rejected_reasoning_not_executed=error.partial_reasoning)
                    _record_response_cost(event, returned)
                self._settle_api_budget(reservation, event)
                if kind is None or attempt == limit:
                    raise
                retry_allowance = hard_limit if hard_limit is not None else timeout
                delay = _RETRY_BACKOFF_SECONDS
                if new_recovery:
                    event["retry_delay_source"] = "transport_default"
                    if kind == "shared_pool_rate_limit":
                        observed_at = datetime.now(timezone.utc)
                        header, retry_after, source = _retry_after(error, observed_at)
                        event.update(retry_after_observed_at_utc=observed_at.isoformat(),
                                     retry_after_header=header, retry_after_seconds=retry_after,
                                     retry_delay_source=source or "shared_pool_default")
                        if retry_after is not None and retry_after > _MAX_RETRY_AFTER_SECONDS:
                            event["retry_skipped"] = "retry_after_exceeds_bounded_wait"
                            raise
                        delay = max(_SHARED_POOL_BACKOFF_SECONDS, retry_after or 0.0)
                    event["retry_delay_seconds"] = delay
                remaining = self._retry_remaining_seconds()
                if remaining is None or remaining < retry_allowance + delay:
                    event["retry_skipped"] = "trusted_deadline_unavailable_or_insufficient"
                    raise
                wait_started = time.monotonic()
                if new_recovery:
                    event["retry_wait_started_at_utc"] = datetime.now(timezone.utc).isoformat()
                time.sleep(delay)
                remaining = self._retry_remaining_seconds()
                if new_recovery:
                    event.update(retry_wait_finished_at_utc=datetime.now(timezone.utc).isoformat(),
                                 retry_delay_elapsed_seconds=max(0.0, time.monotonic() - wait_started),
                                 trusted_phase_seconds_before_retry=remaining)
                if remaining is None or remaining < retry_allowance:
                    event["retry_skipped"] = "trusted_deadline_unavailable_or_insufficient"
                    raise
                event["retry_scheduled"] = True
            else:
                event.update(status="succeeded", response_received=True,
                             response_id=_response_value(response, "id"),
                             elapsed_seconds=max(0.0, time.monotonic() - attempt_started),
                             finished_at_utc=datetime.now(timezone.utc).isoformat())
                _record_response_cost(event, response)
                try:
                    self._settle_api_budget(reservation, event)
                except FriendliAPIBudgetError:
                    returned = _response_dict(response)
                    event.update(status="failed", exception_type="FriendliAPIBudgetError",
                                 response=returned, response_sha256=hashlib.sha256(_encoded(returned)).hexdigest())
                    raise
                return response

    def _query(self, messages: list[dict], **kwargs):
        if not self.config.dynamic_output_cap:
            return self._transport_query(messages, **kwargs)
        parameters = self.config.model_kwargs | kwargs
        ceiling = parameters.get("max_tokens")
        if type(ceiling) is not int or not 0 < ceiling <= OUTPUT_CEILING:
            raise ValueError("Dynamic Friendli cap requires an explicit positive output ceiling <=943718")
        if parameters.get("max_completion_tokens") is not None or parameters.get("stream"):
            raise ValueError("Dynamic Friendli cap requires unambiguous nonstreaming max_tokens")
        if type(parameters.get("num_retries")) is not int or parameters["num_retries"] != 0:
            raise ValueError("Dynamic Friendli cap requires zero model retries")
        managed_proof = {}
        if self.config.context_management:
            ceiling = min(ceiling, self.config.practical_reply_tokens)
            if ceiling < self.config.context_min_reply_tokens:
                raise ValueError("Managed context needs at least its useful minimum reply allowance")
            messages, managed_proof = self._manage_context(messages, ceiling)
        budget = self._estimate_context(messages)
        fingerprints = budget.pop("fingerprints")
        estimate = budget["input_budget_tokens"]
        cap = min(ceiling, CONTEXT_WINDOW - estimate - SAFETY_TOKENS)
        event = {"request": len(self._context_requests) + 1, "status": "started",
                 **budget, **managed_proof,
                 "requested_max_tokens": max(0, cap), "output_ceiling": ceiling,
                 "context_window_tokens": CONTEXT_WINDOW, "safety_tokens": SAFETY_TOKENS}
        self._context_requests.append(event)
        minimum = min(ceiling, self.config.context_min_reply_tokens) if self.config.context_management else 1
        if cap < minimum:
            event["status"] = "blocked_no_context_space"
            raise ValueError("Conversation has exhausted the guarded context window; no model request sent")
        try:
            # The original native history is never changed. The explicit
            # managed policy can use its disclosed replayable transport window.
            response = self._transport_query(messages, **(kwargs | {"max_tokens": cap}))
        except Exception as exc:
            event.update(status="failed", exception_type=type(exc).__name__)
            raise
        usage = getattr(response, "usage", None)
        observed = getattr(usage, "prompt_tokens", None)
        event.update(status="succeeded", response_id=getattr(response, "id", None),
                     actual_prompt_tokens=observed,
                     actual_completion_tokens=getattr(usage, "completion_tokens", None))
        if type(observed) is int and 0 < observed < CONTEXT_WINDOW:
            self._context_anchor = (fingerprints, observed)
            self._context_output = None
            completed = getattr(usage, "completion_tokens", None)
            choices = getattr(response, "choices", None)
            if type(completed) is int and completed > 0 and choices:
                try:
                    assistant = self._prepare_messages_for_api([choices[0].message.model_dump()])[0]
                except ValueError:
                    # This optimization must not hide a completed response
                    # before native Mini persists its exact payload and cost.
                    # Normal preparation of a later continuation still rejects
                    # unsupported metadata without sending another API call.
                    event["completion_accounting_fallback"] = "returned_message_preparation_rejected"
                else:
                    bound = _returned_assistant_bound(assistant, completed)
                    if bound is not None:
                        self._context_output = (hashlib.sha256(_encoded(assistant)).hexdigest(), bound)
        else:
            # Missing accounting never grants extra input capacity.
            self._context_anchor = None
            self._context_output = None
        return response

    def serialize(self) -> dict:
        value = super().serialize()
        budget_proof = self._api_budget_proof()
        if budget_proof is not None:
            value["info"]["picbench_api_budget"] = budget_proof
        if (self.config.transport_retry_limit or self.config.hard_api_deadline_seconds is not None
                or self.config.semantic_error_retry or self.config.shared_pool_rate_limit_retry
                or self.config.malformed_response_retry):
            new_transport = self.config.hard_api_deadline_seconds is not None or self.config.semantic_error_retry
            value["info"]["picbench_transport"] = {
                "schema_version": 2 if new_transport else 1, "retry_limit": self.config.transport_retry_limit,
                **({"hard_api_deadline_seconds": self.config.hard_api_deadline_seconds,
                    "semantic_error_retry": self.config.semantic_error_retry} if new_transport else {}),
                **({"shared_pool_rate_limit_retry": self.config.shared_pool_rate_limit_retry,
                    "malformed_response_retry": self.config.malformed_response_retry,
                    "shared_pool_retry_delay_seconds": _SHARED_POOL_BACKOFF_SECONDS,
                    "retry_after_max_seconds": _MAX_RETRY_AFTER_SECONDS}
                   if self.config.shared_pool_rate_limit_retry or self.config.malformed_response_retry else {}),
                "attempts": copy.deepcopy(self._transport_attempts),
            }
        if self.config.dynamic_output_cap:
            configured_ceiling = self.config.model_kwargs.get("max_tokens")
            effective_ceiling = (min(configured_ceiling, self.config.practical_reply_tokens)
                                 if self.config.context_management and type(configured_ceiling) is int
                                 else configured_ceiling)
            value["info"]["picbench_context_budget"] = {
                "schema_version": 3 if self.config.context_management else 2,
                "context_window_tokens": CONTEXT_WINDOW,
                "completion_usage_policy": "exact_returned_plaintext_equivalent_reasoning_once_v2",
                "output_ceiling": effective_ceiling,
                "safety_tokens": SAFETY_TOKENS, "image_reserve_tokens": IMAGE_RESERVE,
                "history_policy": CONTEXT_HISTORY_POLICY if self.config.context_management else "preserve_all_messages",
                **({"native_history_policy": "preserve_all_messages",
                    "tool_schema_sha256": TOOL_SCHEMA_SHA256,
                    "tool_schema_utf8_bytes": TOOL_SCHEMA_UTF8_BYTES,
                    "context_management": {
                        "trigger_tokens": self.config.context_trigger_tokens,
                        "target_bytes": self.config.context_target_bytes,
                        "tool_output_bytes": self.config.context_tool_output_bytes,
                        "memory_bytes": self.config.context_memory_bytes,
                        "min_reply_tokens": self.config.context_min_reply_tokens,
                        "practical_reply_tokens": self.config.practical_reply_tokens,
                    }, "compactions": copy.deepcopy(self._context_compactions)}
                   if self.config.context_management else {}),
                "requests": copy.deepcopy(self._context_requests),
            }
        return value

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        if self.config.context_management:
            return prepare_native_context_messages(messages, set_cache_control=self.config.set_cache_control)
        prepared = copy.deepcopy(super()._prepare_messages_for_api(messages))
        for message in prepared:
            if "provider_specific_fields" not in message:
                continue
            fields = message.pop("provider_specific_fields")
            if fields is None:
                continue
            if not isinstance(fields, dict):
                raise ValueError("Invalid provider metadata; refusing a lossy continuation")
            for name, value in fields.items():
                if value is None:
                    continue
                if name not in {"reasoning", "reasoning_content", "reasoning_details", "refusal"}:
                    raise ValueError("Unknown provider metadata; refusing a lossy continuation")
                if name == "reasoning_details":
                    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                        raise ValueError("Invalid reasoning details; refusing a lossy continuation")
                elif not isinstance(value, str):
                    raise ValueError("Invalid reasoning/refusal text; refusing a lossy continuation")
                if name in message and message[name] != value:
                    raise ValueError("Conflicting provider metadata; refusing a lossy continuation")
                message[name] = value
        return prepared
