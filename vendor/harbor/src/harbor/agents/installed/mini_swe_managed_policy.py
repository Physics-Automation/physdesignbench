"""SDK-free, replayable policy for the explicitly selected Mini harness study.

These model limits are separate from the historical GLM/Friendli constants.
The context transformations reuse its reviewed complete-turn helpers, which
never mutate the native transcript or edit retained signed reasoning blocks.
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
from pathlib import Path

if __package__:
    from .mini_swe_friendli_context import (
        _encoded, _input_bound, _context_turns, _context_transport_window,
        prepare_native_context_messages, validate_context_request_replay,
        SAFETY_TOKENS, IMAGE_RESERVE, TOOL_SCHEMA_UTF8_BYTES, TOOL_SCHEMA_SHA256,
        CONTEXT_HISTORY_POLICY,
    )
else:
    from picbench_friendli_context import (
        _encoded, _input_bound, _context_turns, _context_transport_window,
        prepare_native_context_messages, validate_context_request_replay,
        SAFETY_TOKENS, IMAGE_RESERVE, TOOL_SCHEMA_UTF8_BYTES, TOOL_SCHEMA_SHA256,
        CONTEXT_HISTORY_POLICY,
    )

POLICY = "managed_openrouter_stream_v1"
# Intentional SDK-free re-exports consumed by the staged transport module.
__all__ = [
    "CONTEXT_HISTORY_POLICY", "IMAGE_RESERVE", "SAFETY_TOKENS",
    "TOOL_SCHEMA_SHA256", "TOOL_SCHEMA_UTF8_BYTES", "_context_transport_window",
    "_context_turns", "_encoded", "_input_bound",
]

COMPLETION_USAGE_POLICY = "provider_usage_and_verified_signed_plaintext_once_v2"
_BENCHMARK_STATUS = Path("/run/picbench/benchmark-status.json")
_TRUSTED_STATUS_UID = 0
MANAGED_MODEL_POLICIES = {
    "openrouter/openai/gpt-6.1-sol@preset/picbench-gpt-6-1-sol-openai-only": {
        "model": "openai/gpt-6.1-sol", "provider": "openai",
        "context_window_tokens": 1050000, "max_output_tokens": 128000,
        "max_prompt_tokens": 922000,
        # OpenRouter's base ``only:[openai]`` excludes service-tier endpoints;
        # the runtime separately rejects service-tier overrides. These ceilings
        # cover standard long-context cache writes/input and output. They are
        # prospective reservations, never an estimated actual charge.
        "input_price_per_million_usd": 5.0, "output_price_per_million_usd": 15.0,
    },
    "openrouter/anthropic/claude-sonnet-5.5@preset/picbench-sonnet-5-5-anthropic-only": {
        "model": "anthropic/claude-sonnet-5.5", "provider": "anthropic",
        "context_window_tokens": 1000000, "max_output_tokens": 128000,
        "max_prompt_tokens": None,
        # Input upper price also covers the advertised one-hour cache write.
        "input_price_per_million_usd": 4.0, "output_price_per_million_usd": 10.0,
    },
}


def managed_policy(model: str) -> dict:
    if model not in MANAGED_MODEL_POLICIES:
        raise ValueError("Managed Mini transport requires an exact reviewed model/provider preset")
    return copy.deepcopy(MANAGED_MODEL_POLICIES[model])


def prepare_managed_messages(messages: list[dict], *, set_cache_control=None) -> list[dict]:
    """Move exact SDK reasoning fields without inventing or altering signatures.

    The shared pure helper accepts opaque signed/encrypted blocks unchanged,
    rejects conflicting duplicate fields, strips only native ``extra``, and
    replays Mini's typed-thinking ordering. Explicit cache markers have not
    been qualified for this transport and are rejected.
    """
    return prepare_native_context_messages(messages, set_cache_control=set_cache_control)


def replay_managed_request(native_messages: list[dict], event: dict, config: dict) -> dict:
    """Reconstruct normalized source and effective transport bytes offline."""
    return validate_context_request_replay(native_messages, event, config)


def message_hashes(messages: list[dict]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(_encoded(message)).hexdigest() for message in messages)


def returned_assistant_bound(assistant: dict, completed: int) -> int | None:
    """Count verified equivalent returned plaintext once, retaining signatures.

    This optimization applies only to the exact next assistant message bound
    to a completed provider response. Every plaintext representation must be
    byte-equal. Signatures/IDs/framing remain at their full byte bound. Opaque
    encrypted blocks, distinct summaries and unfamiliar metadata never receive
    a generated-token optimization.
    """
    allowed = {"role", "content", "tool_calls", "function_call", "reasoning",
               "reasoning_content", "reasoning_details", "refusal"}
    if (type(completed) is not int or completed <= 0 or assistant.get("role") != "assistant"
            or any(key not in allowed and value is not None for key, value in assistant.items())
            or any(assistant.get(key) is not None and not isinstance(assistant[key], str)
                   for key in ("content", "reasoning", "reasoning_content", "refusal"))):
        return None
    representations = [assistant[key] for key in ("reasoning", "reasoning_content") if assistant.get(key)]
    details = assistant.get("reasoning_details")
    if details is not None:
        if not isinstance(details, list):
            return None
        plaintext = []
        for part in details:
            if (not isinstance(part, dict) or part.get("type") != "reasoning.text"
                    or not isinstance(part.get("text"), str)
                    or any(key not in {"type", "text", "signature", "format", "index", "id"}
                           and value is not None for key, value in part.items())
                    or part.get("signature") is not None and not isinstance(part["signature"], str)):
                return None
            plaintext.append(part["text"])
        if plaintext:
            representations.append("".join(plaintext))
    if representations and len(set(representations)) != 1:
        return None
    framing = copy.deepcopy(assistant)
    for key in ("content", "reasoning", "reasoning_content", "refusal"):
        if framing.get(key) is not None:
            framing[key] = ""
    for part in framing.get("reasoning_details") or []:
        part["text"] = ""  # The exact signature stays present and is counted.
    calls = framing.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list):
            return None
        for call in calls:
            if (not isinstance(call, dict) or call.get("type") != "function"
                    or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("arguments"), str)):
                return None
            call["function"]["arguments"] = ""
    function = framing.get("function_call")
    if function is not None:
        if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
            return None
        function["arguments"] = ""
    return completed + max(1024, len(_encoded(framing)))


def conservative_input_bound(messages: list[dict], known_exact_bounds: dict[str, int]) -> int:
    """Use prior provider-bound exact messages in fresh windows as well.

    The caller populates this map only from previously completed response
    messages and their reported completion usage. Unknown/changed/encrypted
    messages retain the ordinary full-byte bound, including image allowances.
    """
    bound = _input_bound(messages)
    for message in messages:
        digest = hashlib.sha256(_encoded(message)).hexdigest()
        known = known_exact_bounds.get(digest)
        size = len(_encoded(message))
        if type(known) is int and 0 < known < size:
            bound -= size - known
    return bound


def request_digest(model: str, messages: list[dict], parameters: dict) -> str:
    # Credentials remain outside all retained request fingerprints/proofs.
    public = {key: value for key, value in parameters.items()
              if key not in {"api_key", "headers", "extra_headers"}}
    return hashlib.sha256(_encoded({"model": model, "messages": messages, "parameters": public})).hexdigest()


class ManagedElapsedDeadlineError(TimeoutError):
    """The complete streamed request exceeded its independent elapsed limit."""


class _ElapsedInterrupt(BaseException):
    # Bypass broad SDK Exception handlers and their internal normalization.
    pass


@contextmanager
def elapsed_api_deadline(seconds: float):
    """POSIX cancellation, including active chunk/heartbeat streams, no worker.

    This is the reviewed Friendli timer algorithm with a model-neutral error.
    It preserves any earlier process alarm and never extends its deadline.
    """
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL")):
        raise RuntimeError("Managed elapsed deadlines require the POSIX main thread")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Managed elapsed deadline must be positive and finite")
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
                previous_deadline = (previous_deadline + (math.floor((now - previous_deadline) / previous_interval) + 1) * previous_interval
                                     if previous_interval > 0 else None)
                if callable(previous_handler):
                    previous_handler(signum, frame)
                elif previous_handler == signal.SIG_DFL:
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
                    signal.raise_signal(signal.SIGALRM)
        finally:
            if time.monotonic() >= own_deadline:
                raise _ElapsedInterrupt()
            arm()

    signal.signal(signal.SIGALRM, alarm)
    arm()
    try:
        yield
    except _ElapsedInterrupt:
        raise ManagedElapsedDeadlineError("Managed model request exceeded its elapsed deadline") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, previous_deadline - time.monotonic()), previous_interval)
        else:
            signal.setitimer(signal.ITIMER_REAL, 0, previous_interval)


def trusted_remaining_seconds(previous_deadline: float | None) -> tuple[float | None, float | None]:
    """Read only the root-owned phase clock and pin its earliest deadline."""
    try:
        parent = _BENCHMARK_STATUS.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != _TRUSTED_STATUS_UID or parent.st_mode & 0o022:
            return None, previous_deadline
        descriptor = os.open(_BENCHMARK_STATUS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            before = os.fstat(handle.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != _TRUSTED_STATUS_UID or before.st_nlink != 1
                    or before.st_mode & 0o222 or not 0 < before.st_size <= 16384):
                return None, previous_deadline
            data = handle.read(16385)
            after = os.fstat(handle.fileno())
            if any(getattr(before, key) != getattr(after, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                return None, previous_deadline
        if len(data.encode("utf-8")) > 16384:
            return None, previous_deadline
        raw = json.loads(data)
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1 or raw.get("source") != "harbor_agent_phase":
            return None, previous_deadline
        deadline = datetime.fromisoformat(raw["timing"]["deadline_at_utc"].replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            return None, previous_deadline
        candidate = time.monotonic() + (deadline - datetime.now(timezone.utc)).total_seconds()
        if not math.isfinite(candidate):
            return None, previous_deadline
        pinned = min(previous_deadline, candidate) if previous_deadline is not None else candidate
        return pinned - time.monotonic(), pinned
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return None, previous_deadline
