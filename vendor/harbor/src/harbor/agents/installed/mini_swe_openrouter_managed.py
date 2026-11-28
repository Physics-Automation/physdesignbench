"""Explicit streamed Mini transport with model-specific time/context budgets.

This changes transport and accounting only. It supplies no task solution,
executes no tools and changes no scientific deadline. The default is zero retries;
an explicitly selected single transport retry only covers a typed transient HTTP
failure before a response stream opens, inside the original logical deadline.
Only a fully assembled successful response can reach Mini's action parser.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import litellm
import httpx
from pydantic import Field
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

if __package__:
    from .mini_swe_managed_policy import (
        POLICY, COMPLETION_USAGE_POLICY, CONTEXT_HISTORY_POLICY, SAFETY_TOKENS, IMAGE_RESERVE, MANAGED_MODEL_POLICIES,
        TOOL_SCHEMA_SHA256, TOOL_SCHEMA_UTF8_BYTES, _encoded, _input_bound,
        _context_turns, _context_transport_window, managed_policy, message_hashes,
        prepare_managed_messages, request_digest, elapsed_api_deadline,
        trusted_remaining_seconds, ManagedElapsedDeadlineError, returned_assistant_bound, conservative_input_bound,
    )
else:
    from picbench_managed_policy import (
        POLICY, COMPLETION_USAGE_POLICY, CONTEXT_HISTORY_POLICY, SAFETY_TOKENS, IMAGE_RESERVE, MANAGED_MODEL_POLICIES,
        TOOL_SCHEMA_SHA256, TOOL_SCHEMA_UTF8_BYTES, _encoded, _input_bound,
        _context_turns, _context_transport_window, managed_policy, message_hashes,
        prepare_managed_messages, request_digest, elapsed_api_deadline,
        trusted_remaining_seconds, ManagedElapsedDeadlineError, returned_assistant_bound, conservative_input_bound,
    )


class ManagedResponseError(RuntimeError):
    """A partial, truncated, malformed or provider-error response is not an action."""

    def __init__(self, reason: str, response: dict | None = None):
        super().__init__(reason)
        self.response = response


class ManagedAPIBudgetError(RuntimeError):
    pass


STREAM_PROTOCOL_POLICY = "openrouter-raw-sse-evidence-v2"


def _dict(value) -> dict:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise ManagedResponseError("Stream returned an unsupported chunk type")


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _transient_http_before_stream(error: Exception, event: dict, parameters: dict) -> dict:
    """Classify original HTTP evidence, never SDK synthetic status or text.

    LiteLLM replaces a ServiceUnavailableError's response with a synthetic 503
    response. Its original HTTPX status exception survives in the cause/context
    chain; only that original response at the actual selected endpoint counts.
    An HTTP-200 SSE error, partial generation or unknown connection failure is
    never admitted by this narrowly bounded recovery policy.
    """
    if (isinstance(error, (ManagedResponseError, ManagedElapsedDeadlineError, ManagedAPIBudgetError))
            or event.get("stream_opened") is not False or event.get("response_received")
            or event.get("stream_chunks", 0) or event.get("raw_stream_chunks", 0)
            or event.get("raw_stream_bytes", 0) or event.get("response_id")):
        return {}
    base = parameters.get("api_base", "https://openrouter.ai/api/v1")
    if not isinstance(base, str):
        return {}
    try:
        expected = urlsplit(base.rstrip("/") + "/chat/completions")
    except ValueError:
        return {}
    pending, seen, candidates = [error], set(), []
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for child in (getattr(current, "__cause__", None), getattr(current, "__context__", None)):
            if isinstance(child, BaseException):
                pending.append(child)
        if not isinstance(current, httpx.HTTPStatusError):
            continue
        response = current.response
        request = current.request
        if not isinstance(response, httpx.Response) or not isinstance(request, httpx.Request):
            continue
        try:
            actual = urlsplit(str(request.url))
            response_request = response.request
        except (ValueError, RuntimeError):
            continue
        if (actual != expected or request.method != "POST" or response_request is not request
                or type(response.status_code) is not int):
            continue
        candidates.append((response.status_code, response.headers.get("retry-after")))
    if not candidates or len(set(candidates)) != 1:
        return {}
    status, retry_after = candidates[0]
    evidence = {"http_status": status, "http_status_source": "httpx_status_response",
                "http_status_request_matches": True}
    if status not in {502, 503, 504, 520}:
        return evidence
    present = retry_after is not None
    delay = 1.0
    if present:
        try:
            # HTTP permits nonnegative decimal integer seconds or an HTTP date.
            if retry_after.isascii() and retry_after.isdecimal():
                delay = float(int(retry_after))
            else:
                stamp = parsedate_to_datetime(retry_after)
                if stamp.tzinfo is None:
                    return evidence | {"retry_after_present": True, "retry_after_valid": False}
                delay = max(0.0, (stamp - datetime.now(timezone.utc)).total_seconds())
            if not math.isfinite(delay) or not 0 <= delay <= 30:
                return evidence | {"retry_after_present": True, "retry_after_valid": False}
        except (TypeError, ValueError, OverflowError):
            return evidence | {"retry_after_present": True, "retry_after_valid": False}
    return evidence | {"retry_classification": "transient_http_before_stream",
                       "retry_after_present": present, "retry_after_valid": True,
                       "retry_after_seconds": delay if present else None,
                       "retry_backoff_seconds": delay}


def _record_cost(event: dict, response: dict) -> None:
    usage = response.get("usage") or {}
    cost = usage.get("cost")
    details = usage.get("cost_details") or {}
    upstream = details.get("upstream_inference_cost")
    if _number(upstream):
        event["provider_upstream_inference_cost_usd"] = upstream
    if _number(cost) and (not _number(upstream) or upstream <= cost):
        event.update(provider_cost_usd=cost, provider_billing_unknown=False)
    else:
        # A zero gateway charge must not erase conflicting upstream billing.
        event.update(provider_cost_usd=cost if _number(cost) else None, provider_billing_unknown=True)


def _close_stream(stream, seconds=.25) -> list[str]:
    """Close the owned HTTP response first within one total cleanup grace.

    A cancelled inference can use at most this separate cleanup grace; neither
    the trusted learner deadline nor an inference worker is created/extended.
    """
    pending, seen, errors, closers = [stream], set(), [], []
    while pending and len(seen) < 32:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            closers.append((0 if isinstance(current, httpx.Response) else 1, close))
        for key in ("completion_stream", "response", "_response", "http_response", "streaming_response", "response_iterator"):
            child = getattr(current, key, None)
            if child is not None:
                pending.append(child)
    try:
        with elapsed_api_deadline(seconds):
            for _priority, close in sorted(closers, key=lambda item: item[0]):
                try:
                    close()
                except ManagedElapsedDeadlineError:
                    raise
                except Exception as error:
                    errors.append(type(error).__name__)
    except ManagedElapsedDeadlineError:
        errors.append("cleanup_elapsed_deadline")
    return errors


def _safe_error_identity(key, value):
    """Only selected model/provider names and OpenRouter generation IDs."""
    if not isinstance(value, str):
        return None
    if key == "model" and value in {item["model"] for item in MANAGED_MODEL_POLICIES.values()}:
        return value
    if key == "provider" and value.lower() in {item["provider"] for item in MANAGED_MODEL_POLICIES.values()}:
        return value.lower()
    if (key == "id" and value.startswith("gen-") and 4 < len(value) <= 128 and value.isascii()
            and all(character.isalnum() or character in "-_" for character in value)):
        return value
    return None


def _provider_error_response(raw):
    """Bounded, secret-free evidence for an HTTP-200 SSE error envelope.

    The SSE error code is not an HTTP status. Upstream messages and metadata
    can contain headers, credentials or echoed request text, so only their
    lengths/digests and narrowly typed identifiers enter the native log.
    """
    encoded = _encoded(raw)
    evidence = {"source": "openrouter_sse_error", "body_sha256": hashlib.sha256(encoded).hexdigest(),
                "body_utf8_bytes": len(encoded)}
    known_codes = {"api_error", "authentication_error", "context_length_exceeded", "insufficient_quota",
                   "invalid_request_error", "overloaded_error", "permission_error", "provider_error",
                   "rate_limit_error", "rate_limit_exceeded", "server_error"}
    error = raw.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if type(code) is int and -1000000 <= code <= 1000000 or isinstance(code, str) and code in known_codes:
            evidence["code"] = code
        elif code is not None:
            evidence["code_sha256"] = hashlib.sha256(_encoded(code)).hexdigest()
        kind = error.get("type")
        if isinstance(kind, str) and kind in known_codes:
            evidence["type"] = kind
        text = error.get("message")
        if isinstance(text, str):
            data = text.encode("utf-8")
            evidence.update(message_sha256=hashlib.sha256(data).hexdigest(), message_utf8_bytes=len(data))
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            provider = _safe_error_identity("provider", metadata.get("provider_name"))
            if provider is not None:
                evidence["provider_name"] = provider
            elif metadata.get("provider_name") is not None:
                evidence["provider_name_sha256"] = hashlib.sha256(_encoded(metadata["provider_name"])).hexdigest()
    response = {"id": None, "choices": [], "usage": {},
                "error": {"message": "Provider returned a streaming error", "evidence": evidence}}
    if "code" in evidence:
        response["error"]["code"] = evidence["code"]
    for key in ("id", "model", "provider"):
        value = _safe_error_identity(key, raw.get(key))
        if value is not None:
            response[key] = value
    usage = raw.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            if _number(usage.get(key)):
                response["usage"][key] = usage[key]
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost") if isinstance(usage.get("cost_details"), dict) else None
        if _number(upstream):
            response["usage"]["cost_details"] = {"upstream_inference_cost": upstream}
    return response


def _capture_openrouter_stream_metadata(stream, event):
    """Keep provider metadata before the pinned SDK discards nonstandard keys.

    Only this request's owned OpenRouter parser instance is instrumented. The
    unmodified raw SSE dictionary remains the authority for provider, terminal
    status, usage and billed cost; no prices are substituted for actual charges.
    """
    handler = getattr(stream, "completion_stream", None)
    parser = getattr(handler, "chunk_parser", None)
    captured = {"provider_terminal_seen": False, "chunks": [], "bytes": 0,
                "sha256": hashlib.sha256(), "failure": None}
    if (not callable(parser) or handler.__class__.__name__ != "OpenRouterChatCompletionStreamingHandler"):
        return captured, None

    def capture_chunk(*args, **kwargs):
        raw = kwargs.get("chunk", args[0] if args else None)
        if isinstance(raw, dict):
            # The SDK can merge/replace signed detail fragments before yielding
            # a normalized chunk. Keep the exact provider deltas first; all
            # critical fields are assembled from this bounded raw copy below.
            captured["bytes"] += len(_encoded(raw))
            if len(captured["chunks"]) >= 250000 or captured["bytes"] > 128 * 1024 * 1024:
                raise ManagedResponseError("Raw provider stream exceeded its bounded allowance")
            captured["chunks"].append(copy.deepcopy(raw))
            data = _encoded(raw)
            captured["sha256"].update(len(data).to_bytes(8, "big") + data)
            event["raw_provider_metadata_captured"] = True
            for key in ("id", "model", "provider"):
                if raw.get(key) is not None:
                    if key in captured and captured[key] != raw[key]:
                        raise ManagedResponseError("Raw SSE changed response identity/provider", copy.deepcopy(raw))
                    captured[key] = copy.deepcopy(raw[key])
            if isinstance(raw.get("usage"), dict):
                captured["usage"] = copy.deepcopy(raw["usage"])
            if "error" in raw:
                response = _provider_error_response(raw)
                for key in ("id", "model", "provider"):
                    value = _safe_error_identity(key, captured.get(key))
                    if value is not None:
                        response[key] = value
                if not response["usage"] and isinstance(captured.get("usage"), dict):
                    response["usage"] = _provider_error_response({"usage": captured["usage"]})["usage"]
                event["raw_provider_error"] = copy.deepcopy(response["error"]["evidence"])
                captured["failure"] = ManagedResponseError("Provider returned a raw SSE error", response)
                raise captured["failure"]
            for choice in raw.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    captured["provider_terminal_seen"] = True
                    captured["finish_reason"] = choice["finish_reason"]
        return parser(*args, **kwargs)

    def capture(*args, **kwargs):
        try:
            return capture_chunk(*args, **kwargs)
        except ManagedResponseError as error:
            captured["failure"] = error
            raise

    handler.chunk_parser = capture
    string_parser = getattr(handler, "_handle_string_chunk", None)
    if callable(string_parser):
        # The pinned SDK checks whether '[DONE]' occurs *anywhere* in the
        # line, before calling chunk_parser. A literal in JSON text/tool args
        # must remain data, while the exact sentinel is never proof of a
        # successful model finish_reason. Malformed JSON must not disappear.
        event.update(raw_sse_done_seen=False, raw_sse_lines=0, raw_sse_bytes=0)

        def parse_line(str_line):
            if not isinstance(str_line, str):
                raise ManagedResponseError("OpenRouter SSE line is not text")
            event["raw_sse_lines"] += 1
            event["raw_sse_bytes"] += len(str_line.encode("utf-8"))
            if event["raw_sse_lines"] > 250000 or event["raw_sse_bytes"] > 128 * 1024 * 1024:
                raise ManagedResponseError("OpenRouter SSE exceeded its bounded allowance")
            if str_line.startswith("data:"):
                payload = str_line[5:].strip()
                if payload == "[DONE]":
                    event["raw_sse_done_seen"] = True
                    return string_parser(str_line)
                try:
                    raw = json.loads(payload)
                except (TypeError, ValueError):
                    event["raw_sse_malformed_json"] = True
                    raise ManagedResponseError("OpenRouter SSE contained malformed JSON") from None
                if not isinstance(raw, dict):
                    raise ManagedResponseError("OpenRouter SSE JSON is not an object")
                return capture(chunk=raw)
            # SSE comments and event/id/retry fields carry no model delta.
            # Suppress the SDK's substring-DONE test on those protocol lines.
            if not str_line.strip() or str_line.startswith((":", "event:", "id:", "retry:")):
                return string_parser("")
            raise ManagedResponseError("OpenRouter SSE contained an unsupported protocol line")

        def capture_line(str_line):
            try:
                return parse_line(str_line)
            except ManagedResponseError as error:
                captured["failure"] = error
                raise

        handler._handle_string_chunk = capture_line

    def restore():
        handler.chunk_parser = parser
        if callable(string_parser):
            handler._handle_string_chunk = string_parser

    return captured, restore


class _CapturedStream:
    """Bounded inert raw chunks, with no network or SDK normalization."""
    def __init__(self, chunks):
        self.chunks = chunks
    def __iter__(self):
        return iter(self.chunks)
    def close(self):
        pass


def _reject_provider_error_content(response: dict) -> None:
    """A typed HTTP-200 provider error is never a successful model action."""
    for choice in response.get("choices") or []:
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            continue
        try:
            envelope = json.loads(content)
        except (ValueError, TypeError):
            continue
        if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
            error = envelope["error"]
            if type(error.get("code")) is int or isinstance(error.get("message"), str):
                raise ManagedResponseError("Provider returned a typed error inside a successful HTTP body", response)


def _merge_detail(target: dict, fragment: dict) -> None:
    """Reassemble declared streamed string fragments; retain opaque signatures."""
    for key, value in fragment.items():
        if value is None:
            continue
        if key in {"text", "summary", "signature", "data"}:
            if not isinstance(value, str) or key in target and not isinstance(target[key], str):
                raise ManagedResponseError("Invalid streamed reasoning string")
            target[key] = target.get(key, "") + value
        elif key in target and target[key] != value:
            raise ManagedResponseError("Conflicting streamed reasoning metadata")
        else:
            target[key] = copy.deepcopy(value)


def collect_stream(stream, event: dict) -> dict:
    """Assemble OpenRouter chunks before exposing any tool arguments to Mini.

    LiteLLM's generic stream builder replaces reasoning-detail lists with their
    last fragment. This assembler merges indexed detail fragments explicitly
    and rejects incomplete tool calls and non-success terminal reasons.
    """
    message = {"role": "assistant", "content": None}
    calls, details = {}, {}
    response = {"id": None, "model": None, "choices": [], "usage": {}}
    terminal = None
    chunk_count = byte_count = 0
    captured, restore = _capture_openrouter_stream_metadata(stream, event)
    try:
        for original in stream:
            raw = _dict(original)
            chunk_count += 1
            byte_count += len(_encoded(raw))
            event.update(stream_chunks=chunk_count, stream_bytes=byte_count)
            if chunk_count > 250000 or byte_count > 128 * 1024 * 1024:
                raise ManagedResponseError("Stream exceeded its bounded response allowance")
            if restore is not None:
                # Consumption remains through the pinned SDK, but its lossy
                # normalized deltas are not authoritative signed/tool content.
                continue
            if raw.get("error"):
                raise ManagedResponseError("Provider returned a streaming error", raw)
            for key in ("id", "model", "provider", "created", "system_fingerprint"):
                if raw.get(key) is not None:
                    if key in {"id", "model", "provider"} and response.get(key) not in (None, raw[key]):
                        raise ManagedResponseError("Stream changed response identity/provider")
                    response[key] = copy.deepcopy(raw[key])
            usage = raw.get("usage")
            if isinstance(usage, dict) and usage:
                # OpenRouter's terminal usage is cumulative, not a token delta.
                # SDKs may emit a synthetic estimate after the real terminal
                # usage. It must not overwrite confirmed provider accounting.
                if _number(usage.get("cost")) or not _number(response["usage"].get("cost")):
                    response["usage"].update(copy.deepcopy(usage))
            choices = raw.get("choices") or []
            if len(choices) > 1 or choices and choices[0].get("index", 0) != 0:
                raise ManagedResponseError("Managed stream requires one choice")
            for choice in choices:
                native = (choice.get("provider_specific_fields") or {}).get("native_finish_reason")
                if choice.get("error") or native in {"error", "content_filter"}:
                    raise ManagedResponseError("Provider rejected the streamed completion", raw)
                finish = choice.get("finish_reason")
                if finish is not None:
                    if terminal is not None and terminal != finish:
                        raise ManagedResponseError("Stream supplied conflicting terminal reasons")
                    terminal = finish
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ManagedResponseError("Malformed streamed delta")
                for key in ("content", "reasoning", "reasoning_content", "refusal"):
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise ManagedResponseError("Unsupported typed streamed text")
                        message[key] = (message.get(key) or "") + value
                wrapper = delta.get("provider_specific_fields") or {}
                if not isinstance(wrapper, dict) or any(key not in {"reasoning_details", "reasoning", "reasoning_content", "refusal"}
                                                       and value is not None for key, value in wrapper.items()):
                    raise ManagedResponseError("Unqualified streamed provider metadata")
                for key in ("reasoning", "reasoning_content", "refusal"):
                    value = wrapper.get(key)
                    if value is not None:
                        if key in delta and delta[key] is not None and delta[key] != value:
                            raise ManagedResponseError("Conflicting streamed provider text")
                        if delta.get(key) is None:
                            if not isinstance(value, str):
                                raise ManagedResponseError("Invalid streamed provider text")
                            message[key] = (message.get(key) or "") + value
                direct = delta.get("reasoning_details")
                wrapped = wrapper.get("reasoning_details")
                if direct is not None and wrapped is not None and direct != wrapped:
                    raise ManagedResponseError("Conflicting streamed signed details")
                fragments = direct if direct is not None else wrapped
                if fragments is not None:
                    if not isinstance(fragments, list):
                        raise ManagedResponseError("Invalid streamed signed details")
                    for ordinal, fragment in enumerate(fragments):
                        if not isinstance(fragment, dict):
                            raise ManagedResponseError("Invalid streamed reasoning fragment")
                        index = fragment.get("index", ordinal)
                        if type(index) is not int or index < 0:
                            raise ManagedResponseError("Invalid streamed reasoning index")
                        _merge_detail(details.setdefault(index, {}), fragment)
                for fragment in delta.get("tool_calls") or []:
                    if not isinstance(fragment, dict) or type(fragment.get("index")) is not int or fragment["index"] < 0:
                        raise ManagedResponseError("Invalid streamed tool index")
                    call = calls.setdefault(fragment["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        if call.get("id") not in (None, fragment["id"]):
                            raise ManagedResponseError("Stream changed tool-call identity")
                        call["id"] = fragment["id"]
                    if fragment.get("type") not in (None, "function"):
                        raise ManagedResponseError("Unqualified streamed tool type")
                    function = fragment.get("function") or {}
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise ManagedResponseError("Invalid streamed tool field")
                            call["function"][key] += value
        event["stream_iterator_exhausted"] = True
        if restore is not None:
            raw_event = {}
            try:
                response = collect_stream(_CapturedStream(captured["chunks"]), raw_event)
            finally:
                if "terminal_validation" in raw_event:
                    event["terminal_validation"] = raw_event["terminal_validation"]
            message = response["choices"][0]["message"]
            terminal = response["choices"][0]["finish_reason"]
            calls = {index: call for index, call in enumerate(message.get("tool_calls") or [])}
            event.update(raw_stream_chunks=len(captured["chunks"]), raw_stream_bytes=captured["bytes"],
                         raw_stream_response_sha256=hashlib.sha256(_encoded(response)).hexdigest())
        if details:
            message["provider_specific_fields"] = {"reasoning_details": [details[index] for index in sorted(details)]}
        if calls:
            ordered = [calls[index] for index in sorted(calls)]
            if any(not call.get("id") or call["function"]["name"] != "bash" for call in ordered):
                raise ManagedResponseError("Incomplete or unqualified streamed tool call")
            message["tool_calls"] = ordered
        response["choices"] = [{"index": 0, "finish_reason": terminal, "message": message}]
        if restore is not None:
            for key in ("id", "model", "provider", "usage"):
                if key in captured:
                    response[key] = copy.deepcopy(captured[key])
            if not captured["provider_terminal_seen"]:
                raise ManagedResponseError("Raw provider stream lacked a terminal completion", response)
            event["provider_terminal_seen"] = True
        if hasattr(stream, "received_finish_reason") and stream.received_finish_reason is None:
            raise ManagedResponseError("SDK synthesized an EOF completion without a provider terminal", response)
        if not response.get("id") or terminal not in {"stop", "tool_calls"}:
            event["terminal_validation"] = "missing_successful_finish_reason" if terminal is None else "unsuccessful_finish_reason"
            raise ManagedResponseError("Stream ended without a successful terminal response", response)
        if calls and terminal != "tool_calls" or terminal == "tool_calls" and not calls:
            raise ManagedResponseError("Terminal reason does not match streamed tools", response)
        _reject_provider_error_content(response)
        event.update(stream_complete=True, finish_reason=terminal)
        return response
    except Exception as error:
        if restore is not None:
            event["stream_failure_stage"] = ("provider_sse_error" if event.get("raw_provider_error")
                                             else "raw_stream_validation" if captured["failure"] is not None
                                             else "stream_consumption_or_completion")
            if captured["failure"] is not None and error is not captured["failure"]:
                # LiteLLM wraps even our own parser rejection in a synthetic
                # MidStreamFallbackError, discarding its response evidence.
                # Recover the owned original; never reinterpret it as HTTP or
                # allow the rejected partial generation to reach Mini.
                event["sdk_exception_type"] = type(error).__name__
                raise captured["failure"] from None
        raise
    finally:
        primary = sys.exc_info()[0] is not None
        if restore is not None:
            event.update(raw_stream_chunks=len(captured["chunks"]), raw_stream_bytes=captured["bytes"],
                         raw_stream_frames_sha256=captured["sha256"].hexdigest(),
                         raw_provider_terminal_seen=captured["provider_terminal_seen"])
        remaining, _ = trusted_remaining_seconds(None)
        grace = min(.25, remaining) if remaining is not None and remaining > 0 else .25
        event.update(stream_cleanup_grace_seconds=grace, stream_cleanup_max_grace_seconds=.25)
        cleanup_started = time.monotonic()
        errors = _close_stream(stream, grace)
        event["stream_cleanup_elapsed_seconds"] = max(0.0, time.monotonic() - cleanup_started)
        if restore is not None:
            restore()
        event["stream_cleanup_complete"] = not errors
        if errors:
            event["stream_cleanup_errors"] = errors
            event["stream_cleanup_timed_out"] = "cleanup_elapsed_deadline" in errors
            if not primary:
                raise ManagedResponseError("Stream cleanup did not complete within its bounded grace", response)


class ManagedLitellmModelConfig(LitellmModelConfig):
    managed_transport: bool = Field(default=True, strict=True)
    dynamic_output_cap: bool = Field(default=True, strict=True)
    context_management: bool = Field(default=True, strict=True)
    context_trigger_tokens: int | None = Field(default=None, strict=True, ge=1)
    context_target_bytes: int = Field(default=700000, strict=True, ge=65536)
    context_tool_output_bytes: int = Field(default=16384, strict=True, ge=1024, le=65536)
    context_memory_bytes: int = Field(default=32768, strict=True, ge=4096, le=65536)
    context_min_reply_tokens: int = Field(default=8192, strict=True, ge=1024, le=32768)
    practical_reply_tokens: int = Field(default=128000, strict=True, ge=8192, le=128000)
    hard_api_deadline_seconds: float = Field(default=3600.0, strict=True, gt=0, allow_inf_nan=False)
    transport_retry_limit: int = Field(default=0, strict=True, ge=0, le=1)
    api_cost_budget_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    api_input_price_per_million_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    api_output_price_per_million_usd: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)


class ManagedOpenRouterModel(LitellmModel):
    abort_exceptions = LitellmModel.abort_exceptions + [ManagedResponseError, ManagedElapsedDeadlineError, ManagedAPIBudgetError]

    def __init__(self, **kwargs):
        super().__init__(config_class=ManagedLitellmModelConfig, **kwargs)
        self._policy = managed_policy(self.config.model_name)
        config = self.config
        body = config.model_kwargs.get("extra_body", {})
        if not isinstance(body, dict):
            raise ValueError("Managed transport requires an object extra_body")
        self._reasoning_policy = copy.deepcopy(body.get("reasoning"))
        if (not isinstance(self._reasoning_policy, dict)
                or set(self._reasoning_policy) != {"effort", "exclude"}
                or self._reasoning_policy.get("exclude") is not False
                or self._reasoning_policy.get("effort") not in {"low", "medium", "high", "xhigh", "max"}):
            raise ValueError("Managed transport needs an explicit preserved native reasoning effort")
        if not config.managed_transport or not config.dynamic_output_cap or not config.context_management:
            raise ValueError("Managed transport requires its complete reviewed context/time policy")
        if config.set_cache_control is not None:
            raise ValueError("Managed transport does not admit unqualified explicit cache markers")
        if len(_encoded([BASH_TOOL])) != TOOL_SCHEMA_UTF8_BYTES or hashlib.sha256(_encoded([BASH_TOOL])).hexdigest() != TOOL_SCHEMA_SHA256:
            raise ValueError("Managed transport requires the pinned Mini bash tool")
        maximum = self._policy["max_output_tokens"]
        trigger = self._policy["context_window_tokens"] - maximum - SAFETY_TOKENS
        if config.context_trigger_tokens is None:
            config.context_trigger_tokens = trigger
        if not config.context_min_reply_tokens <= config.practical_reply_tokens <= maximum:
            raise ValueError("Managed reply allowance exceeds model bounds")
        if config.context_target_bytes >= config.context_trigger_tokens or config.context_trigger_tokens > trigger:
            raise ValueError("Managed context must compact before the maximum-output safety reserve")
        self._context_anchor = None
        self._context_output = None
        self._known_assistant_bounds = {}
        self._context_source_anchor = None
        self._context_cut = None
        self._context_epoch = 0
        self._context_requests, self._context_compactions, self._transport_attempts = [], [], []
        self._deadline_monotonic = None
        self._api_budget_state = None

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return prepare_managed_messages(messages, set_cache_control=self.config.set_cache_control)

    def _remaining_seconds(self):
        remaining, self._deadline_monotonic = trusted_remaining_seconds(self._deadline_monotonic)
        return remaining

    def _estimate_context(self, messages: list[dict]) -> dict:
        fingerprints = message_hashes(messages)
        completed_bound = None
        if self._context_anchor is not None and fingerprints[:len(self._context_anchor[0])] == self._context_anchor[0]:
            base = self._context_anchor[1]
            appended = messages[len(self._context_anchor[0]):]
            growth = conservative_input_bound(appended, self._known_assistant_bounds)
            method = "provider_usage_plus_appended_bytes"
            if appended and self._context_output is not None and fingerprints[len(self._context_anchor[0])] == self._context_output[0]:
                completed_bound = min(_input_bound(appended[:1]), self._context_output[1])
                growth = completed_bound + conservative_input_bound(appended[1:], self._known_assistant_bounds)
                method = "provider_usage_plus_exact_returned_completion_and_appended_bytes"
        else:
            base = None
            growth = conservative_input_bound(messages, self._known_assistant_bounds) + TOOL_SCHEMA_UTF8_BYTES
            method = ("verified_returned_messages_and_full_request_bytes"
                      if growth != _input_bound(messages) + TOOL_SCHEMA_UTF8_BYTES else "full_request_bytes")
        return {"fingerprints": fingerprints, "input_budget_tokens": (base or 0) + growth,
                "anchor_prompt_tokens": base, "new_input_byte_bound": growth, "estimator": method,
                "completion_budget_tokens": completed_bound}

    def _manage_context(self, messages: list[dict], ceiling: int):
        source_fingerprints = message_hashes(messages)
        changed = self._context_source_anchor is not None and source_fingerprints[:len(self._context_source_anchor)] != self._context_source_anchor
        if changed:
            self._context_cut, self._context_anchor, self._context_output = None, None, None
        prefix, turns = _context_turns(messages)
        outgoing, elisions = _context_transport_window(messages, prefix, self._context_cut,
                    self.config.context_tool_output_bytes, self.config.context_memory_bytes)
        estimate = self._estimate_context(outgoing)["input_budget_tokens"]
        minimum = min(ceiling, self.config.context_min_reply_tokens)
        compacted = False
        if estimate >= self.config.context_trigger_tokens or self._policy["context_window_tokens"] - estimate - SAFETY_TOKENS < minimum:
            best = None
            hard_bound = min(self._policy["context_window_tokens"] - SAFETY_TOKENS - minimum,
                             self._policy.get("max_prompt_tokens") or self._policy["context_window_tokens"])
            for begin, _end in reversed(turns):
                if begin <= prefix:
                    break
                candidate, candidate_elisions = _context_transport_window(messages, prefix, begin,
                    self.config.context_tool_output_bytes, self.config.context_memory_bytes)
                bound = conservative_input_bound(candidate, self._known_assistant_bounds) + TOOL_SCHEMA_UTF8_BYTES
                if best is None:
                    if bound > hard_bound:
                        break
                    best = (begin, candidate, candidate_elisions, bound)
                    if bound > self.config.context_target_bytes:
                        break
                elif bound > self.config.context_target_bytes:
                    break
                else:
                    best = (begin, candidate, candidate_elisions, bound)
            if best is not None and (self._context_cut is None or best[0] > self._context_cut):
                self._context_cut, outgoing, elisions, bound = best
                self._context_anchor = None
                self._context_output = None
                self._context_epoch += 1
                compacted = True
                self._context_compactions.append({"epoch": self._context_epoch, "source_message_count": len(messages),
                    "policy": CONTEXT_HISTORY_POLICY,
                    "source_sha256": hashlib.sha256(_encoded(messages)).hexdigest(), "prefix_count": prefix,
                    "window_start": self._context_cut, "effective_sha256": hashlib.sha256(_encoded(outgoing)).hexdigest(),
                    "omitted_prepared_range": [prefix, self._context_cut],
                    "omitted_sha256": hashlib.sha256(_encoded(messages[prefix:self._context_cut])).hexdigest(),
                    "memory_sha256": hashlib.sha256(_encoded(outgoing[prefix])).hexdigest(),
                    "effective_message_count": len(outgoing),
                    "trigger_input_budget_tokens": estimate,
                    "target_exceeded_for_atomic_latest_turn": bound > self.config.context_target_bytes})
        self._context_source_anchor = source_fingerprints
        proof = {"context_epoch": self._context_epoch, "source_history_changed": changed, "compacted_this_request": compacted,
                 "source_message_count": len(messages), "source_sha256": hashlib.sha256(_encoded(messages)).hexdigest(),
                 "effective_message_count": len(outgoing), "effective_sha256": hashlib.sha256(_encoded(outgoing)).hexdigest(),
                 "effective_message_sha256": list(message_hashes(outgoing)), "prefix_count": prefix,
                 "window_start": self._context_cut, "tool_observation_elisions": elisions}
        return outgoing, proof

    def _diagnostic_output_cap(self, messages, cap):
        """Fit a call inside an optional prospective diagnostic spend cap.

        The declared model/output ceiling stays unchanged; only this call's
        actual requested reply allowance is reduced before reservation.
        """
        if self.config.api_cost_budget_usd is None:
            return cap
        prices = (self.config.api_input_price_per_million_usd, self.config.api_output_price_per_million_usd)
        expected = (self._policy["input_price_per_million_usd"], self._policy["output_price_per_million_usd"])
        if any(value is None or value < minimum for value, minimum in zip(prices, expected)):
            raise ValueError("Diagnostic reservation needs the selected model's conservative price ceilings")
        limit, input_price, output_price = (Decimal(str(value)) for value in (self.config.api_cost_budget_usd, *prices))
        state = self._api_budget_state
        if state is not None and state["signature"] != (limit, input_price, output_price):
            raise ValueError("Managed diagnostic budget/price settings changed")
        used = state["reported"] + state["unknown"] if state is not None else Decimal(0)
        bound = _input_bound(messages) + TOOL_SCHEMA_UTF8_BYTES
        remaining = limit - used - Decimal(bound) * input_price / Decimal(1000000)
        affordable = int(remaining * Decimal(1000000) / output_price) if remaining > 0 else 0
        return max(0, min(cap, affordable))

    def _reserve(self, messages, parameters, event):
        if self.config.api_cost_budget_usd is None:
            return None
        prices = (self.config.api_input_price_per_million_usd, self.config.api_output_price_per_million_usd)
        expected = (self._policy["input_price_per_million_usd"], self._policy["output_price_per_million_usd"])
        if any(value is None or value < minimum for value, minimum in zip(prices, expected)):
            raise ValueError("Diagnostic reservation needs the selected model's conservative price ceilings")
        signature = tuple(Decimal(str(value)) for value in (self.config.api_cost_budget_usd, *prices))
        if self._api_budget_state is None:
            self._api_budget_state = {"signature": signature, "reported": Decimal(0), "unknown": Decimal(0), "events": []}
        state = self._api_budget_state
        if state["signature"] != signature:
            raise ValueError("Managed diagnostic budget/price settings changed")
        bound = _input_bound(messages) + TOOL_SCHEMA_UTF8_BYTES
        quote = (Decimal(bound) * signature[1] + Decimal(parameters["max_tokens"]) * signature[2]) / Decimal(1000000)
        record = {"request": event["request"], "attempt": event["attempt"], "input_token_upper_bound": bound,
                  "output_token_limit": parameters["max_tokens"], "reservation_usd": quote,
                  "provider_reported_charge_usd": None, "accounted_usd": Decimal(0), "status": "blocked"}
        state["events"].append(record)
        event["api_budget_reservation_usd"] = float(quote)
        if state["reported"] + state["unknown"] + quote > signature[0]:
            event.update(not_sent=True, provider_cost_usd=0.0, provider_billing_unknown=False, api_budget_status="blocked")
            raise ManagedAPIBudgetError("Cannot reserve the next diagnostic model call")
        state["unknown"] += quote
        record.update(accounted_usd=quote, status="reserved")
        event.update(api_budget_status="reserved", api_budget_accounted_usd=float(quote))
        return record

    def _settle(self, record, event):
        if record is None:
            return
        state = self._api_budget_state
        if record["status"] != "reserved":
            raise ValueError("Diagnostic model reservation was settled more than once")
        quote = record["reservation_usd"]
        state["unknown"] -= quote
        cost = event.get("provider_cost_usd")
        if event.get("provider_billing_unknown") is False and _number(cost):
            accounted = Decimal(str(cost))
            state["reported"] += accounted
            record.update(status="settled_known", provider_reported_charge_usd=accounted)
        else:
            upstream = event.get("provider_upstream_inference_cost_usd")
            accounted = max(quote, Decimal(str(upstream))) if _number(upstream) else quote
            state["unknown"] += accounted
            record["status"] = "retained_unknown"
        record["accounted_usd"] = accounted
        event.update(api_budget_status=record["status"], api_budget_accounted_usd=float(accounted))
        if state["reported"] + state["unknown"] > state["signature"][0]:
            event["api_budget_exceeded"] = True
            raise ManagedAPIBudgetError("Returned diagnostic charge exceeded its reserved budget")

    def _query(self, messages: list[dict], **kwargs):
        parameters = copy.deepcopy(self.config.model_kwargs | kwargs)
        if (os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") != "1"
                or type(parameters.get("num_retries")) is not int or parameters["num_retries"] != 0):
            raise ValueError("Managed Mini requires one model attempt and zero SDK retries")
        if (parameters.get("stream") is not True
                or any(parameters.get(key) is not None for key in ("max_completion_tokens", "max_output_tokens"))):
            raise ValueError("Managed Mini requires explicit streaming and unambiguous max_tokens")
        if any(key in parameters for key in ("model", "messages", "tools", "functions", "function_call", "best_of")):
            raise ValueError("Managed request cannot override its model, messages, tools or generation count")
        if type(parameters.get("n", 1)) is not int or parameters.get("n", 1) != 1:
            raise ValueError("Managed request requires exactly one generated choice")
        if parameters.get("service_tier") not in (None, "default"):
            raise ValueError("Managed transport requires the preset's standard service tier")
        timeout = parameters.get("timeout")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Managed Mini needs an explicit positive network inactivity timeout")
        if self._policy["provider"] == "anthropic" and any(parameters.get(key) is not None for key in ("temperature", "top_p", "top_k")):
            raise ValueError("Sonnet transport does not admit sampling overrides")
        if parameters.get("tool_choice") not in (None, "auto", "none"):
            raise ValueError("Managed transport does not admit forced tool choice")
        extra = parameters.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("Managed transport requires an object extra_body")
        if set(extra) - {"reasoning", "provider"}:
            # LiteLLM merges extra_body after ordinary parameters. Permitting
            # alternate model/stream/token fields here could bypass the exact
            # route, dynamic context cap or prospective spend reservation.
            raise ValueError("Managed extra_body contains unreviewed request overrides")
        if extra.get("reasoning") != self._reasoning_policy:
            raise ValueError("Managed native reasoning settings changed during a run")
        if any(parameters.get(key) is not None or extra.get(key) is not None for key in ("models", "fallbacks")):
            raise ValueError("Managed transport never admits model fallbacks")
        provider = extra.get("provider")
        if provider is not None and provider != {"only": [self._policy["provider"]], "allow_fallbacks": False, "require_parameters": True}:
            raise ValueError("Managed provider policy differs from the exact preset")
        ceiling = parameters.get("max_tokens")
        if type(ceiling) is not int or not 0 < ceiling <= self._policy["max_output_tokens"]:
            raise ValueError("Managed output ceiling exceeds the selected model limit")
        ceiling = min(ceiling, self.config.practical_reply_tokens)
        if ceiling < self.config.context_min_reply_tokens:
            raise ValueError("Managed reply ceiling is smaller than its useful minimum")
        messages, proof = self._manage_context(messages, ceiling)
        budget = self._estimate_context(messages)
        fingerprints = budget.pop("fingerprints")
        cap = min(ceiling, self._policy["context_window_tokens"] - budget["input_budget_tokens"] - SAFETY_TOKENS)
        context_cap = cap
        cap = self._diagnostic_output_cap(messages, cap)
        context = {"request": len(self._context_requests) + 1, "status": "started", **budget, **proof,
                   "requested_max_tokens": max(0, cap), "output_ceiling": ceiling,
                   "context_max_tokens": max(0, context_cap), "api_budget_reduced_output": cap < context_cap,
                   "context_window_tokens": self._policy["context_window_tokens"], "safety_tokens": SAFETY_TOKENS}
        self._context_requests.append(context)
        if context_cap >= self.config.context_min_reply_tokens and cap < self.config.context_min_reply_tokens:
            context["status"] = "blocked_api_budget"
            raise ManagedAPIBudgetError("Diagnostic spend cap cannot reserve a useful reply; no model request sent")
        if (cap < self.config.context_min_reply_tokens
                or self._policy.get("max_prompt_tokens") is not None and budget["input_budget_tokens"] > self._policy["max_prompt_tokens"]):
            context["status"] = "blocked_no_context_space"
            raise ValueError("Managed context cannot reserve a useful reply; no model request sent")
        parameters.update(max_tokens=cap, stream_options={"include_usage": True})
        remaining = self._remaining_seconds()
        logical_started = time.monotonic()
        logical_hard = (min(self.config.hard_api_deadline_seconds, remaining)
                        if remaining is not None and math.isfinite(remaining) and remaining > 0 else None)
        logical_initial_remaining = remaining
        logical_deadline = logical_started + logical_hard if logical_hard is not None else None
        request_hash = request_digest(self.config.model_name, messages, parameters)
        raw = response = None
        for attempt in range(1, self.config.transport_retry_limit + 2):
            remaining = self._remaining_seconds()
            event = {"request": context["request"], "attempt": attempt,
                     "request_sha256": request_hash,
                     "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
                     "response_received": False, "provider_cost_usd": None, "provider_billing_unknown": True,
                     "trusted_phase_seconds_remaining": remaining,
                     "hard_api_deadline_seconds": self.config.hard_api_deadline_seconds,
                     "logical_hard_deadline_seconds": logical_hard,
                     "logical_initial_remaining_seconds": logical_initial_remaining,
                     "stream_opened": False, "stream_complete": False, "stream_chunks": 0}
            self._transport_attempts.append(event)
            context["transport_attempt_count"] = attempt
            if remaining is None or not math.isfinite(remaining) or remaining <= 0 or logical_deadline is None:
                context.update(status="failed", exception_type="ValueError")
                event.update(status="failed", not_sent=True, exception_type="ValueError",
                             provider_billing_unknown=False, provider_cost_usd=0.0)
                raise ValueError("Managed API calls need the original trusted learner clock")
            hard = min(remaining, logical_deadline - time.monotonic())
            if hard <= 0:
                context.update(status="failed", exception_type="ManagedElapsedDeadlineError")
                event.update(status="failed", not_sent=True, exception_type="ManagedElapsedDeadlineError",
                             provider_billing_unknown=False, provider_cost_usd=0.0)
                raise ManagedElapsedDeadlineError("Original logical API deadline expired before retry")
            event["hard_deadline_seconds"] = hard
            reservation = None
            started = time.monotonic()
            try:
                # Retry reservations are independent: an unknown first charge
                # remains accounted, and an identical retry cap is never shrunk.
                reservation = self._reserve(messages, parameters, event)
                remaining_now = self._remaining_seconds()
                if remaining_now is None or not math.isfinite(remaining_now) or remaining_now <= 0:
                    event.update(not_sent=True, provider_cost_usd=0.0, provider_billing_unknown=False)
                    raise ValueError("Original learner clock expired before model inference")
                hard = min(remaining_now, logical_deadline - time.monotonic())
                if hard <= 0:
                    event.update(not_sent=True, provider_cost_usd=0.0, provider_billing_unknown=False)
                    raise ManagedElapsedDeadlineError("Original logical API deadline expired before model inference")
                event.update(trusted_phase_seconds_remaining=remaining_now, hard_deadline_seconds=hard)
                with elapsed_api_deadline(hard):
                    stream = super()._query(copy.deepcopy(messages), **copy.deepcopy(parameters))
                    event["stream_opened"] = True
                    raw = collect_stream(stream, event)
                event.update(response_received=True, response_id=raw.get("id"))
                _record_cost(event, raw)
                if str(raw.get("provider", "")).lower() != self._policy["provider"]:
                    raise ManagedResponseError("Response came from a different provider", raw)
                response_model = str(raw.get("model", "")).removeprefix("openrouter/")
                if response_model != self._policy["model"]:
                    raise ManagedResponseError("Response model differs from the exact selected model", raw)
                reported_usage = raw.get("usage") or {}
                prompt_tokens, completed_tokens = (reported_usage.get(key) for key in ("prompt_tokens", "completion_tokens"))
                if (type(prompt_tokens) is not int or type(completed_tokens) is not int
                        or prompt_tokens <= 0 or not 0 < completed_tokens <= cap
                        or prompt_tokens > budget["input_budget_tokens"]
                        or prompt_tokens + completed_tokens > self._policy["context_window_tokens"]
                        or self._policy.get("max_prompt_tokens") is not None and prompt_tokens > self._policy["max_prompt_tokens"]):
                    raise ManagedResponseError("Provider usage exceeds the selected input/output contract", raw)
                response = litellm.ModelResponse(**raw)
                self._settle(reservation, event)
            except Exception as error:
                if isinstance(error, ManagedResponseError) and error.response is not None:
                    raw = error.response
                    event.update(response_received=True, response_id=raw.get("id"), response=copy.deepcopy(raw),
                                 response_sha256=hashlib.sha256(_encoded(raw)).hexdigest())
                    _record_cost(event, raw)
                elif raw is not None:
                    # Retain rejected complete responses without exposing them
                    # to the agent's action parser or treating them as retries.
                    event.update(response_received=True, response_id=raw.get("id"), response=copy.deepcopy(raw),
                                 response_sha256=hashlib.sha256(_encoded(raw)).hexdigest())
                event.update(status="failed", exception_type=type(error).__name__,
                             elapsed_seconds=max(0.0, time.monotonic() - started),
                             logical_elapsed_seconds=max(0.0, time.monotonic() - logical_started),
                             finished_at_utc=datetime.now(timezone.utc).isoformat())
                if reservation is not None and reservation["status"] == "reserved":
                    self._settle(reservation, event)
                evidence = _transient_http_before_stream(error, event, parameters)
                event.update(evidence)
                delay = evidence.get("retry_backoff_seconds")
                left = self._remaining_seconds()
                retry = (attempt <= self.config.transport_retry_limit
                         and evidence.get("retry_classification") == "transient_http_before_stream"
                         and not event.get("not_sent") and _number(delay)
                         and left is not None and math.isfinite(left)
                         and min(left, logical_deadline - time.monotonic()) > delay)
                event["retry_allowed"] = retry
                if not retry:
                    context.update(status="failed", exception_type=type(error).__name__,
                                   logical_elapsed_seconds=max(0.0, time.monotonic() - logical_started))
                    raise
                context["status"] = "retrying_transport"
                # No generation/action has escaped. Respect Retry-After within
                # the same logical and trusted learner clocks, never renewed.
                backoff_started = time.monotonic()
                try:
                    with elapsed_api_deadline(min(left, logical_deadline - time.monotonic())):
                        time.sleep(delay)
                except ManagedElapsedDeadlineError:
                    event["retry_backoff_elapsed_seconds"] = max(0.0, time.monotonic() - backoff_started)
                    context.update(status="failed", exception_type="ManagedElapsedDeadlineError",
                                   logical_elapsed_seconds=max(0.0, time.monotonic() - logical_started))
                    raise
                event["retry_backoff_elapsed_seconds"] = max(0.0, time.monotonic() - backoff_started)
                continue
            event.update(status="succeeded", elapsed_seconds=max(0.0, time.monotonic() - started),
                         logical_elapsed_seconds=max(0.0, time.monotonic() - logical_started),
                         finished_at_utc=datetime.now(timezone.utc).isoformat())
            context["logical_elapsed_seconds"] = event["logical_elapsed_seconds"]
            break
        usage = raw.get("usage") or {}
        observed = usage.get("prompt_tokens")
        context.update(status="succeeded", response_id=raw.get("id"), actual_prompt_tokens=observed,
                       actual_completion_tokens=usage.get("completion_tokens"))
        confirmed = _number(usage.get("cost")) and type(observed) is int and 0 < observed < self._policy["context_window_tokens"]
        context["provider_usage_confirmed"] = confirmed
        self._context_anchor = (fingerprints, observed) if confirmed else None
        self._context_output = None
        if confirmed:
            completed = usage.get("completion_tokens")
            try:
                assistant = self._prepare_messages_for_api([response.choices[0].message.model_dump()])[0]
                bound = returned_assistant_bound(assistant, completed)
            except ValueError:
                context["completion_accounting_fallback"] = "returned_message_preparation_rejected"
            else:
                if bound is not None:
                    digest = message_hashes([assistant])[0]
                    bound = max(bound, self._known_assistant_bounds.get(digest, 0))
                    self._context_output = (digest, bound)
                    self._known_assistant_bounds[digest] = bound
        return response

    def serialize(self) -> dict:
        value = super().serialize()
        transport = {"schema_version": 1, "policy": POLICY, "model_policy": copy.deepcopy(self._policy),
                     "stream_protocol_policy": STREAM_PROTOCOL_POLICY,
                     "retry_limit": self.config.transport_retry_limit, "hard_api_deadline_seconds": self.config.hard_api_deadline_seconds,
                     "attempts": copy.deepcopy(self._transport_attempts)}
        value["info"]["picbench_managed_transport"] = transport
        # Existing secret-free accounting includes any rejected returned response.
        value["info"]["picbench_transport"] = transport | {"schema_version": 2}
        value["info"]["picbench_context_budget"] = {
            "schema_version": 4, "completion_usage_policy": COMPLETION_USAGE_POLICY,
            "context_window_tokens": self._policy["context_window_tokens"],
            "output_ceiling": min(self.config.model_kwargs.get("max_tokens", self.config.practical_reply_tokens), self.config.practical_reply_tokens),
            "safety_tokens": SAFETY_TOKENS, "image_reserve_tokens": IMAGE_RESERVE,
            "history_policy": CONTEXT_HISTORY_POLICY, "native_history_policy": "preserve_all_messages",
            "tool_schema_sha256": TOOL_SCHEMA_SHA256, "tool_schema_utf8_bytes": TOOL_SCHEMA_UTF8_BYTES,
            "context_management": {"trigger_tokens": self.config.context_trigger_tokens, "target_bytes": self.config.context_target_bytes,
                "tool_output_bytes": self.config.context_tool_output_bytes, "memory_bytes": self.config.context_memory_bytes,
                "min_reply_tokens": self.config.context_min_reply_tokens, "practical_reply_tokens": self.config.practical_reply_tokens},
            "compactions": copy.deepcopy(self._context_compactions), "requests": copy.deepcopy(self._context_requests)}
        if self._api_budget_state is not None:
            state = self._api_budget_state
            limit, input_price, output_price = state["signature"]
            value["info"]["picbench_api_budget"] = {"schema_version": 1, "limit_usd": float(limit),
                "input_price_per_million_usd": float(input_price), "output_price_per_million_usd": float(output_price),
                "reported_charge_usd": float(state["reported"]), "unknown_charge_reserve_usd": float(state["unknown"]),
                "accounted_upper_usd": float(state["reported"] + state["unknown"]),
                "events": [{key: float(item) if isinstance(item, Decimal) else item for key, item in row.items()} for row in state["events"]]}
        return value


ManagedLitellmModel = ManagedOpenRouterModel
