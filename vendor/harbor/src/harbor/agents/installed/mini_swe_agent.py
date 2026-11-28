import hashlib
import json
import math
import shlex
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, override

import yaml
from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
)
from harbor.agents.model_connection import ResolvedModelConnection, ModelConnectionSpec
from harbor.agents.options import Cli, InstalledAgentOptions
from harbor.agents.session_id_headers import validate_session_id_headers
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.logger import logger


_DEFAULT_LITELLM_TIMEOUT_SEC = 3600
_MINI_SWE_AGENT_PYTHON_VERSION = "3.12"


def _normalize_content(raw_content: Any) -> str:
    """Normalize message content which may be a string, list of parts, or None."""
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for part in raw_content:
            if isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(raw_content)


def _iso_timestamp(value: Any) -> str | None:
    """Normalize upstream timestamps to ISO 8601, when available."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    return None


def _message_timestamp(message: dict[str, Any]) -> str | None:
    """Return the best upstream timestamp attached to a mini-swe-agent message."""
    for field in ("created_at", "timestamp", "completed_at"):
        timestamp = _iso_timestamp(message.get(field))
        if timestamp:
            return timestamp

    extra = message.get("extra") or {}
    if isinstance(extra, dict):
        return _iso_timestamp(extra.get("timestamp"))
    return None


def _message_response(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("object") == "response":
        return message
    extra = message.get("extra")
    response = extra.get("response") if isinstance(extra, dict) else None
    return response if isinstance(response, dict) else {}


def _messages_with_transport_responses(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Account for returned provider errors retried before Mini sees a response.

    Schema 2 keeps the original response privately in its attempt evidence.
    Make a derived non-action message for normalization; never modify native
    messages or serialize an error reply as an executed tool call.
    """
    messages = list(trajectory.get("messages") or [])
    transport = (trajectory.get("info") or {}).get("picbench_transport") or {}
    if transport.get("schema_version") != 2:
        return messages
    seen = {_message_response(message).get("id") for message in messages if isinstance(message, dict)}
    for attempt in transport.get("attempts") or []:
        if not isinstance(attempt, dict) or attempt.get("status") != "failed" or attempt.get("response_received") is not True:
            continue
        response = attempt.get("response")
        identifier = attempt.get("response_id")
        if not isinstance(response, dict) or not isinstance(identifier, str) or not identifier or response.get("id") != identifier:
            continue
        encoded = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != attempt.get("response_sha256"):
            continue
        if identifier in seen:
            continue
        seen.add(identifier)
        timestamp = _iso_timestamp(attempt.get("finished_at_utc"))
        derived = {
            "role": "user",
            "content": "Provider error response rejected before tool parsing; no action executed.",
            "extra": {"response": response, "timestamp": timestamp,
                      "picbench_transport_response": True},
        }
        position = next((index for index, message in enumerate(messages)
                         if index > 1 and timestamp and (_message_timestamp(message) or "") >= timestamp),
                        len(messages) - 1 if messages and messages[-1].get("role") == "exit" else len(messages))
        messages.insert(position, derived)
    return messages


def _transport_response_coverage(trajectory: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    transport = (trajectory.get("info") or {}).get("picbench_transport") or {}
    if transport.get("schema_version") != 2:
        return None
    attempts = transport.get("attempts") or []
    returned = [attempt.get("response_id") for attempt in attempts
                if isinstance(attempt, dict) and attempt.get("response_received") is True]
    observed = [_message_response(message).get("id") for message in messages]
    observed_by_id = {_message_response(message).get("id"): _message_response(message) for message in messages}
    payloads_valid = True
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("status") != "failed" or attempt.get("response_received") is not True:
            continue
        response = observed_by_id.get(attempt.get("response_id"))
        encoded = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payloads_valid &= isinstance(response, dict) and hashlib.sha256(encoded).hexdigest() == attempt.get("response_sha256")
    identifiers_valid = all(isinstance(identifier, str) and identifier for identifier in returned + observed)
    complete = bool(payloads_valid and identifiers_valid and len(returned) == len(set(returned))
                    and len(observed) == len(set(observed)) and set(returned) == set(observed))
    return {"response_count_basis": "all_transport_returned_response_ids",
            "transport_returned_response_count": len(returned),
            "transport_response_coverage_complete": complete,
            "response_count_complete": complete}


def _model_messages(trajectory: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """One record per API response, including rejected responses on user errors.

    Upstream saves a FormatError's raw model response on its corrective *user*
    message. Counting assistant roles alone loses both charges and failures.
    Repeated references to an identical response ID are counted only once.
    """
    records = []
    seen: set[str] = set()
    for index, message in enumerate(_messages_with_transport_responses(trajectory)):
        if not isinstance(message, dict):
            continue
        response = _message_response(message)
        if not response and message.get("role") != "assistant":
            continue
        identifier = response.get("id")
        if isinstance(identifier, str) and identifier:
            if identifier in seen:
                continue
            seen.add(identifier)
        records.append((index, message))
    return records


def _response_accounting(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Secret/reasoning-free health counters from every retained API response."""
    records = _model_messages(trajectory)
    rejected = limited = provider_errors = successful = 0
    finish_reasons: dict[str, int] = {}
    for _, message in records:
        response = _message_response(message)
        successful += message.get("role") == "assistant"
        rejected += message.get("role") != "assistant"
        reasons = set()
        for choice in response.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            for key in ("finish_reason", "native_finish_reason"):
                reason = choice.get(key)
                if isinstance(reason, str):
                    reasons.add(reason)
            fields = choice.get("provider_specific_fields") or {}
            native = (
                fields.get("native_finish_reason") if isinstance(fields, dict) else None
            )
            if isinstance(native, str):
                reasons.add(native)
        for reason in reasons:
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        limited += bool(reasons & {"length", "max_tokens", "max_output_tokens"})
        provider_errors += "error" in reasons or bool(response.get("error"))
    info = trajectory.get("info") or {}
    calls = (info.get("model_stats") or {}).get("api_calls")
    result = {
        "schema_version": 1,
        "recorded_api_responses": len(records),
        "reported_api_calls": calls,
        "successful_model_responses": successful,
        "rejected_model_responses": rejected,
        "token_limit_responses": limited,
        "provider_error_responses": provider_errors,
        "finish_reason_counts": finish_reasons,
        "native_exit_status": info.get("exit_status"),
        "response_count_complete": type(calls) is int and calls == len(records),
    }
    transport = info.get("picbench_transport")
    if isinstance(transport, dict) and isinstance(transport.get("attempts"), list):
        attempts = transport["attempts"]
        result.update(transport_attempts=len(attempts),
                      transport_retries=sum(a.get("attempt", 1) > 1 for a in attempts),
                      transport_failed_attempts=sum(a.get("status") == "failed" for a in attempts),
                      transport_billing_unknown_attempts=sum(a.get("provider_billing_unknown") is True for a in attempts))
    coverage = _transport_response_coverage(trajectory, [message for _, message in records])
    if coverage is not None:
        result.update(coverage)
        result["logical_model_calls_reported"] = calls
        result["returned_response_errors_before_parsing"] = sum(
            attempt.get("status") == "failed" and attempt.get("response_received") is True
            for attempt in transport.get("attempts", [])
        )
    return result


def native_exit_error(
    trajectory: dict[str, Any], *, deadline_reached: bool = False
) -> str | None:
    """The stock CLI can return zero even after an unsuccessful native exit."""
    status = (trajectory.get("info") or {}).get("exit_status")
    if status == "Submitted" or (deadline_reached and status in (None, "")):
        return None
    # Do not echo arbitrary provider/error text, task contents or reasoning.
    label = (
        status
        if isinstance(status, str) and status.isidentifier() and len(status) < 80
        else "missing/invalid"
    )
    return f"mini-SWE native exit {label}; not a healthy learner completion"


def _message_usage(message: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized usage dict for a mini-swe-agent message.

    Handles both shapes mini-swe-agent emits depending on model class:
    - ``LitellmModel`` (chat-completions): usage under
      ``extra.response.usage`` with ``prompt_tokens`` / ``completion_tokens``.
    - ``LitellmResponseModel`` (Responses API): the message *is* the response
      object — usage at top level with ``input_tokens`` / ``output_tokens``.

    Always returns chat-completions field names so callers stay simple.
    """
    usage = _message_response(message).get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0,
        "prompt_tokens_details": usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {},
        "completion_tokens_details": usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {},
    }


def _nonnegative_cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _message_cost(message: dict[str, Any]) -> float | None:
    """Read the provider charge, including an explicitly reported free call."""
    response = _message_response(message)
    return _nonnegative_cost((response.get("usage") or {}).get("cost"))


def _message_upstream_cost(message: dict[str, Any]) -> float | None:
    usage = _message_response(message).get("usage") or {}
    details = usage.get("cost_details") or {}
    return _nonnegative_cost(details.get("upstream_inference_cost")) if isinstance(details, dict) else None


def _trajectory_cost(trajectory: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """Prefer complete provider charges; never turn missing pricing into free usage."""
    messages = [m for _, m in _model_messages(trajectory)]
    costs = [_message_cost(m) for m in messages]
    known = [cost for cost in costs if cost is not None]
    upstream = [_message_upstream_cost(message) for message in messages]
    conflicts = sum(cost == 0 and value is not None and value > 0 for cost, value in zip(costs, upstream))
    stats = (trajectory.get("info") or {}).get("model_stats") or {}
    calls = stats.get("api_calls")
    metadata = {
        "schema_version": 2,
        "recorded_model_turns": len(messages),
        "reported_api_calls": calls,
        "provider_cost_calls": len(known),
    }
    if any(value is not None for value in upstream):
        metadata.update(
            reported_zero_cost_responses=sum(cost == 0 for cost in costs),
            provider_upstream_inference_cost_observed_usd=sum(value for value in upstream if value is not None),
            provider_cost_conflict_calls=conflicts,
            upstream_cost_scope="provider_reported_upstream_cost_not_confirmed_customer_charge",
        )
    transport = (trajectory.get("info") or {}).get("picbench_transport") or {}
    coverage = _transport_response_coverage(trajectory, messages)
    if coverage is not None:
        metadata.update(coverage)
    unknown = sum(a.get("provider_billing_unknown") is True for a in transport.get("attempts", []))
    if unknown:
        return None, {**metadata, "status": "unavailable", "reason": "transport_attempt_cost_unknown",
                      "transport_billing_unknown_attempts": unknown, "observed_cost_usd": sum(known)}
    if conflicts:
        return None, {**metadata, "status": "unavailable", "reason": "provider_cost_conflict",
                      "observed_cost_usd": sum(known)}
    if coverage is not None and not coverage["transport_response_coverage_complete"]:
        return None, {**metadata, "status": "unavailable", "reason": "incomplete_transport_response_evidence",
                      "observed_cost_usd": sum(known)}
    if (
        known
        and len(known) == len(messages)
        and (calls is None or calls == len(messages) or coverage is not None and coverage["transport_response_coverage_complete"])
    ):
        return sum(known), {
            **metadata,
            "status": "reported",
            "source": "response.usage.cost",
        }
    if known:
        return None, {
            **metadata,
            "status": "unavailable",
            "reason": "incomplete_provider_costs",
            "observed_cost_usd": sum(known),
        }
    library_cost = _nonnegative_cost(stats.get("instance_cost"))
    if library_cost is not None and library_cost > 0:
        return library_cost, {
            **metadata,
            "status": "estimated",
            "source": "mini_litellm.instance_cost",
        }
    return None, {
        **metadata,
        "status": "unavailable",
        "reason": "provider_cost_and_pricing_unavailable",
    }


def _add_observation_to_last_agent_step(
    steps: list[Step],
    content: str,
    _logger: Any,
    message_index: int,
    timestamp: str | None = None,
    source_call_id: str | None = None,
) -> None:
    """Add observation content to the most recent agent step. ``source_call_id``
    correlates the result with its originating ``ToolCall`` (needed for parallel
    calls)."""
    if steps and steps[-1].source == "agent":
        prev_step = steps[-1]
        if timestamp and prev_step.timestamp is None:
            prev_step.timestamp = timestamp
        result = ObservationResult(content=content, source_call_id=source_call_id)
        if prev_step.observation and prev_step.observation.results:
            prev_step.observation.results.append(result)
        else:
            prev_step.observation = Observation(results=[result])
    else:
        _logger.debug(f"Message at index {message_index} has no preceding agent step")


def _build_step_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    prompt_tokens_details: dict[str, Any],
    completion_tokens_details: dict[str, Any],
    cost_usd: float | None,
) -> Metrics | None:
    """Build metrics for an individual step."""
    if prompt_tokens == 0 and completion_tokens == 0 and cost_usd is None:
        return None

    extra_metrics: dict[str, Any] = {}
    if prompt_tokens_details:
        extra_metrics["prompt_tokens_details"] = prompt_tokens_details
    if completion_tokens_details:
        extra_metrics["completion_tokens_details"] = completion_tokens_details

    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens if cached_tokens > 0 else None,
        cost_usd=cost_usd,
        extra=extra_metrics if extra_metrics else None,
    )


def _normalize_tool_args(raw_args: Any) -> Any:
    """Parse a tool call's ``arguments`` (JSON string or dict), wrapping any
    unparseable value under ``command``. Shared by both wire-format parsers."""
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return {"command": raw_args}
    if isinstance(raw_args, dict):
        return raw_args
    return {"command": str(raw_args)}


def _parse_tool_calls(
    message: dict[str, Any], content: str, step_id: int
) -> tuple[list[ToolCall] | None, str | None]:
    """Parse tool calls from an assistant message into ATIF ToolCall objects."""
    message_tool_calls = message.get("tool_calls")
    if not message_tool_calls:
        return None, content if content else None

    tool_calls: list[ToolCall] = []
    for tc in message_tool_calls:
        tc_id = tc.get("id", f"call_{step_id}_{len(tool_calls) + 1}")
        func = tc.get("function") or {}
        func_name = func.get("name", "bash")
        tool_calls.append(
            ToolCall(
                tool_call_id=tc_id,
                function_name=func_name,
                arguments=_normalize_tool_args(func.get("arguments", "{}")),
            )
        )

    # In tool-calling mode, the content is typically reasoning/thinking
    reasoning = content if content else None
    return tool_calls if tool_calls else None, reasoning


def _parse_response_output(
    message: dict[str, Any], step_id: int
) -> tuple[str, str | None, list[ToolCall] | None]:
    """Responses-API analogue of ``_parse_tool_calls``: a ``LitellmResponseModel``
    turn is the raw response object (no ``role``) with text/reasoning/tool calls in
    ``output``. Returns ``(message_text, reasoning_content, tool_calls)``.
    """
    message_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in message.get("output") or []:
        if not isinstance(item, dict):
            continue  # tolerate a malformed item rather than drop the trajectory
        item_type = item.get("type")
        if item_type == "message":
            text = _normalize_content(item.get("content"))
            if text:
                message_parts.append(text)
        elif item_type == "reasoning":
            # GPT-5.x reasoning text is usually encrypted (empty summary/content).
            for key in ("summary", "content"):
                value = item.get(key)
                if not isinstance(value, list):
                    continue
                for part in value:
                    text = part.get("text") if isinstance(part, dict) else None
                    if text:
                        reasoning_parts.append(text)
        elif item_type == "function_call":
            tool_calls.append(
                ToolCall(
                    tool_call_id=item.get("call_id")
                    or item.get("id")
                    or f"call_{step_id}_{len(tool_calls) + 1}",
                    function_name=item.get("name", "bash"),
                    arguments=_normalize_tool_args(item.get("arguments", "{}")),
                )
            )
    return (
        "\n".join(message_parts),
        "\n".join(reasoning_parts) or None,
        tool_calls or None,
    )


def convert_mini_swe_agent_to_atif(
    mini_swe_agent_trajectory: dict[str, Any],
    session_id: str,
) -> Trajectory:
    """
    Convert mini-swe-agent v2 trajectory format to ATIF format.

    Handles both shapes mini-swe-agent emits:
    - Chat-completions (``LitellmModel``): assistant messages carry ``tool_calls``
      and tool results use ``role: "tool"``.
    - Responses API (``LitellmResponseModel``, used for reasoning models like
      GPT-5.x): each turn is an ``object: "response"`` object (no ``role``) with
      text/reasoning/tool calls in ``output`` and ``function_call_output`` results.

    Args:
        mini_swe_agent_trajectory: The mini-swe-agent trajectory data
        session_id: The session ID for the ATIF trajectory

    Returns:
        Trajectory: The converted ATIF trajectory
    """
    _logger = logger.getChild(__name__)

    # Extract metadata
    info = mini_swe_agent_trajectory.get("info") or {}
    config = info.get("config") or {}
    model_config = config.get("model") or {}
    agent_config = config.get("agent") or {}

    model_name = model_config.get("model_name") or "unknown"
    mini_version = info.get("mini_version") or "unknown"
    trajectory_format = mini_swe_agent_trajectory.get("trajectory_format", "unknown")

    messages = _messages_with_transport_responses(mini_swe_agent_trajectory)
    model_records = dict(_model_messages(mini_swe_agent_trajectory))

    steps: list[Step] = []
    step_id = 1

    # Track cumulative token counts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_cost_usd, cost_capture = _trajectory_cost(mini_swe_agent_trajectory)

    # Count completion tokens across model responses.
    for message in model_records.values():
        total_completion_tokens += _message_usage(message)["completion_tokens"]

    # Process messages
    for i, message in enumerate(messages):
        role = message.get("role")
        content = _normalize_content(message.get("content"))
        timestamp = _message_timestamp(message)
        # Responses-API turns have no role; they are the raw response object.
        is_response = message.get("object") == "response"

        # Extract token usage
        usage = _message_usage(message) if i in model_records else _message_usage({})
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        prompt_tokens_details = usage["prompt_tokens_details"]
        completion_tokens_details = usage["completion_tokens_details"]
        cached_tokens = prompt_tokens_details.get("cached_tokens") or 0
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens") or 0

        total_prompt_tokens += prompt_tokens
        total_cached_tokens += cached_tokens
        total_reasoning_tokens += reasoning_tokens

        if role == "system":
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="system",
                    message=content,
                )
            )
            step_id += 1

        elif role == "user":
            if i in model_records:
                # Preserve a rejected model attempt as its own non-action step;
                # never pretend that mini accepted/executed its tool calls.
                metrics = _build_step_metrics(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    prompt_tokens_details,
                    completion_tokens_details,
                    _message_cost(message),
                )
                if metrics is not None:
                    metrics.extra = {
                        **(metrics.extra or {}),
                        "model_response_rejected": True,
                    }
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=timestamp,
                        source="agent",
                        model_name=model_name,
                        message=("Provider error response rejected before tool parsing."
                                 if message.get("extra", {}).get("picbench_transport_response")
                                 else "Model response rejected by mini-SWE (format error)."),
                        metrics=metrics,
                        observation=Observation(
                            results=[ObservationResult(content=content)]
                        ),
                    )
                )
                step_id += 1
            elif i == 1:
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=timestamp,
                        source="user",
                        message=content,
                    )
                )
                step_id += 1
            else:
                _add_observation_to_last_agent_step(
                    steps, content, _logger, i, timestamp
                )

        elif role == "tool":
            _add_observation_to_last_agent_step(steps, content, _logger, i, timestamp)

        elif message.get("type") == "function_call_output":
            # Responses-API tool result: observation in ``output`` (fall back to
            # ``extra.raw_output``); ``call_id`` links it to its ToolCall.
            obs = message.get("output")
            if not isinstance(obs, str):
                obs = (
                    _normalize_content(obs)
                    if obs is not None
                    else ((message.get("extra") or {}).get("raw_output") or "")
                )
            _add_observation_to_last_agent_step(
                steps, obs, _logger, i, timestamp, source_call_id=message.get("call_id")
            )

        elif role == "assistant" or is_response:
            if is_response:
                content, reasoning, tool_calls = _parse_response_output(
                    message, step_id
                )
            else:
                tool_calls, reasoning = _parse_tool_calls(message, content, step_id)

            metrics = _build_step_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                prompt_tokens_details=prompt_tokens_details,
                completion_tokens_details=completion_tokens_details,
                cost_usd=_message_cost(message),
            )

            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model_name,
                    message=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    llm_call_count=1,
                )
            )
            step_id += 1

    # Build final metrics
    final_extra: dict[str, Any] = {
        "picbench_cost_capture": cost_capture,
        "model_turns": cost_capture["recorded_model_turns"],
        "picbench_response_accounting": _response_accounting(mini_swe_agent_trajectory),
    }
    if total_reasoning_tokens > 0:
        final_extra["total_reasoning_tokens"] = total_reasoning_tokens

    final_metrics = FinalMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached_tokens if total_cached_tokens > 0 else None,
        total_cost_usd=total_cost_usd,
        total_steps=len(steps),
        extra=final_extra if final_extra else None,
    )

    agent = Agent(
        name="mini-swe-agent",
        version=mini_version,
        model_name=model_name,
        extra={
            "original_format": trajectory_format,
            "agent_config": agent_config,
        },
    )

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=agent,
        steps=steps,
        final_metrics=final_metrics,
        notes="Converted from mini-swe-agent trajectory format to ATIF",
    )


def convert_and_save_trajectory(
    mini_swe_agent_trajectory_path: Path,
    atif_trajectory_path: Path,
    session_id: str,
) -> None:
    """
    Convert mini-swe-agent trajectory file to ATIF format and save it.

    Args:
        mini_swe_agent_trajectory_path: Path to mini-swe-agent trajectory.json
        atif_trajectory_path: Path to save the ATIF trajectory.json
        session_id: The session ID for the ATIF trajectory
    """
    _logger = logger.getChild(__name__)

    try:
        mini_swe_agent_trajectory = json.loads(
            mini_swe_agent_trajectory_path.read_text()
        )

        atif_trajectory = convert_mini_swe_agent_to_atif(
            mini_swe_agent_trajectory,
            session_id,
        )

        atif_trajectory_path.write_text(
            json.dumps(atif_trajectory.to_json_dict(), indent=2)
        )

        _logger.debug(
            f"Successfully converted trajectory to ATIF format: {atif_trajectory_path}"
        )

    except Exception as e:
        _logger.error(f"Failed to convert trajectory: {e}")
        raise


class MiniSweAgentOptions(InstalledAgentOptions):
    cost_limit: Annotated[str | None, Cli("--cost-limit")] = Field(
        default="0", description="Cost limit."
    )
    reasoning_effort: str | None = Field(
        default=None, description="Model reasoning effort."
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum completion tokens."
    )
    config_file: str | None = Field(
        default=None, description="Path to a mini-swe-agent YAML config."
    )
    litellm_model_registry_file: str | None = Field(
        default=None, description="Path to a LiteLLM model registry JSON file."
    )
    litellm_model_registry: dict[str, Any] | None = Field(
        default=None, description="Inline LiteLLM model registry."
    )
    session_id_headers: list[str] | None = Field(
        default=None, description="HTTP headers that receive the session ID."
    )


class MiniSweAgent(BaseInstalledAgent):
    """
    The Mini SWE Agent uses the mini-swe-agent tool to solve tasks.

    For Modal Server sticky routing, pass
    ``--ak 'session_id_headers=["Modal-Session-ID"]'``.
    """

    capabilities = AgentCapabilities(atif=True, mcp_servers=True)
    MODEL_CONNECTION = ModelConnectionSpec(
        api_key_envs=("MSWEA_API_KEY",),
        base_url_envs=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        passthrough=True,
    )
    DEFAULT_API_TIMEOUT_SECONDS: int = _DEFAULT_LITELLM_TIMEOUT_SEC

    @property
    @override
    def model_connection(self) -> ResolvedModelConnection:
        # mini-swe accepts either OpenAI base-URL alias; export both when set.
        access = super().model_connection
        url = access.configured_base_url
        if url is None or not any(
            key in access.env for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE")
        ):
            return access
        return replace(
            access,
            env={**access.env, "OPENAI_BASE_URL": url, "OPENAI_API_BASE": url},
        )

    options_model = MiniSweAgentOptions
    options: MiniSweAgentOptions

    def __init__(
        self,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        config_file: str | None = None,
        config: dict[str, Any] | None = None,
        litellm_model_registry_file: str | None = None,
        litellm_model_registry: dict[str, Any] | None = None,
        session_id_headers: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        coerced_max_tokens = self._coerce_max_tokens(max_tokens)
        if config is not None and not isinstance(config, dict):
            raise ValueError(
                "Invalid value for 'config': expected a mapping, "
                f"got {config.__class__.__name__}"
            )
        if config is not None and config_file is not None:
            raise ValueError("'config' and 'config_file' are mutually exclusive")
        config_yaml: str | None = None
        if config is not None:
            config_yaml = yaml.safe_dump(config, sort_keys=False)
        elif config_file:
            config_yaml = Path(config_file).read_text()
        # LiteLLM model registry entries (litellm.utils.register_model format),
        # e.g. to mark a model missing from LiteLLM's bundled model map as
        # supporting reasoning_effort so it isn't silently dropped under
        # drop_params. Named for the mini-swe-agent config key it feeds,
        # `model.litellm_model_registry`, and to stay clear of harbor's own
        # dataset/package registries.
        litellm_model_registry_json: str | None = None
        if litellm_model_registry is not None and litellm_model_registry_file:
            raise ValueError(
                "'litellm_model_registry' and 'litellm_model_registry_file' "
                "are mutually exclusive"
            )
        if litellm_model_registry is not None or litellm_model_registry_file:
            source = (
                "litellm_model_registry_file"
                if litellm_model_registry_file
                else "litellm_model_registry"
            )
            # Decode the file on the host so a malformed registry fails here
            # rather than silently no-opping in the container: mini-swe-agent
            # only registers when the path parses as a model map.
            registry = (
                json.loads(Path(litellm_model_registry_file).read_text())
                if litellm_model_registry_file
                else litellm_model_registry
            )
            if not isinstance(registry, dict):
                raise ValueError(
                    f"Invalid value for {source!r}: expected a mapping, "
                    f"got {registry.__class__.__name__}"
                )
            litellm_model_registry_json = json.dumps(registry, indent=2)
        super().__init__(
            *args,
            reasoning_effort=reasoning_effort,
            max_tokens=coerced_max_tokens,
            config_file=config_file,
            litellm_model_registry_file=litellm_model_registry_file,
            litellm_model_registry=litellm_model_registry,
            session_id_headers=session_id_headers,
            **kwargs,
        )
        self._session_id_headers = validate_session_id_headers(
            self.options.session_id_headers
        )
        self._config_yaml = config_yaml
        self._litellm_model_registry_json = litellm_model_registry_json

    @staticmethod
    def _coerce_max_tokens(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Invalid value for 'max_tokens': expected int")
        if value <= 0:
            raise ValueError("Invalid value for 'max_tokens': must be greater than 0")
        return value

    @staticmethod
    @override
    def name() -> str:
        return AgentName.MINI_SWE_AGENT.value

    @override
    def get_version_command(self) -> str | None:
        return (
            'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
            'else export PATH="$HOME/.local/bin:$PATH"; fi; '
            "uv tool list 2>/dev/null | grep mini-swe-agent"
        )

    @override
    def parse_version(self, stdout: str) -> str:
        # Output: "mini-swe-agent v0.1.2"
        import re

        match = re.search(r"(\d+\.\d+\S*)", stdout)
        return match.group(1) if match else stdout.strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(
            environment,
            ("curl", "bash", "build_tools", "git"),
        )
        version_spec = f"=={self._version}" if self._version else ""
        python_version = shlex.quote(_MINI_SWE_AGENT_PYTHON_VERSION)
        # mini's open-ended LiteLLM dependency can resolve to an incompatible
        # MCP import path. Keep the qualified 2.4.6 runtime on the repository's
        # tested transport version; do not alter other explicitly chosen CLIs.
        dependency_spec = (
            "--with litellm==1.93.0" if self._version == "2.4.6" else "--with litellm"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then"
                "  curl -LsSf https://astral.sh/uv/install.sh | sh;"
                " fi && "
                'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then'
                '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc";'
                " fi && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                'export PATH="$HOME/.local/bin:$PATH" && '
                # Keep the agent toolchain independent from a task image's
                # default Python; current dependencies require modern stdlib
                # typing features.
                f"uv python install {python_version} && "
                # litellm's `proxy` extra is the only thing that pulls
                # pyroscope-io, which publishes no musllinux wheel and whose
                # sdist cannot be built standalone (its Cargo.toml points at
                # ../../ffikit and ../../../ crates that the sdist omits). That
                # makes agent setup fail on musl task images (Alpine). The extra
                # was added for orjson/fastapi on the LiteLLM tools path, so
                # request those directly instead of the whole extra.
                f"uv tool install --python {python_version} "
                f"mini-swe-agent{version_spec} "
                f"{dependency_spec} --with orjson --with fastapi && "
                '"$(uv tool dir)/mini-swe-agent/bin/python" -c "import litellm" && '
                "mini-swe-agent --help"
            ),
        )

    @property
    def _mini_swe_agent_trajectory_path(self) -> PurePosixPath:
        """Path where mini-swe-agent writes its own trajectory format."""
        return EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json"

    @property
    def _atif_trajectory_path(self) -> PurePosixPath:
        """Path where we write the ATIF-formatted trajectory."""
        return EnvironmentPaths.agent_dir / "trajectory.json"

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        # Read the mini-swe-agent trajectory
        mini_trajectory_path = self.logs_dir / "mini-swe-agent.trajectory.json"

        if not mini_trajectory_path.exists():
            self.logger.debug(
                f"Mini-swe-agent trajectory file {mini_trajectory_path} does not exist"
            )
            return

        try:
            mini_trajectory = json.loads(mini_trajectory_path.read_text())
        except Exception as e:
            self.logger.debug(f"Failed to load mini-swe-agent trajectory: {e}")
            return

        # Extract token usage from mini-swe-agent format
        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0
        total_cost, cost_capture = _trajectory_cost(mini_trajectory)
        for _, message in _model_messages(mini_trajectory):
            usage = _message_usage(message)
            n_cache_tokens += usage["prompt_tokens_details"].get("cached_tokens") or 0
            n_input_tokens += usage["prompt_tokens"]
            n_output_tokens += usage["completion_tokens"]

        context.n_input_tokens = n_input_tokens
        context.n_output_tokens = n_output_tokens
        context.n_cache_tokens = n_cache_tokens
        context.cost_usd = total_cost
        context.metadata = {
            **(context.metadata or {}),
            "picbench_cost_capture": cost_capture,
            "model_turns": cost_capture["recorded_model_turns"],
            "picbench_response_accounting": _response_accounting(mini_trajectory),
            "exit_status": (mini_trajectory.get("info") or {}).get("exit_status"),
        }

        # Convert mini-swe-agent trajectory to ATIF format
        atif_trajectory_path = self.logs_dir / "trajectory.json"
        session_id = self.session_id or str(uuid.uuid4())
        try:
            convert_and_save_trajectory(
                mini_swe_agent_trajectory_path=mini_trajectory_path,
                atif_trajectory_path=atif_trajectory_path,
                session_id=session_id,
            )
        except Exception as e:
            self.logger.debug(f"Failed to convert trajectory to ATIF format: {e}")

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        openrouter_model = self.model_name is not None and self.model_name.startswith(
            "openrouter/"
        )
        glm_flash = (
            self.model_name is not None
            and self.model_name.split("@", 1)[0] == "openrouter/z-ai/glm-5.3-flash"
        )
        glm_friendli = self.model_name == (
            "openrouter/z-ai/glm-5.3-flash@preset/picbench-glm-5-3-flash-friendli-only"
        )
        gemini_argon = (
            self.model_name is not None
            and self.model_name.split("@", 1)[0] == "openrouter/google/gemini-4-argon"
        )
        profile = yaml.safe_load(self._config_yaml) if self._config_yaml else {}
        profile_model = profile.get("model", {}) if isinstance(profile, dict) else {}
        profile_environment = profile.get("environment", {}) if isinstance(profile, dict) else {}
        managed_transport = isinstance(profile_model, dict) and profile_model.get("managed_transport") is True
        managed_models = {
            "openrouter/openai/gpt-6.1-sol@preset/picbench-gpt-6-1-sol-openai-only",
            "openrouter/anthropic/claude-sonnet-5.5@preset/picbench-sonnet-5-5-anthropic-only",
        }
        if managed_transport and self.model_name not in managed_models:
            raise ValueError("Managed mini transport requires the exact reviewed GPT-6.1 Sol or Sonnet route")
        if (self.model_name is not None
                and self.model_name.split("@", 1)[0] == "openrouter/anthropic/claude-sonnet-5.5"
                and not managed_transport):
            raise ValueError("Sonnet with mini-SWE requires the managed streaming profile")
        guarded_environment = (
            isinstance(profile_environment, dict)
            and profile_environment.get("environment_class")
            == "picbench_guarded_environment.GuardedLocalEnvironment"
        )
        if guarded_environment and not glm_friendli:
            raise ValueError("The guarded environment requires the exact reviewed Friendli route")
        retry_limit = profile_model.get("transport_retry_limit", 0) if isinstance(profile_model, dict) else 0
        if type(retry_limit) is not int or retry_limit not in (0, 1):
            raise ValueError("transport_retry_limit must be integer0 or1")
        if retry_limit and not (gemini_argon or glm_friendli or managed_transport):
            raise ValueError("transport_retry_limit1 requires the reviewed Argon, Friendli or managed transport")
        effort = self.options.reasoning_effort or (
            "max" if glm_flash else "medium" if gemini_argon else None
        )
        if glm_flash and effort not in {"low", "high", "max"}:
            raise ValueError(
                "GLM-5.3-Flash supports only low, high or max reasoning; no medium or off mode"
            )
        if gemini_argon and effort not in {"low", "medium", "high"}:
            raise ValueError(
                "Gemini-4-Argon supports only low, medium or high reasoning"
            )
        if (
            openrouter_model
            and self._extra_env.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1") != "1"
        ):
            # BaseInstalledAgent applies explicit agent-env overrides last.
            # Reject a conflict instead of silently restoring model retries.
            label = (
                "GLM-5.3-Flash"
                if glm_flash
                else "Gemini-4-Argon"
                if gemini_argon
                else "OpenRouter mini-SWE"
            )
            raise ValueError(f"{label} requires MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1")
        if (gemini_argon or glm_friendli or managed_transport) and "PYTHONPATH" in self._extra_env:
            raise ValueError(
                "The reviewed model transport sets its path inside the container; PYTHONPATH agent-env overrides are not allowed"
            )
        augmented_instruction = instruction
        if self.mcp_servers:
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for s in self.mcp_servers:
                if s.transport == "stdio":
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            augmented_instruction = instruction + mcp_info

        escaped_instruction = shlex.quote(augmented_instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        access = self.model_connection
        env = {
            **access.env,
            "MSWEA_CONFIGURED": "true",  # Disable interactive setup
            "MSWEA_COST_TRACKING": "ignore_errors",  # Ignore unknown model costs
        }
        if openrouter_model:
            # Harbor's no-retry policy alone does not disable mini's model retry
            # loop. Keep every OpenRouter route to one provider attempt per call.
            env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"

        if access.api_key is None:
            raise ValueError(
                f"No API key found for model {self.model_name!r}. "
                "Please set MSWEA_API_KEY or the provider's API key variable."
            )

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""

        # LiteLLM's chat-completion fallback is 600 seconds. We set it higher.
        config_flags = (
            f"-c model.model_kwargs.timeout={self.DEFAULT_API_TIMEOUT_SECONDS} "
        )

        # Write custom config into the container if provided
        if self._config_yaml:
            config_path = "/tmp/mswea-config/custom.yaml"
            heredoc_marker = f"MSWEA_CONFIG_EOF_{uuid.uuid4().hex[:8]}"
            write_config_cmd = (
                f"mkdir -p /tmp/mswea-config\n"
                f"cat > '{config_path}' << '{heredoc_marker}'\n"
                f"{self._config_yaml}\n"
                f"{heredoc_marker}\n"
            )
            await self.exec_as_agent(environment, command=write_config_cmd, env=env)
            config_flags += f"-c {config_path} "
        if self._litellm_model_registry_json:
            registry_path = "/tmp/mswea-config/registry.json"
            heredoc_marker = f"MSWEA_REGISTRY_EOF_{uuid.uuid4().hex[:8]}"
            write_registry_cmd = (
                f"mkdir -p /tmp/mswea-config\n"
                f"cat > '{registry_path}' << '{heredoc_marker}'\n"
                f"{self._litellm_model_registry_json}\n"
                f"{heredoc_marker}\n"
            )
            await self.exec_as_agent(environment, command=write_registry_cmd, env=env)
            config_flags += f"-c model.litellm_model_registry={registry_path} "

        if self.session_id:
            header_names = (
                self._session_id_headers
                if self._session_id_headers is not None
                else ["X-Session-ID"]
            )
            for header_name in header_names:
                session_header_config = (
                    f"model.model_kwargs.extra_headers.{header_name}={self.session_id}"
                )
                config_flags += f"-c {shlex.quote(session_header_config)} "

        if gemini_argon:
            transport = (
                Path(__file__).with_name("mini_swe_openrouter_argon.py").read_text()
            )
            marker = f"MSWEA_TRANSPORT_EOF_{uuid.uuid4().hex}"
            await self.exec_as_agent(
                environment,
                command=(
                    "mkdir -p /tmp/mswea-config\n"
                    f"cat > /tmp/mswea-config/picbench_argon_model.py << '{marker}'\n"
                    f"{transport}\n{marker}\n"
                ),
                env=env,
            )
            config_flags += (
                "-c model.model_class=picbench_argon_model.ArgonLitellmModel "
            )

        if glm_friendli:
            transport = Path(__file__).with_name("mini_swe_openrouter_friendli.py").read_text()
            context_helper = Path(__file__).with_name("mini_swe_friendli_context.py").read_text()
            marker = f"MSWEA_TRANSPORT_EOF_{uuid.uuid4().hex}"
            await self.exec_as_agent(
                environment,
                command=(
                    "mkdir -p /tmp/mswea-config\n"
                    f"cat > /tmp/mswea-config/picbench_friendli_model.py << '{marker}'\n"
                    f"{transport}\n{marker}\n"
                    f"cat > /tmp/mswea-config/picbench_friendli_context.py << '{marker}'\n"
                    f"{context_helper}\n{marker}\n"
                ),
                env=env,
            )
            config_flags += "-c model.model_class=picbench_friendli_model.FriendliLitellmModel "
            if guarded_environment:
                guard = Path(__file__).with_name("mini_swe_guarded_environment.py").read_text()
                marker = f"MSWEA_PROCESS_GUARD_EOF_{uuid.uuid4().hex}"
                await self.exec_as_agent(
                    environment,
                    command=(
                        "mkdir -p /tmp/mswea-config\n"
                        f"cat > /tmp/mswea-config/picbench_guarded_environment.py << '{marker}'\n"
                        f"{guard}\n{marker}\n"
                    ),
                    env=env,
                )
                config_flags += (
                    "-c environment.environment_class="
                    "picbench_guarded_environment.GuardedLocalEnvironment "
                )

        if managed_transport:
            modules = {
                "picbench_managed_model.py": "mini_swe_openrouter_managed.py",
                "picbench_managed_policy.py": "mini_swe_managed_policy.py",
                "picbench_friendli_context.py": "mini_swe_friendli_context.py",
            }
            marker = f"MSWEA_MANAGED_EOF_{uuid.uuid4().hex}"
            await self.exec_as_agent(
                environment,
                command="mkdir -p /tmp/mswea-config\n" + "".join(
                    f"cat > /tmp/mswea-config/{target} << '{marker}'\n"
                    f"{Path(__file__).with_name(source).read_text()}\n{marker}\n"
                    for target, source in modules.items()
                ),
                env=env,
            )
            config_flags += "-c model.model_class=picbench_managed_model.ManagedOpenRouterModel "

        if openrouter_model:
            # Disable LiteLLM's inner retries as well as mini's outer retry loop.
            # Apply after the custom file so it cannot re-enable paid attempts.
            config_flags += "-c model.model_kwargs.num_retries=0 "

        if effort:
            eff = shlex.quote(effort)
            if self.model_name.startswith("openai/"):
                # OpenAI gpt-5.x rejects tools+reasoning_effort on
                # /v1/chat/completions ("use /v1/responses instead").
                # mini-swe-agent's LitellmResponseModel routes via
                # litellm.responses(); effort is reasoning.effort there.
                config_flags += (
                    "-c model.model_class=litellm_response "
                    f"-c model.model_kwargs.reasoning.effort={eff} "
                )
            elif self.model_name.startswith("openrouter/"):
                # OpenRouter's native field is reasoning.effort. An extra-body
                # reasoning_effort field can be silently ignored by the gateway.
                config_flags += (
                    f"-c model.model_kwargs.extra_body.reasoning.effort={eff} "
                )
                if gemini_argon:
                    # Keep Google's signed reasoning details on tool continuations;
                    # mini/LiteLLM must forward the returned blocks unchanged.
                    config_flags += (
                        "-c model.model_kwargs.extra_body.reasoning.exclude=false "
                    )
                if glm_flash:
                    # Use OpenRouter's native field: LiteLLM's top-level
                    # reasoning_effort can translate GLM's native max to xhigh.
                    # Retain reasoning for tool-call continuity without claiming
                    # unsupported native Z.ai thinking.clear_thinking forwarding.
                    config_flags += (
                        "-c model.model_kwargs.extra_body.reasoning.exclude=false "
                        "-c model.model_kwargs.temperature=1 "
                        "-c model.model_kwargs.top_p=0.95 "
                    )
                if managed_transport:
                    config_flags += "-c model.model_kwargs.extra_body.reasoning.exclude=false "
            else:
                # LiteLLM maps top-level reasoning_effort to provider-native
                # params (Anthropic output_config/thinking, Gemini
                # thinking_level, etc.). Nesting under extra_body forwards a
                # literal field that providers like Gemini reject.
                config_flags += f"-c model.model_kwargs.reasoning_effort={eff} "

        if self.options.max_tokens is not None:
            token_key = (
                "max_output_tokens"
                if self.model_name.startswith("openai/")
                and self.options.reasoning_effort
                else "max_tokens"
            )
            config_flags += (
                f"-c model.model_kwargs.{token_key}={self.options.max_tokens} "
            )

        # mini-swe-agent v2's -c is a *spec list*, not a merge-over-defaults:
        # any -c suppresses the packaged config/mini.yaml (system_template etc).
        # Prepend the builtin so dotted/file overrides layer on top of it.
        # Bare "mini" resolves via mini-swe's own get_config_path() to
        # builtin_config_dir/mini.yaml regardless of install location.
        config_flags = "-c mini " + config_flags

        # Docker transports env names through its host Compose process. Keep
        # Python's module search path local to this learner command instead.
        model_command_prefix = "PYTHONPATH=/tmp/mswea-config " if gemini_argon or glm_friendli or managed_transport else ""
        task_flag = f"--task={escaped_instruction} "
        if len(task_flag.encode("utf-8")) > 60000:
            # A long qualification prompt can exceed Linux's per-argument
            # limit before mini even starts. Use the pinned CLI's native
            # run.task config instead, preserving the complete instruction.
            task_path = "/tmp/mswea-config/task.yaml"
            await self.exec_as_agent(environment, command="mkdir -p /tmp/mswea-config", env=env)
            with tempfile.TemporaryDirectory(prefix="harbor-mini-task-") as temporary:
                task_file = Path(temporary) / "task.yaml"
                task_file.write_text(json.dumps({"run": {"task": augmented_instruction}}))
                await environment.upload_file(task_file, task_path)
            config_flags += f"-c {task_path} "
            task_flag = ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -o pipefail; "
                'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
                'else export PATH="$HOME/.local/bin:$PATH"; fi; '
                f"{model_command_prefix}mini-swe-agent --yolo --model={self.model_name} {task_flag}"
                f"--output={self._mini_swe_agent_trajectory_path} {extra_flags}"
                f"{config_flags}"
                f"--exit-immediately 2>&1 </dev/null | tee /logs/agent/mini-swe-agent.txt"
            ),
            env=env,
        )
