"""Narrow OpenRouter signed-reasoning transport for pinned mini/LiteLLM.

This is a model transport plugin, not an agent or a task prompt. Mini keeps the
complete assistant response; LiteLLM 1.93.0 nests OpenRouter reasoning blocks in
provider_specific_fields. OpenRouter requires the unchanged list at top level.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpcore
import httpx
import litellm
from pydantic import Field

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


_BENCHMARK_STATUS = Path("/run/picbench/benchmark-status.json")
_RETRY_BACKOFF_SECONDS = 1.0


class ArgonLitellmModelConfig(LitellmModelConfig):
    transport_retry_limit: int = Field(default=0, strict=True, ge=0, le=1)


def _incomplete_body_error(error: Exception) -> bool:
    """Do not turn arbitrary API, policy, authentication or timeout errors into retries."""
    if type(error) not in (litellm.APIError, httpx.RemoteProtocolError, httpcore.RemoteProtocolError):
        return False
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (httpx.RemoteProtocolError, httpcore.RemoteProtocolError)):
            if "incomplete chunked read" in str(current).lower():
                return True
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return False


class ArgonLitellmModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=ArgonLitellmModelConfig, **kwargs)
        if self.config.model_name.split("@", 1)[0] != "openrouter/google/gemini-4-argon":
            raise ValueError("Argon transport supports only the exact selected Gemini model")
        self._transport_attempts: list[dict] = []
        self._transport_request_count = 0
        self._retry_deadline_monotonic: float | None = None

    def _retry_remaining_seconds(self) -> float | None:
        """Read the existing root-owned phase clock; never create or extend one."""
        try:
            if _BENCHMARK_STATUS.is_symlink() or _BENCHMARK_STATUS.stat().st_size > 16384:
                return None
            status = json.loads(_BENCHMARK_STATUS.read_text())
            if status.get("schema_version") != 1 or status.get("source") != "harbor_agent_phase":
                return None
            deadline = datetime.fromisoformat(status["timing"]["deadline_at_utc"].replace("Z", "+00:00"))
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
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _query(self, messages: list[dict], **kwargs):
        limit = self.config.transport_retry_limit
        combined = self.config.model_kwargs | kwargs
        timeout = combined.get("timeout")
        if limit:
            if (os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") != "1"
                    or type(combined.get("num_retries")) is not int or combined["num_retries"] != 0):
                raise ValueError("Argon transport retry requires Mini attempts1 and LiteLLM num_retries0")
            if ((combined.get("stream") is not None and combined.get("stream") is not False) or isinstance(timeout, bool)
                    or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("Argon transport retry requires a bounded nonstreaming request")
        # Freeze only the opted-in call. Each SDK invocation receives a new copy,
        # so message/parameter mutation cannot change the repeated signed payload.
        frozen_messages = copy.deepcopy(messages) if limit else messages
        frozen_kwargs = copy.deepcopy(combined) if limit else kwargs
        fingerprint = hashlib.sha256(json.dumps({
            "model": self.config.model_name, "messages": frozen_messages,
            "parameters": {k: v for k, v in combined.items()
                           if k in {"extra_body", "max_tokens", "timeout", "stream", "num_retries"}},
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._transport_request_count += 1
        for attempt in range(limit + 1):
            event = {"request": self._transport_request_count, "attempt": attempt + 1,
                     "request_sha256": fingerprint, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "status": "started", "response_received": False,
                     "provider_cost_usd": None, "provider_billing_unknown": True}
            self._transport_attempts.append(event)
            try:
                response = super()._query(
                    copy.deepcopy(frozen_messages) if limit else messages,
                    **(copy.deepcopy(frozen_kwargs) if limit else kwargs),
                )
            except Exception as error:
                classified = _incomplete_body_error(error)
                event.update(status="failed", exception_type=type(error).__name__,
                             incomplete_body=classified, provider_billing_unknown=True,
                             finished_at_utc=datetime.now(timezone.utc).isoformat())
                if not classified or attempt == limit:
                    raise
                remaining = self._retry_remaining_seconds()
                if remaining is None or remaining < timeout + _RETRY_BACKOFF_SECONDS:
                    event["retry_skipped"] = "trusted_deadline_unavailable_or_insufficient"
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS)
                remaining = self._retry_remaining_seconds()
                if remaining is None or remaining < timeout:
                    event["retry_skipped"] = "trusted_deadline_unavailable_or_insufficient"
                    raise
                event["retry_scheduled"] = True
            else:
                usage = getattr(response, "usage", None)
                cost = getattr(usage, "cost", None)
                event.update(status="succeeded", response_received=True,
                             finished_at_utc=datetime.now(timezone.utc).isoformat())
                if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                    event["provider_cost_usd"] = cost
                    event["provider_billing_unknown"] = False
                return response

    def serialize(self) -> dict:
        value = super().serialize()
        value["info"]["picbench_transport"] = {
            "schema_version": 1, "retry_limit": self.config.transport_retry_limit,
            "attempts": copy.deepcopy(self._transport_attempts),
        }
        return value

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = copy.deepcopy(super()._prepare_messages_for_api(messages))
        for message in prepared:
            if message.get("role") != "assistant":
                continue
            fields = message.get("provider_specific_fields")
            if not isinstance(fields, dict) or "reasoning_details" not in fields:
                continue
            details = fields["reasoning_details"]
            if not isinstance(details, list) or any(
                not isinstance(item, dict) for item in details
            ):
                raise ValueError(
                    "Invalid signed reasoning details; refusing a lossy continuation"
                )
            if (
                "reasoning_details" in message
                and message["reasoning_details"] != details
            ):
                raise ValueError(
                    "Conflicting signed reasoning details; refusing a lossy continuation"
                )
            message["reasoning_details"] = details
            del fields["reasoning_details"]
            if not fields:
                del message["provider_specific_fields"]
        return prepared
