"""Guard and retain native Claude Messages traffic without converting protocols.

Only loopback clients and the fixed HTTPS OpenRouter gateway are supported.
Headers/credentials are never persisted. Request/response bodies are private
operational evidence; this module never retries an inference request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import select
import signal
import socket
import struct
import tempfile
import threading
import time
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

MAX_BODY = 24 * 1024 * 1024
REJECTION_RECORD_LIMIT = 16
REJECTION_BODY_LIMIT = 64 * 1024 * 1024
RESERVATION_WAIT_LIMIT = 64
MAX_RESPONSE_BODY = 32 * 1024 * 1024
TIMEOUT = 600
HTTP_RETRY_BACKOFF_SECONDS = 2.0
OUTPUT_CAP = 65536
INPUT_PRICE = Decimal("0.000002")
OUTPUT_PRICE = Decimal("0.000010")
MODEL = "moonshotai/kimi-k3.1-pro@preset/picbench-kimi-k3-1-pro-moonshotai-mxfp4-only"
GLM_MODEL = "z-ai/glm-5.3-flash@preset/picbench-glm-5-3-flash-zai-only"
SONNET_MODEL = "anthropic/claude-sonnet-5.5@preset/picbench-sonnet-5-5-anthropic-only"


def model_policy(model: str) -> dict:
    """Return the exact reviewed route, limits and conservative pricing.

    These are native Messages guard settings, not permission to substitute a
    provider/model. The selected preset remains in every forwarded request.
    """
    if model == MODEL:
        return {
            "output_cap": OUTPUT_CAP,
            "efforts": ("low", "medium", "high", "max"),
            "input_price": INPUT_PRICE,
            "output_price": OUTPUT_PRICE,
            "provider_name": "Moonshot AI",
            "provider_slug": "moonshotai/mxfp4",
            "canonical_model": "moonshotai/kimi-k3.1-pro",
            "request_timeout_seconds": 600,
            "request_timeout_semantics": "read_inactivity",
            "connect_timeout_seconds": 600,
            "api_client_timeout_ms": 7200000,
        }
    if model == GLM_MODEL:
        return {
            "output_cap": 131072,
            "efforts": ("low", "high", "max"),
            "input_price": Decimal("0.00000015"),
            "output_price": Decimal("0.0000005"),
            "provider_name": "Z.AI",
            "provider_slug": "z-ai",
            "canonical_model": "z-ai/glm-5.3-flash",
            "request_timeout_seconds": 1800,
            "request_timeout_semantics": "read_inactivity",
            "connect_timeout_seconds": 60,
            "api_client_timeout_ms": 7200000,
        }
    if model == SONNET_MODEL:
        return {
            "output_cap": 128000,
            "efforts": ("low", "medium", "high", "xhigh", "max"),
            "input_price": Decimal("0.000002"),
            "output_price": Decimal("0.000010"),
            "provider_name": "Anthropic",
            "provider_slug": "anthropic",
            "canonical_model": "anthropic/claude-sonnet-5.5",
            "request_timeout_seconds": 600,
            "request_timeout_semantics": "read_inactivity",
            "connect_timeout_seconds": 600,
            "api_client_timeout_ms": 7200000,
        }
    raise ValueError("Native Messages model is outside the reviewed preset allowlist")


def write_json(path: Path, value):
    publish_private(path, (json.dumps(value, ensure_ascii=False) + "\n").encode())


def publish_private(path: Path, body: bytes, *, exclusive=False):
    """Publish complete bytes atomically, staging outside canonical evidence.

    The fixed root's sibling lies on the same filesystem. Temporary files can
    never become unexpected canonical inventory entries, even after a crash.
    """
    canonical = Path("/root/.picbench-claude-native")
    if not path.is_relative_to(canonical):
        canonical = path.parent.parent if path.parent.name == "openrouter-wire" else path.parent
    staging_parent = canonical.parent
    with tempfile.TemporaryDirectory(prefix=".picbench-native-publish-", dir=staging_parent) as temporary:
        staged = Path(temporary) / "complete"
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(staged, path, follow_symlinks=False)
        else:
            os.replace(staged, path)


def write_private_once(path: Path, body: bytes):
    """Retain exact evidence once; a late SDK request must not replace it."""
    publish_private(path, body, exclusive=True)


def validate_tool_protocol(value, model):
    """Require ordinary, fully declared tools on the reviewed native routes.

    Claude's deferred-tool protocol is not supported on these model routes.
    Inspect only native tool definitions and content blocks: a learner may
    legitimately discuss this protocol in text, thinking or a tool's JSON input.
    The request is rejected intact rather than rewritten or retried.
    """
    if model not in {MODEL, GLM_MODEL, SONNET_MODEL}:
        return

    def is_deferred_tool(tool):
        return any(
            isinstance(tool.get(key), str)
            and (
                tool[key] in {
                    "DeferredToolPlaceholder", "ToolSearch", "tool_search",
                    "openrouter:tool_search",
                }
                or tool[key].startswith("tool_search_tool_")
            )
            for key in ("name", "type")
        )

    tools = value.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Native tool definitions must be a list")
    for tool in tools:
        if isinstance(tool, dict) and (
            "defer_loading" in tool or is_deferred_tool(tool)
        ):
            raise ValueError("Native models require upfront tools without deferred loading or tool search")

    def check_content(content):
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in {"tool_reference", "tool_search_tool_result"} or (
                kind in {"tool_use", "server_tool_use"} and is_deferred_tool(block)
            ):
                raise ValueError("Native models require upfront tools without deferred content blocks")
            if kind == "tool_result":
                check_content(block.get("content"))

    for message in value.get("messages", []):
        if isinstance(message, dict):
            check_content(message.get("content"))


def validate_request(value, model, effort):
    policy = model_policy(model)
    if not isinstance(value, dict) or value.get("model") != model:
        raise ValueError("Messages request model differs from the pinned preset")
    if not isinstance(value.get("output_config"), dict) or not isinstance(
        value.get("thinking"), dict
    ):
        raise ValueError("Native reasoning fields must be objects")
    if value["output_config"].get("effort") != effort or effort not in policy["efforts"]:
        raise ValueError("Native request did not retain the selected reasoning effort")
    if value.get("thinking", {}).get("type") != "adaptive":
        raise ValueError("Native adaptive thinking is required")
    if (
        type(value.get("max_tokens")) is not int
        or not 1 <= value["max_tokens"] <= policy["output_cap"]
    ):
        raise ValueError("Native output cap is missing or exceeds the reviewed limit")
    if any(key in value for key in ("provider", "models", "route", "reasoning", "configuration_update")):
        raise ValueError("Request attempted to override the pinned native route")
    if model == SONNET_MODEL and any(key in value for key in ("temperature", "top_p", "top_k", "fallbacks")):
        raise ValueError("Sonnet 5.5 does not permit sampling or fallback overrides")
    if model == SONNET_MODEL and "tool_choice" in value and (
        not isinstance(value["tool_choice"], dict)
        or value["tool_choice"].get("type") not in {"auto", "none"}
    ):
        raise ValueError("Sonnet 5.5 requires automatic or disabled tool choice")
    if not isinstance(value.get("messages"), list) or not value["messages"]:
        raise ValueError("Native conversation is missing")
    validate_tool_protocol(value, model)
    for message in value["messages"]:
        # A recognized Sonnet 5.5 CLI emits native system messages retaining
        # the already selected effort. They are not permission to vary it.
        fixed_sonnet_effort = (
            model == SONNET_MODEL and isinstance(message, dict)
            and set(message) == {"role", "content", "output_config"}
            and message["role"] == "system"
            and message["output_config"] == {"effort": effort}
            and (isinstance(message["content"], str)
                 or isinstance(message["content"], list)
                 and all(isinstance(block, dict) and block.get("type") == "text"
                         and isinstance(block.get("text"), str)
                         and set(block) <= {"type", "text", "cache_control"}
                         for block in message["content"]))
        )
        if (not isinstance(message, dict)
                or message.get("type") == "configuration_update"
                or message.get("role") == "configuration_update"
                or ("output_config" in message and not fixed_sonnet_effort)
                or any(key in message for key in ("thinking", "reasoning", "configuration_update"))):
            raise ValueError("Message-level reasoning overrides are not permitted")
        # Inspect protocol fields only. Thinking blocks and quoted code/JSON in
        # ordinary text, tool inputs and tool results are conversation content.
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "configuration_update"
            for block in content
        ):
            raise ValueError("Native configuration-update blocks are not permitted")


def normalize_request(value, model, effort):
    """Inherit adaptive thinking only for observed, action-bound CLI helpers.

    Keep native tools and the pinned model/effort unchanged. WebSearch uses the
    documented Anthropic Messages alias; OpenRouter may execute its search via
    Exa while model inference remains on the selected, pinned endpoint.
    """
    def text_block(block):
        return (isinstance(block, dict) and set(block) == {"type", "text"}
                and block["type"] == "text" and isinstance(block["text"], str))

    policy = model_policy(model)
    common = (
        effort in policy["efforts"] and isinstance(value, dict)
        and value.get("model") == model
        and type(value.get("max_tokens")) is int
        and value["max_tokens"] == (65536 if model == MODEL else 128000)
        and value.get("output_config") == {"effort": effort}
        and value.get("stream") is True
        and isinstance(value.get("metadata"), dict) and set(value["metadata"]) == {"user_id"}
        and isinstance(value["metadata"]["user_id"], str)
        and isinstance(value.get("system"), list)
        and all(text_block(block) for block in value["system"])
        and isinstance(value.get("messages"), list) and len(value["messages"]) == 1
        and isinstance(value["messages"][0], dict)
        and set(value["messages"][0]) == {"role", "content"}
        and value["messages"][0]["role"] == "user"
        and isinstance(value["messages"][0]["content"], list)
        and len(value["messages"][0]["content"]) == 1
        and text_block(value["messages"][0]["content"][0])
    )
    keys = {"model", "messages", "system", "tools", "metadata", "max_tokens", "output_config", "stream"}
    fetch = common and set(value) == keys and value["tools"] == [] and len(value["system"]) == 2
    search = (
        common and (model, effort) in {(GLM_MODEL, "high"), (MODEL, "medium"), (SONNET_MODEL, "high")}
        and set(value) - {"thinking"} == keys | {"tool_choice"}
        and ("thinking" not in value or value["thinking"] == {"type": "adaptive"})
        and value["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]
        and value["tool_choice"] == {"type": "auto"} and len(value["system"]) == 3
        and value["messages"][0]["content"][0]["text"].startswith("Perform a web search for the query: ")
        and len(value["messages"][0]["content"][0]["text"].removeprefix("Perform a web search for the query: ").strip()) >= 2
    )
    if server_search_tools(value) and not search:
        raise ValueError("Native server WebSearch requires the reviewed action-bound helper envelope")
    inherited = (fetch or search) and "thinking" not in value
    effective = {**value, "thinking": {"type": "adaptive"}} if inherited else value
    if search and model in {MODEL, SONNET_MODEL}:
        # A fresh client WebSearch action gets one server-tool step, followed
        # by the model's final answer. Preserve all eight declared search uses;
        # parent actions can perform further searches with fresh source credit.
        effective = {**effective, "stop_server_tools_when": [{"type": "step_count_is", "step_count": 1}]}
    validate_request(effective, model, effort)
    return effective, "native_websearch" if search else "native_webfetch_summary" if fetch else "native"


def server_search_tools(value):
    """Identify server-search aliases, never ordinary client WebSearch tools."""
    if not isinstance(value, dict):
        return []
    tools = value.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Native tool definitions must be a list")
    return [tool for tool in tools if isinstance(tool, dict)
            and isinstance(tool.get("type"), str)
            and (tool["type"].startswith("web_search") or tool["type"] == "openrouter:web_search")]


def response_events(body: bytes, content_type: str) -> list[dict]:
    if "text/event-stream" not in content_type:
        value = json.loads(body)
        return [value] if isinstance(value, dict) else []
    values = []
    # Anthropic SSE sends one JSON data line per event; ignore event labels.
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            value = json.loads(line[5:].strip())
            if isinstance(value, dict):
                values.append(value)
    return values


def native_helper_actions(events: list[dict], request: dict, number: int, tool_name: str) -> list[dict]:
    """Extract completed native action metadata, without interpreting page text."""
    if tool_name not in {"WebFetch", "WebSearch"}:
        raise ValueError("Native helper action kind is outside the reviewed allowlist")
    blocks, pending = [], {}
    for event in events:
        content = event.get("content") if event.get("type") == "message" else None
        if event.get("type") == "message_start":
            content = event.get("message", {}).get("content")
        if isinstance(content, list):
            blocks.extend(block for block in content if isinstance(block, dict)
                          and block.get("type") == "tool_use" and block.get("name") == tool_name)
        index = event.get("index")
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == tool_name:
                if index in pending:
                    raise ValueError(f"Native {tool_name} action block was repeated")
                pending[index] = (dict(block), [])
        elif event.get("type") == "content_block_delta" and index in pending:
            delta = event.get("delta", {})
            if delta.get("type") == "input_json_delta":
                if not isinstance(delta.get("partial_json"), str):
                    raise ValueError(f"Native {tool_name} action input fragment is invalid")
                pending[index][1].append(delta["partial_json"])
        elif event.get("type") == "content_block_stop" and index in pending:
            block, fragments = pending.pop(index)
            if fragments:
                block["input"] = json.loads("".join(fragments))
            blocks.append(block)
    if pending:
        raise ValueError(f"Native {tool_name} action input block is incomplete")
    if not blocks:
        return []
    definitions = [tool for tool in request.get("tools", []) if isinstance(tool, dict) and tool.get("name") == tool_name]
    if (len(definitions) != 1 or not isinstance(definitions[0].get("input_schema"), dict)
            or definitions[0]["input_schema"].get("type") != "object" or "defer_loading" in definitions[0]):
        raise ValueError(f"Native {tool_name} action lacks one upfront tool definition")

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    seen, actions = set(), []
    for block in blocks:
        identity, inputs = block.get("id"), block.get("input")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"Native {tool_name} action identity is missing or repeated")
        seen.add(identity)
        if tool_name == "WebFetch":
            if (not isinstance(inputs, dict) or not {"url", "prompt"} <= set(inputs) <= {"url", "prompt", "offset"}
                    or not isinstance(inputs["url"], str) or not isinstance(inputs["prompt"], str)
                    or not inputs["prompt"].strip()):
                raise ValueError("Native WebFetch action input is invalid")
            if "offset" in inputs:
                # CLI 2.1.310 pagination is an optional, declared safe integer.
                # Review this small schema locally: the observer runs with only
                # the standard library, and must not follow schema references.
                schema = definitions[0]["input_schema"]
                properties = schema.get("properties")
                if (set(schema) - {"$schema", "type", "properties", "required", "additionalProperties"}
                        or schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
                        != "https://json-schema.org/draft/2020-12/schema"
                        or schema.get("required") not in (["url", "prompt"], ["prompt", "url"])
                        or schema.get("additionalProperties") is not False
                        or not isinstance(properties, dict) or set(properties) != {"url", "prompt", "offset"}):
                    raise ValueError("Native WebFetch pagination schema is outside the reviewed shape")
                url_schema, prompt_schema, offset_schema = (properties[key] for key in ("url", "prompt", "offset"))
                if (not isinstance(url_schema, dict) or set(url_schema) - {"description", "type", "format"}
                        or url_schema.get("type") != "string" or url_schema.get("format") != "uri"
                        or not isinstance(prompt_schema, dict) or set(prompt_schema) - {"description", "type"}
                        or prompt_schema.get("type") != "string"
                        or not isinstance(offset_schema, dict)
                        or set(offset_schema) - {"description", "type", "minimum", "maximum"}
                        or offset_schema.get("type") != "integer"
                        or type(offset_schema.get("minimum")) is not int
                        or type(offset_schema.get("maximum")) is not int
                        or not 0 <= offset_schema["minimum"] <= offset_schema["maximum"] <= 9007199254740991):
                    raise ValueError("Native WebFetch pagination schema constraints are invalid")
                if (type(inputs["offset"]) is not int
                        or not offset_schema["minimum"] <= inputs["offset"] <= offset_schema["maximum"]):
                    raise ValueError("Native WebFetch action offset is invalid")
            address = urlsplit(inputs["url"])
            if (address.scheme not in {"http", "https"} or not address.hostname
                    or address.username or address.password or any(c.isspace() for c in address.netloc)):
                raise ValueError("Native WebFetch action URL is invalid")
            _ = address.port
        elif (not isinstance(inputs, dict) or not {"query"} <= set(inputs) <= {"query", "allowed_domains", "blocked_domains"}
              or not isinstance(inputs["query"], str) or len(inputs["query"].strip()) < 2
              or any(not isinstance(inputs[key], list) or not all(isinstance(item, str) and item for item in inputs[key])
                     for key in ("allowed_domains", "blocked_domains") if key in inputs)):
            raise ValueError("Native WebSearch action input is invalid")
        actions.append({
            "schema_version": 1, "semantics": f"one_helper_per_native_{tool_name.lower()}_action",
            "association": "completed_response_fifo", "source_request_number": number,
            "tool_use_id": identity, "tool_input_sha256": digest(inputs),
            "tool_definition_sha256": digest(definitions[0]),
        })
    return actions


def webfetch_actions(events: list[dict], request: dict, number: int) -> list[dict]:
    return native_helper_actions(events, request, number, "WebFetch")


def websearch_actions(events: list[dict], request: dict, number: int) -> list[dict]:
    return native_helper_actions(events, request, number, "WebSearch")


def usage_cost(events: list[dict], reservation: Decimal, model: str = MODEL, *, classification="native") -> Decimal:
    policy = model_policy(model)
    usage = {}
    for event in events:
        for source in (event.get("message", {}).get("usage"), event.get("usage")):
            if isinstance(source, dict):
                usage.update(source)
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        if usage.get(key) is None:
            usage[key] = 0  # The native gateway schema marks these optional/nullable.
    complete = (
        any(e.get("type") == "message_stop" for e in events)
        or len(events) == 1
        and events[0].get("type") == "message"
        and events[0].get("stop_reason")
    )
    if (
        not complete
        or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        )
        or any(
            type(usage.get(key, 0)) is not int or usage.get(key, 0) < 0
            for key in ("cache_creation_input_tokens", "cache_read_input_tokens")
        )
    ):
        return reservation  # Unknown interrupted billing retains its whole reservation.
    input_tokens = sum(
        usage.get(key, 0)
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )
    reported = usage.get("cost")
    if isinstance(reported, (int, float, str)) and not isinstance(reported, bool):
        billed = Decimal(str(reported))
        if billed.is_finite() and billed >= 0:
            # This is OpenRouter's authenticated completed-call usage charge,
            # including cached input and reasoning. Keep conservative pricing
            # only when the gateway omits billing, never for an incomplete call.
            return billed
    if classification == "native_websearch":
        # Token counts alone do not attest all server-loop and search charges.
        return reservation
    # Cache writes can cost more than uncached input. Retain a conservative 2x.
    return (
        Decimal(input_tokens) * policy["input_price"] * 2
        + Decimal(usage["output_tokens"]) * policy["output_price"]
    )


def image_token_bound(value) -> int:
    """Conservative decoded-pixel allowance for bounded PNG qualification.

    Compressed file size alone cannot bound vision tokens. Other media require
    an explicit reviewed bound before use with this finite qualification cap.
    """
    if isinstance(value, list):
        return sum(image_token_bound(item) for item in value)
    if not isinstance(value, dict):
        return 0
    if value.get("type") == "image":
        source = value.get("source", {})
        if source.get("type") != "base64" or source.get("media_type") != "image/png":
            raise ValueError("Bounded qualification requires native PNG image bytes")
        image = base64.b64decode(source.get("data", ""), validate=True)
        if len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid native PNG image")
        width, height = struct.unpack(">II", image[16:24])
        if not 0 < width <= 16384 or not 0 < height <= 16384:
            raise ValueError("Native image dimensions are outside qualification limits")
        return width * height * 2 + 1024
    if value.get("type") in {"document", "video", "audio"}:
        raise ValueError("Unbounded media input is outside this qualification")
    return sum(image_token_bound(item) for item in value.values())


def websearch_cost_bound(value: dict, policy: dict) -> Decimal:
    """Reserve every permitted model turn and worst-engine search charge.

    The reviewed Kimi/Sonnet helper adds the Messages step_count_is=1 condition, bounding the outer loop to
    two full-context/output generations. Its original max_uses=8 is preserved,
    including an eight-search USD0.3175 allowance for the worst supported engine
    (Firecrawl,25results). GLM retains its unchanged30-step profile. Result
    character caps never replace a rigorous whole-context bound.
    """
    turns = 2 if value.get("model") in {MODEL, SONNET_MODEL} and value.get("stop_server_tools_when") == [{"type": "step_count_is", "step_count": 1}] else 31
    return (
        Decimal(turns) * (Decimal(1048576) * policy["input_price"] * 2
                          + Decimal(value["max_tokens"]) * policy["output_price"])
        + Decimal(8) * Decimal("0.3175")
    )


class Guard:
    def __init__(self, directory: Path, model: str, effort: str, max_cost: str | None, *, model_http_retries=0):
        self.directory, self.model, self.effort = directory, model, effort
        if type(model_http_retries) is not int or model_http_retries not in {0, 1} or model_http_retries and model != GLM_MODEL:
            raise ValueError("Only the reviewed GLM route permits one explicitly approved HTTP retry")
        self.model_http_retries = model_http_retries
        self.http_retry_unknown_cost = {}
        self.policy = model_policy(model)
        if effort not in self.policy["efforts"]:
            raise ValueError("Native Messages effort is outside the reviewed model policy")
        self.limit = Decimal(max_cost) if max_cost is not None else None
        if self.limit is not None and (not self.limit.is_finite() or self.limit <= 0):
            raise ValueError("Model spend limit must be positive and finite")
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.spent = Decimal(0)
        self.completed_cost = Decimal(0)
        self.inflight = {}
        self.inflight_kinds = {}
        self.waiters = deque()
        self.helper_waiters = deque()
        self.available_actions = deque()
        self.action_credits = {}
        self.claimed_actions = set()
        self.reserved_actions = set()
        self.stopped = False
        self.number = 0
        self.rejection_number = 0
        self.rejection_bytes = 0
        self.failed = False
        self.fingerprints = set()

    def record_failure(self, body: bytes | None, failure: dict) -> dict:
        """Bind the first causal failure and every later rejected raw body.

        Bodies are captured before any parsing/normalization. No headers or
        credentials are copied into metadata. The legacy alias now means the
        first failure's body, never the last SDK/latch rejection.
        """
        with self.lock:
            self.failed = True
            self.condition.notify_all()
            self.rejection_number += 1
            record = {**failure, "rejection_number": self.rejection_number}
            if self.rejection_number > REJECTION_RECORD_LIMIT:
                record.update(request_body_retained=False, retention_limited=True)
                marker = self.directory / "rejection-limit.json"
                if not marker.exists():
                    write_private_once(marker, json.dumps({
                        "rejection_record_limit": REJECTION_RECORD_LIMIT,
                        "rejection_body_limit_bytes": REJECTION_BODY_LIMIT,
                        "further_rejections_not_retained": True,
                        "first_failure_unchanged": True,
                    }).encode() + b"\n")
                return record
            stem = f"rejection-{self.rejection_number:04d}"
            if body is not None:
                record.update(request_body_sha256=hashlib.sha256(body).hexdigest(),
                              request_body_size_bytes=len(body))
                if self.rejection_bytes + len(body) <= REJECTION_BODY_LIMIT:
                    name = stem + "-request.body"
                    write_private_once(self.directory / name, body)
                    self.rejection_bytes += len(body)
                    record.update(request_body_path=name, request_body_retained=True)
                else:
                    record.update(request_body_retained=False, request_body_omitted_reason="total_body_quota")
            else:
                record["request_body_retained"] = False
            first = not (self.directory / "failure.json").exists()
            if first and body is not None:
                write_private_once(self.directory / "first-rejected-request.json", body)
                write_private_once(self.directory / "rejected-request.json", body)
                record["first_rejected_request_path"] = "first-rejected-request.json"
            encoded = json.dumps(record, ensure_ascii=False).encode() + b"\n"
            write_private_once(self.directory / (stem + ".json"), encoded)
            if first:
                write_private_once(self.directory / "failure.json", encoded)
            journal = self.directory / "failures.jsonl"
            try:
                descriptor = os.open(journal, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                previous = b""
            else:
                with os.fdopen(descriptor, "rb") as stream:
                    previous = stream.read()
            publish_private(journal, previous + encoded)
            return record

    def stop(self):
        """Wake admission waiters before shutting down active connections."""
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def settle(self, number: int, cost: Decimal, *, actions=(), failure_callback=None):
        """Commit a completed response; failed/unknown work keeps its reserve."""
        with self.condition:
            try:
                if number not in self.inflight:
                    raise ValueError("Native response reservation was already settled")
                reservation = self.inflight[number]
                if not cost.is_finite() or cost < 0:
                    raise ValueError("Native completed cost is invalid")
                cost += self.http_retry_unknown_cost.get(number, Decimal(0))
                keys = [(action["source_request_number"], action["tool_use_id"]) for action in actions]
                if (actions and self.inflight_kinds[number] != "native"
                        or any(key[0] != number or key in self.action_credits for key in keys)
                        or len(set(keys)) != len(keys)):
                    raise ValueError("Native helper action credit is invalid or repeated")
                self.completed_cost += cost
                self.spent += cost - reservation
                del self.inflight[number]
                del self.inflight_kinds[number]
                if cost > reservation:
                    raise NativeCostBoundError("Native completed cost exceeded its conservative reservation")
                for key, action in zip(keys, actions):
                    self.action_credits[key] = dict(action)
                    self.available_actions.append(key)
            except (ValueError, OSError) as error:
                self.failed = True
                if failure_callback is not None:
                    failure_callback(error)
                self.condition.notify_all()
                raise
            self.condition.notify_all()

    def reserve_http_retry(self, number: int):
        """Keep unknown first-attempt cost and admit a second whole reservation.

        No waiting, borrowing or release of the failed attempt's allowance is
        allowed. If concurrent reservations leave insufficient room, the
        original HTTP failure is returned without another provider request.
        """
        with self.condition:
            if (self.model_http_retries != 1 or self.failed or self.stopped
                    or number not in self.inflight or number in self.http_retry_unknown_cost):
                return None
            extra = self.inflight[number]
            if self.limit is not None and self.spent + extra > self.limit:
                return None
            self.http_retry_unknown_cost[number] = extra
            self.inflight[number] += extra
            self.spent += extra
            return extra

    def claim_helper_action(self, *, deadline, wait_metadata, cancelled=None, failure_callback=None,
                           tool_name="WebFetch"):
        """Wait for a completed action of this helper's kind; consume it once.

        The helper carries no URL/tool ID. FIFO associates admission with one
        declared action; it does not attest which URL produced the page text.
        """
        ticket, wait_started = None, None
        semantics = {"WebFetch": "one_helper_per_native_webfetch_action",
                     "WebSearch": "one_helper_per_native_websearch_action"}.get(tool_name)
        if semantics is None:
            raise ValueError("Native helper action kind is outside the reviewed allowlist")
        wait_metadata.update(classification="none", wait_seconds=0.0, outcome="immediate")
        with self.condition:
            try:
                while True:
                    if self.failed:
                        wait_metadata["outcome"] = "guard_failed"
                        raise ValueError("Inference is disabled after the first failed request")
                    if self.stopped:
                        wait_metadata["outcome"] = "observer_stopping"
                        raise ConnectionAbortedError("Native observer is stopping")
                    if ticket is not None and cancelled is not None and cancelled():
                        wait_metadata["outcome"] = "client_disconnected"
                        raise ConnectionAbortedError("Native helper client disconnected while awaiting its action")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        wait_metadata["outcome"] = "deadline_exceeded"
                        raise TimeoutError(f"Native request deadline reached while awaiting a {tool_name} action")
                    key = next((key for key in self.available_actions
                                if self.action_credits[key]["semantics"] == semantics), None)
                    if key is not None and (not self.helper_waiters or self.helper_waiters[0] is ticket):
                        self.available_actions.remove(key)
                        self.claimed_actions.add(key)
                        if ticket is not None:
                            self.helper_waiters.popleft()
                            ticket = None
                            wait_metadata["outcome"] = "admitted"
                            self.condition.notify_all()
                        return dict(self.action_credits[key])
                    if key is None and "native" not in self.inflight_kinds.values():
                        wait_metadata["outcome"] = "unbacked"
                        raise ValueError(f"Native {tool_name} helper has no unconsumed source action")
                    if ticket is None:
                        if len(self.helper_waiters) >= RESERVATION_WAIT_LIMIT:
                            wait_metadata["outcome"] = "queue_limit"
                            raise ValueError(f"Native {tool_name} helper action wait queue is full")
                        ticket = object()
                        self.helper_waiters.append(ticket)
                        wait_started = time.monotonic()
                        wait_metadata["classification"] = "parent_response_completion"
                    self.condition.wait(min(remaining, 0.1) if cancelled is not None else remaining)
            except (ValueError, OSError) as exc:
                if wait_started is not None:
                    wait_metadata["wait_seconds"] = round(time.monotonic() - wait_started, 6)
                    wait_started = None
                if failure_callback is not None:
                    failure_callback(exc)
                raise
            finally:
                if ticket is not None:
                    self.helper_waiters.remove(ticket)
                    self.condition.notify_all()
                if wait_started is not None:
                    wait_metadata["wait_seconds"] = round(time.monotonic() - wait_started, 6)

    def _retire_returned_helper_actions(self, value):
        """Retire unclaimed fast-path actions, never a claimed/inflight helper.

        Claude may return WebFetch directly without requesting a summarizer.
        A later ordinary native history proves that source action completed.
        Bind its exact context, assistant ordinal, input and upfront definition;
        quoted tool results or a different conversation cannot consume credit.
        The caller holds the admission lock before forwarding this history.
        """
        def context(request):
            first = next((message for message in request["messages"] if message.get("role") == "user"), None)
            return (request.get("system"), first) if first is not None else None

        def digest(item):
            return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

        retired, sources = [], {}
        for key in tuple(self.available_actions):
            if key in self.claimed_actions or key in self.reserved_actions:
                continue
            credit = self.action_credits[key]
            name = {"one_helper_per_native_webfetch_action": "WebFetch",
                    "one_helper_per_native_websearch_action": "WebSearch"}.get(credit.get("semantics"))
            path = self.directory / f"{key[0]:04d}-request.json"
            if name is None:
                continue
            if not path.is_file() or path.stat().st_size > MAX_BODY:
                raise ValueError("Native helper source request evidence is missing or unbounded")
            if key[0] not in sources:
                sources[key[0]] = json.loads(path.read_bytes())
            source = sources[key[0]]
            if context(source) is None or context(source) != context(value):
                continue
            definitions = [[tool for tool in request.get("tools", []) if isinstance(tool, dict) and tool.get("name") == name]
                           for request in (source, value)]
            if any(len(items) != 1 or digest(items[0]) != credit["tool_definition_sha256"] for items in definitions):
                continue
            ordinal = sum(message.get("role") == "assistant" for message in source["messages"])
            seen_assistants, matched = 0, False
            for message in value["messages"]:
                content = message.get("content")
                content = content if isinstance(content, list) else []
                if message.get("role") == "assistant":
                    if seen_assistants == ordinal:
                        matches = [block for block in content if isinstance(block, dict)
                                   and block.get("type") == "tool_use" and block.get("name") == name
                                   and block.get("id") == key[1] and digest(block.get("input")) == credit["tool_input_sha256"]]
                        matched = len(matches) == 1
                    seen_assistants += 1
                elif matched and message.get("role") == "user" and any(
                    isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") == key[1]
                    for block in content
                ):
                    self.available_actions.remove(key)
                    retired.append(key)
                    break
        return retired

    def reserve(self, body: bytes, value: dict, *, deadline=None, wait_metadata=None, cancelled=None,
                failure_callback=None, classification="native", helper_action_credit=None):
        validate_request(value, self.model, self.effort)
        if server_search_tools(value) and classification != "native_websearch":
            raise ValueError("Native server WebSearch requires an action-bound helper reservation")
        # One byte per input token plus 2x cache-write allowance is conservative
        # for UTF-8 text, schemas and base64 images. Output includes thinking.
        media_tokens = image_token_bound(value) if self.limit is not None else 0
        reservation = (
            Decimal(len(body) + media_tokens) * self.policy["input_price"] * 2
            + Decimal(value["max_tokens"]) * self.policy["output_price"]
        )
        if classification == "native_websearch":
            reservation = max(reservation, websearch_cost_bound(value, self.policy))
        fingerprint = hashlib.sha256(
            json.dumps(
                {k: v for k, v in value.items() if k != "metadata"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        wait_started = None
        ticket = None
        metadata = wait_metadata if wait_metadata is not None else {}
        metadata.update(classification="none", wait_seconds=0.0, outcome="immediate")
        admission_deadline = deadline if deadline is not None else time.monotonic() + self.policy["request_timeout_seconds"]
        with self.condition:
            try:
                if self.failed:
                    raise ValueError("Inference is disabled after the first failed request")
                if self.stopped:
                    raise ConnectionAbortedError("Native observer is stopping")
                if classification == "native":
                    self._retire_returned_helper_actions(value)
                if classification not in {"native", "native_webfetch_summary", "native_websearch"}:
                    raise ValueError("Native request classification is invalid")
                credited_helper = classification != "native"
                if credited_helper:
                    inherited_keys = {"thinking"}
                    if classification == "native_websearch" and self.model in {MODEL, SONNET_MODEL}:
                        inherited_keys.add("stop_server_tools_when")
                    raw_shape = {key: item for key, item in value.items() if key not in inherited_keys}
                    effective_shape, kind = normalize_request(raw_shape, self.model, self.effort)
                    if kind != classification or effective_shape != value:
                        raise ValueError("Native credited helper does not retain the observed envelope")
                    name = "websearch" if classification == "native_websearch" else "webfetch"
                    if (not isinstance(helper_action_credit, dict)
                            or helper_action_credit.get("semantics") != f"one_helper_per_native_{name}_action"):
                        raise ValueError("Native helper requires a fresh matching source action")
                    key = (helper_action_credit.get("source_request_number"), helper_action_credit.get("tool_use_id"))
                    if (key not in self.claimed_actions or key in self.reserved_actions
                            or self.action_credits.get(key) != helper_action_credit):
                        raise ValueError("Native helper source action was not freshly claimed")
                    if classification == "native_websearch":
                        # Empty optional lists add no filter. Preserve the full
                        # source digest; nonempty domain constraints cannot be
                        # dropped by this observed query-only callback.
                        query = value["messages"][0]["content"][0]["text"].removeprefix("Perform a web search for the query: ")
                        query_digests = {hashlib.sha256(json.dumps(
                            {"query": query, **empty}, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                        ).encode()).hexdigest() for empty in (
                            {}, {"allowed_domains": []}, {"blocked_domains": []},
                            {"allowed_domains": [], "blocked_domains": []},
                        )}
                        if helper_action_credit["tool_input_sha256"] not in query_digests:
                            raise ValueError("Native WebSearch helper query differs from its source action")
                    self.reserved_actions.add(key)
                if fingerprint in self.fingerprints and not credited_helper:
                    self.failed = True
                    self.condition.notify_all()
                    raise ValueError("An identical inference request cannot be retried")
                self.fingerprints.add(fingerprint)
                while True:
                    if self.failed:
                        metadata["outcome"] = "guard_failed"
                        raise ValueError("Inference is disabled after the first failed request")
                    if self.stopped:
                        metadata["outcome"] = "observer_stopping"
                        raise ConnectionAbortedError("Native observer is stopping")
                    if ticket is not None and cancelled is not None and cancelled():
                        metadata["outcome"] = "client_disconnected"
                        raise ConnectionAbortedError("Native client disconnected while awaiting budget capacity")
                    if self.limit is not None and self.completed_cost + reservation > self.limit:
                        metadata["outcome"] = "completed_budget_exhausted"
                        raise ValueError("Qualification model budget cannot cover this request")
                    remaining = admission_deadline - time.monotonic()
                    if remaining <= 0:
                        metadata["outcome"] = "deadline_exceeded"
                        raise TimeoutError("Native request deadline reached while awaiting budget capacity")
                    fits = self.limit is None or self.spent + reservation <= self.limit
                    first = not self.waiters or self.waiters[0] is ticket
                    if fits and first:
                        self.spent += reservation
                        self.number += 1
                        self.inflight[self.number] = reservation
                        self.inflight_kinds[self.number] = classification
                        if ticket is not None:
                            self.waiters.popleft()
                            ticket = None
                            metadata["outcome"] = "admitted"
                            self.condition.notify_all()
                        return self.number, reservation
                    if ticket is None:
                        if len(self.waiters) >= RESERVATION_WAIT_LIMIT:
                            metadata["outcome"] = "queue_limit"
                            raise ValueError("Native budget reservation wait queue is full")
                        ticket = object()
                        self.waiters.append(ticket)
                        wait_started = time.monotonic()
                        metadata["classification"] = "inflight_reservations"
                    self.condition.wait(min(remaining, 0.1) if cancelled is not None else remaining)
            except (ValueError, OSError) as exc:
                if wait_started is not None:
                    metadata["wait_seconds"] = round(time.monotonic() - wait_started, 6)
                    wait_started = None
                if failure_callback is not None:
                    failure_callback(exc)
                raise
            finally:
                if ticket is not None:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
                if wait_started is not None:
                    metadata["wait_seconds"] = round(time.monotonic() - wait_started, 6)


class IncompleteNativeResponse(ValueError):
    pass


class UpstreamHTTPError(ValueError):
    pass


class NativeResponseLimitError(ValueError):
    pass


class NativeCostBoundError(ValueError):
    pass


def _handler_class(directory: Path, model: str, effort: str, guard: Guard, token: str):
    """Keep the production gateway fixed; the factory permits offline socket tests."""
    policy = model_policy(model)
    timeout_fields = {key: policy[key] for key in (
        "request_timeout_seconds", "request_timeout_semantics",
        "connect_timeout_seconds", "api_client_timeout_ms",
    )}
    active, active_lock = set(), threading.Lock()
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        @staticmethod
        def stop_connections():
            stopping.set()
            guard.stop()
            with active_lock:
                sockets = list(active)
            for sock in sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

        def log_message(self, *args):
            pass

        def reply_error(self, status: int, reason: str):
            payload = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": reason},
                }
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            parsed = urlsplit(self.path)
            if parsed.path not in {
                "/v1/messages",
                "/v1/messages/count_tokens",
            } or parsed.query not in {"", "beta=true"}:
                guard.record_failure(None, {"error_type": "GatewayPathRejected",
                                           "error_phase": "path", "request_number": None})
                self.reply_error(
                    403, "Gateway path is outside the native Messages allowlist"
                )
                return
            number, value, body, incoming_body = None, None, None, None
            upstream, upstream_socket, result = None, None, None
            watchdog, response_file = None, None
            response = bytearray()
            started = time.monotonic()
            deadline = started + policy["request_timeout_seconds"]
            started_at = datetime.now(UTC).isoformat()
            first_byte_at, last_byte_at = None, None
            upstream_status, content_type = None, None
            response_headers, failure = {}, None
            headers_sent, disconnected = False, False
            http_attempts, http_retry_count = [], 0
            attempt = None
            reservation_wait = {"classification": "none", "wait_seconds": 0.0, "outcome": "immediate"}
            helper_wait = {"classification": "none", "wait_seconds": 0.0, "outcome": "immediate"}
            phase = "request_body"

            def preserve_failure(exc):
                # Budget admission/settlement can wake other handlers. Capture
                # their initiating failure while its condition lock is held so
                # a newly awakened latch rejection cannot become the cause.
                nonlocal failure, disconnected
                disconnected = disconnected or (not stopping.is_set() and phase.startswith("downstream_") and isinstance(exc, OSError))
                failure = guard.record_failure(incoming_body, {
                    "error_type": type(exc).__name__, "request_number": number,
                    "guard_reason": str(exc) if type(exc) is ValueError and phase == "guard" else None,
                    "error_phase": phase, "upstream_status": upstream_status,
                    "observer_stopping": stopping.is_set(),
                    "response_bytes_retained": len(response),
                    "started_at_utc": started_at,
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "budget_reservation_wait": reservation_wait,
                    "helper_action_wait": helper_wait,
                })

            try:
                if stopping.is_set():
                    raise ConnectionAbortedError("Native observer is stopping")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Unbounded or missing native request body")
                incoming_body = self.rfile.read(size)
                body = incoming_body
                value = json.loads(body)
                counting = parsed.path.endswith("/count_tokens")
                phase = "guard"
                retry_headers = self.headers.get_all("x-stainless-retry-count", [])
                if len(retry_headers) > 1 or retry_headers and retry_headers != ["0"]:
                    raise ValueError("Native automatic API retries are not permitted")
                sdk_retry_count = 0 if retry_headers else None
                if counting:
                    if value.get("model") != model:
                        raise ValueError(
                            "Token-count model differs from the pinned model"
                        )
                    validate_tool_protocol(value, model)
                else:
                    value, classification = normalize_request(value, model, effort)
                    if model == SONNET_MODEL and any(
                        isinstance(message, dict) and "output_config" in message
                        for message in value["messages"]
                    ) and "per-turn-control-2026-07-01" not in {
                        part.strip() for part in self.headers.get("anthropic-beta", "").split(",")
                    }:
                        raise ValueError("Native Sonnet effort markers require the pinned CLI beta header")
                    if classification != "native":
                        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
                    def client_disconnected():
                        try:
                            if not select.select([self.connection], [], [], 0)[0]:
                                return False
                            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
                        except BlockingIOError:
                            return False
                        except (OSError, ValueError):
                            return True

                    helper_action_credit = None
                    if classification in {"native_webfetch_summary", "native_websearch"}:
                        if sdk_retry_count != 0:
                            raise ValueError("Native credited helper requires an explicit zero SDK retry count")
                        helper_action_credit = guard.claim_helper_action(
                            deadline=deadline, wait_metadata=helper_wait, cancelled=client_disconnected,
                            failure_callback=preserve_failure,
                            tool_name="WebSearch" if classification == "native_websearch" else "WebFetch",
                        )
                    number, reservation = guard.reserve(
                        body, value, deadline=deadline, wait_metadata=reservation_wait,
                        cancelled=client_disconnected, failure_callback=preserve_failure,
                        classification=classification, helper_action_credit=helper_action_credit,
                    )
                    write_private_once(directory / f"{number:04d}-request.raw.body", incoming_body)
                    write_private_once(directory / f"{number:04d}-request.json", body)
                    write_private_once(directory / f"{number:04d}-request-policy.json", json.dumps({
                        "schema_version": 1,
                        "classification": classification,
                        "inherited_fields": ["thinking"] if classification != "native" and "thinking" not in json.loads(incoming_body) else [],
                        "added_fields": sorted(set(value) - set(json.loads(incoming_body))),
                        "raw_request_sha256": hashlib.sha256(incoming_body).hexdigest(),
                        "effective_request_sha256": hashlib.sha256(body).hexdigest(),
                        "sdk_retry_count": sdk_retry_count,
                        "helper_action_credit": helper_action_credit,
                        **({"native_beta_headers": [part.strip() for part in self.headers.get("anthropic-beta", "").split(",") if part.strip()]}
                           if model == SONNET_MODEL else {}),
                    }).encode() + b"\n")
                    response_file = (directory / f"{number:04d}-response.body").open("xb")
                    os.chmod(response_file.name, 0o600)
                headers = {
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                }
                for name in ("anthropic-version", "anthropic-beta", "accept"):
                    if self.headers.get(name):
                        headers[name] = self.headers[name]
                def remaining():
                    if stopping.is_set():
                        raise ConnectionAbortedError("Native observer is stopping")
                    seconds = policy["request_timeout_seconds"]
                    if policy["request_timeout_semantics"] == "total":
                        seconds = deadline - time.monotonic()
                        if seconds <= 0:
                            raise TimeoutError("Native request deadline reached")
                    elif len(http_attempts) > 1 and phase in {"connect", "request_send", "response_headers"}:
                        seconds = min(seconds, deadline - time.monotonic())
                        if seconds <= 0:
                            raise TimeoutError("Native retry pre-response deadline reached")
                    upstream.timeout = seconds
                    if upstream_socket is not None:
                        upstream_socket.settimeout(seconds)
                    self.connection.settimeout(seconds)
                    return seconds

                def expire():
                    if upstream_socket is not None:
                        try:
                            upstream_socket.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    if upstream is not None:
                        upstream.close()

                while True:
                    phase = "connect"
                    if stopping.is_set() or number is not None and client_disconnected():
                        disconnected = not stopping.is_set()
                        raise ConnectionAbortedError("Native request stopped before an upstream attempt")
                    if guard.model_http_retries and number is not None:
                        attempt = {"attempt": len(http_attempts) + 1, "status": None, "content_type": None,
                                   "request_sha256": hashlib.sha256(body).hexdigest(), "request_sent": False,
                                   "response_body_path": f"{number:04d}-response.body", "response_complete": False,
                                   "downstream_headers_sent": False, "downstream_body_bytes": 0,
                                   "started_at_utc": datetime.now(UTC).isoformat(), "completed_at_utc": None,
                                   "retry_decision": "not_eligible", "generation_headers": {}}
                        http_attempts.append(attempt)
                    upstream = http.client.HTTPSConnection("openrouter.ai", timeout=min(
                        policy["connect_timeout_seconds"], max(0.001, deadline - time.monotonic())
                    ) if len(http_attempts) > 1 else policy["connect_timeout_seconds"])
                    if policy["request_timeout_semantics"] == "total":
                        watchdog = threading.Timer(remaining(), expire)
                        watchdog.daemon = True
                        watchdog.start()
                    upstream.connect()
                    upstream_socket = upstream.sock
                    if upstream_socket is None:
                        raise ConnectionError("Native upstream did not establish a socket")
                    with active_lock:
                        active.add(upstream_socket)
                    remaining()
                    phase = "request_send"
                    if attempt is not None:
                        with guard.condition:
                            if guard.failed or guard.stopped:
                                raise ConnectionAbortedError("Native guard stopped before the next HTTP attempt")
                            attempt["request_sent"] = True
                            http_retry_count = int(attempt["attempt"] > 1)
                    upstream.request("POST", "/api" + self.path, body, headers)
                    remaining()
                    phase = "response_headers"
                    result = upstream.getresponse()
                    upstream_status = result.status
                    content_type = result.getheader("Content-Type", "application/json")
                    response_headers = {key: result.getheader(key) for key in (
                        "x-generation-id", "x-openrouter-generation-id", "x-request-id",
                    ) if result.getheader(key)}
                    candidate = bool(attempt is not None and attempt["attempt"] == 1
                                     and upstream_status in {429, 520} and "text/event-stream" not in content_type.lower())
                    if attempt is not None:
                        attempt.update(status=upstream_status, content_type=content_type, generation_headers=response_headers,
                                       retry_decision="not_needed" if upstream_status == 200 else "not_eligible")
                    if not candidate:
                        phase = "downstream_headers"
                        self.send_response(upstream_status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Connection", "close")
                        headers_sent = True
                        self.end_headers()
                        if attempt is not None:
                            attempt["downstream_headers_sent"] = True
                    while True:
                        phase = "response_read"
                        remaining()
                        chunk = result.read1(65536)
                        if not chunk:
                            if getattr(result, "length", None) not in {None, 0}:
                                raise IncompleteNativeResponse("Native HTTP body ended before its declared length")
                            break
                        now = datetime.now(UTC).isoformat()
                        first_byte_at = first_byte_at or now
                        last_byte_at = now
                        available = MAX_RESPONSE_BODY - len(response)
                        retained = chunk[:available]
                        response.extend(retained)
                        if response_file is not None:
                            response_file.write(retained)
                            response_file.flush()
                        if len(chunk) > available:
                            raise NativeResponseLimitError("Native response exceeds the private evidence bound")
                        if not candidate:
                            phase = "downstream_write"
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            if attempt is not None:
                                attempt["downstream_body_bytes"] += len(chunk)
                    if attempt is not None:
                        attempt.update(response_sha256=hashlib.sha256(response).hexdigest(),
                                       response_bytes_retained=len(response), response_complete=True,
                                       completed_at_utc=datetime.now(UTC).isoformat())
                    if not candidate:
                        break
                    error_name = f"{number:04d}-http-attempt-1-response.body"
                    write_private_once(directory / error_name, bytes(response))
                    attempt["response_body_path"] = error_name
                    phase = "http_retry_admission"
                    if client_disconnected():
                        disconnected = True
                        attempt["retry_decision"] = "client_disconnected"
                        raise ConnectionAbortedError("Native retry client disconnected")
                    enough_time = deadline - time.monotonic() > HTTP_RETRY_BACKOFF_SECONDS
                    extra = guard.reserve_http_retry(number) if enough_time else None
                    attempt["retry_decision"] = "budget_unavailable" if enough_time else "deadline_unavailable"
                    if extra is None:
                        phase = "downstream_headers"
                        self.send_response(upstream_status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Connection", "close")
                        headers_sent = True
                        self.end_headers()
                        attempt["downstream_headers_sent"] = True
                        phase = "downstream_write"
                        self.wfile.write(response)
                        self.wfile.flush()
                        attempt["downstream_body_bytes"] = len(response)
                        break
                    result.close()
                    upstream.close()
                    with active_lock:
                        active.discard(upstream_socket)
                    result, upstream, upstream_socket = None, None, None
                    phase = "http_retry_backoff"
                    wake = time.monotonic() + HTTP_RETRY_BACKOFF_SECONDS
                    while time.monotonic() < wake:
                        if stopping.is_set() or guard.failed or guard.stopped or client_disconnected() or time.monotonic() >= deadline:
                            disconnected = not stopping.is_set() and client_disconnected()
                            attempt["retry_decision"] = "cancelled_before_retry"
                            raise ConnectionAbortedError("Native request stopped during bounded retry backoff")
                        stopping.wait(min(0.05, max(0, wake - time.monotonic())))
                    attempt["retry_decision"] = "retried"
                    response.clear()
                    response_file.seek(0)
                    response_file.truncate()
                    first_byte_at, last_byte_at = None, None
                    upstream_status, content_type, response_headers = None, None, {}
                if number is not None:
                    phase = "response_validate"
                    if upstream_status != 200:
                        raise UpstreamHTTPError("Native upstream returned an unsuccessful HTTP status")
                    phase = "response_decode"
                    events = response_events(bytes(response), content_type)
                    complete = (
                        any(e.get("type") == "message_stop" for e in events)
                        or len(events) == 1
                        and events[0].get("type") == "message"
                        and events[0].get("stop_reason")
                    )
                    phase = "response_validate"
                    if not complete or any(e.get("type") == "error" for e in events):
                        raise IncompleteNativeResponse("Native response is incomplete or contains an error event")
                    if classification == "native_webfetch_summary":
                        for event in events:
                            blocks = [event.get("content_block")]
                            for source in (event, event.get("message")):
                                if isinstance(source, dict) and isinstance(source.get("content"), list):
                                    blocks.extend(source["content"])
                            if any(isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}
                                   for block in blocks):
                                raise IncompleteNativeResponse("Native tool-less summary attempted a tool call")
                    elif classification == "native_websearch":
                        for event in events:
                            blocks = [event.get("content_block")]
                            for source in (event, event.get("message")):
                                if isinstance(source, dict) and isinstance(source.get("content"), list):
                                    blocks.extend(source["content"])
                            if any(isinstance(block, dict) and (block.get("type") == "tool_use" or (
                                block.get("type") == "server_tool_use" and block.get("name") != "web_search"
                            )) for block in blocks):
                                raise IncompleteNativeResponse("Native WebSearch helper attempted an undeclared tool call")
                    try:
                        cost = usage_cost(events, reservation, model, classification=classification)
                    except ArithmeticError:
                        raise ValueError("Native reported usage cost is invalid") from None
                    actions = (webfetch_actions(events, value, number) + websearch_actions(events, value, number)
                               if classification == "native" else [])
                    guard.settle(number, cost, actions=actions, failure_callback=preserve_failure)
                self.close_connection = True
            except (ValueError, OSError, http.client.HTTPException) as exc:
                # Retain classifications only, never headers, secrets or an
                # upstream exception string that might contain private content.
                if failure is None:
                    preserve_failure(exc)
                self.close_connection = True
                # A secondary broken pipe must never replace the first cause,
                # and another HTTP header inside an SSE body corrupts its bytes.
                if not headers_sent:
                    try:
                        self.reply_error(
                            400 if isinstance(exc, ValueError) else 502,
                            "Native gateway guard failed; inspect private qualification evidence",
                        )
                    except OSError:
                        pass
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                if upstream is not None:
                    with active_lock:
                        active.discard(upstream_socket)
                    if result is not None:
                        result.close()
                    upstream.close()
                if response_file is not None:
                    response_file.close()
                if number is not None:
                    if attempt is not None and not attempt.get("completed_at_utc"):
                        attempt.update(completed_at_utc=datetime.now(UTC).isoformat(),
                                       response_sha256=hashlib.sha256(response).hexdigest(), response_bytes_retained=len(response))
                    write_json(directory / f"{number:04d}-exchange.json", {
                        "status": upstream_status, "content_type": content_type,
                        "request_sha256": hashlib.sha256((directory / f"{number:04d}-request.json").read_bytes()).hexdigest(),
                        "response_sha256": hashlib.sha256(response).hexdigest(),
                        "upstream": "https://openrouter.ai/api/v1/messages",
                        "automatic_retries": guard.model_http_retries, "client_disconnected": disconnected,
                        **({"model_http_retries": guard.model_http_retries} if model == GLM_MODEL else {}),
                        **({"http_attempts": http_attempts, "http_attempt_count": len(http_attempts),
                            "http_retry_count": http_retry_count, "request_reservation_usd": str(reservation),
                            "http_retry_unknown_cost_upper_bound_usd": str(guard.http_retry_unknown_cost.get(number, Decimal(0)))}
                           if guard.model_http_retries else {}),
                        "generation_headers": response_headers, **timeout_fields,
                        "cost_upper_bound_usd": str(guard.spent),
                        "started_at_utc": started_at, "first_byte_at_utc": first_byte_at,
                        "last_byte_at_utc": last_byte_at,
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                        "budget_reservation_wait": reservation_wait,
                        "helper_action_wait": helper_wait,
                        "response_bytes_retained": len(response),
                        "failure": failure,
                    })

    return Handler


def serve(directory: Path, model: str, effort: str, max_cost: str | None, port: int, model_http_retries=0):
    os.umask(0o077)
    if os.geteuid() != 0 or os.getegid() != 0:
        raise ValueError("Native observer must run as the protected root actor")
    parent = directory.parent
    if (parent != Path("/root/.picbench-claude-native") or parent.is_symlink()
            or parent.stat().st_uid != 0 or parent.stat().st_gid != 0
            or parent.stat().st_mode & 0o777 != 0o700):
        raise ValueError("Native observer evidence root is not private")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    guard = Guard(directory, model, effort, max_cost, model_http_retries=model_http_retries)
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not token:
        raise ValueError("OpenRouter credential missing")
    Handler = _handler_class(directory, model, effort, guard, token)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = False

    def stop(*_):
        Handler.stop_connections()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    write_json(
        directory / "ready.json",
        {
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "pid": os.getpid(),
            "model": model,
            "reasoning_effort": effort,
            "max_cost_usd": max_cost,
            "automatic_retries": model_http_retries,
            **({"model_http_retries": model_http_retries} if model == GLM_MODEL else {}),
            "native_request_policy_version": 4 if model == SONNET_MODEL else 3,
            "native_missing_thinking_policy": "webfetch_websearch_adaptive_v1",
            **{key: model_policy(model)[key] for key in (
                "request_timeout_seconds", "request_timeout_semantics",
                "connect_timeout_seconds", "api_client_timeout_ms",
            )},
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "execution_uid": os.geteuid(),
            "execution_gid": os.getegid(),
        },
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--effort", required=True, choices=("low", "medium", "high", "max")
    )
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-http-retries", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    serve(args.state_dir, args.model, args.effort, args.max_cost_usd, args.port, args.model_http_retries)
