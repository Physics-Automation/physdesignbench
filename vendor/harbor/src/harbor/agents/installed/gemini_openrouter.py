"""Run-scoped Google-format transport for the unmodified Gemini CLI.

OpenRouter does not expose Google's generateContent API. This bridge uses
LiteLLM's tool/config conversion, but preserves original OpenRouter assistant
messages itself: the generic converter loses reasoning_details and call IDs.
Upstream SSE is consumed immediately; native headers and keepalives are sent
while the complete signed response is assembled. No transport retries.
"""

from __future__ import annotations

import base64
import copy
import hmac
import json
import math
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from typing import Any, override

import httpx

MAX_BODY = 32 * 1024 * 1024
PROVIDERS = {
    "only": ["google-ai-studio", "google-vertex"],
    "allow_fallbacks": False,
    "require_parameters": True,
}


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args)

    @override
    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    @override
    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    @override
    def handle_error(self, request, client_address):
        # HTTP errors must not leak request bodies or provider credentials.
        pass


def _merge_stream_delta(target: dict[str, Any], incoming: dict[str, Any]):
    """Merge Chat Completions delta fields without discarding provider extensions."""
    for key, value in incoming.items():
        if value is None:
            continue
        if key not in target or target[key] is None:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge_stream_delta(target[key], value)
        elif isinstance(value, str) and isinstance(target[key], str):
            if key in {"id", "type", "format", "role"}:
                if target[key] != value:
                    raise ValueError("Conflicting stream identity")
            else:
                target[key] += value
        elif isinstance(value, list) and isinstance(target[key], list):
            target[key].extend(copy.deepcopy(value))
        elif target[key] != value:
            raise ValueError("Conflicting stream field")


class _OpenRouterStream:
    """Accumulate exact tool IDs/arguments and indexed encrypted reasoning blocks."""

    def __init__(self):
        self.result: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                }
            ],
            "usage": {},
        }
        self.tools = {}
        self.reasoning = {}

    def add(self, event):
        if event.get("error"):
            raise ValueError("OpenRouter stream returned an error")
        for name in ("id", "model", "provider"):
            if event.get(name) is not None:
                self.result[name] = event[name]
        if event.get("usage") is not None:
            self.result["usage"].update(event["usage"])
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected additional stream candidate")
            target = self.result["choices"][0]
            if choice.get("finish_reason") is not None:
                target["finish_reason"] = choice["finish_reason"]
            delta = copy.deepcopy(choice.get("delta") or {})
            for field, blocks in (
                ("tool_calls", self.tools),
                ("reasoning_details", self.reasoning),
            ):
                for position, block in enumerate(delta.pop(field, None) or []):
                    index = block.get("index", position)
                    if not isinstance(index, int):
                        raise ValueError("Invalid indexed stream block")
                    # Google can reuse an index for text and encrypted reasoning.
                    key = (
                        (index, block.get("type"))
                        if field == "reasoning_details"
                        else index
                    )
                    _merge_stream_delta(blocks.setdefault(key, {}), block)
            _merge_stream_delta(target["message"], delta)

    def complete(self):
        if not self.result["choices"][0]["finish_reason"]:
            raise ValueError("OpenRouter stream ended before its final candidate")
        message = self.result["choices"][0]["message"]
        if self.tools:
            message["tool_calls"] = [
                {k: v for k, v in self.tools[i].items() if k != "index"}
                for i in sorted(self.tools)
            ]
        if self.reasoning:
            message["reasoning_details"] = [
                self.reasoning[i]
                for i in sorted(self.reasoning, key=lambda key: key[0])
            ]
        return self.result


class GeminiOpenRouterBridge:
    def __init__(self, key: str, model: str, host: str, logs_dir: Path):
        if (
            not key
            or not model.startswith("google/gemini-")
            or "@" in model
            or ":" in model
        ):
            raise ValueError(
                "OpenRouter Gemini bridge requires an explicit Google Gemini model"
            )
        self.model = model
        self.cli_model = model.split("/", 1)[1]
        self.key = key
        self.token = secrets.token_urlsafe(32)
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        # Never use environment proxy settings or automatic transport retries.
        self.client = httpx.Client(timeout=360, trust_env=False)
        self.server = _BridgeServer((host, 0), self._handler())
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.client.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

        allowed = {
            "contents",
            "systemInstruction",
            "tools",
            "toolConfig",
            "generationConfig",
            "safetySettings",
        }
        if set(payload) - allowed:
            raise ValueError("Unsupported Gemini request fields")
        for tool in payload.get("tools", []):
            if set(tool) != {"functionDeclarations"}:
                raise ValueError(
                    "Only client function tools are supported by this transport"
                )
        config = payload.get("generationConfig", {})
        supported = {
            "temperature",
            "maxOutputTokens",
            "topP",
            "topK",
            "stopSequences",
            "thinkingConfig",
            "responseMimeType",
            "responseSchema",
            "responseJsonSchema",
            "candidateCount",
            "seed",
        }
        if set(config) - supported:
            raise ValueError("Unsupported generation configuration")
        if config.get("topK") is not None or payload.get("safetySettings"):
            raise ValueError("OpenRouter cannot preserve topK/safety settings")
        request = GoogleGenAIAdapter().translate_generate_content_to_completion(
            self.model,
            [],
            config=config,
            tools=payload.get("tools"),
            toolConfig=payload.get("toolConfig"),
        )
        declarations = [
            decl
            for tool in payload.get("tools", [])
            for decl in tool["functionDeclarations"]
        ]
        for native, translated in zip(
            declarations, request.get("tools", []), strict=True
        ):
            if set(native) - {
                "name",
                "description",
                "parameters",
                "parametersJsonSchema",
            }:
                raise ValueError("Unsupported function declaration fields")
            if "parameters" in native:
                if "parametersJsonSchema" in native:
                    raise ValueError("Ambiguous function parameter schema")
                translated["function"]["parameters"] = self.json_schema(
                    native["parameters"]
                )
        tool_config = payload.get("toolConfig", {})
        if set(tool_config) - {"functionCallingConfig"}:
            raise ValueError("Unsupported tool configuration")
        calling = tool_config.get("functionCallingConfig", {})
        if set(calling) - {"mode", "allowedFunctionNames"} or calling.get(
            "mode", "AUTO"
        ) not in {"AUTO", "ANY", "NONE"}:
            raise ValueError("Unsupported function calling configuration")
        names = calling.get("allowedFunctionNames")
        if names:
            if calling.get("mode") != "ANY" or not set(names).issubset(
                {d["name"] for d in declarations}
            ):
                raise ValueError("Unsupported allowed function names")
            request["tools"] = [
                tool for tool in request["tools"] if tool["function"]["name"] in names
            ]
        if config.get("candidateCount", 1) != 1:
            raise ValueError("Only one candidate per native request is supported")
        if "seed" in config:
            request["seed"] = config["seed"]
        mime = config.get("responseMimeType")
        if mime not in {None, "text/plain", "application/json"}:
            raise ValueError("Unsupported response MIME type")
        schema = config.get("responseJsonSchema") or config.get("responseSchema")
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "gemini_response",
                    "schema": self.json_schema(schema),
                },
            }
        elif mime == "application/json":
            request["response_format"] = {"type": "json_object"}
        request["messages"] = self.messages(payload)
        request["provider"] = copy.deepcopy(PROVIDERS)
        request["stream"] = False
        thinking = config.get("thinkingConfig", {})
        if thinking:
            if set(thinking) - {"thinkingLevel", "includeThoughts", "thinkingBudget"}:
                raise ValueError("Unsupported thinking configuration")
            reasoning = {"exclude": not thinking.get("includeThoughts", True)}
            if "thinkingLevel" in thinking:
                reasoning["effort"] = thinking["thinkingLevel"].lower()
            elif "thinkingBudget" in thinking and thinking["thinkingBudget"] > 0:
                reasoning["max_tokens"] = thinking["thinkingBudget"]
            else:
                raise ValueError("Explicit thinking level or positive budget required")
            request["reasoning"] = reasoning
        return request

    @staticmethod
    def json_schema(value):
        """Normalize Google's schema spelling without modifying property names."""
        if not isinstance(value, dict):
            return value
        result = copy.deepcopy(value)
        if isinstance(result.get("type"), str):
            result["type"] = result["type"].lower()
        if result.pop("nullable", False) and isinstance(result.get("type"), str):
            result["type"] = [result["type"], "null"]
        if "propertyOrdering" in result:
            raise ValueError("Explicit JSON property ordering is not supported")
        for field in ("properties", "$defs", "definitions"):
            if field in result:
                result[field] = {
                    name: GeminiOpenRouterBridge.json_schema(schema)
                    for name, schema in result[field].items()
                }
        for name in ("items", "additionalProperties"):
            if isinstance(result.get(name), dict):
                result[name] = GeminiOpenRouterBridge.json_schema(result[name])
        for name in ("anyOf", "allOf", "oneOf", "prefixItems"):
            if name in result:
                result[name] = [
                    GeminiOpenRouterBridge.json_schema(schema)
                    for schema in result[name]
                ]
        return result

    def messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        message: dict[str, Any]
        messages = []
        instruction = payload.get("systemInstruction", {})
        if instruction:
            messages.append(
                {
                    "role": "system",
                    "content": "\n".join(p["text"] for p in instruction["parts"]),
                }
            )
        pending: list[dict[str, Any]] = []
        returned = {}
        for content in payload["contents"]:
            role, parts = content.get("role", "user"), content["parts"]
            if role == "model":
                signatures = {
                    p["thoughtSignature"] for p in parts if p.get("thoughtSignature")
                }
                if signatures:
                    with self.lock:
                        originals = [
                            copy.deepcopy(self.history[s])
                            for s in signatures
                            if s in self.history
                        ]
                    if len(originals) != 1 or len(signatures) != 1:
                        raise ValueError(
                            "Reasoning continuation is missing from this run's bridge"
                        )
                    message = originals[0]
                else:
                    # Unsigned assistant text may be synthesized by CLI compaction.
                    if any("functionCall" in p or p.get("thought") for p in parts):
                        raise ValueError("Signed tool/reasoning history was lost")
                    message = {
                        "role": "assistant",
                        "content": "".join(p["text"] for p in parts),
                    }
                messages.append(message)
                pending = copy.deepcopy(message.get("tool_calls", []))
                returned = {}
            elif role == "user":
                text_parts = []
                for part in parts:
                    if "functionResponse" in part:
                        response = part["functionResponse"]
                        response_id = response.get("id")
                        if response_id and response_id in returned:
                            # Native parallel-tool retries can repeat the same
                            # response batch. Emit each tool result exactly once.
                            if returned[response_id] != response:
                                raise ValueError("Conflicting repeated tool response")
                            continue
                        candidates = [
                            c
                            for c in pending
                            if c["function"]["name"] == response["name"]
                        ]
                        if not candidates:
                            raise ValueError("Tool response has no matching call")
                        if response.get("id"):
                            matches = [
                                c for c in candidates if c["id"] == response["id"]
                            ]
                        else:
                            matches = candidates if len(candidates) == 1 else []
                        if len(matches) != 1:
                            raise ValueError("Ambiguous or unknown tool-call ID")
                        call = matches[0]
                        pending.remove(call)
                        returned[call["id"]] = copy.deepcopy(response)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(response["response"]),
                            }
                        )
                    elif "text" in part:
                        text_parts.append({"type": "text", "text": part["text"]})
                    elif "inlineData" in part and part["inlineData"][
                        "mimeType"
                    ].startswith("image/"):
                        data = part["inlineData"]
                        text_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{data['mimeType']};base64,{data['data']}"
                                },
                            }
                        )
                    else:
                        raise ValueError(
                            "Unsupported content part; refusing lossy conversion"
                        )
                if text_parts:
                    messages.append({"role": "user", "content": text_parts})
            else:
                raise ValueError("Unsupported content role")
        return messages

    def _record_usage(self, result, cli_requested_model=None, *, complete=True):
        usage = result.get("usage") or {}
        record = {
            "id": result.get("id"),
            "model": result.get("model"),
            "requested_model": self.model,
            "cli_requested_model": cli_requested_model or self.cli_model,
            "provider": result.get("provider"),
            "usage": usage,
            "cost_source": "openrouter_usage",
            "response_complete": complete,
        }
        with self.lock:
            self.records.append(record)
            with (self.logs_dir / "openrouter-usage.jsonl").open("a") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")

    def response(
        self, result: dict[str, Any], *, cli_requested_model: str | None = None
    ) -> dict[str, Any]:
        # Persist billable provider evidence before any conversion can fail.
        self._record_usage(result, cli_requested_model)
        if result.get("error") or not result.get("choices"):
            raise ValueError("OpenRouter returned no candidate")
        choice = result["choices"][0]
        message = copy.deepcopy(choice["message"])
        parts: list[dict[str, Any]] = []
        if message.get("reasoning"):
            parts.append({"text": message["reasoning"], "thought": True})
        if message.get("content"):
            if not isinstance(message["content"], str):
                raise ValueError("Unsupported response content")
            parts.append({"text": message["content"]})
        for call in message.get("tool_calls", []):
            parts.append(
                {
                    "functionCall": {
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "args": json.loads(call["function"]["arguments"]),
                    }
                }
            )
        if not parts:
            parts = [{"text": ""}]
        # An opaque native signature points to the EXACT original assistant message,
        # including all encrypted reasoning_details, IDs, and provider extensions.
        signature = base64.b64encode(secrets.token_bytes(32)).decode()
        parts[-1]["thoughtSignature"] = signature
        with self.lock:
            self.history[signature] = message
        usage = result.get("usage") or {}
        completion = usage.get("completion_tokens")
        thoughts = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ) or 0
        native_usage = {}
        for dest, value in {
            "promptTokenCount": usage.get("prompt_tokens"),
            "candidatesTokenCount": max(0, completion - thoughts)
            if completion is not None
            else None,
            "thoughtsTokenCount": thoughts,
            "cachedContentTokenCount": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "totalTokenCount": usage.get("total_tokens"),
        }.items():
            if value is not None:
                native_usage[dest] = value
        finish = {
            "stop": "STOP",
            "tool_calls": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
        }.get(choice.get("finish_reason"))
        if finish is None:
            raise ValueError("Unsupported upstream finish reason")
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": finish,
                    "index": 0,
                }
            ],
            "usageMetadata": native_usage,
            "modelVersion": self.cli_model,
            "responseId": result.get("id"),
        }

    def populate_context(self, context):
        # Count provider-reported calls, including native CLI helper requests.
        with self.lock:
            usages = [record["usage"] for record in self.records]

        def total(key):
            values = [u.get(key) for u in usages]
            return (
                sum(values)
                if values
                and all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    and v >= 0
                    for v in values
                )
                else None
            )

        context.n_input_tokens = total("prompt_tokens")
        context.n_output_tokens = total("completion_tokens")
        cached = [
            (u.get("prompt_tokens_details") or {}).get("cached_tokens") for u in usages
        ]
        context.n_cache_tokens = (
            sum(cached) if cached and all(isinstance(v, int) for v in cached) else None
        )
        context.cost_usd = total("cost")
        context.metadata = {
            **(context.metadata or {}),
            "cost_source": "openrouter_usage",
            "transport": "gemini-openrouter-streamed-upstream",
            "helper_model_policy": "selected_model",
            "model_calls": len(usages),
        }

    def _handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            @override
            def setup(self):
                super().setup()
                self.connection.settimeout(30)

            @override
            def log_message(self, format: str, *args: Any):
                pass

            def reply(self, code, body, content_type="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.connection.settimeout(390)
                target = urlsplit(self.path)
                path = unquote(target.path)
                endpoint = re.fullmatch(
                    r"/v1beta/models/(gemini-[A-Za-z0-9_.-]+):(generateContent|streamGenerateContent)",
                    path,
                )
                if not endpoint or target.query not in {"", "alt=sse"}:
                    self.reply(
                        404,
                        b'{"error":{"code":404,"message":"Unsupported Gemini endpoint"}}',
                    )
                    return
                cli_requested_model = endpoint.group(1)
                auth = self.headers.get("x-goog-api-key", "")
                if not hmac.compare_digest(auth.encode(), bridge.token.encode()):
                    self.reply(
                        403,
                        b'{"error":{"code":403,"message":"Invalid bridge credential"}}',
                    )
                    return
                try:
                    lengths = self.headers.get_all("Content-Length", [])
                    if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                        raise ValueError("Invalid body framing")
                    length = int(lengths[0])
                    if not 0 < length <= MAX_BODY:
                        raise ValueError("Invalid body size")
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Truncated request")
                    request = bridge.request(json.loads(raw))
                except (ValueError, TypeError, KeyError):
                    self.reply(
                        400,
                        b'{"error":{"code":400,"message":"Unsupported Gemini request; see transport documentation"}}',
                    )
                    return
                streaming = path.endswith(":streamGenerateContent")
                headers_sent = False
                accumulator = _OpenRouterStream()
                recorded = False

                def keepalive():
                    # Blank lines are legal SSE and JSON whitespace. The pinned
                    # Google SDK accepts whitespace before data; colon comments
                    # would block its anchored SSE parser.
                    self.wfile.write(b"\n")
                    self.wfile.flush()

                def native_headers():
                    nonlocal headers_sent
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "text/event-stream" if streaming else "application/json",
                    )
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.flush()
                    headers_sent = True
                    self.close_connection = True

                request.update(stream=True, stream_options={"include_usage": True})
                try:
                    with bridge.client.stream(
                        "POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=request,
                        headers={
                            "Authorization": "Bearer " + bridge.key,
                            "Content-Type": "application/json",
                        },
                    ) as result:
                        if result.status_code != 200:
                            self.reply(
                                result.status_code,
                                json.dumps(
                                    {
                                        "error": {
                                            "code": result.status_code,
                                            "message": "OpenRouter request failed",
                                        }
                                    }
                                ).encode(),
                            )
                            return
                        # OpenRouter stream headers arrive before generation is
                        # complete. Forward them now, not after a long completion.
                        native_headers()
                        if "text/event-stream" not in result.headers.get(
                            "Content-Type", ""
                        ):
                            # Some compatible servers return a complete JSON
                            # response even when asked to stream (also fixtures).
                            result.read()
                            complete = result.json()
                        else:
                            data = []
                            done = False
                            for line in result.iter_lines():
                                if line.startswith("data:"):
                                    data.append(line[5:].lstrip())
                                elif not line:
                                    if data:
                                        payload = "\n".join(data)
                                        data = []
                                        if payload == "[DONE]":
                                            done = True
                                            break
                                        accumulator.add(json.loads(payload))
                                    keepalive()
                                elif line.startswith(":"):
                                    keepalive()
                            if data and not done:
                                payload = "\n".join(data)
                                if payload != "[DONE]":
                                    accumulator.add(json.loads(payload))
                            complete = accumulator.complete()
                        # response() journals returned usage before conversion.
                        recorded = True
                        native = bridge.response(
                            complete, cli_requested_model=cli_requested_model
                        )
                        encoded = json.dumps(native, allow_nan=False).encode()
                        self.wfile.write(
                            b"data: " + encoded + b"\n\n" if streaming else encoded
                        )
                        self.wfile.flush()
                except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError):
                    if accumulator.result.get("id") and not recorded:
                        bridge._record_usage(
                            accumulator.result, cli_requested_model, complete=False
                        )
                    error = b'{"error":{"code":502,"status":"UNAVAILABLE","message":"OpenRouter transport failed"}}'
                    try:
                        if headers_sent:
                            # Google's SDK recognizes raw JSON errors mid-stream.
                            # Never emit an invented candidate or partial tool call.
                            self.wfile.write(error)
                            self.wfile.flush()
                        else:
                            self.reply(502, error)
                    except OSError:
                        pass

        return Handler
