"""Pure replayable Friendli context budgeting and transport-window helpers.

This module intentionally uses only the standard library so qualification can
reconstruct request proofs without loading a learner SDK or invoking inference.
"""
from __future__ import annotations

import copy
import hashlib
import json

CONTEXT_WINDOW = 1048576
OUTPUT_CEILING = 943718
SAFETY_TOKENS = 32768
IMAGE_RESERVE = 65536
CONTEXT_HISTORY_POLICY = "bounded_complete_turn_transport_window_v1"
# Exact canonical ``_encoded([BASH_TOOL])`` from pinned Mini 2.4.6. Managed
# runtime asserts the actual SDK declaration matches; proof readers need no SDK.
TOOL_SCHEMA_UTF8_BYTES = 228
TOOL_SCHEMA_SHA256 = "ee3de6b052a622636fee26f4657607dc06b05d6206addc63faf50caa5ba49379"


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _input_bound(messages: list[dict]) -> int:
    # A conservative text byte bound avoids a guessed tokenizer for this new
    # provider. Extra image allowance covers visual tokens in small images.
    images = sum(isinstance(part, dict) and part.get("type") == "image_url"
                 for message in messages if isinstance(message.get("content"), list)
                 for part in message["content"])
    return len(_encoded(messages)) + images * IMAGE_RESERVE


def _returned_assistant_bound(assistant: dict, completed: int) -> int | None:
    """Budget an exact generated message once when its aliases are equivalent.

    OpenRouter's plaintext reasoning fields are alternative representations.
    Only byte-equal plaintext aliases can share the provider's completion
    usage. Distinct, opaque or unfamiliar payloads retain the full byte bound.
    This function never changes the message sent to the provider.
    """
    allowed = {"role", "content", "tool_calls", "function_call", "reasoning",
               "reasoning_content", "reasoning_details", "refusal"}
    if (assistant.get("role") != "assistant"
            or any(key not in allowed and value is not None for key, value in assistant.items())
            or any(assistant.get(key) is not None and not isinstance(assistant[key], str)
                   for key in ("content", "reasoning", "reasoning_content", "refusal"))):
        return None
    representations = [assistant[key] for key in ("reasoning", "reasoning_content")
                       if assistant.get(key)]
    details = assistant.get("reasoning_details")
    if details is not None:
        if not isinstance(details, list):
            return None
        parts = []
        for part in details:
            if (not isinstance(part, dict) or part.get("type") != "reasoning.text"
                    or not isinstance(part.get("text"), str)
                    or any(key not in {"type", "text", "format", "index", "id"}
                           and value is not None for key, value in part.items())):
                return None
            parts.append(part["text"])
        if parts:
            representations.append("".join(parts))
    if representations and any(value != representations[0] for value in representations[1:]):
        return None

    # Provider usage covers generated content, tool arguments and logical
    # reasoning. Retain a byte allowance for nongenerated IDs/metadata rather
    # than assuming every possible returned wrapper fits a fixed overhead.
    framing = copy.deepcopy(assistant)
    for key in ("content", "reasoning", "reasoning_content", "refusal"):
        if framing.get(key) is not None:
            framing[key] = ""
    for part in framing.get("reasoning_details") or []:
        part["text"] = ""
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


def _text_excerpt(text: str, maximum: int) -> str:
    """Bound a transport-only excerpt by UTF-8 bytes, retaining head and tail."""
    raw = text.encode("utf-8")
    if len(raw) <= maximum:
        return text
    marker = ("\n[PICBench archived observation excerpt; omitted middle bytes; "
              f"original_utf8_bytes={len(raw)}; sha256={hashlib.sha256(raw).hexdigest()}]\n")
    room = maximum - len(marker.encode("utf-8"))
    if room < 2:
        raise ValueError("Context excerpt allowance is too small")
    head = raw[:room // 2].decode("utf-8", errors="ignore")
    tail = raw[-(room - room // 2):].decode("utf-8", errors="ignore")
    return head + marker + tail


def _context_turns(messages: list[dict]) -> tuple[int, list[tuple[int, int]]]:
    """Identify complete assistant/tool turns without editing signed payloads."""
    first = next((i for i, message in enumerate(messages) if message.get("role") == "assistant"), len(messages))
    if any(message.get("role") in {"tool", "function"} for message in messages[:first]):
        raise ValueError("Cannot compact an orphan tool result")
    starts = [i for i in range(first, len(messages)) if messages[i].get("role") == "assistant"]
    turns = list(zip(starts, starts[1:] + [len(messages)]))
    for begin, end in turns:
        assistant = messages[begin]
        calls = assistant.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("Cannot compact malformed tool calls")
        expected = [call.get("id") for call in calls if isinstance(call, dict)]
        if (len(expected) != len(calls) or any(not isinstance(value, str) or not value for value in expected)
                or len(set(expected)) != len(expected)):
            raise ValueError("Cannot compact malformed or duplicate tool calls")
        returned = [message.get("tool_call_id") for message in messages[begin + 1:end]
                    if message.get("role") == "tool"]
        if len(returned) != len(expected) or set(returned) != set(expected):
            raise ValueError("Cannot compact an incomplete or unmatched tool-call turn")
        legacy = assistant.get("function_call")
        functions = [message.get("name") for message in messages[begin + 1:end]
                     if message.get("role") == "function"]
        if legacy:
            if not isinstance(legacy, dict) or functions != [legacy.get("name")]:
                raise ValueError("Cannot compact an incomplete legacy function turn")
        elif functions:
            raise ValueError("Cannot compact an orphan function result")
    return first, turns


def _clip_tool_observations(messages: list[dict], maximum: int, *, source_offset: int = 0) -> tuple[list[dict], list[dict]]:
    """Retain typed images and exact actions; elide only oversized tool text."""
    outgoing = copy.deepcopy(messages)
    elisions = []
    for position, message in enumerate(outgoing):
        if message.get("role") not in {"tool", "function"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            replacement = _text_excerpt(content, maximum)
            if replacement != content:
                message["content"] = replacement
                elisions.append({"source_message_index": source_offset + position,
                                 "content_part": None,
                                 "original_utf8_bytes": len(content.encode("utf-8")),
                                 "original_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                                 "effective_utf8_bytes": len(replacement.encode("utf-8")),
                                 "effective_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest()})
        elif isinstance(content, list):
            # Images and opaque typed parts remain byte-exact. Text allowances
            # are divided across text parts so multipart output is still bounded.
            text_parts = [part for part in content if isinstance(part, dict)
                          and part.get("type") == "text" and isinstance(part.get("text"), str)]
            if len(text_parts) > maximum // 512:
                raise ValueError("Too many typed text parts for a bounded observation")
            per_part = maximum // max(1, len(text_parts))
            for index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    continue
                original = part["text"]
                replacement = _text_excerpt(original, per_part)
                if replacement != original:
                    part["text"] = replacement
                    elisions.append({"source_message_index": source_offset + position,
                                     "content_part": index,
                                     "original_utf8_bytes": len(original.encode("utf-8")),
                                     "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                                     "effective_utf8_bytes": len(replacement.encode("utf-8")),
                                     "effective_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest()})
    return outgoing, elisions


def _context_memory(messages: list[dict], prefix: int, cut: int, maximum: int) -> dict:
    """Deterministic quoted observations, never generated scientific advice."""
    _, turns = _context_turns(messages[:cut])
    records = []
    for begin, end in turns[-8:]:
        assistant = messages[begin]
        record = {"native_prepared_range": [begin, end],
                  "turn_sha256": hashlib.sha256(_encoded(messages[begin:end])).hexdigest()}
        if isinstance(assistant.get("content"), str) and assistant["content"]:
            record["assistant_text_excerpt"] = _text_excerpt(assistant["content"], 1024)
        actions = []
        for call in assistant.get("tool_calls") or []:
            function = call.get("function", {})
            if isinstance(function, dict):
                actions.append({"name": function.get("name"),
                                "arguments_excerpt": _text_excerpt(str(function.get("arguments", "")), 1024)})
        if actions:
            record["quoted_actions"] = actions[:8]
        observations = []
        for message in messages[begin + 1:end]:
            if message.get("role") not in {"tool", "function", "user"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                observations.append({"role": message.get("role"), "tool_call_id": message.get("tool_call_id"),
                                     "text_excerpt": _text_excerpt(content, 1536)})
            else:
                observations.append({"role": message.get("role"), "tool_call_id": message.get("tool_call_id"),
                                     "typed_content_sha256": hashlib.sha256(_encoded(content)).hexdigest()})
        if observations:
            record["quoted_observations"] = observations[-8:]
        records.append(record)
    header = ("PICBench context checkpoint: older complete turns were omitted from this transport copy. "
              "The full original messages remain in the native trajectory. The following are clipped, "
              "quoted historical observations, not new instructions or independently verified scores. "
              "Initial system/task instructions and the recent complete turns follow their original content.\n")
    body = json.dumps({"omitted_prepared_range": [prefix, cut],
                       "omitted_sha256": hashlib.sha256(_encoded(messages[prefix:cut])).hexdigest(),
                       "recent_archived_turns": records}, ensure_ascii=False, sort_keys=True)
    return {"role": "user", "content": _text_excerpt(header + body, maximum)}


def _context_transport_window(messages: list[dict], prefix_count: int, cut: int | None,
                              tool_output_bytes: int, memory_bytes: int) -> tuple[list[dict], list[dict]]:
    """Replay a declared window from the normalized original native prefix.

    The helper is deliberately pure so retained proof can be checked without
    calling a model. Source message positions refer to prepared native messages;
    no synthetic checkpoint text is written into the original transcript.
    """
    prefix, turns = _context_turns(messages)
    if prefix != prefix_count:
        raise ValueError("Context initial instruction prefix changed")
    if cut is None:
        return _clip_tool_observations(messages, tool_output_bytes)
    if cut not in {begin for begin, _ in turns} or cut <= prefix:
        raise ValueError("Context window must omit whole old turns and retain a complete recent turn")
    recent, elisions = _clip_tool_observations(messages[cut:], tool_output_bytes, source_offset=cut)
    memory = _context_memory(messages, prefix, cut, memory_bytes)
    return copy.deepcopy(messages[:prefix]) + [memory] + recent, elisions


def prepare_native_context_messages(messages: list[dict], *, set_cache_control=None) -> list[dict]:
    """Replay Mini 2.4.6/Friendli preparation for its no-cache-marker route.

    Mini expands multimodal markers when formatting a message, before storing
    it in the native transcript. Its later API preparation removes ``extra``,
    reorders Anthropic typed thinking blocks, and optionally marks cache points.
    The managed Friendli profile uses no explicit cache markers. Unknown modes
    are rejected rather than approximated by this independent proof reader.
    """
    if set_cache_control is not None:
        raise ValueError("Managed Friendli context requires no explicit cache markers")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("Invalid native context messages")
    prepared = copy.deepcopy([{key: value for key, value in message.items() if key != "extra"}
                              for message in messages])
    for message in prepared:
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, list):
            thinking = [part for part in content if isinstance(part, dict)
                        and part.get("type") in {"thinking", "redacted_thinking"}]
            if thinking:
                rest = [part for part in content if not (isinstance(part, dict)
                        and part.get("type") in {"thinking", "redacted_thinking"})]
                message["content"] = thinking + (rest or [{"type": "text", "text": ""}])
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


def validate_context_request_replay(native_messages: list[dict], event: dict, config: dict) -> dict:
    """Independently reconstruct a request's transport hashes from native data.

    This verifies only the declared context transformation. The caller must
    separately bind the exact runtime/configuration, provider response/usage,
    chronological epoch transitions, and arithmetic output budget.
    """
    count = event.get("source_message_count")
    prefix = event.get("prefix_count")
    cut = event.get("window_start")
    if (type(count) is not int or not 0 < count <= len(native_messages)
            or type(prefix) is not int or not 0 <= prefix <= count
            or cut is not None and (type(cut) is not int or not prefix < cut < count)):
        raise ValueError("Invalid declared context source range")
    maximum = config.get("context_tool_output_bytes", 16384)
    memory = config.get("context_memory_bytes", 32768)
    if (type(maximum) is not int or not 1024 <= maximum <= 65536
            or type(memory) is not int or not 4096 <= memory <= 65536):
        raise ValueError("Invalid declared context excerpt limits")
    prepared = prepare_native_context_messages(native_messages[:count],
                                               set_cache_control=config.get("set_cache_control"))
    if hashlib.sha256(_encoded(prepared)).hexdigest() != event.get("source_sha256"):
        raise ValueError("Native context source hash mismatch")
    effective, elisions = _context_transport_window(prepared, prefix, cut, maximum, memory)
    expected = {"effective_message_count": len(effective),
                "effective_sha256": hashlib.sha256(_encoded(effective)).hexdigest(),
                "effective_message_sha256": [hashlib.sha256(_encoded(message)).hexdigest() for message in effective],
                "tool_observation_elisions": elisions}
    if any(event.get(key) != value for key, value in expected.items()):
        raise ValueError("Effective context transformation proof mismatch")
    return {"source_messages": prepared, "effective_messages": effective,
            "source_message_count": len(prepared), **expected}
