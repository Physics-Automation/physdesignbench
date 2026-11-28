"""Trusted benchmark status initialization for Harbor agent phases."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)
BENCHMARK_STATUS_DIR = "/run/picbench"
BENCHMARK_STATUS_PATH = f"{BENCHMARK_STATUS_DIR}/benchmark-status.json"
BENCHMARK_STATUS_OWNER_UID = 0
BENCHMARK_STATUS_OWNER_GID = 0
_ENV_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_SHELL_VARIABLE_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_BRACED_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}]+\}\}")
_ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<[^<>]+>")
_MASK_PATTERN = re.compile(r"(?:\*{3,}|\u2022{3,})")
_PLACEHOLDER_WORDS = frozenset(
    {
        "changeme",
        "masked",
        "placeholder",
        "redacted",
        "replace-me",
        "replace_me",
    }
)


def _usable_value(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""

    value = raw.strip()
    normalized = value.casefold()
    if (
        not value
        or any(character in value for character in ("\x00", "\r", "\n"))
        or _ENV_TEMPLATE_PATTERN.fullmatch(value)
        or _SHELL_VARIABLE_PATTERN.fullmatch(value)
        or _BRACED_PLACEHOLDER_PATTERN.fullmatch(value)
        or _ANGLE_PLACEHOLDER_PATTERN.fullmatch(value)
        or _MASK_PATTERN.search(value)
        or normalized in _PLACEHOLDER_WORDS
    ):
        return ""
    return value


def _mapping_from_verifier(owner: Any) -> Mapping[str, Any]:
    verifier = getattr(owner, "verifier", None)
    candidate = getattr(verifier, "env", {}) or {}
    return candidate if isinstance(candidate, Mapping) else {}


def _resolve_env_template(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw

    match = _ENV_TEMPLATE_PATTERN.fullmatch(raw.strip())
    if not match:
        return raw

    variable_name, default = match.groups()
    if variable_name in os.environ:
        return os.environ[variable_name]
    return default or ""


def _initial_benchmark_status(
    trial: Any,
    timeout_sec: float | int | None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    timeout_value = float(timeout_sec) if timeout_sec is not None else None
    agent = getattr(trial, "agent", None)
    if timeout_value is not None and getattr(agent, "_continue_until_timeout", False) is True:
        # Persistent Codex stops working before the outer Harbor deadline to
        # copy logs and shut down. Do not advertise that reserve as usable time.
        reserve = getattr(agent, "_shutdown_grace_sec", 0.0)
        if isinstance(reserve, (int, float)) and not isinstance(reserve, bool) and math.isfinite(reserve):
            timeout_value = max(0.0, timeout_value - max(0.0, reserve))
    deadline_at = started_at + timedelta(seconds=timeout_value) if timeout_value is not None else None
    started_iso = started_at.isoformat().replace("+00:00", "Z")
    deadline_iso = deadline_at.isoformat().replace("+00:00", "Z") if deadline_at is not None else None

    return {
        "schema_version": 1,
        "generated_at": started_iso,
        "source": "harbor_agent_phase",
        "timing": {
            "generated_at": started_iso,
            "started_at_utc": started_iso,
            "deadline_at_utc": deadline_iso,
            "time_limit_seconds": timeout_value,
            "elapsed_seconds": 0,
            "remaining_seconds": timeout_value,
        },
    }


def _initialize_benchmark_status_command(status: Mapping[str, Any]) -> str:
    payload = json.dumps(status, sort_keys=True, separators=(",", ":"))
    return f"""
python3 - <<'PY'
import json
import os
import tempfile
from pathlib import Path

status_dir = Path({BENCHMARK_STATUS_DIR!r})
status_file = Path({BENCHMARK_STATUS_PATH!r})
payload = json.loads({payload!r})

if status_dir.is_symlink():
    raise SystemExit("refusing to write through symlinked benchmark status directory")
status_dir.mkdir(parents=True, exist_ok=True)
os.chown(status_dir, {BENCHMARK_STATUS_OWNER_UID}, {BENCHMARK_STATUS_OWNER_GID})
status_dir.chmod(0o700)
if status_file.is_symlink():
    raise SystemExit("refusing to overwrite symlinked benchmark status")
if status_file.exists() and not status_file.is_file():
    raise SystemExit("refusing to overwrite non-file benchmark status")

descriptor, temporary_name = tempfile.mkstemp(
    dir=status_dir,
    prefix=".benchmark-status.",
    suffix=".tmp",
)
temporary = Path(temporary_name)
try:
    os.fchown(descriptor, {BENCHMARK_STATUS_OWNER_UID}, {BENCHMARK_STATUS_OWNER_GID})
    os.fchmod(descriptor, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(status_file)
    os.chown(status_file, {BENCHMARK_STATUS_OWNER_UID}, {BENCHMARK_STATUS_OWNER_GID})
    status_file.chmod(0o444)
finally:
    temporary.unlink(missing_ok=True)
status_dir.chmod(0o555)
PY
"""


async def _initialize_benchmark_status(
    trial: Any,
    timeout_sec: float | int | None,
) -> None:
    environment = getattr(trial, "agent_environment", None)
    if environment is None or not callable(getattr(environment, "exec", None)):
        LOGGER.warning("Could not initialize benchmark status timing: Harbor agent environment does not support exec.")
        return

    try:
        result = await environment.exec(
            command=_initialize_benchmark_status_command(_initial_benchmark_status(trial, timeout_sec)),
            user="root",
            timeout_sec=30,
        )
    except Exception:
        LOGGER.warning("Could not initialize benchmark status timing.")
        return
    if getattr(result, "return_code", 0) != 0:
        LOGGER.warning(
            "Could not initialize benchmark status timing (exit code %s).",
            getattr(result, "return_code", "unknown"),
        )
