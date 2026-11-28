"""Opt-in protection against accidental broad process cleanup by Mini tools.

This is an operational guard for observed cleanup mistakes, not a security
boundary for hostile shell/Python programs. Ordinary explicit-PID cleanup is
left available. Native actions and results are retained by upstream Mini.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

_BENCHMARK_STATUS = Path("/run/picbench/benchmark-status.json")
_TRUSTED_STATUS_UID = 0
WORKFLOW_POLICY = "bounded_commands_full_budget_v1"

_NAME_KILL = re.compile(
    r"(?m)(?:^|[;&|\n])\s*(?:(?:then|do|exec|command|sudo)\s+)*"
    r"(?:[\w./-]+/)?(?:pkill|killall)(?=\s|$)"
)
_PYTHON_NAME_KILL = re.compile(
    r"(?:os\.system|subprocess\.(?:run|Popen|call|check_call|check_output))"
    r"\s*\(\s*(?:\[\s*)?['\"](?:[\w./-]+/)?(?:pkill|killall)(?:['\"]|\s)"
)
_ENUMERATION = re.compile(
    r"/proc(?:/|\b)|\b(?:pgrep|ps)\s|\b(?:process_iter|pids)\s*\("
)
_SIGNAL = re.compile(
    r"\b(?:kill|killpg|killpgid)\b|\.\s*(?:terminate|send_signal)\s*\("
)
_GROUP_SIGNAL = re.compile(
    r"(?m)(?:^|[;&|\n])\s*(?:command\s+)?kill\s+"
    r"(?:(?:-s|--signal)\s+(?:\w+)\s+|-\w+\s+)?(?:--\s+)?(?:0|-[1-9]\d*)(?=\s|$)"
)
_GUARD_NOTICE = (
    "Process cleanup was blocked because it selects processes by name, command "
    "line, or a broad process group. Those selectors can also match the agent "
    "or its tool shell. Save the positive PID when starting your own child "
    "process, terminate only that recorded child PID, and wait for it to exit. "
    "Do not use pkill/killall or scan /proc, ps, pgrep or process_iter to kill "
    "matching commands. The command was not executed."
)


def process_cleanup_reason(command: str) -> str | None:
    """Classify the observed unsafe cleanup shapes without executing anything."""
    if _NAME_KILL.search(command) or _PYTHON_NAME_KILL.search(command):
        return "process_name_cleanup"
    if _ENUMERATION.search(command) and _SIGNAL.search(command):
        return "enumerated_process_cleanup"
    if _GROUP_SIGNAL.search(command):
        return "broad_process_group_cleanup"
    return None


def workflow_command_kind(command: str) -> str:
    """Recognize a literal public probe invocation, rather than a mention.

    This is a conservative operational classifier, not a hostile-shell security
    boundary. Heredocs, substitutions and shell functions receive the ordinary
    limit. The public clients accept no positional/options arguments.
    """
    if len(command) > 1048576 or any(marker in command for marker in ("$(", "`")):
        return "ordinary_command"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return "ordinary_command"
    if any("<<" in token or token in {"(", ")", "()"} for token in tokens):
        return "ordinary_command"
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {";", "&&", "||", "|", "&", "\n"}:
            segments.append([])
        else:
            segments[-1].append(token)
    commands: list[list[str]] = []
    for segment in segments:
        argv: list[str] = []
        index = 0
        while index < len(segment):
            token = segment[index]
            if token in {"<", ">", ">>", ">|", "<&", ">&", "<>", "&>"}:
                if index + 1 >= len(segment):
                    return "ordinary_command"
                if argv and argv[-1].isdigit():
                    argv.pop()
                index += 2
                continue
            argv.append(token)
            index += 1
        while argv and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[0], re.S):
            argv.pop(0)
        if argv and argv[0] in {"command", "exec"}:
            argv.pop(0)
        if argv and argv[0] == "env":
            argv.pop(0)
            if argv and argv[0] == "--":
                argv.pop(0)
            while argv and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[0], re.S):
                argv.pop(0)
        if (len(argv) >= 3 and argv[0] == "timeout"
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[smhd]?", argv[1])):
            argv = argv[2:]
        if argv:
            commands.append(argv)
    probes = [index for index, argv in enumerate(commands)
              if len(argv) == 1 and argv[0] in {"run_probe", "/usr/local/bin/run_probe"}]
    if not probes:
        return "ordinary_command"
    if len(probes) == 1:
        if len(commands) == 1:
            return "probe_command"
        if (len(commands) == 2 and probes == [1] and len(commands[0]) == 2
                and commands[0][0] == "cd" and not commands[0][1].startswith("-")
                and not any(marker in commands[0][1] for marker in ("$", "*", "?", "[", "~"))):
            return "probe_command"
    return "mixed_probe_blocked"


def recognized_probe_command(command: str) -> bool:
    return workflow_command_kind(command) == "probe_command"


class GuardedLocalEnvironmentConfig(LocalEnvironmentConfig):
    ordinary_command_timeout_seconds: int | None = Field(default=None, strict=True, ge=1, le=300)
    early_submission_remaining_seconds: int | None = Field(default=None, strict=True, ge=1, le=60)


class GuardedLocalEnvironment(LocalEnvironment):
    """LocalEnvironment with an explicit, native-recorded cleanup safeguard."""

    def __init__(self, **kwargs):
        super().__init__(config_class=GuardedLocalEnvironmentConfig, **kwargs)
        self._process_guard_lock = threading.Lock()
        self._process_guard_command_count = 0
        self._process_guard_blocked: list[dict[str, Any]] = []
        self._workflow_events: list[dict[str, Any]] = []
        self._workflow_local = threading.local()
        self._workflow_deadline_monotonic: float | None = None
        self._workflow_deadline_text: str | None = None

    def _workflow_active(self) -> bool:
        return (self.config.ordinary_command_timeout_seconds is not None
                or self.config.early_submission_remaining_seconds is not None)

    def _workflow_clock(self) -> dict | None:
        """Read the existing immutable phase clock and pin its earliest end."""
        try:
            parent = _BENCHMARK_STATUS.parent.lstat()
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != _TRUSTED_STATUS_UID
                    or parent.st_mode & 0o022):
                return None
            descriptor = os.open(_BENCHMARK_STATUS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                before = os.fstat(handle.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != _TRUSTED_STATUS_UID
                        or before.st_nlink != 1 or before.st_mode & 0o222
                        or not 0 < before.st_size <= 16384):
                    return None
                raw = handle.read(16385)
                after = os.fstat(handle.fileno())
            if (len(raw.encode("utf-8")) > 16384
                    or any(getattr(before, key) != getattr(after, key) for key in
                           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))):
                return None
            status = json.loads(raw)
            if (not isinstance(status, dict) or type(status.get("schema_version")) is not int
                    or status.get("schema_version") != 1 or status.get("source") != "harbor_agent_phase"):
                return None
            timing = status.get("timing")
            text = timing.get("deadline_at_utc") if isinstance(timing, dict) else None
            if not isinstance(text, str):
                return None
            deadline = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                return None
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            candidate = time.monotonic() + remaining
            if not math.isfinite(candidate):
                return None
            with self._process_guard_lock:
                if self._workflow_deadline_monotonic is None or candidate < self._workflow_deadline_monotonic:
                    self._workflow_deadline_monotonic = candidate
                    self._workflow_deadline_text = text
                remaining = max(0.0, self._workflow_deadline_monotonic - time.monotonic())
                pinned_text = self._workflow_deadline_text
            return {"trusted_remaining_seconds": remaining, "trusted_deadline_at_utc": pinned_text,
                    "trusted_clock_source": "harbor_agent_phase",
                    "trusted_status_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            return None

    def _check_finished(self, output: dict):
        event = getattr(self._workflow_local, "event", None)
        if event is not None:
            event.update(returncode=output.get("returncode"),
                         timeout_observed=output.get("extra", {}).get("exception_type") == "TimeoutExpired")
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        requested = bool(lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                         and output.get("returncode") == 0)
        if not requested:
            return super()._check_finished(output)
        allowance = self.config.early_submission_remaining_seconds
        if allowance is None:
            if event is not None:
                event["submission_decision"] = "allowed_guard_disabled"
            return super()._check_finished(output)
        clock = self._workflow_clock()
        if event is not None:
            event.update(clock or {"trusted_remaining_seconds": None, "trusted_deadline_at_utc": None,
                                   "trusted_clock_source": None, "trusted_status_sha256": None})
        if clock is not None and clock["trusted_remaining_seconds"] <= allowance:
            if event is not None:
                event["submission_decision"] = "allowed_near_deadline"
            return super()._check_finished(output)
        decision = "blocked_clock_unavailable" if clock is None else "blocked_early"
        if event is not None:
            event.update(submission_decision=decision, returncode=125)
        if clock is None:
            notice = "Final submission was held because the trusted benchmark clock is unavailable. "
        else:
            notice = (f"Final submission was held because the original learner clock has "
                      f"{clock['trusted_remaining_seconds']:.1f} seconds remaining. ")
        output.update(output=notice + f"This run permits final submission only within its last {allowance} seconds. "
                      "Continue working in the existing task workspace. The command already executed; "
                      "do not repeat any side effects. The original deadline was not extended.",
                      returncode=125, exception_info="")
        output.setdefault("extra", {})["picbench_workflow_guard"] = {"submission_decision": decision}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        if not isinstance(command, str):
            raise ValueError("Guarded local environment requires a string command")
        with self._process_guard_lock:
            self._process_guard_command_count += 1
            command_index = self._process_guard_command_count
        active = self._workflow_active()
        event = None
        started = time.monotonic()
        if active:
            if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("Workflow guard requires a positive finite explicit timeout")
            selected = min(self.config.timeout, timeout if timeout is not None else self.config.timeout)
            kind = workflow_command_kind(command)
            probe = kind == "probe_command"
            if not probe and self.config.ordinary_command_timeout_seconds is not None:
                selected = min(selected, self.config.ordinary_command_timeout_seconds)
            event = {"command_index": command_index, "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                     "kind": kind,
                     "configured_timeout_seconds": self.config.timeout, "explicit_timeout_seconds": timeout,
                     "selected_timeout_seconds": selected, "executed": True, "returncode": None,
                     "timeout_observed": False, "submission_decision": "not_requested",
                     "started_at_utc": datetime.now(timezone.utc).isoformat()}
            with self._process_guard_lock:
                self._workflow_events.append(event)
        reason = process_cleanup_reason(command)
        if reason is not None:
            blocked = {
                "command_index": command_index,
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "reason": reason,
                "returncode": 126,
                "executed": False,
            }
            with self._process_guard_lock:
                self._process_guard_blocked.append(blocked)
            if event is not None:
                event.update(kind="cleanup_blocked", selected_timeout_seconds=None, executed=False,
                             returncode=126, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                             elapsed_seconds=max(0.0, time.monotonic() - started))
            return {
                "output": _GUARD_NOTICE,
                "returncode": 126,
                "exception_info": "",
                "extra": {"picbench_process_guard": dict(blocked)},
            }
        if event is not None and event["kind"] == "mixed_probe_blocked":
            event.update(selected_timeout_seconds=None, executed=False, returncode=126,
                         finished_at_utc=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=max(0.0, time.monotonic() - started))
            return {"output": "A probe was combined with other work in this action. "
                    "Invoke the public run_probe command separately (an optional literal cd prefix and "
                    "output redirection are supported). This preserves its long client allowance while "
                    "ordinary commands retain their configured limit. No part of this action executed.",
                    "returncode": 126, "exception_info": "",
                    "extra": {"picbench_workflow_guard": {"reason": "mixed_probe_blocked"}}}
        previous = getattr(self._workflow_local, "event", None)
        self._workflow_local.event = event
        try:
            # Pinned Mini's runner starts a new process session and kills its
            # whole group on timeout. Delegation preserves that cleanup.
            return super().execute(action, cwd, timeout=event["selected_timeout_seconds"] if event is not None else timeout)
        finally:
            self._workflow_local.event = previous
            if event is not None:
                with self._process_guard_lock:
                    event.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                                 elapsed_seconds=max(0.0, time.monotonic() - started))

    def serialize(self) -> dict:
        value = super().serialize()
        with self._process_guard_lock:
            value["info"]["picbench_process_guard"] = {
                "schema_version": 1,
                "policy": "explicit_pid_cleanup_v1",
                "active": True,
                "command_count": self._process_guard_command_count,
                "blocked_count": len(self._process_guard_blocked),
                "blocked_commands": [dict(event) for event in sorted(
                    self._process_guard_blocked, key=lambda item: item["command_index"])],
                "scope": "accidental_cleanup_safeguard_not_security_boundary",
            }
            if self._workflow_active():
                value["info"]["picbench_workflow_guard"] = {
                    "schema_version": 1, "policy": WORKFLOW_POLICY, "active": True,
                    "ordinary_command_timeout_seconds": self.config.ordinary_command_timeout_seconds,
                    "early_submission_remaining_seconds": self.config.early_submission_remaining_seconds,
                    "command_count": self._process_guard_command_count,
                    "events": [dict(event) for event in sorted(self._workflow_events, key=lambda item: item["command_index"])],
                }
        return value
