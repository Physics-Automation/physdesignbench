#!/usr/bin/env python3
"""Private, budgeted broker for anisotropic-SWG beamsplitter evaluations."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 1024 * 1024
STATE_PATH = Path("/var/lib/picbench/anisotropic-swg-beamsplitter-v0/state/probe_state.json")
STATE_DIR = STATE_PATH.parent
STATE_LOCK = STATE_DIR / "probe.lock"
PROBE_HISTORY_PATH = STATE_DIR / "probe_history.json"
SUBMISSION_PATH = Path("/submission/design.py")
SUBMISSION_DIR = SUBMISSION_PATH.parent
RESULT_PATH = Path("/logs/artifacts/eval.json")
PAID_RECEIPT_PATH = STATE_DIR / "paid_usage_receipt.json"
PROBE_DOWNLOAD_DIR = Path("/logs/probe-downloads")
EVALUATE = Path("/opt/picbench/evaluator/bin/evaluate")
PROXY = Path("/opt/picbench/evaluator/runtime/layout_license_proxy.py")
EVALUATION_LOCK = threading.Lock()
PINNED_IMAGE_RE = re.compile(r"^[^@\s]+@(?P<digest>sha256:[0-9a-f]{64})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PROBE_HISTORY_BYTES = 256 * 1024
MAX_PUBLIC_FEEDBACK_BYTES = 8 * 1024
MAX_PUBLIC_PROBE_RESPONSE_BYTES = 64 * 1024
DEFAULT_MAX_PAID_LEARNER_PROBES = 30
DEFAULT_CUMULATIVE_FLEXCREDIT_LIMIT = 100.0
MAX_FLEXCREDIT_PER_CANDIDATE_EVALUATION = 4.0
# Protect one fresh final evaluation; probes cannot spend this allocation.
FINAL_RESERVE_FLEXCREDIT = MAX_FLEXCREDIT_PER_CANDIDATE_EVALUATION
LEARNER_THERMAL_REMAINING_ENV = "PICBENCH_LEARNER_THERMAL_REMAINING_FLEXCREDIT"
PAID_LEARNER_PROBES_REMAINING_ENV = "PICBENCH_PAID_LEARNER_PROBES_REMAINING"
PROBE_RESPONSE_FIELDS = {
    "schema_version",
    "probe_id",
    "probes_remaining",
    "submission_sha256",
    "status",
    "valid",
    "gate_pass",
    "score",
    "metric",
    "budget",
    "diagnostic_artifacts",
    "best_valid_snapshot",
    "task_feedback",
}


def _require_image_identity() -> None:
    image = os.environ.get("PICBENCH_EVALUATOR_IMAGE", "").strip()
    override = os.environ.get("PICBENCH_ALLOW_MUTABLE_IMAGES", "0").strip().lower()
    if override not in {"0", "1", "false", "true", "no", "yes"}:
        raise RuntimeError("PICBENCH_ALLOW_MUTABLE_IMAGES has an invalid value")
    match = PINNED_IMAGE_RE.fullmatch(image)
    if match is None and override not in {"1", "true", "yes"}:
        raise RuntimeError("probe service requires an evaluator image pinned by sha256 digest")
    if match is not None:
        resolved = match.group("digest")
        supplied = os.environ.get("PICBENCH_EVALUATOR_IMAGE_DIGEST", "").strip()
        if supplied and supplied != resolved:
            raise RuntimeError("evaluator image digest metadata does not match its image reference")
        os.environ["PICBENCH_EVALUATOR_IMAGE_DIGEST"] = resolved


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _probe_history_unavailable(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "trusted_probe_sidecar_state",
        "role": "diagnostic_probe_provenance",
        "score_authority": False,
        "available": False,
        "reason": reason,
        "attempts": [],
        "probes": [],
        "totals": {"accepted_probe_count": 0},
    }


def _new_probe_history() -> dict[str, Any]:
    history = _probe_history_unavailable("no_successful_probes_recorded")
    history["available"] = True
    history.pop("reason", None)
    return history


def _public_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return value if math.isfinite(float(value)) else None
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str) and len(value) <= 160:
        return value
    return None


def _bounded_history_files(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    files: dict[str, dict[str, Any]] = {}
    for name, metadata in list(value.items())[:8]:
        if not isinstance(name, str) or not isinstance(metadata, dict):
            continue
        digest = metadata.get("sha256")
        size_bytes = metadata.get("size_bytes")
        if isinstance(digest, str) and SHA256_RE.fullmatch(digest) and isinstance(size_bytes, int) and size_bytes >= 0:
            files[name[:64]] = {"sha256": digest, "size_bytes": size_bytes}
    return files


def _build_probe_history_record(
    response: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    summary = feedback.get("summary") if isinstance(feedback, dict) else {}
    if not isinstance(summary, dict):
        summary = feedback if isinstance(feedback, dict) else {}
    diagnostic_artifacts = response.get("diagnostic_artifacts")
    if not isinstance(diagnostic_artifacts, dict):
        diagnostic_artifacts = {}
    summary_keys = (
        "status",
        "valid",
        "design_valid",
        "simulation_valid",
        "evidence_valid",
        "gate_pass",
        "candidate_subruns",
        "symmetry_gate_pass",
        "input_coverage_pass",
        "official_score",
        "task_metric",
        "paper_margin",
        "estimated_flexcredit",
        "max_flexcredit_per_run",
    )
    return {
        "schema_version": 1,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_id": int(response["probe_id"]),
        "measurement_probes_remaining": int(
            response.get("measurement_probes_remaining", response.get("probes_remaining", 0))
        ),
        "submission_sha256": response["submission_sha256"],
        "manifest_sha256": response.get("manifest_sha256"),
        "evaluator_returncode": int(response.get("evaluator_returncode", 0)),
        "summary": {
            key: scalar
            for key in summary_keys
            if (scalar := _public_scalar(summary.get(key))) is not None
        },
        "thermal_budget": response.get("budget") if isinstance(response.get("budget"), dict) else {},
        "diagnostic_artifacts": {
            "available": bool(diagnostic_artifacts.get("available")),
            "files": _bounded_history_files(diagnostic_artifacts.get("files")),
        },
    }


def _validated_probe_history_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise RuntimeError("probe history record schema is invalid")
    for key in ("probe_id", "measurement_probes_remaining", "evaluator_returncode"):
        value = record.get(key)
        minimum = 1 if key == "probe_id" else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise RuntimeError("probe history record integer is invalid")
    submission_sha256 = record.get("submission_sha256")
    if not isinstance(submission_sha256, str) or SHA256_RE.fullmatch(submission_sha256) is None:
        raise RuntimeError("probe history submission hash is invalid")
    manifest_sha256 = record.get("manifest_sha256")
    if manifest_sha256 is not None and (
        not isinstance(manifest_sha256, str) or SHA256_RE.fullmatch(manifest_sha256) is None
    ):
        raise RuntimeError("probe history manifest hash is invalid")
    if not isinstance(record.get("summary"), dict) or not isinstance(record.get("thermal_budget"), dict):
        raise RuntimeError("probe history record body is invalid")
    return record


def _load_probe_history_unlocked() -> dict[str, Any]:
    if not PROBE_HISTORY_PATH.exists():
        return _new_probe_history()
    if PROBE_HISTORY_PATH.is_symlink() or PROBE_HISTORY_PATH.stat().st_size > MAX_PROBE_HISTORY_BYTES:
        raise RuntimeError("probe history file is unsafe")
    history = json.loads(PROBE_HISTORY_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(history, dict)
        or history.get("schema_version") != 1
        or history.get("source") != "trusted_probe_sidecar_state"
        or history.get("role") != "diagnostic_probe_provenance"
        or history.get("score_authority") is not False
        or history.get("available") is not True
        or not isinstance(history.get("probes"), list)
    ):
        raise RuntimeError("probe history file is malformed")
    history["probes"] = [_validated_probe_history_record(record) for record in history["probes"]]
    return history


_ATTEMPT_EVENTS = {
    "evaluator_launch_attempted": "evaluator_launch_attempted_at_utc",
    "evaluator_process_started": "evaluator_process_started_at_utc",
    "evaluator_finished": "evaluator_finished_at_utc",
    "response_write_attempted": "response_write_attempted_at_utc",
    "response_write_completed": "response_write_completed_at_utc",
}
_ATTEMPT_FIELDS = {
    "probe_id",
    "submission_sha256",
    "measurement_probes_remaining_after_allocation",
    "allocated_at_utc",
    "evaluator_launch_attempted_at_utc",
    "evaluator_process_started_at_utc",
    "evaluator_finished_at_utc",
    "timeout_observed",
    "process_returncode",
    "response_write_attempted_at_utc",
    "response_write_completed_at_utc",
    "http_status_sent",
}
_ATTEMPT_CONTEXT = threading.local()


def _attempt_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validated_probe_attempt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ATTEMPT_FIELDS:
        raise ValueError("probe attempt schema is invalid")
    if (
        not isinstance(value["probe_id"], int)
        or isinstance(value["probe_id"], bool)
        or value["probe_id"] < 1
        or not isinstance(value["submission_sha256"], str)
        or SHA256_RE.fullmatch(value["submission_sha256"]) is None
        or not isinstance(
            value["measurement_probes_remaining_after_allocation"], int
        )
        or isinstance(
            value["measurement_probes_remaining_after_allocation"], bool
        )
        or value["measurement_probes_remaining_after_allocation"] < 0
        or not isinstance(value["allocated_at_utc"], str)
        or not value["allocated_at_utc"]
        or len(value["allocated_at_utc"]) > 32
        or not isinstance(value["timeout_observed"], bool)
    ):
        raise ValueError("probe attempt fields are invalid")
    for field in (
        "evaluator_launch_attempted_at_utc",
        "evaluator_process_started_at_utc",
        "evaluator_finished_at_utc",
        "response_write_attempted_at_utc",
        "response_write_completed_at_utc",
    ):
        item = value[field]
        if item is not None and (
            not isinstance(item, str) or not item or len(item) > 32
        ):
            raise ValueError("probe attempt timestamp is invalid")
    returncode = value["process_returncode"]
    if returncode is not None and (
        not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        raise ValueError("probe attempt return code is invalid")
    http_status = value["http_status_sent"]
    if http_status is not None and (
        not isinstance(http_status, int)
        or isinstance(http_status, bool)
        or not 100 <= http_status <= 599
    ):
        raise ValueError("probe attempt HTTP status is invalid")
    return dict(value)


def _probe_history_lock_path() -> Path:
    lock_path = globals().get("LOCK_FILE", globals().get("STATE_LOCK"))
    if not isinstance(lock_path, Path):
        raise RuntimeError("probe history lock path is unavailable")
    return lock_path


def _mutate_probe_attempt(
    probe_id: int,
    *,
    submission_sha256: str | None = None,
    measurement_probes_remaining: int | None = None,
    event: str | None = None,
    process_returncode: int | None = None,
    http_status_sent: int | None = None,
) -> bool:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            _probe_history_lock_path(),
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            history = _load_probe_history_unlocked()
            raw_attempts = history.get("attempts", [])
            if not isinstance(raw_attempts, list):
                raise ValueError("probe attempt history is invalid")
            attempts = [_validated_probe_attempt(item) for item in raw_attempts]
            if len({item["probe_id"] for item in attempts}) != len(attempts):
                raise ValueError("probe attempt IDs are not unique")
            matching = [item for item in attempts if item["probe_id"] == probe_id]
            if len(matching) > 1:
                raise ValueError("probe attempt ID is duplicated")
            if matching:
                attempt = matching[0]
                if submission_sha256 is not None and (
                    attempt["submission_sha256"] != submission_sha256
                ):
                    raise ValueError("probe attempt submission hash changed")
                if measurement_probes_remaining is not None and (
                    attempt["measurement_probes_remaining_after_allocation"]
                    != measurement_probes_remaining
                ):
                    raise ValueError("probe attempt budget snapshot changed")
            else:
                if (
                    not isinstance(probe_id, int)
                    or isinstance(probe_id, bool)
                    or probe_id < 1
                    or not isinstance(submission_sha256, str)
                    or SHA256_RE.fullmatch(submission_sha256) is None
                    or not isinstance(measurement_probes_remaining, int)
                    or isinstance(measurement_probes_remaining, bool)
                    or measurement_probes_remaining < 0
                ):
                    raise ValueError("new probe attempt identity is invalid")
                if event is not None:
                    raise ValueError("probe attempt must be allocated before updates")
                attempt = {
                    "probe_id": probe_id,
                    "submission_sha256": submission_sha256,
                    "measurement_probes_remaining_after_allocation": (
                        measurement_probes_remaining
                    ),
                    "allocated_at_utc": _attempt_timestamp(),
                    "evaluator_launch_attempted_at_utc": None,
                    "evaluator_process_started_at_utc": None,
                    "evaluator_finished_at_utc": None,
                    "timeout_observed": False,
                    "process_returncode": None,
                    "response_write_attempted_at_utc": None,
                    "response_write_completed_at_utc": None,
                    "http_status_sent": None,
                }
                attempts.append(attempt)
            if event is not None:
                if event == "timeout_observed":
                    attempt["timeout_observed"] = True
                else:
                    timestamp_field = _ATTEMPT_EVENTS.get(event)
                    if timestamp_field is None:
                        raise ValueError("probe attempt event is invalid")
                    if attempt[timestamp_field] is None:
                        attempt[timestamp_field] = _attempt_timestamp()
            if process_returncode is not None:
                current = attempt["process_returncode"]
                if current is not None and current != process_returncode:
                    raise ValueError("probe attempt return code changed")
                attempt["process_returncode"] = process_returncode
                if attempt["evaluator_finished_at_utc"] is None:
                    attempt["evaluator_finished_at_utc"] = _attempt_timestamp()
            if http_status_sent is not None:
                current = attempt["http_status_sent"]
                if current is not None and current != http_status_sent:
                    raise ValueError("probe attempt HTTP status changed")
                attempt["http_status_sent"] = http_status_sent
            history["attempts"] = sorted(
                (_validated_probe_attempt(item) for item in attempts),
                key=lambda item: item["probe_id"],
            )
            payload = _json_bytes(history)
            if len(payload) > MAX_PROBE_HISTORY_BYTES:
                raise ValueError("probe history exceeds size limit")
            _atomic_write(PROBE_HISTORY_PATH, payload, mode=0o600)
            return True
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "probe_attempt_history_update_failed",
                    "exception_type": type(exc).__name__,
                    "score_authority": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return False


def _begin_probe_attempt(
    probe_id: int,
    measurement_probes_remaining: int,
    submission_sha256: str,
) -> bool:
    recorded = _mutate_probe_attempt(
        probe_id,
        submission_sha256=submission_sha256,
        measurement_probes_remaining=measurement_probes_remaining,
    )
    _ATTEMPT_CONTEXT.probe_id = probe_id
    return recorded


def _update_probe_attempt(
    probe_id: int,
    *,
    event: str,
    process_returncode: int | None = None,
    http_status_sent: int | None = None,
) -> bool:
    return _mutate_probe_attempt(
        probe_id,
        event=event,
        process_returncode=process_returncode,
        http_status_sent=http_status_sent,
    )


def _append_probe_history(record: dict[str, Any]) -> bool:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        clean_record = _validated_probe_history_record(record)
        descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            history = _load_probe_history_unlocked()
            history["probes"].append(clean_record)
            history["totals"] = {
                "accepted_probe_count": len(history["probes"]),
                "measurement_probes_remaining": clean_record["measurement_probes_remaining"],
            }
            payload = _json_bytes(history)
            if len(payload) > MAX_PROBE_HISTORY_BYTES:
                raise RuntimeError("probe history exceeds size limit")
            _atomic_write(PROBE_HISTORY_PATH, payload, mode=0o600)
            return True
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except Exception:
        return False


def _initial_probe_state(budget: int) -> dict[str, Any]:
    paid_limit = int(
        os.environ.get(
            "PICBENCH_MAX_PAID_LEARNER_PROBES",
            str(DEFAULT_MAX_PAID_LEARNER_PROBES),
        )
    )
    cost_limit = float(
        os.environ.get(
            "PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT",
            str(DEFAULT_CUMULATIVE_FLEXCREDIT_LIMIT),
        )
    )
    if (
        not 0 <= paid_limit <= DEFAULT_MAX_PAID_LEARNER_PROBES
        or not math.isfinite(cost_limit)
        or not FINAL_RESERVE_FLEXCREDIT <= cost_limit <= DEFAULT_CUMULATIVE_FLEXCREDIT_LIMIT
    ):
        raise RuntimeError("paid probe accounting limits are invalid")
    return {
        "schema_version": 1,
        "next_probe_id": 1,
        "measurement_budget": {"limit": budget, "used": 0, "remaining": budget},
        "paid_budget": {"limit": paid_limit, "used": 0, "remaining": paid_limit},
        "cost_budget": {
            "limit_flexcredit": cost_limit,
            "final_reserve_flexcredit": FINAL_RESERVE_FLEXCREDIT,
            "used_flexcredit": 0.0,
            "learner_remaining_flexcredit": cost_limit - FINAL_RESERVE_FLEXCREDIT,
        },
        "task_accounting": {
            "confirmed_paid_starts": 0,
            "uncertain_paid_start_attempts": 0,
            "accounted_probe_ids": [],
        },
    }


def _read_probe_state(budget: int) -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _initial_probe_state(budget)
    if STATE_PATH.is_symlink() or STATE_PATH.stat().st_size > 4096:
        raise RuntimeError("probe state file is unsafe")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "next_probe_id",
        "measurement_budget",
            "paid_budget",
            "cost_budget",
            "task_accounting",
    }:
        raise RuntimeError("probe state schema is invalid")
    measurement = state.get("measurement_budget")
    paid = state.get("paid_budget")
    cost = state.get("cost_budget")
    accounting = state.get("task_accounting")
    expected = _initial_probe_state(budget)
    if (
        state.get("schema_version") != 1
        or not isinstance(state.get("next_probe_id"), int)
        or isinstance(state.get("next_probe_id"), bool)
        or not isinstance(measurement, dict)
        or set(measurement) != {"limit", "used", "remaining"}
        or any(
            not isinstance(measurement.get(key), int)
            or isinstance(measurement.get(key), bool)
            for key in ("limit", "used", "remaining")
        )
        or measurement["limit"] != budget
        or not 0 <= measurement["used"] <= budget
        or measurement["remaining"] != budget - measurement["used"]
        or state["next_probe_id"] != measurement["used"] + 1
        or not isinstance(paid, dict)
        or set(paid) != {"limit", "used", "remaining"}
        or any(
            not isinstance(paid.get(key), int) or isinstance(paid.get(key), bool)
            for key in ("limit", "used", "remaining")
        )
        or paid["limit"] != expected["paid_budget"]["limit"]
        or not 0 <= paid["used"] <= paid["limit"]
        or paid["remaining"] != paid["limit"] - paid["used"]
        or not isinstance(cost, dict)
        or set(cost)
        != {
            "limit_flexcredit",
            "final_reserve_flexcredit",
            "used_flexcredit",
            "learner_remaining_flexcredit",
        }
        or any(
            isinstance(cost.get(key), bool) or not isinstance(cost.get(key), (int, float))
            for key in cost
        )
        or not all(math.isfinite(float(value)) for value in cost.values())
        or float(cost["limit_flexcredit"])
        != expected["cost_budget"]["limit_flexcredit"]
        or float(cost["final_reserve_flexcredit"]) != FINAL_RESERVE_FLEXCREDIT
        or not 0.0 <= float(cost["used_flexcredit"]) <= float(cost["limit_flexcredit"]) - FINAL_RESERVE_FLEXCREDIT
        or not math.isclose(
            float(cost["learner_remaining_flexcredit"]),
            float(cost["limit_flexcredit"]) - FINAL_RESERVE_FLEXCREDIT - float(cost["used_flexcredit"]),
            abs_tol=1.0e-12,
        )
        or not isinstance(accounting, dict)
        or set(accounting)
        != {
            "confirmed_paid_starts",
            "uncertain_paid_start_attempts",
            "accounted_probe_ids",
        }
        or any(
            not isinstance(accounting.get(key), int)
            or isinstance(accounting.get(key), bool)
            or accounting[key] < 0
            for key in ("confirmed_paid_starts", "uncertain_paid_start_attempts")
        )
        or not isinstance(accounting.get("accounted_probe_ids"), list)
        or any(
            not isinstance(probe_id, int) or isinstance(probe_id, bool) or probe_id < 1
            for probe_id in accounting["accounted_probe_ids"]
        )
        or any(
            probe_id >= state["next_probe_id"]
            for probe_id in accounting["accounted_probe_ids"]
        )
        or len(set(accounting["accounted_probe_ids"]))
        != len(accounting["accounted_probe_ids"])
        or len(accounting["accounted_probe_ids"]) != paid["used"]
        or paid["used"]
        != accounting["confirmed_paid_starts"]
        + accounting["uncertain_paid_start_attempts"]
    ):
        raise RuntimeError("probe state is invalid")
    return state


def _probe_accounting_environment() -> dict[str, str]:
    budget = int(os.environ.get("PICBENCH_PROBE_BUDGET", "30"))
    descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_probe_state(budget)
        return {
            PAID_LEARNER_PROBES_REMAINING_ENV: str(state["paid_budget"]["remaining"]),
            LEARNER_THERMAL_REMAINING_ENV: str(
                state["cost_budget"]["learner_remaining_flexcredit"]
            ),
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _probe_budget_envelope(probe_id: int, measurement_remaining: int) -> dict[str, Any]:
    """Return the authoritative post-evaluation accounting snapshot.

    Measurement allocations and paid cloud starts are separate counters: a
    pre-cloud geometry failure consumes a measurement probe but no paid-probe
    or FlexCredit allowance.  The caller reconciles any trusted worker receipt
    before requesting this snapshot.
    """

    budget = int(os.environ.get("PICBENCH_PROBE_BUDGET", "30"))
    descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_probe_state(budget)
        measurement = state["measurement_budget"]
        paid = state["paid_budget"]
        cost = state["cost_budget"]
        if (
            measurement["used"] != probe_id
            or measurement["remaining"] != measurement_remaining
        ):
            raise RuntimeError("probe response accounting snapshot is inconsistent")
        return {
            "measurement_probes_used": measurement["used"],
            "measurement_probes_remaining": measurement["remaining"],
            "paid_probes_used": paid["used"],
            "paid_probes_remaining": paid["remaining"],
            "estimated_flexcredit_used": float(cost["used_flexcredit"]),
            "estimated_flexcredit_remaining": float(
                cost["learner_remaining_flexcredit"]
            ),
            # The receipt conservatively accounts provider estimates and an
            # ambiguous start; it does not attest exact provider billing.
            "billed_flexcredit_used": None,
            "per_run_cap": MAX_FLEXCREDIT_PER_CANDIDATE_EVALUATION,
            "cumulative_cap": float(cost["limit_flexcredit"]),
            "final_reserve": float(cost["final_reserve_flexcredit"]),
            "budget_authority": "private_evaluator_sidecar",
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _record_paid_receipt(
    probe_id: int | None = None, submission_sha256: str | None = None
) -> None:
    try:
        receipt_info = PAID_RECEIPT_PATH.lstat()
    except FileNotFoundError:
        return
    if (
        PAID_RECEIPT_PATH.is_symlink()
        or not PAID_RECEIPT_PATH.is_file()
        or receipt_info.st_uid != os.geteuid()
        or receipt_info.st_mode & 0o077
        or receipt_info.st_size > 4096
    ):
        raise RuntimeError("paid usage receipt is not a private bounded regular file")
    try:
        receipt = json.loads(PAID_RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("paid usage receipt is invalid") from exc
    estimate = receipt.get("estimated_flexcredit") if isinstance(receipt, dict) else None
    receipt_probe_id = receipt.get("probe_id") if isinstance(receipt, dict) else None
    receipt_sha256 = receipt.get("submission_sha256") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version",
            "probe_id",
            "submission_sha256",
            "estimated_flexcredit",
            "start_attempted",
            "start_accepted",
        }
        or receipt.get("schema_version") != 1
        or isinstance(receipt_probe_id, bool)
        or not isinstance(receipt_probe_id, int)
        or receipt_probe_id < 1
        or not isinstance(receipt_sha256, str)
        or SHA256_RE.fullmatch(receipt_sha256) is None
        or (probe_id is not None and receipt_probe_id != probe_id)
        or (submission_sha256 is not None and receipt_sha256 != submission_sha256)
        or receipt.get("start_attempted") is not True
        or not isinstance(receipt.get("start_accepted"), bool)
        or isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(float(estimate))
        or not 0.0 <= float(estimate) <= MAX_FLEXCREDIT_PER_CANDIDATE_EVALUATION
    ):
        raise RuntimeError("paid usage receipt fields are invalid")
    probe_id = receipt_probe_id
    budget = int(os.environ.get("PICBENCH_PROBE_BUDGET", "30"))
    descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_probe_state(budget)
        accounting = state["task_accounting"]
        if probe_id < 1 or probe_id >= state["next_probe_id"]:
            raise RuntimeError("paid usage receipt references an unallocated probe")
        if probe_id not in accounting["accounted_probe_ids"]:
            paid = state["paid_budget"]
            cost = state["cost_budget"]
            if paid["remaining"] <= 0:
                raise RuntimeError("paid probe accounting exceeded its trusted limit")
            new_cost = float(cost["used_flexcredit"]) + float(estimate)
            probe_limit = float(cost["limit_flexcredit"]) - FINAL_RESERVE_FLEXCREDIT
            if new_cost > probe_limit:
                raise RuntimeError("probe FlexCredit accounting exceeded its trusted limit")
            paid["used"] += 1
            paid["remaining"] -= 1
            cost["used_flexcredit"] = new_cost
            cost["learner_remaining_flexcredit"] = max(
                0.0, probe_limit - new_cost
            )
            key = (
                "confirmed_paid_starts"
                if receipt["start_accepted"]
                else "uncertain_paid_start_attempts"
            )
            accounting[key] += 1
            accounting["accounted_probe_ids"].append(probe_id)
            _atomic_json(STATE_PATH, state)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    # Once durable state includes the receipt, consume it. Otherwise an old
    # receipt could make a later pre-cloud failure look like a mismatched start.
    PAID_RECEIPT_PATH.unlink(missing_ok=True)


def _reconcile_pending_paid_receipt() -> None:
    _record_paid_receipt()


def _initialize_probe_state() -> None:
    budget = int(os.environ.get("PICBENCH_PROBE_BUDGET", "30"))
    if budget < 0 or budget > 30:
        raise RuntimeError("PICBENCH_PROBE_BUDGET must be in [0, 30]")
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_probe_state(budget)
        if not STATE_PATH.exists():
            _atomic_json(STATE_PATH, state)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    _reconcile_pending_paid_receipt()



def _allocate_probe() -> tuple[int, int]:
    budget = int(os.environ.get("PICBENCH_PROBE_BUDGET", "30"))
    if budget < 0 or budget > 30:
        raise RuntimeError("PICBENCH_PROBE_BUDGET must be in [0, 30]")
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(STATE_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_probe_state(budget)
        measurement = state["measurement_budget"]
        if measurement["remaining"] <= 0:
            raise RuntimeError("probe budget exhausted")
        probe_id = state["next_probe_id"]
        state["next_probe_id"] += 1
        measurement["used"] += 1
        measurement["remaining"] -= 1
        _atomic_json(STATE_PATH, state)
        return probe_id, measurement["remaining"]
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_submission(source: bytes) -> None:
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".design.", suffix=".py", dir=SUBMISSION_PATH.parent)
    temporary = Path(name)
    try:
        view = memoryview(source)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, SUBMISSION_PATH)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _public_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Build bounded task feedback without copying raw evaluator structures."""

    status = payload.get("status")
    if status not in {"score_valid", "evaluation_error"}:
        status = "evaluation_error"
    summary: dict[str, Any] = {"status": status}
    for key in (
        "valid",
        "design_valid",
        "simulation_valid",
        "evidence_valid",
        "gate_pass",
        "score_eligible",
        "symmetry_gate_pass",
        "input_coverage_pass",
        "cost_guard_pass",
        "paper_envelope_pass",
    ):
        value = payload.get(key)
        if isinstance(value, bool):
            summary[key] = value
    for key in (
        "candidate_runs",
        "candidate_subruns",
        "physical_input_count",
        "simulated_input_count",
        "derived_input_count",
    ):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 8:
            summary[key] = value
    for key in (
        "task_metric",
        "official_score",
        "reference_baseline",
        "paper_margin",
        "T3_1550",
        "T4_1550",
        "excess_loss_db_1550",
        "imbalance_db_1550",
        "phase_error_deg_1550",
        "reflection_db_1550",
        "worst_excess_loss_db",
        "worst_imbalance_db",
        "worst_phase_error_deg",
        "worst_wavelength_nm",
        "estimated_flexcredit",
        "max_flexcredit_per_run",
    ):
        value = payload.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            summary[key] = float(value)

    if summary.get("valid") is True:
        blocking_stage = "none"
        public_error_codes: list[str] = []
        next_steps: list[str] = []
    elif summary.get("design_valid") is False:
        blocking_stage = "public_geometry_contract"
        public_error_codes = ["geometry_contract_failed"]
        next_steps = [
            "Review the public mask, landing, and mirror-symmetry contract before probing again."
        ]
    elif summary.get("simulation_valid") is False:
        blocking_stage = "trusted_simulation_or_cost_guard"
        public_error_codes = ["trusted_simulation_unavailable"]
        next_steps = [
            "Review the public input-coverage and cost-cap diagnostics before probing again."
        ]
    else:
        blocking_stage = "trusted_evaluation"
        public_error_codes = ["trusted_evaluation_failed"]
        next_steps = ["Review the public task contract and rerun a bounded probe."]

    feedback = {
        "schema_version": 1,
        "task_slug": "anisotropic-swg-beamsplitter-v0",
        "feedback_kind": "bounded_beamsplitter_diagnostics",
        "blocking_stage": blocking_stage,
        "public_error_codes": public_error_codes,
        "summary": summary,
        "next_steps": next_steps,
    }
    if payload.get("gate_pass") is not True:
        summary.pop("official_score", None)
    if len(_json_bytes(feedback)) > MAX_PUBLIC_FEEDBACK_BYTES:
        raise RuntimeError("public task feedback exceeds its trusted size limit")
    return feedback


def _public_response(
    payload: dict[str, Any], probe_id: int, remaining: int, submission_sha256: str
) -> dict[str, Any]:
    """Build the canonical bounded 13-field probe envelope."""

    score = _public_scalar(payload.get("official_score"))
    metric = _public_scalar(payload.get("task_metric"))
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or payload.get("gate_pass") is not True
    ):
        score = None
    if not isinstance(metric, (int, float)) or isinstance(metric, bool):
        metric = None
    response = {
        "schema_version": 1,
        "probe_id": probe_id,
        "probes_remaining": remaining,
        "submission_sha256": submission_sha256,
        "status": "completed",
        "valid": payload.get("valid") is True,
        "gate_pass": payload.get("gate_pass") is True,
        "score": score,
        "metric": metric,
        "budget": _probe_budget_envelope(probe_id, remaining),
        "diagnostic_artifacts": {
            "available": False,
            "location": None,
            "reason": "not_exposed_by_this_task",
            "files": {},
            "score_authority": False,
        },
        "best_valid_snapshot": None,
        "task_feedback": _public_result(payload),
    }
    if set(response) != PROBE_RESPONSE_FIELDS:
        raise RuntimeError("probe response field contract mismatch")
    if (
        len(json.dumps(response, sort_keys=True, allow_nan=False).encode("utf-8"))
        > MAX_PUBLIC_PROBE_RESPONSE_BYTES
    ):
        raise RuntimeError("public probe response exceeds its trusted size limit")
    return response


def _public_exception_category(exc: Exception) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        return "evaluation_timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_evaluator_result"
    if isinstance(exc, ValueError):
        return "invalid_request"
    if isinstance(exc, RuntimeError) and str(exc) == "probe budget exhausted":
        return "probe_budget_exhausted"
    return "evaluation_failed"


class Handler(BaseHTTPRequestHandler):
    server_version = "PICBenchAnisotropicSWGBeamsplitterProbe/1.0"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30.0)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _respond(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        probe_id: int | None = None,
    ) -> None:
        if probe_id is None and isinstance(payload.get("probe_id"), int):
            probe_id = payload["probe_id"]
        if probe_id is None:
            context_probe_id = getattr(_ATTEMPT_CONTEXT, "probe_id", None)
            if isinstance(context_probe_id, int):
                probe_id = context_probe_id
        if probe_id is not None:
            _update_probe_attempt(
                probe_id,
                event="response_write_attempted",
            )
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if probe_id is not None:
            _update_probe_attempt(
                probe_id,
                event="response_write_attempted",
                http_status_sent=int(status),
            )
        self.wfile.write(body)
        if probe_id is not None:
            _update_probe_attempt(probe_id, event="response_write_completed")
            if getattr(_ATTEMPT_CONTEXT, "probe_id", None) == probe_id:
                del _ATTEMPT_CONTEXT.probe_id

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._respond(404, {"error": "not_found"})
            return
        self._respond(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/probe":
            self._respond(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_SOURCE_BYTES:
            self._respond(413, {"error": "submission must be a non-empty source file no larger than 1 MiB"})
            return
        source = self.rfile.read(length)
        if len(source) != length:
            self._respond(400, {"error": "truncated submission"})
            return
        with EVALUATION_LOCK:
            probe_id: int | None = None
            try:
                source_sha256 = hashlib.sha256(source).hexdigest()
                probe_id, remaining = _allocate_probe()
                _begin_probe_attempt(probe_id, remaining, source_sha256)
                _write_submission(source)
                try:
                    completed = _run_evaluator(probe_id)
                finally:
                    _record_paid_receipt(probe_id, source_sha256)
                if RESULT_PATH.exists():
                    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
                else:
                    result = {"status": "evaluation_error", "valid": False, "errors": ["evaluator published no result"]}
                if result.get("submission_sha256") != source_sha256:
                    raise RuntimeError("canonical evaluator result did not match the frozen probe submission")
                if completed.returncode != 0 and "errors" not in result:
                    result["errors"] = [completed.stderr.decode("utf-8", "replace")[-1000:]]
                response = _public_response(result, probe_id, remaining, source_sha256)
                _append_probe_history(
                    _build_probe_history_record(response, response["task_feedback"])
                )
                self._respond(200, response)
            except Exception as exc:
                self._respond(
                    409,
                    {
                        "error": "probe_request_failed",
                        "category": _public_exception_category(exc),
                    },
                    probe_id=probe_id,
                )


class BoundedProbeServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._slots = threading.BoundedSemaphore(2)
        super().__init__(*args, **kwargs)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _run_evaluator(probe_id: int) -> subprocess.CompletedProcess[bytes]:
    evaluator_env = os.environ.copy()
    evaluator_env["PICBENCH_EVALUATION_MODE"] = "probe"
    evaluator_env["PICBENCH_PROBE_ID"] = str(probe_id)
    evaluator_env["PICBENCH_PAID_USAGE_RECEIPT_PATH"] = str(PAID_RECEIPT_PATH)
    _reconcile_pending_paid_receipt()
    evaluator_env.update(_probe_accounting_environment())
    PAID_RECEIPT_PATH.unlink(missing_ok=True)
    _update_probe_attempt(probe_id, event="evaluator_launch_attempted")
    process = subprocess.Popen(
        [str(EVALUATE), str(SUBMISSION_PATH)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=evaluator_env,
        start_new_session=True,
    )
    _update_probe_attempt(probe_id, event="evaluator_process_started")
    try:
        stdout, stderr = process.communicate(timeout=1500)
    except subprocess.TimeoutExpired:
        _update_probe_attempt(probe_id, event="timeout_observed")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
        _update_probe_attempt(
            probe_id,
            event="evaluator_finished",
            process_returncode=int(process.returncode),
        )
        raise RuntimeError("canonical probe evaluation timed out")
    _update_probe_attempt(
        probe_id,
        event="evaluator_finished",
        process_returncode=int(process.returncode),
    )
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _start_proxy() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, str(PROXY), "--host", "0.0.0.0", "--port", "8787"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    for _ in range(100):
        if process.poll() is not None:
            error = (process.stderr.read() if process.stderr else b"").decode("utf-8", "replace")
            raise RuntimeError(f"layout proxy failed: {error[-1000:]}")
        try:
            with socket.create_connection(("127.0.0.1", 8787), timeout=0.2):
                return process
        except OSError:
            time.sleep(0.05)
    process.terminate()
    raise RuntimeError("layout proxy startup timed out")


def main() -> int:
    if os.geteuid() != 0:
        raise RuntimeError("probe service must run as root")
    if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError("could not protect credential-stage probe service memory")
    _require_image_identity()
    _initialize_probe_state()
    proxy = _start_proxy()
    try:
        server = BoundedProbeServer(("0.0.0.0", 8788), Handler)
        server.serve_forever()
    finally:
        proxy.terminate()
        proxy.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
