#!/usr/bin/env python3
"""Local license-only proxy for learner PhotonForge layout imports.

The learner receives a fake cloud key plus this localhost endpoint.  This
proxy forwards only PhotonForge's license check with root-staged credentials
and blocks every other cloud API path, so normal learner commands can build and
export layouts without being able to launch/download simulations.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

UPSTREAM = "https://tidy3d-api.simulation.cloud"
LICENSE_PATH = "/tidy3d/licenses/check"
LICENSE_QUERY = "type=PY_LOCAL"
ENV_FILES = ("/root/.trusted_runtime_env",)
MAX_CONCURRENT_REQUESTS = 4
MAX_REQUESTS_PER_WINDOW = 60
REQUEST_WINDOW_SECONDS = 60.0
MAX_TOTAL_REQUESTS = 1000
CLIENT_SOCKET_TIMEOUT_SECONDS = 35.0
MAX_UPSTREAM_RESPONSE_BYTES = 1024 * 1024


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never replay the credential-bearing request to another URL."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


UPSTREAM_OPENER = urllib.request.build_opener(NoRedirectHandler())


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Bound root-owned proxy work before a handler thread is created."""

    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        max_requests_per_window: int = MAX_REQUESTS_PER_WINDOW,
        request_window_seconds: float = REQUEST_WINDOW_SECONDS,
        max_total_requests: int = MAX_TOTAL_REQUESTS,
    ) -> None:
        if min(max_concurrent, max_requests_per_window, max_total_requests) < 1:
            raise ValueError("proxy request limits must be positive")
        if request_window_seconds <= 0:
            raise ValueError("proxy request window must be positive")
        self._request_slots = threading.BoundedSemaphore(max_concurrent)
        self._budget_lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._request_total = 0
        self._max_requests_per_window = max_requests_per_window
        self._request_window_seconds = request_window_seconds
        self._max_total_requests = max_total_requests
        super().__init__(server_address, handler_class)

    def _consume_request_budget(self, *, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else float(now)
        cutoff = timestamp - self._request_window_seconds
        with self._budget_lock:
            while self._request_times and self._request_times[0] <= cutoff:
                self._request_times.popleft()
            if (
                self._request_total >= self._max_total_requests
                or len(self._request_times) >= self._max_requests_per_window
            ):
                return False
            self._request_total += 1
            self._request_times.append(timestamp)
            return True

    def _reject_connection(self, request: socket.socket, status: int, message: str) -> None:
        body = _json_bytes({"error": "layout_proxy_capacity_exceeded", "message": message})
        reason = "Too Many Requests" if status == 429 else "Service Unavailable"
        response = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body
        try:
            request.sendall(response)
        except OSError:
            pass
        self.shutdown_request(request)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        # ThreadingMixIn normally allocates a thread immediately.  Acquire the
        # root-work slot and rate/total budget first, rejecting synchronously.
        if not self._request_slots.acquire(blocking=False):
            self._reject_connection(request, 503, "concurrent request limit reached")
            return
        if not self._consume_request_budget():
            self._request_slots.release()
            self._reject_connection(request, 429, "request budget exhausted")
            return
        try:
            request.settimeout(CLIENT_SOCKET_TIMEOUT_SECONDS)
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def _parse_export_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _load_root_env_files() -> None:
    for raw_path in ENV_FILES:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parsed = _parse_export_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key in {"TIDY3D_API_KEY", "SIMCLOUD_APIKEY"} and value and key not in os.environ:
                os.environ[key] = value


def _cloud_key() -> str:
    _load_root_env_files()
    for name in ("SIMCLOUD_APIKEY", "TIDY3D_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value and value != "agenticpic-layout-token":
            return value
    return ""


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _read_bounded_response(response: Any) -> bytes:
    body = response.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
    if len(body) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise ValueError("upstream license response exceeded the size limit")
    return body


def _is_exact_license_request(raw_path: str) -> bool:
    parsed = urllib.parse.urlsplit(raw_path)
    return (
        not parsed.scheme
        and not parsed.netloc
        and parsed.path == LICENSE_PATH
        and parsed.query == LICENSE_QUERY
        and not parsed.fragment
    )


def _redact_cloud_key(body: bytes, key: str) -> bytes:
    """Remove direct and URL-encoded credential echoes from upstream bodies."""
    replacements = {
        key.encode("utf-8"),
        urllib.parse.quote(key, safe="").encode("ascii"),
        urllib.parse.quote_plus(key, safe="").encode("ascii"),
    }
    redacted = body
    for secret in replacements:
        if secret:
            redacted = redacted.replace(secret, b"[redacted]")
    return redacted


class LicenseProxyHandler(BaseHTTPRequestHandler):
    server_version = "AgenticPICLayoutLicenseProxy/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if not _is_exact_license_request(self.path):
            self._send_json(
                403,
                {
                    "error": "blocked_by_agenticpic_layout_proxy",
                    "message": "Only PhotonForge license checks are allowed in learner commands. Use run_probe for full validation.",
                    "path": self.path,
                },
            )
            return

        key = _cloud_key()
        if not key:
            self._send_json(
                503,
                {
                    "error": "missing_root_cloud_credentials",
                    "message": "Root-staged credentials are unavailable for the layout license proxy.",
                },
            )
            return

        headers = {
            "Accept": "application/json",
            "User-Agent": "AgenticPICLayoutLicenseProxy/1.0",
            "simcloud-api-key": key,
        }
        upstream_url = f"{UPSTREAM}{LICENSE_PATH}?{LICENSE_QUERY}"
        request = urllib.request.Request(upstream_url, headers=headers, method="GET")
        try:
            with UPSTREAM_OPENER.open(request, timeout=30) as response:
                body = _redact_cloud_key(_read_bounded_response(response), key)
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                self._send_json(
                    502,
                    {
                        "error": "upstream_license_redirect_blocked",
                        "message": "The credential-bearing license request cannot be redirected.",
                    },
                )
                return
            try:
                body = _redact_cloud_key(_read_bounded_response(exc), key)
            except ValueError as size_error:
                self._send_json(
                    502,
                    {
                        "error": "upstream_license_check_failed",
                        "message": str(size_error),
                    },
                )
                return
            self.send_response(exc.code)
            self.send_header(
                "Content-Type",
                exc.headers.get("Content-Type", "application/json"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # pragma: no cover - network/environment failure
            self._send_json(502, {"error": "upstream_license_check_failed", "message": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        self._send_json(
            403,
            {
                "error": "blocked_by_agenticpic_layout_proxy",
                "message": "Cloud mutation/compute requests are blocked in learner commands.",
                "path": self.path,
            },
        )


def main() -> int:
    if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise RuntimeError("could not protect credential-bearing proxy memory")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTICPIC_LAYOUT_LICENSE_PROXY_PORT", "8787")))
    args = parser.parse_args()

    server = BoundedThreadingHTTPServer((args.host, args.port), LicenseProxyHandler)
    print(f"AgenticPIC layout license proxy listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
