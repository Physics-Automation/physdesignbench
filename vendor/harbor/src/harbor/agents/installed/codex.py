from inspect import getattr_static
import base64
import binascii
import hashlib
import json
import math
import os
import shlex
import stat
import time
import uuid
from urllib.parse import urlsplit
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Annotated, Any, Literal, override

import toml
from pydantic import Field, model_validator

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    filter_jsonl_events,
    with_prompt_template,
)
from harbor.agents.options import Cli, InstalledAgentOptions
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.agents.protocols import ACPAgentMixin
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.bridge import BridgeKind
from harbor.models.trajectories import (
    Agent,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trajectories.subagent_trajectory_ref import SubagentTrajectoryRef
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value
from harbor.utils.path_safety import validate_output_path
from harbor.utils.scripts import (
    ensure_acp_node_command,
    pinned_bin_wrapper_command,
    safe_bin_symlink_command,
)
from harbor.utils.trajectory_utils import format_trajectory_json

_ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
_IMAGE_EXTENSIONS: dict[_ImageMediaType, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}

_TOOL_OUTPUT_TEXT_TYPES = frozenset({"text", "input_text"})
_TOOL_OUTPUT_IMAGE_TYPES = frozenset({"image", "input_image"})
_TOOL_OUTPUT_CONTENT_TYPES = (
    _TOOL_OUTPUT_TEXT_TYPES
    | _TOOL_OUTPUT_IMAGE_TYPES
    | frozenset(
        {"audio", "input_audio", "resource", "resource_link", "encrypted_content"}
    )
)
_BASE64_DATA_URL = re.compile(
    r"data:(?P<media_type>[^;,]+)(?:;[^;,]*)*;base64,(?P<data>.*)", re.DOTALL
)


class CodexOptions(InstalledAgentOptions):
    continue_until_timeout: str | bool = True
    shutdown_grace_sec: float = 20.0
    max_idle_turns: int = 2
    openrouter_base_url: str | None = None

    reasoning_effort: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None,
        Cli("-c", format="-c model_reasoning_effort={value}"),
    ] = Field(
        default=None,
        description="Model reasoning effort.",
    )
    reasoning_summary: Annotated[
        Literal["auto", "concise", "detailed", "none"] | None,
        Cli("-c", format="-c model_reasoning_summary={value}"),
    ] = Field(
        default=None,
        description="How model reasoning is summarized.",
    )
    web_search: Annotated[
        Literal["disabled", "cached", "live"] | None,
        Cli("-c", format="-c web_search={value}"),
    ] = Field(
        default=None,
        description="Web search mode.",
    )
    disable_web_search: bool = Field(
        default=False, description="Same as web_search=disabled."
    )
    dangerously_bypass_hook_trust: Annotated[
        bool, Cli("--dangerously-bypass-hook-trust")
    ] = Field(
        default=False,
        description="Run enabled Codex hooks without persisted trust; use only with vetted hook sources.",
    )

    @model_validator(mode="after")
    def _disable_web_search(self):
        if self.disable_web_search:
            self.web_search = "disabled"
        return self


class Codex(BaseInstalledAgent, ACPAgentMixin):
    """
    The Codex agent uses OpenAI's Codex CLI tool to solve tasks.

    A native Codex ``config.toml`` path or inline JSON object can be supplied
    with ``config``. Harbor uses it as the base, then applies explicit runtime
    inputs such as ``OPENAI_BASE_URL`` and task MCP servers before uploading the
    effective file to ``$CODEX_HOME/config.toml``.
    """

    capabilities = AgentCapabilities(
        atif=True,
        resume=True,
        load_native_trajectory=True,
        load_atif_trajectory=True,
        native_config=True,
        skills=True,
        mcp_servers=True,
        bridges=frozenset({BridgeKind.ACP}),
    )
    MODEL_CONNECTION = ModelConnectionSpec(default_provider="openai")
    _OUTPUT_FILENAME = "codex.txt"
    _REMOTE_CODEX_HOME = PurePosixPath("/tmp/codex-home")
    _REMOTE_CODEX_SECRETS_DIR = PurePosixPath("/tmp/codex-secrets")
    _REMOTE_ACP_CODEX_HOME = PurePosixPath("/tmp/codex-acp-home")
    CODEX_ACP_VERSION = "1.12.0"
    _ROLLOUT_FILENAME_RE = re.compile(r"^rollout-(\d{4})-(\d{2})-(\d{2})T\S+\.jsonl$")
    SUPPORTS_PROBE_SUSPENSION: bool = True
    _LOCAL_BINARY_ENV = "HARBOR_CODEX_LOCAL_BINARY"
    _INSTALL_CHECK_COMMAND = (
        "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
        "command -v codex >/dev/null 2>&1"
    )
    _INSTALL_VERSION_COMMAND = (
        "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version"
    )
    _HARBOR_TIMEOUT_METADATA_KEY = "harbor_agent_timeout_sec"
    _OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    _MAX_SESSION_FILES = 256
    _MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024
    _MAX_SESSION_BYTES = 512 * 1024 * 1024
    _TRAJECTORY_INDEX_SCHEMA = "harbor-codex-trajectories-v1"
    _CONTINUATION_PROMPT = (
        "Continue actively working on the original task. Do not merely summarize "
        "the previous turn. Inspect the current files, make further concrete "
        "improvements, run useful local checks, and persist the best current "
        "submission. Continue until Harbor externally ends the working period."
    )

    options_model = CodexOptions
    options: CodexOptions

    @override
    def _validate_native_load_trajectory(self, path: Path) -> None:
        if not self._ROLLOUT_FILENAME_RE.match(path.name):
            raise ValueError(
                "Codex load_trajectory must be a native rollout file named "
                "rollout-<timestamp>-<session-id>.jsonl or an ATIF "
                f"trajectory .json file; got {path.name!r}"
            )

    @override
    async def _upload_load_trajectory(
        self, environment: BaseEnvironment, source: Path
    ) -> None:
        match = self._ROLLOUT_FILENAME_RE.match(source.name)
        if match is None:
            raise ValueError(f"Invalid Codex rollout filename: {source.name!r}")
        target_dir = (
            self.environment_logs_dir / "sessions" / match[1] / match[2] / match[3]
        )
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {shlex.quote(target_dir.as_posix())}",
        )
        await self._upload_agent_owned_file(
            environment, source, (target_dir / source.name).as_posix()
        )

    @staticmethod
    def _rollout_text(message: str | list[ContentPart] | None) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        parts = [
            part.text if part.type == "text" and part.text else "[image omitted]"
            for part in message
        ]
        return "\n".join(parts)

    @staticmethod
    def _rollout_timestamp(trajectory: Trajectory) -> datetime:
        """Session start time for the rollout filename and meta line."""
        for step in trajectory.steps:
            if step.timestamp:
                try:
                    return datetime.fromisoformat(step.timestamp.replace("Z", "+00:00"))
                except ValueError:
                    break
        return datetime.now(timezone.utc)

    @staticmethod
    def _rollout_custom_tool_input(step: Step, tool_call: ToolCall) -> str | None:
        """The ``custom_tool_call`` input string when the recorded call was a
        free-form tool call; None means render a structured ``function_call``."""
        details = (step.extra or {}).get("tool_call_details") or {}
        entry = details.get(tool_call.tool_call_id) or {}
        raw = entry.get("raw_arguments")
        if not isinstance(raw, str):
            return None
        item_type = entry.get("item_type")
        if item_type is not None:
            return raw if item_type == "custom_tool_call" else None
        try:
            return None if json.loads(raw) == tool_call.arguments else raw
        except ValueError:
            return raw

    @override
    def atif_to_native_trajectory(
        self, trajectory: Trajectory, session_id: str
    ) -> tuple[str, str]:
        """Render an ATIF trajectory as a native Codex rollout transcript.

        The inverse of the rollout-to-ATIF conversion in this class: ATIF steps
        become the ``session_meta``, ``response_item``, and ``event_msg`` lines
        that Codex stores at
        ``$CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<ts>-<id>.jsonl``,
        so ``codex exec resume`` picks the conversation up as history.
        """
        timestamp = self._rollout_timestamp(trajectory)
        filename = (
            f"rollout-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}-{session_id}.jsonl"
        )
        iso_timestamp = (
            timestamp.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        version = trajectory.agent.version if trajectory.agent else None

        lines: list[dict[str, Any]] = [
            {
                "timestamp": iso_timestamp,
                "type": "session_meta",
                "payload": {
                    "session_id": session_id,
                    "id": session_id,
                    "timestamp": iso_timestamp,
                    "cwd": "/app",
                    "originator": "codex_exec",
                    "cli_version": version or "unknown",
                    "source": "exec",
                    "thread_source": "user",
                },
            }
        ]
        last_timestamp = iso_timestamp

        def append_item(payload: dict[str, Any], timestamp: str) -> None:
            lines.append(
                {"timestamp": timestamp, "type": "response_item", "payload": payload}
            )

        def append_message(role: str, text: str, timestamp: str) -> None:
            content_type = "output_text" if role == "assistant" else "input_text"
            append_item(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": text}],
                },
                timestamp,
            )

        def append_event(payload: dict[str, Any], timestamp: str) -> None:
            lines.append(
                {"timestamp": timestamp, "type": "event_msg", "payload": payload}
            )

        for step in trajectory.steps:
            line_timestamp = step.timestamp or last_timestamp
            last_timestamp = line_timestamp

            if step.source == "system":
                append_message(
                    "developer", self._rollout_text(step.message), line_timestamp
                )
                continue
            if step.source == "user":
                text = self._rollout_text(step.message)
                append_message("user", text, line_timestamp)
                # Codex's session listing (used by `codex exec resume`) only
                # includes threads that carry a user_message event.
                append_event({"type": "user_message", "message": text}, line_timestamp)
                continue

            text = self._rollout_text(step.message)
            if text:
                append_message("assistant", text, line_timestamp)
                append_event({"type": "agent_message", "message": text}, line_timestamp)

            tool_calls = step.tool_calls or []
            custom_call_ids = set()
            for tool_call in tool_calls:
                raw_input = self._rollout_custom_tool_input(step, tool_call)
                if raw_input is not None:
                    custom_call_ids.add(tool_call.tool_call_id)
                    append_item(
                        {
                            "type": "custom_tool_call",
                            "status": "completed",
                            "call_id": tool_call.tool_call_id,
                            "name": tool_call.function_name,
                            "input": raw_input,
                        },
                        line_timestamp,
                    )
                else:
                    append_item(
                        {
                            "type": "function_call",
                            "call_id": tool_call.tool_call_id,
                            "name": tool_call.function_name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                        line_timestamp,
                    )

            results = step.observation.results if step.observation else []
            results_by_call_id = {
                result.source_call_id: result
                for result in results
                if result.source_call_id is not None
            }
            for tool_call in tool_calls:
                call_id = tool_call.tool_call_id
                result = results_by_call_id.get(call_id)
                is_custom = call_id in custom_call_ids
                append_item(
                    {
                        "type": (
                            "custom_tool_call_output"
                            if is_custom
                            else "function_call_output"
                        ),
                        "call_id": call_id,
                        "output": (
                            self._rollout_text(result.content)
                            if result is not None
                            else "[no output recorded]"
                        ),
                    },
                    line_timestamp,
                )

        content = "".join(json.dumps(line) + "\n" for line in lines)
        return filename, content

    def __init__(
        self,
        *args: Any,
        continue_until_timeout: str | bool = True,
        shutdown_grace_sec: str | int | float = 20.0,
        max_idle_turns: str | int = 2,
        openrouter_base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._openrouter_base_url = self._OPENROUTER_BASE_URL
        if openrouter_base_url is not None:
            # Operator-supplied host broker; never accept shell/TOML injection
            # or account-path/query overrides in generated provider config.
            if not isinstance(openrouter_base_url, str):
                raise ValueError("Invalid OpenRouter broker base URL")
            parsed = urlsplit(openrouter_base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path != "/api/v1"
                or any(char in openrouter_base_url for char in '\\"\n\r$`')
                or any(char.isspace() for char in openrouter_base_url)
            ):
                raise ValueError("Invalid OpenRouter broker base URL")
            self._openrouter_base_url = openrouter_base_url
        self._continue_until_timeout = parse_bool_env_value(
            continue_until_timeout,
            name="continue_until_timeout",
        )
        try:
            self._shutdown_grace_sec = float(shutdown_grace_sec)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid value for 'shutdown_grace_sec': expected a finite "
                "non-negative number"
            ) from exc
        if not math.isfinite(self._shutdown_grace_sec) or self._shutdown_grace_sec < 0:
            raise ValueError(
                "Invalid value for 'shutdown_grace_sec': expected a finite "
                "non-negative number"
            )
        try:
            self._max_idle_turns = int(max_idle_turns)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid value for 'max_idle_turns': expected an integer in [1, 100]"
            ) from exc
        if isinstance(max_idle_turns, bool) or not 1 <= self._max_idle_turns <= 100:
            raise ValueError(
                "Invalid value for 'max_idle_turns': expected an integer in [1, 100]"
            )
        self._acp_codex_executable: str | None = None
        super().__init__(*args, **kwargs)
        self._base_config = self._load_base_config()
        self._trajectory_session_id: str | None = None
        self._last_invocation_completed = False
        self._session_export_failed = False
        self._completed_session_exports = 0

    @staticmethod
    @override
    def name() -> str:
        return AgentName.CODEX.value

    @property
    def _trajectory_path(self) -> PurePosixPath:
        return self.environment_logs_dir / "trajectory.json"

    @override
    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version"

    @override
    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line.removeprefix("codex-cli").strip()
        return text

    @override
    def _classify_exec_error(
        self, command: str, result: Any
    ) -> NonZeroAgentExitCodeError:
        """Classify the ``codex exec --json`` launch from harness-emitted
        errors only; ``item.*`` events carry tool output and model text."""
        if "codex exec" not in command:
            return super()._classify_exec_error(command, result)
        evidence = SimpleNamespace(
            return_code=result.return_code,
            stdout=filter_jsonl_events(
                result.stdout, lambda e: e.get("type") in {"turn.failed", "error"}
            ),
            stderr=result.stderr,
        )
        return super()._classify_exec_error(command, evidence)

    async def _installed_codex_satisfies_version(
        self, environment: BaseEnvironment
    ) -> bool:
        if self._version is None:
            check_result = await environment.exec(command=self._INSTALL_CHECK_COMMAND)
            return check_result.return_code == 0

        version_result = await environment.exec(command=self._INSTALL_VERSION_COMMAND)
        if version_result.return_code != 0:
            return False

        installed_version = self.parse_version(version_result.stdout or "")
        return installed_version == self._version

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_codex_satisfies_version(environment):
            self.logger.debug("Codex is already available at the requested version")
            return

        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "nodejs", "npm", "ripgrep")
        )

        # Allow controlled benchmark hosts to inject a trusted standalone
        # Codex build without downloading Node and npm packages per container.
        local_binary_value = os.environ.get(self._LOCAL_BINARY_ENV)
        if local_binary_value:
            local_binary = Path(local_binary_value).expanduser().resolve(strict=True)
            if not local_binary.is_file():
                raise ValueError(
                    f"{self._LOCAL_BINARY_ENV} must reference a regular file"
                )
            code_mode_host = local_binary.with_name("codex-code-mode-host")
            if not code_mode_host.is_file():
                raise ValueError(
                    f"{self._LOCAL_BINARY_ENV} must be beside "
                    "a codex-code-mode-host regular file"
                )
            remote_binary = PurePosixPath("/tmp/harbor-codex-standalone")
            remote_code_mode_host = PurePosixPath(
                "/tmp/harbor-codex-code-mode-host-standalone"
            )
            await environment.upload_file(local_binary, str(remote_binary))
            await environment.upload_file(code_mode_host, str(remote_code_mode_host))
            await self.exec_as_root(
                environment,
                command=(
                    f"install -m 0755 {shlex.quote(str(remote_binary))} "
                    '"/usr/local/bin/codex" && '
                    f"install -m 0755 {shlex.quote(str(remote_code_mode_host))} "
                    '"/usr/local/bin/codex-code-mode-host" && '
                    f"rm -f {shlex.quote(str(remote_binary))} "
                    f"{shlex.quote(str(remote_code_mode_host))} && "
                    "/usr/local/bin/codex --version"
                ),
            )
            return

        # Install codex (as default user)
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
                f"  npm install -g @openai/codex{version_spec};"
                " else"
                "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&"
                '  export NVM_DIR="$HOME/.nvm" &&'
                '  \\. "$NVM_DIR/nvm.sh" || true &&'
                "  command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } &&"
                "  nvm install 22 && nvm alias default 22 && npm -v &&"
                f"  npm install -g @openai/codex{version_spec};"
                " fi && "
                "codex --version"
            ),
            env={"NVM_NODEJS_ORG_MIRROR": "https://nodejs.org/dist"},
        )
        # Symlink node/codex to /usr/local/bin (root)
        await self.exec_as_root(
            environment,
            command=(
                "for bin in node codex; do"
                '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
                '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
                '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
                "  fi;"
                " done"
            ),
        )

    @override
    def acp_command(self) -> list[str]:
        command = [
            "env",
            "NO_BROWSER=1",
            "INITIAL_AGENT_MODE=agent-full-access",
            'DEFAULT_AUTH_REQUEST={"methodId":"api-key"}',
            f"CODEX_HOME={self._REMOTE_ACP_CODEX_HOME.as_posix()}",
        ]
        session_config: dict[str, Any] = {}
        if self.model_name:
            session_config["model"] = self.model_name.split("/")[-1]
        for key, value in (
            ("model_reasoning_effort", self.options.reasoning_effort),
            ("model_reasoning_summary", self.options.reasoning_summary),
            ("web_search", self.options.web_search),
        ):
            if value is not None:
                session_config[key] = value
        if self.options.dangerously_bypass_hook_trust:
            session_config["bypass_hook_trust"] = True
        if session_config:
            command.append(f"CODEX_CONFIG={json.dumps(session_config)}")
        if self._acp_codex_executable:
            command.append(f"CODEX_PATH={self._acp_codex_executable}")
        command.append("codex-acp")
        return command

    @override
    def acp_env(self) -> dict[str, str]:
        # Keep the adapter's credential separate from an OpenAI user agent's key.
        api_key = self.model_connection.api_key
        return {"CODEX_API_KEY": api_key} if api_key else {}

    @override
    async def acp_install(self, environment: BaseEnvironment) -> None:
        if not self.model_name:
            raise ValueError("Model name is required")
        if self._resolve_auth_json_path() is not None:
            raise ValueError(
                "Codex auth.json authentication is not supported in ACP mode"
            )

        home = shlex.quote(self._REMOTE_ACP_CODEX_HOME.as_posix())
        result = await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"mkdir -p {home} && chmod 700 {home}; "
                '_hb_system_node="$(command -v node 2>/dev/null || true)"; '
                'if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi; '
                # Capture Harbor's CLI before installing the adapter/toolchain.
                '_hb_codex="$(command -v codex 2>/dev/null || true)"; '
                'if [ -z "$_hb_codex" ]; then '
                'echo "Could not locate the installed Codex binary for ACP mode" >&2; '
                "exit 1; fi; " + ensure_acp_node_command() + " && npm install -g "
                f"@agentclientprotocol/codex-acp@{self.CODEX_ACP_VERSION} && "
                'echo "${_hb_system_node:-none}:$(command -v node)'
                ':$(command -v codex-acp):${_hb_codex}"'
            ),
        )
        lines = (result.stdout or "").strip().splitlines()
        fields = lines[-1].split(":") if lines else []
        if len(fields) != 4 or not all(fields[1:]):
            raise RuntimeError(
                "Could not locate node/codex-acp/codex binaries for ACP mode"
            )
        system_node, node_path, adapter_path, codex_path = fields
        # Otherwise the adapter silently uses its bundled Codex version.
        self._acp_codex_executable = codex_path
        command = pinned_bin_wrapper_command(
            node_path, adapter_path, "/usr/local/bin/codex-acp"
        )
        if system_node == "none":
            command += " && " + safe_bin_symlink_command(
                node_path, "/usr/local/bin/node"
            )
        await self.exec_as_root(environment, command=command)

        config = self._build_effective_config(self.model_connection.configured_base_url)
        # API-key login can persist credentials; keep them in the temporary home
        # rather than a keyring that acp_teardown cannot clean up.
        config["cli_auth_credentials_store"] = "file"
        await self._upload_effective_config(
            environment,
            config,
            (self._REMOTE_ACP_CODEX_HOME / "config.toml").as_posix(),
        )
        if skills_command := self._build_register_skills_command():
            await self.exec_as_agent(environment, command=skills_command)

    @override
    async def acp_teardown(self, environment: BaseEnvironment) -> None:
        home = shlex.quote(self._REMOTE_ACP_CODEX_HOME.as_posix())
        sessions = shlex.quote((self.environment_logs_dir / "sessions").as_posix())
        logs = shlex.quote(self.environment_logs_dir.as_posix())
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    f"if [ -d {home}/sessions ]; then "
                    f"mkdir -p {logs} && rm -rf {sessions} && "
                    f"cp -R {home}/sessions {sessions}; fi"
                ),
            )
        finally:
            # Native run()'s finally block is bypassed for bridge targets.
            await self.exec_as_agent(environment, command=f"rm -rf {home}")

    @property
    @override
    def remote_session_logs_dir(self) -> PurePosixPath:
        return self._REMOTE_CODEX_HOME / "sessions"

    @override
    def convert_trajectory(self, logs_dir: Path) -> Trajectory | None:
        session_dir = self._get_session_dir(logs_dir)
        return self._convert_events_to_trajectory(session_dir) if session_dir else None

    def _get_session_dir(self, logs_dir: Path | None = None) -> Path | None:
        """Return the archive root; a session may span multiple date folders."""
        sessions_dir = (
            logs_dir if logs_dir is not None else self.logs_dir
        ) / "sessions"
        if not sessions_dir.exists():
            return None
        if sessions_dir.is_symlink() or not sessions_dir.is_dir():
            raise ValueError("Codex session archive is not a regular directory")
        return sessions_dir

    @staticmethod
    def _canonical_session_id(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = str(uuid.UUID(value))
        except (ValueError, AttributeError):
            return None
        return parsed if value == parsed else None

    def _capture_session_id(self, stdout: str, expected: str | None = None) -> None:
        """Bind collected history to the CLI's parent thread, never an arbitrary child."""
        thread_ids = set()
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict) and event.get("type") == "thread.started":
                thread_id = self._canonical_session_id(event.get("thread_id"))
                if thread_id is None:
                    raise ValueError("Codex emitted an invalid parent thread ID")
                thread_ids.add(thread_id)
        wanted = expected or self._trajectory_session_id
        if len(thread_ids) > 1 or (wanted and thread_ids and thread_ids != {wanted}):
            raise ValueError("Codex parent thread changed during continuation")
        if wanted:
            self._trajectory_session_id = wanted
        elif thread_ids:
            self._trajectory_session_id = next(iter(thread_ids))

    def _read_session_archive(self, session_dir: Path) -> dict[str, dict[str, Any]]:
        """Read only bounded, stable regular JSONL files, without following links."""
        sessions: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        file_count = 0
        directory_count = 0
        pending = [session_dir]
        while pending:
            pending.sort(reverse=True)
            directory = pending.pop()
            directory_count += 1
            if (
                directory_count > 1024
                or len(directory.relative_to(session_dir).parts) > 16
            ):
                raise ValueError("Codex session directory limit exceeded")
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Codex session archive contains a directory link")
            for path in sorted(directory.iterdir()):
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(
                        "Codex session archive contains a link or special file"
                    )
                file_count += 1
                total_bytes += info.st_size
                if (
                    file_count > self._MAX_SESSION_FILES
                    or info.st_size > self._MAX_SESSION_FILE_BYTES
                    or total_bytes > self._MAX_SESSION_BYTES
                ):
                    raise ValueError("Codex session archive size limit exceeded")
                if path.suffix != ".jsonl":
                    continue
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as handle:
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode) or (
                        before.st_dev,
                        before.st_ino,
                    ) != (info.st_dev, info.st_ino):
                        raise ValueError("Codex session file changed during collection")
                    data = handle.read(self._MAX_SESSION_FILE_BYTES + 1)
                    after = os.fstat(handle.fileno())
                if len(data) != before.st_size or (
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("Codex session file changed during collection")
                events = []
                malformed_lines = 0
                for line in data.splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        malformed_lines += 1
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                metas = [
                    e.get("payload", {})
                    for e in events
                    if e.get("type") == "session_meta"
                ]
                ids = {
                    self._canonical_session_id(m.get("id"))
                    for m in metas
                    if isinstance(m, dict)
                }
                if len(ids) != 1 or None in ids:
                    raise ValueError("Codex rollout lacks one valid session identity")
                session_id = next(iter(ids))
                if session_id is None:
                    raise ValueError("Codex session identity is missing")
                group = sessions.setdefault(
                    session_id,
                    {
                        "events": [],
                        "raw_files": [],
                        "malformed_lines": 0,
                        "event_chunks": [],
                    },
                )
                first_timestamp = next(
                    (
                        e["timestamp"]
                        for e in events
                        if isinstance(e.get("timestamp"), str)
                    ),
                    "",
                )
                group["event_chunks"].append((first_timestamp, events))
                group["malformed_lines"] += malformed_lines
                group["raw_files"].append(
                    {
                        "path": path.relative_to(session_dir).as_posix(),
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        # Preserve append order within each JSONL file, including untimestamped
        # token-count events. Order resumed fragments by their first timestamp.
        for group in sessions.values():
            chunks = group.pop("event_chunks")
            chunks.sort(key=lambda item: item[0])
            group["events"] = [event for _, events in chunks for event in events]
        return sessions

    @staticmethod
    def _extract_message_text(content: list[Any]) -> str:
        """Extract joined text from Codex content blocks."""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    def _parse_output_blob(
        self, raw: Any
    ) -> tuple[str | list[ContentPart] | None, dict[str, Any] | None]:
        """Extract the observation content and metadata from a Codex tool output."""
        if raw is None:
            return None, None

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return raw, None
        else:
            parsed = raw

        if isinstance(parsed, dict):
            output = parsed.get("output")
            if output is None and parsed:
                # dumping remaining structure if output missing
                output = json.dumps(parsed, ensure_ascii=False)
            elif output is not None and not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            metadata = parsed.get("metadata")
            return output, metadata if isinstance(metadata, dict) else None

        if self._is_tool_output_content(parsed):
            return self._tool_output_content(parsed), None

        if isinstance(raw, str):
            return raw, None
        return json.dumps(parsed, ensure_ascii=False), None

    @staticmethod
    def _is_tool_output_content(value: Any) -> bool:
        """Whether a tool output is a list of MCP content blocks or content items."""
        if not isinstance(value, list) or not value:
            return False
        for block in value:
            if not isinstance(block, dict):
                return False
            block_type = block.get("type")
            if (
                not isinstance(block_type, str)
                or block_type not in _TOOL_OUTPUT_CONTENT_TYPES
            ):
                return False
        return True

    def _tool_output_content(
        self, blocks: list[dict[str, Any]]
    ) -> str | list[ContentPart]:
        """Convert tool output content blocks into ATIF observation content.

        Text-only output becomes a plain string, parts joined by newlines as
        ``_rollout_text`` renders them. Images are saved next to the trajectory
        and referenced from image ``ContentPart``s. Other blocks (audio,
        resources, encrypted content) are kept as their JSON text.
        """
        parts: list[ContentPart] = []
        for block in blocks:
            block_type = block["type"]
            text = block.get("text")
            if block_type in _TOOL_OUTPUT_TEXT_TYPES and isinstance(text, str):
                parts.append(ContentPart(type="text", text=text))
            elif block_type in _TOOL_OUTPUT_IMAGE_TYPES:
                parts.append(self._tool_output_image_part(block))
            else:
                parts.append(
                    ContentPart(type="text", text=json.dumps(block, ensure_ascii=False))
                )

        if any(part.type == "image" for part in parts):
            return parts
        return "\n".join(part.text or "" for part in parts)

    def _tool_output_image_part(self, block: dict[str, Any]) -> ContentPart:
        """The content part for an MCP image block or an ``input_image`` item.

        Inline base64 images are saved next to the trajectory. A remote URL is
        kept in text: ``ImageSource`` accepts URLs, but Harbor's consumers
        (the viewer, RewardKit judges) only resolve local files, and fetching
        the image here is not an option. Images that cannot be referenced at
        all (file ids, unsupported types, undecodable data) are omitted.
        """
        if block["type"] == "image":
            data = block.get("data")
            media_type = block.get("mimeType") or block.get("mime_type")
        else:
            data = block.get("image_url")
            media_type = None
        if not isinstance(data, str):
            return ContentPart(type="text", text="[image omitted]")
        if "://" in data and not data.startswith("data:"):
            return ContentPart(type="text", text=f"[image: {data}]")

        source = self._save_tool_output_image(data, media_type)
        if source is None:
            return ContentPart(type="text", text="[image omitted]")
        return ContentPart(type="image", source=source)

    def _save_tool_output_image(
        self, data: str, media_type: str | None
    ) -> ImageSource | None:
        """Save an inline tool output image under ``images/`` in the logs dir.

        ``data`` is raw base64 with a separate ``media_type`` (an MCP image
        block) or a base64 ``data:`` URL (an ``input_image`` item). Returns
        None for anything that cannot be stored as a file.
        """
        data_url = _BASE64_DATA_URL.fullmatch(data)
        if data_url is not None:
            media_type, data = data_url["media_type"], data_url["data"]
        if not isinstance(media_type, str):
            return None

        valid_media_type = self._image_media_type(media_type)
        if valid_media_type is None:
            self.logger.debug("Unsupported Codex tool image type: %s", media_type)
            return None
        extension = _IMAGE_EXTENSIONS[valid_media_type]

        try:
            image_bytes = base64.b64decode("".join(data.split()), validate=True)
        except (binascii.Error, ValueError):
            self.logger.debug("Failed to decode Codex tool image", exc_info=True)
            return None
        if not image_bytes:
            return None

        digest = hashlib.sha256(image_bytes).hexdigest()[:16]
        filename = f"codex_{digest}.{extension}"
        image_path = self.logs_dir / "images" / filename
        temporary_path = image_path.with_name(f"{filename}.tmp")
        try:
            validate_output_path(image_path, self.logs_dir)
            validate_output_path(temporary_path, self.logs_dir)
            if not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path.write_bytes(image_bytes)
                temporary_path.replace(image_path)
        except (OSError, ValueError):
            self.logger.debug("Failed to save Codex tool image", exc_info=True)
            return None
        return ImageSource(media_type=valid_media_type, path=f"images/{filename}")

    @staticmethod
    def _image_media_type(value: str) -> _ImageMediaType | None:
        """The ``ImageSource`` media type for a MIME type string, if supported."""
        media_type = value.strip().lower()
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        for valid_media_type in _IMAGE_EXTENSIONS:
            if media_type == valid_media_type:
                return valid_media_type
        return None

    @staticmethod
    def _group_events_by_api_call_id(
        normalized_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge assistant events from the same Codex model request into one step."""
        result: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}
        group_order: list[str] = []

        def flush() -> None:
            for group_id in group_order:
                group = groups.pop(group_id, None)
                if group is None:
                    continue
                group["tool_calls"].sort(key=lambda tc: tc.get("tool_order", 0))
                message_parts = [
                    part
                    for part in group.pop("message_parts")
                    if isinstance(part, str) and part
                ]
                group["text"] = "\n\n".join(message_parts)
                result.append(group)
            group_order.clear()

        for event in normalized_events:
            api_call_id = event.get("api_call_id")
            kind = event.get("kind")
            role = event.get("role")

            if kind == "message" and role != "assistant":
                flush()
                result.append(event)
                continue

            if not isinstance(api_call_id, str):
                flush()
                result.append(event)
                continue

            if api_call_id not in groups:
                groups[api_call_id] = {
                    "kind": "bundled",
                    "api_call_id": api_call_id,
                    "codex_turn_id": event.get("codex_turn_id"),
                    "timestamp": event.get("timestamp"),
                    "message_parts": [],
                    "reasoning": None,
                    "tool_calls": [],
                    "metrics": event.get("metrics"),
                }
                group_order.append(api_call_id)

            group = groups[api_call_id]
            if kind == "message":
                text = event.get("text")
                if isinstance(text, str) and text:
                    group["message_parts"].append(text)
                if event.get("reasoning"):
                    group["reasoning"] = event["reasoning"]
                if event.get("timestamp"):
                    group["timestamp"] = event["timestamp"]
            elif kind == "tool_call":
                group["tool_calls"].append(event)
                if not group["reasoning"] and event.get("reasoning"):
                    group["reasoning"] = event["reasoning"]
                if not group.get("metrics") and event.get("metrics"):
                    group["metrics"] = event["metrics"]

        flush()
        return result

    @staticmethod
    def _metrics_from_token_count_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        info = payload.get("info")
        if not isinstance(info, dict):
            return None

        last_usage = info.get("last_token_usage")
        if not isinstance(last_usage, dict):
            return None

        prompt_tokens = last_usage.get("input_tokens")
        completion_tokens = last_usage.get("output_tokens")
        cached_tokens = last_usage.get("cached_input_tokens")
        cache_write_tokens = last_usage.get("cache_write_input_tokens")
        reasoning_tokens = last_usage.get("reasoning_output_tokens")
        total_tokens = last_usage.get("total_tokens")

        extra = {
            "reasoning_output_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }
        if cache_write_tokens is not None:
            extra["cache_write_input_tokens"] = cache_write_tokens

        return {
            "prompt_tokens": prompt_tokens if prompt_tokens else None,
            "completion_tokens": completion_tokens or None,
            "cached_tokens": cached_tokens or None,
            "extra": extra,
        }

    def _convert_event_to_step(self, event: dict[str, Any], step_id: int) -> Step:
        """Convert a normalized Codex event dictionary into an ATIF step."""
        kind = event.get("kind")
        timestamp = event.get("timestamp")

        if kind == "message":
            role = event.get("role", "user")
            text = event.get("text", "")
            reasoning = event.get("reasoning")
            source: Literal["system", "user", "agent"]
            if role == "assistant":
                source = "agent"
            elif role == "user":
                source = "user"
            else:
                source = "system"

            extra = event.get("extra")

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source=source,
                message=text,
                reasoning_content=reasoning
                if source == "agent" and reasoning
                else None,
                model_name=self.model_name
                if source == "agent" and self.model_name
                else None,
                llm_call_count=1 if source == "agent" else None,
                extra=extra if extra else None,
            )

        if kind == "tool_call":
            call_id = event.get("call_id", "")
            tool_name = event.get("tool_name", "")
            reasoning = event.get("reasoning")
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}

            tool_call = ToolCall(
                tool_call_id=call_id,
                function_name=tool_name,
                arguments=arguments,
            )

            observation: Observation | None = None
            output_text = event.get("output")
            if output_text is not None:
                observation = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=call_id or None,
                            content=output_text,
                        )
                    ]
                )

            metrics_payload = event.get("metrics")
            metrics: Metrics | None = None
            if isinstance(metrics_payload, dict):
                metrics = Metrics(**metrics_payload)

            extra: dict[str, Any] | None = None
            metadata = event.get("metadata")
            if metadata:
                extra = {"tool_metadata": metadata}
            raw_arguments = event.get("raw_arguments")
            if raw_arguments:
                extra = extra or {}
                extra["raw_arguments"] = raw_arguments
            status = event.get("status")
            if status:
                extra = extra or {}
                extra["status"] = status
            api_call_id = event.get("api_call_id")
            if api_call_id:
                extra = extra or {}
                extra["api_call_id"] = api_call_id
            codex_turn_id = event.get("codex_turn_id")
            if codex_turn_id:
                extra = extra or {}
                extra["codex_turn_id"] = codex_turn_id

            if event.get("item_type"):
                extra = extra or {}
                extra["tool_call_details"] = {
                    call_id: {
                        key: event[key]
                        for key in ("item_type", "metadata", "raw_arguments", "status")
                        if event.get(key) is not None
                    }
                }
            message = event.get("message") or ""

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source="agent",
                message=message,
                tool_calls=[tool_call],
                observation=observation,
                model_name=self.model_name if self.model_name else None,
                reasoning_content=reasoning if reasoning else None,
                metrics=metrics,
                llm_call_count=1,
                extra=extra,
            )

        if kind == "bundled":
            text = event.get("text", "")
            reasoning = event.get("reasoning")

            tool_calls: list[ToolCall] = []
            observation_results: list[ObservationResult] = []
            for tc in event.get("tool_calls", []):
                call_id = tc.get("call_id", "")
                arguments = tc.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}

                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=tc.get("tool_name", ""),
                        arguments=arguments,
                    )
                )
                if "output" in tc:
                    observation_results.append(
                        ObservationResult(
                            source_call_id=call_id or None, content=tc["output"]
                        )
                    )

            extra: dict[str, Any] | None = None
            api_call_id = event.get("api_call_id")
            if api_call_id:
                extra = {"api_call_id": api_call_id}
            codex_turn_id = event.get("codex_turn_id")
            if codex_turn_id:
                extra = extra or {}
                extra["codex_turn_id"] = codex_turn_id

            tool_details: dict[str, Any] = {}
            for tc in event.get("tool_calls", []):
                call_id = tc.get("call_id", "")
                details: dict[str, Any] = {}
                for source_key, target_key in (
                    ("metadata", "metadata"),
                    ("raw_arguments", "raw_arguments"),
                    ("item_type", "item_type"),
                    ("status", "status"),
                ):
                    value = tc.get(source_key)
                    if value:
                        details[target_key] = value
                if details:
                    tool_details[call_id] = details
            if tool_details:
                extra = extra or {}
                extra["tool_call_details"] = tool_details
                if any(d.get("status") == "incomplete" for d in tool_details.values()):
                    extra["status"] = "incomplete"

            observation = (
                Observation(results=observation_results)
                if observation_results
                else None
            )

            return Step(
                step_id=step_id,
                timestamp=event.get("timestamp"),
                source="agent",
                message=text,
                model_name=self.model_name if self.model_name else None,
                reasoning_content=reasoning if reasoning else None,
                tool_calls=tool_calls or None,
                observation=observation,
                metrics=Metrics(**event["metrics"]) if event.get("metrics") else None,
                llm_call_count=1,
                extra=extra,
            )

        raise ValueError(f"Unsupported event kind '{kind}'")

    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cache_write_tokens: int | None,
        model_name: str | None = None,
    ) -> float | None:
        """Compute one API call's cost in USD via LiteLLM's pricing logic.

        LiteLLM selects context-dependent rates from the token count for this
        individual request. Returns None when the model is missing from its
        pricing table or the calculation fails.
        """
        resolved_model_name = model_name or self.model_name
        if not resolved_model_name:
            return None

        try:
            import litellm
        except ImportError:
            self.logger.debug("litellm not available; leaving codex cost_usd as None")
            return None

        pricing_model_name: str | None = None
        for key in (
            resolved_model_name,
            resolved_model_name.split("/", 1)[-1],
        ):
            if litellm.model_cost.get(key):
                pricing_model_name = key
                break

        if pricing_model_name is None:
            self.logger.debug(
                "No LiteLLM pricing entry for model '%s'; leaving codex "
                "cost_usd as None",
                resolved_model_name,
            )
            return None

        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=pricing_model_name,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                cache_creation_input_tokens=cache_write_tokens or 0,
                cache_read_input_tokens=cached_tokens or 0,
            )
        except Exception:
            self.logger.debug(
                "Failed to calculate Codex cost for model '%s'",
                resolved_model_name,
                exc_info=True,
            )
            return None

        return float(input_cost + output_cost)

    def _convert_events_to_trajectory(
        self,
        session_dir: Path,
        *,
        session_id: str | None = None,
        raw_events: list[dict[str, Any]] | None = None,
    ) -> Trajectory | None:
        """Convert Codex session JSONL events into an ATIF trajectory."""
        if raw_events is None:
            sessions = self._read_session_archive(session_dir)
            session_id = session_id or self._trajectory_session_id
            if session_id is None:
                # Backward-compatible import of a single unambiguous old run.
                # Multiple rollouts must be bound to the captured parent ID.
                if len(sessions) != 1:
                    raise ValueError("Codex parent session identity is required")
                session_id = next(iter(sessions))
            if session_id not in sessions:
                raise ValueError("Captured Codex parent session is absent from archive")
            raw_events = sessions[session_id]["events"]

        if not raw_events:
            return None

        session_meta = next(
            (e for e in raw_events if e.get("type") == "session_meta"), None
        )
        recorded_id = self._canonical_session_id(
            (session_meta or {}).get("payload", {}).get("id")
        )
        if recorded_id is None or (
            session_id is not None and recorded_id != session_id
        ):
            raise ValueError("Codex session metadata does not match parent binding")
        session_id = recorded_id

        agent_version = "unknown"
        agent_extra: dict[str, Any] | None = None
        default_model_name: str | None = None

        if session_meta:
            payload = session_meta.get("payload", {})
            agent_version = payload.get("cli_version") or agent_version
            extra: dict[str, Any] = {}
            for key in ("originator", "cwd", "git", "instructions", "source"):
                value = payload.get(key)
                if value is not None:
                    extra[key] = value
            agent_extra = extra or None

        if agent_version == "unknown" and self._version:
            agent_version = self._version

        for event in raw_events:
            if event.get("type") == "turn_context":
                model_name = event.get("payload", {}).get("model")
                if isinstance(model_name, str):
                    default_model_name = model_name
                    break

        if default_model_name is None:
            default_model_name = self.model_name

        # normalize events to a structure suitable for conversion into Steps
        normalized_events: list[dict[str, Any]] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        pending_reasoning: str | None = None
        reasoning_timestamp: str | None = None
        codex_turn_id: str | None = None
        api_call_index = 1
        current_api_call_id = f"api_call_{api_call_index}"
        api_call_metrics: dict[str, dict[str, Any]] = {}
        saw_model_output_in_api_call = False
        tool_order_counter = 0

        def record_model_output() -> None:
            nonlocal saw_model_output_in_api_call
            saw_model_output_in_api_call = True

        def finish_api_call(token_count_payload: dict[str, Any]) -> None:
            nonlocal api_call_index, current_api_call_id, saw_model_output_in_api_call
            nonlocal tool_order_counter

            if not saw_model_output_in_api_call:
                return

            metrics = self._metrics_from_token_count_payload(token_count_payload)
            if metrics:
                cache_write_tokens = metrics["extra"].get("cache_write_input_tokens")
                metrics["cost_usd"] = self._compute_cost_from_pricing(
                    prompt_tokens=metrics.get("prompt_tokens"),
                    completion_tokens=metrics.get("completion_tokens"),
                    cached_tokens=metrics.get("cached_tokens"),
                    cache_write_tokens=cache_write_tokens,
                    model_name=default_model_name,
                )
                api_call_metrics[current_api_call_id] = metrics

            api_call_index += 1
            current_api_call_id = f"api_call_{api_call_index}"
            saw_model_output_in_api_call = False
            tool_order_counter = 0

        for event in raw_events:
            etype = event.get("type")
            payload = event.get("payload", {})
            timestamp = event.get("timestamp")
            if not isinstance(payload, dict):
                continue

            if etype == "event_msg" and isinstance(payload, dict):
                event_type = payload.get("type")
                if event_type in {"task_started", "turn_started"}:
                    turn_id = payload.get("turn_id")
                    codex_turn_id = turn_id if isinstance(turn_id, str) else None
                elif event_type in {"task_complete", "turn_complete", "turn_aborted"}:
                    codex_turn_id = None
                elif event_type == "token_count":
                    # A token_count event closes one model API call.
                    finish_api_call(payload)
                continue

            if etype == "turn_context":
                turn_id = payload.get("turn_id") if isinstance(payload, dict) else None
                if isinstance(turn_id, str) and codex_turn_id is None:
                    codex_turn_id = turn_id
                continue

            if etype != "response_item":
                continue

            payload_type = payload.get("type")
            if payload_type == "reasoning":
                reasoning_timestamp = timestamp
                summary = payload.get("summary")
                if isinstance(summary, list) and summary:
                    reasoning_parts: list[str] = []
                    for item in summary:
                        if isinstance(item, str):
                            reasoning_parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                reasoning_parts.append(text)
                    pending_reasoning = (
                        "\n".join(reasoning_parts) if reasoning_parts else None
                    )
                else:
                    pending_reasoning = None
                continue

            if payload_type == "message":
                content = payload.get("content", [])
                text = (
                    self._extract_message_text(content)
                    if isinstance(content, list)
                    else ""
                )
                normalized_events.append(
                    {
                        "kind": "message",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "timestamp": timestamp,
                        "role": payload.get("role", "user"),
                        "text": text,
                        "reasoning": pending_reasoning
                        if payload.get("role") == "assistant"
                        else None,
                    }
                )
                if payload.get("role") == "assistant":
                    record_model_output()
                pending_reasoning = None
                continue

            if payload_type == "web_search_call":
                action = payload.get("action") or {}
                action_type = action.get("type", "")
                arguments: dict[str, Any] = {"action_type": action_type}
                if "query" in action:
                    arguments["query"] = action["query"]
                if "queries" in action:
                    arguments["queries"] = action["queries"]
                if "url" in action:
                    arguments["url"] = action["url"]

                native_call_id = payload.get("id")
                call_id = (
                    native_call_id
                    if isinstance(native_call_id, str) and native_call_id.strip()
                    else f"{current_api_call_id}_web_search_{tool_order_counter}"
                )

                normalized_events.append(
                    {
                        "kind": "tool_call",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "tool_order": tool_order_counter,
                        "timestamp": timestamp,
                        "call_id": call_id,
                        "tool_name": "web_search_call",
                        "arguments": arguments,
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": payload.get("status"),
                        "message": None,
                    }
                )
                tool_order_counter += 1
                record_model_output()
                pending_reasoning = None
                continue

            if payload_type in {"function_call", "custom_tool_call"}:
                call_id = payload.get("call_id")
                if not call_id:
                    continue

                raw_args_key = (
                    "arguments" if payload_type == "function_call" else "input"
                )
                raw_arguments = payload.get(raw_args_key)
                try:
                    parsed_args = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError):
                    if isinstance(raw_arguments, str):
                        parsed_args = {"input": raw_arguments}
                    elif raw_arguments is None:
                        parsed_args = {}
                    else:
                        parsed_args = {"value": raw_arguments}

                pending_calls[call_id] = {
                    "kind": "tool_call",
                    "api_call_id": current_api_call_id,
                    "codex_turn_id": codex_turn_id,
                    "tool_order": tool_order_counter,
                    "timestamp": timestamp,
                    "call_id": call_id,
                    "tool_name": payload.get("name") or "",
                    "arguments": parsed_args,
                    "raw_arguments": raw_arguments,
                    "item_type": payload_type,
                    "reasoning": pending_reasoning,
                    "status": "incomplete",
                    "message": None,
                }
                # Persist calls at dispatch time, including calls interrupted
                # before their output. Updating this dictionary on completion
                # also preserves the order of concurrent calls.
                normalized_events.append(pending_calls[call_id])
                tool_order_counter += 1
                record_model_output()
                pending_reasoning = None
                continue

            if payload_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id")
                output, metadata = self._parse_output_blob(payload.get("output"))

                call_info = pending_calls.pop(call_id, None) if call_id else None

                if call_info is None:
                    call_info = {
                        "kind": "tool_call",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "tool_order": tool_order_counter,
                        "timestamp": timestamp,
                        "call_id": call_id or "",
                        "tool_name": payload.get("name", "") or "",
                        "arguments": {},
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": None,
                        "message": None,
                    }
                    normalized_events.append(call_info)
                    tool_order_counter += 1

                call_info["output"] = output
                call_info["metadata"] = metadata
                call_info["status"] = "completed"
                call_info["timestamp"] = call_info.get("timestamp") or timestamp
                pending_reasoning = None
                continue

        if pending_reasoning:
            # A canceled model turn can end after an emitted summary and
            # before its assistant message/tool. Preserve that emitted text.
            normalized_events.append(
                {
                    "kind": "message",
                    "role": "assistant",
                    "text": "",
                    "timestamp": reasoning_timestamp,
                    "reasoning": pending_reasoning,
                    "extra": {
                        "status": "incomplete",
                        "event_type": "reasoning_summary",
                    },
                }
            )
        for norm_event in normalized_events:
            event_api_call_id = norm_event.get("api_call_id")
            if (
                isinstance(event_api_call_id, str)
                and event_api_call_id in api_call_metrics
            ):
                norm_event["metrics"] = api_call_metrics[event_api_call_id]

        # Legacy/partial logs without API boundaries cannot safely be grouped.
        grouped_events = (
            self._group_events_by_api_call_id(normalized_events)
            if api_call_index > 1
            else normalized_events
        )

        steps: list[Step] = []
        skipped_events = 0
        for idx, norm_event in enumerate(grouped_events, start=1):
            try:
                step = self._convert_event_to_step(norm_event, idx)
            except ValueError as exc:
                skipped_events += 1
                self.logger.debug(f"Skipping event during step conversion: {exc}")
                continue

            # Provide default model name if not set for agent steps
            if step.source == "agent" and not step.model_name and default_model_name:
                step.model_name = default_model_name

            steps.append(step)

        if not steps:
            self.logger.debug("No valid steps produced from Codex session")
            return None

        estimated_total_cost_usd: float | None = 0.0 if api_call_metrics else None
        for metrics in api_call_metrics.values():
            call_cost = metrics["cost_usd"]
            if call_cost is None:
                estimated_total_cost_usd = None
                break
            if estimated_total_cost_usd is not None:
                estimated_total_cost_usd += call_cost

        # Extract final metrics from the last token_count event with totals
        total_metrics: FinalMetrics | None = None
        for event in reversed(raw_events):
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue

            info = payload.get("info")
            if not isinstance(info, dict):
                continue

            total_usage = info.get("total_token_usage")
            if not isinstance(total_usage, dict):
                continue

            prompt_tokens = total_usage.get("input_tokens")
            completion_tokens = total_usage.get("output_tokens")
            reasoning_tokens = total_usage.get("reasoning_output_tokens")
            cached_tokens = total_usage.get("cached_input_tokens")
            cache_write_tokens = total_usage.get("cache_write_input_tokens")
            overall_tokens = total_usage.get("total_tokens")

            # Codex CLI does not include cost in token_count events, so fall
            # back to LiteLLM's pricing table when the field is absent. Use
            # explicit `is None` checks so a legitimate $0 cost reported by
            # the CLI isn't mistaken for a missing field.
            total_cost_usd = info.get("total_cost")
            if total_cost_usd is None:
                total_cost_usd = info.get("cost_usd")
            if total_cost_usd is None:
                total_cost_usd = estimated_total_cost_usd

            final_extra: dict[str, Any] | None = {
                "reasoning_output_tokens": reasoning_tokens,
                "total_tokens": overall_tokens,
                "last_token_usage": info.get("last_token_usage"),
            }
            if cache_write_tokens is not None:
                final_extra["total_cache_write_input_tokens"] = cache_write_tokens

            total_metrics = FinalMetrics(
                total_prompt_tokens=prompt_tokens if prompt_tokens else None,
                total_completion_tokens=completion_tokens or None,
                total_cached_tokens=cached_tokens or None,
                total_cost_usd=total_cost_usd,
                total_steps=len(steps),
                extra=final_extra,
            )
            break

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name="codex",
                version=agent_version,
                model_name=default_model_name,
                extra=agent_extra,
            ),
            steps=steps,
            final_metrics=total_metrics,
            trajectory_id=session_id,
            extra={
                "incomplete_tool_calls": sorted(pending_calls),
                "raw_event_count": len(raw_events),
                "conversion_skipped_events": skipped_events,
                "emitted_reasoning_only": True,
            },
        )

        return trajectory

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after Codex finishes executing.

        Converts the Codex session JSONL file into an ATIF trajectory, persists it,
        and propagates usage metrics back to the Harbor context.
        """
        try:
            trajectory = self._write_trajectory_bundle()
        except Exception as exc:
            self.logger.exception(
                "Failed to convert Codex session archive; raw history retained"
            )
            context.metadata = {
                **(context.metadata or {}),
                "codex_trajectory_retention": {
                    "status": "unavailable",
                    "reason": type(exc).__name__,
                    "raw_archive_retained": True,
                },
            }
            return
        if trajectory is None:
            return

        if trajectory.final_metrics:
            metrics = trajectory.final_metrics
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0

    def _write_trajectory_file(self, filename: str, data: bytes) -> None:
        temporary = self.logs_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.logs_dir / filename)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_trajectory_bundle(self) -> Trajectory | None:
        session_dir = self._get_session_dir()
        if session_dir is None:
            return None
        sessions = self._read_session_archive(session_dir)
        if not sessions:
            return None
        parent_id = self._trajectory_session_id
        if parent_id is None:
            # exec can raise before returning stdout. Recover only a unique
            # CLI parent ID from the original append-only stream on that path.
            output_file = self.logs_dir / self._OUTPUT_FILENAME
            if output_file.exists() and not output_file.is_symlink():
                info = output_file.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size > self._MAX_SESSION_BYTES
                ):
                    raise ValueError("Codex output is not a bounded regular log")
                fd = os.open(output_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as handle:
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode) or (
                        before.st_dev,
                        before.st_ino,
                    ) != (info.st_dev, info.st_ino):
                        raise ValueError("Codex output changed during collection")
                    ids = set()
                    consumed = 0
                    discard_line = False
                    while line := handle.readline(1024 * 1024 + 1):
                        consumed += len(line)
                        if consumed > self._MAX_SESSION_BYTES:
                            raise ValueError("Codex output size limit exceeded")
                        if discard_line or len(line) > 1024 * 1024:
                            discard_line = not line.endswith(b"\n")
                            continue
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if (
                            isinstance(event, dict)
                            and event.get("type") == "thread.started"
                        ):
                            recorded = self._canonical_session_id(
                                event.get("thread_id")
                            )
                            if recorded:
                                ids.add(recorded)
                    after = os.fstat(handle.fileno())
                    if (before.st_size, before.st_mtime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise ValueError("Codex output changed during collection")
                    if len(ids) == 1:
                        parent_id = next(iter(ids))
            if parent_id is None and len(sessions) == 1:
                parent_id = next(iter(sessions))
        if parent_id not in sessions:
            raise ValueError("Captured Codex parent session is absent from archive")

        parent_links: dict[str, str | None] = {}
        trajectories: dict[str, Trajectory] = {}
        for session_id, group in sessions.items():
            meta = next(e for e in group["events"] if e.get("type") == "session_meta")[
                "payload"
            ]
            source = meta.get("source")
            spawn = (
                source.get("subagent", {}).get("thread_spawn", {})
                if isinstance(source, dict) and isinstance(source.get("subagent"), dict)
                else {}
            )
            parent_links[session_id] = (
                self._canonical_session_id(spawn.get("parent_thread_id"))
                if isinstance(spawn, dict)
                else None
            )
            converted = self._convert_events_to_trajectory(
                session_dir, session_id=session_id, raw_events=group["events"]
            )
            if converted is not None:
                trajectories[session_id] = converted

        def filename(session_id: str) -> str:
            return (
                "trajectory.json"
                if session_id == parent_id
                else f"trajectory-{session_id}.json"
            )

        def is_child(session_id: str) -> bool:
            seen = {session_id}
            ancestor = parent_links.get(session_id)
            while ancestor and ancestor not in seen:
                if ancestor == parent_id:
                    return True
                seen.add(ancestor)
                ancestor = parent_links.get(ancestor)
            return False

        for session_id, converted in trajectories.items():
            refs = [
                SubagentTrajectoryRef(
                    session_id=child_id,
                    trajectory_id=child_id,
                    trajectory_path=filename(child_id),
                )
                for child_id in sorted(trajectories)
                if parent_links.get(child_id) == session_id
            ]
            converted.extra = {
                **(converted.extra or {}),
                "trajectory_index": "trajectory_index.json",
                "subagent_trajectory_refs": [
                    ref.model_dump(exclude_none=True) for ref in refs
                ],
            }
            for step in converted.steps:
                if step.observation is None:
                    continue
                for result in step.observation.results:
                    if isinstance(result.content, str):
                        matching = [
                            ref
                            for ref in refs
                            if ref.session_id is not None
                            and ref.session_id in result.content
                        ]
                        if matching:
                            result.subagent_trajectory_ref = matching

        index: dict[str, Any] = {
            "schema": self._TRAJECTORY_INDEX_SCHEMA,
            "parent_session_id": parent_id,
            "score_authority": False,
            "trajectories": [],
        }
        for session_id, group in sorted(sessions.items()):
            converted = trajectories.get(session_id)
            entry: dict[str, Any] = {
                "session_id": session_id,
                "parent_session_id": parent_links[session_id],
                "role": "parent"
                if session_id == parent_id
                else ("child" if is_child(session_id) else "other"),
                "trajectory_file": None,
                "sha256": None,
                "size_bytes": None,
                "raw_files": group["raw_files"],
                "malformed_lines": group["malformed_lines"],
                "status": "partial",
            }
            if converted is not None:
                data = format_trajectory_json(converted.to_json_dict()).encode("utf-8")
                if len(data) > self._MAX_SESSION_FILE_BYTES:
                    raise ValueError(
                        "Codex ATIF trajectory size limit exceeded; raw archive retained"
                    )
                self._write_trajectory_file(filename(session_id), data)
                entry.update(
                    trajectory_file=filename(session_id),
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                )
                completed = (
                    self._last_invocation_completed
                    if session_id == parent_id
                    else any(
                        e.get("type") == "event_msg"
                        and isinstance(e.get("payload"), dict)
                        and e["payload"].get("type")
                        in {"task_complete", "turn_complete"}
                        for e in group["events"][-8:]
                    )
                )
                if (
                    completed
                    and not (converted.extra or {}).get("incomplete_tool_calls")
                    and not (converted.extra or {}).get("conversion_skipped_events")
                    and not any(
                        (step.extra or {}).get("status") == "incomplete"
                        for step in converted.steps
                    )
                    and not group["malformed_lines"]
                ):
                    entry["status"] = "complete"
            index["trajectories"].append(entry)
        index_data = (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        if len(index_data) > 1024 * 1024:
            raise ValueError("Codex trajectory index size limit exceeded")
        self._write_trajectory_file("trajectory_index.json", index_data)
        return trajectories.get(parent_id)

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Codex's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def _load_base_config(self) -> dict[str, Any]:
        """Load a Codex TOML file or validate an inline JSON configuration."""
        config_source = self.config_source
        if config_source is None:
            return {}
        if isinstance(config_source, dict):
            try:
                rendered = toml.dumps(config_source)
                round_tripped = toml.loads(rendered)
            except (TypeError, ValueError, toml.TomlDecodeError) as exc:
                raise ValueError(
                    f"Invalid inline Codex config for TOML conversion: {exc}"
                ) from exc
            if round_tripped != config_source:
                raise ValueError(
                    "Invalid inline Codex config: JSON object cannot be "
                    "represented losslessly as TOML"
                )
            return config_source

        try:
            config = toml.loads(config_source.read_text())
        except (OSError, UnicodeError, toml.TomlDecodeError) as exc:
            raise ValueError(
                f"Invalid Codex config file {config_source}: {exc}"
            ) from exc

        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid Codex config file {config_source}: expected a TOML table"
            )
        return config

    def _build_effective_config(
        self, openai_base_url: str | None = None
    ) -> dict[str, Any]:
        """Merge Harbor runtime configuration on top of the user's base config."""
        config = deepcopy(self._base_config)

        if openai_base_url:
            configured_base_url = config.get("openai_base_url")
            if configured_base_url not in (None, openai_base_url):
                self.logger.warning(
                    "OPENAI_BASE_URL overrides openai_base_url from Codex config.toml"
                )
            config["openai_base_url"] = openai_base_url

        if not self.mcp_servers:
            return config

        mcp_servers = config.setdefault("mcp_servers", {})
        if not isinstance(mcp_servers, dict):
            raise ValueError(
                "Invalid Codex config: mcp_servers must be a TOML table when "
                "Harbor task MCP servers are configured"
            )

        for server in self.mcp_servers:
            if server.name in mcp_servers:
                self.logger.warning(
                    "Harbor MCP server '%s' overrides the same server in Codex "
                    "config.toml",
                    server.name,
                )

            if server.transport == "stdio":
                mcp_servers[server.name] = {
                    "command": server.command,
                    "args": list(server.args),
                }
            else:
                mcp_servers[server.name] = {"url": server.url}

        return config

    async def _upload_effective_config(
        self,
        environment: BaseEnvironment,
        config: dict[str, Any],
        remote_path: str,
    ) -> None:
        """Render and upload the temporary effective Codex configuration."""
        if not config:
            return

        await self._upload_config_text(
            environment,
            content=toml.dumps(config),
            remote_path=remote_path,
            filename="config.toml",
        )

    def _resolve_auth_json_path(self) -> Path | None:
        """Resolve which auth.json to inject, if any.

        Defaults to None (OPENAI_API_KEY auth). Opt into auth.json auth via:
          - CODEX_AUTH_JSON_PATH=<path> → use that specific file
          - CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        """
        explicit = self._get_env("CODEX_AUTH_JSON_PATH")
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise ValueError(
                    f"CODEX_AUTH_JSON_PATH points to non-existent file: {explicit}"
                )
            return p

        if parse_bool_env_value(
            self._get_env("CODEX_FORCE_AUTH_JSON"),
            name="CODEX_FORCE_AUTH_JSON",
            default=False,
        ):
            default = Path.home() / ".codex" / "auth.json"
            if not default.is_file():
                raise ValueError(
                    f"CODEX_FORCE_AUTH_JSON is set but {default} does not exist"
                )
            return default

        return None

    def _persistent_work_budget_sec(
        self,
        context: AgentContext,
        *,
        run_started_at: float,
    ) -> int | None:
        """Return the cleanly usable wall-clock budget for persistent Codex turns."""
        if not self._continue_until_timeout:
            return None

        metadata = context.metadata
        if not isinstance(metadata, dict):
            return None
        raw_timeout = metadata.get(self._HARBOR_TIMEOUT_METADATA_KEY)
        if raw_timeout is None or isinstance(raw_timeout, bool):
            return None
        try:
            timeout_sec = float(raw_timeout)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            return None

        elapsed_sec = time.monotonic() - run_started_at
        work_budget_sec = int(timeout_sec - elapsed_sec - self._shutdown_grace_sec)
        return work_budget_sec if work_budget_sec > 0 else None

    def _persistent_codex_command(
        self,
        *,
        escaped_instruction: str,
        model: str,
        cli_flags_arg: str,
        work_budget_sec: int,
    ) -> str:
        """Build a same-session loop that ends cleanly just before Harbor's timeout."""
        output_path = shlex.quote(
            (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        )
        continuation_prompt = shlex.quote(self._CONTINUATION_PROMPT)
        common_flags = (
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {shlex.quote(model)} "
            "--json "
            "--enable unified_exec "
            f"{cli_flags_arg}"
        )
        first_command = f"codex exec {common_flags}-- {escaped_instruction}"
        resume_command = (
            f"codex exec resume --last {common_flags}-- {continuation_prompt}"
        )

        return (
            "umask 077; "
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f": > {output_path}; "
            f"codex_deadline=$(( $(date +%s) + {work_budget_sec} )); "
            "codex_turn=0; "
            "codex_idle_turns=0; "
            "codex_workspace_fingerprint() { "
            "if [ -d /workspace ]; then "
            "find /workspace -xdev -type f -size -2097153c "
            "! -path '/workspace/.git/*' -print0 2>/dev/null | "
            "sort -z | xargs -0r sha256sum -- 2>/dev/null | sha256sum | cut -d' ' -f1; "
            "else printf '%s\\n' workspace-unavailable; fi; "
            "}; "
            "codex_workspace_before=$(codex_workspace_fingerprint); "
            "while :; do "
            "codex_remaining=$(( codex_deadline - $(date +%s) )); "
            'if [ "$codex_remaining" -le 0 ]; then break; fi; '
            'if [ "$codex_turn" -eq 0 ]; then '
            f'timeout --signal=TERM --kill-after=5s "${{codex_remaining}}s" '
            f"{first_command} 2>&1 </dev/null | tee -a {output_path}; "
            "else "
            f'timeout --signal=TERM --kill-after=5s "${{codex_remaining}}s" '
            f"{resume_command} 2>&1 </dev/null | tee -a {output_path}; "
            "fi; "
            "codex_status=${PIPESTATUS[0]}; "
            'if [ "$codex_status" -ne 0 ]; then '
            "codex_now=$(date +%s); "
            'if { [ "$codex_status" -eq 124 ] || '
            '[ "$codex_status" -eq 137 ] || '
            '[ "$codex_status" -eq 143 ]; } && '
            '[ "$codex_now" -ge "$((codex_deadline - 1))" ]; then '
            "break; "
            "fi; "
            'exit "$codex_status"; '
            "fi; "
            "codex_workspace_after=$(codex_workspace_fingerprint); "
            'if [ "$codex_turn" -gt 0 ] && '
            '[ "$codex_workspace_after" = "$codex_workspace_before" ]; then '
            "codex_idle_turns=$((codex_idle_turns + 1)); "
            "else codex_idle_turns=0; fi; "
            "codex_workspace_before=$codex_workspace_after; "
            "codex_turn=$((codex_turn + 1)); "
            f'if [ "$codex_idle_turns" -ge {self._max_idle_turns} ]; then break; fi; '
            "done"
        )

    def _is_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.startswith("openrouter/"))

    def _suspendable_codex_command(self, *, session_id, prompt, model, cli_flags_arg):
        """One completed call, never an internal resume/poll loop.

        The trusted host controller validates session IDs and decides whether
        another call may start. Existing provider/model routing is unchanged.
        """
        import uuid

        if session_id is not None:
            if (
                not isinstance(session_id, str)
                or str(uuid.UUID(session_id)) != session_id
            ):
                raise ValueError("Invalid Codex continuation identity")
        invocation = "codex exec "
        if session_id is not None:
            invocation += f"resume {shlex.quote(session_id)} "
        return (
            "umask 077; "
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            + invocation
            + "--dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "
            + f"--model {shlex.quote(model)} --json --enable unified_exec "
            + f"{cli_flags_arg}-- {shlex.quote(prompt)} "
            + f"2>&1 </dev/null | tee -a {shlex.quote((EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix())}"
        )

    def _codex_cli_model(self) -> str:
        if not self.model_name:
            raise ValueError("Model name is required")
        if self._is_openrouter_model():
            model = self.model_name.split("/", 1)[1]
            if not model:
                raise ValueError(
                    "OpenRouter model name must include a provider/model slug"
                )
            return model
        return self.model_name.split("/")[-1]

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        run_started_at = time.monotonic()
        escaped_instruction = shlex.quote(instruction)

        use_openrouter = self._is_openrouter_model()
        model = self._codex_cli_model()

        # Build command with optional CLI config flags from descriptors.
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        # Auth resolution:
        #   1. CODEX_AUTH_JSON_PATH=<path> → use that specific auth.json file
        #   2. CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        #   3. Default: use OPENAI_API_KEY
        auth_json_path = None if use_openrouter else self._resolve_auth_json_path()
        access = self.model_connection

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()
        remote_config_path = (self._REMOTE_CODEX_HOME / "config.toml").as_posix()
        agent_sessions_dir = (self.environment_logs_dir / "sessions").as_posix()

        env: dict[str, str] = {
            "CODEX_HOME": remote_codex_home,
        }
        if use_openrouter:
            openrouter_key = self._get_env("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for openrouter/* models"
                )
            env["OPENROUTER_API_KEY"] = openrouter_key

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}\n"
                "umask 077\n"
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)}\n'
                f'chmod 700 "$CODEX_HOME" {shlex.quote(remote_secrets_dir)}'
            ),
            env=env,
        )

        if self._load:
            await self._seed_load_trajectory(environment)
        resume_run = self._resume or self._load

        if auth_json_path:
            self.logger.debug("Codex auth: using auth.json from %s", auth_json_path)
            await environment.upload_file(auth_json_path, remote_auth_path)
            # upload_file copies as root. Lock the mode before handing ownership
            # to the agent so no other container UID can read the credential.
            secure_auth_command = f"chmod 600 {shlex.quote(remote_auth_path)}"
            if environment.default_user is not None:
                secure_auth_command += (
                    " && chown "
                    f"{shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(remote_auth_path)}"
                )
            await self.exec_as_root(
                environment,
                command=secure_auth_command,
            )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"\n'
            )
        elif use_openrouter:
            self.logger.debug("Codex auth: using OpenRouter custom provider")
            setup_command = ""
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env["OPENAI_API_KEY"] = access.api_key or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"chmod 600 {shlex.quote(remote_auth_path)}\n"
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"\n'
            )

        openai_base_url = access.configured_base_url
        if openai_base_url:
            env["OPENAI_BASE_URL"] = openai_base_url

        # codex 0.118.0 only honors openai_base_url from config.toml, not the env var.
        effective_config = self._build_effective_config(openai_base_url)
        if use_openrouter:
            effective_config["preferred_auth_method"] = "apikey"
            effective_config["model_provider"] = "openrouter"
            effective_config.setdefault("model_providers", {})["openrouter"] = {
                "name": "OpenRouter",
                "base_url": self._openrouter_base_url,
                "env_key": "OPENROUTER_API_KEY",
                "wire_api": "responses",
            }
        await self._upload_effective_config(
            environment, effective_config, remote_config_path
        )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        if resume_run:
            setup_command += (
                f"\nif [ ! -d {shlex.quote(agent_sessions_dir)} ]; then\n"
                '  echo "Cannot resume Codex: no previous session logs found" >&2\n'
                "  exit 1\n"
                "fi\n"
                'rm -rf "$CODEX_HOME/sessions"\n'
                f"cp -R {shlex.quote(agent_sessions_dir)} "
                '"$CODEX_HOME/sessions"'
            )

        setup_command = "umask 077\n" + setup_command
        setup_command += (
            f'\nchmod 700 "$CODEX_HOME" {shlex.quote(remote_secrets_dir)}\n'
            f"if [ -f {shlex.quote(remote_auth_path)} ]; then\n"
            f"  chmod 600 {shlex.quote(remote_auth_path)}\n"
            "fi\n"
            'if [ -f "$CODEX_HOME/config.toml" ]; then\n'
            '  chmod 600 "$CODEX_HOME/config.toml"\n'
            "fi"
        )

        if setup_command.strip():
            await self.exec_as_agent(
                environment,
                command=setup_command,
                env=env,
            )
        persistent_work_budget_sec = self._persistent_work_budget_sec(
            context,
            run_started_at=run_started_at,
        )
        suspension = getattr_static(environment, "_harbor_probe_suspension", None)
        try:
            if suspension is not None:

                async def invoke(session_id, prompt, timeout):
                    self._last_invocation_completed = False
                    result = await self.exec_as_agent(
                        environment,
                        command=self._suspendable_codex_command(
                            session_id=session_id,
                            prompt=prompt,
                            model=model,
                            cli_flags_arg=cli_flags_arg,
                        ),
                        env=env,
                        timeout_sec=max(1, math.ceil(timeout)),
                    )
                    self._capture_session_id(result.stdout, expected=session_id)
                    self._last_invocation_completed = True
                    # The controller pauses Docker only after invoke returns.
                    # Export here while learner exec is still available.
                    await self._export_codex_sessions(environment, env)
                    return result.stdout

                await suspension.run_turns(invoke, instruction)
            elif persistent_work_budget_sec is not None:
                result = await self.exec_as_agent(
                    environment,
                    command=self._persistent_codex_command(
                        escaped_instruction=escaped_instruction,
                        model=model,
                        cli_flags_arg=cli_flags_arg,
                        work_budget_sec=persistent_work_budget_sec,
                    ),
                    env=env,
                )
                self._capture_session_id(result.stdout)
                self._last_invocation_completed = True
            else:
                result = await self.exec_as_agent(
                    environment,
                    command=(
                        "umask 077; "
                        "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                        f"codex exec {'resume --last ' if resume_run else ''}"
                        "--dangerously-bypass-approvals-and-sandbox "
                        "--skip-git-repo-check "
                        f"--model {model} "
                        "--json "
                        "--enable unified_exec "
                        f"{cli_flags_arg}"
                        "-- "  # end of flags
                        f"{escaped_instruction} "
                        f"2>&1 </dev/null | tee {
                            EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
                        }"
                    ),
                    env=env,
                )
                self._capture_session_id(result.stdout)
                self._last_invocation_completed = True
        finally:
            # A failed/cancelled handoff can deliberately leave the complete
            # learner container frozen. Never exec into it for cleanup: the
            # outer lifecycle will stop it before collection/final verification.
            if suspension is None or not suspension.learner_paused:
                try:
                    if not self._session_export_failed:
                        await self._cleanup_codex_environment(
                            environment, env, remote_secrets_dir
                        )
                finally:
                    if self._session_export_failed:
                        context.metadata = {
                            **(context.metadata or {}),
                            **(self.history_recovery_metadata() or {}),
                        }

    def history_recovery_metadata(self) -> dict[str, dict[str, Any]] | None:
        """Ask teardown to retain uncopied raw history, including partial turns."""
        if not self._session_export_failed:
            return None
        return {
            "codex_trajectory_retention": {
                "status": "unavailable",
                "reason": "session_export_failed",
                "source_history_preserved": True,
                "source_location": "learner_codex_home",
                "recovery_required_before_environment_cleanup": True,
                "previous_snapshot_exported": self._completed_session_exports > 0,
            }
        }

    def _session_export_command(self) -> str:
        """Portable bounded copy; never follow links or destroy the prior archive."""
        logs = shlex.quote(EnvironmentPaths.agent_dir.as_posix())
        required = "1" if self._trajectory_session_id else "0"
        return f"""set -e
umask 077
codex_archive_root={logs}
mkdir -p "$codex_archive_root"
if [ -L "$CODEX_HOME" ] || [ -L "$CODEX_HOME/sessions" ]; then
  echo 'Codex session export rejects directory links' >&2; exit 1
fi
if [ ! -d "$CODEX_HOME/sessions" ]; then
  if [ {required} = 1 ]; then echo 'Codex session export is missing the parent history' >&2; exit 1; fi
  exit 0
fi
if [ {required} = 1 ] && [ -z "$(find "$CODEX_HOME/sessions" -type f -name '*.jsonl' -print -quit)" ]; then
  echo 'Codex session export has no rollout history' >&2; exit 1
fi
codex_validate_archive() {{
  [ -z "$(find "$1" ! -type d ! -type f -print -quit)" ] || return 1
  [ -z "$(find "$1" -type f -links +1 -print -quit)" ] || return 1
  [ -z "$(find "$1" -type f -size +{self._MAX_SESSION_FILE_BYTES}c -print -quit)" ] || return 1
  [ "$(find "$1" -type d | wc -l)" -le 1024 ] || return 1
  find "$1" -type f -exec stat -c '%s' {{}} + | awk '
    {{ count++; bytes += $1 }}
    END {{ if (count > {self._MAX_SESSION_FILES} || bytes > {self._MAX_SESSION_BYTES}) exit 1 }}'
}}
codex_validate_archive "$CODEX_HOME/sessions" || {{ echo 'Codex session export exceeds safe archive bounds' >&2; exit 1; }}
codex_archive_stage=$(mktemp -d "$codex_archive_root/.codex-sessions-export.XXXXXXXX")
codex_archive_backup="$codex_archive_stage/previous"
codex_archive_destination="$codex_archive_root/sessions"
codex_archive_committed=0
codex_restore_archive() {{
  if [ "$codex_archive_committed" = 0 ] && [ -d "$codex_archive_backup" ] && [ ! -e "$codex_archive_destination" ]; then
    mv "$codex_archive_backup" "$codex_archive_destination" || {{ echo 'Codex previous archive preserved in export staging' >&2; return; }}
  fi
  rm -rf "$codex_archive_stage"
}}
trap codex_restore_archive EXIT
cp -R -P "$CODEX_HOME/sessions" "$codex_archive_stage/sessions"
codex_validate_archive "$codex_archive_stage/sessions" || {{ echo 'Codex copied session archive failed validation' >&2; exit 1; }}
if [ -e "$codex_archive_destination" ] || [ -L "$codex_archive_destination" ]; then
  [ -d "$codex_archive_destination" ] && [ ! -L "$codex_archive_destination" ] || exit 1
  mv "$codex_archive_destination" "$codex_archive_backup"
fi
mv "$codex_archive_stage/sessions" "$codex_archive_destination"
codex_archive_committed=1
"""

    async def _export_codex_sessions(self, environment, env) -> None:
        try:
            await self.exec_as_agent(
                environment,
                command=self._session_export_command(),
                env=env,
                timeout_sec=60,
            )
            self._completed_session_exports += 1
        except BaseException as exc:
            self._session_export_failed = True
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                "Codex session export failed; original CODEX_HOME history was preserved "
                "in the learner container and requires recovery before environment cleanup"
            ) from exc

    async def _cleanup_codex_environment(self, environment, env, remote_secrets_dir):
        # Export failures propagate and leave the original sessions available
        # to operator recovery. Never remove CODEX_HOME after a failed copy.
        await self._export_codex_sessions(environment, env)
        # cleanup - best effort
        try:
            await self.exec_as_agent(
                environment,
                command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                env=env,
            )
        except Exception:
            pass
