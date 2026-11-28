from inspect import getattr_static

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Annotated, Any, Literal, override
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    filter_jsonl_events,
    with_prompt_template,
)
from harbor.agents.options import Cli, Env, InstalledAgentOptions
from harbor.agents.model_connection import (
    ResolvedModelConnection,
    ModelConnectionSpec,
    without_inferred_endpoint,
)
from harbor.agents.protocols import ACPAgentMixin
from harbor.agents.installed.claude_openrouter import GLM_MODEL, MODEL, SONNET_MODEL, model_policy
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.bridge import BridgeKind
from harbor.models.trajectories import (
    Agent,
    ContentPart,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.trial.errors import AgentCleanupError, AgentEvidenceIntegrityError
from harbor.utils.env import parse_bool_env_value
from harbor.utils.scripts import (
    ensure_acp_node_command,
    pinned_bin_wrapper_command,
    safe_bin_symlink_command,
)


class ClaudeCodeOptions(InstalledAgentOptions):
    max_turns: Annotated[
        int | None, Cli("--max-turns", fallback="CLAUDE_CODE_MAX_TURNS")
    ] = Field(default=None, description="Maximum agent turns.")
    reasoning_effort: Annotated[
        Literal["low", "medium", "high", "xhigh", "max"] | None,
        Cli("--effort", fallback="CLAUDE_CODE_EFFORT_LEVEL"),
    ] = Field(default=None, description="Model reasoning effort.")
    thinking: Annotated[
        Literal["enabled", "adaptive", "disabled"] | None, Cli("--thinking")
    ] = Field(default=None, description="Native thinking mode.")
    model_http_retries: int = Field(
        default=0, strict=True, ge=0, le=1,
        description="Guarded GLM only: one pre-stream HTTP429/520 retry when explicitly enabled.",
    )
    thinking_display: Annotated[
        Literal["summarized", "omitted"] | None,
        Cli("--thinking-display"),
    ] = Field(default=None, description="How thinking is displayed.")
    max_budget_usd: Annotated[str | None, Cli("--max-budget-usd")] = Field(
        default=None, description="Maximum spend in USD."
    )
    fallback_model: Annotated[str | None, Cli("--fallback-model")] = Field(
        default=None, description="Fallback model name."
    )
    append_system_prompt: Annotated[str | None, Cli("--append-system-prompt")] = Field(
        default=None, description="Text appended to the system prompt."
    )
    allowed_tools: Annotated[str | None, Cli("--allowedTools")] = Field(
        default=None, description="Comma-separated allowed tools."
    )
    disallowed_tools: Annotated[str | None, Cli("--disallowedTools")] = Field(
        default=None, description="Comma-separated disallowed tools."
    )
    disable_web_search: bool = Field(
        default=False, description="Disallow the WebSearch and WebFetch tools."
    )

    @model_validator(mode="after")
    def _disallow_web_tools(self):
        if self.disable_web_search:
            tools = [self.disallowed_tools] if self.disallowed_tools else []
            self.disallowed_tools = ",".join([*tools, "WebSearch", "WebFetch"])
        return self

    permission_mode: Annotated[
        Literal[
            "default",
            "acceptEdits",
            "plan",
            "auto",
            "dontAsk",
            "bypassPermissions",
        ]
        | None,
        Cli("--permission-mode", format="--permission-mode={value}"),
    ] = Field(default="bypassPermissions", description="Permission mode.")
    max_thinking_tokens: Annotated[
        int | None,
        Env("MAX_THINKING_TOKENS", fallback="MAX_THINKING_TOKENS"),
    ] = Field(default=None, description="Maximum thinking tokens.")
    memory_dir: str | None = Field(
        default=None, description="Host directory uploaded as Claude memory."
    )


class ClaudeCode(BaseInstalledAgent, ACPAgentMixin):
    capabilities = AgentCapabilities(
        atif=True,
        resume=True,
        load_native_trajectory=True,
        load_atif_trajectory=True,
        handoff=True,
        native_config=True,
        bridges=frozenset({BridgeKind.ACP}),
        skills=True,
        mcp_servers=True,
    )
    MODEL_CONNECTION = ModelConnectionSpec(
        default_provider="anthropic",
        api_key_envs=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        base_url_envs=("ANTHROPIC_BASE_URL",),
    )

    @property
    @override
    def model_connection(self) -> ResolvedModelConnection:
        # Bedrock / bearer-token routing has no Anthropic API endpoint to infer.
        access = super().model_connection
        if self._get_env("AWS_BEARER_TOKEN_BEDROCK"):
            return without_inferred_endpoint(access)
        try:
            if parse_bool_env_value(
                self._get_env("CLAUDE_CODE_USE_BEDROCK"),
                name="CLAUDE_CODE_USE_BEDROCK",
                default=False,
            ):
                return without_inferred_endpoint(access)
        except ValueError:
            self.logger.debug(
                "Ignoring invalid boolean environment value CLAUDE_CODE_USE_BEDROCK"
            )
        return access

    # Zed ACP adapter version (bridges ACP <-> the Claude Agent SDK).
    CLAUDE_CODE_ACP_VERSION = "0.16.2"
    _REMOTE_SETTINGS_DIR = PurePosixPath("/tmp/claude-code-settings")
    _REMOTE_SETTINGS_PATH = _REMOTE_SETTINGS_DIR / "settings.json"
    _INSTALL_CHECK_COMMAND = (
        'export PATH="$HOME/.local/bin:$PATH"; command -v claude >/dev/null 2>&1'
    )
    _INSTALL_VERSION_COMMAND = 'export PATH="$HOME/.local/bin:$PATH"; claude --version'

    SUPPORTS_PROBE_SUSPENSION: bool = True
    _OPENROUTER_ANTHROPIC_BASE_URL = "https://openrouter.ai/api"

    _KIMI_MODEL = "openrouter/moonshotai/kimi-k3.1-pro"
    _KIMI_REQUEST_MODEL = MODEL
    _GLM_MODEL = "openrouter/z-ai/glm-5.3-flash"
    _GLM_REQUEST_MODEL = GLM_MODEL
    _SONNET_MODEL = "openrouter/anthropic/claude-sonnet-5.5"
    _SONNET_REQUEST_MODEL = SONNET_MODEL
    _NATIVE_PRIVATE_ROOT = "/root/.picbench-claude-native"
    _NATIVE_CLEANUP_TIMEOUT_SECONDS = 120
    # Preserve the ordinary default; allow an explicit foreground wait covering
    # MZI-v2's 11460s client, 120s artifact retrieval and 120s local margin.
    # Harbor's independent 7200s learner deadline remains authoritative.
    _KIMI_BASH_TIMEOUT_MS = 3720000
    _NATIVE_BASH_MAX_TIMEOUT_MS = 11700000
    _MODEL_ROLE_KEYS = (
        "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_FABLE_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
    )

    options_model = ClaudeCodeOptions
    options: ClaudeCodeOptions

    @staticmethod
    @override
    def name() -> str:
        return AgentName.CLAUDE_CODE.value

    def __init__(
        self,
        logs_dir: Path,
        memory_dir: str | None = None,
        *args,
        **kwargs,
    ):
        self.memory_dir = memory_dir
        self._acp_claude_executable: str | None = None
        super().__init__(logs_dir, *args, memory_dir=memory_dir, **kwargs)
        self._base_settings = self._load_base_settings()
        self._model_http_retries = self.options.model_http_retries
        if self._model_http_retries and (
            not self.model_name or self.model_name.split("@", 1)[0] != self._GLM_MODEL
        ):
            raise ValueError("model_http_retries=1 requires guarded GLM 5.3 Flash")

    @override
    def _validate_native_load_trajectory(self, path: Path) -> None:
        try:
            is_session_file = path.suffix == ".jsonl" and bool(uuid.UUID(path.stem))
        except ValueError:
            is_session_file = False
        if not is_session_file:
            raise ValueError(
                "Claude Code load_trajectory must be a native session file "
                "named <session-id>.jsonl or an ATIF trajectory .json file; "
                f"got {path.name!r}"
            )

    @override
    async def _upload_load_trajectory(
        self, environment: BaseEnvironment, source: Path
    ) -> None:
        target = (
            EnvironmentPaths.agent_dir / "sessions" / "projects" / "-app" / source.name
        ).as_posix()
        await self._upload_agent_owned_file(environment, source, target)

    @override
    @classmethod
    def handoff(cls, trial_dir: Path, cwd: Path) -> list[str]:
        """Copy the trial's session into the local Claude Code project for ``cwd``."""
        if shutil.which("claude") is None:
            raise ValueError(
                "claude CLI not found on PATH; install Claude Code first: "
                "npm install -g @anthropic-ai/claude-code"
            )
        sessions = [
            session
            for session_dir in cls._session_dirs(trial_dir / "agent")
            for session in sorted(session_dir.glob("*.jsonl"))
        ]
        if len(sessions) != 1:
            raise ValueError(
                f"Expected exactly 1 session in {trial_dir / 'agent' / 'sessions'}, "
                f"found {len(sessions)}; handoff supports single-session trials only"
            )
        session = sessions[0]
        config_dir = Path(
            os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
        ).expanduser()
        project_dir = config_dir / "projects" / re.sub(r"[^A-Za-z0-9-]", "-", str(cwd))
        project_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(session, project_dir / session.name)
        return ["claude", "--resume", session.stem]

    @staticmethod
    def _session_text(message: str | list[ContentPart] | None) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        parts = [
            part.text if part.type == "text" and part.text else "[image omitted]"
            for part in message
        ]
        return "\n".join(parts)

    @classmethod
    def _session_tool_result_content(cls, result: ObservationResult) -> Any:
        """Prefer the exact tool_result content recorded by a Claude Code run."""
        raw = ((result.extra or {}).get("tool_result_metadata") or {}).get(
            "raw_tool_result"
        )
        if isinstance(raw, dict) and "content" in raw:
            return raw["content"]
        return cls._session_text(result.content)

    @override
    def atif_to_native_trajectory(
        self, trajectory: Trajectory, session_id: str
    ) -> tuple[str, str]:
        """Render an ATIF trajectory as a native Claude Code session transcript.

        The inverse of the session-to-ATIF conversion in this class: ATIF steps
        become the ``user``/``assistant`` event chain that Claude Code stores at
        ``$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl``, threaded
        through ``uuid``/``parentUuid`` so ``claude --resume <session-id>``
        picks the conversation up as history.
        """
        version = trajectory.agent.version if trajectory.agent else None
        default_model = trajectory.agent.model_name if trajectory.agent else None

        events: list[dict[str, Any]] = []
        parent_uuid: str | None = None
        last_timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        def base_event(event_type: str, timestamp: str) -> dict[str, Any]:
            nonlocal parent_uuid
            event: dict[str, Any] = {
                "parentUuid": parent_uuid,
                "isSidechain": False,
                "type": event_type,
                "uuid": str(uuid.uuid4()),
                "timestamp": timestamp,
                "sessionId": session_id,
                "userType": "external",
                "cwd": "/app",
                "gitBranch": "",
            }
            if version:
                event["version"] = version
            parent_uuid = event["uuid"]
            return event

        def append_step(step: Step) -> None:
            nonlocal last_timestamp
            timestamp = step.timestamp or last_timestamp
            last_timestamp = timestamp

            if step.source == "user":
                event = base_event("user", timestamp)
                event["message"] = {
                    "role": "user",
                    "content": self._session_text(step.message),
                }
                events.append(event)
                return

            content: list[dict[str, Any]] = []
            text = self._session_text(step.message)
            if text:
                content.append({"type": "text", "text": text})
            tool_calls = step.tool_calls or []
            for tool_call in tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.tool_call_id,
                        "name": tool_call.function_name,
                        "input": tool_call.arguments,
                    }
                )
            if content:
                metrics = step.metrics
                event = base_event("assistant", timestamp)
                event["message"] = {
                    "model": step.model_name or default_model,
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": (metrics.prompt_tokens if metrics else None)
                        or 0,
                        "output_tokens": (
                            metrics.completion_tokens if metrics else None
                        )
                        or 0,
                    },
                }
                events.append(event)

            results = step.observation.results if step.observation else []
            results_by_call_id = {
                result.source_call_id: result
                for result in results
                if result.source_call_id is not None
            }
            result_blocks = []
            for tool_call in tool_calls:
                result = results_by_call_id.get(tool_call.tool_call_id)
                result_blocks.append(
                    {
                        "tool_use_id": tool_call.tool_call_id,
                        "type": "tool_result",
                        "content": (
                            self._session_tool_result_content(result)
                            if result is not None
                            else "[no output recorded]"
                        ),
                    }
                )
            if result_blocks:
                event = base_event("user", timestamp)
                event["message"] = {"role": "user", "content": result_blocks}
                events.append(event)

        for step in trajectory.steps:
            # Claude Code sessions carry no system-prompt events; system steps
            # from other agents' trajectories are their agent-specific
            # scaffolding.
            if step.source == "system":
                continue
            append_step(step)

        content = "".join(json.dumps(event) + "\n" for event in events)
        return f"{session_id}.jsonl", content

    def _load_base_settings(self) -> dict[str, Any]:
        """Load and validate the user-supplied Claude settings JSON."""
        config_source = self.config_source
        if config_source is None:
            return {}
        if isinstance(config_source, dict):
            return config_source

        try:
            settings = json.loads(config_source.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid Claude Code settings file {config_source}: {exc}"
            ) from exc

        if not isinstance(settings, dict):
            raise ValueError(
                f"Invalid Claude Code settings file {config_source}: "
                "expected a JSON object"
            )
        return settings

    async def _upload_base_settings(self, environment: BaseEnvironment) -> None:
        """Upload validated settings for Claude's per-run ``--settings`` layer."""
        if self.config_source is None:
            return

        await self._upload_config_text(
            environment,
            content=json.dumps(self._base_settings, indent=2),
            remote_path=self._REMOTE_SETTINGS_PATH.as_posix(),
            filename="settings.json",
        )

    @override
    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; claude --version'

    @override
    def parse_version(self, stdout: str) -> str:
        # Output formats seen: "1.0.18 (Claude Code)" or "claude v1.2.3"
        import re

        text = stdout.strip()
        match = re.search(r"(\d+\.\d+\.\d+)", text)
        if match:
            return match.group(1)
        return text

    @override
    def _classify_exec_error(
        self, command: str, result: Any
    ) -> NonZeroAgentExitCodeError:
        """Classify the stream-json launch from the terminal ``result`` and
        ``error`` events only; assistant, user and system events carry model
        text and tool output that can echo error phrases."""
        if "--output-format=stream-json" not in command:
            return super()._classify_exec_error(command, result)

        def is_harness_error(event: dict[str, Any]) -> bool:
            if event.get("type") == "error":
                return True
            return event.get("type") == "result" and (
                event.get("is_error") is True
                or str(event.get("subtype", "")).startswith("error")
            )

        evidence = SimpleNamespace(
            return_code=result.return_code,
            stdout=filter_jsonl_events(result.stdout, is_harness_error),
            stderr=result.stderr,
        )
        return super()._classify_exec_error(command, evidence)

    async def _installed_claude_satisfies_version(
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
        if await self._installed_claude_satisfies_version(environment):
            self.logger.debug(
                "Claude Code is already available at the requested version"
            )
            return

        if self._get_env("PICBENCH_RF_MODEL_BROKER_URL"):
            raise RuntimeError(
                "The isolated RF image must preinstall the pinned Claude Code version"
            )
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "nodejs", "npm", "procps")
        )
        # Install claude-code (as default user)
        version_flag = f" {self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if command -v apk &> /dev/null; then"
                f"  npm install -g @anthropic-ai/claude-code{'@' + self._version if self._version else ''};"
                " else"
                f"  curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s --{version_flag};"
                " fi && "
                "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                "claude --version"
            ),
        )

    @classmethod
    def _session_dirs(cls, agent_dir: Path) -> list[Path]:
        """Directories under an agent logs dir holding Claude session files.

        Sessions live at ``<agent_dir>/sessions/projects/<cwd-slug>/*.jsonl``;
        subagent transcripts are excluded. Shared by the post-run
        native-to-ATIF conversion (rooted at ``self.logs_dir``) and by
        ``handoff`` (rooted at ``trial_dir / "agent"``).
        """
        project_root = agent_dir / "sessions" / "projects"
        if not project_root.is_dir():
            return []
        return sorted(
            {
                f.parent
                for f in project_root.rglob("*.jsonl")
                if "subagents" not in f.parent.parts
            }
        )

    @override
    def acp_command(self) -> list[str]:
        if self._is_guarded_openrouter_model():
            raise ValueError("Guarded native models require the observed native CLI route, not ACP")
        command = ["env", "-u", "CLAUDECODE"]
        model = self._resolved_model_name()
        if model:
            command.append(f"ANTHROPIC_MODEL={model}")
            alias_var = self._model_alias_env_var(model)
            if alias_var:
                command.append(f"{alias_var}={model}")
        command.append(
            f"CLAUDE_CONFIG_DIR={(self.environment_logs_dir / 'sessions').as_posix()}"
        )
        # acp adapter's Agent SDK has older bundled Claude Code build that
        # needs to be overridden, otherwise it silently drops the
        # intended effort level.
        if self._acp_claude_executable:
            command.append(f"CLAUDE_CODE_EXECUTABLE={self._acp_claude_executable}")
        effort = self.options.reasoning_effort
        if effort is not None:
            command.append(f"CLAUDE_CODE_EFFORT_LEVEL={effort}")
        max_thinking = self._resolved_env_vars.get("MAX_THINKING_TOKENS")
        if max_thinking is not None:
            command.append(f"MAX_THINKING_TOKENS={max_thinking}")
        command.append("claude-code-acp")
        return command

    @override
    async def acp_install(self, environment: BaseEnvironment) -> None:
        if self._is_guarded_openrouter_model():
            raise ValueError("Guarded native models require the observed native CLI route, not ACP")
        sessions_dir = (self.environment_logs_dir / "sessions").as_posix()
        # The adapter force-selects a model on every new session (ignoring
        # ANTHROPIC_MODEL); settings.json is its only pre-session selection
        # channel.
        model = self._resolved_model_name()
        settings_command = ""
        if model:
            settings_json = shlex.quote(json.dumps({"model": model}))
            settings_command = (
                f"printf '%s' {settings_json} > {sessions_dir}/settings.json; "
            )
        # The bootstrap.sh install lands in ~/.local/bin, which non-login
        # exec shells don't have on PATH, hence the fallback probe.
        result = await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"mkdir -p {sessions_dir}; "
                + settings_command
                + '_hb_system_node="$(command -v node 2>/dev/null || true)"; '
                + '_hb_claude="$(command -v claude 2>/dev/null || true)"; '
                + 'if [ -z "$_hb_claude" ] && [ -x "$HOME/.local/bin/claude" ]; then '
                + '_hb_claude="$HOME/.local/bin/claude"; fi; '
                + ensure_acp_node_command()
                + " && "
                "npm install -g "
                f"@zed-industries/claude-code-acp@{self.CLAUDE_CODE_ACP_VERSION} && "
                'echo "${_hb_system_node:-none}:$(command -v node)'
                ":$(command -v claude-code-acp)"
                ':${_hb_claude:-none}"'
            ),
        )
        binary_paths = (result.stdout or "").strip().splitlines()[-1]
        system_node, node_path, adapter_path, claude_path = (
            binary_paths.split(":")
            if binary_paths.count(":") == 3
            else ("", "", "", "")
        )
        if not node_path or not adapter_path:
            raise RuntimeError(
                "Could not locate node/claude-code-acp binaries for ACP mode "
                f"(got {result.stdout!r})"
            )
        if not claude_path or claude_path == "none":
            raise RuntimeError(
                "Could not locate the native claude binary for ACP mode; "
                "without it the adapter would fall back to the Agent SDK's "
                f"bundled Claude Code build (got {result.stdout!r})"
            )
        self._acp_claude_executable = claude_path
        link_command = pinned_bin_wrapper_command(
            node_path, adapter_path, "/usr/local/bin/claude-code-acp"
        )
        if system_node == "none":
            link_command += " && " + safe_bin_symlink_command(
                node_path, "/usr/local/bin/node"
            )
        await self.exec_as_root(environment, command=link_command)

    @property
    @override
    def remote_session_logs_dir(self) -> PurePosixPath:
        return self.environment_logs_dir / "sessions"

    @override
    def convert_trajectory(self, logs_dir: Path) -> Trajectory | None:
        session_dir = self._get_session_dir(logs_dir)
        return (
            self._convert_events_to_trajectory(session_dir, logs_dir=logs_dir)
            if session_dir
            else None
        )

    def _get_session_dir(self, logs_dir: Path | None = None) -> Path | None:
        """Identify the Claude session directory containing the primary JSONL log"""
        session_dirs = self._session_dirs(
            logs_dir if logs_dir is not None else self.logs_dir
        )
        if not session_dirs:
            return None
        if len(session_dirs) == 1:
            return session_dirs[0]

        self.logger.debug(
            "Multiple Claude Code session directories found; "
            "could not identify the correct one"
        )
        return None

    def _convert_event_to_step(self, event: dict[str, Any], step_id: int) -> Step:
        """Convert a normalized Claude Code event dictionary into an ATIF step."""
        kind = event.get("kind")
        timestamp = event.get("timestamp")

        if kind == "message":
            role = event.get("role", "user")
            text = event.get("text", "")
            reasoning = event.get("reasoning")
            metrics = event.get("metrics")
            extra = event.get("extra")
            model_name = event.get("model_name")

            if role == "assistant":
                source = "agent"
            elif role == "user":
                source = "user"
            else:
                source = "system"

            step = Step(
                step_id=step_id,
                timestamp=timestamp,
                source=source,
                message=text,
                llm_call_count=1 if source == "agent" else None,
            )

            if source == "agent":
                if reasoning:
                    step.reasoning_content = reasoning
                if model_name:
                    step.model_name = model_name
                elif self.model_name:
                    step.model_name = self.model_name

            if metrics:
                step.metrics = metrics
            if extra:
                step.extra = extra

            return step

        if kind == "agent_step":
            text = event.get("text") or ""
            reasoning = event.get("reasoning")
            metrics = event.get("metrics")
            extra = event.get("extra")
            model_name = event.get("model_name") or self.model_name
            tool_specs = event.get("tool_calls") or []

            tool_calls: list[ToolCall] = []
            results: list[ObservationResult] = []
            for spec in tool_specs:
                spec_call_id = spec.get("call_id")
                if not spec_call_id:
                    continue
                tool_calls.append(
                    ToolCall(
                        tool_call_id=spec_call_id,
                        function_name=spec.get("tool_name") or "",
                        arguments=spec.get("arguments") or {},
                        extra=spec.get("extra"),
                    )
                )
                if spec.get("output") is not None:
                    results.append(
                        ObservationResult(
                            source_call_id=spec_call_id,
                            content=spec.get("output"),
                            subagent_trajectory_ref=None,
                            extra=spec.get("result_extra"),
                        )
                    )

            step = Step(
                step_id=step_id,
                timestamp=timestamp,
                source="agent",
                message=text,
                tool_calls=tool_calls or None,
                observation=Observation(results=results) if results else None,
                llm_call_count=1,
            )
            if reasoning:
                step.reasoning_content = reasoning
            if model_name:
                step.model_name = model_name
            if metrics:
                step.metrics = metrics
            if extra:
                step.extra = extra

            return step

        if kind == "tool_call":
            call_id = event.get("call_id")
            tool_name = event.get("tool_name")
            if not call_id or not tool_name:
                raise ValueError("Tool call event missing call_id or tool_name")

            arguments = event.get("arguments") or {}
            raw_arguments = event.get("raw_arguments")
            reasoning = event.get("reasoning")
            metrics = event.get("metrics")
            extra = event.get("extra")
            status = event.get("status")
            message = event.get("message")
            output = event.get("output")
            metadata = event.get("metadata")
            model_name = event.get("model_name") or self.model_name

            tool_call = ToolCall(
                tool_call_id=call_id,
                function_name=tool_name,
                arguments=arguments,
            )

            observation_result = ObservationResult(
                source_call_id=call_id,
                content=output,
                subagent_trajectory_ref=None,
            )

            observation = (
                Observation(results=[observation_result])
                if output is not None
                else None
            )

            extra = extra or {}
            updates = {
                "metadata": metadata,
                "raw_arguments": raw_arguments,
                "status": status,
            }
            for key, value in updates.items():
                if value is not None:
                    extra.setdefault(key, value)

            step = Step(
                step_id=step_id,
                timestamp=timestamp,
                source="agent",
                message=message or "",
                tool_calls=[tool_call],
                observation=observation,
                llm_call_count=1,
            )

            if model_name:
                step.model_name = model_name
            if reasoning:
                step.reasoning_content = reasoning
            if metrics:
                step.metrics = metrics
            if extra:
                step.extra = extra

            return step

        raise ValueError(f"Unsupported event kind '{kind}'")

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @classmethod
    def _extract_text_reasoning_tool_uses(
        cls, content: Any
    ) -> tuple[str, str | None, list[dict[str, Any]]]:
        if isinstance(content, str):
            text = content.strip()
            return text, None, []

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_blocks: list[dict[str, Any]] = []

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(cls._stringify(block))
                    continue

                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_blocks.append(block)
                    continue

                if block_type in {"thinking", "reasoning", "analysis"}:
                    text_value = (
                        block.get("text")
                        if block.get("text") is not None
                        else block.get("thinking")
                    )
                    if isinstance(text_value, str):
                        reasoning_parts.append(text_value.strip())
                    else:
                        reasoning_parts.append(cls._stringify(text_value))
                    continue

                if block_type == "redacted_thinking":
                    # Anthropic `redacted_thinking` blocks carry encrypted
                    # reasoning in `data` that clients cannot decrypt — they
                    # are meant to be passed back unchanged on subsequent API
                    # calls. OpenRouter, however, mis-uses this envelope to
                    # pass through PLAIN reasoning from non-Anthropic models
                    # routed through their proxy: the `data` field is
                    # `openrouter.reasoning:<b64>`, where the base64 decodes
                    # to plain JSON `{"text": "…", "type": "reasoning.text"}`.
                    # Detect that shape and surface the inner text as
                    # reasoning. For genuine Anthropic ciphertext we drop the
                    # block rather than dump the raw envelope into the
                    # human-readable message field.
                    data = block.get("data")
                    if isinstance(data, str) and data.startswith(
                        "openrouter.reasoning:"
                    ):
                        try:
                            payload = data[len("openrouter.reasoning:") :]
                            decoded = base64.b64decode(payload + "==").decode(
                                "utf-8", "replace"
                            )
                            inner = json.loads(decoded)
                            inner_text = inner.get("text")
                            if isinstance(inner_text, str):
                                reasoning_parts.append(inner_text.strip())
                        except (ValueError, json.JSONDecodeError):
                            pass
                    continue

                if block_type == "code" and isinstance(block.get("code"), str):
                    text_parts.append(block["code"])
                    continue

                text_value = block.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                else:
                    text_parts.append(cls._stringify(block))
        elif content is not None:
            text_parts.append(cls._stringify(content))

        text = "\n\n".join(
            part.strip() for part in text_parts if part and str(part).strip()
        )
        reasoning = "\n\n".join(
            part.strip() for part in reasoning_parts if part and str(part).strip()
        )

        return text, (reasoning or None), tool_blocks

    @staticmethod
    def _build_metrics(usage: Any) -> Metrics | None:
        if not isinstance(usage, dict):
            return None

        # Use ``or 0`` rather than a ``get`` default: a streaming response that is
        # interrupted (e.g. the agent is killed on timeout) can leave a usage field
        # present but ``None`` (``"output_tokens": null``), and ``get(key, 0)`` only
        # falls back when the key is absent. Without this guard the arithmetic below
        # raises ``TypeError: int + NoneType``, which aborts the whole trajectory
        # conversion and discards token accounting for the entire trial.
        cached_tokens = usage.get("cache_read_input_tokens") or 0
        creation = usage.get("cache_creation_input_tokens") or 0
        input_tokens = usage.get("input_tokens") or 0
        # Align with Anthropic session totals: input + cache read + cache creation.
        prompt_tokens = input_tokens + cached_tokens + creation
        completion_tokens = usage.get("output_tokens") or 0

        extra: dict[str, Any] = {}
        for key, value in usage.items():
            if key in {"input_tokens", "output_tokens"}:
                continue
            extra[key] = value

        if (
            prompt_tokens is None
            and completion_tokens is None
            and cached_tokens is None
            and not extra
        ):
            return None

        return Metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=None,
            extra=extra or None,
        )

    @classmethod
    def _format_tool_result(
        cls, block: dict[str, Any], tool_use_result: dict[str, Any] | None
    ) -> tuple[str | None, dict[str, Any] | None]:
        parts: list[str] = []

        content = block.get("content")
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                text_value = cls._stringify(item)
                if text_value.strip():
                    parts.append(text_value.strip())
        elif content not in (None, ""):
            parts.append(cls._stringify(content))

        metadata: dict[str, Any] | None = None
        if tool_use_result and isinstance(tool_use_result, dict):
            metadata = {"tool_use_result": tool_use_result}
            stdout = tool_use_result.get("stdout")
            stderr = tool_use_result.get("stderr")
            exit_code = tool_use_result.get("exitCode") or tool_use_result.get(
                "exit_code"
            )
            interrupted = tool_use_result.get("interrupted")
            is_image = tool_use_result.get("isImage")

            formatted_chunks: list[str] = []
            if stdout:
                formatted_chunks.append(f"[stdout]\n{stdout}".rstrip())
            if stderr:
                formatted_chunks.append(f"[stderr]\n{stderr}".rstrip())
            if exit_code not in (None, 0):
                formatted_chunks.append(f"[exit_code] {exit_code}")
            if interrupted:
                formatted_chunks.append(f"[interrupted] {interrupted}")
            if is_image:
                formatted_chunks.append(f"[is_image] {is_image}")

            remaining_meta = {
                key: value
                for key, value in tool_use_result.items()
                if key
                not in {
                    "stdout",
                    "stderr",
                    "exitCode",
                    "exit_code",
                    "interrupted",
                    "isImage",
                }
            }
            if remaining_meta:
                formatted_chunks.append(
                    f"[metadata] {json.dumps(remaining_meta, ensure_ascii=False)}"
                )

            if formatted_chunks:
                parts.append("\n".join(chunk for chunk in formatted_chunks if chunk))

        if block.get("is_error") is True:
            parts.append("[error] tool reported failure")
            metadata = metadata or {}
            metadata["is_error"] = True

        if metadata is not None:
            metadata.setdefault("raw_tool_result", block)

        result_text = "\n\n".join(part for part in parts if part).strip()
        return (result_text or None), metadata

    def _parse_total_cost_from_stream_json(
        self, logs_dir: Path | None = None
    ) -> float | None:
        """Extract authoritative `total_cost_usd` from Claude Code's stdout stream.

        Claude Code's `--output-format=stream-json --print` mode emits a final
        ``{"type":"result", ..., "total_cost_usd": <float>, ...}`` line to stdout,
        which Harbor tees to ``<logs_dir>/claude-code.txt``. Returns ``None`` if
        the file is missing, malformed, or the result event lacks the field.
        """
        stream_path = (
            logs_dir if logs_dir is not None else self.logs_dir
        ) / "claude-code.txt"
        try:
            content = stream_path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in content.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                cost = event.get("total_cost_usd")
                if cost is None:
                    return None
                try:
                    return float(cost)
                except (TypeError, ValueError):
                    return None
        return None

    def _estimate_step_costs(self, steps: list[Step]) -> float | None:
        """Price each step with LiteLLM, writing ``metrics.cost_usd`` on it.

        Returns the sum, or None when any step with token usage could not be
        priced, since a partial sum would understate the total.
        """
        try:
            import litellm
        except ImportError:
            self.logger.debug("LiteLLM is unavailable; cannot estimate Claude cost")
            return None

        total_cost = 0.0
        priced_any_step = False
        priced_all_steps = True
        for step in steps:
            metrics = step.metrics
            if metrics is None:
                continue

            prompt_tokens = metrics.prompt_tokens or 0
            completion_tokens = metrics.completion_tokens or 0
            if prompt_tokens <= 0 and completion_tokens <= 0:
                continue
            if not step.model_name:
                self.logger.debug("Cannot estimate Claude cost without a step model")
                priced_all_steps = False
                continue

            extra = metrics.extra or {}
            cache_creation_tokens = extra.get("cache_creation_input_tokens")
            if not isinstance(cache_creation_tokens, int):
                cache_creation_tokens = 0
            cache_read_tokens = extra.get("cache_read_input_tokens")
            if not isinstance(cache_read_tokens, int):
                cache_read_tokens = metrics.cached_tokens or 0
            service_tier = extra.get("service_tier")
            if not isinstance(service_tier, str):
                service_tier = None

            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=step.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_creation_input_tokens=cache_creation_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                    service_tier=service_tier,
                )
            except Exception as exc:
                self.logger.debug(
                    "Cannot estimate Claude cost for model '%s': %s",
                    step.model_name,
                    exc,
                )
                priced_all_steps = False
                continue

            metrics.cost_usd = prompt_cost + completion_cost
            metrics.extra = {**extra, "cost_source": "litellm_estimate"}
            total_cost += metrics.cost_usd
            priced_any_step = True

        return total_cost if priced_any_step and priced_all_steps else None

    @staticmethod
    def _first_event_model(
        events: list[dict[str, Any]], *, include_sidechain: bool
    ) -> str | None:
        for event in events:
            if not include_sidechain and event.get("isSidechain"):
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            model_name = message.get("model")
            if isinstance(model_name, str) and model_name:
                return model_name
        return None

    @staticmethod
    def _event_identity_extra(event: dict[str, Any]) -> dict[str, Any]:
        """Keep each Claude session event's agent identity on its ATIF step."""
        extra = {"is_sidechain": event.get("isSidechain", False)}
        if agent_id := event.get("agentId"):
            extra["agent_id"] = agent_id
        return extra

    def _convert_events_to_trajectory(
        self, session_dir: Path, *, logs_dir: Path | None = None
    ) -> Trajectory | None:
        """Convert Claude session into an ATIF trajectory."""
        # Newer Claude Code versions write each subagent's transcript to its
        # own JSONL under a `subagents/` subdirectory (e.g.
        # `<session-id>/subagents/agent-<id>.jsonl`) instead of inlining
        # sidechain events in the main session file; read both.
        session_files = sorted(session_dir.glob("*.jsonl")) + sorted(
            session_dir.rglob("subagents/*.jsonl")
        )

        if not session_files:
            self.logger.debug(f"No Claude Code session files found in {session_dir}")
            return None

        raw_events: list[dict[str, Any]] = []
        for session_file in session_files:
            with open(session_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        raw_events.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        self.logger.debug(
                            f"Skipping malformed JSONL line in {session_file}: {exc}"
                        )

        if not raw_events:
            return None

        seen_event_uuids: set[str] = set()
        deduped_raw_events: list[dict[str, Any]] = []
        for event in raw_events:
            uuid = event.get("uuid")
            if isinstance(uuid, str) and uuid:
                if uuid in seen_event_uuids:
                    self.logger.debug(
                        f"Skipping duplicate Claude Code session event {uuid}"
                    )
                    continue
                seen_event_uuids.add(uuid)
            deduped_raw_events.append(event)
        raw_events = deduped_raw_events

        # Keep events in chronological order across the main chain and any
        # subagent sidechains, so the first user step remains the instruction
        # (downstream byte-identity checks rely on this) and step timestamps
        # stay monotonic; sidechain steps carry `extra.is_sidechain` and
        # `extra.agent_id` so consumers can distinguish concurrent subagents.
        raw_events.sort(key=lambda e: e.get("timestamp", ""))
        events = raw_events

        session_id: str = session_dir.name
        for event in events:
            sid = event.get("sessionId")
            if isinstance(sid, str):
                session_id = sid
                break

        agent_version: str = "unknown"
        for event in events:
            ver = event.get("version")
            if isinstance(ver, str) and ver:
                agent_version = ver
                break

        cwds = {
            event.get("cwd")
            for event in events
            if isinstance(event.get("cwd"), str) and event.get("cwd")
        }
        git_branches = {
            event.get("gitBranch")
            for event in events
            if isinstance(event.get("gitBranch"), str) and event.get("gitBranch")
        }
        agent_ids = {
            event.get("agentId")
            for event in events
            if isinstance(event.get("agentId"), str) and event.get("agentId")
        }

        agent_extra: dict[str, Any] | None = {}
        if cwds:
            agent_extra["cwds"] = cwds
        if git_branches:
            agent_extra["git_branches"] = git_branches
        if agent_ids:
            agent_extra["agent_ids"] = agent_ids
        if not agent_extra:
            agent_extra = None

        # Prefer the main chain's model so a subagent running a different
        # model (e.g. a small/fast model) cannot become the trajectory's
        # `agent.model_name` or the fallback for model-less steps.
        default_model_name = (
            self._first_event_model(events, include_sidechain=False)
            or self._first_event_model(events, include_sidechain=True)
            or self.model_name
        )

        # Per message id, keep the last usage (streaming updates it on each chunk).
        last_usage_by_msg_id: dict[str, Any] = {}
        for ev in events:
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message")
            if not isinstance(msg, dict):
                continue
            mid = msg.get("id")
            usage = msg.get("usage")
            if mid and usage is not None:
                last_usage_by_msg_id[mid] = usage

        normalized_events: list[dict[str, Any]] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        completed_call_ids: set[str] = set()
        seen_message_ids: set[str] = set()
        # Maps an assistant `message.id` to the single agent_step it bundles, so
        # text / reasoning / every tool_use from one LLM inference land on one
        # ATIF step even if the session log splits them across events.
        turn_by_msgid: dict[str, dict[str, Any]] = {}

        for event in events:
            message = event.get("message")
            if not isinstance(message, dict):
                continue

            event_type = event.get("type")
            timestamp = event.get("timestamp")

            if event_type == "assistant":
                text, reasoning, tool_blocks = self._extract_text_reasoning_tool_uses(
                    message.get("content")
                )

                msg_id = message.get("id")
                if msg_id and msg_id in seen_message_ids:
                    metrics = None
                else:
                    # Claude-Code accumulates usage, so the "real" usage is the last
                    # message in the chain. We still report this usage on the first
                    # shown in the trajectory.
                    usage = (
                        last_usage_by_msg_id.get(msg_id, message.get("usage"))
                        if msg_id
                        else message.get("usage")
                    )
                    metrics = self._build_metrics(usage)
                    if msg_id:
                        seen_message_ids.add(msg_id)
                extra: dict[str, Any] = {}
                for key in ("stop_reason", "stop_sequence", "requestId"):
                    value = message.get(key)
                    if value is not None:
                        extra[key] = value
                if event.get("id"):
                    extra["id"] = event["id"]
                extra.update(self._event_identity_extra(event))
                if event.get("cwd"):
                    extra.setdefault("cwd", event["cwd"])
                if event.get("userType") and event.get("userType") != "external":
                    extra["user_type"] = event["userType"]

                model_name = message.get("model") or default_model_name

                # Bundle one LLM inference (text, reasoning, and all tool_use
                # calls share a `message.id`) into a single ATIF step, per
                # RFC-0001 (`step` == one turn; `tool_calls` is multi-valued).
                # Reuse the turn when the same `message.id` is split across
                # several session-log events.
                turn = turn_by_msgid.get(msg_id) if msg_id else None
                if turn is None:
                    turn = {
                        "kind": "agent_step",
                        "timestamp": timestamp,
                        "role": message.get("role", "assistant"),
                        "text": "",
                        "reasoning": None,
                        "metrics": None,
                        "extra": extra or None,
                        "model_name": model_name,
                        "tool_calls": [],
                    }
                    normalized_events.append(turn)
                    if msg_id:
                        turn_by_msgid[msg_id] = turn

                if text:
                    turn["text"] = (
                        f"{turn['text']}\n\n{text}".strip() if turn["text"] else text
                    )
                if reasoning and message.get("role") == "assistant":
                    turn["reasoning"] = (
                        f"{turn['reasoning']}\n\n{reasoning}"
                        if turn["reasoning"]
                        else reasoning
                    )
                if turn["metrics"] is None and metrics is not None:
                    turn["metrics"] = metrics
                    metrics = None

                turn_calls = turn["tool_calls"]
                if not isinstance(turn_calls, list):
                    turn_calls = []
                    turn["tool_calls"] = turn_calls
                for tool_block in tool_blocks:
                    call_id = tool_block.get("id") or tool_block.get("tool_use_id")
                    if not call_id:
                        continue
                    # A call is keyed by call_id; skip a tool_use already seen
                    # (e.g. an event replayed after compaction) so it is not
                    # bundled twice.
                    if call_id in pending_calls or call_id in completed_call_ids:
                        continue

                    raw_arguments = tool_block.get("input")
                    if isinstance(raw_arguments, dict):
                        arguments = raw_arguments
                    else:
                        arguments = {"input": raw_arguments}

                    call_extra: dict[str, Any] = {}
                    if raw_arguments is not None:
                        call_extra["raw_arguments"] = raw_arguments
                    if tool_block.get("status") is not None:
                        call_extra["status"] = tool_block.get("status")
                    if tool_block.get("is_error") is not None:
                        call_extra["tool_use_is_error"] = tool_block.get("is_error")
                    if tool_block.get("name"):
                        call_extra.setdefault("tool_use_name", tool_block.get("name"))

                    tool_call_spec: dict[str, Any] = {
                        "call_id": call_id,
                        "tool_name": tool_block.get("name") or "",
                        "arguments": arguments or {},
                        "extra": call_extra or None,
                        "output": None,
                        "result_extra": None,
                    }
                    turn_calls.append(tool_call_spec)
                    pending_calls[call_id] = tool_call_spec

                continue

            if event_type == "user":
                content = message.get("content")
                if isinstance(content, str):
                    # Preserve the raw bytes of the user message so that
                    # downstream byte-identity checks (e.g. sha256 of the
                    # canonical instruction.md vs the first user step) hold;
                    # `if text.strip()` still drops empty / whitespace-only
                    # entries to match the previous skip semantics. The
                    # `text` truthy check is redundant — an empty str
                    # also strips to an empty str, which is falsy — so a
                    # single `text.strip()` does both jobs.
                    text = content
                    if text.strip():
                        extra = self._event_identity_extra(event)
                        normalized_events.append(
                            {
                                "kind": "message",
                                "timestamp": timestamp,
                                "role": "user",
                                "text": text,
                                "extra": extra,
                            }
                        )
                    continue

                if isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        # Byte-faithful: when a block is a plain text
                        # content block (`{"type": "text", "text": "..."}`),
                        # use the inner string verbatim instead of letting
                        # `_stringify` json-encode the whole dict. This
                        # keeps trailing/internal whitespace intact for
                        # downstream sha256-of-user-message checks.
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            text_parts.append(block["text"])
                            continue

                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            call_id = block.get("tool_use_id")
                            formatted_output, metadata = self._format_tool_result(
                                block, event.get("toolUseResult")
                            )
                            call_info = (
                                pending_calls.pop(call_id, None) if call_id else None
                            )
                            if call_info is not None:
                                # Matched a pending tool call: attach the result
                                # in place on its bundled turn (already appended
                                # to normalized_events as one agent_step).
                                result_extra: dict[str, Any] = {}
                                if metadata:
                                    result_extra["tool_result_metadata"] = metadata
                                if block.get("is_error") is not None:
                                    result_extra["tool_result_is_error"] = block.get(
                                        "is_error"
                                    )
                                call_info["output"] = formatted_output
                                call_info["result_extra"] = result_extra or None
                                if call_id:
                                    completed_call_ids.add(call_id)
                                continue

                            # Orphan tool_result with no matching tool_use in
                            # this window (e.g. replayed after compaction): keep
                            # the legacy single-call handling so its output is
                            # not lost. A duplicate of an already-completed call
                            # is dropped; a result with no tool name is skipped.
                            if call_id and call_id in completed_call_ids:
                                self.logger.debug(
                                    f"Skipping duplicate Claude Code tool result {call_id}"
                                )
                                continue
                            tool_name = (
                                block.get("name") or block.get("tool_name") or ""
                            )
                            if not tool_name:
                                self.logger.debug(
                                    "Skipping orphan Claude Code tool result "
                                    f"{call_id or '<missing>'} without tool name"
                                )
                                continue
                            call_info = {
                                "kind": "tool_call",
                                "timestamp": timestamp,
                                "call_id": call_id or "",
                                "tool_name": tool_name,
                                "arguments": {},
                                "raw_arguments": None,
                                "reasoning": None,
                                "status": None,
                                "message": None,
                                "extra": None,
                                "metrics": None,
                                "model_name": default_model_name,
                            }

                            extra_val = call_info.get("extra")
                            extra = extra_val if isinstance(extra_val, dict) else {}
                            extra.update(self._event_identity_extra(event))
                            if metadata:
                                extra.setdefault("tool_result_metadata", metadata)
                            if block.get("is_error") is not None:
                                extra.setdefault(
                                    "tool_result_is_error", block.get("is_error")
                                )

                            call_info["extra"] = extra or None
                            call_info["output"] = formatted_output
                            call_info["metadata"] = metadata
                            call_info["timestamp"] = (
                                call_info.get("timestamp") or timestamp
                            )
                            call_info.setdefault("model_name", default_model_name)

                            normalized_events.append(call_info)
                            if call_id:
                                completed_call_ids.add(call_id)
                            continue

                        # Anthropic content blocks that aren't tool_result
                        # (notably {"type":"text","text":"…"} text blocks
                        # injected as user content, e.g. when Claude Code
                        # loads a Skill) should have their inner string
                        # surfaced, not the whole block dict JSON-encoded.
                        # Without this, the ATIF user message ends up as a
                        # raw envelope like
                        #   '{"type":"text","text":"Base directory for…"}'
                        # which downstream renderers can't read as message
                        # text.
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            text_parts.append(block["text"])
                            continue

                        text_parts.append(self._stringify(block))

                    # Drop the per-part `.strip()` so byte spans inside each
                    # part round-trip; still filter parts that are empty or
                    # whitespace-only so the join doesn't materialise
                    # `\n\n` separators between nothing. ``part`` is always
                    # the output of `block["text"]` or `_stringify(block)`,
                    # both of which return ``str``, so a plain
                    # ``part.strip()`` is the single sufficient filter.
                    text_message = "\n\n".join(
                        part for part in text_parts if part.strip()
                    )
                    if text_message:
                        normalized_events.append(
                            {
                                "kind": "message",
                                "timestamp": timestamp,
                                "role": "user",
                                "text": text_message,
                                "extra": self._event_identity_extra(event),
                            }
                        )
                    continue

                if content not in (None, ""):
                    # Same byte-faithful rule as above: keep the raw bytes
                    # of the stringified content and use `.strip()` only
                    # for the empty-skip filter, not as a transform.
                    text = self._stringify(content)
                    if text.strip():
                        normalized_events.append(
                            {
                                "kind": "message",
                                "timestamp": timestamp,
                                "role": "user",
                                "text": text,
                                "extra": self._event_identity_extra(event),
                            }
                        )

        # Leftover pending calls (a tool_use that never received a result) are
        # already embedded in their bundled turn's `tool_calls`, so there is
        # nothing to flush here; they render as a ToolCall with no observation.

        steps: list[Step] = []
        for norm_event in normalized_events:
            try:
                step = self._convert_event_to_step(norm_event, len(steps) + 1)
            except ValueError as exc:
                self.logger.debug(f"Skipping event during step conversion: {exc}")
                continue

            if step.source == "agent" and not step.model_name and default_model_name:
                step.model_name = default_model_name

            steps.append(step)

        if not steps:
            self.logger.debug("No valid steps produced from Claude Code session")
            return None

        prompt_values = [
            step.metrics.prompt_tokens
            for step in steps
            if step.metrics and step.metrics.prompt_tokens is not None
        ]
        completion_values = [
            step.metrics.completion_tokens
            for step in steps
            if step.metrics and step.metrics.completion_tokens is not None
        ]
        cached_values = [
            step.metrics.cached_tokens
            for step in steps
            if step.metrics and step.metrics.cached_tokens is not None
        ]

        total_prompt_tokens = sum(prompt_values) if prompt_values else None
        total_completion_tokens = sum(completion_values) if completion_values else None
        total_cached_tokens = sum(cached_values) if cached_values else None

        service_tiers: set[str] = set()
        cache_creation_total, cache_read_total = 0, 0
        cache_creation_seen, cache_read_seen = False, False
        for step in steps:
            if not step.metrics or not step.metrics.extra:
                continue
            extra = step.metrics.extra
            tier = extra.get("service_tier")
            if isinstance(tier, str):
                service_tiers.add(tier)
            cache_creation = extra.get("cache_creation_input_tokens")
            if isinstance(cache_creation, int):
                cache_creation_total += cache_creation
                cache_creation_seen = True
            cache_read = extra.get("cache_read_input_tokens")
            if isinstance(cache_read, int):
                cache_read_total += cache_read
                cache_read_seen = True

        final_extra: dict[str, Any] = {}
        if service_tiers:
            final_extra["service_tiers"] = sorted(service_tiers)
        if cache_creation_seen:
            final_extra["total_cache_creation_input_tokens"] = cache_creation_total
        if cache_read_seen:
            final_extra["total_cache_read_input_tokens"] = cache_read_total

        estimated_cost_usd = self._estimate_step_costs(steps)
        total_cost_usd = self._parse_total_cost_from_stream_json(logs_dir)
        if total_cost_usd is None:
            total_cost_usd = estimated_cost_usd
            if total_cost_usd is not None:
                final_extra["cost_source"] = "litellm_estimate"

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cached_tokens=total_cached_tokens,
            total_cost_usd=total_cost_usd,
            total_steps=len(steps),
            extra=final_extra or None,
        )

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name=AgentName.CLAUDE_CODE.value,
                version=agent_version,
                model_name=default_model_name,
                extra=agent_extra,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

        return trajectory

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        try:
            trajectory = self.convert_trajectory(self.logs_dir)
        except Exception as exc:
            self.logger.debug(
                f"Failed to convert Claude Code events to trajectory: {exc}"
            )
            return
        if not trajectory:
            self.logger.debug("Failed to convert Claude Code session to trajectory")
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(
                    trajectory.to_json_dict(), handle, indent=2, ensure_ascii=False
                )
            self.logger.debug(f"Wrote Claude Code trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        if trajectory.final_metrics:
            metrics = trajectory.final_metrics
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills from the environment to Claude's config.

        Copies the contents of ``self.skills_dir`` into
        ``$CLAUDE_CONFIG_DIR/skills/`` so that Claude Code picks them up
        automatically.
        """
        if not self.skills_dir:
            return None
        return (
            f"(cp -r {shlex.quote(self.skills_dir)}/* "
            "$CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true)"
        )

    def _build_register_memory_command(self) -> str | None:
        """Return a shell command that copies memory files into Claude's config.

        Copies the contents of ``self.options.memory_dir`` into
        ``$CLAUDE_CONFIG_DIR/projects/-app/memory/`` so that Claude Code
        picks them up as pre-seeded auto-memory at session start.
        """
        if not self.options.memory_dir:
            return None
        return (
            f"(mkdir -p $CLAUDE_CONFIG_DIR/projects/-app/memory && "
            f"cp -r {shlex.quote(self.options.memory_dir)}/* "
            "$CLAUDE_CONFIG_DIR/projects/-app/memory/ 2>/dev/null || true)"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Return a shell command that writes MCP config to ~/.claude.json.

        User-scoped servers are loaded without a trust dialog, unlike
        project-scoped .mcp.json which requires explicit enablement.
        """
        if not self.mcp_servers:
            return None
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "type": "stdio",
                    "command": server.command,
                    "args": server.args,
                }
            else:
                # Claude Code uses "http" for streamable-http transport
                transport = (
                    "http"
                    if server.transport == "streamable-http"
                    else server.transport
                )
                servers[server.name] = {"type": transport, "url": server.url}
        claude_json = json.dumps({"mcpServers": servers}, indent=2)
        escaped = shlex.quote(claude_json)
        return f"echo {escaped} > $CLAUDE_CONFIG_DIR/.claude.json"

    def _resolved_model_name(self) -> str | None:
        """Effective ``ANTHROPIC_MODEL`` value, shared by ``run()`` and the
        ACP launch command."""
        if self.model_name:
            access = self.model_connection
            if access.provider is None:
                if "/" in self.model_name:
                    return self.model_name.split("/", 1)[-1]
                return self.model_name
            if access.configured_base_url:
                return self.model_name
            return self.model_name.split("/")[-1]
        return self._get_env("ANTHROPIC_MODEL") or None

    @staticmethod
    def _model_alias_env_var(model: str) -> str | None:
        """Tier-alias env var (``ANTHROPIC_DEFAULT_*_MODEL``) to pin, since
        the ACP adapter may resolve the session model through a tier alias
        ("opus"/"sonnet"/"haiku") rather than the exact ID."""
        lowered = model.lower()
        for tier in ("opus", "sonnet", "haiku"):
            if tier in lowered:
                return f"ANTHROPIC_DEFAULT_{tier.upper()}_MODEL"
        return None

    def _should_force_oauth(self) -> bool:
        """Whether to drop the API key so the CLI uses CLAUDE_CODE_OAUTH_TOKEN.

        Opt in via CLAUDE_FORCE_OAUTH=<truthy>; default keeps ANTHROPIC_API_KEY.
        Mirrors codex's CODEX_FORCE_AUTH_JSON.
        """
        return parse_bool_env_value(
            self._get_env("CLAUDE_FORCE_OAUTH"),
            name="CLAUDE_FORCE_OAUTH",
            default=False,
        )

    def _is_kimi_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.split("@", 1)[0] == self._KIMI_MODEL)

    def _is_sonnet_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.split("@", 1)[0] == self._SONNET_MODEL)

    def _is_glm_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.split("@", 1)[0] == self._GLM_MODEL)

    def _native_model_identity_settings(self) -> dict[str, Any]:
        """Recognize the gateway's pinned Sonnet ID as native Sonnet 5.5.

        The documented override controls CLI capability detection while every
        actual request still carries the same exact OpenRouter preset.
        """
        if self._is_sonnet_openrouter_model():
            return {"modelOverrides": {"claude-sonnet-5-5": self._SONNET_REQUEST_MODEL}}
        return {}


    def _is_guarded_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.split("@", 1)[0] in {
            self._KIMI_MODEL, self._GLM_MODEL, self._SONNET_MODEL,
        })


    def _guarded_request_model(self) -> str:
        if self._is_sonnet_openrouter_model():
            return self._SONNET_REQUEST_MODEL
        if self.model_name and self.model_name.split("@", 1)[0] == self._GLM_MODEL:
            return self._GLM_REQUEST_MODEL
        return self._KIMI_REQUEST_MODEL


    def _native_stream_watchdog_environment(self) -> dict[str, str]:
        if not self._is_guarded_openrouter_model():
            return {}
        # Claude's header timer, fixed body timer and streaming watchdogs are
        # independent. Let the reviewed proxy detect inactivity, without the
        # CLI's default five-minute body timer preempting an active response.
        idle_ms = str(model_policy(self._guarded_request_model())["request_timeout_seconds"] * 1000)
        return {
            "API_FORCE_IDLE_TIMEOUT": "0",
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": idle_ms,
            "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": idle_ms,
            "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
            "CLAUDE_ENABLE_BYTE_WATCHDOG": "1",
        }


    def _native_cli_retry_environment(self) -> dict[str, str]:
        if not self._is_guarded_openrouter_model():
            return {}
        # Only the operator-selected wire guard may retry eligible HTTP errors.
        # Disable native CLI resends without changing streaming watchdogs.
        environment = {"CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_RETRY_WATCHDOG": "0"}
        if self._is_kimi_openrouter_model() or self._is_sonnet_openrouter_model():
            # The pinned CLI treats a stream failure as a separate nonstreaming
            # fallback even when SDK retries are disabled. Kimi/Sonnet permit neither.
            environment["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] = "1"
        return environment


    def _validate_guarded_configuration(self) -> None:
        """Reject ambiguous native routing before consulting any credential."""
        request_model = self._guarded_request_model()
        policy = model_policy(request_model)
        if self.model_name != "openrouter/" + request_model:
            raise ValueError("Native Claude routing requires the exact reviewed OpenRouter preset")
        if self._version != "2.1.310":
            raise ValueError("Native OpenRouter routing requires pinned Claude Code version2.1.310")
        if self._resolved_flags.get("reasoning_effort") not in policy["efforts"]:
            raise ValueError("Native OpenRouter routing requires an explicit supported model effort")
        if self._resolved_flags.get("thinking") != "adaptive":
            raise ValueError("Native OpenRouter routing requires adaptive thinking")
        if self.options.max_thinking_tokens is not None:
            raise ValueError("Native effort selection cannot be combined with a thinking-token override")
        if self._resolved_flags.get("fallback_model") is not None:
            raise ValueError("Guarded OpenRouter routing does not permit a fallback model")
        identity = self._native_model_identity_settings()
        if identity and self._base_settings.get("modelOverrides") not in (None, identity["modelOverrides"]):
            raise ValueError("Sonnet model identity must retain the exact native-to-preset mapping")
        if self.model_connection.provider is None:
            raise ValueError("Guarded OpenRouter routing cannot be combined with Bedrock")
        for key in ("CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_FORCE_OAUTH",
                    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"):
            if (self._get_env(key) or "").strip().lower() not in {"", "0", "false", "no", "off"}:
                raise ValueError("Guarded OpenRouter routing does not permit provider or thinking overrides")
        if self._extra_env.get("CLAUDE_CODE_EFFORT_LEVEL", self._resolved_flags["reasoning_effort"]) != self._resolved_flags["reasoning_effort"]:
            raise ValueError("Effort override conflicts with the selected model effort")
        for key in self._MODEL_ROLE_KEYS:
            if key in self._extra_env and self._extra_env[key] != request_model:
                raise ValueError("Model-role overrides must retain the exact pinned route")
        for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
            if key in self._extra_env:
                raise ValueError("Native observer routing does not permit credential or endpoint overrides")
        for key in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            if self._extra_env.get(key):
                raise ValueError("Native observer routing does not permit credential or endpoint overrides")
        for key, expected in (("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(policy["output_cap"])),
                              ("API_TIMEOUT_MS", str(policy["api_client_timeout_ms"]))):
            if self._get_env(key) not in (None, expected):
                raise ValueError("Native output cap and API timeout must retain the reviewed values")
        for key, expected in self._native_stream_watchdog_environment().items():
            if self._get_env(key) not in (None, expected):
                raise ValueError("Native streaming timeouts must retain the reviewed values")
        for key, expected in self._native_cli_retry_environment().items():
            if self._get_env(key) not in (None, expected):
                if key == "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK":
                    raise ValueError("Native nonstreaming fallback must remain disabled")
                raise ValueError("Native automatic CLI retries must remain disabled")
        for key, expected in (("BASH_DEFAULT_TIMEOUT_MS", self._KIMI_BASH_TIMEOUT_MS),
                              ("BASH_MAX_TIMEOUT_MS", self._NATIVE_BASH_MAX_TIMEOUT_MS)):
            if self._get_env(key) not in (None, str(expected)):
                raise ValueError("Native Bash timeouts must cover the reviewed PIC probe wait")
        if self._get_env("ENABLE_TOOL_SEARCH") not in (None, "false"):
            raise ValueError("Native tool search must be disabled; load tools upfront")
        budget = self._resolved_flags.get("max_budget_usd")
        if budget is not None:
            try:
                valid_budget = math.isfinite(float(budget)) and float(budget) > 0
            except (TypeError, ValueError):
                valid_budget = False
            if not valid_budget:
                raise ValueError("Native model-spend cap must be a finite positive amount")


    async def _start_native_observer(self, environment: BaseEnvironment, api_key: str) -> dict:
        """Use the same native Messages observer in qualification and scored runs."""
        helper = Path(__file__).with_name("claude_openrouter.py")
        if helper.is_symlink() or not helper.is_file():
            raise RuntimeError("The frozen Claude OpenRouter observer is missing")
        helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
        private_root = self._NATIVE_PRIVATE_ROOT
        prepare = (
            "import os,pathlib,stat\n"
            "assert os.geteuid()==0 and os.getegid()==0\n"
            "parent=pathlib.Path('/root')\n"
            "assert not parent.is_symlink() and parent.stat().st_uid==0 and stat.S_IMODE(parent.stat().st_mode)==0o700\n"
            f"path=pathlib.Path({private_root!r})\n"
            "path.mkdir(mode=0o700,exist_ok=False)\n"
        )
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(prepare), user="root")
        if result.return_code != 0:
            raise RuntimeError("Private native observer root could not be established")
        remote = private_root + "/claude-openrouter.py"
        await environment.upload_file(helper, remote)
        version_result = await environment.exec(command=self.get_version_command())
        version = self.parse_version(version_result.stdout)
        if version_result.return_code != 0 or version != self._version:
            raise RuntimeError("Actual Claude Code version differs from the pinned native version")
        budget = self._resolved_flags.get("max_budget_usd")
        effort = self._resolved_flags["reasoning_effort"]
        arguments = ["/usr/local/bin/python3", "-I", "-B", remote, "--state-dir", private_root + "/openrouter-wire",
                     "--model", self._guarded_request_model(), "--effort", effort, "--port", "0"]
        if self._is_glm_openrouter_model():
            arguments += ["--model-http-retries", str(self._model_http_retries)]
        if budget is not None:
            arguments += ["--max-cost-usd", str(budget)]
        launcher = (
            "import hashlib,os,pathlib,subprocess,time\n"
            "assert os.geteuid()==0 and os.getegid()==0\n"
            f"helper=pathlib.Path({remote!r})\n"
            "assert helper.is_file() and not helper.is_symlink()\n"
            "os.chown(helper,0,0); helper.chmod(0o600)\n"
            f"assert hashlib.sha256(helper.read_bytes()).hexdigest()=={helper_sha!r}\n"
            f"ready=pathlib.Path({private_root + '/openrouter-wire/ready.json'!r})\n"
            "if ready.exists(): raise RuntimeError('Observer evidence already exists')\n"
            "os.umask(0o077)\n"
            f"log=open({private_root + '/openrouter-observer.log'!r},'xb')\n"
            f"child=subprocess.Popen({arguments!r},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
            "for _ in range(150):\n"
            " if ready.is_file(): break\n"
            " if child.poll() is not None: raise RuntimeError('Observer exited before readiness')\n"
            " time.sleep(0.1)\n"
            "else:\n"
            " child.terminate()\n"
            " raise RuntimeError('Observer readiness timed out')\n"
            "print(ready.read_text())\n"
        )
        # Bypass BaseInstalledAgent's debug environment logging for this one
        # credential-bearing helper launch. Neither argv nor receipts carry it.
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(launcher),
                                        env={"OPENROUTER_API_KEY": api_key}, timeout_sec=20, user="root")
        try:
            ready = json.loads(result.stdout)
            endpoint = urlsplit(ready["base_url"])
            valid = (
                result.return_code == 0
                and set(ready) == {"base_url", "pid", "model", "reasoning_effort", "max_cost_usd",
                                   "automatic_retries", "request_timeout_seconds", "request_timeout_semantics",
                                   "connect_timeout_seconds", "api_client_timeout_ms",
                                   "helper_sha256", "execution_uid", "execution_gid",
                                   "native_request_policy_version", "native_missing_thinking_policy"}
                                   | ({"model_http_retries"} if self._is_glm_openrouter_model() else set())
                and endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1"
                and endpoint.port is not None and 0 < endpoint.port < 65536
                and not endpoint.username and not endpoint.password and endpoint.path == ""
                and not endpoint.query and not endpoint.fragment
                and type(ready["pid"]) is int and ready["pid"] > 0
                and ready["model"] == self._guarded_request_model() and ready["reasoning_effort"] == effort
                and ready["helper_sha256"] == helper_sha
                and type(ready["automatic_retries"]) is int
                and ready["automatic_retries"] == self._model_http_retries
                and (not self._is_glm_openrouter_model() or (
                    type(ready.get("model_http_retries")) is int
                    and ready["model_http_retries"] == self._model_http_retries
                ))
                and type(ready["request_timeout_seconds"]) is int
                and ready["request_timeout_seconds"] == model_policy(self._guarded_request_model())["request_timeout_seconds"]
                and ready["request_timeout_semantics"] == model_policy(self._guarded_request_model())["request_timeout_semantics"]
                and type(ready["connect_timeout_seconds"]) is int
                and ready["connect_timeout_seconds"] == model_policy(self._guarded_request_model())["connect_timeout_seconds"]
                and type(ready["api_client_timeout_ms"]) is int
                and ready["api_client_timeout_ms"] == model_policy(self._guarded_request_model())["api_client_timeout_ms"]
                and ready["max_cost_usd"] == budget
                and type(ready["execution_uid"]) is int and ready["execution_uid"] == 0
                and type(ready["execution_gid"]) is int and ready["execution_gid"] == 0
                and type(ready["native_request_policy_version"]) is int
                and ready["native_request_policy_version"] == (4 if self._is_sonnet_openrouter_model() else 3)
                and ready["native_missing_thinking_policy"] == (
                    "webfetch_websearch_adaptive_v1"
                )
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError("Claude OpenRouter observer readiness does not match the frozen configuration")
        ready["version"] = version
        return ready


    async def _record_native_configuration(self, environment: BaseEnvironment, ready: dict, env: dict) -> None:
        policy = model_policy(self._guarded_request_model())
        receipt = {
            "schema_version": 1, "agent": "claude-code", "model": self._guarded_request_model(),
            "reasoning_effort": self._resolved_flags["reasoning_effort"], "thinking": "adaptive",
            "version": ready["version"], "role_models": {key: env[key] for key in self._MODEL_ROLE_KEYS},
            "helper_sha256": ready["helper_sha256"], "max_output_tokens_per_call": int(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]),
            "request_timeout_seconds": policy["request_timeout_seconds"],
            "request_timeout_semantics": policy["request_timeout_semantics"],
            "connect_timeout_seconds": policy["connect_timeout_seconds"],
            "api_client_timeout_ms": int(env["API_TIMEOUT_MS"]),
            "automatic_retries": self._model_http_retries, "retry_authority": "wire_guard",
            "bash_default_timeout_ms": int(env["BASH_DEFAULT_TIMEOUT_MS"]),
            "bash_max_timeout_ms": int(env["BASH_MAX_TIMEOUT_MS"]),
            "max_budget_usd": float(ready["max_cost_usd"]) if ready["max_cost_usd"] is not None else None,
            "endpoint_sha256": hashlib.sha256(ready["base_url"].encode()).hexdigest(),
            "native_effort_translation": "not_inferred_by_adapter",
            "native_evidence_scope": "operator_private",
            "observer_execution_uid": ready["execution_uid"],
            "observer_execution_gid": ready["execution_gid"],
            "native_request_policy_version": ready["native_request_policy_version"],
            "native_missing_thinking_policy": ready["native_missing_thinking_policy"],
        }
        receipt["tool_search_enabled"] = env["ENABLE_TOOL_SEARCH"] != "false"
        receipt["native_cli_retry_environment"] = {
            key: env[key] for key in self._native_cli_retry_environment()
        }
        receipt["stream_watchdog_environment"] = {
            key: env[key] for key in self._native_stream_watchdog_environment()
        }
        identity = self._native_model_identity_settings()
        if identity:
            receipt["model_identity_overrides"] = identity["modelOverrides"]
            receipt["canonical_cli_model"] = "claude-sonnet-5-5"
        if self._is_glm_openrouter_model():
            receipt["model_http_retries"] = self._model_http_retries
        code = ("import os,pathlib; assert os.geteuid()==0 and os.getegid()==0; "
                f"p=pathlib.Path({self._NATIVE_PRIVATE_ROOT + '/claude-code-config.json'!r}); "
                f"p.write_text({json.dumps(json.dumps(receipt, sort_keys=True) + chr(10))}); p.chmod(0o600)")
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(code), user="root")
        if result.return_code != 0:
            raise RuntimeError("Claude native configuration receipt was not retained")


    async def _stop_native_observer(self, environment: BaseEnvironment, ready: dict) -> None:
        helper = (self._NATIVE_PRIVATE_ROOT + "/claude-openrouter.py").encode()
        code = (
            "import json,os,pathlib,signal,time\n"
            "assert os.geteuid()==0 and os.getegid()==0\n"
            f"pid={ready['pid']}\n"
            f"helper={helper!r}\n"
            "stopped=False\n"
            "try:\n"
            " command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()\n"
            " if helper in command: os.kill(pid,signal.SIGTERM)\n"
            "except ProcessLookupError: pass\n"
            "except FileNotFoundError: pass\n"
            "for _ in range(50):\n"
            " try: command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()\n"
            " except FileNotFoundError: command=b''\n"
            " if helper not in command:\n"
            "  stopped=True\n"
            "  break\n"
            " time.sleep(0.1)\n"
            "if not stopped:\n"
            " os.kill(pid,signal.SIGKILL)\n"
            " for _ in range(50):\n"
            "  try: command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()\n"
            "  except FileNotFoundError: command=b''\n"
            "  if helper not in command:\n"
            "   stopped=True; break\n"
            "  time.sleep(0.1)\n"
            "assert stopped, 'Native observer did not stop'\n"
            f"p=pathlib.Path({self._NATIVE_PRIVATE_ROOT + '/openrouter-wire/observer-stop.json'!r})\n"
            "p.write_text(json.dumps({'pid':pid,'stopped':stopped,'grace_seconds':5,'execution_uid':os.geteuid(),'execution_gid':os.getegid()})+'\\n')\n"
            "p.chmod(0o600)\n"
        )
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(code), timeout_sec=15, user="root")
        if result.return_code != 0:
            raise self._native_integrity_error("Native observer did not stop before evidence export")


    def _native_evidence_snapshot_code(self) -> str:
        helper_sha = hashlib.sha256(Path(__file__).with_name("claude_openrouter.py").read_bytes()).hexdigest()
        return (
            "import hashlib,json,os,pathlib,re,stat,sys\n"
            "sys.dont_write_bytecode=True\n"
            f"base=pathlib.Path({self._NATIVE_PRIVATE_ROOT!r})\n"
            "class NativeEvidenceViolation(Exception):\n"
            " def __init__(self,rule,path):\n"
            "  label='/root' if path==pathlib.Path('/root') else str(path.relative_to(base))\n"
            "  self.record={'rule':rule,'path':label[:240]}\n"
            "def require(condition,rule,path):\n"
            " if not condition: raise NativeEvidenceViolation(rule,path)\n"
            "def private(path,directory):\n"
            " require(path.exists() or path.is_symlink(),'missing',path)\n"
            " info=path.lstat()\n"
            " require(not stat.S_ISLNK(info.st_mode),'symlink',path)\n"
            " require(info.st_uid==info.st_gid==0,'root_owner',path)\n"
            " require(stat.S_IMODE(info.st_mode)==(0o700 if directory else 0o600),'private_mode',path)\n"
            " require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),'file_kind',path)\n"
            "def snapshot():\n"
            " require(os.geteuid()==os.getegid()==0,'root_actor',base)\n"
            " private(pathlib.Path('/root'),True); private(base,True)\n"
            " required={'claude-openrouter.py','claude-code-config.json','openrouter-observer.log','openrouter-wire/ready.json','openrouter-wire/observer-stop.json'}\n"
            " files={}\n"
            " for here,dirs,names in os.walk(base,followlinks=False):\n"
            "  path=pathlib.Path(here); private(path,True)\n"
            "  if path==base:\n"
            "   if 'export' in dirs:\n"
            "    require(not (base/'export').is_symlink(),'symlink',base/'export')\n"
            "   dirs[:]=[name for name in dirs if name!='export']\n"
            "  for name in dirs:\n"
            "   child=path/name; private(child,True)\n"
            "   require(child==base/'openrouter-wire','unexpected_directory',child)\n"
            "  for name in names:\n"
            "   if path==base and name=='privacy.json':\n"
            "    private(path/name,False); continue\n"
            "   file=path/name; private(file,False)\n"
            "   relative=str(file.relative_to(base))\n"
            "   known=(relative in required or path==base/'openrouter-wire' and (name in {'failure.json','failures.jsonl','rejected-request.json','first-rejected-request.json','rejection-limit.json'} or re.fullmatch(r'(?:[0-9]{4,}-(?:request\\.json|request\\.raw\\.body|request-policy\\.json|exchange\\.json|response\\.body)|rejection-[0-9]{4,}(?:-request\\.body|\\.json))',name)))\n"
            f"   known=known or ({self._model_http_retries == 1!r} and path==base/'openrouter-wire' and re.fullmatch(r'[0-9]{{4,}}-http-attempt-1-response\\.body',name))\n"
            "   require(bool(known),'unexpected_file',file)\n"
            "   files[relative]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'size_bytes':file.stat().st_size}\n"
            " for name in required: require(name in files,'required_evidence',base/name)\n"
            f" require(files['claude-openrouter.py']['sha256']=={helper_sha!r},'helper_source_hash',base/'claude-openrouter.py')\n"
            " configuration=json.loads((base/'claude-code-config.json').read_text())\n"
            " require(configuration.get('helper_sha256')==files['claude-openrouter.py']['sha256'],'configuration_helper_hash',base/'claude-code-config.json')\n"
            " return files\n"
        )


    def _native_evidence_checked_code(self, body: str) -> str:
        """Emit only a bounded protocol diagnostic when a root check fails."""
        return self._native_evidence_snapshot_code() + (
            "try:\n" + "".join(" " + line + "\n" for line in body.splitlines())
            + "except Exception as exc:\n"
            " record=exc.record if isinstance(exc,NativeEvidenceViolation) else {'rule':'snapshot_error','path':'.','error_type':type(exc).__name__}\n"
            " print(json.dumps({'native_evidence_failure':record},sort_keys=True))\n"
            " raise SystemExit(1)\n"
        )


    def _record_native_cleanup_failure(self, error: BaseException, phase: str, *, original: BaseException | None = None) -> None:
        """Retain failure classifications outside the learner and canonical tree."""
        diagnostic = {
            "schema_version": 1, "phase": phase,
            "error_type": type(error).__name__,
            "integrity_failure": isinstance(error, AgentEvidenceIntegrityError),
            "original_error_type": type(original).__name__ if original is not None else None,
            "diagnostic": getattr(error, "native_diagnostic", None),
        }
        target = self.logs_dir.parent / "operator"
        try:
            if any(path.is_symlink() for path in (target, *target.parents)):
                return
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.stat().st_mode & 0o777 != 0o700:
                return
            path = target / "claude-native-failures.jsonl"
            if path.is_symlink() or path.exists() and not path.is_file():
                return
            if path.exists() and path.stat().st_mode & 0o777 != 0o600:
                return
            with open(path, "a", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, 0o600)) as stream:
                stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
        except OSError:
            # The original classified failure remains authoritative if even the
            # separate operator destination is unavailable. Never repair source.
            pass


    def _native_integrity_error(self, message: str, *, result=None) -> AgentEvidenceIntegrityError:
        error = AgentEvidenceIntegrityError(message)
        if result is not None:
            try:
                value = json.loads(result.stdout)
                record = value["native_evidence_failure"]
                if (isinstance(record, dict) and isinstance(record.get("rule"), str)
                        and isinstance(record.get("path"), str)):
                    error.native_diagnostic = {key: str(record[key])[:240]
                                               for key in ("rule", "path", "error_type") if key in record}
                    error.add_note("Native evidence rule/path: " + json.dumps(error.native_diagnostic, sort_keys=True))
            except (ValueError, KeyError, TypeError):
                pass
        return error


    async def _finish_native_evidence(
        self, environment: BaseEnvironment, ready: dict,
    ) -> tuple[asyncio.CancelledError | None, BaseException | None]:
        """Stop inference and verify export despite an outer learner deadline.

        A cancellation can arrive after the learner has returned, while its
        finally block is copying the private evidence. It is a deadline, not
        evidence corruption. Keep cleanup bounded and protected from that
        cancellation, then return it to the caller without accepting an
        unverified or incomplete export.
        """
        async def finish() -> None:
            async with asyncio.timeout(self._NATIVE_CLEANUP_TIMEOUT_SECONDS):
                await self._stop_native_observer(environment, ready)
                await self._export_native_evidence(environment)

        task = asyncio.create_task(finish(), name="claude-native-evidence-cleanup")
        cancellation = None
        while True:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    # Cancellation inside cleanup has not established source
                    # integrity. The caller must still classify it fail-closed.
                    return cancellation, exc
                cancellation = cancellation or exc
            except BaseException as exc:
                return cancellation, exc
            else:
                return cancellation, None


    async def _export_native_evidence(self, environment: BaseEnvironment) -> None:
        """Copy stopped root evidence to an operator directory without changing it.

        The downloader may chown its staging source. Only a distinct copy is
        handed to it; strict canonical metadata and hashes are checked before
        and after the transfer, including when the transfer itself fails.
        """
        code = self._native_evidence_checked_code(
            "import shutil\n"
            "files=snapshot()\n"
            "require(json.loads((base/'openrouter-wire/observer-stop.json').read_text()).get('stopped') is True,'observer_not_stopped',base/'openrouter-wire/observer-stop.json')\n"
            "privacy={'schema_version':1,'source_root':str(base),'source_uid':0,'source_gid':0,'source_mode':0o700,'wire_uid':0,'wire_gid':0,'wire_mode':0o700,'non_private_entries':[],'observer_execution_uid':0,'observer_execution_gid':0,'observer_stopped':True,'exported_after_observer_stop':True,'native_evidence_scope':'operator_private','canonical_files':files}\n"
            "privacy_path=base/'privacy.json'; require(not privacy_path.exists(),'existing_attestation',privacy_path)\n"
            "privacy_path.write_text(json.dumps(privacy,sort_keys=True)+'\\n'); privacy_path.chmod(0o600)\n"
            "export=base/'export'; export.mkdir(mode=0o700,exist_ok=False)\n"
            "for name in [*files,'privacy.json']:\n"
            " source=base/name; target=export/name\n"
            " target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)\n"
            " shutil.copyfile(source,target); target.chmod(0o600)\n"
            "print(json.dumps({'canonical_files':files,'canonical_snapshot_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'privacy_sha256':hashlib.sha256(privacy_path.read_bytes()).hexdigest()}))\n"
        )
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(code), user="root")
        if result.return_code != 0:
            error = self._native_integrity_error("Canonical native evidence was not private at export", result=result)
            self._record_native_cleanup_failure(error, "export_source")
            raise error
        try:
            before = json.loads(result.stdout)
            if (not isinstance(before["canonical_files"], dict)
                    or not isinstance(before["canonical_snapshot_sha256"], str)
                    or not isinstance(before["privacy_sha256"], str)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise self._native_integrity_error("Native canonical snapshot receipt is invalid") from None
        target = self.logs_dir.parent / "operator" / "claude-native"
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise self._native_integrity_error("Native operator export destination contains a symlink")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.parent.stat().st_mode & 0o777 != 0o700:
            raise self._native_integrity_error("Native operator export parent is not private")
        target.mkdir(mode=0o700, exist_ok=False)
        transfer_error = None
        try:
            await environment.download_dir(self._NATIVE_PRIVATE_ROOT + "/export", target)
        except Exception as exc:
            transfer_error = exc
        # A failed downloader never exempts the source from integrity checks.
        after_code = self._native_evidence_checked_code("print(json.dumps(snapshot(),sort_keys=True))\n")
        result = await environment.exec(command="python3 -I -B -c " + shlex.quote(after_code), user="root")
        if result.return_code != 0:
            error = self._native_integrity_error("Canonical native evidence changed during operator export", result=result)
            self._record_native_cleanup_failure(error, "export_source_after_transfer")
            raise error from None
        try:
            after = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise self._native_integrity_error("Native canonical recheck receipt is invalid") from None
        if after != before["canonical_files"]:
            raise self._native_integrity_error("Canonical native evidence changed during operator export")
        if transfer_error is not None:
            if isinstance(transfer_error, AgentEvidenceIntegrityError):
                raise transfer_error
            error = AgentCleanupError("Native operator evidence transfer failed after unchanged-source verification")
            error.native_diagnostic = {"rule": "download_transfer", "path": "export", "error_type": type(transfer_error).__name__}
            self._record_native_cleanup_failure(error, "export_transfer")
            raise error from None
        expected_files = set(before["canonical_files"]) | {"privacy.json"}
        actual_files = set()
        expected_dirs = {str(parent) for name in expected_files for parent in Path(name).parents if str(parent) != "."}
        for path in target.rglob("*"):
            relative = str(path.relative_to(target))
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise self._native_integrity_error("Native operator export contains a linked or special file")
            if path.is_dir() and relative not in expected_dirs:
                raise self._native_integrity_error("Native operator export contains an unexpected directory")
            if path.is_file():
                actual_files.add(relative)
        if actual_files != expected_files:
            raise self._native_integrity_error("Native operator export file inventory differs from canonical evidence")
        for name, binding in before["canonical_files"].items():
            path = target / name
            if (path.resolve() != path.absolute() or not path.is_file()
                    or path.stat().st_size != binding["size_bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]):
                raise self._native_integrity_error("Native operator export does not match canonical evidence")
        if hashlib.sha256((target / "privacy.json").read_bytes()).hexdigest() != before["privacy_sha256"]:
            raise self._native_integrity_error("Native privacy attestation changed during export")
        for path in target.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        receipt = {
            "schema_version": 1, "source_root": self._NATIVE_PRIVATE_ROOT,
            "canonical_snapshot_sha256": before["canonical_snapshot_sha256"],
            "canonical_unchanged_after_export": True, "exported_after_observer_stop": True,
            "downloaded_source": "protected_export_staging", "native_evidence_scope": "operator_private",
        }
        path = target / "operator-export.json"
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        path.chmod(0o600)


    def _resolve_auth_env(self) -> dict[str, str]:
        if self._is_openrouter_model():
            key = (self._get_env("OPENROUTER_API_KEY") or "").strip()
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is required for openrouter/* models"
                )
            return {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": key,
                "OPENROUTER_API_KEY": key,
                "ANTHROPIC_BASE_URL": self._get_env("ANTHROPIC_BASE_URL")
                or self._OPENROUTER_ANTHROPIC_BASE_URL,
            }
        access = self.model_connection
        use_bedrock = access.provider is None
        # CLAUDE_FORCE_OAUTH drops the API key so the CLI uses the subscription
        # token; default keeps the key (the CLI prefers it). Mirrors codex. Only
        # the per-call env is dropped, not a key set in extra_env.
        force_oauth = self._should_force_oauth()
        oauth_token = (self._get_env("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()

        if force_oauth and not oauth_token:
            raise RuntimeError(
                "CLAUDE_FORCE_OAUTH is set but CLAUDE_CODE_OAUTH_TOKEN is not. "
                "Run `claude setup-token` to get one, or unset CLAUDE_FORCE_OAUTH."
            )

        if force_oauth or use_bedrock:
            api_key = ""
        else:
            api_key = access.api_key or ""
            if api_key and oauth_token:
                self.logger.debug(
                    "API key and OAuth token both set; using the API key "
                    "(set CLAUDE_FORCE_OAUTH=1 to use the subscription)."
                )

        env: dict[str, str | None] = {
            # "" when forced, then dropped by the empty-value filter below.
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_BASE_URL": (None if use_bedrock else access.configured_base_url),
            "CLAUDE_CODE_OAUTH_TOKEN": oauth_token,
        }

        # Bedrock configuration: pass through AWS credentials and region
        if use_bedrock:
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"

            # AWS Bedrock API key auth (Option E from Bedrock docs)
            bedrock_token = self._get_env("AWS_BEARER_TOKEN_BEDROCK") or ""
            if bedrock_token:
                env["AWS_BEARER_TOKEN_BEDROCK"] = bedrock_token

            # Standard AWS credential chain (Option B from Bedrock docs)
            for aws_var in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_PROFILE",
            ):
                val = self._get_env(aws_var) or ""
                if val:
                    env[aws_var] = val

            # AWS_REGION is required for Bedrock; default to us-east-1
            env["AWS_REGION"] = self._get_env("AWS_REGION") or "us-east-1"

            # Optional: separate region for the small/fast model (Haiku)
            small_model_region = (
                self._get_env("ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION") or ""
            )
            if small_model_region:
                env["ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION"] = small_model_region

            # Optional: disable prompt caching (not available in all regions)
            if (self._get_env("DISABLE_PROMPT_CACHING") or "").strip() == "1":
                env["DISABLE_PROMPT_CACHING"] = "1"

        # Remove empty auth credentials to allow Claude CLI to prioritize the available method
        # When both are empty, Claude CLI will fail with a clear authentication error
        return {k: v for k, v in env.items() if v}

    @override
    def acp_env(self) -> dict[str, str]:
        if self._is_guarded_openrouter_model():
            raise ValueError("Guarded native models require the observed native CLI route, not ACP")
        # Model selection (ANTHROPIC_MODEL, tier aliases, effort) stays in
        # acp_command(): its `env KEY=VAL` prefixes win at the target process,
        # while this dict lands in the shared exec overlay where model vars
        # would clobber the user agent's own selection.
        return self._resolve_auth_env()

    @override
    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        guarded = self._is_guarded_openrouter_model()
        if guarded:
            self._validate_guarded_configuration()
        env = self._resolve_auth_env()
        openrouter_key = env.get("OPENROUTER_API_KEY", "")
        max_output_tokens = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        if max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = max_output_tokens

        # Handle model name based on whether using custom API base or Bedrock
        resolved_model = (
            self._anthropic_model_name()
            if self._is_openrouter_model()
            else self._resolved_model_name()
        )
        if resolved_model:
            env["ANTHROPIC_MODEL"] = resolved_model

        # When using custom base URL, set all model aliases to the same model
        if "ANTHROPIC_BASE_URL" in env and "ANTHROPIC_MODEL" in env:
            env["ANTHROPIC_DEFAULT_FABLE_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]

        # Disable adaptive thinking if requested
        if os.environ.get("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "").strip() == "1":
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"

        # Disable non-essential traffic (telemetry, etc.)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        # Allow bypassPermissions mode when running as root inside containers
        env["IS_SANDBOX"] = "1"

        # Merge declarative env vars (e.g. MAX_THINKING_TOKENS)
        env.update(self._resolved_env_vars)

        env["CLAUDE_CONFIG_DIR"] = (self.environment_logs_dir / "sessions").as_posix()

        ready = None
        original_error = None
        try:
            if guarded:
                ready = await self._start_native_observer(environment, openrouter_key)
                policy = model_policy(self._guarded_request_model())
                env.update({"ANTHROPIC_BASE_URL": ready["base_url"],
                            "ANTHROPIC_AUTH_TOKEN": "picbench-wire-observer", "OPENROUTER_API_KEY": "",
                            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(policy["output_cap"]),
                            "API_TIMEOUT_MS": str(policy["api_client_timeout_ms"]),
                            "BASH_DEFAULT_TIMEOUT_MS": str(self._KIMI_BASH_TIMEOUT_MS),
                            "BASH_MAX_TIMEOUT_MS": str(self._NATIVE_BASH_MAX_TIMEOUT_MS)})
                env.update(self._native_stream_watchdog_environment())
                env.update(self._native_cli_retry_environment())
                # The reviewed model routes require complete native tool
                # definitions; the wire guard independently rejects deferred
                # tools on every request, without rewriting their schemas.
                env["ENABLE_TOOL_SEARCH"] = "false"
                # The key belongs to the observer process, not Claude's tools.
                self._extra_env.pop("OPENROUTER_API_KEY", None)
                await self._record_native_configuration(environment, ready, env)

            setup_command = (
                "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
                "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
                "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
                "if [ -d ~/.claude/skills ]; then "
                "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "
                "fi"
            )
            if self.config_source is not None:
                setup_command += (
                    f" && mkdir -p {shlex.quote(self._REMOTE_SETTINGS_DIR.as_posix())}"
                )

            skills_command = self._build_register_skills_command()
            if skills_command:
                setup_command += f" && {skills_command}"

            memory_command = self._build_register_memory_command()
            if memory_command:
                setup_command += f" && {memory_command}"

            mcp_command = self._build_register_mcp_servers_command()
            if mcp_command:
                setup_command += f" && {mcp_command}"

            cli_flags = self.build_cli_flags()
            extra_flags = (cli_flags + " ") if cli_flags else ""
            settings_flag = (
                f"--settings {shlex.quote(self._REMOTE_SETTINGS_PATH.as_posix())} "
                if self.config_source is not None
                else ""
            )
            identity = self._native_model_identity_settings()
            if identity:
                # Supply the model identity inline at the native command's
                # settings precedence, including suspended/resumed turns.
                # The learner cannot alter this serialized mapping on disk.
                native_settings = {**self._base_settings, **identity}
                settings_flag = f"--settings {shlex.quote(json.dumps(native_settings))} "
            resume_flag = "--continue " if self._resume else ""

            await self.exec_as_agent(
                environment,
                command=setup_command,
                env=env,
            )

            if self._load:
                load_session_id = await self._seed_load_trajectory(environment)
                resume_flag = f"--resume {load_session_id} "

            try:
                await self._upload_base_settings(environment)
                suspension = getattr_static(environment, "_harbor_probe_suspension", None)
                if suspension is not None:

                    async def invoke(session_id, prompt, timeout):
                        result = await self.exec_as_agent(
                            environment,
                            command=self._suspendable_claude_command(
                                session_id, prompt, settings_flag + extra_flags
                            ),
                            env=env,
                            timeout_sec=max(1, math.ceil(timeout)),
                        )
                        return result.stdout

                    await suspension.run_turns(invoke, instruction)
                    return

                instruction_shell_var = f"harbor_claude_code_instruction_{uuid.uuid4().hex}"
                instruction_env_var = instruction_shell_var.upper()
                run_env = {**env, instruction_env_var: instruction}

                await self.exec_as_agent(
                    environment,
                    command=(
                        'export PATH="$HOME/.local/bin:$PATH"; '
                        f'{instruction_shell_var}="${instruction_env_var}"; '
                        f"unset {instruction_env_var}; "
                        f'printf "%s" "${instruction_shell_var}" | '
                        f"claude --verbose --output-format=stream-json "
                        f"{settings_flag}"
                        f"{extra_flags}"
                        f"{resume_flag}"
                        f"--print 2>&1 | tee "
                        f"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}"
                    ),
                    env=run_env,
                )
            finally:
                if self.config_source is not None:
                    try:
                        await self.exec_as_agent(
                            environment,
                            command=(
                                f"rm -rf "
                                f"{shlex.quote(self._REMOTE_SETTINGS_DIR.as_posix())}"
                            ),
                            env=env,
                        )
                    except Exception:
                        pass
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            if guarded and ready is not None:
                cancellation, cleanup_error = await self._finish_native_evidence(environment, ready)
                if cancellation is not None and original_error is None:
                    original_error = cancellation
                if cleanup_error is not None:
                    if not isinstance(cleanup_error, (AgentCleanupError, AgentEvidenceIntegrityError)):
                        error = self._native_integrity_error("Native observer cleanup did not establish stopped, unchanged evidence")
                        error.native_diagnostic = {"rule": "cleanup_unverified", "path": ".", "error_type": type(cleanup_error).__name__}
                        cleanup_error = error
                    self._record_native_cleanup_failure(cleanup_error, "cleanup", original=original_error)
                    if isinstance(cleanup_error, AgentEvidenceIntegrityError):
                        # A privacy/source failure must never become an accepted
                        # ordinary learner cutoff. Retain its original cause.
                        if original_error is not None:
                            raise cleanup_error from original_error
                        raise cleanup_error from None
                    # A normal cutoff must not hide missing required operator
                    # evidence. The central lifecycle policy consumes this
                    # trusted, bounded marker separately from the learner error.
                    if context.metadata is None:
                        context.metadata = {}
                    failures = context.metadata.get("agent_cleanup_failures")
                    if not isinstance(failures, list):
                        failures = []
                        context.metadata["agent_cleanup_failures"] = failures
                    failures.append({
                        "status": "failed", "error_type": type(cleanup_error).__name__,
                        "phase": "native_evidence_cleanup",
                        "diagnostic_path": "operator/claude-native-failures.jsonl",
                    })
                    if original_error is not None:
                        # The transfer is operational and canonical evidence was
                        # independently unchanged. Keep the actual learner error.
                        original_error.add_note("Operational native evidence cleanup also failed; see operator/claude-native-failures.jsonl")
                    else:
                        raise cleanup_error from None
                if cancellation is not None and original_error is cancellation:
                    raise cancellation

    def _is_openrouter_model(self) -> bool:
        return bool(self.model_name and self.model_name.startswith("openrouter/"))

    def _anthropic_model_name(self) -> str:
        if not self.model_name:
            raise ValueError("Model name is required")
        if self._is_openrouter_model():
            model = self.model_name.split("/", 1)[1]
            if not model:
                raise ValueError(
                    "OpenRouter model name must include a provider/model slug"
                )
            return model
        return self.model_name

    def _suspendable_claude_command(
        self, session_id: str | None, prompt: str, flags: str
    ) -> str:
        """One native completed call, resuming only the host-bound session."""
        from uuid import UUID

        continuation = ""
        if session_id is not None:
            if not isinstance(session_id, str) or str(UUID(session_id)) != session_id:
                raise ValueError("Invalid Claude Code continuation identity")
            continuation = f"--resume {shlex.quote(session_id)} "
        output = shlex.quote(
            (EnvironmentPaths.agent_dir / "claude-code.txt").as_posix()
        )
        return (
            'set -o pipefail; umask 077; export PATH="$HOME/.local/bin:$PATH"; '
            f"claude --verbose --output-format=stream-json {flags}{continuation}"
            f"--print -- {shlex.quote(prompt)} 2>&1 </dev/null | tee -a {output}"
        )
