import json
import logging
import os
import re
import stat
import tempfile
import math
from inspect import getattr_static
from pathlib import Path
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from harbor.utils.env import is_sensitive_env_key, resolve_env_vars
from harbor.utils.scripts import (
    build_execution_command,
    needs_chmod,
    quote_shell_arg,
)


class AddTestsDirError(Exception):
    pass


class VerifierOutputParseError(Exception):
    pass


class DownloadVerifierDirError(Exception):
    pass


class RewardFileNotFoundError(FileNotFoundError):
    pass


class RewardFileEmptyError(Exception):
    pass


_MAX_REWARD_BYTES = 1024 * 1024
_MAX_FAILURE_OUTPUT_BYTES = 1024 * 1024
_REWARD_MISSING_EXIT_CODE = 66


def _atomic_write_text(path: Path, content: str, *, staging_dir: Path) -> None:
    """Replace a host log without following an untrusted directory entry.

    The temporary inode is created outside the agent-mounted verifier
    directory so an in-container process cannot swap the staging name before
    the atomic replacement.
    """
    if path.parent.is_symlink() or staging_dir.is_symlink():
        raise RuntimeError(
            "Refusing to publish verifier output through a symlinked output "
            f"or staging directory ({path.parent}, {staging_dir})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=staging_dir,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            # `fdopen` owns and normally closes the descriptor.
            pass
        temporary_path.unlink(missing_ok=True)


def _combined_exec_output(stdout: str | None, stderr: str | None) -> str:
    """Preserve Harbor's historical combined verifier-output artifact."""
    output = stdout or ""
    if stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += stderr
    return output


def _sanitized_failure_output(output: str, env: dict[str, str] | None) -> str:
    """Retain bounded diagnostics, not credentials or a synthetic reward."""
    secrets = {
        value
        for key, value in {**os.environ, **(env or {})}.items()
        if value and is_sensitive_env_key(key)
    }
    for value in sorted(secrets, key=len, reverse=True):
        output = output.replace(value, "[REDACTED]")
    # Also cover labelled credentials not present in Harbor's environment.
    output = re.sub(
        r"""(?<![\w-])(["']?[\w-]{0,128}(?:api[_-]?key|secret|token|password|credential)[\w-]{0,128}["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)""",
        r"\1[REDACTED]",
        output,
        flags=re.IGNORECASE,
    )
    output = re.sub(r"\bBearer\s+\S+", "Bearer [REDACTED]", output, flags=re.IGNORECASE)
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) > _MAX_FAILURE_OUTPUT_BYTES:
        notice = b"\n[Harbor: verifier failure output truncated]\n"
        output = encoded[: _MAX_FAILURE_OUTPUT_BYTES - len(notice)].decode(
            "utf-8", errors="ignore"
        )
        output += notice.decode()
    return output


class Verifier(BaseVerifier):
    def __init__(
        self,
        task: Task,
        trial_paths: TrialPaths,
        environment: BaseEnvironment,
        override_env: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
        skip_tests_upload: bool = False,
        verifier_env: dict[str, str] | None = None,
        step_name: str | None = None,
        include_logs: list[str] | None = None,
        exclude_logs: list[str] | None = None,
    ):
        super().__init__(
            task=task,
            trial_paths=trial_paths,
            environment=environment,
            override_env=override_env,
            logger=logger,
            verifier_env=verifier_env,
            step_name=step_name,
            include_logs=include_logs,
            exclude_logs=exclude_logs,
        )
        self._skip_tests_upload = skip_tests_upload

    @staticmethod
    def _parse_reward_text_content(
        content: str, source: Path | str
    ) -> dict[str, float | int]:
        if not content:
            raise RewardFileEmptyError(f"Reward file is empty at {source}")
        try:
            value = float(content)
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from text file {source}"
            ) from e
        if not math.isfinite(value):
            raise VerifierOutputParseError(
                f"Non-finite reward {value!r} in text file {source}"
            )
        return {"reward": value}

    @staticmethod
    def _parse_reward_json_content(
        content: str, source: Path | str
    ) -> dict[str, float | int]:
        if not content:
            raise RewardFileEmptyError(f"Reward file is empty at {source}")
        try:
            rewards = json.loads(content)
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from JSON file {source}"
            ) from e
        # Reject non-numeric and non-finite reward values. json.loads parses
        # NaN/Infinity tokens *and* numeric overflows like 1e309 into non-finite
        # floats, and string values like "NaN" would otherwise be coerced to
        # float("nan") later by pydantic, so require real finite numbers here.
        values = rewards.values() if isinstance(rewards, dict) else [rewards]
        for value in values:
            if not isinstance(value, (int, float)):
                raise VerifierOutputParseError(
                    f"Non-numeric reward {value!r} in JSON file {source}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise VerifierOutputParseError(
                    f"Non-finite reward {value!r} in JSON file {source}"
                )
        return rewards

    def _parse_reward_text(self) -> dict[str, float | int]:
        return self._parse_reward_text_content(
            self.trial_paths.reward_text_path.read_text(),
            self.trial_paths.reward_text_path,
        )

    def _parse_reward_json(self) -> dict[str, float | int]:
        return self._parse_reward_json_content(
            self.trial_paths.reward_json_path.read_text(),
            self.trial_paths.reward_json_path,
        )

    def _uses_secure_mounted_reward_capture(self) -> bool:
        capabilities = self.environment.capabilities
        return (
            capabilities.mounted
            and getattr_static(capabilities, "docker_compose", False) is True
            and self.environment.os != TaskOS.WINDOWS
        )

    async def _prepare_mounted_verifier_outputs(
        self, env_paths: EnvironmentPaths
    ) -> None:
        """Remove canonical output links before the trusted test opens them.

        Root verifiers additionally keep the bind mount root-owned and mode
        0700 for the whole test/capture/publication transaction. This prevents
        a pre-existing learner symlink from being followed by an ordinary
        shell redirection in the verifier script.
        """
        verifier_dir = quote_shell_arg(str(env_paths.verifier_dir), TaskOS.LINUX)
        names = " ".join(
            quote_shell_arg(name, TaskOS.LINUX)
            for name in (
                env_paths.reward_json_path.name,
                env_paths.reward_text_path.name,
                self.trial_paths.test_stdout_path.name,
            )
        )
        effective_user = getattr(self.environment, "default_user", None)
        root_verifier = effective_user in {0, "0", "root"}
        lock = (
            'chmod 0000 "$verifier_dir"; '
            'chown 0:0 "$verifier_dir"; '
            'chmod 0700 "$verifier_dir"; '
            if root_verifier
            else ""
        )
        result = await self.environment.exec(
            command=(
                "set -eu; "
                f"verifier_dir={verifier_dir}; "
                'if [ -L "$verifier_dir" ]; then exit 74; fi; '
                f"{lock}"
                'cd "$verifier_dir"; '
                f"rm -f -- {names}"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise VerifierOutputParseError(
                "Failed to prepare canonical mounted verifier outputs safely"
            )

    async def _capture_mounted_reward(
        self, env_paths: EnvironmentPaths
    ) -> tuple[Path, str, str, dict[str, float | int]]:
        """Lock and copy a Docker bind-mounted reward without following links.

        The verifier may intentionally leave `/logs/verifier` root-owned and
        mode 0700. Docker's no-`-L` copy preserves a source symlink rather than
        following it; Harbor then accepts only one bounded regular file in a
        host-private staging directory.
        """
        verifier_dir = quote_shell_arg(str(env_paths.verifier_dir), TaskOS.LINUX)
        reward_json_name = quote_shell_arg(
            env_paths.reward_json_path.name, TaskOS.LINUX
        )
        reward_text_name = quote_shell_arg(
            env_paths.reward_text_path.name, TaskOS.LINUX
        )
        stdout_name = quote_shell_arg(
            self.trial_paths.test_stdout_path.name, TaskOS.LINUX
        )
        command = f"""
set -eu
verifier_dir={verifier_dir}
if [ -L "$verifier_dir" ]; then
    echo 'Verifier output directory is a symlink' >&2
    exit 74
fi
chown 0:0 "$verifier_dir"
chmod 0700 "$verifier_dir"
cd "$verifier_dir"
rm -f {stdout_name}
if [ -e {reward_json_name} ] || [ -L {reward_json_name} ]; then
    reward_file={reward_json_name}
    reward_kind=json
elif [ -e {reward_text_name} ] || [ -L {reward_text_name} ]; then
    reward_file={reward_text_name}
    reward_kind=text
else
    echo 'No reward file found' >&2
    exit {_REWARD_MISSING_EXIT_CODE}
fi
if [ -L "$reward_file" ] || [ ! -f "$reward_file" ]; then
    echo 'Reward entry is not a regular nofollow file' >&2
    exit 74
fi
reward_size=$(stat -c %s "$reward_file" 2>/dev/null) || {{
    echo 'Could not inspect reward file size' >&2
    exit 74
}}
case "$reward_size" in
    ''|*[!0-9]*) echo 'Reward file size is invalid' >&2; exit 74 ;;
esac
if [ "$reward_size" -gt {_MAX_REWARD_BYTES} ]; then
    echo 'Reward file exceeds Harbor size limit' >&2
    exit 75
fi
printf '%s\\n' "$reward_kind"
""".strip()
        result = await self.environment.exec(command=command, user="root")
        if result.return_code == _REWARD_MISSING_EXIT_CODE:
            raise RewardFileNotFoundError(
                f"No reward file found at {self.trial_paths.reward_text_path} or "
                f"{self.trial_paths.reward_json_path}"
            )
        if result.return_code != 0:
            output = (result.stdout or result.stderr or "<none>").strip()
            raise VerifierOutputParseError(
                "Failed to lock verifier output safely inside the container "
                f"(return code {result.return_code}): {output}"
            )

        reward_kind = (result.stdout or "").strip()
        if reward_kind == "json":
            container_path = str(env_paths.reward_json_path)
            host_path = self.trial_paths.reward_json_path
        elif reward_kind == "text":
            container_path = str(env_paths.reward_text_path)
            host_path = self.trial_paths.reward_text_path
        else:
            raise VerifierOutputParseError(
                f"Secure verifier reward selection returned {reward_kind!r}"
            )

        with tempfile.TemporaryDirectory(
            dir=self.trial_paths.trial_dir,
            prefix=".harbor-reward-capture-",
        ) as temporary_dir:
            captured_path = Path(temporary_dir) / host_path.name
            copied = await self.environment.download_file_nofollow(
                container_path, captured_path
            )
            if not copied:
                raise VerifierOutputParseError(
                    f"Docker could not copy selected reward file {container_path}"
                )
            metadata = captured_path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise VerifierOutputParseError(
                    f"Reward entry {container_path} is not a nofollow regular file"
                )
            if metadata.st_size > _MAX_REWARD_BYTES:
                raise VerifierOutputParseError(
                    f"Reward file {container_path} exceeds {_MAX_REWARD_BYTES} bytes"
                )
            content = captured_path.read_text()

        if reward_kind == "json":
            rewards = self._parse_reward_json_content(content, host_path)
        else:
            rewards = self._parse_reward_text_content(content, host_path)
        return host_path, container_path, content, rewards

    async def _publish_locked_mounted_outputs(
        self,
        *,
        env_paths: EnvironmentPaths,
        reward_container_path: str | None,
        reward_content: str | None,
        test_output: str,
    ) -> None:
        """Publish canonical files while the verifier mount remains root-locked."""
        verifier_dir = quote_shell_arg(str(env_paths.verifier_dir), TaskOS.LINUX)
        reward_json_name = quote_shell_arg(
            env_paths.reward_json_path.name, TaskOS.LINUX
        )
        reward_text_name = quote_shell_arg(
            env_paths.reward_text_path.name, TaskOS.LINUX
        )
        stdout_name = quote_shell_arg(
            self.trial_paths.test_stdout_path.name, TaskOS.LINUX
        )
        remove_result = await self.environment.exec(
            command=(
                f"set -eu; cd {verifier_dir}; "
                f"rm -f {reward_json_name} {reward_text_name} {stdout_name}"
            ),
            user="root",
        )
        if remove_result.return_code != 0:
            raise VerifierOutputParseError(
                "Failed to clear canonical verifier entries before safe publication"
            )

        with tempfile.TemporaryDirectory(
            dir=self.trial_paths.trial_dir,
            prefix=".harbor-verifier-publish-",
        ) as temporary_dir:
            temporary_root = Path(temporary_dir)
            reward_source = temporary_root / "reward"
            stdout_source = temporary_root / "test-stdout.txt"
            stdout_source.write_text(test_output)
            if reward_container_path is not None and reward_content is not None:
                reward_source.write_text(reward_content)
                await self.environment.upload_file(reward_source, reward_container_path)
            await self.environment.upload_file(
                stdout_source,
                str(env_paths.verifier_dir / self.trial_paths.test_stdout_path.name),
            )

        # Keep the live mount readable by the host but immutable to a detached
        # non-root agent process. `find` does not follow symlinks by default.
        finalize_result = await self.environment.exec(
            command=f"""
set -eu
verifier_dir={verifier_dir}
find "$verifier_dir" -xdev -type d -exec chown 0:0 {{}} \\;
find "$verifier_dir" -xdev -type f -exec chown 0:0 {{}} \\;
find "$verifier_dir" -xdev -type d -exec chmod 0755 {{}} \\;
find "$verifier_dir" -xdev -type f -exec chmod 0644 {{}} \\;
""".strip(),
            user="root",
        )
        if finalize_result.return_code != 0:
            raise VerifierOutputParseError(
                "Failed to expose verifier artifacts as root-owned read-only files"
            )

    async def _retain_failure_output(
        self, env_paths: EnvironmentPaths, output: str, env: dict[str, str] | None
    ) -> None:
        """Best-effort diagnostic retention must never mask the original failure."""
        try:
            output = _sanitized_failure_output(output, env)
            if self._uses_secure_mounted_reward_capture():
                await self._prepare_mounted_verifier_outputs(env_paths)
                await self._publish_locked_mounted_outputs(
                    env_paths=env_paths,
                    reward_container_path=None,
                    reward_content=None,
                    test_output=output,
                )
            else:
                _atomic_write_text(
                    self.trial_paths.test_stdout_path,
                    output,
                    staging_dir=self.trial_paths.trial_dir,
                )
        except Exception:
            self.logger.warning("Could not retain verifier failure output safely")

    def _resolve_tests(self) -> tuple[list[Path], Path, Path]:
        if self._skip_tests_upload:
            # The verifier image already owns /tests/test.{sh,bat}.
            return (
                [],
                self.task.paths.tests_dir,
                self.task.paths.test_path_for(self.environment.os),
            )

        if self.step_name is None:
            discovered = self.task.paths.discovered_test_path_for(self.environment.os)
            if discovered is None:
                raise FileNotFoundError(
                    f"No test script found in: {self.task.paths.tests_dir} "
                    f"(target OS: {self.environment.os.value})"
                )
            return [self.task.paths.tests_dir], self.task.paths.tests_dir, discovered

        step_tests_dir = self.task.paths.step_tests_dir(self.step_name)
        source_dirs = []
        if self.task.paths.tests_dir.exists():
            source_dirs.append(self.task.paths.tests_dir)
        if step_tests_dir.exists():
            source_dirs.append(step_tests_dir)

        step_test_path = self.task.paths.discovered_step_test_path_for(
            self.step_name, self.environment.os
        )
        shared_test_path = self.task.paths.discovered_test_path_for(self.environment.os)
        if step_test_path is not None:
            return source_dirs, step_tests_dir, step_test_path
        if shared_test_path is not None:
            return source_dirs, self.task.paths.tests_dir, shared_test_path

        raise FileNotFoundError(
            f"No {self.environment.os.value} test script found for step "
            f"'{self.step_name}': expected "
            f"{self.task.paths.step_test_path_for(self.step_name, self.environment.os)} "
            f"or {self.task.paths.test_path_for(self.environment.os)}"
        )

    @override
    async def verify(self) -> VerifierResult:
        """
        Grades the agents performance based on the environment.
        Returns:
            (VerifierResult): The result of the verifier.
        """
        env_paths = EnvironmentPaths.for_os(self.environment.os)
        test_source_dirs, tests_source_dir, host_test_path = self._resolve_tests()

        if not self._skip_tests_upload:
            try:
                for source_dir in test_source_dirs:
                    self.task.paths.validate_input_tree(source_dir)
                for source_dir in test_source_dirs:
                    await self.environment.upload_dir(
                        source_dir=source_dir,
                        target_dir=str(env_paths.tests_dir),
                    )
            except Exception as e:
                raise AddTestsDirError(
                    "Failed to add tests directory to environment."
                ) from e

        merged_env = {
            **self.task.config.verifier.env,
            **(self.verifier_env or {}),
            **self.override_env,
        }
        env = None
        if merged_env:
            for key in merged_env:
                if "api_key" in key.lower():
                    self.logger.debug(
                        "The verifier.env contains an API key (often the case for LLM-"
                        "based verifiers). You will incur costs associated with the "
                        "API calls."
                    )
            env = resolve_env_vars(merged_env)

        test_script_path = str(
            env_paths.tests_dir
            / host_test_path.relative_to(tests_source_dir).as_posix()
        )
        command = build_execution_command(
            test_script_path,
            task_os=self.environment.os,
        )

        # A separate verifier can use a launcher already baked into a
        # read-only evaluator image. In that mode the image build is
        # responsible for making the launcher executable; attempting chmod
        # here would fail before verification starts.
        if needs_chmod(test_script_path) and not self._skip_tests_upload:
            await self.environment.exec(
                command=f"chmod +x {quote_shell_arg(test_script_path, self.environment.os)}",
                user="root",
            )

        if self._uses_secure_mounted_reward_capture():
            await self._prepare_mounted_verifier_outputs(env_paths)

        # Runs as ``environment.default_user``, which the caller must set to the
        # effective verifier user (step-level override or task-level fallback).
        test_result = await self.environment.exec(
            command=command,
            env=env,
        )
        test_output = _combined_exec_output(test_result.stdout, test_result.stderr)

        if self._uses_secure_mounted_reward_capture():
            try:
                (
                    _reward_host_path,
                    reward_container_path,
                    reward_content,
                    rewards,
                ) = await self._capture_mounted_reward(env_paths)
            except Exception:
                await self._retain_failure_output(env_paths, test_output, env)
                raise
            await self._publish_locked_mounted_outputs(
                env_paths=env_paths,
                reward_container_path=reward_container_path,
                reward_content=reward_content,
                test_output=test_output,
            )
            return VerifierResult(rewards=rewards)

        if not self.environment.capabilities.mounted:
            try:
                if self.include_logs or self.exclude_logs:
                    await self.environment.download_dir_filtered(
                        source_dir=str(env_paths.verifier_dir),
                        target_dir=self.trial_paths.verifier_dir,
                        include=self.include_logs or None,
                        exclude=self.exclude_logs or None,
                        protect=[
                            self.trial_paths.reward_text_path.name,
                            self.trial_paths.reward_json_path.name,
                        ],
                    )
                else:
                    await self.environment.download_dir(
                        source_dir=str(env_paths.verifier_dir),
                        target_dir=self.trial_paths.verifier_dir,
                    )
            except Exception as e:
                raise DownloadVerifierDirError(
                    "Failed to download verifier directory from environment"
                ) from e

        try:
            if self.trial_paths.reward_json_path.exists():
                rewards = self._parse_reward_json()
            elif self.trial_paths.reward_text_path.exists():
                rewards = self._parse_reward_text()
            else:
                raise RewardFileNotFoundError(
                    f"No reward file found at {self.trial_paths.reward_text_path} or {
                        self.trial_paths.reward_json_path
                    }"
                )
        except Exception:
            await self._retain_failure_output(env_paths, test_output, env)
            raise

        # Non-Docker environments publish captured output from the host. The
        # staging inode lives outside any agent-mounted verifier directory and
        # replacement never follows a destination symlink.
        _atomic_write_text(
            self.trial_paths.test_stdout_path,
            test_output,
            staging_dir=self.trial_paths.trial_dir,
        )

        return VerifierResult(rewards=rewards)
