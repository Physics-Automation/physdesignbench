"""Unix/Linux container operations for DockerEnvironment."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import stat
import tempfile
import io
import sys
import tarfile
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from harbor.constants import MAIN_SERVICE_NAME

if TYPE_CHECKING:
    from harbor.environments.docker.docker import DockerEnvironment


_DOWNLOAD_REGULAR_FILE_SCRIPT = r"""
path=$1
if [ ! -f "$path" ] || [ -L "$path" ]; then
    echo "artifact source is not a non-symlink regular file" >&2
    exit 66
fi
cat "$path"
""".strip()

# Bounded, uncompressed stream from the *live* mount namespace. Docker cp can
# silently copy an empty directory for tmpfs. Neither chown nor staging files
# inside the trusted sidecar is necessary. Paths remain separate argv values.
_MAX_SIDECAR_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_SIDECAR_ARCHIVE_MEMBERS = 100_000
_DOWNLOAD_DIRECTORY_SCRIPT = r"""
set -euo pipefail
artifact_dir=$1
if [ ! -d "$artifact_dir" ] || [ -L "$artifact_dir" ]; then
    echo "artifact source is not a non-symlink directory" >&2
    exit 66
fi
tar -C "$artifact_dir" -cf - . | head -c "$2"
""".strip()


def _extract_sidecar_directory(archive_path: Path, target: Path) -> None:
    """Extract only bounded regular files/directories into a private fresh tree."""
    if archive_path.stat().st_size > _MAX_SIDECAR_ARCHIVE_BYTES:
        raise ValueError("sidecar archive exceeds byte limit")
    seen: set[str] = set()
    total = 0
    with tarfile.open(archive_path, mode="r:") as archive:
        for index, member in enumerate(archive, start=1):
            if index > _MAX_SIDECAR_ARCHIVE_MEMBERS:
                raise ValueError("sidecar archive exceeds entry limit")
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe sidecar archive path")
            if not (member.isdir() or member.isreg()) or member.sparse is not None:
                raise ValueError("sidecar archive contains a link or special file")
            name = relative.as_posix()
            if name in seen or (name == "." and not member.isdir()):
                raise ValueError("duplicate or invalid sidecar archive path")
            seen.add(name)
            if name == ".":
                continue
            destination = target.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            total += member.size
            if member.size < 0 or total > _MAX_SIDECAR_ARCHIVE_BYTES:
                raise ValueError("sidecar archive exceeds extracted byte limit")
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("missing sidecar archive file contents")
            with source, destination.open("xb") as output:
                destination.chmod(0o600)
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.stat().st_size != member.size:
                raise ValueError("truncated sidecar archive file")


class UnixOps:
    """File transfer and exec operations for Linux containers."""

    def __init__(self, env: DockerEnvironment) -> None:
        self._env = env

    @staticmethod
    def _reset_tar_owner(info: tarfile.TarInfo) -> tarfile.TarInfo:
        # Some runtimes cannot map arbitrary host UIDs into the container.
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    @staticmethod
    def _container_file_target(
        source_path: Path | str, target_path: str, target_is_dir: bool = False
    ) -> tuple[str, str]:
        if target_is_dir or target_path.endswith("/"):
            return target_path.rstrip("/") or "/", Path(source_path).name

        target = PurePosixPath(target_path)
        return target.parent.as_posix(), target.name

    async def _container_path_is_dir(self, target_path: str) -> bool:
        result = await self._env._run_docker_compose_command(
            [
                "exec",
                "-T",
                "-u",
                "root",
                MAIN_SERVICE_NAME,
                "test",
                "-d",
                target_path,
            ],
            check=False,
        )
        return result.return_code == 0

    def _tar_dir_contents(self, source_dir: Path | str) -> bytes:
        source = Path(source_dir)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for path in source.rglob("*"):
                tar.add(
                    path,
                    arcname=path.relative_to(source).as_posix(),
                    recursive=False,
                    filter=self._reset_tar_owner,
                )
        return buffer.getvalue()

    def _tar_file(self, source_path: Path | str, arcname: str) -> bytes:
        source = Path(source_path)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(source, arcname=arcname, filter=self._reset_tar_owner)
        return buffer.getvalue()

    async def _extract_tar(self, tar_bytes: bytes, target_dir: str) -> None:
        await self._env._run_docker_compose_command(
            [
                "exec",
                "-T",
                "-u",
                "root",
                MAIN_SERVICE_NAME,
                "mkdir",
                "-p",
                target_dir,
            ],
            check=True,
        )
        await self._env._run_docker_compose_command(
            [
                "exec",
                "-T",
                "-u",
                "root",
                MAIN_SERVICE_NAME,
                "tar",
                "-xf",
                "-",
                "-C",
                target_dir,
            ],
            check=True,
            stdin_data=tar_bytes,
        )

    async def _upload_file_with_tar(
        self, source_path: Path | str, target_path: str
    ) -> None:
        target_is_dir = False
        if not target_path.endswith("/"):
            target_is_dir = await self._container_path_is_dir(target_path)
        target_dir, target_name = self._container_file_target(
            source_path, target_path, target_is_dir=target_is_dir
        )
        await self._extract_tar(self._tar_file(source_path, target_name), target_dir)

    async def _upload_dir_with_tar(
        self, source_dir: Path | str, target_dir: str
    ) -> None:
        await self._extract_tar(self._tar_dir_contents(source_dir), target_dir)

    async def _fallback_to_tar(
        self, cp_error: RuntimeError, operation: Callable[[], Awaitable[None]]
    ) -> None:
        self._env.logger.debug(
            "docker compose cp failed; retrying upload with tar stream: %s",
            cp_error,
        )
        try:
            await operation()
        except Exception as tar_error:
            raise RuntimeError(
                "Docker compose cp failed, and tar upload fallback also failed. "
                f"cp error: {cp_error}. tar error: {tar_error}"
            ) from tar_error

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if self._env.task_env_config.docker_read_only:
            source_info = source.stat()
            if not stat.S_ISREG(source_info.st_mode):
                raise ValueError(
                    f"Read-only Docker upload source must be a regular file: {source}"
                )
            target = shlex.quote(target_path)
            mode = stat.S_IMODE(source_info.st_mode)
            await self._env._run_docker_compose_command(
                [
                    "exec",
                    "-T",
                    "--user",
                    "0",
                    MAIN_SERVICE_NAME,
                    "sh",
                    "-c",
                    f"umask 077; cat > {target} && chmod {mode:o} {target}",
                ],
                check=True,
                stdin_path=source,
            )
            return
        try:
            await self._env._run_docker_compose_command(
                ["cp", str(source_path), f"{MAIN_SERVICE_NAME}:{target_path}"],
                check=True,
            )
        except RuntimeError as cp_error:
            await self._fallback_to_tar(
                cp_error, lambda: self._upload_file_with_tar(source_path, target_path)
            )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if self._env.task_env_config.docker_read_only:
            if source.is_symlink() or not source.is_dir():
                raise ValueError(
                    f"Read-only Docker upload source must be a non-symlink directory: {source}"
                )

            target_root = PurePosixPath(target_dir)
            target_dirs = [target_root.as_posix()]
            files: list[tuple[Path, str]] = []
            for current, dirnames, filenames in os.walk(
                source,
                topdown=True,
                followlinks=False,
            ):
                dirnames.sort()
                filenames.sort()
                current_path = Path(current)
                relative = current_path.relative_to(source)
                for dirname in dirnames:
                    directory = current_path / dirname
                    if directory.is_symlink():
                        raise ValueError(
                            f"Read-only Docker directory upload rejects symlinks: {directory}"
                        )
                    target_dirs.append((target_root / relative / dirname).as_posix())
                for filename in filenames:
                    file_path = current_path / filename
                    file_info = file_path.lstat()
                    if not stat.S_ISREG(file_info.st_mode):
                        raise ValueError(
                            "Read-only Docker directory upload requires regular files: "
                            f"{file_path}"
                        )
                    files.append(
                        (
                            file_path,
                            (target_root / relative / filename).as_posix(),
                        )
                    )

            result = await self._env.ensure_dirs(target_dirs, chmod=True)
            if result is not None and result.return_code != 0:
                raise RuntimeError(
                    "Could not prepare read-only Docker upload directories: "
                    f"{result.stderr or result.stdout or '<no diagnostics>'}"
                )
            for source_path, target_path in files:
                await self.upload_file(source_path, target_path)
            return

        try:
            await self._env._run_docker_compose_command(
                ["cp", f"{source_dir}/.", f"{MAIN_SERVICE_NAME}:{target_dir}"],
                check=True,
            )
        except RuntimeError as cp_error:
            await self._fallback_to_tar(
                cp_error, lambda: self._upload_dir_with_tar(source_dir, target_dir)
            )
        # Fix CRLF line endings when the host is Windows: shell scripts with
        # Windows line endings fail to execute inside the Linux container.
        if sys.platform == "win32":
            await self._env._run_docker_compose_command(
                [
                    "exec",
                    MAIN_SERVICE_NAME,
                    "bash",
                    "-c",
                    f"find {target_dir} -type f \\( -name '*.sh' -o -name '*.py' "
                    "-o -name '*.ps1' -o -name '*.cmd' -o -name '*.bat' \\) "
                    "-exec sed -i 's/\\r$//' {} \\;",
                ],
                check=False,
            )

    async def _resolve_service_container(self, service: str) -> str:
        """Return a container ID/name for *service* in this compose project.

        Compose V2 exposes ``compose cp``. podman-compose 1.x does not, so
        downloads fall back to ``podman cp`` / ``docker cp`` against the
        container. Names differ by frontend (``project_service_1`` vs
        ``project-service-1``).
        """
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        project = _sanitize_docker_compose_project_name(self._env.session_id)
        proc = await asyncio.create_subprocess_exec(
            *type(self._env)._engine_cmd(
                "ps",
                "-a",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.ID}}\t{{.Names}}",
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        suffixes = (f"_{service}_1", f"-{service}-1")
        for line in stdout.decode(errors="replace").splitlines():
            cid, _, name = line.partition("\t")
            name = name.strip()
            cid = cid.strip()
            if name.endswith(suffixes):
                return cid or name
        raise RuntimeError(
            f"Could not resolve container for compose service {service!r} "
            f"in project {project!r}."
        )

    async def _engine_cp(self, source: str, destination: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            *type(self._env)._engine_cmd("cp", source, destination),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            output = stdout.decode(errors="replace") if stdout else ""
            raise RuntimeError(
                f"Engine cp failed ({source!r} -> {destination!r}): {output}"
            )

    async def _download_via_engine_cp(
        self, container_path: str, target_path: Path | str, service: str
    ) -> None:
        container = await self._resolve_service_container(service)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await self._engine_cp(f"{container}:{container_path}", str(target_path))

    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
        service: str | None = None,
    ) -> None:
        service = service or MAIN_SERVICE_NAME
        if type(self._env).runtime().supports_compose_cp:
            try:
                await self._env._run_docker_compose_command(
                    ["cp", f"{service}:{source_path}", str(target_path)],
                    check=True,
                )
                return
            except RuntimeError as cp_error:
                self._env.logger.debug(
                    "compose cp failed; retrying download with engine cp: %s",
                    cp_error,
                )
        await self._download_via_engine_cp(source_path, target_path, service)

    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        service: str | None = None,
    ) -> None:
        service = service or MAIN_SERVICE_NAME
        if type(self._env).runtime().supports_compose_cp:
            try:
                await self._env._run_docker_compose_command(
                    ["cp", f"{service}:{source_dir}/.", str(target_dir)],
                    check=True,
                )
                return
            except RuntimeError as cp_error:
                self._env.logger.debug(
                    "compose cp failed; retrying download with engine cp: %s",
                    cp_error,
                )
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._download_via_engine_cp(f"{source_dir}/.", target_dir, service)

    @staticmethod
    def exec_shell_args(command: str) -> list[str]:
        """Return the shell wrapper for executing *command* in a Linux container."""
        return ["bash", "-c", command]

    async def download_running_service_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str,
    ) -> None:
        """Stream a trusted sidecar file through its live mount namespace.

        ``docker compose cp`` can report a file as missing when the source is
        held on a container tmpfs. Probe histories deliberately use such a
        tmpfs, so collect them through a non-TTY root exec while the trusted
        sidecar is still running. The fixed shell program receives the path as
        an argument (not interpolated shell text), rejects symlinks and
        directories, and streams stdout directly to an atomic host-side file.
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                await self._env._run_docker_compose_command(
                    [
                        "exec",
                        "-T",
                        "--user",
                        "0",
                        service,
                        "sh",
                        "-c",
                        _DOWNLOAD_REGULAR_FILE_SCRIPT,
                        "harbor-service-artifact-download",
                        source_path,
                    ],
                    check=True,
                    timeout_sec=120,
                    stdout_file=output,
                )
            os.replace(temporary, target)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)

    async def download_running_service_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str,
    ) -> None:
        """Collect live sidecar directories without losing tmpfs contents.

        Validate in a private sibling staging directory, then atomically publish
        into an absent/empty destination. On any failure, retain the destination
        unchanged. Do not apply archive ownership, permissions or links on host.
        """
        target = Path(target_dir)
        if any(parent.is_symlink() for parent in (target, *target.parents)):
            raise ValueError("sidecar artifact destination contains a symlink")
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError("sidecar artifact destination must be absent or empty")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.", dir=target.parent
        ) as temporary:
            staging = Path(temporary)
            archive_path = staging / "download.tar"
            with archive_path.open("xb") as output:
                archive_path.chmod(0o600)
                await self._env._run_docker_compose_command(
                    [
                        "exec",
                        "-T",
                        "--user",
                        "0",
                        service,
                        "bash",
                        "-c",
                        _DOWNLOAD_DIRECTORY_SCRIPT,
                        "harbor-service-directory-download",
                        source_dir,
                        str(_MAX_SIDECAR_ARCHIVE_BYTES + 1),
                    ],
                    check=True,
                    timeout_sec=120,
                    stdout_file=output,
                )
            tree = staging / "contents"
            tree.mkdir(mode=0o700)
            _extract_sidecar_directory(archive_path, tree)
            os.replace(tree, target)
