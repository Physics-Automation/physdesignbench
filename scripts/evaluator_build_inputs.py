#!/usr/bin/env python3
"""Hash the effective local inputs of a task's evaluator image build."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import stat
from pathlib import Path

EVALUATOR_BUILD_INPUTS_LABEL = "org.picbench.evaluator-build-inputs-sha256"
LEARNER_BUILD_INPUTS_LABEL = "org.picbench.learner-build-inputs-sha256"
_DIGEST_DOMAIN = b"picbench.evaluator-build-inputs.v1\0"
_LEARNER_DIGEST_DOMAIN = b"picbench.learner-build-inputs.v1\0"
_STANDARD_EXCLUSIONS = frozenset({"**/__pycache__/", "**/*.py[cod]", "**/.pytest_cache/", "**/.ruff_cache/"})


def _dockerfile_lines(dockerfile: Path) -> list[tuple[int, str]]:
    logical_lines: list[tuple[int, str]] = []
    pending = ""
    pending_line = 0
    for line_number, raw_line in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if not pending:
            pending_line = line_number
        pending += stripped
        if pending.endswith("\\"):
            pending = pending[:-1] + " "
            continue
        logical_lines.append((pending_line, pending))
        pending = ""
    if pending:
        logical_lines.append((pending_line, pending))
    return logical_lines


def _copy_sources(context_dir: Path, dockerfile: Path) -> set[Path]:
    sources: set[Path] = set()
    for line_number, instruction in _dockerfile_lines(dockerfile):
        fields = shlex.split(instruction)
        if not fields:
            continue
        directive = fields[0].upper()
        if directive == "ADD":
            raise ValueError(f"unsupported ADD instruction at {dockerfile}:{line_number}")
        if directive != "COPY":
            continue
        if len(fields) != 3 or fields[1].startswith("--"):
            raise ValueError(f"unsupported COPY instruction at {dockerfile}:{line_number}")
        source_relative = Path(fields[1])
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError(f"COPY source escapes task directory at {dockerfile}:{line_number}")
        source = context_dir / source_relative
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(f"missing COPY source at {dockerfile}:{line_number}: {fields[1]}")
        sources.add(source)
    return sources


def _dockerignore_allowlist(context_dir: Path) -> set[Path]:
    dockerignore = context_dir / ".dockerignore"
    rules = [
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rules or rules[0] != "*":
        raise ValueError(f"{dockerignore} must use a deny-all allowlist")

    included: set[Path] = set()
    for rule in rules[1:]:
        if rule in _STANDARD_EXCLUSIONS:
            continue
        if not rule.startswith("!"):
            raise ValueError(f"unsupported .dockerignore rule in {dockerignore}: {rule}")
        relative_text = rule[1:].rstrip("/")
        if not relative_text or any(character in relative_text for character in "*?["):
            raise ValueError(f"unsupported .dockerignore allowlist rule in {dockerignore}: {rule}")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"allowlist path escapes build context in {dockerignore}: {rule}")
        included.add(relative)
    return included


def _input_paths(task_dir: Path) -> list[Path]:
    copy_sources = _copy_sources(task_dir, task_dir / "evaluator/Dockerfile")
    paths: set[Path] = {
        task_dir / "build_images.sh",
        task_dir / "evaluator/Dockerfile",
        task_dir / ".dockerignore",
    }
    allowlist = _dockerignore_allowlist(task_dir)
    for root in copy_sources:
        if not root.exists() and not root.is_symlink():
            continue
        candidates = {root}
        if root.is_dir() and not root.is_symlink():
            candidates.update(root.rglob("*"))
        paths.update(path for path in candidates if path.relative_to(task_dir) in allowlist)
    missing_copy_roots = sorted(
        root.relative_to(task_dir).as_posix() for root in copy_sources if root.relative_to(task_dir) not in allowlist
    )
    if missing_copy_roots:
        raise ValueError(
            "Dockerfile COPY source is absent from .dockerignore allowlist: " + ", ".join(missing_copy_roots)
        )
    return sorted(paths, key=lambda path: path.relative_to(task_dir).as_posix())


def _learner_input_paths(task_dir: Path) -> list[Path]:
    context_dir = task_dir / "environment"
    dockerfile = context_dir / "Dockerfile"
    copy_sources = _copy_sources(context_dir, dockerfile)
    paths: set[Path] = {
        task_dir / "build_images.sh",
        dockerfile,
        context_dir / ".dockerignore",
    }
    allowlist = _dockerignore_allowlist(context_dir)
    for root in copy_sources:
        if not root.exists() and not root.is_symlink():
            continue
        candidates = {root}
        if root.is_dir() and not root.is_symlink():
            candidates.update(root.rglob("*"))
        paths.update(
            path
            for path in candidates
            if path.relative_to(context_dir) in allowlist
        )
    missing_copy_roots = sorted(
        root.relative_to(context_dir).as_posix()
        for root in copy_sources
        if root.relative_to(context_dir) not in allowlist
    )
    if missing_copy_roots:
        raise ValueError(
            "Dockerfile COPY source is absent from .dockerignore allowlist: "
            + ", ".join(missing_copy_roots)
        )
    return sorted(paths, key=lambda path: path.relative_to(task_dir).as_posix())


def _hash_input_paths(task_dir: Path, paths: list[Path], domain: bytes) -> str:
    digest = hashlib.sha256(domain)
    for path in paths:
        relative = path.relative_to(task_dir).as_posix().encode("utf-8")
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            kind = b"symlink"
            payload = os.readlink(path).encode("utf-8")
        elif stat.S_ISDIR(metadata.st_mode):
            kind = b"directory"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"file"
            payload = path.read_bytes()
        else:
            raise ValueError(f"unsupported build input type: {path}")
        for field in (kind, relative, f"{mode:o}".encode("ascii"), payload):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def evaluator_build_inputs_sha256(task_dir: Path) -> str:
    """Return a stable digest over evaluator image build inputs."""

    task_dir = task_dir.resolve()
    return _hash_input_paths(task_dir, _input_paths(task_dir), _DIGEST_DOMAIN)


def learner_build_inputs_sha256(task_dir: Path) -> str:
    """Return a stable digest over learner-image build inputs."""

    task_dir = task_dir.resolve()
    return _hash_input_paths(
        task_dir,
        _learner_input_paths(task_dir),
        _LEARNER_DIGEST_DOMAIN,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument(
        "--image",
        choices=("evaluator", "learner"),
        default="evaluator",
    )
    args = parser.parse_args(argv)
    digest = (
        learner_build_inputs_sha256(args.task_dir)
        if args.image == "learner"
        else evaluator_build_inputs_sha256(args.task_dir)
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
