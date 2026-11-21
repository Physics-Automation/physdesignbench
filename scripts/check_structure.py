#!/usr/bin/env python3
"""Lightweight structural checks, not physics/evaluator validation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
required_docs = [
    "CONTRIBUTING.md",
    "docs/task-guidelines.md",
    "docs/task-format.md",
    "docs/review-guidelines.md",
    ".github/ISSUE_TEMPLATE/task-proposal.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
]
errors = []
for path in required_docs:
    if not (ROOT / path).is_file():
        errors.append(f"Missing contributor file: {path}")

tasks = ROOT / "tasks"
if tasks.exists():
    for task in sorted(tasks.iterdir()):
        if not task.is_dir() or task.name.startswith("."):
            continue
        for name in ("instruction.md", "task.toml", "environment", "solution", "tests"):
            if not (task / name).exists():
                errors.append(f"{task.relative_to(ROOT)}: missing {name}")

if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print("Contribution structure checks passed (not a physics validation).")
