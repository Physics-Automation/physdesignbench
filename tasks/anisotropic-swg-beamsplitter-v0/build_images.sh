#!/bin/sh
# Compatibility entry point; release logic lives in the shared Python tooling.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$task_root/../.." && pwd)
cd "$repo_root"
exec uv run python -m scripts.release build --task "$(basename "$task_root")" "$@"
