#!/bin/sh
set -eu
if ! printf '%s\n' "${PICBENCH_EVALUATOR_IMAGE_DIGEST:-}" \
    | grep -Eq '^sha256:[0-9a-f]{64}$'; then
    echo "missing or invalid PICBENCH_EVALUATOR_IMAGE_DIGEST" >&2
    exit 1
fi

exec /opt/picbench/evaluator/bin/evaluate /submission/design.py
