#!/usr/bin/env bash
set -euo pipefail

cat > /workspace/README_solution_note.md <<'EOF'
This is an open-ended implementation/optimization benchmark. No private oracle
geometry or score is installed by the solution hook.

Implement build_submission() in /workspace/design.py. It must return one
complete PhotonForge Component containing silicon geometry on GDS layer (1, 0)
and residual bottom-void footprints on (2, 0). Start from the published Halir
anisotropic-SWG MMI, use
run_probe for evaluator-owned feedback, and leave the best complete component
as the sole final submission.
EOF
