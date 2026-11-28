# PhysDesignBench image pilot 0.1.0-pilot.1

Only the anisotropic SWG beamsplitter Reimplementation task is released here.
Both images are published. Harbor lifecycle and source/provenance checks pass;
scientific certification is blocked by missing runtime simulator credentials.

- `image-manifest.json`: immutable image bindings, source commit and layer checks.
- `harbor-validation.json`: executed lifecycle validation, including its limits.
- `scientific-validation.json`: blocked provider preflight; no paid job started.
- `evidence/`: exact probe feedback, independent final artifact and provenance.

See [the validation report](../../docs/releases/pilot-validation.md) for commands
and interpretation. Generate the digest-pinned task package from the manifest
using `python -m scripts.release pin`; development configurations are unchanged.
