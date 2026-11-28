# Pilot validation — 0.1.0-pilot.1

The anisotropic SWG beamsplitter images are published and the release tooling
works. **The pilot is not scientifically certified:** runtime simulator
credentials are unavailable, so the current-profile reference final has not run.

## Executed checks

| Check | Result |
|---|---|
| Source preservation | The original 31 task files were compared against their recorded hashes. Only naming/distribution configuration, the build launcher and README differ; scientific code, profiles, instructions, reference hook and budgets are unchanged. |
| Host tests | 51 passed; six host simulator-dependent tests skipped. |
| Docker builds | Both existing Dockerfiles build as `linux/amd64`, locally and on native AMD64 CI. |
| Image audit | 53,249 file entries across all layers inspected; credential-pattern, forbidden-path, platform/revision and offline software checks pass. PhotonForge package discovery is offline; its licensed import is not claimed. |
| Registry | Both versioned images were pushed using the existing Actions secrets, resolved to registry digests, and independently pulled. A fresh local classic image store pulled both public images without building. |
| Release copy | Learner, probe sidecar and separate verifier use immutable digest references. No Compose build remains; Harbor resources and artifact contracts are unchanged. |
| Harbor lifecycle | Learner and evaluator start; health checks pass; the native probe returns a source-bound envelope; final source is independently collected and graded in a fresh verifier. |
| Final/probe separation | Probe and final contain different source bytes. The final artifact has the final source hash, `mode=final` and no probe ID. |
| Scientific reference | Blocked before starting a provider job: neither runtime simulator credential is configured. No paid FDTD job was submitted. |

The credential-free invalid-source case demonstrates lifecycle and provenance,
not successful physical validation. Its final rejection occurs during PhotonForge
licence setup, before the deliberate invalid source executes. Reference validity,
objective agreement and scientific invalid-source rejection remain unverified on
the published-image path. Existing offline evaluator tests pass; older-profile
reference evidence is not substituted for a current simulation.

## Published images

Repository: `docker.io/physicsautomation/physdesignbench`.

| Role | Version tag | Immutable digest |
|---|---|---|
| Learner | `anisotropic-swg-beamsplitter-v0-learner-v0.1.0-pilot.1` | `sha256:5636798a97b08b335d3e606e3c5de051d71d7ab69ceafdfc79b9607771308264` |
| Evaluator | `anisotropic-swg-beamsplitter-v0-evaluator-v0.1.0-pilot.1` | `sha256:bf35b3e02b80b6f639a561865230baae369d005ebaa8866dec5b3b947dca8c66` |

Image source commit: `8c3afbefe4bbfaf2c11a862a7fb0185a16524777`.
Later commits improve host release/validation tooling; the task inputs remain
identical. The [versioned manifest](../../releases/0.1.0-pilot.1/image-manifest.json)
records full image bindings, input hashes and inspection results.

The [manual publication run](https://github.com/Physics-Automation/physdesignbench/actions/runs/46145929398)
built, inspected, pushed, pulled and assembled both images successfully, but its
last artifact-upload step failed because hidden directories were excluded.
`include-hidden-files` is now enabled, and the
[corrected CI run](https://github.com/Physics-Automation/physdesignbench/actions/runs/46146395214)
passed and retained its build manifests. The released manifest was recovered
through the reusable `resolve` command after pulling and re-inspecting the actual
published images; no version tag was overwritten. Local Docker 29.5's incomplete
containerd exports were handled by using the dedicated VM's classic image store,
without changing any task container restrictions.

## Reproduce without local image builds

From this PR branch after `uv sync --locked --extra dev`:

```sh
uv run python -m scripts.release pin \
  --manifest releases/0.1.0-pilot.1/image-manifest.json \
  --destination .release/package-0.1.0-pilot.1
uv run python -m scripts.release validate \
  --manifest releases/0.1.0-pilot.1/image-manifest.json \
  --package .release/package-0.1.0-pilot.1 --harbor
```

On a compatible Linux AMD64 host, inject a simulator credential at runtime
through a private environment file and perform the remaining minimal check:

```sh
uv run --env-file /path/to/private.env python -m scripts.release validate \
  --manifest releases/0.1.0-pilot.1/image-manifest.json \
  --package .release/package-0.1.0-pilot.1 --scientific \
  --reference tests/tasks/anisotropic-swg-beamsplitter-v0/fixtures/paper_candidate_design.py \
  --output .release/scientific-validation.json
```

This first confirms actual execution and rejection of the deliberately invalid
source, then performs one independent reference final using the unchanged task
budget and guards. It must pass before completing scientific certification.
No simulator credentials should be committed or included in image builds.
