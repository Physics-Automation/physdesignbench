# Single-task Docker and Harbor pilot

## Verified paper example

The manuscript's active worked examples select WDM-v1 for Design and the
**anisotropic SWG beamsplitter** for Reimplementation. The latter is the
Gemini 4 Argon / mini-SWE-agent example, titled “Reimplementation: SWG
beamsplitter”, and its instructions identify the Halir-style anisotropic
sub-wavelength-grating 2 × 2 MMI device. This maps to
`tasks/anisotropic-swg-beamsplitter-v0/`; `metadata.task_mode` is
`implementation_optimization`. The six-channel WDM task is not this example.

Identification was checked against the active manuscript inputs
`working_examples/index.tex`, `working_examples/swg_gemini.tex`, and the task
metadata and instruction text. The inactive SRSN worked example is commented out.
Task file provenance and the source implementation revision are recorded in
[pilot-source.json](pilot-source.json).

## Image and service inventory

| Role | Dockerfile | Build context | Harbor use |
|---|---|---|---|
| Learner | `environment/Dockerfile` | `environment/` | `[environment]`, Compose `main` |
| Evaluator | `evaluator/Dockerfile` | task directory | Compose `evaluator` for probes and a fresh `[verifier.environment]` for final grading |

Two images are sufficient. The evaluator image is reused for two different
lifecycle instances; it is not two different evaluator images. Learner and
evaluator containers are not combined. The platform is `linux/amd64`, matching
the original Linux sandbox/seccomp contract.

The learner constructs a PhotonForge component. A restricted nested-user-
namespace builder exports canonical GDS; the evaluator builds one Tidy3D FDTD
excitation under the fixed y-mirror symmetry contract. It checks silicon and
bottom-void masks and evaluates the worst normalized excess-loss, imbalance,
and quadrature-error penalty across 1250–1750 nm. The task's raw objective,
reference value, simulator settings and numerical gates are unchanged.

## Runtime contract

- Agent: 7,200 seconds, user `agent`, 4 CPUs, 8,192 MiB, public network.
- Probe sidecar: 4 CPUs, 8,192 MiB; read-only root; bounded writable tmpfs;
  all capabilities dropped except the task's existing six; no-new-privileges;
  original seccomp policy and PID limit.
- Final verifier: separate fresh container; 1,800-second timeout. Source is
  exported through the existing root-only collection command before grading.
- Probe budget: 30, with at most 30 paid learner probes. The 100-FlexCredit
  guard includes the existing protected final reserve and 4-credit per-evaluation
  guard. No release script changes these limits.
- Real `TIDY3D_API_KEY` / `SIMCLOUD_APIKEY` values are supplied at runtime only
  to trusted evaluator environments. The learner's fixed proxy token is a dummy
  local protocol value, not a provider credential.
- Ports 8787 (layout licence proxy) and 8788 (probe API) remain internal services.
  Health checks and collection of probe history, state and usage receipts remain.

## Public-image review

The evaluator image intentionally contains the evaluator code and fixed physical
profile. These are inspectable in a public registry and must not be described as
secret. The integrity boundary is independent execution of unchanged trusted
code, root-owned state, restricted submission building, frozen-source binding,
and provider credentials injected at runtime—not obscurity of image contents.

The learner Dockerfile uses its existing deny-all context allowlist and never
copies the evaluator or solution. The evaluator uses a separate deny-all context
allowlist and does not copy the solution or author fixture. Neither image
copies `.env`, runtime authorization receipts, host credentials, or provider
keys. The reference fixture is retained only in host-side validation tests.
Layer inspection checks every saved image layer, including files deleted by
later layers, and reports only paths if it finds a credential shape or configured
secret. Images cannot be published through the release CLI before these checks
and offline dependency imports have passed.

## Existing evidence and certification limits

Historical author evidence records a valid raw objective `J = 0.666180569542655`
under the earlier v2 physical profile. The current profile is
`v4-y-mirror-input-coverage`; old certificates explicitly do not certify current
source bytes. The historical readiness report passed offline checks and image
bindings but did not grant release authority. The migrated tests retain this
distinction and recompute the historical objective without claiming a fresh
simulation.

The existing `solution/solve.sh` writes guidance and does not install an oracle
geometry. It is preserved byte for byte. For validation, the release scripts can
make a disposable copy with a standard Harbor oracle hook that installs the
existing author fixture. This is not a modification of the distributed solution,
a new agent framework, or a benchmark run. A provider-connected reference final
is still required before describing the pilot as scientifically certified.

## Shared runtime dependencies

The existing Harbor compatibility dependency is bundled under `vendor/harbor`
with its Apache-2.0 licence. The upstream base is
`b53b8134e1241686dca7759af188f987ecc48e8b`; the imported compatibility revision is
`3e6cf0b4ee76aa2574d81586e7a17220823b42bd`. This preserves separate-verifier
collection, image-template resolution, resource/security settings and process
lifecycle fixes. Only the small benchmark-clock helper required by that runtime
is included from the task runner integration. The project-specific viewer, historical run stores,
agent campaign runners and other 25 tasks are not migrated.

## Release infrastructure reference

The implementation follows Terminal-Bench's release-copy approach: leave source
Dockerfiles and development task configuration in place, build the required
surfaces, publish task/role/version tags to one repository, and generate a copied
task tree with digest pins for the learner, verifier and Compose services.
The reference inspected was commit `5411da4d8aed9e522b8ebdf440bfcb1ca5b25072`:

- [Release tooling](https://github.com/harbor-framework/terminal-bench/blob/5411da4d8aed9e522b8ebdf440bfcb1ca5b25072/scripts/release-build/release_prebuilt.py)
- [Release workflow](https://github.com/harbor-framework/terminal-bench/blob/5411da4d8aed9e522b8ebdf440bfcb1ca5b25072/.github/workflows/release-prebuilt.yml)

This pilot deliberately does not publish a Harbor Hub task collection or dataset.
A future versioned dataset can use `physics-automation/physdesignbench` after
individual tasks and dataset membership have been reviewed.

## Executed pilot checks

See [pilot-validation.md](pilot-validation.md) for the published image references,
executed checks, retained evidence and remaining scientific-validation blocker.
