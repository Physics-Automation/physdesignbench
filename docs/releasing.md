# Releasing task images

This pilot distributes **one Reimplementation task** using the existing Harbor
architecture. It does not publish a full benchmark dataset. See the
[pilot audit](releases/pilot-audit.md) for identification, images, trust boundaries
and scientific-validation limits.

## Install

Requires Python 3.12+, uv, Git, Docker with Buildx and a `linux/amd64` Linux
container runtime. The original restricted builder needs nested user namespaces;
validate this on the intended host rather than weakening its sandbox.

```sh
uv sync --locked --extra dev
```

The bundled Harbor runtime and compatibility helper make this checkout
self-contained. No original implementation checkout is required. Provider
credentials stay in your shell or an untracked env file, never build arguments.

## Build and inspect

```sh
uv run python -m scripts.release inventory \
  --task anisotropic-swg-beamsplitter-v0 --version 0.1.0-pilot.1 \
  --manifest .release/plan.json

# Print the plan without building, logging in, or publishing.
uv run python -m scripts.release build \
  --task anisotropic-swg-beamsplitter-v0 --version 0.1.0-pilot.1 --dry-run

# Real release builds require committed, clean source.
uv run python -m scripts.release build \
  --task anisotropic-swg-beamsplitter-v0 --version 0.1.0-pilot.1 \
  --manifest .release/build.json
uv run python -m scripts.release check --manifest .release/build.json
```

`--allow-dirty` permits a local development build that cannot be published.
The build command never pushes. Buildx reuses its local layer cache. The checks
inspect all layers, forbidden learner paths, baked environment credentials,
platform/revision labels and offline dependency imports. Exact public AWS
documentation example IDs are exempted by content hash; matching key-header text
without encoded key material and unbounded binary substrings are not credentials.
The exact SHA256 of Debian libgnutls30t64 3.8.9-3+deb13u4 is recorded as a public
self-test-key fixture, verified against the official Debian package. Any changed
library bytes are scanned normally; configured runtime secret values are never
exempted. Runtime credentials are
not required for these checks. PhotonForge package discovery is checked offline;
its import performs online licence verification and is exercised only during
provider-connected validation.

## Publish explicitly

Use the manual **Task image release** GitHub Actions workflow. Select the task,
source ref and version. Leave `dry_run=true` for a build-only check; select
`dry_run=false` only to publish. It uses the existing `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` secrets. Pull requests build and inspect without those secrets
and can never trigger the publish step. No pull request or source change is
merged automatically.

The same operation can run locally after `docker login`; the CLI reads no Docker
Hub token itself:

```sh
uv run python -m scripts.release publish --manifest .release/build.json \
  --execute --output .release/image-manifest.json
```

Images use the single repository `docker.io/physicsautomation/physdesignbench`,
with tags `<task-id>-<image-role>-v<release-version>`. A version tag cannot be
overwritten by this publisher. Use a new explicit version for changed images.
A failed multi-image publication retains an **incomplete** report, not a complete
release manifest. No script claims a partially published release succeeded.

The successful manifest records each task, role, registry, repository, versioned
tag, immutable registry digest, source commit, Dockerfile, build context and
platform. It also binds checked local image IDs and exact task-input hashes.

## Recover a manifest after an upload failure

Existing version tags are never overwritten. If registry publication succeeded
but artifact upload failed, retain the original clean build plan, pull its
versioned images, and rerun `check` against those images. Then resolve every
registry digest and compare the pulled image bytes against the inspected IDs:

```sh
uv run python -m scripts.release check --manifest .release/build.json
uv run python -m scripts.release resolve --manifest .release/build.json \
  --output .release/image-manifest.json
```

This recovery command does not push. Missing images, changed task inputs,
revision-label mismatches, or mismatched image bytes prevent a complete manifest.

## Pin release copies and pull

```sh
uv run python -m scripts.release pin --manifest .release/image-manifest.json \
  --destination .release/package-0.1.0-pilot.1
uv run python -m scripts.release validate --manifest .release/image-manifest.json \
  --package .release/package-0.1.0-pilot.1 --output .release/validation.json
```

The source task is unchanged. Release copies use `repository@sha256:...` in
Harbor's learner/verifier fields and every configured Compose service, with no
Compose builds. Evaluator provenance and digest environment bindings are pinned
together; simulator credentials remain runtime substitutions. Mutable `latest`
references are never authoritative. The workflow uploads the manifest, validation
report and tarred Harbor task package as artifacts.

## Harbor and scientific verification

```sh
# One invalid probe followed by a different invalid final source; no paid solve.
uv run python -m scripts.release validate --manifest .release/image-manifest.json \
  --package .release/package-0.1.0-pilot.1 --harbor

# Explicitly requests one provider-connected final of the existing author fixture.
# Supply runtime credentials in the shell, or use uv run --env-file <private-file>.
uv run python -m scripts.release validate --manifest .release/image-manifest.json \
  --package .release/package-0.1.0-pilot.1 --scientific \
  --reference tests/tasks/anisotropic-swg-beamsplitter-v0/fixtures/paper_candidate_design.py
```

The latter may incur simulation charges, subject to the task's existing guards.
It is not run automatically by the image publisher. Each validation case uses
Harbor's standard oracle agent in a disposable task copy. The distributed
`solution/solve.sh` is preserved. Final evaluation must be bound to the final
source hash, not a probe result. To compare against a verified current objective,
add `--expected-j <value>`; do not substitute an old-profile certificate.

For an ordinary experiment using a released package:

```sh
uv run harbor run --path .release/package-0.1.0-pilot.1/tasks/anisotropic-swg-beamsplitter-v0 \
  --agent oracle --env docker --no-force-build --max-retries 0
```

The existing oracle hook is guidance-only, so that last command is a lifecycle
check, not a reference solution. Choose a supported agent/model for an actual
benchmark attempt. Scientific results always use independent final grading.

## Add the next task

1. Migrate that task and only its required shared dependencies.
2. Add `release.toml` declaring each image role's task-relative `context`,
   `dockerfile`, Harbor `config_paths`, Compose `compose_services`, optional
   `build_args`, runtime `smoke_commands`, and image-content `forbidden_paths`.
3. Declare the existing image/digest variable names if the task uses them.
   Existing compatibility labels can use `input_hash_kind` / `input_hash_label`.
4. Audit public distribution of evaluator contents and set
   `release.public_evaluator_reviewed=true` only after that review.
5. Repeat inventory, build, inspect, publish, pin and validation with the new ID.

Single-image tasks declare only the learner role. Tasks with separate verifiers
or additional services declare additional roles and bindings. Multiple
`--task` arguments or `--task all` select configured tasks. Unmanaged Compose
image/build surfaces, missing contexts and path escapes fail closed. Large
collections should dispatch tasks separately to stay within runner disk/time
limits. Platform expansion and Harbor Hub publication are future release steps,
not enabled by this pilot.
