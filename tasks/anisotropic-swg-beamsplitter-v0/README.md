# Anisotropic SWG 2 x 2 Beamsplitter Implementation/Optimization

This task asks the learner to implement the Halir-style anisotropic SWG 2 x 2
MMI beamsplitter from the packaged paper as one PhotonForge component,
reproduce the paper geometry during development, and improve its broadband
splitting response.

The submitted `design.py` exposes `build_submission()` and returns one final
component containing the silicon and residual bottom-void masks. A restricted
private builder exports canonical GDS; the evaluator owns the fixed optical
stack, excitation, monitors, cloud run, and metric. Both submitted masks must
be strictly invariant under reflection about `y=0`. The evaluator excites the
`y=+1 um` input once; Maxwell symmetry then gives the `y=-1 um` response with
the two output labels exchanged, so this one subrun covers both physical inputs.

The lower-is-better objective is the worst sampled normalized penalty against
the paper's `1 dB` excess-loss, `1 dB` imbalance, and `5 deg` quadrature-error
envelope over `1250–1750 nm`. The paper reference is `1.0`; it is comparison
metadata and is not rerun for each candidate.

Learners receive the paper, `references/paper_parameters.json`,
`task_profile.json`, `harness_contract.md`, and the backend guide. `run_probe`
provides bounded diagnostic feedback. Learner metrics, cloud evidence, and
artifacts are not score authority; final verification evaluates the submitted
source afresh.

## Published-image pilot

This task is the Reimplementation worked example selected for the first
PhysDesignBench distribution pilot. See [release instructions](../../docs/releasing.md)
and the [image/isolation audit](../../docs/releases/pilot-audit.md).

There are separate learner and evaluator images. Public evaluator source and
physical profiles are inspectable; evaluation integrity depends on independent
execution and runtime-only credentials, not hidden image contents. Reference
fixtures are not included in either image. Source development configuration is
retained; release copies are generated with registry digests pinned throughout.
