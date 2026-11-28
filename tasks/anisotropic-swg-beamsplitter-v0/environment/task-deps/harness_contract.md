# Implementation harness contract

## Submission

Work in `/workspace` and edit only `design.py`. It must define
`build_submission()` and return one PhotonForge component matching the exact
interface and geometry contract in `task_profile.json`.

You may run the source locally for layout debugging. Learner-created GDS,
metrics, JSON, cloud task IDs, downloaded results, HDF5, S-parameters, and
plots are diagnostic only and never final-score authority.

## Probe

```bash
run_probe
```

The unprivileged fixed-endpoint client uploads only the current bounded source
to the private evaluator sidecar. The sidecar freezes and hashes it, builds
fresh canonical GDS in the restricted non-root builder, validates the public
contract, runs the canonical evaluator, and writes and prints a unique
`/workspace/probe_results_N.json`.

The trial allows up to 30 probes and 100 FlexCredits in total. Each evaluation
is capped at 4 FC. The private broker protects 4 FC for fresh final verification,
leaving at most 96 FC for probes; 30 paid probes are not guaranteed. Failed or
uncertain cloud starts retain conservative cost reservations. There are no
automatic paid retries.

Probe state and budget remain private evaluator state. A probe never changes,
replaces, or supplies evidence to final verification.

## Probe result envelope

Each accepted probe saves and prints the same version-1 envelope, with integer
`schema_version` `1` and a benchmark-owned monotonic accepted `probe_id`:

- `submission_sha256` is the frozen-source digest when available; `status` is
  normally `completed` for accepted probes; `valid` and `gate_pass` are task
  booleans when defined; trusted diagnostic `score` and `metric` may be `null`.
- `probes_remaining` is the remaining accepted-probe count or `null`; `budget`
  holds measurement and paid-probe counts, estimates, caps, reserve, and
  `budget_authority` = `private_evaluator_sidecar`.
- `diagnostic_artifacts` is bounded metadata for installed learner-visible
  files; `best_valid_snapshot` is checkpoint metadata or `null`;
  `task_feedback` contains bounded task-specific diagnostics.

`task_feedback` is the only task-specific probe namespace; task-specific
resource fields stay under `budget`. Artifacts and snapshots have
`score_authority: false` and never supply final evidence. The snapshot remains
`null` unless the task explicitly maintains learner-visible checkpoint files.

For this task, `task_feedback` includes a bounded beamsplitter summary, public
gates, coarse failure stage and codes, and fixed next steps. It excludes raw
errors, tracebacks, private paths, provider identifiers, and hidden thresholds.

Probe results are feedback only. Final verification freezes the final
submission in fresh evaluator state and never scores a cached probe result.

## Benchmark status

Use `benchmark_status` to inspect benchmark-owned elapsed and remaining
agent-phase time. Missing or untrusted timing returns
`benchmark_status_unavailable` with nullable values. The clock is advisory,
independent of probe budgets and scoring; final verification remains
authoritative.

## Final verification

Harbor invokes a root-owned transport helper that safely freezes only
`/workspace/design.py` into `/submission/design.py`. A fresh verifier
container from the private evaluator image:

1. verifies the frozen source identity;
2. executes learner code only in the restricted non-root builder;
3. promotes only canonical `final_pic.gds`;
4. validates the silicon, residual-void, fixed-port, and strict `y=0` mirror
   contracts;
5. constructs and runs the evaluator-owned Tidy3D simulation for the
   `y=+1 um` input; strict mirror symmetry derives the second physical input
   response with output labels exchanged;
6. computes incident/reflected and two output TE0 spectra from 1250 to 1750 nm
   on a 10 nm grid
   and the worst normalized loss/balance/phase envelope penalty;
7. publishes a manifest-bound result and viewer artifacts.

Probe geometry and simulation results are never reused by the final verifier.

## Result semantics

- `design_valid`: canonical GDS passed the public geometry contract.
- `simulation_valid`: evaluator-owned FDTD completed with physical outputs.
- `evidence_valid`: the objective came from that evaluator-owned run.
- `input_coverage_pass`: both physical inputs are covered: one by the trusted
  subrun and one by the evaluator-validated mirror relation.
- `task_metric`, `official_score`, `official_metric`, and `objective_value`: the
  same worst sampled normalized splitter-envelope penalty `J`; lower is better.
- `reference_baseline`: `1.0`, meaning all sampled wavelengths satisfy the
  paper's 1 dB loss, 1 dB imbalance, and 5 degree phase-error envelope.
- `paper_margin`: published-envelope reference minus candidate `J`, retained as
  a diagnostic only.
- `reward`: Harbor compatibility adapter `1/(1+J)` for an eligible result and
  zero otherwise; it is not the official score.
