# Task: Implement and Optimize the Anisotropic SWG 2 x 2 Beamsplitter

Implement the Halir-style anisotropic sub-wavelength-grating 2 x 2 MMI
beamsplitter from the provided paper as one final layout component. First
reproduce the paper geometry and use trusted probes to debug it; then optimize
the permitted geometry over `1250–1750 nm`.

## Objective

At each sampled wavelength, compute the normalized envelope penalty

```text
max(max(excess_loss_dB, 0) / 1 dB,
    imbalance_dB / 1 dB,
    quadrature_phase_error_deg / 5 deg)
```

The official score `J` is `max_wavelength` of this penalty over the 10-nm
wavelength grid. Lower is better. The evaluator simulates the `y=+1 um` input.
Eligibility requires exact `y=0` mirror symmetry of both submitted mask unions;
therefore the `y=-1 um` input has the same `J` with the two output labels
exchanged. A geometrically asymmetric candidate is invalid rather than being
scored from only one input.

## Paper baseline

The paper reports an operating envelope of at most `1 dB` excess loss, `1 dB`
imbalance, and `5 deg` quadrature phase error across `1250–1750 nm`. The
normalized reference is therefore `1.0`. It is not a hidden design target, a
final eligibility gate, or a baseline evaluation repeated for every candidate.
Only the submitted candidate is evaluated.

## Required files

- `references/paper.pdf`: source paper.
- `references/paper_parameters.json`: extracted geometry and paper limits.
- `task_profile.json`: authoritative submission, layout, objective, and
  evaluation contract.
- `harness_contract.md`: source-freezing, probe, and final-verifier workflow.

## Submission

Write `/workspace/design.py` with:

```python
def build_submission():
    return final_component
```

Return one final layout component containing the complete silicon core mask on
layer `(1, 0)` and residual bottom-void footprints on `(2, 0)`. The device
starts at `x=0`, propagates along `+x`, and uses micrometers.

The evaluator owns the material stack, layer extrusions, fixed access
extensions, source, monitors, mesh, boundaries, normalization, and official
metric. Do not add any other layers or submit computed metrics or simulation
artifacts.

## Scientific workflow

1. Reconstruct the paper starting point: 190-nm period, 50% duty cycle,
   30-period inverse-SWG tapers, and the 74-period, 3.25-um-wide MMI.
2. Use trusted probes until the geometry is valid and the broadband splitter
   response is physically plausible.
3. Optimize the bridge profile, SWG bar-width profile, bounded MMI period count
   and width, and residual-void footprints.
4. Leave `build_submission()` returning only the best final component.

## Geometry rules

- Use only layers `(1, 0)` and `(2, 0)` with the meanings in
  `task_profile.json`.
- Keep the device within the public coordinate, length, area, polygon, feature,
  and vertex bounds.
- Preserve the two 0.50-um-wide input/output landings centered at
  `y=-1.0 um` and `y=+1.0 um`.
- Keep the complete union on each mask layer strictly invariant under
  reflection about `y=0`; this is the scientific basis for deriving the second
  physical input from the evaluator's one simulated input.
- Keep every polygon positive-area and maintain the required masks and
  connectivity.
- Submit one component, not separate baseline and optimized components.

## Probe

Use `run_probe` for bounded evaluator-owned feedback; see
`harness_contract.md`. Task-specific scientific guidance appears under
`task_feedback`. Probe results are diagnostic, and final verification evaluates
the final source afresh.

## Execution policy

Harbor provides a fixed wall-clock working period. Continue actively working
until Harbor externally terminates the agent at that deadline. Do not emit a
final response early, even if you believe the design is complete or cannot
immediately resolve a problem.

Use `benchmark_status` to inspect benchmark-owned wall-clock status. It
reports elapsed and remaining time when runner timing metadata is provided;
final verification remains authoritative.

If the primary implementation appears complete, continue with independent
audits, layout optimization, geometry inspection and optimization, parameter
optimization, code simplification, and robustness checks. Save the best
current submission file frequently because Harbor may terminate the session at
the deadline.
