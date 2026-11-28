# PhotonForge Component and Trusted Tidy3D Evaluation

The learner implements layout only. Use PhotonForge to construct one component
with the complete silicon core on `(1, 0)` and residual bottom-void footprints
on `(2, 0)`.

## Minimal interface

```python
import photonforge as pf


def build_submission():
    component = pf.Component("ANISOTROPIC_SWG_BEAMSPLITTER_FINAL")
    # Add the complete learner-authored masks on (1, 0) and (2, 0).
    return component
```

This scaffold shows only the interface. Derive and implement the paper
geometry yourself. PhotonForge dimensions are in micrometers; place the input
face at `x=0` and propagate along `+x`.

## Trusted simulation

The private evaluator imports canonical GDS and supplies all score-bearing
physics: the 220-nm SOI stack, mask extrusions, access extensions, fundamental
TE0 source and mode monitors, mesh, boundaries, normalization, cloud cost
guard, and broadband metric extraction. Both submitted mask unions must be
strictly invariant under reflection about `y=0`. One run excites the `y=+1 um`
input, samples `1250–1750 nm` every 10 nm, and evaluates excess loss, output
imbalance, and quadrature phase error. The reflected `y=-1 um` excitation has
the same objective with the two output labels exchanged.
The benchmark uses fixed, nondispersive material indices across this band;
it does not claim equivalence to the paper's dispersive electromagnetic model.

Do not call Tidy3D cloud APIs in the submission. Learner task IDs, result
files, HDF5, S-parameters, and reported metrics are ignored.

The trusted evaluator uploads the evaluator-constructed simulation to obtain a
provider estimate and blocks before `web.start` when the estimate is missing,
invalid, or above the configured ceiling.

## References

- PhotonForge: [https://docs.flexcompute.com/projects/photonforge/en/latest/](https://docs.flexcompute.com/projects/photonforge/en/latest/)
- Tidy3D: [https://docs.flexcompute.com/projects/tidy3d/en/latest/](https://docs.flexcompute.com/projects/tidy3d/en/latest/)

The images pin PhotonForge and Tidy3D. If online examples differ from the installed runtime, inspect the pinned API in the learner container.
