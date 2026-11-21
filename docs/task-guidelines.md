# Task guidelines

These guidelines describe the criteria used when evaluating new PhysDesignBench proposals. They are not a guarantee that a proposed task will be accepted.

## Engineering problem

- The task is grounded in a real physical design or reimplementation problem.
- A contributor identifies the design variables, objective, operating conditions, and physical and fabrication constraints.
- The problem requires meaningful design choices and simulation or analysis, rather than recalling a known answer.
- The task states software, dependencies, licensing restrictions, and approximate resource requirements.

## Independent and reproducible evaluation

- The evaluator operates on the submitted artifact independently of the agent's account of its performance.
- Validity checks are explicit and include physical and numerical feasibility, with appropriate tolerances.
- The objective is deterministic when practical; where solvers introduce variability, document tolerances and repeatability.
- Evaluator code, dependencies, seeds, and solver settings are documented sufficiently for maintainers to reproduce results.
- A contributor tests failure cases, including invalid geometry or artifacts, and checks that the evaluator rejects them appropriately.
- If a design uses a proprietary simulator, document licensing, version, hardware, and access limitations clearly.

## Reference solution and scoring

- Supply at least one reference design known to pass all validity checks.
- Record the reference objective and the procedure used to reproduce it.
- Specify whether each objective is minimized or maximized; define constraints and units.
- Define engineering performance targets separately from basic validity checks.
- Do not use an agent's own reported score in place of independent evaluation.

## Runtime and difficulty

- Specify the intended agent time limit and evaluator-probe limit in the proposal. **Do not assume the same budget for all task families.**
- A reference design must be runnable within the stated environment.
- Difficulty should arise from engineering choices, coupling, optimization, or verification, not obscure instructions or intentional evaluator loopholes.
- Where possible, include an initial design and a short explanation of why optimizing it is nontrivial.

## Submission evidence

A pull request should include the runnable task package, reference design, test commands and outcomes, and any licenses or third-party sources needed to reproduce the evaluation.

See the [task format](task-format.md) for package requirements.
