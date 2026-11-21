# Contributing to PhysDesignBench

PhysDesignBench welcomes contributions from researchers and engineers working in photonics, electromagnetics, RF and microwave design, multiphysics, and other simulation-based engineering fields.

Please propose a task **before** investing time in a full implementation. Maintainers and domain experts can help review the scope and evaluation plan.

## Ways to contribute

- **Propose a task.** [Open a task proposal](https://github.com/Physics-Automation/physdesignbench/issues/new/choose) describing the physical system, design variables, objective, validity constraints, and simulation tools.
- **Review a task.** Check physical assumptions, constraints, evaluator behavior, and reference designs.
- **Evaluate an agent.** Follow the [Run guide](https://physdesignbench.com/run), and coordinate with maintainers before proposing results for official rankings.
- **Report a problem.** Open an issue about faulty evaluators, unclear specifications, or reproducibility.

## Task contribution process

1. **Propose.** Open an issue with the problem, objective, constraints, simulator, and available reference design. Maintainers assess relevance, evaluability, runtime, and overlap.
2. **Agree on scope.** Discuss requirements, evaluation boundaries, performance targets, and budget before writing the task.
3. **Implement.** Use an existing [PICBench task](https://github.com/MomeniAli/PICBench/tree/main/tasks) as a reference while this repository is being populated. Follow the [task format](../task-format.md).
4. **Test.** Verify the reference design, validity checks, and scoring. Run at least one end-to-end agent attempt and record the commands, software versions, and results.
5. **Review.** Open a pull request and fill in the checklist. Maintainers and domain experts review physical correctness, evaluator isolation, and reproducibility.

See the [task guidelines](../task-guidelines.md) for acceptance criteria and [review guidelines](../review-guidelines.md) for the detailed checklist.

## Task acceptance and benchmark releases

Merging a contribution into the repository does not automatically include it in a benchmark release or leaderboard. Release membership and metric definitions are reviewed separately.

## Repository transition

The original benchmark tasks currently remain in [MomeniAli/PICBench](https://github.com/MomeniAli/PICBench). They have not been moved into this repository yet. Task migration will be handled separately.

## Questions

If you have a physics design problem but are unsure how to evaluate it automatically, [open a proposal](https://github.com/Physics-Automation/physdesignbench/issues/new/choose) describing what you can currently simulate and verify.
