# Contributing to PhysDesignBench

PhysDesignBench welcomes contributions across computational physics and physics-based engineering, including but not limited to photonics, electromagnetics, RF and microwave engineering, mechanics, fluid dynamics, thermal physics, and multiphysics.

Please propose a task **before** investing time in a full implementation. Maintainers and domain experts can help review the scope and evaluation plan.

## Task contribution process

1. **Propose.** Open an issue with the problem, objective, constraints, simulator, and available reference design. Maintainers assess relevance, evaluability, runtime, and overlap.
2. **Agree on scope.** Discuss requirements, evaluation boundaries, performance targets, and budget before writing the task.
3. **Implement.** Follow the [task format](../task-format.md) and the agreed evaluation plan.
4. **Test.** Verify the reference design, validity checks, and scoring. Run at least one end-to-end agent attempt and record the commands, software versions, and results.
5. **Review.** Open a pull request and fill in the checklist. Maintainers and domain experts review physical correctness, evaluator isolation, and reproducibility.

See the [task guidelines](../task-guidelines.md) for acceptance criteria and [review guidelines](../review-guidelines.md) for the detailed checklist.

## Task acceptance and benchmark releases

Merging a contribution into the repository does not automatically include it in a benchmark release or leaderboard. Release membership and metric definitions are reviewed separately.

## Other ways to contribute

New tasks are the main contribution, but there are other ways to help:

- **Review a task.** Check physical assumptions, constraints, evaluator behavior, and reference designs.
- **Evaluate an agent.** Follow the [Run guide](https://physdesignbench.com/run), and coordinate with maintainers before proposing results for official rankings.
- **Report a problem.** Open an issue about faulty evaluators, unclear specifications, or reproducibility.

## Questions

If you have a physics design problem but are unsure how to evaluate it automatically, [open a proposal](https://github.com/Physics-Automation/physdesignbench/issues/new/choose) describing what you can currently simulate and verify.
