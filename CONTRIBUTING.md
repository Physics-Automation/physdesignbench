# Contributing to PhysDesignBench

PhysDesignBench evaluates AI agents on physical engineering design tasks. We welcome contributions from domain experts in photonics, electromagnetics, RF and microwave engineering, multiphysics, and other simulation-based engineering fields.

A new task should begin with a proposal, **before** the contributor spends time implementing it. Maintainers and domain experts review both the physical problem and its evaluator.

## Contribution paths

- **Propose a task:** [Open a task proposal](https://github.com/Physics-Automation/physdesignbench/issues/new/choose). Explain the physical system, design variables, objective, validity constraints, and evaluation software.
- **Review a task:** Check physical assumptions, reference solutions, constraints, and evaluator behavior.
- **Report a problem:** Open an issue for faulty evaluators, confusing instructions, or reproducibility failures.
- **Evaluate an agent:** Follow the [Run guide on the website](https://physdesignbench.com/run). Coordinate with maintainers before asking for results to be included in an official ranking.

## From proposal to accepted task

1. **Discuss the proposal.** Use the issue template. The maintainers assess scientific relevance, evaluability, runtime, and overlap with existing tasks.
2. **Implement the task.** Start from an existing [PICBench task](https://github.com/MomeniAli/PICBench/tree/main/tasks) for reference while the new repository is being populated. Follow the [task format](docs/task-format.md).
3. **Test it.** Verify the reference design, run validity and scoring checks, and execute at least one end-to-end agent attempt. Document the commands and outcomes.
4. **Open a pull request.** Fill in the PR checklist and attach reproducibility evidence. Maintainers and a domain expert review the evaluator and physical assumptions.
5. **Revise and merge.** Merging a task does **not** automatically add it to an official benchmark release or leaderboard. Benchmark membership and scoring policy are decided separately.

Read the [task guidelines](docs/task-guidelines.md) and [review guidelines](docs/review-guidelines.md) before submitting.

## Questions

If you have a promising design problem but are unsure how to make its evaluation reproducible, open a proposal anyway. Describe what is currently measurable and where evaluation needs work.

## A note on the repository transition

The original task implementations are presently maintained in [MomeniAli/PICBench](https://github.com/MomeniAli/PICBench). They have **not** yet been moved here. This repository initially hosts the contribution process; task migration will be handled separately.
