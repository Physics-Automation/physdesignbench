# Task review guidelines

Use the following questions during issue triage and pull-request review.

## Proposal review

- Is the problem a credible physics or engineering design problem?
- Are design variables, operating conditions, measurable objectives, and constraints stated?
- Can a submitted design be evaluated independently?
- Does the proposer have a reproducible reference design or a viable plan for creating one?
- Is the simulation environment realistically accessible and runnable?
- Does the problem add meaningful coverage to the existing task set?

A proposal may be revised before implementation. A proposal discussion is not an acceptance decision.

## Implementation review

- **Physical assumptions:** Are governing models, units, materials, boundary conditions, and constraints physically defensible?
- **Validity:** Are there tests for malformed outputs and common physically invalid cases?
- **Objective:** Do numerical outputs correspond to the intended engineering metric, including correct optimization direction?
- **Reference:** Does the reference pass all validity checks? Can maintainers independently reproduce its score?
- **Isolation:** Does the agent lack access to hidden evaluator details and privileged reference outputs?
- **Reproducibility:** Are versions, dependencies, licenses, seeds, execution commands, and tolerances documented?
- **Runtime:** Do the evaluator and reference run within the declared resource and probe budgets?
- **Evidence:** Is at least one full agent run documented, with artifacts and failures where relevant?

Where possible, obtain a domain-expert review separate from the implementation author.

## Release policy

A merged task is available in the repository but is **not automatically** included in an official benchmark release, a published task count, or the public leaderboard. Changes to a released evaluator or scoring policy require explicit versioning and a documented rationale.
