# Task format

PhysDesignBench tasks use a Harbor-compatible layout. Exact internal evaluator packaging can differ between task families; follow an agreed proposal and a reference implementation.

```text
tasks/<task-id>/
├── instruction.md
├── task.toml
├── environment/
├── solution/
└── tests/
```

| Path | Purpose |
| --- | --- |
| `instruction.md` | Agent-facing engineering problem, constraints, and submission format |
| `task.toml` | Harbor configuration and execution settings |
| `environment/` | Initial design, simulator configuration, and agent runtime |
| `solution/` | Reference design and any scripts needed to reproduce it |
| `tests/` | Evaluation entry points, validity and objective checks, and tests |

Some existing tasks separate evaluator implementation into further directories. **Do not expose hidden evaluator logic or reference answers inside the agent workspace.** The isolation boundary matters more than the presence of a particular folder name.

## Before opening a PR

1. Confirm the instructions identify the final submission artifact unambiguously.
2. Rebuild and evaluate the reference using the same evaluator pathway intended for submitted artifacts.
3. Check representative invalid submissions and document what the evaluator returns.
4. Record the simulator, runtime, version, resource limits, and evaluation commands.
5. Document performance targets, objective direction, and reference values.

Until representative tasks have been migrated, use an existing [PICBench task folder](https://github.com/MomeniAli/PICBench/tree/main/tasks) as the implementation starting point. This repository's structural CI is intentionally lightweight and is **not** a substitute for running Harbor or validating the physics.
