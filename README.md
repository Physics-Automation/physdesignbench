<h1 align="center"><img src="docs/assets/physdesignbench_logo.svg" alt="PhysDesignBench logo" height="110"><br/>PhysDesignBench</h1>

<p align="center"><em>Can AI agents turn engineering specifications into working physical designs?</em></p>

<p align="center">
PhysDesignBench evaluates AI agents on physics-based engineering design. Agents use simulation tools to design and optimize physical devices; their final submissions are independently rebuilt and checked against engineering constraints and performance objectives.
</p>

## Links

- [Website](https://physdesignbench.com)
- [Leaderboard](https://physdesignbench.com/leaderboard)
- [Tasks](https://physdesignbench.com/tasks)
- [Explore results](https://physdesignbench.com/extended)
- [Run the benchmark](https://physdesignbench.com/run)
- [Citation (BibTeX)](https://physdesignbench.com/citation.bib)

## Benchmark

The paper evaluates **26 tasks**: 16 Design tasks and 10 Reimplementation tasks, spanning photonics, electromagnetics, and coupled multiphysics systems.

Agents work with domain-specific simulators under fixed time and evaluation budgets. The benchmark reports the **mean normalized score (MNS)** of final submissions and **engineering pass rate (EPR)**, which measures how often final designs meet the selected performance targets.

For task specifications, design examples, and results, see the [website](https://physdesignbench.com).

## Running the benchmark

See the [run guide](https://physdesignbench.com/run) for environment setup and execution instructions.

Existing task implementations are currently hosted in the [original PICBench repository](https://github.com/MomeniAli/PICBench). They will be migrated here separately; this repository presently hosts the contribution pipeline.

## Contributing

We welcome new design problems, expert reviews, and evaluations of additional agents.

Read the [contribution guide](docs/contributing/README.md) for task requirements and the submission process, or [propose a task](https://github.com/Physics-Automation/physdesignbench/issues/new/choose).

## Citation

If you use PhysDesignBench in your research, please cite the paper. The current [BibTeX entry](https://physdesignbench.com/citation.bib) is maintained on the project website.
