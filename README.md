# AutoAgora

:warning: **Beware! This software is still experimental, use at your own risk!**
:warning:

An [Agora](https://github.com/graphprotocol/agora) cost model automation tool for The
Graph indexers:

- Automates the creation of relative prices for commonly seen query skeletons.
  (Off by default).
- Continuously tunes a per-subgraph absolute price multiplier, using continuous
  online reinforcement learning.

## Installation

### Installation directly from the source code

To install AutoAgora directly from the source code please clone the repository and install package in the virtual environment using `poetry`:
```console
git clone git@gitlab.com:semiotic-ai/the-graph/autoagora.git
cd autoagora
poetry install
```

## Running the AutoAgora code

All scripts should be executed in the virtual environment managed by `poetry`.
### Running the test suite

```console
poetry run python -m pytest
```

### Running the bandit-related scripts

There are three scripts in the bandit_scripts folder:

* `show_simulated_subgraph.py` - runs and (optionally) visualizes the simulation (only environment)
* `show_bandit.py` - runs and (optionally) visualizes the simulation (both agent and environment)
* `train_bandit.py` - runs the simulation and trains an agent in a given environment (no visualization, logging to TensorBoard)

Each script can be parametrized by arguments passed from the command line. For example, to run the simulation with `ppo` agent in the `noisy_static` environment please run:

```console
poetry run python bandit_scripts/show_bandit.py -a ppo -e noisy_static --show
```

More details on arguments can be accessed in help (`--h`).

## AutoAgora sub-components repos

For AutoAgora to function correctly, you will also need to set up:

- [AutoAgora indexer-service](https://gitlab.com/semiotic-ai/the-graph/autoagora-indexer-service)
- [AutoAgora Sidecar](https://gitlab.com/semiotic-ai/the-graph/autoagora-sidecar)
- [AutoAgora Processor](https://gitlab.com/semiotic-ai/the-graph/autoagora-processor)
