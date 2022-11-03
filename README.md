[![Coveralls](https://img.shields.io/coveralls/github/semiotic-ai/autoagora)](https://coveralls.io/github/semiotic-ai/autoagora)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Semantic Versioning](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/spec/v2.0.0.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# AutoAgora

**AutoAgora is *beta* software. Please use responsibly, monitor it (Prometheus metrics)
, [write issues and contribute](CONTRIBUTING.md)!**

An [Agora](https://github.com/graphprotocol/agora) cost model automation tool for The Graph indexers:

- Automates the creation of relative prices for commonly seen query skeletons. (Off by default).
- Continuously tunes a per-subgraph absolute price multiplier, using continuous online reinforcement learning.

## Indexer's guide

### Build

Just build the container!

```sh
docker build -t autoagora .
```

### Usage

For AutoAgora to function correctly, you will also need to set up:

- [AutoAgora indexer-service](https://github.com/semiotic-ai/autoagora-indexer-service)
- [AutoAgora Processor](https://github.com/semiotic-ai/autoagora-processor)

AutoAgora will continously:

- Watch for the indexer's current allocations by querying the `indexer-agent`'s management GraphQL endpoint.
- Analyze the query logs stored in a PostgreSQL database -- logs that were previously processed by the `AutoAgora
indexer-service` wrapper and the `AutoAgora Processor`.
- Gather query metrics from the `indexer-service`'s prometheus metrics endpoint.
- Update the allocated subgraph's cost models by sending mutations to the `indexer-agent`'s management GraphQL endpoint.

Therefore, only a single instance of `AutoAgora` should be running against an `indexer-agent`.

Configuration:

```txt
usage: autoagora [-h] --indexer-agent-mgmt-endpoint INDEXER_AGENT_MGMT_ENDPOINT --indexer-service-metrics-endpoint
                 INDEXER_SERVICE_METRICS_ENDPOINT --logs-postgres-host LOGS_POSTGRES_HOST
                 [--logs-postgres-port LOGS_POSTGRES_PORT] --logs-postgres-database LOGS_POSTGRES_DATABASE
                 --logs-postgres-username LOGS_POSTGRES_USERNAME --logs-postgres-password LOGS_POSTGRES_PASSWORD
                 [--agora-models-refresh-interval AGORA_MODELS_REFRESH_INTERVAL] [--experimental-model-builder]

optional arguments:
  -h, --help            show this help message and exit
  --indexer-agent-mgmt-endpoint INDEXER_AGENT_MGMT_ENDPOINT
                        URL to the indexer-agent management GraphQL endpoint. [env var: INDEXER_AGENT_MGMT_ENDPOINT]
                        (default: None)
  --indexer-service-metrics-endpoint INDEXER_SERVICE_METRICS_ENDPOINT
                        HTTP endpoint for the indexer-service metrics. [env var: INDEXER_SERVICE_METRICS_ENDPOINT]
                        (default: None)
  --logs-postgres-host LOGS_POSTGRES_HOST
                        Host of the postgres instance storing the logs. [env var: LOGS_POSTGRES_HOST] (default: None)
  --logs-postgres-port LOGS_POSTGRES_PORT
                        Port of the postgres instance storing the logs. [env var: LOGS_POSTGRES_PORT] (default: 5432)
  --logs-postgres-database LOGS_POSTGRES_DATABASE
                        Name of the logs database. [env var: LOGS_POSTGRES_DATABASE] (default: None)
  --logs-postgres-username LOGS_POSTGRES_USERNAME
                        Username for the logs database. [env var: LOGS_POSTGRES_USERNAME] (default: None)
  --logs-postgres-password LOGS_POSTGRES_PASSWORD
                        Password for the logs database. [env var: LOGS_POSTGRES_PASSWORD] (default: None)
  --agora-models-refresh-interval AGORA_MODELS_REFRESH_INTERVAL
                        Interval in seconds between rebuilds of the Agora models. [env var:
                        AGORA_MODELS_REFRESH_INTERVAL] (default: 3600)
  --experimental-model-builder
                        Activates the relative query cost discovery. Otherwise only builds a default query pricing
                        model with automated market price discovery. [env var: EXPERIMENTAL_MODEL_BUILDER] (default:
                        False)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        [env var: LOG_LEVEL] (default: WARNING)
  --json-logs JSON_LOGS
                        Output logs in JSON format. Compatible with GKE. [env var: JSON_LOGS] (default: False)
```

AutoAgora also exposes metrics for Prometheus on port `8000` (Example values):

```txt
# HELP bandit_reward Reward of the bandit training: queries_per_second * price_multiplier.
# TYPE bandit_reward gauge
bandit_reward{subgraph="QmRDGLp6BHwiH9HAE2NYEE3f7LrKuRqziHBv76trT4etgU"} 1.577651313168855e-07
# HELP bandit_price_multiplier Price multiplier sampled from the Gaussian model.
# TYPE bandit_price_multiplier gauge
bandit_price_multiplier{subgraph="QmRDGLp6BHwiH9HAE2NYEE3f7LrKuRqziHBv76trT4etgU"} 2.60150080442184e-07
# HELP bandit_stddev Standard deviation of the Gaussian price multiplier model.
# TYPE bandit_stddev gauge
bandit_stddev{subgraph="QmRDGLp6BHwiH9HAE2NYEE3f7LrKuRqziHBv76trT4etgU"} 1.843469500541687
# HELP bandit_mean Mean of the Gaussian price multiplier model.
# TYPE bandit_mean gauge
bandit_mean{subgraph="QmRDGLp6BHwiH9HAE2NYEE3f7LrKuRqziHBv76trT4etgU"} 3.653126148672616e-05
```

Where "bandit" refers to the reinforcement learning method (Continuum-armed bandit) used to track the market price for each subgraph.

## Developer's guide

### Installation directly from the source code

To install AutoAgora directly from the source code please clone the repository and install package in the virtual environment using `poetry`:
```console
git clone https://github.com/semiotic-ai/autoagora.git
cd autoagora
poetry install
```

### Running the AutoAgora code

All scripts should be executed in the virtual environment managed by `poetry`.

### Running the test suite

```console
poetry run python -m pytest
```
