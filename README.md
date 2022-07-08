# AutoAgora

:warning: **Beware! This software is still experimental, use at your own risk!**
:warning:

An [Agora](https://github.com/graphprotocol/agora) cost model automation tool for The
Graph indexers:

- Automates the creation of relative prices for commonly seen query skeletons.
  (Off by default).
- Continuously tunes a per-subgraph absolute price multiplier, using continuous
  online reinforcement learning.

## AutoAgora sub-components repos

For AutoAgora to function correctly, you will also need to set up:

- [AutoAgora indexer-service](https://gitlab.com/semiotic-ai/the-graph/autoagora-indexer-service)
- [AutoAgora Sidecar](https://gitlab.com/semiotic-ai/the-graph/autoagora-sidecar)
- [AutoAgora Processor](https://gitlab.com/semiotic-ai/the-graph/autoagora-processor)
