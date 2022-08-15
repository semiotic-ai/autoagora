# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

from agents.agent_factory import AgentFactory
from environments.environment_factory import EnvironmentFactory


def init_simulation(parser: argparse.ArgumentParser):

    # Parse arguments.
    args = parser.parse_args()

    # Open JSON file.
    with open(args.config) as f:
        # Load the configuration.
        config = json.loads(f.read())

    # Instantiate the agent.
    agents = {}
    for agent_name, agent_spec in config["agents"].items():
        # Make sure there is only one agent specified per section.
        if len(agent_spec.items()) == 1:

            # Get agent specification.
            agent_type, properties = next(iter(agent_spec.items()))

            # Instantiate the agent.
            agents[agent_name] = AgentFactory(agent_type=agent_type, **properties)

        elif len(agent_spec.items()) == 2:  
            # ... or there is only agent spec + number of instances.

            num_instances = agent_spec.pop("num_instances")
            # Get agent specification.
            agent_type, properties = next(iter(agent_spec.items()))

            # Instatiate n instances.
            for i in range(num_instances):
                agents[f"{agent_name}{i}"] = AgentFactory(agent_type=agent_type, **properties)

        else:
            raise ValueError(f"Section {agent_name} is invalid!")

    # Make sure there is only one environment specified.
    assert len(config["environment"].items()) == 1

    # Get env specification.
    environment_type, properties = next(iter(config["environment"].items()))

    # Instantiate the environment.
    environment = EnvironmentFactory(environment_type=environment_type, **properties)

    return args, environment, agents
