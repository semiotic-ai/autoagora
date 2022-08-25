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

    agents = {}
    # Instantiate agents.
    for agent_name, agent_section in config["agents"].items():

        # Get number of instances.
        num_instances = agent_section.pop("num_instances", 1)

        # Instantiate a single agent.
        if num_instances == 1:
            agents[agent_name] = AgentFactory(
                agent_name=agent_name, agent_section=agent_section
            )
        else:
            # Instatiate n instances.
            for i in range(num_instances):
                subagent_name = f"{agent_name}{i}"
                agents[subagent_name] = AgentFactory(
                    agent_name=subagent_name, agent_section=agent_section
                )

    # Make sure there is only one environment specified.
    assert len(config["environment"].items()) == 1

    # Get env specification.
    environment_type, properties = next(iter(config["environment"].items()))

    # Instantiate the environment.
    environment = EnvironmentFactory(
        environment_type_name=environment_type, **properties
    )

    return args, environment, agents
