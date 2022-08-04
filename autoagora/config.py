# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional, Sequence

import configargparse


class _Args(configargparse.Namespace):
    def __getattribute__(self, name) -> Any:
        return object.__getattribute__(self, name)


# `args` will contain the global argument values after calling `init_config()`
args = _Args()


def init_config(argv: Optional[Sequence[str]] = None):
    """Parses the arguments from the global argument parser and sets logging config.

    To add arguments, simply add them to the global argument parser from any module, as
    long as the modules are imported before the invocation of that function.

    Argument values are added to the global `args` namespace object declared in this
    module.
    """
    argsparser = configargparse.get_argument_parser()
    argsparser.add_argument(
        "--log-level",
        env_var="LOG_LEVEL",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        required=False,
    )
    argsparser.parse_args(args=argv, namespace=args)

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.root.setLevel(args.log_level)
