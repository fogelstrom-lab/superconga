#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Common functionality for updating configuration parameters with arguments from the CLI.

As well as the valid choices of certain parameters.
"""

from typing import Optional, Union

# Valid choices for certain commands.
VALID_PRECISIONS = [32, 64]
VALID_GAUGES = ["landau", "solenoid", "symmetric"]
VALID_CHARGE_SIGNS = [-1, +1]
VALID_NORMS = ["l1", "l2", "linf"]
VALID_ACCELERATORS = [
    "anderson",
    "barzilai-borwein",
    "bb",
    "congacc",
    "picard",
    "polyak",
]
VALID_DATA_FORMATS = ["csv", "h5"]


def set_value(
    config: dict,
    key: str,
    value: Union[dict, int, float, list],
    group: Optional[str] = None,
) -> None:
    """Set value from command-line argument"""

    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        if group is None:
            config[key] = value
        else:
            if group in config:
                config[group][key] = value
            else:
                config[group] = {key: value}


def set_flag(
    config: dict,
    key: str,
    on: bool,
    off: bool,
    group: Optional[str] = None,
) -> None:
    """Set value from command-line flag (on/off for true/false)."""

    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        if group is None:
            config[key] = on
        else:
            config[group][key] = on
