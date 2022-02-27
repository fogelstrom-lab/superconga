#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

from typing import Optional, Union

# Valid choices for certain commands.
VALID_PRECISIONS = [32, 64]
VALID_GAUGES = ["radial", "solenoid"]
VALID_CHARGE_SIGNS = [-1, +1]
VALID_NORMS = ["L1", "L2", "LInf"]
VALID_ACCELERATORS = [
    "Adaptive-Ball",
    "AB",
    "Anderson",
    "Barzilai-Borwein",
    "BB",
    "Picard",
    "Polyak",
]
VALID_DATA_FORMATS = ["h5", "csv"]

# Set value from command-line argument.
def set_value(
    config: dict,
    key: str,
    value: Union[dict, int, float, list],
    group: Optional[str] = None,
) -> None:
    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        if group is None:
            config[key] = value
        else:
            if group in config:
                config[group][key] = value
            else:
                config[group] = {key: value}


# Set value from command-line flag (on/off for true/false).
def set_flag(
    config: dict,
    key: str,
    on: bool,
    off: bool,
    group: Optional[str] = None,
) -> None:
    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        if group is None:
            config[key] = on
        else:
            config[group][key] = on
