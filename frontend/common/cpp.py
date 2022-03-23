#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Functionality for writing a configuration to file and calling the relevant C++ binary."""

import datetime
import json
import os
import subprocess
import sys


def run(config: dict, cpp_path: str):
    """Call a C++ binary using a given configuration."""

    # Create tmp directory if it does not exist.
    tmp_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../tmp"))
    os.makedirs(tmp_path, exist_ok=True)

    # Write config file to tmp folder.
    dt_date = datetime.datetime.now()
    config_path = os.path.join(tmp_path, dt_date.isoformat() + ".json")
    with open(config_path, "w") as file:
        json.dump(config, file, indent="\t", ensure_ascii=False)

    # Run the C++ program as a subprocess.
    if os.path.exists(cpp_path) and os.access(cpp_path, os.X_OK):
        subprocess.run([cpp_path, config_path])
    else:
        sys.exit(
            f"-- ERROR! Cannot find the C++ executable '{cpp_path}'. See the README for how to compile it."
        )

    # Remove temporary config file.
    if os.path.exists(config_path):
        os.remove(config_path)
