#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Convenience functions for calling C++ binary."""

# Built-in modules and third-party libraries.
import sys
import os
import argparse
import json
import subprocess
from pathlib import Path

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries.settings import configuration

# ------------------------------------------------------------------------------
# Run C++ program as a subprocess.
# ------------------------------------------------------------------------------
def run_cpp_binary(cpp_binary_str: str, sim_params: dict) -> int:
    # Write temporary config file.
    with open(configuration.temp_config_file_name, "w") as file:
        json.dump(sim_params, file, indent="\t", ensure_ascii=False)

    # Construct Path object.
    cpp_binary_path = Path(cpp_binary_str)

    # Run the C++ program as a subprocess.
    if cpp_binary_path.is_file() and os.access(cpp_binary_path, os.X_OK):
        status_code = subprocess.call(
            [str(cpp_binary_path), configuration.temp_config_file_name]
        )
    else:
        status_code = 1
        print(
            f"--Cannot find the C++ executable '{cpp_binary_path}'. See the README for how to compile it."
        )

    # Remove temporary config file.
    if Path(configuration.temp_config_file_name).is_file():
        os.remove(configuration.temp_config_file_name)

    return status_code
