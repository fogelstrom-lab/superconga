#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Various helpful methods for file IO."""

# Built-in modules and third-party libraries.
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Union

# Custom libraries.

# Global settings.

# -----------------------------------------------------------------------------
# VALIDATION/TESTING
# -----------------------------------------------------------------------------
# Check that directory exists and is readable.
def check_dir(
    data_path: Union[str, bytes, os.PathLike],
    verbose: bool = True,
    exit_on_failure: bool = True,
) -> bool:

    success = True

    if (not data_path.is_dir()) or (not os.access(data_path, os.F_OK)):
        success = False

        if verbose:
            print(f"Error: Cannot find directory: '{data_path}'.")

        if exit_on_failure:
            status_code = 1
            sys.exit(status_code)
    elif not os.access(data_path, os.R_OK):
        success = False

        if verbose:
            print(f"Error: Cannot read directory: '{data_path}'.")

        if exit_on_failure:
            status_code = 1
            sys.exit(status_code)

    return success


# Check that file exists and is readable.
def check_file(
    data_path: Union[str, bytes, os.PathLike],
    verbose: bool = True,
    exit_on_failure: bool = True,
) -> bool:

    success = True

    if (not data_path.is_file()) or (not os.access(data_path, os.F_OK)):
        success = False

        if verbose:
            print(f"Error: Cannot find file: '{data_path}'.")

        if exit_on_failure:
            status_code = 1
            sys.exit(status_code)
    elif not os.access(data_path, os.R_OK):
        success = False

        if verbose:
            print(f"Error: Cannot read file: '{data_path}'.")

        if exit_on_failure:
            status_code = 1
            sys.exit(status_code)

    return success


# -----------------------------------------------------------------------------
# FILE PARSING: CSV
# -----------------------------------------------------------------------------
# General parser for csv files: return result as pandas DataFrame.
def parse_general_csv(file_path: str) -> pd.DataFrame:
    # Check that file exists and is readable: exit on failure.
    check_file(file_path)

    # Attempt to parse CSV file.
    data = pd.read_csv(file_path)

    return data