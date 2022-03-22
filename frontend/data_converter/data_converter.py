#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
The main part of the data-format converter.
"""

import argparse
import json
import numpy as np
import os
import sys


from .arguments import make_parser

sys.path.insert(0, "../")
from common import io


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Run data-format converter."""

    # Working directory.
    data_dir = args.load_path

    # Format to convert from.
    src_data_format = args.src_data_format

    # Format to convert to.
    dst_data_format = args.dst_data_format

    # What to convert.
    what = args.what

    # Nothing to do if the data formats are equal.
    if src_data_format == dst_data_format:
        sys.exit("-- Data formats are equal.")

    # Create a list of keys, i.e. stuff to convert.
    if what == "simulation":
        # Everything except the order parameter.
        keys = [
            "domain",
            "current_density",
            "vector_potential",
            "magnetic_flux_density",
        ]

        # Read the configuration file.
        config = io.read_simulation_config(data_dir=data_dir)

        # Append order-parameter components.
        for symmetry in config["physics"]["order_parameter"].keys():
            keys.append("order_parameter_" + symmetry)
    elif what == "postprocess":
        keys = ["ldos"]
    else:
        sys.exit(f"-- ERROR! {what} data not understood.")

    # Loop over keys and convert the data.
    success = True
    for key in keys:
        # Read the data from the old format.
        data = io.read_data(data_dir=data_dir, key=key, data_format=src_data_format)

        # Write the data to the new format.
        success = io.write_data(
            data_dir=data_dir,
            key=key,
            data=data,
            data_format=dst_data_format,
        )

        # Abort if not successful.
        if not success:
            print(f"-- ERROR! Could not convert {key}.")
            break

        # Read the data from the new format.
        test_data = io.read_data(
            data_dir=data_dir, key=key, data_format=dst_data_format
        )

        # Compute the maximum error.
        error = np.amax(np.abs(data - test_data))

        # Abort if not successful.
        success = error < 1e-15
        if not success:
            print(
                f"-- ERROR! The newly written data for {key} is not identical to the old."
            )
            break

    if success:
        print(
            f"-- SUCCESS! Converted data in {data_dir} from {src_data_format} to {dst_data_format}."
        )

        # Remove files with the old data format.
        if args.clean:
            if src_data_format == "csv":
                for key in keys:
                    if "order_parameter" in key:
                        filename = "order_parameter.csv"
                    else:
                        filename = key + ".csv"
                    path = os.path.join(data_dir, filename)
                    os.remove(path)
                    print(f"-- Removed {path}.")
            elif src_data_format == "h5":
                filename = what + ".h5"
                path = os.path.join(data_dir, filename)
                os.remove(path)
                print(f"-- Removed {path}.")
            else:
                print(
                    f"-- WARNING! Removal of {src_data_format} files is not implemented."
                )

    else:
        print(
            f"-- FAILURE! Could not convert data in {data_dir} from {src_data_format} to {dst_data_format}."
        )
