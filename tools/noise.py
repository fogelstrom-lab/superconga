#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Simple script for adding noise to the order parameter and vector potential."""

import argparse
import json
import numpy as np
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../frontend"))
from common import io

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    # Create parser and subparsers.
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to order parameter and vector potential."
    )
    parser.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Path to original data. (Default: %(default)s)",
    )
    parser.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Where to put the noisy data. Files will be overwritten if they exist. (Default: %(default)s)",
    )
    parser.add_argument(
        "--op-noise",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise to apply to the order parameter. (Default: %(default)s)",
    )
    parser.add_argument(
        "--vp-noise",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise to apply to the vector potential. (Default: %(default)s)",
    )

    # Parse arguments.
    args = parser.parse_args()

    # Print help if nothing is set.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Data directories.
    load_dir = os.path.normpath(args.load_path)
    save_dir = os.path.normpath(args.save_path)

    # Check if load_dir is set.
    if load_dir is None:
        sys.exit("-- ERROR! Please specify '--load-path'.")

    # Check if save_dir is set.
    if save_dir is None:
        sys.exit("-- ERROR! Please specify '--save-path'.")

    # Standard deviation of the Gaussian noise.
    op_noise = args.op_noise
    vp_noise = args.vp_noise

    # Read config.
    config = io.read_simulation_config(data_dir=load_dir)

    # Data format.
    data_format = config["misc"]["data_format"]

    # Copy files.
    if not load_dir == save_dir:
        os.makedirs(save_dir)
        if data_format == "csv":
            for filename in ["domain.csv", "order_parameter.csv", "vector_potential.csv"]:
                path = os.path.join(load_dir, filename)
                shutil.copy(path, save_dir)
        elif data_format == "h5":
            path = os.path.join(load_dir, "simulation.h5")
            shutil.copy(path, save_dir)
        else:
            raise ValueError(f"-- ERROR! Data format {data_format} not supported.")

    # Get keys.
    keys = ["vector_potential"]
    for symmetry in config["physics"]["order_parameter"].keys():
        keys.append("order_parameter_" + symmetry)

    # Loop over keys and add noise.
    success = True
    for key in keys:
        # Read the data.
        data = io.read_data(data_dir=save_dir, key=key, data_format=data_format)

        # Add noise.
        if "order_parameter" in key:
            stddev = op_noise
        elif "vector_potential" in key:
            stddev = vp_noise
        else:
            continue
        data += np.random.normal(loc=0.0, scale=stddev, size=data.shape)

        # Write the data.
        success = io.write_data(
            data_dir=save_dir,
            key=key,
            data=data,
            data_format=data_format,
        )

        # Abort if not successful.
        if not success:
            print(f"-- ERROR! Could not write {key}.")
            break

    if success:
        print(
            f"-- SUCCESS! Loaded data from {load_dir}, added noise, and saved in {save_dir}."
        )
    else:
        print(
            f"-- FAILURE! Could not load data from {load_dir}, add noise, and save in {save_dir}."
        )


if __name__ == "__main__":
    main()
