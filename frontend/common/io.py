#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Functionality for reading/writing data from/to disk."""

import json
import os

try:  # Optional: only needed for reading/writing HDF5 data (not for setup/compile).
    import h5py
except ImportError:  # pragma: no cover
    h5py = None
try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None
import sys


def read_config(data_dir, filenames):
    """Read a JSON configuration file."""

    if not isinstance(filenames, list):
        filenames = [filenames]

    success = False
    for filename in filenames:
        config_path = os.path.join(data_dir, filename)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = f.read()
            config = json.loads(data)
            success = True
            break

    if not success:
        sys.exit("-- ERROR! Could not find config file.")

    return config


def read_simulation_config(data_dir):
    """Read a JSON simulation configuration file."""

    # Supporting old filename.
    filenames = [
        "simulation_config.json",
        "config.json",
    ]
    return read_config(data_dir=data_dir, filenames=filenames)


def read_postprocess_config(data_dir):
    """Read a JSON postprocess configuration file."""

    # Supporting old filename.
    filenames = [
        "postprocess_config.json",
        "postprocess.json",
    ]
    return read_config(data_dir=data_dir, filenames=filenames)


# Read data for a certain key from h5 file.
def read_data_h5(hf, key):
    """Read a dataset from an H5 file."""

    if not key in hf.keys():
        sys.exit(f"-- ERROR! Could not find {key} in h5 file.")

    # Fetch dimensions.
    dim_x = hf[key].attrs["dim_x"]
    dim_y = hf[key].attrs["dim_y"]

    # Convert to NumPy.
    data = np.array(hf[key])
    return np.reshape(data, [-1, dim_y, dim_x])


# Read data for a certain key from Pandas dataframe.
def read_data_csv(df, key):
    """Read a column from a CSV file."""

    if not ((key in df) and ("x_idx" in df) and ("y_idx" in df)):
        sys.exit(f"-- ERROR! Could not find {key}, or indices, in csv file.")

    # Fetch dimensions.
    dim_x = df["x_idx"].max() + 1
    dim_y = df["y_idx"].max() + 1

    # Convert to NumPy.
    data = df[key].to_numpy()
    return np.reshape(data, [-1, dim_y, dim_x])


def get_filename(data_dir, filenames):
    """Get the first existing filename from a list of filenames."""

    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return filename
    return filenames[0]


def read_data(data_dir, key, data_format):
    """Read data, e.g. the order parameter, from a directory."""

    if data_format == "h5":
        # Check if H5 file exists.
        # TODO(niclas): Check data format in config? I.e. use CSV even though h5 file exists.
        if key in ["ldos"]:
            filenames = ["postprocess.h5", "spectroscopy.h5"]
        else:
            filenames = ["simulation.h5"]
        filename = get_filename(data_dir=data_dir, filenames=filenames)

        # Full path.
        h5_path = os.path.join(data_dir, filename)

        # Check if it exists.
        if not os.path.exists(h5_path):
            sys.exit(f"-- ERROR! The file {h5_path} does not exist.")

        # Open H5 file.
        hf = h5py.File(h5_path, "r")

        # Read data.
        data = read_data_h5(hf=hf, key=key)
    elif data_format == "csv":
        # Open CSV file.
        if "order_parameter" in key:
            filename = "order_parameter.csv"
        else:
            filename = key + ".csv"
        csv_path = os.path.join(data_dir, filename)

        # The file does not exist.
        if not os.path.exists(csv_path):
            sys.exit(f"-- ERROR! The file {csv_path} does not exist.")

        # Open file.
        df = pd.read_csv(csv_path)

        # Unfortunately the data is saved in a rather annoying way.
        if "order_parameter" in key:
            symmetry = key[(len("order_parameter") + 1) :]
            key_real = symmetry + "_re"
            key_imag = symmetry + "_im"
            data_real = read_data_csv(df=df, key=key_real)
            data_imag = read_data_csv(df=df, key=key_imag)
            data = np.concatenate([data_real, data_imag], axis=0)
        elif "current_density" in key:
            data_x = read_data_csv(df=df, key="j_x")
            data_y = read_data_csv(df=df, key="j_y")
            data = np.concatenate([data_x, data_y], axis=0)
        elif "vector_potential" in key:
            data_x = read_data_csv(df=df, key="a_x")
            data_y = read_data_csv(df=df, key="a_y")
            data = np.concatenate([data_x, data_y], axis=0)
        elif "flux_density" in key:
            data = read_data_csv(df=df, key="b_z")
        elif "domain" in key:
            data = read_data_csv(df=df, key=key)
        elif "ldos" in key:
            data = read_data_csv(df=df, key=key)
        else:
            raise ValueError(f"-- ERROR! Can not find data for {key}.")
    else:
        raise ValueError(f"-- ERROR! Data format {data_format} is not supported.")

    return data


# Write data for a certain key to h5 file.
def write_data_h5(hf, key, data):
    """Write a dataset to an H5 file."""

    # Get data dimensions.
    dims = np.shape(data)
    if len(dims) == 2:
        dim_z = 1
        dim_y, dim_x = np.shape(data)
    elif len(dims) == 3:
        dim_z, dim_y, dim_x = np.shape(data)
    else:
        print("-- ERROR! The data should be 2D or 3D.")
        return False

    # TODO(Niclas): Do it in a better way.
    if "order_parameter" in key:
        is_complex = 1
        assert dim_z % 2 == 0
        dim_z = dim_z // 2
    else:
        is_complex = 0

    # Delete dataset if it exists.
    if key in hf.keys():
        del hf[key]

    # Create dataset and set attributes.
    hf.create_dataset(key, data=data.flatten())
    hf[key].attrs["dim_x"] = dim_x
    hf[key].attrs["dim_y"] = dim_y
    hf[key].attrs["dim_z"] = dim_z
    hf[key].attrs["is_complex"] = is_complex
    return True


# Write data for a certain key to csv file.
def write_data_csv(filename, key, data):
    """Write a column to a CSV file."""

    # Remove dimensions.
    data = np.squeeze(data)

    # Read file if it exists.
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame()

    # Get data dimensions.
    dims = np.shape(data)

    # TODO(Niclas): Unnecessary to do this every time we're adding a column.
    if len(dims) == 2:
        dim_y, dim_x = np.shape(data)

        # Create meshgrid.
        vx = np.array(list(range(dim_x)))
        vy = np.array(list(range(dim_y)))
        X, Y = np.meshgrid(vx, vy)

        # Add/update columns.
        df["x_idx"] = X.flatten()
        df["y_idx"] = Y.flatten()
    elif len(dims) == 3:
        dim_z, dim_y, dim_x = np.shape(data)

        # Create meshgrid.
        vx = np.array(list(range(dim_x)))
        vy = np.array(list(range(dim_y)))
        vz = np.array(list(range(dim_z)))
        X, Y, Z = np.meshgrid(vx, vy, vz)

        # Add/update columns.
        df["x_idx"] = X.flatten()
        df["y_idx"] = Y.flatten()
        df["e_idx"] = Z.flatten()
    else:
        print("-- ERROR! The data should be 2D or 3D.")
        return False

    # Add column.
    df[key] = data.flatten()

    # Write.
    df.to_csv(filename, header=True, index=False, float_format="%.16e")

    return True


def write_data(data_dir, key, data, data_format):
    """Write data, e.g. the order parameter, to a directory."""

    if data_format == "h5":
        if key in ["ldos"]:
            filename = "postprocess.h5"
        else:
            filename = "simulation.h5"

        # Open H5 file.
        h5_path = os.path.join(data_dir, filename)
        hf = h5py.File(h5_path, "a")

        # Write data.
        success = write_data_h5(hf=hf, key=key, data=data)
    elif data_format == "csv":
        # Determine filename.
        if "order_parameter" in key:
            filename = "order_parameter.csv"
        else:
            filename = key + ".csv"
        csv_path = os.path.join(data_dir, filename)

        # Unfortunately the data is saved in a rather annoying way.
        if "order_parameter" in key:
            symmetry = key[(len("order_parameter") + 1) :]
            key_real = symmetry + "_re"
            key_imag = symmetry + "_im"
            data_real, data_imag = np.split(data, indices_or_sections=2, axis=0)
            success_real = write_data_csv(
                filename=csv_path, key=key_real, data=data_real
            )
            success_imag = write_data_csv(
                filename=csv_path, key=key_imag, data=data_imag
            )
            success = success_real and success_imag
        elif "current_density" in key:
            data_x, data_y = np.split(data, indices_or_sections=2, axis=0)
            success_x = write_data_csv(filename=csv_path, key="j_x", data=data_x)
            success_y = write_data_csv(filename=csv_path, key="j_y", data=data_y)
            success = success_x and success_y
        elif "vector_potential" in key:
            data_x, data_y = np.split(data, indices_or_sections=2, axis=0)
            success_x = write_data_csv(filename=csv_path, key="a_x", data=data_x)
            success_y = write_data_csv(filename=csv_path, key="a_y", data=data_y)
            success = success_x and success_y
        elif "flux_density" in key:
            success = write_data_csv(filename=csv_path, key="b_z", data=data)
        elif "domain" in key:
            success = write_data_csv(filename=csv_path, key=key, data=data)
        elif "ldos" in key:
            success = write_data_csv(filename=csv_path, key=key, data=data)
        else:
            raise ValueError(f"-- ERROR! Can not find data for {key}.")
    else:
        raise ValueError(f"-- ERROR! Data format {data_format} is not supported.")

    return success
