#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import h5py
import json
import numpy as np
import os
import pandas as pd
import sys


def get_config(data_dir, filenames):
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


def get_simulation_config(data_dir):
    # Supporting old filename.
    filenames = [
        "simulation_config.json",
        "config.json",
    ]
    return get_config(data_dir=data_dir, filenames=filenames)


def get_postprocess_config(data_dir):
    # Supporting old filename.
    filenames = [
        "postprocess_config.json",
        "postprocess.json",
    ]
    return get_config(data_dir=data_dir, filenames=filenames)


# Get data for a certain key from h5 file.
def get_data_h5(hf, key):
    # Fetch dimensions.
    dim_x = hf[key].attrs["dim_x"]
    dim_y = hf[key].attrs["dim_y"]

    # Convert to NumPy.
    data = np.array(hf[key])
    return np.reshape(data, [-1, dim_y, dim_x])


# Get data for a certain key from Pandas dataframe.
def get_data_csv(df, key):
    # Fetch dimensions.
    dim_x = df["x_idx"].max() + 1
    dim_y = df["y_idx"].max() + 1

    # Convert to NumPy.
    data = df[key].to_numpy()
    return np.reshape(data, [-1, dim_y, dim_x])


def get_filename(data_dir, filenames):
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return filename
    return None


def get_data(data_dir, key):
    # Check if H5 file exists.
    # TODO(niclas): Check data format in config? I.e. use CSV even though h5 file exists.
    if key in ["ldos"]:
        filenames = ["postprocess.h5", "spectroscopy.h5"]
    else:
        filenames = ["simulation.h5"]
    filename = get_filename(data_dir=data_dir, filenames=filenames)

    if filename is not None:
        # Open H5 file.
        h5_path = os.path.join(data_dir, filename)
        hf = h5py.File(h5_path, "r")

        # Get data.
        data = get_data_h5(hf=hf, key=key)
    else:
        # Open CSV file.
        if "order_parameter" in key:
            csv_path = os.path.join(data_dir, "order_parameter.csv")
        else:
            csv_path = os.path.join(data_dir, key + ".csv")
        df = pd.read_csv(csv_path)

        # Unfortunately the data is saved in a rather annoying way.
        if "order_parameter" in key:
            symmetry = key[(len("order_parameter") + 1) :]
            key_real = symmetry + "_re"
            key_imag = symmetry + "_im"
            data_real = get_data_csv(df=df, key=key_real)
            data_imag = get_data_csv(df=df, key=key_imag)
            data = np.concatenate([data_real, data_imag], axis=0)
        elif "current_density" in key:
            data_x = get_data_csv(df=df, key="j_x")
            data_y = get_data_csv(df=df, key="j_y")
            data = np.concatenate([data_x, data_y], axis=0)
        elif "vector_potential" in key:
            data_x = get_data_csv(df=df, key="a_x")
            data_y = get_data_csv(df=df, key="a_y")
            data = np.concatenate([data_x, data_y], axis=0)
        elif "flux_density" in key:
            data = get_data_csv(df=df, key="b_z")
        elif "domain" in key:
            data = get_data_csv(df=df, key=key)
        elif "ldos" in key:
            data = get_data_csv(df=df, key=key)
        else:
            raise ValueError(f"-- ERROR! Can not find data for {key}.")

    return data
