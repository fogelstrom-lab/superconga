#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Parse observables/quantities saved on grid lattice from SuperConga."""

# Built-in modules and third-party libraries.
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Union

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries.file_io import utils
from python_libraries.settings import configuration

# Global settings.


# -----------------------------------------------------------------------------
# FILE PARSING: H5
# -----------------------------------------------------------------------------
# Get data for certain key from h5 file.
def get_data_h5(h5file, key):
    # Fetch dimensions.
    dim_x = h5file[key].attrs["dim_x"]
    dim_y = h5file[key].attrs["dim_y"]

    # Convert to NumPy.
    data = np.array(h5file[key])
    return np.reshape(data, [-1, dim_y, dim_x])


# -----------------------------------------------------------------------------
# FILE PARSING: PANDAS (CSV)
# -----------------------------------------------------------------------------
# Get data for certain key from Pandas dataframe.
def get_data_csv(df, key):
    # Fetch dimensions.
    dim_x = df["x_idx"].max() + 1
    dim_y = df["y_idx"].max() + 1

    # Convert to NumPy.
    data = df[key].to_numpy()
    return np.reshape(data, [-1, dim_y, dim_x])


# Parse 2D grid of geometry mask from csv to dict of numpy arrays, with a
# structure which is appropriate for i.e. matplotlib.pyplot.imshow().
def parse_geometry_numpy(base_path: str) -> dict:
    # Try opening H5 file.
    path = os.path.join(base_path, configuration.filenames["simulation"])
    key = "domain"
    if os.path.exists(path):
        h5file = h5py.File(path, "r")
        data = get_data_h5(h5file=h5file, key=key)
    else:
        # Try reading CSV file instead.
        path = os.path.join(base_path, "domain.csv")
        df = pd.read_csv(path)
        data = get_data_csv(df=df, key=key)

    # Storage for return data.
    out_data_dict = {}

    out_data_dict["mask"] = np.squeeze(data)

    return out_data_dict


# Parse 2D grid observable/quantity from csv to dict of numpy arrays, with a
# structure which is appropriate for i.e. matplotlib.pyplot.imshow().
def parse_field_numpy(base_path: str, field_name: str) -> dict:
    # Get the field type: vector-, scalar- or complex- field.
    field_type = configuration.grid_field_types[field_name]

    # Storage for return data.
    out_data_dict = {}

    # Path to default H5 file.
    path = os.path.join(base_path, configuration.filenames["simulation"])

    # Fetch data depending on field type.
    if field_type == "complex":
        # Storage for magnitudes and phases.
        magnitudes = []
        phases = []

        if os.path.exists(path):
            # TODO(patric): h5file is never closed!?
            h5file = h5py.File(path, "r")

            # An H5 key for the order parameter is e.g. "order_parameter_s". So we can fetch the keys which has "order_parameter" in them.
            for key in h5file.keys():
                if not field_name in key:
                    continue
                data = get_data_h5(h5file=h5file, key=key)
                magnitudes.append(np.hypot(data[0, :, :], data[1, :, :]))
                phases.append(np.arctan2(data[1, :, :], data[0, :, :]))
        else:
            # Try reading CSV file instead.
            path = os.path.join(base_path, field_name + ".csv")
            df = pd.read_csv(path)
            for symmetry in configuration.valid_symmetries:
                key_real = symmetry + "_re"
                key_imag = symmetry + "_im"
                if (key_real in df.keys()) and (key_imag in df.keys()):
                    data_real = get_data_csv(df=df, key=key_real)
                    data_imag = get_data_csv(df=df, key=key_imag)
                    magnitudes.append(np.squeeze(np.hypot(data_real, data_imag)))
                    phases.append(np.squeeze(np.arctan2(data_imag, data_real)))

        assert len(magnitudes) == len(phases)

        if len(magnitudes) == 1:
            magnitudes = magnitudes[0]
            phases = phases[0]

        out_data_dict["magnitude"] = magnitudes
        out_data_dict["phase"] = phases
    else:
        if os.path.exists(path):
            h5file = h5py.File(path, "r")
            data = get_data_h5(h5file=h5file, key=field_name)
        else:
            # Try reading CSV file instead.
            path = os.path.join(base_path, field_name + ".csv")
            df = pd.read_csv(path)
            if field_name == "current_density":
                data_x = get_data_csv(df=df, key="j_x")
                data_y = get_data_csv(df=df, key="j_y")
                data = np.concatenate((data_x, data_y), axis=0)
            elif field_name == "vector_potential":
                data_x = get_data_csv(df=df, key="a_x")
                data_y = get_data_csv(df=df, key="a_y")
                data = np.concatenate((data_x, data_y), axis=0)
            elif field_name == "magnetic_flux_density":
                data = get_data_csv(df=df, key="b_z")

        if field_type == "vector":
            # Get x- and y-component of vector field.
            out_data_dict["x"] = np.squeeze(data[0, :, :])
            out_data_dict["y"] = np.squeeze(data[1, :, :])
            # Compute the magnitude.
            out_data_dict["magnitude"] = np.hypot(data[0, :, :], data[1, :, :])
        elif field_type == "scalar":
            out_data_dict["magnitude"] = np.squeeze(data)
        else:
            print(
                f"-- ERROR! parse_grid_quantity.py: Unimplemented field type: {field_type}."
            )
            status_code = 1
            sys.exit(status_code)

    return out_data_dict


# Wrapper to parse all observables/quantities indicated by plot_parameters.
def parse_all_fields(base_path: str):
    # Data storage as dictionary.
    data_dict = {}

    # Loop over all data fields.
    for field_name in sorted(configuration.grid_field_types.keys()):
        # Check if field is to be plotted.
        data_dict[field_name] = parse_field_numpy(
            base_path=base_path,
            field_name=field_name,
        )

    # Return all data in easily parseable dictionary.
    return data_dict


# Parse LDOS data.
def parse_ldos(base_path: str) -> dict:
    # Data storage as dictionary.
    data_dict = {}

    # Try opening H5 file.
    path = os.path.join(base_path, configuration.filenames["spectroscopy"])
    key = "ldos"
    if os.path.exists(path):
        h5file = h5py.File(path, "r")
        data = get_data_h5(h5file=h5file, key=key)
    else:
        # Try reading CSV file instead.
        path = os.path.join(base_path, "ldos.csv")
        df = pd.read_csv(path)
        data = get_data_csv(df=df, key=key)
    data_dict["ldos"] = data

    # Make indices.
    dim_y, dim_x = data.shape[1:]
    vx = np.array(list(range(dim_x)))
    vy = np.array(list(range(dim_x)))
    x, y = np.meshgrid(vx, vy)
    data_dict["x"] = x
    data_dict["y"] = y

    # Return all data in easily parseable dictionary.
    return data_dict
