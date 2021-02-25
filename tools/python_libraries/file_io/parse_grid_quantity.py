#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Parse observables/quantities saved on grid lattice from SuprConGPU."""

# Built-in modules and third-party libraries.
import os
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
# FILE PARSING: PANDAS
# -----------------------------------------------------------------------------
# Parse 2D grid observable/quantity from csv to pandas DataFrame object.
def parse_field_pandas(base_path: str, field_name: str) -> pd.DataFrame:
    # Construct full file path.
    data_path = Path(base_path) / configuration.filenames[field_name]

    return utils.parse_general_csv(data_path)


# Parse 2D grid observable/quantity from csv to dict of numpy arrays, with a
# structure which is appropriate for i.e. matplotlib.pyplot.imshow().
def parse_field_numpy(base_path: str, field_name: str) -> dict:
    # Parse CSV with pandas.
    data_pandas = parse_field_pandas(base_path=base_path, field_name=field_name)

    # Convert to numpy data.
    data_numpy = data_pandas.to_numpy()

    # Get number of points per side in grid. Assuming uniform grid.
    x_coordinate_grid = data_numpy[:, 0]
    points_per_side = int(np.sqrt(len(x_coordinate_grid)))

    # Get the field type: vector-, scalar- or complex- field.
    field_type = configuration.grid_field_types[field_name]

    # Storage for return data.
    out_data_dict = {}

    # Fetch data depending on field type.
    if field_type == "vector":
        # Get x- and y-component of vector field.
        out_data_dict["x"] = data_numpy[:, 2].reshape(points_per_side, points_per_side)
        out_data_dict["y"] = data_numpy[:, 3].reshape(points_per_side, points_per_side)
        # Compute the magnitude.
        out_data_dict["magnitude"] = np.sqrt(
            out_data_dict["x"] ** 2 + out_data_dict["y"] ** 2
        )
    elif field_type == "scalar":
        out_data_dict["magnitude"] = data_numpy[:, 2].reshape(
            points_per_side, points_per_side
        )
    elif field_type == "complex":
        # Check if multiple components exist (i.e. d+is for order parameter):
        # subtract two for x- and y-coordinates, divide by two for real and
        # imaginary part for each component.
        num_components = int((len(data_pandas.columns) - 2) / 2)

        # Storage for magnitudes and phases.
        magnitudes = []
        phases = []

        # Loop over components.
        for component_idx in range(0, num_components):
            # GET real and imaginary parts.
            re_offset = 2 + 2 * component_idx
            real_part = data_numpy[:, re_offset].reshape(
                points_per_side, points_per_side
            )
            imaginary_offset = 3 + 2 * component_idx
            imaginary_part = data_numpy[:, imaginary_offset].reshape(
                points_per_side, points_per_side
            )

            # Compute absolute square.
            magnitudes.append(np.sqrt(real_part ** 2 + imaginary_part ** 2))

            # Compute phase using atan2 (see e.g. wikipedia).
            phases.append(np.arctan2(imaginary_part, real_part))

        if num_components == 1:
            magnitudes = magnitudes[0]
            phases = phases[0]

        out_data_dict["magnitude"] = magnitudes
        out_data_dict["phase"] = phases
    else:
        print(
            f"ERROR in parse_grid_quantity.py! Unimplemented field type: {field_type}."
        )
        status_code = 1
        sys.exit(status_code)

    return out_data_dict


# Wrapper to parse all observables/quantities indicated by plot_parameters.
def parse_all_fields(base_path: str, as_numpy: bool = True):
    # Data storage as dictionary.
    data_dict = {}

    # Loop over all data fields.
    for field_name in sorted(configuration.grid_field_types.keys()):
        # Check if field is to be plotted.
        if as_numpy:
            data_dict[field_name] = parse_field_numpy(
                base_path=base_path,
                field_name=field_name,
            )
        else:
            data_dict[field_name] = parse_field_pandas(
                base_path=base_path,
                field_name=field_name,
            )

    # Return all data in easily parseable dictionary.
    return data_dict


# Parse LDOS data.
def parse_ldos(base_path: str, internal_parameters: dict) -> dict:
    # Data storage as dictionary.
    data_dict = {}

    field_name = "ldos"

    # Parse CSV with pandas.
    data_pandas = parse_field_pandas(base_path=base_path, field_name=field_name)

    # Get the lattice side length.
    num_lattice_points = internal_parameters["num_lattice_points"]
    points_per_side = int(np.sqrt(num_lattice_points))

    assert points_per_side ** 2 == num_lattice_points

    # Convert to NumPy arrays.
    x = np.reshape(
        data_pandas["x_idx"].tolist(), [-1, points_per_side, points_per_side]
    )
    y = np.reshape(
        data_pandas["y_idx"].tolist(), [-1, points_per_side, points_per_side]
    )
    data_dict["ldos"] = np.reshape(
        data_pandas["ldos"].tolist(), [-1, points_per_side, points_per_side]
    )

    # The x/y-coordinates are the same for every energy. So we can extract them for a single energy.
    data_dict["x"] = x[0, :, :]
    data_dict["y"] = y[0, :, :]

    # Return all data in easily parseable dictionary.
    return data_dict
