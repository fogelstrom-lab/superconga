#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Various settings/config/defines tied to SuprConGPU."""

# Built-in modules and third-party libraries.
import os
import sys
import json
import pprint
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Union

# Custom libraries.

# Global values.

# -----------------------------------------------------------------------------
# SUPRCONGPU SETTINGS
# -----------------------------------------------------------------------------
# Valid order parameter symmetries.
valid_symmetries = ["s", "dx2-y2", "dxy", "g"]
symmetry_latex_format = {
    "s": r"s",
    "dx2-y2": r"d_{x^2-y^2}",
    "dxy": r"d_{xy}",
    "g": r"g",
}

# Filenames of output data.
filenames = {
    "config": "config.json",
    "convergence": "convergence.csv",
    "current_density": "current_density.csv",
    "dos": "dos.csv",
    "geometry": "domain.csv",
    "internal_parameters": "internal_parameters.json",
    "ldos": "ldos.csv",
    "magnetic_flux_density": "magnetic_flux_density.csv",
    "order_parameter": "order_parameter.csv",
    "post_processing": "postprocess.json",
    "results": "results.json",
    "superflow": "superflow.csv",
    "vector_potential": "vector_potential.csv",
}

# Components for each field type. Note that this is hardcoded in
# parse_grid_quantity.py. The only way to avoid this would be to provide a
# lookup table for which component is which column in the CSV files, and
# how to use these to e.g. get the magnitude and phase from real and imaginary
# parts.
grid_field_components = {
    "vector": ["x", "y", "magnitude"],
    "scalar": ["magnitude"],
    "complex": ["magnitude", "phase"],
}

# Fields and components for grid data objects. The key order is essential to
# field_plotting.plot_all().
grid_field_types = {
    "order_parameter": "complex",
    "magnetic_flux_density": "scalar",
    "current_density": "vector",
    "vector_potential": "vector",
}
#    "superflow": "vector",

# Name of temporary json file sent to C++/CUDA binary.
temp_config_file_name = ".tmp_config.json"

# Default config file for plot parameters.
default_plot_config = "tools/python_libraries/plotting/plot_default.json"

# Mask color: color "outside" grain.
mask_color = "#c0c0c0"