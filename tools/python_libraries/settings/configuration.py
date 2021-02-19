#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Various settings/config/defines tied to SuperConga."""

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
# SUPERCONGA SETTINGS
# -----------------------------------------------------------------------------
# Valid order parameter symmetries.
valid_symmetries = ["s", "dx2-y2", "dxy", "g"]
valid_norms = ["L1", "L2", "LInf"]
valid_accelerators = [
    "Adaptive-Ball",
    "AB",
    "Anderson",
    "Barzilai-Borwein",
    "BB",
    "Picard",
    "Polyak",
]
valid_data_formats = ["h5", "csv"]
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
    "internal_parameters": "internal_parameters.json",
    "post_processing": "postprocess.json",
    "results": "results.json",
    "simulation": "simulation.h5",
    "spectroscopy": "spectroscopy.h5",
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

# Default config file for plot parameters.
default_plot_config = os.path.join(
    os.path.dirname(__file__), "../plotting/plot_default.json"
)

# Mask color: color "outside" grain.
mask_color = "#c0c0c0"
