#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Various utils functions."""

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
# FUNCTIONS
# -----------------------------------------------------------------------------
# Transform a complex two-component field to the linear combination
# representation. For example, for a chiral superconductor
# Delta = Delta_1 + i*Delta_2, the transformation is to
# Delta_+/- = (Delta_1 +/- i*Delta_2)/sqrt(2), where now Delta_+/- can be
# analyzed (plotted) instead of Delta_1/2.
def chiral_transformation(data_fields_in: dict, component_index: int) -> dict:
    # Extract magnitude and phase for the two components.
    magnitude_1 = data_fields_in["magnitude"][0]
    magnitude_2 = data_fields_in["magnitude"][1]
    phase_1 = data_fields_in["phase"][0]
    phase_2 = data_fields_in["phase"][1]

    # Choose chirality for new representation:  + or - for first or second
    # component, respectively.
    if component_index == 0:
        chirality = +1
    else:
        chirality = -1

    # Compute the linear combination.
    complex_field = (
        magnitude_1 * np.exp(1.0j * phase_1)
        + chirality * magnitude_2 * 1.0j * np.exp(1.0j * phase_2)
    ) / np.sqrt(2)

    # Return the new magnitude and phase.
    return {"magnitude": np.abs(complex_field), "phase": np.angle(complex_field)}
