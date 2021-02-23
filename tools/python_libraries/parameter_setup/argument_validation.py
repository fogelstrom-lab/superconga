#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Validation of command-line input parsed by argparse."""

# Built-in modules and third-party libraries.
import argparse
import json
import numpy as np
import os
from pathlib import Path
import pprint
import sys
from typing import Optional, Union

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries.file_io import utils
from python_libraries.settings import configuration

# ------------------------------------------------------------------------------
# Validation helper functions: check values and dictionaries against conditions.
# ------------------------------------------------------------------------------
# Validate command-line arguments based on check list, defined in validate().
def check_value(key: str, value: Union[bool, float, int, str], checks: dict) -> bool:
    # Check that the value is of the expected type.
    expected_type = checks["type"]
    if not isinstance(value, expected_type):
        print(f"--The field '{key}' must be of type {expected_type}.")
        return False

    # Run recursively if list or dictionary.
    if isinstance(value, list):
        success = True
        for elem in value:
            success &= check_value(key=key, value=elem, checks=checks["sub"])
        return success
    elif isinstance(value, dict):
        success = True
        success &= check_dict(sim_params=value, checks=checks["fields"])
        return success

    # Check that the value is larger than 'low'.
    if "low" in checks:
        low = checks["low"]
        if not value >= low:
            print(f"--The field '{key}' must be larger than or equal to {low}.")
            return False

    # Check that the value is smaller than 'high'.
    if "high" in checks:
        high = checks["high"]
        if not value <= high:
            print(f"--The field '{key}' must be smaller than or equal to {high}.")
            return False

    # Check that the value is not in the excluded list.
    if "excluded" in checks:
        for exc in checks["excluded"]:
            if value == exc:
                print(f"--The field '{key}' cannot have value {exc}.")
                return False

    # Check that the value is in some pre-defined set.
    if "one_of" in checks:
        valid_values = checks["one_of"]
        if not value in valid_values:
            print(f"--The field '{key}' must be one of {valid_values}.")
            return False

    # No errors.
    return True


# Check that the necessary fields are set.
def check_dict(sim_params: dict, checks: dict) -> bool:
    success = True

    # Loop over the keys in the checks so we can spot if something is missing.
    for key, value_checks in checks.items():
        if not key in sim_params:
            # Skip if the parameter is optional.
            if value_checks.get("optional", False):
                continue
            else:
                print(f"--The field '{key}' must exist.")
                success = False
                continue

        value = sim_params[key]

        # Check that the value is of the expected type.
        expected_type = value_checks["type"]
        if not isinstance(value, expected_type):
            print(f"--The field '{key}' must be of type {expected_type}.")
            success = False
            continue

        # Check if field is a dict: if so call the function again (recursively).
        if isinstance(value, dict):
            success &= check_dict(sim_params=value, checks=value_checks["fields"])
        else:
            # If not a dict then check the values.
            success &= check_value(key=key, value=value, checks=value_checks)

    return success


# ------------------------------------------------------------------------------
# VALIDATION DICTIONARIES
# ------------------------------------------------------------------------------
# Validation checks for runner: a dict with the checks needed for all parameters.
runner_checks = {
    "convergence_criterion": {"type": float, "low": 0.0},
    "crystal_axes_rotation": {"type": float},
    "data_path": {"type": str},
    "external_flux_quanta": {"type": float},
    "geometry": {
        "type": dict,
        "fields": {
            "num_outer_edges": {"type": int, "excluded": [0, 1, 2]},
            "num_inner_edges": {"type": int, "excluded": [0, 1, 2]},
            "hole_scale": {"type": float},
        },
    },
    "grain_side_length": {"type": float, "low": 0.0},
    "points_per_coherence_length": {"type": float, "low": 0.0},
    "num_fermi_momenta": {"type": int, "low": 0},
    "num_iterations_max": {"type": int, "low": 0},
    "num_iterations_min": {"type": int, "low": 0},
    "num_ozaki_energies": {"type": int, "low": 0},
    "optimizer": {
        "type": dict,
        "fields": {
            "anderson_always": {"type": bool},
            "inertia": {"type": float},
            "max_mixing": {"type": float, "low": 0.0},
            "min_mixing": {"type": float, "low": 0.0},
            "rank": {"type": int, "low": 0},
            "replace_oldest": {"type": bool},
            "step_size": {"type": float},
        },
    },
    "order_parameter": {
        "type": list,
        "sub": {
            "type": dict,
            "fields": {
                "critical_temperature": {"type": float, "low": 0.0},
                "initial_phase_shift": {"type": float},
                "initial_phase_winding": {"type": float},
                "initial_random_phase_weight": {"type": float},
                "symmetry": {"type": str, "one_of": configuration.valid_symmetries},
            },
        },
    },
    "penetration_depth": {"type": float},
    "save_data": {"type": bool},
    "save_data_interval": {"type": int},
    "temperature": {"type": float, "low": 0.0},
    "verbose": {"type": bool},
    "visualize": {"type": bool},
}

# General fields and parameters for scalar field.
plotter_checks_scalar_field = {
    "type": dict,
    "fields": {
        "plot": {"type": bool},
        "colormap": {"type": str},
    },
}

# General fields and parameters for vector field.
plotter_checks_vector_field = {
    "type": dict,
    "fields": {
        "magnitude": plotter_checks_scalar_field,
        "x": plotter_checks_scalar_field,
        "y": plotter_checks_scalar_field,
    },
}

# Validation checks for plotter: a dict with the checks needed for all parameters.
plotter_checks = {
    # General plotting parameters.
    "general": {
        "type": dict,
        "fields": {"data_path": {"type": str}, "gui": {"type": bool, "optional": True}},
    },
    # Current density plotting parameters.
    "current_density": plotter_checks_vector_field,
    # Magnetic flux density plotting parameters.
    "magnetic_flux_density": {
        "type": dict,
        "fields": {"magnitude": plotter_checks_scalar_field},
    },
    # Order parameter plotting parameters.
    "order_parameter": {
        "type": dict,
        "fields": {
            "magnitude": plotter_checks_scalar_field,
            "phase": plotter_checks_scalar_field,
        },
    },
    # Superflow plotting parameters.
    "superflow": plotter_checks_vector_field,
    # Vector potential plotting parameters.
    "vector_potential": plotter_checks_vector_field,
}

# ------------------------------------------------------------------------------
# FRONTEND-SPECIFIC VALIDATION
# ------------------------------------------------------------------------------
# Specific checks for main runner.
def validate_runner_arguments(
    sim_params: dict,
) -> bool:
    success = True

    # Normalize critical temperature so that the maximum is one.
    Tc_max = 0
    for comp in sim_params["order_parameter"]:
        Tc_max = max(Tc_max, comp["critical_temperature"])
    for i in range(len(sim_params["order_parameter"])):
        sim_params["order_parameter"][i]["critical_temperature"] /= Tc_max

    # Check that the minimum number of iterations is smaller than the maximum.
    if not (sim_params["num_iterations_max"] >= sim_params["num_iterations_min"]):
        print("--'num_iterations_max' must be larger than 'num_iterations_min'.")
        success = False

    # Check that minimum value of the optimizer parameter 'p' is smallar than the maximum.
    if (
        not sim_params["optimizer"]["max_mixing"]
        >= sim_params["optimizer"]["min_mixing"]
    ):
        print("--'max_mixing' must be larger than, or equal to, 'min_mixing'.")
        success = False

    return success


# Specific checks for plotter.
def validate_plotter_arguments(
    plot_params: dict,
) -> bool:
    success = True

    # Check that path is valid and contains data files.
    data_path = Path(plot_params["general"]["data_path"])

    if not utils.check_dir(data_path, verbose=False, exit_on_failure=False):
        print(f"Error! Invalid data path: '{data_path}'")
        success = False
        return success

    plot_current = (
        plot_params["current_density"]["magnitude"]["plot"]
        or plot_params["current_density"]["x"]["plot"]
        or plot_params["current_density"]["y"]["plot"]
    )

    plot_mfield = plot_params["magnetic_flux_density"]["magnitude"]["plot"]

    plot_op = (
        plot_params["order_parameter"]["magnitude"]["plot"]
        or plot_params["order_parameter"]["phase"]["plot"]
    )

    plot_superflow = (
        plot_params["superflow"]["magnitude"]["plot"]
        or plot_params["superflow"]["x"]["plot"]
        or plot_params["superflow"]["y"]["plot"]
    )

    plot_vector_potential = (
        plot_params["vector_potential"]["magnitude"]["plot"]
        or plot_params["vector_potential"]["x"]["plot"]
        or plot_params["vector_potential"]["y"]["plot"]
    )

    # Check that at least one field is to be plotted.
    if (
        (not plot_current)
        and (not plot_mfield)
        and (not plot_op)
        and (not plot_superflow)
        and (not plot_vector_potential)
    ):
        print(
            f"Error: No quantity was chosen for plotting. Please specify at least one quantity, using e.g. '--plot-current'. See '--help' for all options."
        )
        success = False
        return success

    # Check that parameter file exists.
    config_file = data_path / configuration.filenames["config"]
    if not utils.check_file(config_file, verbose=False, exit_on_failure=False):
        print(
            f"Error: No valid config file found in specified data path: '{config_file}'."
        )
        success = False
        return success

    # Check that internal parameter file exists.
    internal_param_file = data_path / configuration.filenames["internal_parameters"]
    if not utils.check_file(internal_param_file, verbose=False, exit_on_failure=False):
        print(
            f"Error: No valid 'internal parameters' file found in specified data path: '{internal_param_file}'."
        )
        success = False
        return success

    # Check that results file exists.
    results_file = data_path / configuration.filenames["results"]
    if not utils.check_file(results_file, verbose=False, exit_on_failure=False):
        print(
            f"Error: No valid 'scalar results' file found in specified data path: '{results_file}'."
        )
        success = False
        return success

    # Check that convergence file exists.
    convergence_file = data_path / configuration.filenames["convergence"]
    if not utils.check_file(convergence_file, verbose=False, exit_on_failure=False):
        print(
            f"Error: No valid convergence file found in specified data path: '{convergence_file}'."
        )
        success = False
        return success

    # Check that chosen observables exist. Note that the superflow is calculated from the order parameter and external vector potential!
    if plot_current:
        current_file = data_path / configuration.filenames["current_density"]
        if not utils.check_file(current_file, verbose=False, exit_on_failure=False):
            print(
                f"Error: No valid current density file found in specified data path: '{current_file}'."
            )
            success = False
            return success
    if plot_mfield:
        magnetic_field_file = (
            data_path / configuration.filenames["magnetic_flux_density"]
        )
        if not utils.check_file(
            magnetic_field_file, verbose=False, exit_on_failure=False
        ):
            print(
                f"Error: No valid magnetic field file found in specified data path: '{magnetic_field_file}'."
            )
            success = False
            return success
    if plot_op:
        op_file = data_path / configuration.filenames["order_parameter"]
        if not utils.check_file(op_file, verbose=False, exit_on_failure=False):
            print(
                f"Error: No valid order parameter file found in specified data path: '{op_file}'."
            )
            success = False
            return success
    if plot_superflow:
        superflow_file = data_path / configuration.filenames["superflow"]
        if not utils.check_file(superflow_file, verbose=False, exit_on_failure=False):
            print(
                f"Error: No valid superflow file found in specified data path: '{superflow_file}'."
            )
            success = False
            return success
    if plot_vector_potential:
        vector_potential_file = data_path / configuration.filenames["vector_potential"]
        if not utils.check_file(
            vector_potential_file, verbose=False, exit_on_failure=False
        ):
            print(
                f"Error: No valid vector potential file found in specified data path: '{vector_potential_file}'."
            )
            success = False
            return success

    return success


# ------------------------------------------------------------------------------
# VALIDATION WRAPPER
# ------------------------------------------------------------------------------
# Validate arguments based on checking criteria.
def validate_arguments(
    arguments_dict: dict,
    frontend: str = "runner",
) -> bool:
    # Sanity-check user input.
    if frontend == "runner":
        # General checks against dictionary.
        success = check_dict(arguments_dict, runner_checks)

        # Specific checks.
        success &= validate_runner_arguments(arguments_dict)
    elif frontend == "plotter":
        # General checks against dictionary.
        success = check_dict(arguments_dict, plotter_checks)

        # Specific checks.
        success &= validate_plotter_arguments(arguments_dict)
    else:
        print(f"ERROR validate_arguments(): Unknown frontend: '{frontend}'.")
        status_code = 1
        sys.exit(status_code)

    if not success:
        print(
            "ERROR encountered during user input validation [validate_arguments() of argument_validation.py]! Parameter validation failed; the run was aborted. Please see the above terminal output."
        )
        status_code = 1
        sys.exit(status_code)

    return success
