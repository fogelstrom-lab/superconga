#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Validation of command-line input parsed by argparse."""

# Built-in modules and third-party libraries.
from pathlib import Path
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
    # Check that the value is of the expected type. Or if the expected type is float, accept int as well.
    expected_type = checks["type"]
    is_float = expected_type == float
    is_expected = isinstance(value, expected_type) or (
        is_float and isinstance(value, int)
    )
    if not is_expected:
        print(f"-- ERROR! The field '{key}' must be of type {expected_type}.")
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
        if not value > low:
            print(f"-- ERROR! The field '{key}' must be larger than {low}.")
            return False

    # Check that the value is larger than, or equal to, 'low'.
    if "low_include" in checks:
        low = checks["low_include"]
        if not value >= low:
            print(
                f"-- ERROR! The field '{key}' must be larger than, or equal to, {low}."
            )
            return False

    # Check that the value is smaller than 'high'.
    if "high" in checks:
        high = checks["high"]
        if not value < high:
            print(f"-- ERROR! The field '{key}' must be smaller than {high}.")
            return False

    # Check that the value is smaller than, or equal to, 'high'.
    if "high_include" in checks:
        high = checks["high_include"]
        if not value <= high:
            print(
                f"-- ERROR! The field '{key}' must be smaller than, or equal to, {high}."
            )
            return False

    # Check that the value is not in the excluded list.
    if "excluded" in checks:
        for exc in checks["excluded"]:
            if value == exc:
                print(f"-- ERROR! The field '{key}' cannot have value {exc}.")
                return False

    # Check that the value is in some pre-defined set.
    if "one_of" in checks:
        valid_values = checks["one_of"]
        if not value in valid_values:
            print(f"-- ERROR! The field '{key}' must be one of {valid_values}.")
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
                print(f"-- ERROR! The field '{key}' must exist.")
                success = False
                continue

        value = sim_params[key]

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
    "physics": {
        "type": dict,
        "fields": {
            "temperature": {"type": float, "low": 0.0},
            "external_flux_quanta": {"type": float},
            "penetration_depth": {"type": float},
            "crystal_axes_rotation": {"type": float},
            "gauge": {"type": str, "one_of": configuration.valid_gauges},
            "charge_sign": {"type": int, "one_of": [-1, +1]},
            "order_parameter": {
                "type": dict,
                "fields": {
                    "dx2-y2": {
                        "type": dict,
                        "optional": True,
                        "fields": {
                            "critical_temperature": {"type": float, "low": 0.0},
                            "initial_phase_shift": {"type": float},
                            "initial_phase_winding": {"type": float},
                            "initial_noise_stddev": {"type": float},
                        },
                    },
                    "dxy": {
                        "type": dict,
                        "optional": True,
                        "fields": {
                            "critical_temperature": {"type": float, "low": 0.0},
                            "initial_phase_shift": {"type": float},
                            "initial_phase_winding": {"type": float},
                            "initial_noise_stddev": {"type": float},
                        },
                    },
                    "g": {
                        "type": dict,
                        "optional": True,
                        "fields": {
                            "critical_temperature": {"type": float, "low": 0.0},
                            "initial_phase_shift": {"type": float},
                            "initial_phase_winding": {"type": float},
                            "initial_noise_stddev": {"type": float},
                        },
                    },
                    "s": {
                        "type": dict,
                        "optional": True,
                        "fields": {
                            "critical_temperature": {"type": float, "low": 0.0},
                            "initial_phase_shift": {"type": float},
                            "initial_phase_winding": {"type": float},
                            "initial_noise_stddev": {"type": float},
                        },
                    },
                },
            },
            "vortex_lattice": {
                "type": dict,
                "optional": True,
                "fields": {
                    "vortices_centers_x": {"type": list, "sub": {"type": float}},
                    "vortices_centers_y": {"type": list, "sub": {"type": float}},
                    "vortices_winding_numbers": {"type": list, "sub": {"type": float}},
                },
            },
        },
    },
    "geometry": {
        "type": list,
        "sub": {
            "type": dict,
            "fields": {
                "disc": {
                    "type": dict,
                    "optional": True,
                    "fields": {
                        "add": {"type": bool},
                        "diameter": {"type": float, "low": 0.0},
                        "center_x": {"type": float},
                        "center_y": {"type": float},
                    },
                },
                "polygon": {
                    "type": dict,
                    "optional": True,
                    "fields": {
                        "add": {"type": bool},
                        "vertices_x": {"type": list, "sub": {"type": float}},
                        "vertices_y": {"type": list, "sub": {"type": float}},
                    },
                },
                "regular_polygon": {
                    "type": dict,
                    "optional": True,
                    "fields": {
                        "add": {"type": bool},
                        "side_length": {"type": float, "low": 0.0},
                        "center_x": {"type": float},
                        "center_y": {"type": float},
                        "num_edges": {"type": int, "excluded": [0, 1, 2]},
                        "rotation": {"type": float},
                    },
                },
            },
        },
    },
    "accuracy": {
        "type": dict,
        "fields": {
            "convergence_criterion": {"type": float, "low": 0.0},
            "norm": {"type": str, "one_of": configuration.valid_norms},
            "energy_cutoff": {"type": float, "low": 0.0},
            "num_fermi_momenta": {"type": int, "low": 0},
            "num_iterations_burnin": {"type": int, "low_include": 0},
            "num_iterations_max": {"type": int, "low": 0},
            "num_iterations_min": {"type": int, "low_include": 0},
            "points_per_coherence_length": {"type": float, "low": 0.0},
            "vector_potential_error": {"type": float, "low_include": 0.0},
        },
    },
    "accelerator": {
        "type": dict,
        "fields": {
            "name": {"type": str, "one_of": configuration.valid_accelerators},
            "step_size": {"type": float, "low": 0.0, "optional": True},
            "inertia": {"type": float, "low": 0.0, "optional": True},
            "rank": {"type": int, "low": 1, "optional": True},
            "step_size_increase": {"type": float, "low": 0.0, "optional": True},
            "step_size_decrease": {"type": float, "low": 0.0, "optional": True},
        },
    },
    "misc": {
        "type": dict,
        "fields": {
            "data_format": {"type": str, "one_of": configuration.valid_data_formats},
            "load_path": {"type": str},
            "save_frequency": {"type": int},
            "save_path": {"type": str},
            "verbose": {"type": bool},
            "visualize": {"type": bool},
        },
    },
}

postprocess_checks = {
    "convergence_criterion": {"type": float, "low": 0.0},
    "energy_max": {"type": float, "low_include": 0.0},
    "energy_min": {"type": float, "low_include": 0.0},
    "energy_smearing": {"type": float, "low": 0.0},
    "num_energies": {"type": int, "low": 0},
    "num_energies_per_block": {"type": int, "low": 0},
    "num_fermi_momenta": {"type": int, "low": 0},
    "num_iterations_max": {"type": int, "low": 0},
    "num_iterations_min": {"type": int, "low_include": 0},
    "load_path": {"type": str},
    "save_path": {"type": str},
    "verbose": {"type": bool},
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
        "fields": {
            "data_path": {"type": str},
            "chiral_transform": {"type": bool},
        },
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

# Validation checks for DOS plotter: a dict with the checks needed for all parameters.
ldos_plotter_checks = {
    # General plotting parameters.
    "general": {
        "type": dict,
        "fields": {"data_path": {"type": str}},
    }
}

# ------------------------------------------------------------------------------
# FRONTEND-SPECIFIC VALIDATION
# ------------------------------------------------------------------------------
# Specific checks for main runner.
def validate_runner_arguments(
    sim_params: dict,
) -> bool:
    success = True

    # Check that geometry is not empty.
    if len(sim_params["geometry"]) > 0:
        # Loop over geometry components.
        num_added_geometry_objects = 0
        for geometry_component_dict in sim_params["geometry"]:
            # Fetch type of geometry object (e.g. "disc" or "regular_polygon").
            geometry_component = list(geometry_component_dict.keys())[0]

            # There should only be one dict per geometry component.
            if len(geometry_component_dict.keys()) != 1:
                print(
                    f"-- ERROR! Each geometry component should be a single dictionary. Found the following keys: '{list(geometry_component_dict.keys())}'."
                )
                success = False

            # Count number of added objects, used later to verify that something was added.
            if geometry_component_dict[geometry_component]["add"]:
                num_added_geometry_objects += 1

            # Special checks for polygons.
            if geometry_component == "polygon":
                # Check that polygon vertices have equal length.
                num_x_vertices = len(
                    geometry_component_dict[geometry_component]["vertices_x"]
                )
                num_y_vertices = len(
                    geometry_component_dict[geometry_component]["vertices_y"]
                )
                if num_x_vertices != num_y_vertices:
                    print(
                        f"-- ERROR! The number of x- and y-vertex coordinates are not the same: {num_x_vertices} != {num_y_vertices}"
                    )
                    success = False

                min_num_vertices = 3
                if min(num_x_vertices, num_y_vertices) < min_num_vertices:
                    print(
                        f"-- ERROR! The minimum number of vertices is {min_num_vertices}."
                    )
                    success = False

                # TODO(patric): Check that polygon does not have intersecting edges.

        # Check that at least one geometry object is "added".
        if num_added_geometry_objects == 0:
            print(
                f"-- ERROR! No geometry components with 'add=true' were found. Please make sure that at least one component is added to create a valid geometry."
            )
            success = False
    else:
        print("-- ERROR! Geometry list empty. Please add some geometry objects!")
        success = False

    if len(sim_params["physics"]["order_parameter"]) > 0:
        # Normalize critical temperature so that the maximum is one.
        Tc_max = 0
        for comp in sim_params["physics"]["order_parameter"].values():
            Tc_max = max(Tc_max, comp["critical_temperature"])
        if Tc_max > 0:
            for key in sim_params["physics"]["order_parameter"].keys():
                sim_params["physics"]["order_parameter"][key][
                    "critical_temperature"
                ] /= Tc_max
        else:
            print("-- ERROR! The maximum critical temperature must be > 0!")
            success = False
    else:
        print("-- ERROR! Order-parameter list empty. Please add some components!")
        success = False

    # Check that the minimum number of iterations is smaller than the maximum.
    if not (
        sim_params["accuracy"]["num_iterations_max"]
        >= sim_params["accuracy"]["num_iterations_min"]
    ):
        print(
            "-- ERROR! 'num_iterations_max' must be larger than 'num_iterations_min'."
        )
        success = False

    # Check if vortex lattice is supplied.
    if "vortex_lattice" in sim_params["physics"]:
        # Check that lengths are equal.
        if not (
            len(sim_params["physics"]["vortex_lattice"]["vortices_centers_x"])
            == len(sim_params["physics"]["vortex_lattice"]["vortices_centers_y"])
            == len(sim_params["physics"]["vortex_lattice"]["vortices_winding_numbers"])
        ):
            print(
                "-- ERROR! The following lists must all be of equal length: 'vortices_centers_x', 'vortices_centers_y' and 'vortices_winding_numbers'."
            )
            success = False

    return success


# Specific checks for postprocess.
def validate_postprocess_arguments(
    sim_params: dict,
) -> bool:
    success = True

    # Check that the minimum number of iterations is smaller than the maximum.
    if not (sim_params["num_iterations_max"] >= sim_params["num_iterations_min"]):
        print(
            "-- ERROR! 'num_iterations_max' must be larger than 'num_iterations_min'."
        )
        success = False

    # Check that minimum value of the accelerator parameter 'p' is smallar than the maximum.
    if not sim_params["energy_max"] >= sim_params["energy_min"]:
        print("-- ERROR! 'energy_max' must be larger than, or equal to, 'energy_min'.")
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
        print(f"-- Error! Invalid data path: '{data_path}'")
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
            f"-- ERROR! No quantity was chosen for plotting. Please specify at least one quantity, using e.g. '--plot-current'. See '--help' for all options."
        )
        success = False

    return success


# Specific checks for plotter.
def validate_ldos_plotter_arguments(
    plot_params: dict,
) -> bool:
    success = True

    # Check that path is valid and contains data files.
    data_path = Path(plot_params["general"]["data_path"])

    if not utils.check_dir(data_path, verbose=False, exit_on_failure=False):
        print(f"-- ERROR! Invalid data path: '{data_path}'")
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
        if success:
            success = validate_runner_arguments(arguments_dict)
    elif frontend == "postprocess":
        # General checks against dictionary.
        success = check_dict(arguments_dict, postprocess_checks)

        # Specific checks.
        if success:
            success = validate_postprocess_arguments(arguments_dict)
    elif frontend == "plotter":
        # General checks against dictionary.
        success = check_dict(arguments_dict, plotter_checks)

        # Specific checks.
        if success:
            success = validate_plotter_arguments(arguments_dict)
    elif frontend == "ldos_plotter":
        # General checks against dictionary.
        success = check_dict(arguments_dict, ldos_plotter_checks)

        # Specific checks.
        if success:
            success = validate_ldos_plotter_arguments(arguments_dict)
    else:
        print(f"-- ERROR! validate_arguments(): Unknown frontend: '{frontend}'.")
        success = False

    return success
