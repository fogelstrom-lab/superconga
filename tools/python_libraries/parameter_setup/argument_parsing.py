#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Convenience functions for parsing command-line input."""

# Built-in modules and third-party libraries.
import argparse
import json
import math
from pathlib import Path
import os
import sys
from typing import Optional, Union

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries import __version__
from python_libraries.file_io import utils
from python_libraries.settings import configuration

# -----------------------------------------------------------------------------
# ARGUMENT SPECIFICATION
# -----------------------------------------------------------------------------
# Add program arguments.
def make_default_parser(
    program_name: str,
    program_description: str,
    frontend: str = "runner",
):
    parser = argparse.ArgumentParser(
        prog=program_name,
        description=program_description,
    )

    if frontend == "runner":
        add_runner_arguments(parser)
    elif frontend == "postprocess":
        add_postprocess_arguments(parser)
    elif frontend == "plotter":
        add_plotter_arguments(parser)
    elif frontend == "ldos_plotter":
        add_ldos_plotter_arguments(parser)
    else:
        print(f"-- ERROR! make_default_parser(): Unknown frontend: '{frontend}'.")
        status_code = 1
        sys.exit(status_code)

    # Return argument parser.
    return parser


# -----------------------------------------------------------------------------
# ARGUMENT SPECIFICATION: RUNNER
# -----------------------------------------------------------------------------
# Add arguments specific for the runner.
def add_runner_arguments(parser) -> None:
    # Config file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the simulation settings.",
    )

    # C++ binary and precision.
    group_cpp = parser.add_mutually_exclusive_group()
    group_cpp.add_argument(
        "--cpp-path",
        type=str,
        default=None,
        help="Path to the C++ binary.",
    )
    group_cpp.add_argument(
        "-P",
        "--precision",
        type=int,
        choices=[32, 64],
        default=64,
        help="The floating point precision. The C++ binary 'build/release/bin/suprcongpu_fXX', where XX is the precision, relative to the supercongpu.py script will be used. Note that 32 bit precision is less accurate.",
    )

    # Version number.
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )

    # Dump config parameters.
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print parameters and exit.",
    )

    # =============== PHYSICS =============== #
    physics = parser.add_argument_group(
        title="Physics",
        description="The order-parameter components take four arguments. The arguments are critival temperature, initial phase shift (in units of 2pi), initial phase winding (in units of 2pi), and initial random phase weight. The critical temperatures will be scaled so that the maximum is Tc = 1.",
    )

    # Temperature.
    physics.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Temperature in units of the critical temperature T_c.",
    )

    # External flux quanta.
    physics.add_argument(
        "-B",
        "--external-flux-quanta",
        type=float,
        default=None,
        help="External magnetic field, perpendicular to sample, in units of flux quanta. For non-solenoid field, it is distributed uniformly over the sample.",
    )

    # Penetration depth.
    physics.add_argument(
        "-k",
        "--penetration-depth",
        type=float,
        default=None,
        help="The London penetration depth, expressed in coherence lengths (i.e. the Ginzburg-Landau coefficient). If the value is less than zero, the induced vector-potential will not be computed. In other words, the penetration depth will be infinite.",
    )

    # Crystal axes rotation.
    physics.add_argument(
        "-r",
        "--crystal-axes-rotation",
        type=float,
        default=None,
        help="Rotation between grain and crystal ab-axes, in units of 2pi.",
    )

    # Gauge.
    gauge = physics.add_mutually_exclusive_group()
    gauge.add_argument(
        "--gauge-radial",
        action="store_true",
        help="Use the radial gauge, A = 0.5 * B * (-y, x).",
    )
    gauge.add_argument(
        "--gauge-solenoid",
        action="store_true",
        help="Use the solenoid gauge, A = 0.5 * B * Area (-y, x) / (x^2 + y^2).",
    )

    # =============== ORDER PARAMETER =============== #

    # D_{x^2-y^2}-wave.
    dx2_y2_wave = physics.add_mutually_exclusive_group()
    dx2_y2_wave.add_argument(
        "--no-dx2-y2-wave",
        action="store_true",
        help="No D_(x^2-y^2)-wave order-parameter component.",
    )
    dx2_y2_wave.add_argument(
        "--dx2-y2-wave",
        metavar=(
            "critical_temperature",
            "initial_phase_shift",
            "initial_phase_winding",
            "initial_random_phase_weight",
        ),
        type=float,
        default=None,
        nargs=4,
        help="D_(x^2-y^2)-wave order-parameter component.",
    )

    # D_{xy}-wave.
    dxy_wave = physics.add_mutually_exclusive_group()
    dxy_wave.add_argument(
        "--no-dxy-wave",
        action="store_true",
        help="No D_(xy)-wave order-parameter component.",
    )
    dxy_wave.add_argument(
        "--dxy-wave",
        metavar=(
            "critical_temperature",
            "initial_phase_shift",
            "initial_phase_winding",
            "initial_random_phase_weight",
        ),
        type=float,
        default=None,
        nargs=4,
        help="D_(xy)-wave order-parameter component.",
    )

    # G-wave.
    g_wave = physics.add_mutually_exclusive_group()
    g_wave.add_argument(
        "--no-g-wave",
        action="store_true",
        help="No G-wave order-parameter component.",
    )
    g_wave.add_argument(
        "--g-wave",
        metavar=(
            "critical_temperature",
            "initial_phase_shift",
            "initial_phase_winding",
            "initial_random_phase_weight",
        ),
        type=float,
        default=None,
        nargs=4,
        help="G-wave order-parameter component.",
    )

    # S-wave.
    s_wave = physics.add_mutually_exclusive_group()
    s_wave.add_argument(
        "--no-s-wave",
        action="store_true",
        help="No S-wave order-parameter component.",
    )
    s_wave.add_argument(
        "--s-wave",
        metavar=(
            "critical_temperature",
            "initial_phase_shift",
            "initial_phase_winding",
            "initial_random_phase_weight",
        ),
        type=float,
        default=None,
        nargs=4,
        help="S-wave order-parameter component.",
    )

    # =============== GEOMETRY =============== #
    geometry = parser.add_argument_group(
        title="Geometry",
        description="Note that when specifying the geometry via the CLI all components set here will be added before any are removed. If the desired geometry depends on the order of the components, use the config file instead. Furthermore, general polygons, with arbitrary vertex coordinates, is only supported in the config file.",
    )

    # Ignore geometry in config file.
    geometry.add_argument(
        "--geometry-clear",
        action="store_true",
        help="Ignore the geometry specified in the config file.",
    )

    # Add disc.
    geometry.add_argument(
        "--add-disc",
        metavar=("diameter", "center_x", "center_y"),
        type=float,
        action="append",
        nargs=3,
        default=None,
        help="Add disc to geometry. The arguments are diameter, center x, and center y.",
    )

    # Remove disc.
    geometry.add_argument(
        "--remove-disc",
        metavar=("diameter", "center_x", "center_y"),
        type=float,
        action="append",
        nargs=3,
        default=None,
        help="Remove disc from geometry. The arguments are diameter, center x, and center y.",
    )

    # Add regular polygon.
    geometry.add_argument(
        "--add-regular-polygon",
        metavar=("side_length", "center_x", "center_y", "num_edges", "rotation"),
        type=float,
        action="append",
        nargs=5,
        default=None,
        help="Add regular polygon to geometry. The arguments are side length, center x, center y, the number of edges, and the rotation (in units of 2pi).",
    )

    # Remove regular polygon.
    geometry.add_argument(
        "--remove-regular-polygon",
        metavar=("side_length", "center_x", "center_y", "num_edges", "rotation"),
        type=float,
        action="append",
        nargs=5,
        default=None,
        help="Remove regular polygon from geometry. The arguments are side length, center x, center y, the number of edges, and the rotation (in units of 2pi).",
    )

    # =============== ACCURACY =============== #
    accuracy = parser.add_argument_group(title="Accuracy")

    # Convergence criterion.
    accuracy.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criterion for self-consistency.",
    )

    # Num Ozaki energies.
    accuracy.add_argument(
        "-e",
        "--energy-cutoff",
        type=float,
        default=None,
        help="The energy cutoff (in units of 2*pi*kB*Tc).",
    )

    # Num Fermi momenta.
    accuracy.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Recommended minimum: 100.",
    )

    # Num iterations max.
    accuracy.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run.",
    )

    # Num iterations min.
    accuracy.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run.",
    )

    # Lattice side length.
    accuracy.add_argument(
        "-N",
        "--points-per-coherence-length",
        type=float,
        default=None,
        help="Number of lattice points per coherence length. Recommended value 10 (or larger).",
    )

    # Green's function accuracy.
    accuracy.add_argument(
        "--vp-error",
        type=float,
        default=None,
        help="The maximum error (relative Frobenious norm) of the Green's function used when computing the induced vector-potential.",
    )

    # =============== OPTIMIZER =============== #
    optimizer = parser.add_argument_group(title="Optimizer")

    optimizer.add_argument(
        "--opt-step-size",
        type=float,
        default=None,
        help="The step size in the optimizer.",
    )
    optimizer.add_argument(
        "--opt-inertia",
        type=float,
        default=None,
        help="The inertia of the optimizer.",
    )
    group_optimiser_adaptive = optimizer.add_mutually_exclusive_group()
    group_optimiser_adaptive.add_argument(
        "--opt-adaptive",
        action="store_true",
        help="Change the step size and inertia adaptively each iteration.",
    )
    group_optimiser_adaptive.add_argument(
        "--opt-no-adaptive",
        action="store_true",
        help="Do not change the step size and inertia adaptively each iteration.",
    )
    optimizer.add_argument(
        "--opt-rank",
        type=int,
        default=None,
        help="The maximal rank of the system of equations in the Anderson step, i.e. the maximum number of stored steps.",
    )
    optimizer.add_argument(
        "--opt-min-mixing",
        type=float,
        default=None,
        help="The minimal value of the mixing parameter used in Anderson steps. Only applicable if rank > 1.",
    )
    optimizer.add_argument(
        "--opt-max-mixing",
        type=float,
        default=None,
        help="The maximal value of the mixing parameter used in Anderson steps. Only applicable if rank > 1.",
    )
    group_optimiser_replace = optimizer.add_mutually_exclusive_group()
    group_optimiser_replace.add_argument(
        "--opt-replace-worst",
        action="store_true",
        help="Replace the worst stored step in the optimizer, when the number of stored steps is equal to the rank. Only applicable if rank > 1.",
    )
    group_optimiser_replace.add_argument(
        "--opt-replace-oldest",
        action="store_true",
        help="Replace the oldest stored step in the optimizer, when the number of stored steps is equal to the rank. Only applicable if rank > 1.",
    )
    group_optimiser_anderson = optimizer.add_mutually_exclusive_group()
    group_optimiser_anderson.add_argument(
        "--opt-anderson-always",
        action="store_true",
        help="Take Anderson steps all the time. Only applicable if rank > 1.",
    )
    group_optimiser_anderson.add_argument(
        "--opt-anderson-full",
        action="store_true",
        help="Take Anderson steps only when the storage is full. Only applicable if rank > 1.",
    )

    # =============== MISC =============== #
    misc = parser.add_argument_group(title="Misc")

    # Verbose.
    group_verbose = misc.add_mutually_exclusive_group()
    group_verbose.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print iteration status and other output to terminal.",
    )
    group_verbose.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print iteration status and other output to terminal.",
    )

    # Save data.
    misc.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Directory where the data will be saved. If files already exist, they will be overwritten. The exception is the file convergence.csv, which will have data appended to it.",
    )
    misc.add_argument(
        "-F",
        "--save-frequency",
        type=int,
        default=None,
        help="How frequently to save data (every Nth iteration). Zero means not saving data at all, and a negative value means only saving at the end.",
    )

    # Load data.
    group_load = misc.add_mutually_exclusive_group()
    group_load.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Directory containing data. The simulation will resume from this data.",
    )
    group_load.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume the simulation from the data in the '--save-path' directory.",
    )

    # Visualize.
    group_visualize = misc.add_mutually_exclusive_group()
    group_visualize.add_argument(
        "--visualize",
        action="store_true",
        help="Real-time plotting. The units are E_0=2*pi*kB*Tc, j_d = E_0*|e|*v_F*N_F, A_0 = Phi_0 / (pi*xi_0), and B_0 = \Phi_0/(pi*xi^2_0), where xi_0 = hbar*vF/E_0, and Phi_0 = pi*hbar/|e|.",
    )
    group_visualize.add_argument(
        "--no-visualize", action="store_true", help="No real-time plotting."
    )


# -----------------------------------------------------------------------------
# ARGUMENT SPECIFICATION: POSTPROCESS
# -----------------------------------------------------------------------------
# Add arguments specific for the postprocess.
def add_postprocess_arguments(parser) -> None:
    # Config file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the settings for calculating the density of states.",
    )

    # C++ binary and precision.
    group_cpp = parser.add_mutually_exclusive_group()
    group_cpp.add_argument(
        "--cpp-path",
        type=str,
        default=None,
        help="Path to the C++ binary.",
    )
    group_cpp.add_argument(
        "-P",
        "--precision",
        type=int,
        choices=[32, 64],
        default=64,
        help="The floating point precision. The C++ binary 'build/release/bin/postprocess_fXX', where XX is the precision, relative to the supercongpu.py script will be used. Note that 32 bit precision is less accurate.",
    )

    # Dump config parameters.
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print parameters and exit.",
    )

    # Convergence criterion.
    parser.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criterion for self-consistency.",
    )

    # Save data.
    parser.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Directory where the data will be saved. If files already exist, they will be overwritten. If not set the '--load-path' directory will be used.",
    )

    # Load data.
    parser.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Directory containing the data of a converged simulation.",
    )

    # Energy max.
    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="The maximum energy to compute density of states at.",
    )

    # Energy min.
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="The minimum energy to compute density of states at.",
    )

    # Energy smearing.
    parser.add_argument(
        "--energy-smearing",
        type=float,
        default=None,
        help="The smearing (broadening) factor to the energy in the propagator, used to compute the density of states.",
    )

    # Num energies.
    parser.add_argument(
        "-e",
        "--num-energies",
        type=int,
        default=None,
        help="Number of energies to compute the density of states at, between (and including) the min/max energies.",
    )

    # Num energies per block.
    parser.add_argument(
        "-E",
        "--num-energies-per-block",
        type=int,
        default=None,
        help="Number of energies per block. The density of states calculation is memory intensive, which might lead to the GPU running out of memory. To solve this, the problem is divided into multiple chunks (blocks) calculated in serial.",
    )

    # Num iterations max.
    parser.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run.",
    )

    # Num iterations min.
    parser.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run.",
    )

    # Num Fermi momenta.
    parser.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Recommended minimum: 700.",
    )

    # Verbose.
    group_verbose = parser.add_mutually_exclusive_group()
    group_verbose.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print iteration status and other output to terminal.",
    )
    group_verbose.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print iteration status and other output to terminal.",
    )

    # Version number.
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )


# -----------------------------------------------------------------------------
# ARGUMENT SPECIFICATION: PLOTTER
# -----------------------------------------------------------------------------
# Add arguments specific for the plotter.
def add_plotter_arguments(parser) -> None:
    # Path to data.
    parser.add_argument(
        "-L",
        "--load-path",
        dest="data_path",
        type=str,
        default=None,
        help="Path to data folder.",
    )

    # Plotting configuration file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the plot settings.",
    )

    # Dump config parameters.
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print parameters and exit.",
    )

    # Use chiral transformation when plotting a two-component superconductor.
    group_chiral_transform = parser.add_mutually_exclusive_group()
    group_chiral_transform.add_argument(
        "--chiral-transform",
        action="store_true",
        help="Use chiral transformation (d+id' becomes d_+ + d_-) when plotting a two-component superconductor.",
    )
    group_chiral_transform.add_argument(
        "--no-chiral-transform",
        action="store_true",
        help="Do not use chiral transformation (d+id' becomes d_+ + d_-) when plotting a two-component superconductor.",
    )

    # Plot order parameter.
    parser.add_argument(
        "--plot-order-parameter",
        action="store_true",
        default=None,
        help="Plot order parameter (amplitude and phase).",
    )

    # Plot order parameter.
    parser.add_argument(
        "--plot-order-parameter-magnitude",
        action="store_true",
        default=None,
        help="Plot order parameter magnitude.",
    )

    # Plot order parameter.
    parser.add_argument(
        "--plot-order-parameter-phase",
        action="store_true",
        default=None,
        help="Plot order parameter phase.",
    )

    # Plot current.
    parser.add_argument(
        "--plot-current",
        action="store_true",
        default=None,
        help="Plot current (magnitude, x- and y-component).",
    )

    # Plot current.
    parser.add_argument(
        "--plot-current-magnitude",
        action="store_true",
        default=None,
        help="Plot current magnitude.",
    )

    # Plot current.
    parser.add_argument(
        "--plot-current-x",
        action="store_true",
        default=None,
        help="Plot current x-component.",
    )

    # Plot current.
    parser.add_argument(
        "--plot-current-y",
        action="store_true",
        default=None,
        help="Plot current y-component.",
    )

    # Plot superflows.
    parser.add_argument(
        "--plot-superflow",
        action="store_true",
        default=None,
        help="Plot superflow (magnitude, x- and y-component).",
    )

    # Plot superflows.
    parser.add_argument(
        "--plot-superflow-magnitude",
        action="store_true",
        default=None,
        help="Plot superflow magnitude.",
    )

    # Plot superflows.
    parser.add_argument(
        "--plot-superflow-x",
        action="store_true",
        default=None,
        help="Plot superflow x-component.",
    )

    # Plot superflows.
    parser.add_argument(
        "--plot-superflow-y",
        action="store_true",
        default=None,
        help="Plot superflow y-component.",
    )

    # Plot induced magnetic flux density.
    parser.add_argument(
        "--plot-mfield",
        action="store_true",
        default=None,
        help="Plot induced magnetic flux density.",
    )

    # Plot induced vector potential.
    parser.add_argument(
        "--plot-vector-potential",
        action="store_true",
        default=None,
        help="Plot induced vector potential (magnitude, x- and y-component).",
    )

    # Plot induced vector potential.
    parser.add_argument(
        "--plot-vector-potential-magnitude",
        action="store_true",
        default=None,
        help="Plot induced vector potential magnitude.",
    )

    # Plot induced vector potential.
    parser.add_argument(
        "--plot-vector-potential-x",
        action="store_true",
        default=None,
        help="Plot induced vector potential x-component.",
    )

    # Plot induced vector potential.
    parser.add_argument(
        "--plot-vector-potential-y",
        action="store_true",
        default=None,
        help="Plot induced vector potential y-component.",
    )

    # Version number.
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )


# -----------------------------------------------------------------------------
# ARGUMENT SPECIFICATION: DENSITY OF STATES PLOTTER
# -----------------------------------------------------------------------------
# Add arguments specific for the LDOS plotter.
def add_ldos_plotter_arguments(parser) -> None:
    # Path to data.
    parser.add_argument(
        "-L",
        "--load-path",
        dest="data_path",
        type=str,
        default=None,
        help="Path to folder with density of states data.",
    )

    # Plotting configuration file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the plot settings.",
    )

    # Dump config parameters.
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print parameters and exit.",
    )

    # Version number.
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
# Set value from command-line argument.
def set_value(
    simparams: dict,
    key: str,
    value: Union[dict, int, float, list],
    group: Optional[str] = None,
) -> None:
    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        if group is None:
            simparams[key] = value
        else:
            if isinstance(value, list):
                if group in simparams:
                    # The group, e.g. 'order_parameter' already exists.
                    # Check that the lengths of the lists agree.
                    assert isinstance(simparams[group], list)
                    assert len(value) == len(simparams[group])
                    for i, elem in enumerate(value):
                        # TODO(niclas): Add dummy value so one can only specify e.g. the second critical temperature.
                        simparams[group][i][key] = elem
                else:
                    # The group, e.g. 'order_parameter' does not exist, so lets create a dict for every element of 'value'.
                    tmp = []
                    for elem in value:
                        tmp.append({key: elem})
                    simparams[group] = tmp
            else:
                simparams[group][key] = value


# Set value from command-line flag (on/off for true/false).
def set_flag(
    simparams: dict,
    key: str,
    on: bool,
    off: bool,
    group: Optional[str] = None,
) -> None:
    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        if group is None:
            simparams[key] = on
        else:
            simparams[group][key] = on


# -----------------------------------------------------------------------------
# COMMAND-LINE ARGUMENT PARSING
# -----------------------------------------------------------------------------
# Override json dictionary with values from command line through argparse.
def set_arguments_from_cli(
    sim_params,
    args,
    frontend: str = "runner",
) -> None:

    # Update parameters from command-line input.
    if frontend == "runner":
        set_runner_arguments_from_cli(sim_params, args)
    elif frontend == "postprocess":
        set_postprocess_arguments_from_cli(sim_params, args)
    elif frontend == "plotter":
        set_plotter_arguments_from_cli(sim_params, args)
    elif frontend == "ldos_plotter":
        set_ldos_plotter_arguments_from_cli(sim_params, args)
    else:
        print(f"-- ERROR! set_from_cli(): Unknown frontend: '{frontend}'.")
        status_code = 1
        sys.exit(status_code)


# Main runner: Override json dictionary with command-line input.
def set_runner_arguments_from_cli(
    sim_params,
    args,
) -> None:
    # ================== PHYSICS ================== #
    group = "physics"
    set_value(sim_params, "temperature", args.temperature, group)
    set_value(sim_params, "external_flux_quanta", args.external_flux_quanta, group)
    set_value(sim_params, "penetration_depth", args.penetration_depth, group)
    set_value(sim_params, "crystal_axes_rotation", args.crystal_axes_rotation, group)
    set_flag(
        sim_params, "gauge_solenoid", args.gauge_solenoid, args.gauge_radial, group
    )

    # ================== PHYSICS: ORDER PARAMETER ================== #
    component = "dx2-y2"
    if args.no_dx2_y2_wave and (
        component in sim_params["physics"]["order_parameter"].keys()
    ):
        del sim_params["physics"]["order_parameter"][component]
    elif args.dx2_y2_wave is not None:
        Tc, phase_shift, phase_winding, random_phase = args.dx2_y2_wave
        sim_params["physics"]["order_parameter"][component] = {
            "critical_temperature": Tc,
            "initial_phase_shift": phase_shift,
            "initial_phase_winding": phase_winding,
            "initial_random_phase_weight": random_phase,
        }

    component = "dxy"
    if args.no_dxy_wave and (
        component in sim_params["physics"]["order_parameter"].keys()
    ):
        del sim_params["physics"]["order_parameter"][component]
    elif args.dxy_wave is not None:
        Tc, phase_shift, phase_winding, random_phase = args.dxy_wave
        sim_params["physics"]["order_parameter"][component] = {
            "critical_temperature": Tc,
            "initial_phase_shift": phase_shift,
            "initial_phase_winding": phase_winding,
            "initial_random_phase_weight": random_phase,
        }

    component = "g"
    if args.no_g_wave and (
        component in sim_params["physics"]["order_parameter"].keys()
    ):
        del sim_params["physics"]["order_parameter"][component]
    elif args.g_wave:
        Tc, phase_shift, phase_winding, random_phase = args.g_wave
        sim_params["physics"]["order_parameter"][component] = {
            "critical_temperature": Tc,
            "initial_phase_shift": phase_shift,
            "initial_phase_winding": phase_winding,
            "initial_random_phase_weight": random_phase,
        }

    component = "s"
    if args.no_s_wave and (
        component in sim_params["physics"]["order_parameter"].keys()
    ):
        del sim_params["physics"]["order_parameter"][component]
    elif args.s_wave:
        Tc, phase_shift, phase_winding, random_phase = args.s_wave
        sim_params["physics"]["order_parameter"][component] = {
            "critical_temperature": Tc,
            "initial_phase_shift": phase_shift,
            "initial_phase_winding": phase_winding,
            "initial_random_phase_weight": random_phase,
        }

    # ================== GEOMETRY ================== #
    group = "geometry"
    if args.geometry_clear:
        sim_params[group] = []

    # Add discs.
    if args.add_disc is not None:
        for disc in args.add_disc:
            diameter, center_x, center_y = disc
            sim_params[group].append(
                {
                    "disc": {
                        "add": True,
                        "diameter": diameter,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                }
            )

    # Add regular polygons.
    if args.add_regular_polygon is not None:
        for regular_polygon in args.add_regular_polygon:
            side_length, center_x, center_y, num_edges, rotation = regular_polygon
            sim_params[group].append(
                {
                    "regular_polygon": {
                        "add": True,
                        "side_length": side_length,
                        "center_x": center_x,
                        "center_y": center_y,
                        "num_edges": int(num_edges),
                        "rotation": rotation,
                    }
                }
            )

    # Remove discs.
    if args.remove_disc is not None:
        for disc in args.remove_disc:
            diameter, center_x, center_y = disc
            sim_params[group].append(
                {
                    "disc": {
                        "add": False,
                        "diameter": diameter,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                }
            )

    # Remove regular polygons.
    if args.remove_regular_polygon is not None:
        for regular_polygon in args.remove_regular_polygon:
            side_length, center_x, center_y, num_edges, rotation = regular_polygon
            sim_params[group].append(
                {
                    "regular_polygon": {
                        "add": False,
                        "side_length": side_length,
                        "center_x": center_x,
                        "center_y": center_y,
                        "num_edges": int(num_edges),
                        "rotation": rotation,
                    }
                }
            )

    # ================== ACCURACY ================== #
    group = "accuracy"
    set_value(sim_params, "convergence_criterion", args.convergence_criterion, group)
    set_value(sim_params, "energy_cutoff", args.energy_cutoff, group)
    set_value(sim_params, "num_fermi_momenta", args.num_fermi_momenta, group)
    set_value(sim_params, "num_iterations_max", args.num_iterations_max, group)
    set_value(sim_params, "num_iterations_min", args.num_iterations_min, group)
    set_value(
        sim_params,
        "points_per_coherence_length",
        args.points_per_coherence_length,
        group,
    )
    set_value(sim_params, "vector_potential_error", args.vp_error, group)

    # ================== OPTIMIZER ================== #
    group = "optimizer"
    set_value(sim_params, "step_size", args.opt_step_size, group)
    set_value(sim_params, "inertia", args.opt_inertia, group)
    set_flag(
        sim_params,
        "adaptive",
        args.opt_adaptive,
        args.opt_no_adaptive,
        group,
    )
    set_value(sim_params, "rank", args.opt_rank, group)
    set_value(sim_params, "min_mixing", args.opt_min_mixing, group)
    set_value(sim_params, "max_mixing", args.opt_max_mixing, group)
    set_flag(
        sim_params,
        "replace_oldest",
        args.opt_replace_oldest,
        args.opt_replace_worst,
        group,
    )
    set_flag(
        sim_params,
        "anderson_always",
        args.opt_anderson_always,
        args.opt_anderson_full,
        group,
    )

    # ================== MISC ================== #
    group = "misc"
    set_value(sim_params, "save_path", args.save_path, group)
    set_value(sim_params, "save_frequency", args.save_frequency, group)
    if args.do_continue:
        load_path = sim_params["misc"]["save_path"]
    else:
        load_path = args.load_path
    set_value(sim_params, "load_path", load_path, group)
    set_flag(sim_params, "verbose", args.verbose, args.quiet, group)
    set_flag(sim_params, "visualize", args.visualize, args.no_visualize, group)


# Postprocess: Override json dictionary with command-line input.
def set_postprocess_arguments_from_cli(
    sim_params,
    args,
) -> None:
    # Override values from the command line.
    set_value(sim_params, "convergence_criterion", args.convergence_criterion)
    set_value(sim_params, "energy_max", args.energy_max)
    set_value(sim_params, "energy_min", args.energy_min)
    set_value(sim_params, "energy_smearing", args.energy_smearing)
    set_value(sim_params, "num_energies", args.num_energies)
    set_value(sim_params, "num_energies_per_block", args.num_energies_per_block)
    set_value(sim_params, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(sim_params, "num_iterations_max", args.num_iterations_max)
    set_value(sim_params, "num_iterations_min", args.num_iterations_min)
    set_value(sim_params, "load_path", args.load_path)
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = sim_params["load_path"]
    set_value(sim_params, "save_path", save_path)
    set_flag(sim_params, "verbose", args.verbose, args.quiet)


# Plotter: Override json dictionary with command-line input.
def set_plotter_arguments_from_cli(
    plot_params,
    args,
) -> None:
    # General plot settings.
    group = "general"
    set_value(plot_params, "data_path", args.data_path, group)
    set_flag(
        plot_params,
        "chiral_transform",
        args.chiral_transform,
        args.no_chiral_transform,
        group,
    )

    # Control which fields to plot. All components can be turned on (not off)
    # with the base parameter, e.g. 'plot-current' plots all current
    # components. If it is off, the components can be plotted individually by
    # 'plot-current-x' etc.

    # Current density plot settings.
    if args.plot_current != None:
        # Magnitude: current density.
        group = "magnitude"
        set_flag(
            plot_params["current_density"],
            "plot",
            args.plot_current,
            (not args.plot_current),
            group,
        )
        if args.plot_current_magnitude != None:
            set_flag(
                plot_params["current_density"],
                "plot",
                args.plot_current_magnitude,
                (not args.plot_current_magnitude),
                group,
            )

        # x-component: current density.
        group = "x"
        set_flag(
            plot_params["current_density"],
            "plot",
            args.plot_current,
            (not args.plot_current),
            group,
        )
        if args.plot_current_x != None:
            set_flag(
                plot_params["current_density"],
                "plot",
                args.plot_current_x,
                (not args.plot_current_x),
                group,
            )

        # y-component: current density.
        group = "y"
        set_flag(
            plot_params["current_density"],
            "plot",
            args.plot_current,
            (not args.plot_current),
            group,
        )
        if args.plot_current_y != None:
            set_flag(
                plot_params["current_density"],
                "plot",
                args.plot_current_y,
                (not args.plot_current_y),
                group,
            )

    # Magnetic flux density plot settings.
    if args.plot_mfield != None:
        group = "magnitude"
        set_flag(
            plot_params["magnetic_flux_density"],
            "plot",
            args.plot_mfield,
            (not args.plot_mfield),
            group,
        )

    # Order parameter plot settings.
    if args.plot_order_parameter != None:
        # Magnitude: order parameter.
        group = "magnitude"
        set_flag(
            plot_params["order_parameter"],
            "plot",
            args.plot_order_parameter,
            (not args.plot_order_parameter),
            group,
        )
        if args.plot_order_parameter_magnitude != None:
            set_flag(
                plot_params["order_parameter"],
                "plot",
                args.plot_order_parameter_magnitude,
                (not args.plot_order_parameter_magnitude),
                group,
            )

        # Phase: order parameter.
        group = "phase"
        set_flag(
            plot_params["order_parameter"],
            "plot",
            args.plot_order_parameter,
            (not args.plot_order_parameter),
            group,
        )
        if args.plot_order_parameter_phase != None:
            set_flag(
                plot_params["order_parameter"],
                "plot",
                args.plot_order_parameter_phase,
                (not args.plot_order_parameter_phase),
                group,
            )

    # Superflow plot settings.
    group = "superflow"
    if args.plot_superflow != None:
        # Magnitude: superflow.
        group = "magnitude"
        set_flag(
            plot_params["superflow"],
            "plot",
            args.plot_superflow,
            (not args.plot_superflow),
            group,
        )
        if args.plot_superflow_magnitude != None:
            set_flag(
                plot_params["superflow"],
                "plot",
                args.plot_superflow_magnitude,
                (not args.plot_superflow_magnitude),
                group,
            )

        # x-component: superflow.
        group = "x"
        set_flag(
            plot_params["superflow"],
            "plot",
            args.plot_superflow,
            (not args.plot_superflow),
            group,
        )
        if args.plot_superflow_x != None:
            set_flag(
                plot_params["superflow"],
                "plot",
                args.plot_superflow_x,
                (not args.plot_superflow_x),
                group,
            )

        # y-component: superflow.
        group = "y"
        set_flag(
            plot_params["superflow"],
            "plot",
            args.plot_superflow,
            (not args.plot_superflow),
            group,
        )
        if args.plot_superflow_y != None:
            set_flag(
                plot_params["superflow"],
                "plot",
                args.plot_superflow_y,
                (not args.plot_superflow_y),
                group,
            )

    # Vector potential plot settings.
    group = "vector_potential"
    if args.plot_vector_potential != None:
        # Magnitude: vector potential.
        group = "magnitude"
        set_flag(
            plot_params["vector_potential"],
            "plot",
            args.plot_vector_potential,
            (not args.plot_vector_potential),
            group,
        )
        if args.plot_vector_potential_magnitude != None:
            set_flag(
                plot_params["vector_potential"],
                "plot",
                args.plot_vector_potential_magnitude,
                (not args.plot_vector_potential_magnitude),
                group,
            )

        # x-component: vector potential.
        group = "x"
        set_flag(
            plot_params["vector_potential"],
            "plot",
            args.plot_vector_potential,
            (not args.plot_vector_potential),
            group,
        )
        if args.plot_vector_potential_x != None:
            set_flag(
                plot_params["vector_potential"],
                "plot",
                args.plot_vector_potential_x,
                (not args.plot_vector_potential_x),
                group,
            )

        # y-component: vector potential.
        group = "y"
        set_flag(
            plot_params["vector_potential"],
            "plot",
            args.plot_vector_potential,
            (not args.plot_vector_potential),
            group,
        )
        if args.plot_vector_potential_y != None:
            set_flag(
                plot_params["vector_potential"],
                "plot",
                args.plot_vector_potential_y,
                (not args.plot_vector_potential_y),
                group,
            )

    # Add convenient aliases.
    plot_current = (
        plot_params["current_density"]["magnitude"]["plot"]
        or plot_params["current_density"]["x"]["plot"]
        or plot_params["current_density"]["y"]["plot"]
    )
    plot_params["current_density"]["plot"] = plot_current

    plot_params["magnetic_flux_density"]["plot"] = plot_params["magnetic_flux_density"][
        "magnitude"
    ]["plot"]

    plot_order_parameter = (
        plot_params["order_parameter"]["magnitude"]["plot"]
        or plot_params["order_parameter"]["phase"]["plot"]
    )
    plot_params["order_parameter"]["plot"] = plot_order_parameter

    plot_superflow = (
        plot_params["superflow"]["magnitude"]["plot"]
        or plot_params["superflow"]["x"]["plot"]
        or plot_params["superflow"]["y"]["plot"]
    )
    plot_params["superflow"]["plot"] = plot_superflow

    plot_vector_potential = (
        plot_params["vector_potential"]["magnitude"]["plot"]
        or plot_params["vector_potential"]["x"]["plot"]
        or plot_params["vector_potential"]["y"]["plot"]
    )
    plot_params["vector_potential"]["plot"] = plot_vector_potential


# LDOS Plotter: Override json dictionary with command-line input.
def set_ldos_plotter_arguments_from_cli(
    plot_params,
    args,
) -> None:
    # General plot settings.
    group = "general"
    set_value(plot_params, "data_path", args.data_path, group)


# -----------------------------------------------------------------------------
# DEPRECATED ARGUMENTS PARSING
# -----------------------------------------------------------------------------
# If a deprecated argument is set, change to the new argument.
def set_deprecated_arguments(
    sim_params,
    frontend: str = "runner",
) -> None:

    if frontend == "runner":
        set_runner_deprecated_arguments(sim_params)
    elif frontend == "postprocess":
        pass
    elif frontend == "plotter":
        pass
    elif frontend == "ldos_plotter":
        pass
    else:
        print(f"-- ERROR! set_deprecated_arguments(): Unknown frontend: '{frontend}'.")
        status_code = 1
        sys.exit(status_code)


# Change deprecated arguments to supported ones.
def set_runner_deprecated_arguments(sim_params) -> None:
    geometry = sim_params["geometry"]
    for component in geometry:
        if "regular_polygon" in component:
            regular_polygon = component["regular_polygon"]
            if "bounding_disc_diameter" in regular_polygon:
                diameter = regular_polygon["bounding_disc_diameter"]
                num_edges = regular_polygon["num_edges"]
                side_length = diameter * math.sin(math.pi / num_edges)
                regular_polygon.pop("bounding_disc_diameter")
                regular_polygon.update({"side_length": side_length})
                print(
                    "-- WARNING! The argument 'bounding_disc_diameter' is deprecated. Use 'side_length' instead."
                )
                print(
                    "   The relation is: side_length = bounding_disc_diameter * sin(pi / num_edges)."
                )
                print(
                    f"   Changing 'bounding_disc_diameter' = {diameter} to 'side_length' = {side_length}."
                )


# -----------------------------------------------------------------------------
# WRAPPER: PARSE ARGUMENTS
# -----------------------------------------------------------------------------
# Read command-line arguments. Look for valid .json file and parse it.
def parse_command_line_arguments(
    program_name: str,
    program_description: str,
    frontend: str = "runner",
):
    # Create the argument parser.
    parser = make_default_parser(
        program_name,
        program_description,
        frontend,
    )

    # Parse all command-line arguments.
    args = parser.parse_args()

    # Display help and exit if program is called without arguments.
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # Set up dictionary storage for simulation parameters.
    sim_params = {}

    # If a json config file is set, load it first.
    if args.file is not None:
        # Check that file is valid and can be read: exit if failure.
        json_file_path = Path(args.file)
        utils.check_file(json_file_path)

        # Read the JSON file.
        with open(args.file, "r") as f:
            data = f.read()

        # Parse the JSON file.
        sim_params = json.loads(data)
    elif (args.file is None) and (frontend == "plotter"):
        # Use default plotting config file.
        default_plot_config = configuration.default_plot_config
        default_plot_config_path = Path(default_plot_config)

        # Check that default file exists and can be read: exit if failure.
        utils.check_file(default_plot_config_path)

        # Read the JSON file.
        with open(default_plot_config, "r") as f:
            data = f.read()

        # Parse the JSON file.
        sim_params = json.loads(data)
    elif (args.file is None) and (frontend == "ldos_plotter"):
        # Set default dict structure.
        sim_params = {"general": {}}

    # Overwrite arguments from the command line.
    set_arguments_from_cli(
        sim_params,
        args,
        frontend,
    )

    # Change deprecated arguments.
    set_deprecated_arguments(sim_params, frontend)

    # Dump parameters to terminal and exit.
    if args.dump:
        print(json.dumps(sim_params, sort_keys=False, indent=4))
        sys.exit(0)

    # Get the relative C++ binary path.
    if frontend in ["runner", "postprocess"]:
        cpp_path = args.cpp_path
        cpp_precision = args.precision
    else:
        cpp_path = None
        cpp_precision = None

    return sim_params, (cpp_path, cpp_precision)
