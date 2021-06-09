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

    parser.add_argument(
        "--op-path",
        type=str,
        default=None,
        help="Path to order-parameter file. The simulation will start from this order parameter.",
    )

    parser.add_argument(
        "--vp-path",
        type=str,
        default=None,
        help="Path to vector-potential file. The simulation will start from this vector potential.",
    )

    # =============== PHYSICS =============== #

    # Symmetry.
    parser.add_argument(
        "-H",
        "--op-sym",
        type=str,
        default=None,
        choices=configuration.valid_symmetries,
        nargs="+",
        help="Order parameter pairing symmetry. Multiple values can be passed, one for each order parameter component.",
    )

    # Relative Tc.
    parser.add_argument(
        "-C",
        "--op-tc",
        type=float,
        default=None,
        nargs="+",
        help="Order parameter critical temperature. Multiple values can be passed, one for each order parameter component. Note that all values will be scaled so that the maximum is 1.",
    )

    # Phase shift.
    parser.add_argument(
        "-J",
        "--op-init-phase-shift",
        type=float,
        default=None,
        nargs="+",
        help="The phase shift of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Num phase windings.
    parser.add_argument(
        "-W",
        "--op-init-phase-winding",
        type=float,
        default=None,
        nargs="+",
        help="The phase winding of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Random phase weight.
    parser.add_argument(
        "-R",
        "--op-init-rand-phase-weight",
        type=float,
        default=None,
        nargs="+",
        help="The weight of the random phase of the initial order parameter guess. Multiple values can be passed, one for each order parameter component.",
    )

    # Temperature.
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Temperature in units of the critical temperature T_c.",
    )

    # External flux quanta.
    parser.add_argument(
        "-B",
        "--external-flux-quanta",
        type=float,
        default=None,
        help="External magnetic field, perpendicular to sample, in units of flux quanta. For non-solenoid field, it is distributed uniformly over the sample.",
    )

    # Gauge.
    group_guage = parser.add_mutually_exclusive_group()
    group_guage.add_argument(
        "--gauge-radial",
        action="store_true",
        help="Use the radial gauge, A = 0.5 * B * (-y, x).",
    )
    group_guage.add_argument(
        "--gauge-solenoid",
        action="store_true",
        help="Use the solenoid gauge, A = 0.5 * B * Area (-y, x) / (x^2 + y^2).",
    )

    # Penetration depth.
    parser.add_argument(
        "-k",
        "--penetration-depth",
        type=float,
        default=None,
        help="The London penetration depth, expressed in coherence lengths (i.e. the Ginzburg-Landau coefficient). If the value is less than zero, the induced vector-potential will not be computed. In other words, the penetration depth will be infinite.",
    )

    # Crystal axes rotation.
    parser.add_argument(
        "-r",
        "--crystal-axes-rotation",
        type=float,
        default=None,
        help="Rotation between grain and crystal ab-axes, in units of 2pi.",
    )

    # =============== ACCURACY =============== #

    # Optimizer.
    parser.add_argument(
        "--opt-step-size",
        type=float,
        default=None,
        help="The step size in the optimizer.",
    )
    parser.add_argument(
        "--opt-inertia",
        type=float,
        default=None,
        help="The inertia of the optimizer.",
    )
    group_optimiser_adaptive = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--opt-rank",
        type=int,
        default=None,
        help="The maximal rank of the system of equations in the Anderson step, i.e. the maximum number of stored steps.",
    )
    parser.add_argument(
        "--opt-min-mixing",
        type=float,
        default=None,
        help="The minimal value of the mixing parameter used in Anderson steps. Only applicable if rank > 1.",
    )
    parser.add_argument(
        "--opt-max-mixing",
        type=float,
        default=None,
        help="The maximal value of the mixing parameter used in Anderson steps. Only applicable if rank > 1.",
    )
    group_optimiser_replace = parser.add_mutually_exclusive_group()
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
    group_optimiser_anderson = parser.add_mutually_exclusive_group()
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

    # Convergence criterion.
    parser.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criterion for self-consistency.",
    )

    # Lattice side length.
    parser.add_argument(
        "-N",
        "--points-per-coherence-length",
        type=float,
        default=None,
        help="Number of lattice points per coherence length. Recommended value 10 (or larger).",
    )

    # Num Fermi momenta.
    parser.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Recommended minimum: 100.",
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

    # Num Ozaki energies.
    parser.add_argument(
        "-e",
        "--num-ozaki-energies",
        type=int,
        default=None,
        help="Number of Ozaki energies (similar to Matsubara energies, see the documentation).",
    )

    # Green's function accuracy.
    parser.add_argument(
        "--vp-error",
        type=float,
        default=None,
        help="The maximum error (relative Frobenious norm) of the Green's function used when computing the induced vector-potential.",
    )

    # =============== MISC =============== #

    # Version number.
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )

    # Data path.
    parser.add_argument(
        "-o",
        "--data-path",
        type=str,
        default=None,
        help="Base folder for output (data and parameter files).",
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

    # Save.
    group_save = parser.add_mutually_exclusive_group()
    group_save.add_argument(
        "--save-data",
        action="store_true",
        help="Save data files to disk. If files already exist, they will be overwritten. The exception is the file convergence.csv, which will have data appended to it.",
    )
    group_save.add_argument(
        "--no-save-data", action="store_true", help="Do not save data files to disk."
    )

    # Save data interval.
    parser.add_argument(
        "--save-data-interval",
        type=int,
        default=None,
        help="How frequently to save data (every Nth iteration). Zero means not saving periodically.",
    )

    # Visualize.
    group_visualize = parser.add_mutually_exclusive_group()
    group_visualize.add_argument(
        "--visualize", action="store_true", help="Run with real-time plotting."
    )
    group_visualize.add_argument(
        "--no-visualize", action="store_true", help="Run without real-time plotting."
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

    # Convergence criterion.
    parser.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criterion for self-consistency.",
    )

    # Data path.
    parser.add_argument(
        "-p",
        "--data-path",
        type=str,
        default=None,
        help="Base folder for input/output (data and parameter files).",
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
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Recommended minimum: 700.",
    )

    # Save.
    group_save = parser.add_mutually_exclusive_group()
    group_save.add_argument(
        "--save-data",
        action="store_true",
        help="Save data files to disk. If files already exist, they will be overwritten.",
    )
    group_save.add_argument(
        "--no-save-data", action="store_true", help="Do not save data files to disk."
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
        "-p",
        "--data-path",
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
        "-p",
        "--data-path",
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
    # Override values from the command line.
    set_value(sim_params, "convergence_criterion", args.convergence_criterion)
    set_value(sim_params, "crystal_axes_rotation", args.crystal_axes_rotation)
    set_value(sim_params, "data_path", args.data_path)
    set_value(sim_params, "external_flux_quanta", args.external_flux_quanta)

    set_flag(sim_params, "gauge_solenoid", args.gauge_solenoid, args.gauge_radial)
    set_value(
        sim_params, "points_per_coherence_length", args.points_per_coherence_length
    )
    set_value(sim_params, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(sim_params, "num_iterations_max", args.num_iterations_max)
    set_value(sim_params, "num_iterations_min", args.num_iterations_min)
    set_value(sim_params, "num_ozaki_energies", args.num_ozaki_energies)

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

    group = "order_parameter"
    set_value(sim_params, "critical_temperature", args.op_tc, group)
    set_value(sim_params, "initial_phase_shift", args.op_init_phase_shift, group)
    set_value(sim_params, "initial_phase_winding", args.op_init_phase_winding, group)
    set_value(
        sim_params,
        "initial_random_phase_weight",
        args.op_init_rand_phase_weight,
        group,
    )
    set_value(sim_params, "symmetry", args.op_sym, group)
    set_value(sim_params, "penetration_depth", args.penetration_depth)

    set_flag(sim_params, "save_data", args.save_data, args.no_save_data)
    set_value(sim_params, "save_data_interval", args.save_data_interval)
    set_value(sim_params, "temperature", args.temperature)
    set_flag(sim_params, "verbose", args.verbose, args.quiet)
    set_flag(sim_params, "visualize", args.visualize, args.no_visualize)

    set_value(sim_params, "order_parameter_path", args.op_path)
    set_value(sim_params, "vector_potential_error", args.vp_error)
    set_value(sim_params, "vector_potential_path", args.vp_path)


# Postprocess: Override json dictionary with command-line input.
def set_postprocess_arguments_from_cli(
    sim_params,
    args,
) -> None:
    # Override values from the command line.
    set_value(sim_params, "convergence_criterion", args.convergence_criterion)
    set_value(sim_params, "data_path", args.data_path)
    set_value(sim_params, "energy_max", args.energy_max)
    set_value(sim_params, "energy_min", args.energy_min)
    set_value(sim_params, "energy_smearing", args.energy_smearing)
    set_value(sim_params, "num_energies", args.num_energies)
    set_value(sim_params, "num_energies_per_block", args.num_energies_per_block)
    set_value(sim_params, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(sim_params, "num_iterations_max", args.num_iterations_max)
    set_value(sim_params, "num_iterations_min", args.num_iterations_min)
    set_flag(sim_params, "save_data", args.save_data, args.no_save_data)
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

    # Get the relative C++ binary path.
    if frontend in ["runner", "postprocess"]:
        cpp_path = args.cpp_path
        cpp_precision = args.precision
    else:
        cpp_path = None
        cpp_precision = None

    return sim_params, (cpp_path, cpp_precision)
