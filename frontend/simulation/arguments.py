#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import argparse
import sys

sys.path.insert(0, "../")
from common import cli


def make_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(which="simulation")

    parser.description = "Run a simulation."

    # Config file.
    parser.add_argument(
        "-C",
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file. If this is a path to a directory it will use 'simulation_config.json' within that directory.",
    )

    # C++ binary and precision.
    group_cpp = parser.add_mutually_exclusive_group()
    group_cpp.add_argument(
        "-P",
        "--precision",
        type=int,
        choices=cli.VALID_PRECISIONS,
        default=32,
        help="The floating point precision. The C++ binary 'build/release/bin/simulation_fXX', where XX is the precision, relative to the superconga.py script will be used. Note that 32 bit precision is faster but less accurate. (Default: %(default)s)",
    )
    group_cpp.add_argument(
        "--cpp-path",
        type=str,
        default=None,
        help="Path to the C++ binary.",
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
        description="The order-parameter components take three arguments. The arguments are critical temperature, initial phase shift (in units of 2pi), and standard deviation of the additive initial (Gaussian) noise. The critical temperatures will be scaled so that the maximum is Tc = 1. Vortices can be added with '--vortex'. Note that vortices added using the CLI will be added to all order-parameter components. To set them separately use the config file.",
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
        help="The London penetration depth, expressed in coherence lengths (i.e. the Ginzburg-Landau coefficient). If the value is zero or less, the induced vector-potential will not be computed. In other words, the penetration depth will be infinite.",
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
    physics.add_argument(
        "-g",
        "--gauge",
        type=str,
        choices=cli.VALID_GAUGES,
        default=None,
        help="Set the gauge. Radial: A = 0.5 * B * (-y, x). Solenoid: A = 0.5 * B * Area (-y, x) / (x^2 + y^2).",
    )

    # Charge sign.
    physics.add_argument(
        "--charge-sign",
        type=int,
        choices=cli.VALID_CHARGE_SIGNS,
        default=None,
        help="Set sign of the charge. Use -1 (+1) for electrons (holes).",
    )

    # =============== PHYSICS: ORDER PARAMETER =============== #

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
            "initial_noise_stddev",
        ),
        type=float,
        default=None,
        nargs=3,
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
            "initial_noise_stddev",
        ),
        type=float,
        default=None,
        nargs=3,
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
            "initial_noise_stddev",
        ),
        type=float,
        default=None,
        nargs=3,
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
            "initial_noise_stddev",
        ),
        type=float,
        default=None,
        nargs=3,
        help="S-wave order-parameter component.",
    )

    # =============== PHYSICS: VORTICES =============== #

    # Ignore geometry in config file.
    physics.add_argument(
        "--clear-vortices",
        action="store_true",
        help="Ignore the vortices specified in the config file.",
    )

    # Vortices.
    physics.add_argument(
        "--vortex",
        metavar=("center_x", "center_y", "winding_number"),
        type=float,
        action="append",
        nargs=3,
        help="Add a vortex in all order-parameter components.",
    )

    # =============== GEOMETRY =============== #
    geometry = parser.add_argument_group(
        title="Geometry",
        description="Note that when specifying the geometry via the CLI all components set here will be added before any are removed. If the desired geometry depends on the order of the components, use the config file instead. Furthermore, general polygons, with arbitrary vertex coordinates, is only supported in the config file.",
    )

    # Ignore geometry in config file.
    geometry.add_argument(
        "--clear-geometry",
        action="store_true",
        help="Ignore the geometry specified in the config file.",
    )

    # Add disc.
    geometry.add_argument(
        "--add-disc",
        metavar=("center_x", "center_y", "radius"),
        type=float,
        action="append",
        nargs=3,
        default=None,
        help="Add disc to geometry.",
    )

    # Remove disc.
    geometry.add_argument(
        "--remove-disc",
        metavar=("center_x", "center_y", "radius"),
        type=float,
        action="append",
        nargs=3,
        default=None,
        help="Remove disc from geometry.",
    )

    # Add regular polygon.
    geometry.add_argument(
        "--add-regular-polygon",
        metavar=("center_x", "center_y", "num_edges", "rotation", "side_length"),
        type=float,
        action="append",
        nargs=5,
        default=None,
        help="Add regular polygon to geometry. The arguments are side length, center x, center y, the number of edges, and the rotation (in units of 2pi).",
    )

    # Remove regular polygon.
    geometry.add_argument(
        "--remove-regular-polygon",
        metavar=("center_x", "center_y", "num_edges", "rotation", "side_length"),
        type=float,
        action="append",
        nargs=5,
        default=None,
        help="Remove regular polygon from geometry. The arguments are side length, center x, center y, the number of edges, and the rotation (in units of 2pi).",
    )

    # =============== NUMERICS =============== #
    numerics = parser.add_argument_group(title="Numerics")

    # Convergence criterion.
    numerics.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criterion for self-consistency.",
    )

    # Norm.
    numerics.add_argument(
        "-n",
        "--norm",
        type=str,
        choices=cli.VALID_NORMS,
        default=None,
        help="Which norm to use when computing the residuals. In order from less to more strict; L1, L2, LInf.",
    )

    # Num Ozaki energies.
    numerics.add_argument(
        "-e",
        "--energy-cutoff",
        type=float,
        default=None,
        help="The energy cutoff (in units of 2*pi*kB*Tc).",
    )

    # Num Fermi momenta.
    numerics.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of momenta in Fermi-surface average. If divisible by 4 Boole's method is used. If divisible by 3 Simpson's 3/8 rule is used. If divisible by 2 Simpson's 1/3 rule is used. Otherwise the trapezoidal method is used. Recommended minimum: 100.",
    )

    # Num iterations burn-in.
    numerics.add_argument(
        "-b",
        "--num-iterations-burnin",
        type=int,
        default=None,
        help="The number of iterations to run just converging the boundary. If negative the boundary will be iterated until convergence, which is recommended when using '--continue' or '--load-path'.",
    )

    # Num iterations max.
    numerics.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run.",
    )

    # Num iterations min.
    numerics.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run.",
    )

    # Lattice side length.
    numerics.add_argument(
        "-N",
        "--points-per-coherence-length",
        type=float,
        default=None,
        help="Number of lattice points per coherence length. Recommended value 10 (or larger).",
    )

    # Green's function numerics.
    numerics.add_argument(
        "--vp-error",
        type=float,
        default=None,
        help="The maximum error (relative Frobenious norm) of the Green's function used when computing the induced vector-potential.",
    )

    # =============== ACCELERATOR =============== #
    accelerator = parser.add_argument_group(title="Accelerator")

    accelerator.add_argument(
        "-A",
        "--accelerator",
        type=str,
        choices=cli.VALID_ACCELERATORS,
        default=None,
        help="Which accelerator to use. Note that AB = Adaptive-Ball and BB = Barzilai-Borwein.",
    )
    accelerator.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="The step size in the accelerator.",
    )
    accelerator.add_argument(
        "--step-size-max",
        type=float,
        default=None,
        help="The maximum step size. Only used in accelerators with an adaptive step size, e.g. Adaptive-Ball and Barzilai-Borwein.",
    )
    accelerator.add_argument(
        "--step-size-min",
        type=float,
        default=None,
        help="The minimum step size. Only used in accelerators with an adaptive step size, e.g. Adaptive-Ball and Barzilai-Borwein.",
    )
    accelerator.add_argument(
        "--drag",
        type=float,
        default=None,
        help="The drag in the Polyak accelerator.",
    )
    accelerator.add_argument(
        "--step-size-factor",
        type=float,
        default=None,
        help="The step-size factor, determining how much the step size should increase/decrease, in the Adaptive-Ball accelerator.",
    )
    accelerator.add_argument(
        "--cos-similarity-min",
        type=float,
        default=None,
        help="The minimum cosine similarity to tolerate in the Adaptive-Ball accelerator",
    )
    accelerator.add_argument(
        "--rank",
        type=int,
        default=None,
        help="The maximum rank of the system of equations in the Anderson accelerator, i.e. the maximum number of stored steps.",
    )

    # =============== MISC =============== #
    misc = parser.add_argument_group(title="Misc")

    # Data format.
    misc.add_argument(
        "-D",
        "--data-format",
        type=str,
        choices=cli.VALID_DATA_FORMATS,
        default=None,
        help="What format use when saving data. H5 files are much smaller and faster to read and write. If SuperConga is built without HDF5 support, CSV files will be saved.",
    )

    # Load data.
    group_load = misc.add_mutually_exclusive_group()
    group_load.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Directory containing data. The simulation will resume from this data. Burn-in until convergence '-b -1' is recommended.",
    )
    group_load.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume the simulation from the data in the '--save-path' directory. Burn-in until convergence '-b -1' is recommended.",
    )

    # Save data.
    misc.add_argument(
        "-F",
        "--save-frequency",
        type=int,
        default=None,
        help="How frequently to save data (every Nth iteration). Zero means not saving data at all, and a negative value means only saving at the end.",
    )
    misc.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Directory where the data will be saved. If files already exist, they will be overwritten. The exception is the file convergence.csv, which will have data appended to it.",
    )

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

    # Visualize.
    group_visualize = misc.add_mutually_exclusive_group()
    group_visualize.add_argument(
        "--visualize",
        action="store_true",
        help="Real-time plotting. The units are E_0=2*pi*kB*Tc, j_d = E_0*|e|*v_F*N_F, A_0 = Phi_0 / (pi*xi_0), and B_0 = \Phi_0/(pi*xi^2_0), where xi_0 = hbar*vF/E_0, and Phi_0 = pi*hbar/|e|. Remember that blue is positive in all plots, and red negative.",
    )
    group_visualize.add_argument(
        "--no-visualize", action="store_true", help="No real-time plotting."
    )


def update_config_from_cli(config: dict, args: argparse.Namespace) -> None:
    # ================== PHYSICS ================== #
    group = "physics"
    cli.set_value(
        config=config,
        key="temperature",
        value=args.temperature,
        group=group,
    )
    cli.set_value(
        config=config,
        key="external_flux_quanta",
        value=args.external_flux_quanta,
        group=group,
    )
    cli.set_value(
        config=config,
        key="penetration_depth",
        value=args.penetration_depth,
        group=group,
    )
    cli.set_value(
        config=config,
        key="crystal_axes_rotation",
        value=args.crystal_axes_rotation,
        group=group,
    )
    cli.set_value(
        config=config,
        key="gauge",
        value=args.gauge,
        group=group,
    )
    cli.set_value(
        config=config,
        key="charge_sign",
        value=args.charge_sign,
        group=group,
    )

    # ================== PHYSICS: ORDER PARAMETER ================== #
    # Get vortices.
    vortices = []
    if args.vortex is not None:
        for vortex in args.vortex:
            center_x, center_y, winding_number = vortex
            vortices.append(
                {
                    "center_x": center_x,
                    "center_y": center_y,
                    "winding_number": winding_number,
                }
            )

    # Fetch the arguments for all components so we can loop over them.
    components = {
        "dx2-y2": {"no_wave": args.no_dx2_y2_wave, "args": args.dx2_y2_wave},
        "dxy": {"no_wave": args.no_dxy_wave, "args": args.dxy_wave},
        "g": {"no_wave": args.no_g_wave, "args": args.g_wave},
        "s": {"no_wave": args.no_s_wave, "args": args.s_wave},
    }

    # Loop over components and update parameters.
    for key, value in components.items():
        key_exists = key in config["physics"]["order_parameter"].keys()
        if value["no_wave"] and key_exists:
            # If the user has set "--no-X-wave" the component is removed.
            del config["physics"]["order_parameter"][key]
        elif value["args"] is not None:
            # The arguments for a component has been set, so lets update everything.
            Tc, phase_shift, noise_stddev = value["args"]

            # Fetch vortices if component was set in config.
            if key_exists:
                component_vortices = config["physics"]["order_parameter"][key][
                    "vortices"
                ]
            else:
                component_vortices = []

            # Update all fields of the component.
            config["physics"]["order_parameter"][key] = {
                "critical_temperature": Tc,
                "initial_phase_shift": phase_shift,
                "initial_noise_stddev": noise_stddev,
                "vortices": component_vortices,
            }

        # If the component exists we must update the vortices if set.
        if key in config["physics"]["order_parameter"].keys():
            if args.clear_vortices:
                config["physics"]["order_parameter"][key]["vortices"] = vortices
            else:
                config["physics"]["order_parameter"][key]["vortices"].extend(vortices)

    # ================== GEOMETRY ================== #
    group = "geometry"
    if args.clear_geometry:
        config[group] = []

    # Add discs.
    if args.add_disc is not None:
        for disc in args.add_disc:
            center_x, center_y, radius = disc
            config[group].append(
                {
                    "disc": {
                        "add": True,
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                    }
                }
            )

    # Add regular polygons.
    if args.add_regular_polygon is not None:
        for regular_polygon in args.add_regular_polygon:
            center_x, center_y, num_edges, rotation, side_length = regular_polygon
            config[group].append(
                {
                    "regular_polygon": {
                        "add": True,
                        "center_x": center_x,
                        "center_y": center_y,
                        "num_edges": int(num_edges),
                        "rotation": rotation,
                        "side_length": side_length,
                    }
                }
            )

    # Remove discs.
    if args.remove_disc is not None:
        for disc in args.remove_disc:
            center_x, center_y, radius = disc
            config[group].append(
                {
                    "disc": {
                        "add": False,
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                    }
                }
            )

    # Remove regular polygons.
    if args.remove_regular_polygon is not None:
        for regular_polygon in args.remove_regular_polygon:
            center_x, center_y, num_edges, rotation, side_length = regular_polygon
            config[group].append(
                {
                    "regular_polygon": {
                        "add": False,
                        "center_x": center_x,
                        "center_y": center_y,
                        "num_edges": int(num_edges),
                        "rotation": rotation,
                        "side_length": side_length,
                    }
                }
            )

    # ================== ACCURACY ================== #
    group = "numerics"
    cli.set_value(
        config=config,
        key="convergence_criterion",
        value=args.convergence_criterion,
        group=group,
    )
    cli.set_value(
        config=config,
        key="norm",
        value=args.norm,
        group=group,
    )
    cli.set_value(
        config=config,
        key="energy_cutoff",
        value=args.energy_cutoff,
        group=group,
    )
    cli.set_value(
        config=config,
        key="num_fermi_momenta",
        value=args.num_fermi_momenta,
        group=group,
    )
    cli.set_value(
        config=config,
        key="num_iterations_burnin",
        value=args.num_iterations_burnin,
        group=group,
    )
    cli.set_value(
        config=config,
        key="num_iterations_max",
        value=args.num_iterations_max,
        group=group,
    )
    cli.set_value(
        config=config,
        key="num_iterations_min",
        value=args.num_iterations_min,
        group=group,
    )
    cli.set_value(
        config=config,
        key="points_per_coherence_length",
        value=args.points_per_coherence_length,
        group=group,
    )
    cli.set_value(
        config=config,
        key="vector_potential_error",
        value=args.vp_error,
        group=group,
    )

    # ================== ACCELERATOR ================== #
    group = "accelerator"
    cli.set_value(
        config=config,
        key="name",
        value=args.accelerator,
        group=group,
    )
    cli.set_value(
        config=config,
        key="step_size",
        value=args.step_size,
        group=group,
    )
    cli.set_value(
        config=config,
        key="step_size_max",
        value=args.step_size_max,
        group=group,
    )
    cli.set_value(
        config=config,
        key="step_size_min",
        value=args.step_size_min,
        group=group,
    )
    cli.set_value(
        config=config,
        key="drag",
        value=args.drag,
        group=group,
    )
    cli.set_value(
        config=config,
        key="step_size_factor",
        value=args.step_size_factor,
        group=group,
    )
    cli.set_value(
        config=config,
        key="cos_similarity_min",
        value=args.cos_similarity_min,
        group=group,
    )
    cli.set_value(
        config=config,
        key="rank",
        value=args.rank,
        group=group,
    )

    # ================== MISC ================== #
    group = "misc"
    cli.set_value(
        config=config,
        key="data_format",
        value=args.data_format,
        group=group,
    )
    cli.set_value(
        config=config,
        key="save_path",
        value=args.save_path,
        group=group,
    )
    cli.set_value(
        config=config,
        key="save_frequency",
        value=args.save_frequency,
        group=group,
    )
    if args.do_continue:
        load_path = config["misc"]["save_path"]
    else:
        load_path = args.load_path
    cli.set_value(
        config=config,
        key="load_path",
        value=load_path,
        group=group,
    )
    cli.set_flag(
        config=config,
        key="verbose",
        on=args.verbose,
        off=args.quiet,
        group=group,
    )
    cli.set_flag(
        config=config,
        key="visualize",
        on=args.visualize,
        off=args.no_visualize,
        group=group,
    )


def get_checks() -> dict:
    checks = {
        "physics": {
            "type": dict,
            "fields": {
                "temperature": {"type": float, "low": 0.0},
                "external_flux_quanta": {"type": float},
                "penetration_depth": {"type": float},
                "crystal_axes_rotation": {"type": float},
                "gauge": {"type": str, "one_of": cli.VALID_GAUGES},
                "charge_sign": {"type": int, "one_of": cli.VALID_CHARGE_SIGNS},
                "order_parameter": {
                    "type": dict,
                    "fields": {
                        "dx2-y2": {
                            "type": dict,
                            "optional": True,
                            "fields": {
                                "critical_temperature": {"type": float, "low": 0.0},
                                "initial_phase_shift": {"type": float},
                                "initial_noise_stddev": {"type": float},
                                "vortices": {
                                    "type": list,
                                    "sub": {
                                        "type": dict,
                                        "fields": {
                                            "center_x": {"type": float},
                                            "center_y": {"type": float},
                                            "winding_number": {"type": float},
                                        },
                                    },
                                },
                            },
                        },
                        "dxy": {
                            "type": dict,
                            "optional": True,
                            "fields": {
                                "critical_temperature": {"type": float, "low": 0.0},
                                "initial_phase_shift": {"type": float},
                                "initial_noise_stddev": {"type": float},
                                "vortices": {
                                    "type": list,
                                    "sub": {
                                        "type": dict,
                                        "fields": {
                                            "center_x": {"type": float},
                                            "center_y": {"type": float},
                                            "winding_number": {"type": float},
                                        },
                                    },
                                },
                            },
                        },
                        "g": {
                            "type": dict,
                            "optional": True,
                            "fields": {
                                "critical_temperature": {"type": float, "low": 0.0},
                                "initial_phase_shift": {"type": float},
                                "initial_noise_stddev": {"type": float},
                                "vortices": {
                                    "type": list,
                                    "sub": {
                                        "type": dict,
                                        "fields": {
                                            "center_x": {"type": float},
                                            "center_y": {"type": float},
                                            "winding_number": {"type": float},
                                        },
                                    },
                                },
                            },
                        },
                        "s": {
                            "type": dict,
                            "optional": True,
                            "fields": {
                                "critical_temperature": {"type": float, "low": 0.0},
                                "initial_phase_shift": {"type": float},
                                "initial_noise_stddev": {"type": float},
                                "vortices": {
                                    "type": list,
                                    "sub": {
                                        "type": dict,
                                        "fields": {
                                            "center_x": {"type": float},
                                            "center_y": {"type": float},
                                            "winding_number": {"type": float},
                                        },
                                    },
                                },
                            },
                        },
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
                            "center_x": {"type": float},
                            "center_y": {"type": float},
                            "radius": {"type": float, "low": 0.0},
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
                            "center_x": {"type": float},
                            "center_y": {"type": float},
                            "num_edges": {"type": int, "excluded": [0, 1, 2]},
                            "rotation": {"type": float},
                            "side_length": {"type": float, "low": 0.0},
                        },
                    },
                },
            },
        },
        "numerics": {
            "type": dict,
            "fields": {
                "convergence_criterion": {"type": float, "low": 0.0},
                "norm": {"type": str, "one_of": cli.VALID_NORMS},
                "energy_cutoff": {"type": float, "low": 0.0},
                "num_fermi_momenta": {"type": int, "low": 0},
                "num_iterations_burnin": {"type": int},
                "num_iterations_max": {"type": int, "low": 0},
                "num_iterations_min": {"type": int, "low_include": 0},
                "points_per_coherence_length": {"type": float, "low": 0.0},
                "vector_potential_error": {"type": float, "low_include": 0.0},
            },
        },
        "accelerator": {
            "type": dict,
            "fields": {
                "name": {"type": str, "one_of": cli.VALID_ACCELERATORS},
                "step_size": {"type": float, "low": 0.0, "optional": True},
                "step_size_max": {"type": float, "low": 0.0, "optional": True},
                "step_size_min": {"type": float, "low": 0.0, "optional": True},
                "drag": {"type": float, "low": 0.0, "high": 1.0, "optional": True},
                "rank": {"type": int, "low": 1, "optional": True},
                "step_size_factor": {"type": float, "low": 1.0, "optional": True},
                "cos_similarity_min": {
                    "type": float,
                    "low": 0.0,
                    "high": 1.0,
                    "optional": True,
                },
            },
        },
        "misc": {
            "type": dict,
            "fields": {
                "data_format": {"type": str, "one_of": cli.VALID_DATA_FORMATS},
                "load_path": {"type": str},
                "save_frequency": {"type": int},
                "save_path": {"type": str},
                "verbose": {"type": bool},
                "visualize": {"type": bool},
            },
        },
    }

    return checks
