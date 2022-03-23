#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Functionality for creating the argument parser for the postprocess, as well as validating the inputs."""

import argparse
import sys

sys.path.insert(0, "../")
from common import cli


def make_parser(parser: argparse.ArgumentParser) -> None:
    """Given a parser, add all arguments for the postprocess."""

    parser.set_defaults(which="postprocess")

    parser.description = "Compute the spectra from a converged simulation, e.g. the local density of states (LDOS)."

    # Config file.
    parser.add_argument(
        "-C",
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file. If this is a path to a directory it will use 'postprocess_config.json' within that directory.",
    )

    # C++ binary and precision.
    group_cpp = parser.add_mutually_exclusive_group()
    group_cpp.add_argument(
        "-P",
        "--precision",
        type=int,
        choices=cli.VALID_PRECISIONS,
        default=32,
        help="The floating point precision. The C++ binary 'build/release/bin/postprocess_fXX', where XX is the precision, relative to the superconga.py script will be used. Note that 32 bit precision is faster but less accurate. (Default: %(default)s)",
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

    # =============== SPECTROSCOPY =============== #
    spectroscopy = parser.add_argument_group(title="Spectroscopy")

    # Energy max.
    spectroscopy.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="The maximum energy to compute spectra at.",
    )

    # Energy min.
    spectroscopy.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="The minimum energy to compute spectra at.",
    )

    # Energy broadening.
    spectroscopy.add_argument(
        "--energy-broadening",
        type=float,
        default=None,
        help="The broadening factor of the energy, i.e. the imaginary part.",
    )

    # Num energies.
    spectroscopy.add_argument(
        "-e",
        "--num-energies",
        type=int,
        default=None,
        help="Number of energies to compute the spectra at, between (and including) the min/max energies.",
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

    # Num energies per block.
    numerics.add_argument(
        "-E",
        "--num-energies-per-block",
        type=int,
        default=None,
        help="Number of energies per block. A negative number means all of them. The computation can be memory intensive, which might lead to the GPU running out of memory. To solve this, the problem is divided into multiple blocks calculated in serial.",
    )

    # Num Fermi momenta.
    numerics.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of momenta in Fermi-surface average. If divisible by 4 Boole's method is used. If divisible by 3 Simpson's 3/8 rule is used. If divisible by 2 Simpson's 1/3 rule is used. Otherwise the trapezoidal method is used. Recommended minimum: 700.",
    )

    # Num iterations burn-in.
    numerics.add_argument(
        "-b",
        "--num-iterations-burnin",
        type=int,
        default=None,
        help="The number of iterations to run just converging the boundary. If negative the boundary will be iterated until convergence, which is recommended.",
    )

    # Num iterations max.
    numerics.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run per energy block.",
    )

    # Num iterations min.
    numerics.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run per energy block.",
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
    misc.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Directory containing the data of a converged simulation.",
    )

    # Save data.
    misc.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Directory where the data will be saved. If files already exist, they will be overwritten. If not set the '--load-path' directory will be used.",
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


def update_config_from_cli(config: dict, args: argparse.Namespace) -> None:
    """Update the configuration parameters with arguments from the CLI."""

    group = "spectroscopy"
    cli.set_value(
        config=config,
        key="energy_max",
        value=args.energy_max,
        group=group,
    )
    cli.set_value(
        config=config,
        key="energy_min",
        value=args.energy_min,
        group=group,
    )
    cli.set_value(
        config=config,
        key="energy_broadening",
        value=args.energy_broadening,
        group=group,
    )

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
        key="num_energies",
        value=args.num_energies,
        group=group,
    )
    cli.set_value(
        config=config,
        key="num_energies_per_block",
        value=args.num_energies_per_block,
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

    group = "misc"
    cli.set_value(
        config=config,
        key="data_format",
        value=args.data_format,
        group=group,
    )
    cli.set_value(
        config=config,
        key="load_path",
        value=args.load_path,
        group=group,
    )
    if args.save_path is None:
        save_path = config[group]["load_path"]
    else:
        save_path = args.save_path
    cli.set_value(
        config=config,
        key="save_path",
        value=save_path,
        group=group,
    )
    cli.set_flag(
        config=config,
        key="verbose",
        on=args.verbose,
        off=args.quiet,
        group=group,
    )


def get_checks() -> dict:
    """Get the sanity checks to perform on the configuration parameters."""

    checks = {
        "spectroscopy": {
            "type": dict,
            "fields": {
                "energy_max": {"type": float, "low_include": 0.0},
                "energy_min": {"type": float, "low_include": 0.0},
                "energy_broadening": {"type": float, "low": 0.0},
                "num_energies": {"type": int, "low": 0},
            },
        },
        "numerics": {
            "type": dict,
            "fields": {
                "convergence_criterion": {"type": float, "low": 0.0},
                "norm": {"type": str, "one_of": cli.VALID_NORMS},
                "num_energies_per_block": {"type": int, "excluded": [0]},
                "num_fermi_momenta": {"type": int, "low": 0},
                "num_iterations_max": {"type": int, "low": 0},
                "num_iterations_min": {"type": int, "low_include": 0},
            },
        },
        "misc": {
            "type": dict,
            "fields": {
                "data_format": {"type": str, "one_of": cli.VALID_DATA_FORMATS},
                "load_path": {"type": str},
                "save_path": {"type": str},
                "verbose": {"type": bool},
            },
        },
    }
    return checks
