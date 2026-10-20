#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "frontend"))
from setup import setup
from compilation import compilation
from simulation import simulation
from postprocess import postprocess
from simulation_plotter import simulation_plotter
from postprocess_plotter import postprocess_plotter
from convergence_plotter import convergence_plotter
from data_converter import data_converter

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    # Create parser and subparsers.
    parser = argparse.ArgumentParser(
        description="SuperConga frontend. Quick start: Run 'python superconga.py setup --type <type>' followed by 'python superconga.py compile --type <type>' in order to build the C++ backend, where <type> is e.g. 'Release' or 'Debug'. Then run 'python superconga.py simulate -C <config_path>' to run a simulation, where <config_path> is the configuration file path, e.g. any of the example directories in 'examples/'."
    )

    # Version number.
    parser.add_argument("-V", "--version", action="version", version="v1.0.0")

    # Create subparsers.
    subparsers = parser.add_subparsers(
        help="SuperConga subcommands. Run 'superconga.py <subcommand> --help' for help about the specific subcommand.",
    )

    # Add setup parser.
    setup_parser = subparsers.add_parser("setup")
    setup.make_parser(setup_parser)

    # Add compilation parser.
    compilation_parser = subparsers.add_parser("compile")
    compilation.make_parser(compilation_parser)

    # Add simulation parser.
    simulation_parser = subparsers.add_parser("simulate")
    simulation.make_parser(simulation_parser)

    # Add postprocess parser.
    postprocess_parser = subparsers.add_parser("postprocess")
    postprocess.make_parser(postprocess_parser)

    # Add simulation plotter parser.
    simulation_plotter_parser = subparsers.add_parser("plot-simulation")
    simulation_plotter.make_parser(simulation_plotter_parser)

    # Add postprocess plotter parser.
    postprocess_plotter_parser = subparsers.add_parser("plot-postprocess")
    postprocess_plotter.make_parser(postprocess_plotter_parser)

    # Add convergence plotter parser.
    convergence_plotter_parser = subparsers.add_parser("plot-convergence")
    convergence_plotter.make_parser(convergence_plotter_parser)

    # Add data-converter parser.
    data_converter_parser = subparsers.add_parser("convert")
    data_converter.make_parser(data_converter_parser)

    # Parse arguments.
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Print help if no subparser specified.
        parser.print_help()
    elif len(sys.argv) == 2:
        # Print help of subparser if no additional arguments specified.
        if args.which == "setup":
            setup_parser.print_help()
        elif args.which == "compilation":
            compilation_parser.print_help()
        elif args.which == "simulation":
            simulation_parser.print_help()
        elif args.which == "postprocess":
            postprocess_parser.print_help()
        elif args.which == "simulation_plotter":
            simulation_plotter_parser.print_help()
        elif args.which == "postprocess_plotter":
            postprocess_plotter_parser.print_help()
        elif args.which == "convergence_plotter":
            convergence_plotter_parser.print_help()
        elif args.which == "data_converter":
            data_converter_parser.print_help()
        else:
            print(f"-- ERROR! Command '{args.which}' not implemented.")
    else:
        # Run.
        if args.which == "setup":
            setup.main(args)
        elif args.which == "compilation":
            compilation.main(args)
        elif args.which == "simulation":
            simulation.main(args)
        elif args.which == "postprocess":
            postprocess.main(args)
        elif args.which == "simulation_plotter":
            simulation_plotter.main(args)
        elif args.which == "postprocess_plotter":
            postprocess_plotter.main(args)
        elif args.which == "convergence_plotter":
            convergence_plotter.main(args)
        elif args.which == "data_converter":
            data_converter.main(args)
        else:
            print(f"-- ERROR! Command '{args.which}' not implemented.")


if __name__ == "__main__":
    main()
