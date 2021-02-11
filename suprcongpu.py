#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Short description.

Long description.
"""

# Built-in modules and third-party libraries.
import argparse
import json
import os
import sys
from pathlib import Path

# Custom libraries.
sys.path.insert(0, "tools/")
from python_libraries.gui import gui
from python_libraries.parameter_setup import argument_parsing as AP
from python_libraries.parameter_setup import argument_validation as AV
from python_libraries.runner import binary_caller
from python_libraries.settings import configuration

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    # ======================================================================#
    #                            ARGUMENT PARSING                           #
    # ======================================================================#
    # Front-end is e.g. runner or plotter.
    frontend = "runner"
    # Program name and description for command-line help message.
    program_name = "SuprConGPU"
    program_description = "SuprConGPU Main file for setting up self-consistent simulations in a confined geometry (a mesoscopic grain). Different superconducting order parameters and geometries can be specified, as well as physical parameters such as temperature and external fields. To set these, create a .json parameter-file as in examples and give the path to this file with the -f option. All parameters can be overridden using the command-line options specified below."
    # Parse command-line arguments, and possibly a JSON file which specifies
    # config plotting parameters. The JSON file is provided via: -f|--file.
    simulation_parameters, parser_args = AP.parse_command_line_arguments(
        program_name=program_name,
        program_description=program_description,
        frontend=frontend,
    )

    # ======================================================================#
    #                   ARGUMENT VALIDATION AND PROCESSING                  #
    # ======================================================================#
    # Status of CPP binary execution.
    status_code = 0
    if parser_args.gui:
        status_code = gui.show_runner_gui(
            simulation_parameters=simulation_parameters,
            cpp_binary_path=parser_args.cpp_binary,
        )
    else:
        # Validate user input (e.g. check that data path exists).
        # Exit if there are errors.
        success = AV.validate_arguments(
            arguments_dict=simulation_parameters, frontend=frontend
        )
        if not success:
            sys.exit(1)

        # Launch binary.
        binary_caller.run_cpp_binary(parser_args.cpp_binary, simulation_parameters)

    # ======================================================================#
    #                            POST-PROCESSING                            #
    # ======================================================================#
    # Remove temporary config file.
    if Path(configuration.temp_config_file_name).is_file():
        os.remove(configuration.temp_config_file_name)

    # Quit the program.
    sys.exit(status_code)


if __name__ == "__main__":
    main()
