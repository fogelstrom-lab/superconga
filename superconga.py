#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Frontend for SuperConga: Superconductivity simulation framework on GPU.

SuperConga Main file for setting up self-consistent simulations in a confined
geometry (a mesoscopic grain). Different superconducting order parameters and
geometries can be specified, as well as physical parameters such as temperature
and external fields. To set these, create a .json parameter-file as in examples
and give the path to this file with the -f option. All parameters, except the geometry, 
can be overridden using the command-line options.
"""

# Built-in modules and third-party libraries.
import os
import sys
from pathlib import Path

# Custom libraries.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools/"))
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
    program_name = "SuperConga"
    program_description = "SuperConga Main file for setting up self-consistent simulations in a confined geometry (a mesoscopic grain). Different superconducting order parameters and geometries can be specified, as well as physical parameters such as temperature and external fields. To set these, create a .json parameter-file as in the folders in 'examples/', and give the path to this file with the -f option. All parameters can be overridden using the command-line options specified below."
    # Parse command-line arguments, and possibly a JSON file which specifies
    # config plotting parameters. The JSON file is provided via: -f|--file.
    simulation_parameters, cpp_info = AP.parse_command_line_arguments(
        program_name=program_name,
        program_description=program_description,
        frontend=frontend,
    )

    # ======================================================================#
    #                   ARGUMENT VALIDATION AND PROCESSING                  #
    # ======================================================================#
    # Status of CPP binary execution.
    status_code = 0

    # Validate user input (e.g. check that data path exists).
    # Exit if there are errors.
    success = AV.validate_arguments(
        arguments_dict=simulation_parameters, frontend=frontend
    )
    if not success:
        sys.exit(1)

    # Launch binary.
    cpp_path, cpp_precision = cpp_info
    if cpp_path is None:
        cpp_path = os.path.join(
            os.path.dirname(__file__),
            "build/release/bin/superconga_f" + str(cpp_precision),
        )
    binary_caller.run_cpp_binary(cpp_path, simulation_parameters)

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
