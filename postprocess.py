#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Short description.

Long description.
"""

# Built-in modules and third-party libraries.
import os
import sys
from pathlib import Path

# Custom libraries.
sys.path.insert(0, "tools/")
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
    frontend = "postprocess"
    # Program name and description for command-line help message.
    program_name = "SuprConGPU postprocess"
    program_description = "Compute the LDOS and DOS given a converged simulation."
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
