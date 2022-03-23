#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Functionality for creating the argument parser for the compilation.
"""

import argparse
import sys

sys.path.insert(0, "../")
from common import cli


def make_parser(parser: argparse.ArgumentParser):
    """Given a parser, add all arguments for the compilation."""

    parser.set_defaults(which="compilation")

    parser.description = "Compile a SuperConga build. Note that 'setup' must be run before 'compile'. When completed the C++ backend is ready to rumble."

    parser.add_argument(
        "--type",
        dest="build_type",
        type=str,
        choices=["Debug", "Release"],
        default="Release",
        help="What build to compile. (Default: %(default)s)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after the compilation. (Default: %(default)s)",
    )
