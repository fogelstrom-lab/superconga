#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Functionality for creating the argument parser for the setup tool.
"""

import argparse
import sys

sys.path.insert(0, "../")
from common import cli


def make_parser(parser: argparse.ArgumentParser):
    """Given a parser, add all arguments for the setup tool."""

    parser.set_defaults(which="setup")

    parser.description = "Setup a SuperConga build."

    parser.add_argument(
        "--type",
        dest="build_type",
        type=str,
        choices=["Debug", "Release"],
        default="Release",
        help="What build to setup. (Default: %(default)s)",
    )

    parser.add_argument(
        "--cuda-arch",
        type=int,
        default=None,
        help="If the CUDA architecture auto-detection fails you can specify the architecture manually, e.g. 61 or 75. (Default: %(default)s)",
    )

    parser.add_argument(
        "--no-ninja",
        action="store_true",
        help="Do not use Ninja build system, let CMake decide instead. (Default: %(default)s)",
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Build without visualization support. (Default: %(default)s)",
    )

    parser.add_argument(
        "--use-cuda-opengl-interop",
        action="store_true",
        help="Build with CUDA-OpenGL interopability. If using a single GPU then this yields faster visualization. (Default: %(default)s)",
    )

    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="Build without HDF5 support. (Default: %(default)s)",
    )

    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Do not build tests. (Default: %(default)s)",
    )
