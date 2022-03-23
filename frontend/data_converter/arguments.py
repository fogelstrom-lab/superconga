#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Functionality for creating the argument parser for the data-format converter.
"""

import argparse
import sys

sys.path.insert(0, "../")
from common import cli


def make_parser(parser: argparse.ArgumentParser):
    """Given a parser, add all arguments for the data-format converter."""

    parser.set_defaults(which="data_converter")

    parser.description = "Convert data produced by the 'simulate' or 'postprocess' commands between different data formats."

    parser.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Path to data directory. (Default: %(default)s)",
    )

    parser.add_argument(
        "--what",
        type=str,
        choices=["simulation", "postprocess"],
        default="simulation",
        help="What kind of data to convert. (Default: %(default)s)",
    )

    parser.add_argument(
        "--from",
        dest="src_data_format",
        type=str,
        choices=cli.VALID_DATA_FORMATS,
        default="csv",
        help="Data format before conversion. (Default: %(default)s)",
    )

    parser.add_argument(
        "--to",
        dest="dst_data_format",
        type=str,
        choices=cli.VALID_DATA_FORMATS,
        default="h5",
        help="Data format after conversion. (Default: %(default)s)",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the the old data if the conversion succeeded. Use with care! (Default: %(default)s)",
    )
