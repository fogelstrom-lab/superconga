#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Functionality for creating the argument parser for the postprocess plotter.
"""

import argparse

from . import default


def make_parser(parser: argparse.ArgumentParser):
    """Given a parser, add all arguments for the postprocess plotter."""

    parser.set_defaults(which="postprocess_plotter")

    parser.description = "Tool for exploring the postprocess data. The plots are interactive. Just click either plot to update the energy or point being visualized."

    parser.add_argument(
        "-L",
        "--load-path",
        required=True,
        type=str,
        default=None,
        help="Path to data directory. (Default: %(default)s)",
    )

    parser.add_argument(
        "--limit",
        type=float,
        default=default.settings["limit"],
        help="Upper limit of DoS plot y-axis. If negative it will be adjusted to the data. (Default: %(default)s)",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=default.settings["margin"],
        help="Margin along the y-axis in the DoS plot. A value of 0.1 means an extra 10%%. Only used if 'limit' is negative. (Default: %(default)s)",
    )

    # ==================== SIZE ==================== #

    size = parser.add_argument_group(title="Size")

    size.add_argument(
        "--subplot-width",
        type=float,
        default=default.settings["subplot_width"],
        help="The width of the subplots in inches (sigh...). (Default: %(default)s)",
    )

    size.add_argument(
        "--subplot-height",
        type=float,
        default=default.settings["subplot_height"],
        help="The height of the subplots in inches (sigh...). (Default: %(default)s)",
    )

    # ==================== TEXT ==================== #

    text = parser.add_argument_group(title="Text")

    text.add_argument(
        "--fontsize",
        type=int,
        default=default.settings["fontsize"],
        help="The fontsize to use. (Default: %(default)s)",
    )

    text.add_argument(
        "--no-alphabet",
        action="store_true",
        help="Do not add alphabetical labels, e.g. (a). (Default: %(default)s)",
    )

    text.add_argument(
        "--no-path",
        action="store_true",
        help="Do not show path of data directory in window title. (Default: %(default)s)",
    )

    text.add_argument(
        "--no-tex",
        action="store_true",
        help="Do not use TeX for text and math. (Default: %(default)s)",
    )

    # ==================== COLOR ==================== #

    color = parser.add_argument_group(
        title="Color",
        description="See https://matplotlib.org/stable/tutorials/colors/colormaps.html and https://matplotlib.org/stable/gallery/color/named_colors.html for all available colormaps and colors.",
    )

    color.add_argument(
        "--cmap",
        type=str,
        default=default.settings["colormap"],
        help="Colormap for the LDoS. (Default: %(default)s)",
    )

    color.add_argument(
        "--exterior-color",
        type=str,
        default=default.settings["exterior_color"],
        help="Color for points outside of the grain. (Default: %(default)s)",
    )

    color.add_argument(
        "--dos-color",
        type=str,
        default=default.settings["dos_color"],
        help="Color for the DoS plot. (Default: %(default)s)",
    )

    color.add_argument(
        "--ldos-color",
        type=str,
        default=default.settings["ldos_color"],
        help="Color for the LDoS plot. (Default: %(default)s)",
    )

    color.add_argument(
        "--marker-color",
        type=str,
        default=default.settings["marker_color"],
        help="Color for the marker in the LDoS image. (Default: %(default)s)",
    )
