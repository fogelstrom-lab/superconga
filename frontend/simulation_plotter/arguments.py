#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
Functionality for creating the argument parser for the simulation plotter.
"""

import argparse
from . import default


def make_parser(parser: argparse.ArgumentParser):
    """Given a parser, add all arguments for the simulation plotter."""

    parser.set_defaults(which="simulation_plotter")

    parser.description = (
        "Plot the simulation results produced by the 'simulate' command."
    )

    parser.add_argument(
        "-L",
        "--load-path",
        type=str,
        default=None,
        help="Path to data directory. (Default: %(default)s)",
    )

    parser.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Save the figure instead of showing it. E.g. '-S image.png' or '-S image.pdf'. (Default: %(default)s)",
    )

    parser.add_argument(
        "--view",
        type=str,
        nargs="+",
        choices=default.settings["view"],
        default=default.settings["view"],
        help="What to view. D = order parameter, J = current density, A = induced vector potential, B = magnetic flux density. (Default: %(default)s)",
    )

    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="Use Cartesian coordinates, i.e. (real, imag) and (x, y). "
        "Otherwise polar coordinates, (magnitude, phase) and (magnitude, angle), will be used. (Default: %(default)s)",
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

    size.add_argument(
        "--no-landscape",
        action="store_true",
        help="Tall, instead of wide, figure. (Default: %(default)s)",
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
        "--cmap-default",
        type=str,
        default=default.settings["colormaps"]["default"],
        help="Colormap for quantities that are not magnitudes or angles. (Default: %(default)s)",
    )

    color.add_argument(
        "--cmap-magnitude",
        type=str,
        default=default.settings["colormaps"]["magnitude"],
        help="Colormap for magnitudes. (Default: %(default)s)",
    )

    color.add_argument(
        "--cmap-angle",
        type=str,
        default=default.settings["colormaps"]["angle"],
        help="Colormap for angles. (Default: %(default)s)",
    )

    color.add_argument(
        "--exterior-color",
        type=str,
        default=default.settings["exterior_color"],
        help="Color for points outside of the grain. (Default: %(default)s)",
    )
