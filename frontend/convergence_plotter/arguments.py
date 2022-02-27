#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import argparse
from . import default


def make_parser(parser: argparse.ArgumentParser):
    parser.set_defaults(which="convergence_plotter")

    parser.description = "Plot the residuals (and free energy) versus iteration number for a computation, either simulation or postprocess."

    parser.add_argument(
        "-L",
        "--load-path",
        required=True,
        type=str,
        default=None,
        help="Path to data directory. (Default: %(default)s)",
    )

    parser.add_argument(
        "-S",
        "--save-path",
        type=str,
        default=None,
        help="Save the figure instead of showing it. (Default: %(default)s)",
    )

    parser.add_argument(
        "--view",
        type=str,
        choices=["simulation", "postprocess"],
        default=default.settings["view"],
        help="What to view. (Default: %(default)s)",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=default.settings["sigma"],
        help="Standard deviation of Gaussian blur. If zero or less no blur will be applied. (Default: %(default)s)",
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
        description="See https://matplotlib.org/stable/gallery/color/named_colors.html for all available colors.",
    )

    color.add_argument(
        "--boundary-coherence-color",
        type=str,
        default=default.settings["colors"]["boundary_coherence_residual"],
        help="Color for the boundary coherence. (Default: %(default)s)",
    )

    color.add_argument(
        "--order-parameter-color",
        type=str,
        default=default.settings["colors"]["order_parameter_residual"],
        help="Color for the order parameter. (Default: %(default)s)",
    )

    color.add_argument(
        "--current-density-color",
        type=str,
        default=default.settings["colors"]["current_density_residual"],
        help="Color for the charge-current density. (Default: %(default)s)",
    )

    color.add_argument(
        "--vector-potential-color",
        type=str,
        default=default.settings["colors"]["vector_potential_residual"],
        help="Color for the vector potential. (Default: %(default)s)",
    )

    color.add_argument(
        "--free-energy-color",
        type=str,
        default=default.settings["colors"]["free_energy_residual"],
        help="Color for the free energy. (Default: %(default)s)",
    )

    color.add_argument(
        "--ldos-color",
        type=str,
        default=default.settings["colors"]["ldos_residual"],
        help="Color for the LDoS. (Default: %(default)s)",
    )

    color.add_argument(
        "--alpha",
        type=float,
        default=default.settings["alpha"],
        help="Alpha value, i.e. transparency, for the actual signals when using blur. (Default: %(default)s)",
    )
