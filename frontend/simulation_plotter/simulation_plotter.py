#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import Optional

from . import default
from .arguments import make_parser

sys.path.insert(0, "../")
from common import plotting, read


def get_rows_cols(view: list, num_op_comps: int) -> tuple:
    rows = 0
    cols = 0
    for what in view:
        if what == "D":
            rows += num_op_comps
            cols = max(cols, 2)
        elif what == "J":
            rows += 1
            cols = max(cols, 2)
        elif what == "A":
            rows += 1
            cols = max(cols, 2)
        elif what == "B":
            rows += 1
            cols = max(cols, 1)
        else:
            raise ValueError(f"-- ERROR! View '{what}' is not understood.")
    return rows, cols


def run(
    data_dir: str,
    save_path: Optional[str] = None,
    view: Optional[list] = default.settings["view"],
    cartesian: Optional[bool] = default.settings["cartesian"],
    subplot_width: Optional[int] = default.settings["subplot_width"],
    subplot_height: Optional[int] = default.settings["subplot_height"],
    landscape: Optional[bool] = default.settings["landscape"],
    fontsize: Optional[int] = default.settings["fontsize"],
    alphabet: Optional[bool] = default.settings["alphabet"],
    show_path: Optional[bool] = default.settings["show_path"],
    use_tex: Optional[bool] = default.settings["use_tex"],
    colormaps: Optional[dict] = default.settings["colormaps"],
    exterior_color: Optional[str] = default.settings["exterior_color"],
) -> None:
    # Read configuration file.
    config = read.get_simulation_config(data_dir=data_dir)

    # Set global parameters, i.e. for all plots. Must be done before the figure is made.
    plotting.set_global_parameters(fontsize=fontsize, use_tex=use_tex)

    # Compute the number of needed rows and columns.
    num_op_comps = len(config["physics"]["order_parameter"])
    rows, cols = get_rows_cols(view=view, num_op_comps=num_op_comps)

    # Make figure.
    if landscape:
        rows, cols = cols, rows
    name = data_dir if show_path else None
    fig, axes = plotting.make_figure(
        rows=rows,
        cols=cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
        sharex="all",
        sharey="all",
        name=name,
    )

    # Get the domain so we can color the outside of the domain gray.
    domain = read.get_data(data_dir=data_dir, key="domain")
    domain = domain.astype(float)
    domain[domain == 0] = np.nan

    # Extent.
    _, dim_y, dim_x = domain.shape
    ppxi = config["numerics"]["points_per_coherence_length"]
    extent = plotting.get_extent(ppxi=ppxi, dim_x=dim_x, dim_y=dim_y)

    # Loop over plot objects and plot them.
    idx = 0
    for what in view:
        # Get the actual keys and whether to "erase" the values outside of the domain.
        if what == "D":
            keys = []
            for symmetry in config["physics"]["order_parameter"].keys():
                keys.append("order_parameter_" + symmetry)
            erase = True
        elif what == "J":
            keys = ["current_density"]
            erase = True
        elif what == "A":
            keys = ["vector_potential"]
            erase = False
        elif what == "B":
            keys = ["magnetic_flux_density"]
            erase = False
        else:
            continue

        # Loop over keys and plot the data.
        for key in keys:
            # Extract data from file.
            data = read.get_data(data_dir=data_dir, key=key)

            # Erase data outside of the domain, i.e. set to NaN.
            if erase:
                data *= domain

            # Get labels.
            labels = (
                default.labels[key]["cartesian"]
                if cartesian
                else default.labels[key]["polar"]
            )

            # Plot a row (or column if landscape == True).
            plotting.make_plot_row(
                fig=fig,
                axes=axes,
                idx=idx,
                data=data,
                extent=extent,
                labels=labels,
                colormaps=colormaps,
                landscape=landscape,
                cartesian=cartesian,
                bad_color=exterior_color,
            )

            # Increase index.
            idx += 1

    # Set ticks and labels, e.g. x/\xi_0.
    plotting.set_axes_ticks_and_labels(axes=axes, extent=extent)

    # Set alphabetic label, e.g. (a) at the top left.
    if alphabet:
        plotting.set_alphabetic_labels(axes=axes)

    # To save some space.
    fig.tight_layout()

    # Show or save.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f"-- Wrote plot to {save_path}")

    # Close stuff.
    plt.close()


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    colormaps = {
        "default": args.cmap_default,
        "magnitude": args.cmap_magnitude,
        "angle": args.cmap_angle,
    }

    # Run the plotter.
    run(
        data_dir=args.load_path,
        save_path=args.save_path,
        view=args.view,
        cartesian=args.cartesian,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
        landscape=(not args.no_landscape),
        fontsize=args.fontsize,
        alphabet=(not args.no_alphabet),
        show_path=(not args.no_path),
        use_tex=(not args.no_tex),
        colormaps=colormaps,
        exterior_color=args.exterior_color,
    )
