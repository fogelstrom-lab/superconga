#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
The main part of the postprocess plotter.
"""

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


def get_title_ldos(energy):
    """Given the energy get the title of the LDoS plot."""

    return (
        r"$N(\mathbf{R}, \epsilon) / (2 N_\mathrm{F}) \; | \; \epsilon = %0.2f [2\pi k_\mathrm{B} T_\mathrm{c}]$"
        % energy
    )


def get_title_dos(x, y):
    """Given the position, get the title of the DoS plot."""

    if np.isnan(x) or np.isnan(y):
        title = r"$N(\mathbf{R}, \epsilon) / (2 N_\mathrm{F}) \; | \; \mathbf{R} = (?, ?) [\xi_0]$"
    else:
        title = (
            r"$N(\mathbf{R}, \epsilon) / (2 N_\mathrm{F}) \; | \; \mathbf{R} = (%0.2f, %0.2f) [\xi_0]$"
            % (x, y)
        )
    return title


def run(
    data_dir: str,
    limit: Optional[float] = default.settings["limit"],
    margin: Optional[float] = default.settings["margin"],
    subplot_width: Optional[int] = default.settings["subplot_width"],
    subplot_height: Optional[int] = default.settings["subplot_height"],
    fontsize: Optional[int] = default.settings["fontsize"],
    alphabet: Optional[bool] = default.settings["alphabet"],
    show_path: Optional[bool] = default.settings["show_path"],
    use_tex: Optional[bool] = default.settings["use_tex"],
    colormap: Optional[dict] = default.settings["colormap"],
    exterior_color: Optional[str] = default.settings["exterior_color"],
    dos_color: Optional[str] = default.settings["dos_color"],
    ldos_color: Optional[str] = default.settings["ldos_color"],
    marker_color: Optional[str] = default.settings["marker_color"],
) -> None:
    """Run the postprocess plotter."""

    # Read configuration files.
    simulation_config = read.get_simulation_config(data_dir=data_dir)
    postprocess_config = read.get_postprocess_config(data_dir=data_dir)

    # Set global parameters, i.e. for all plots. Must be done before the figure is made.
    plotting.set_global_parameters(
        fontsize=fontsize,
        use_tex=use_tex,
        formatter_limits=(-3, 3),
    )

    # Make figure.
    rows, cols = (1, 2)
    name = data_dir if show_path else None
    fig, axes = plotting.make_figure(
        rows=rows,
        cols=cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
        squeeze=True,
        sharex=False,
        sharey=False,
        name=name,
    )

    # For convenience.
    ax_LDoS, ax_DoS = axes

    # Get the domain so we can color the outside of the domain gray.
    domain = read.get_data(data_dir=data_dir, key="domain")
    domain = domain.astype(float)
    domain[domain == 0] = np.nan

    # Extent.
    _, dim_y, dim_x = domain.shape
    ppxi = simulation_config["numerics"]["points_per_coherence_length"]
    extent = plotting.get_extent(ppxi=ppxi, dim_x=dim_x, dim_y=dim_y)

    # Get the LDoS.
    LDoS = read.get_data(data_dir=data_dir, key="ldos")
    LDoS *= domain

    # Compute the DoS.
    DoS = np.nanmean(LDoS, axis=(1, 2))
    DoS_max = np.amax(DoS)

    energy_min = postprocess_config["spectroscopy"]["energy_min"]
    energy_max = postprocess_config["spectroscopy"]["energy_max"]
    num_energies = postprocess_config["spectroscopy"]["num_energies"]
    energies = np.linspace(energy_min, energy_max, num_energies)
    energy_diff = np.diff(energies)[0]

    # ============ LDoS subplot ============ #

    # Plot LDOS.
    energy_idx = 0
    image_LDoS, cbar_LDoS = plotting.imshow_and_colorbar(
        fig=fig,
        ax=ax_LDoS,
        data=np.squeeze(LDoS[energy_idx, :, :]),
        label=None,
        extent=extent,
        colormap=colormap,
        plot_type=plotting.PlotType.POSITIVE,
        bad_color=exterior_color,
    )

    # Set ticks and labels, e.g. x/\xi_0.
    plotting.set_axes_ticks_and_labels(axes=ax_LDoS, extent=extent)

    # Marker for point LDoS in the DoS subplot.
    (point_marker,) = ax_LDoS.plot(
        np.NaN,
        np.NaN,
        linestyle=":",
        marker="X",
        markersize=10,
        markerfacecolor=marker_color,
        markeredgecolor="white",
    )

    # Title text with the current energy.
    title_LDoS = ax_LDoS.text(
        0.5,
        1.02,
        get_title_ldos(energy=energies[energy_idx]),
        transform=ax_LDoS.transAxes,
        ha="center",
    )

    # ============ DoS subplot ============ #

    # Massage margin.
    margin += 1

    # Plot DoS.
    ax_DoS.plot(
        np.concatenate([-np.flip(energies), energies]),
        np.concatenate([np.flip(DoS), DoS]),
        color=dos_color,
        label=r"$N(\epsilon)$",
    )
    ax_DoS.set_ylabel(r"$N / (2N_{\mathrm{F}})$")
    ax_DoS.set_xlabel(r"$\epsilon/(2 \pi k_{\mathrm{B}} T_{\mathrm{c}})$")
    if limit > 0:
        ax_DoS.set_ylim([0, limit])
    else:
        ax_DoS.set_ylim([0, DoS_max * margin])
    ax_DoS.set_xlim([-energy_max, energy_max])

    # LDoS at specific point.
    LDoS_point = np.zeros_like(DoS) * np.NaN

    # Create plot for LDoS at specific point.
    (plot_LDoS,) = ax_DoS.plot(
        np.concatenate([-np.flip(energies), energies]),
        np.concatenate([np.flip(LDoS_point), LDoS_point]),
        color=ldos_color,
        label=r"$N(\mathbf{R},\epsilon)$",
    )

    # Vertical line marking the current energy.
    energy_marker = ax_DoS.axvline(
        x=energies[energy_idx],
        color="gray",
        linestyle="--",
    )

    # Title text with the current point.
    title_DoS = ax_DoS.text(
        0.5,
        1.02,
        get_title_dos(x=np.NaN, y=np.NaN),
        transform=ax_DoS.transAxes,
        ha="center",
    )

    # Create legend.
    ax_DoS.legend(loc="upper right")

    # Set alphabetic label, e.g. (a) at the top left.
    if alphabet:
        plotting.set_alphabetic_labels(axes=axes)

    # --------------------------------------------------
    # ADD INTERACTIVE CLICKING
    # --------------------------------------------------
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            if event.inaxes == ax_LDoS:
                # Compute coordinates.
                x_scale = (dim_x - 1) / (extent[1] - extent[0])
                y_scale = (dim_y - 1) / (extent[3] - extent[2])

                x_idx = int((event.xdata - extent[0]) * x_scale)
                y_idx = int((event.ydata - extent[2]) * y_scale)

                # Update point-LDOS data.
                LDoS_point = LDoS[:, y_idx, x_idx]

                # Update point-LDoS plot.
                plot_LDoS.set_ydata([np.flip(LDoS_point), LDoS_point])

                # Update y-axis limits.
                if limit < 0:
                    # TODO(Niclas): Handle "RuntimeWarning: All-NaN slice encountered", i.e. set LDoS_max to zero.
                    LDoS_max = np.nanmax(LDoS_point)
                    if not np.isfinite(LDoS_max):
                        LDoS_max = 0
                    ylim = margin * max(LDoS_max, DoS_max)
                    ax_DoS.set_ylim([0, ylim])

                # Update point marker.
                x = x_idx / x_scale + extent[0]
                y = y_idx / x_scale + extent[2]
                point_marker.set_data(x, y)

                # Update DoS title.
                title_DoS.set_text(get_title_dos(x=x, y=y))

                # Re-draw figure.
                fig.canvas.draw()

            if event.inaxes == ax_DoS:
                # Compute index.
                energy_idx = int(abs(event.xdata) / energy_diff)

                # Compute actual energy.
                energy = np.sign(event.xdata) * energies[energy_idx]

                # Update LDoS title.
                title_LDoS.set_text(get_title_ldos(energy=energy))

                # LDoS for particular energy.
                slice_LDoS = np.squeeze(LDoS[energy_idx, :, :])

                # Min/max values.
                vmax = np.nanmax(slice_LDoS)
                vmin = 0

                # Update data
                image_LDoS.set_data(slice_LDoS)
                image_LDoS.set_clim(vmin=vmin, vmax=vmax)

                # Update colorbar ticks and tick labels.
                plotting.set_colorbar_ticks_and_labels(
                    colorbar=cbar_LDoS, vmin=vmin, vmax=vmax
                )

                # Update vertical line.
                energy_marker.set_xdata(energy)

                # Re-draw figure.
                fig.canvas.draw()

    # To save some space.
    fig.tight_layout()

    # Create interactive event handling.
    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # Show the plot.
    plt.show()

    # Close stuff.
    plt.close()


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Main entry of the postprocess plotter."""

    # Run the plotter.
    run(
        data_dir=args.load_path,
        limit=args.limit,
        margin=args.margin,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
        fontsize=args.fontsize,
        alphabet=(not args.no_alphabet),
        show_path=(not args.no_path),
        use_tex=(not args.no_tex),
        colormap=args.cmap,
        exterior_color=args.exterior_color,
        dos_color=args.dos_color,
        ldos_color=args.ldos_color,
        marker_color=args.marker_color,
    )
