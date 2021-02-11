#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Plot observables/quantities from SuprConGPU."""

# Built-in modules and third-party libraries.
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries.file_io import utils
from python_libraries.plotting import colormaps_extra
from python_libraries.plotting import plotting_parameters as PP
from python_libraries.settings import configuration

# Global settings.

# Register extra colormaps.
colormaps_extra.reg_cmaps()

# Initiate latex plotting with rcParams.
PP.init_latex_plotting()

# -----------------------------------------------------------------------------
# IMSHOW 2D PLOT: GENERAL
# -----------------------------------------------------------------------------
def plot_2D_imshow(
    fig_in,
    ax_in,
    cbar_ax_in,
    data_in,
    colormap_in,
    max_value_in,
    min_value_in,
    extent_in,
    cbar_ticks_in,
    cbar_tick_labels_in,
    cbar_label_in,
    x_label_in=r"$x/\xi_0$",
    y_label_in=r"$y/\xi_0$",
    interpolation_in="none",
    aspect_in=1.0,
    origin_in="lower",
    cbar_orientation_in="vertical",
    cbar_label_pad_in=0,
    mask_nan=True,
):
    # Mask the exterior of the grain by using a separate color.
    # Matplotlib gives warnings/errors if editing a colormap directly, and
    # suggests to use copy.copy() to edit the colormap.
    colormap = copy.copy(mpl.cm.get_cmap(colormap_in))
    if mask_nan:
        # Use separate color for NaN entries.
        colormap.set_bad(PP.mask_color, 1.0)

    # --------------------------------------------------
    # PLOT
    # --------------------------------------------------
    # Plot 2D heatmap with imshow. Imshow expects regularly spaced data as a
    # numpy array, where each entry denotes the amplitude. The x- and y-indices
    # of the value automatically correspond to discrete coordinates. To convert
    # to physical coordinates, use extent. Interpolation can be set if desired,
    # but should generally be avoided as it might give an incorrect
    # representation/impression of the data. To this end, aspect should also be
    # set to 1.0 to avoid arbitrary scaling/distortion. By default, origin is
    # in top left corner, but here we change default to lower left corner.
    cax_out = ax_in.imshow(
        data_in,
        cmap=colormap_in,
        vmax=max_value_in,
        vmin=min_value_in,
        extent=extent_in,
        interpolation=interpolation_in,
        aspect=aspect_in,
        origin=origin_in,
    )

    # --------------------------------------------------
    # AXES SETTINGS
    # --------------------------------------------------
    # Set x- and y-labels.
    ax_in.set_ylabel(y_label_in, fontsize=PP.font_size)
    ax_in.set_xlabel(x_label_in, fontsize=PP.font_size)
    # Adjust ticks and font size of tick labels.
    ax_in.tick_params(axis="both", which="major", labelsize=PP.tick_size)
    ax_in.xaxis.set_ticks_position("bottom")
    ax_in.yaxis.set_ticks_position("left")

    # --------------------------------------------------
    # COLORBAR SETTINGS
    # --------------------------------------------------
    # Create colorbar.
    cbar_out = fig_in.colorbar(
        cax_out,
        cax=cbar_ax_in,
        orientation=cbar_orientation_in,
        ticks=cbar_ticks_in,
    )
    # Set colorbar ticks and tick labels.
    cbar_out.set_ticks(cbar_ticks_in)
    cbar_out.set_ticklabels(cbar_tick_labels_in)
    # Set colorbar label.
    cbar_out.set_label(
        cbar_label_in,
        fontsize=PP.font_size,
        labelpad=cbar_label_pad_in,
    )
    # Adjust ticks and font size of tick labels.
    cbar_out.ax.tick_params(axis="both", which="major", labelsize=PP.cbar_size)
    cbar_out.ax.xaxis.set_ticks_position("top")
    cbar_out.ax.xaxis.set_label_position("top")
    # Make the colorbar appear nicer in vectorized representation.
    cbar_out.solids.set_edgecolor("face")

    return cax_out, cbar_out


# -----------------------------------------------------------------------------
# PLOT ALL OBSERVABLES IN A COLLAGE
# -----------------------------------------------------------------------------
# Plot imshow collage of all observables/quantities indicated by plot_parameters.
# data is a dict of 2D numpy ndarrays (imshow expects the latter).
def plot_all(data: dict, plot_parameters: dict, simulation_parameters: dict) -> None:
    # --------------------------------------------------
    # FIX PLOT LAYOUT: GRID OF IMSHOW PLOTS
    # --------------------------------------------------
    # Calculate number of rows/columns of plots. Each column contains one
    # observable/quantity (except induced B-field which is plotted below OP).
    # There are a maximum of 3 rows, since the maximum number of components is
    # 3 (for vector fields).
    num_plot_rows = 3
    num_plot_columns = 0
    for field_name in sorted(configuration.grid_field_types.keys()):
        if plot_parameters[field_name]["plot"]:
            num_plot_columns += 1
    # OP and magnetic field is plotted in the same column. Reduce double count.
    if (
        plot_parameters["order_parameter"]["plot"]
        and plot_parameters["magnetic_flux_density"]["plot"]
    ):
        num_plot_columns -= 1

    # Setup figure object.
    fig_width_scaling = 0.7 * num_plot_columns
    fig_height_scaling = 0.9 * num_plot_rows

    fig, axes = plt.subplots(
        nrows=num_plot_rows,
        ncols=num_plot_columns,
        figsize=(
            PP.fig_width_inch * fig_width_scaling,
            PP.fig_height_inch * fig_height_scaling,
        ),
        dpi=PP.dpi,
    )

    # Make axes iterable.
    ax = {}
    cbar_ax = {}

    # --------------------------------------------------
    # PLOT LOOP
    # --------------------------------------------------
    active_plots = []
    # Loop over fields.
    column_idx = 0
    for field_name in configuration.grid_field_types.keys():
        if plot_parameters[field_name]["plot"]:
            # Get the field type: vector-, scalar- or complex- field.
            field_type = configuration.grid_field_types[field_name]
            ax[column_idx] = {}
            cbar_ax[column_idx] = {}

            # Loop over field_components.
            row_idx = 0
            for component_name in sorted(
                configuration.grid_field_components[field_type]
            ):
                # Plot magnetic flux density in same column as order parameter.
                if (
                    field_name == "magnetic_flux_density"
                    and plot_parameters["order_parameter"]["plot"]
                ):
                    row_idx = 2
                    column_idx = 0

                # Set axes locations.
                ax[column_idx][row_idx] = axes[row_idx][column_idx]

                # Add colorbar axis.
                cbar_ax[column_idx][row_idx] = make_axes_locatable(
                    ax[column_idx][row_idx]
                ).append_axes("right", size="5%", pad=0.1)

                # Keep track of which plots are active to hide inactive subplots.
                active_plots.append((row_idx, column_idx))

                # Normalize order parameter phase with pi.
                if field_name == "order_parameter" and component_name == "phase":
                    data[field_name][component_name] /= np.pi

                # Fetch data: compute max/min values and data ranges.
                max_value = np.nanmax(np.abs(data[field_name][component_name]))
                if component_name == "magnitude" and field_type != "scalar":
                    min_value = 0.0
                    cbar_ticks = [min_value, max_value]
                    cbar_tick_labels = [r"$0$", r"$%2.2f$" % max_value]
                    cbar_label_pad = -20
                else:
                    min_value = -max_value
                    cbar_ticks = [min_value, 0, max_value]
                    cbar_tick_labels = [
                        r"$%2.2f$" % min_value,
                        r"$0$",
                        r"$%2.2f$" % max_value,
                    ]
                    cbar_label_pad = -30

                # Compute extent: scale coordinates to coherence lengths.
                y_num, x_num = np.shape(data[field_name][component_name])
                x_min = 0
                x_max = x_num / simulation_parameters["points_per_coherence_length"]
                y_min = 0
                y_max = y_num / simulation_parameters["points_per_coherence_length"]
                extent = [x_min, x_max, y_min, y_max]

                # Draw 2D plot with imshow.
                plot_2D_imshow(
                    fig_in=fig,
                    ax_in=ax[column_idx][row_idx],
                    cbar_ax_in=cbar_ax[column_idx][row_idx],
                    data_in=data[field_name][component_name],
                    colormap_in=plot_parameters[field_name][component_name]["colormap"],
                    max_value_in=max_value,
                    min_value_in=min_value,
                    extent_in=extent,
                    cbar_ticks_in=cbar_ticks,
                    cbar_tick_labels_in=cbar_tick_labels,
                    cbar_label_in=plot_parameters[field_name][component_name][
                        "colorbar_label"
                    ],
                    cbar_label_pad_in=cbar_label_pad,
                )
                row_idx += 1
            column_idx += 1

    # Hide inactive subplots.
    for column_idx in range(0, num_plot_columns):
        for row_idx in range(0, num_plot_rows):
            if not (row_idx, column_idx) in active_plots:
                axes[row_idx][column_idx].axis("off")

    # Tight figure layout: adjust axes margins.
    fig.tight_layout()