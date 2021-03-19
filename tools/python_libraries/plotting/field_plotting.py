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
from matplotlib.widgets import Slider, Button, RadioButtons, Cursor
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
    mask_nan_in=True,
    norm_in=None,
):
    # Mask the exterior of the grain by using a separate color.
    # Matplotlib gives warnings/errors if editing a colormap directly, and
    # suggests to use copy.copy() to edit the colormap.
    colormap = copy.copy(mpl.cm.get_cmap(colormap_in))
    if mask_nan_in:
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
    if norm_in == None:
        cax_out = ax_in.imshow(
            data_in,
            cmap=colormap,
            vmax=max_value_in,
            vmin=min_value_in,
            extent=extent_in,
            interpolation=interpolation_in,
            aspect=aspect_in,
            origin=origin_in,
        )
    else:
        cax_out = ax_in.imshow(
            data_in,
            cmap=colormap,
            extent=extent_in,
            interpolation=interpolation_in,
            aspect=aspect_in,
            origin=origin_in,
            norm=norm_in,
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
    if cbar_ax_in != None:
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
    else:
        cbar_out = None

    return cax_out, cbar_out


# -----------------------------------------------------------------------------
# PLOT ALL OBSERVABLES IN A COLLAGE
# -----------------------------------------------------------------------------
# Plot imshow collage of all observables/quantities indicated by plot_parameters.
# data is a dict of 2D numpy ndarrays (imshow expects the latter).
def plot_all(data: dict, geometry: dict, plot_parameters: dict, simulation_parameters: dict) -> None:
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

                # Mask data points outside grain for quantities that are
                # undefined there.
                if field_name in ["order_parameter", "current_density", "superflow"]:
                    data[field_name][component_name][geometry["mask"] == 0] = np.nan

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


# -----------------------------------------------------------------------------
# PLOT LDOS AND DOS
# -----------------------------------------------------------------------------
# Plot DOS as a line plot, and LDOS as 2D heatmap with imshow.
def plot_ldos(
    ldos_data: dict,
    dos_data: dict,
    plot_parameters: dict,
    simulation_parameters: dict,
    post_processing_parameters: dict,
    mask_nan_in=True,
):
    # --------------------------------------------------
    # FIGURE SETUP
    # --------------------------------------------------
    fig_width_scaling = 2.0
    fig_height_scaling = 1.3

    # Create figure object.
    fig = plt.figure(
        figsize=(
            PP.fig_width_inch * fig_width_scaling,
            PP.fig_height_inch * fig_height_scaling,
        ),
        dpi=PP.dpi,
    )

    # Create axis layout.
    gs = gridspec.GridSpec(70, 200)

    ax_LDOS = plt.subplot(gs[0:50, :100])
    ax_DOS = plt.subplot(gs[0:50, 110:])
    ax_slider = plt.subplot(gs[60:65, 33:83])
    ax_slider.set_facecolor("#ffffcc")

    ax_test = plt.subplot(gs[60:65, 110:])
    ax_test.axis("off")

    # Create axis object for heatmap colorbar.
    ax_cbar_LDOS = make_axes_locatable(ax_LDOS).append_axes("right", size="5%", pad=0.1)

    # Plot line at N=0 and N=1 in DOS figure.
    ax_DOS.axhline(y=0, c="gray", lw=1)
    # ax_DOS.axhline(y=1, c="gray", lw=1, ls="--")

    # --------------------------------------------------
    # PLOT LDOS
    # --------------------------------------------------
    # Compute extent: scale coordinates to coherence lengths.
    y_num, x_num = np.shape(ldos_data["ldos"][0])
    x_min = 0
    x_max = x_num / simulation_parameters["points_per_coherence_length"]
    y_min = 0
    y_max = y_num / simulation_parameters["points_per_coherence_length"]
    extent = [x_min, x_max, y_min, y_max]

    # Energy index to plot LDOS at.
    LDOS_energy_idx = 0
    vline1 = ax_DOS.axvline(
        x=dos_data["energy"][LDOS_energy_idx], c="0.75", lw=1.5, ls="--"
    )

    LDOS_max = np.nanmax(ldos_data["ldos"][LDOS_energy_idx])
    LDOS_min = 0

    cbar_ticks = [LDOS_min, LDOS_max]
    cbar_tick_labels = [r"$%d$" % LDOS_min, r"$%2.1f$" % LDOS_max]

    # Plot LDOS.
    ldos_kwargs = {
        "fig_in": fig,
        "ax_in": ax_LDOS,
        "cbar_ax_in": ax_cbar_LDOS,
        "colormap_in": "YlGnBu",
        "max_value_in": LDOS_max,
        "min_value_in": LDOS_min,
        "extent_in": extent,
        "cbar_ticks_in": cbar_ticks,
        "cbar_tick_labels_in": cbar_tick_labels,
        "cbar_label_in": r"$N / 2N_{\mathrm{F}}$",
        "cbar_label_pad_in": -20,
        "mask_nan_in": mask_nan_in,
    }
    cax_LDOS, cbar_LDOS = plot_2D_imshow(
        data_in=ldos_data["ldos"][LDOS_energy_idx], **ldos_kwargs
    )

    # --------------------------------------------------
    # PLOT DOS
    # --------------------------------------------------
    dos_slice_x_idx = int(x_num * 0.5)
    dos_slice_y_idx = int(y_num * 0.5)

    ax_DOS.plot(
        dos_data["energy"],
        dos_data["dos"],
        "k-",
        label=r"$\text{Area-averaged\, DOS}$",
    )
    ax_DOS.plot(
        -np.flip(dos_data["energy"]),
        np.flip(dos_data["dos"]),
        "k-",
    )

    (DOS_p1,) = ax_DOS.plot(
        dos_data["energy"],
        ldos_data["ldos"][:, dos_slice_y_idx, dos_slice_x_idx],
        "r-",
        label=r"$\mathrm{Point\, DOS},\, x=%2.1f\xi_0,\, y=%2.1f\xi_0$"
        % (
            dos_slice_x_idx / simulation_parameters["points_per_coherence_length"],
            dos_slice_y_idx / simulation_parameters["points_per_coherence_length"],
        ),
    )
    (DOS_p2,) = ax_DOS.plot(
        -np.flip(dos_data["energy"]),
        np.flip(ldos_data["ldos"][:, dos_slice_y_idx, dos_slice_x_idx]),
        "r-",
    )

    """
    (DOS_y1,) = ax_DOS.plot(
        dos_data["energy"],
        np.average(ldos_data["ldos"][:, dos_slice_y_idx, :], axis=1),
        "b-",
        label=r"$x\mathrm{-averaged\, DOS},\, y_i=%d$" % (dos_slice_y_idx),
    )
    (DOS_y2,) = ax_DOS.plot(
        -np.flip(dos_data["energy"]),
        np.flip(np.average(ldos_data["ldos"][:, dos_slice_y_idx, :], axis=1)),
        "b-",
    )

    (DOS_x1,) = ax_DOS.plot(
        dos_data["energy"],
        np.average(ldos_data["ldos"][:, :, dos_slice_x_idx], axis=1),
        "m-",
        label=r"$y\mathrm{-averaged\, DOS},\, x_i=%d$" % (dos_slice_x_idx),
    )
    (DOS_x2,) = ax_DOS.plot(
        -np.flip(dos_data["energy"]),
        np.flip(np.average(ldos_data["ldos"][:, :, dos_slice_x_idx], axis=1)),
        "m-",
    )"""

    # Marker indicating point where "point-LDOS" is plotted.
    (LDOS_marker,) = ax_LDOS.plot(
        dos_slice_x_idx / simulation_parameters["points_per_coherence_length"],
        dos_slice_y_idx / simulation_parameters["points_per_coherence_length"],
        lw=0,
        ls=None,
        marker="x",
        ms=7,
        markeredgecolor="r",
        markerfacecolor="none",
        markeredgewidth=1.5,
    )

    # --------------------------------------------------
    # TIDY UP LDOS PLOT
    # --------------------------------------------------

    LDOS_text = ax_LDOS.text(
        0.02,
        1.01,
        r"$\text{(a)\, LDOS},\, \epsilon=%2.1fk_{\mathrm{B}}T_{\mathrm{c}}$"
        % (dos_data["energy"][LDOS_energy_idx]),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax_LDOS.transAxes,
        fontsize=PP.font_size - 2,
        # bbox=props,
    )

    # --------------------------------------------------
    # TIDY UP DOS PLOT
    # --------------------------------------------------
    # Set x- and y-labels.
    ax_DOS.set_ylabel(r"$N / 2N_{\mathrm{F}}$", fontsize=PP.font_size)
    ax_DOS.set_xlabel(r"$\epsilon/k_{\mathrm{B}}T_{\mathrm{c}}$", fontsize=PP.font_size)
    # Adjust ticks and font size of tick labels.
    ax_DOS.tick_params(axis="both", which="major", labelsize=PP.tick_size)
    ax_DOS.xaxis.set_ticks_position("bottom")
    ax_DOS.yaxis.set_ticks_position("left")
    ax_DOS.set_ylim([0, 5])
    ax_DOS.set_xlim([-dos_data["energy"][-1], dos_data["energy"][-1]])
    # ax_DOS.set_yticks([0, 4])
    # ax_DOS.set_xticks([0, 10, 20, 30, 40, 50])

    ax_DOS.text(
        0.02,
        1.01,
        r"$\text{(b)\, DOS}$",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax_DOS.transAxes,
        fontsize=PP.font_size - 2,
        # bbox=props,
    )

    DOS_legend = ax_DOS.legend(
        prop={"size": PP.font_size - 4},
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.1,
        # borderaxespad=1.0,
        # fancybox=True,
        # handlelength=3.0,
        framealpha=0.0,
        # handletextpad=1.5,
    )

    ax_test.text(
        0.5,
        0.5,
        r"$\mathrm{Click\, in\, the\, heatmap\, to\, plot\, LDOS\, at\, that\, point}$",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax_test.transAxes,
        fontsize=PP.font_size - 2,
        # bbox=props,
    )

    # --------------------------------------------------
    # ADD INTERACTIVE SLIDER
    # --------------------------------------------------
    energy_min = dos_data["energy"][0]
    energy_max = dos_data["energy"][-1]
    energy_num = len(dos_data["energy"])
    delta_energy = np.diff(dos_data["energy"])[0]

    # Define an axes area and draw a slider in it
    energy_slider = Slider(
        ax_slider,
        r"$\mathrm{Choose\, energy}\, (k_{\mathrm{B}}T_{\mathrm{c}})$",
        dos_data["energy"][0],
        dos_data["energy"][-1],
        valinit=dos_data["energy"][0],
        valfmt="%2.1f",
        valstep=delta_energy,
    )
    energy_slider.label.set_fontsize(PP.font_size - 4)
    # energy_slider.val.set_size(PP.font)

    def update(val):
        LDOS_energy = energy_slider.val
        LDOS_energy_idx = int(LDOS_energy / delta_energy)
        update_ldos_data = ldos_data["ldos"][LDOS_energy_idx]

        LDOS_max = np.nanmax(update_ldos_data)
        LDOS_min = 0

        cbar_ticks = [LDOS_min, LDOS_max]
        cbar_tick_labels = [r"$%d$" % LDOS_min, r"$%2.1f$" % LDOS_max]

        # Update data
        cax_LDOS.set_data(update_ldos_data)
        cax_LDOS.set_clim(vmin=LDOS_min, vmax=LDOS_max)

        # Update colorbar ticks and tick labels.
        cbar_LDOS.set_ticks(cbar_ticks)
        cbar_LDOS.set_ticklabels(cbar_tick_labels)

        # Update text label.

        LDOS_text.set_text(
            r"$\text{(a)\, LDOS},\, \epsilon=%2.1fk_{\mathrm{B}}T_{\mathrm{c}}$"
            % LDOS_energy
        )

        # Update horizontal bar.
        vline1.set_xdata(LDOS_energy)

        fig.canvas.draw()

    energy_slider.on_changed(update)

    # --------------------------------------------------
    # ADD INTERACTIVE CLICKING
    # --------------------------------------------------
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            if event.inaxes == ax_LDOS:
                # Update coordinates.
                dos_slice_x_idx = int(
                    event.xdata * simulation_parameters["points_per_coherence_length"]
                )
                dos_slice_y_idx = int(
                    event.ydata * simulation_parameters["points_per_coherence_length"]
                )

                # Update point-LDOS plot.
                DOS_p1.set_ydata(ldos_data["ldos"][:, dos_slice_y_idx, dos_slice_x_idx])
                DOS_p2.set_ydata(
                    np.flip(ldos_data["ldos"][:, dos_slice_y_idx, dos_slice_x_idx])
                )

                # Update marker position in heatmap.
                LDOS_marker.set_data((event.xdata, event.ydata))

                # Update coordinates in label.
                DOS_legend.get_texts()[-1].set_text(
                    r"$\mathrm{Point\, DOS},\, x=%2.1f\xi_0,\, y=%2.1f\xi_0$"
                    % (event.xdata, event.ydata)
                )

                # Re-draw figure.
                fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # --------------------------------------------------
    # FIGURE LAYOUT
    # --------------------------------------------------

    fig.subplots_adjust(
        top=0.95, bottom=0.0, left=0.0, right=0.99, hspace=0.2, wspace=0.2
    )

    # Return reference to energy slider. This is to prevent object to be
    # garbage collected, rendering the slider inactive.
    return energy_slider