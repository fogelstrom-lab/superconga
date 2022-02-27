#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

from enum import Enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_figure(
    rows,
    cols,
    subplot_width=3.5,
    subplot_height=2.9,
    dpi=100,
    squeeze=False,
    sharex=False,
    sharey=False,
    name=None,
):
    # Compute figure size.
    width = cols * subplot_width
    height = rows * subplot_height

    # Set name.
    if name is None:
        name = "SuperConga"
    else:
        name = "SuperConga - " + name

    return plt.subplots(
        rows,
        cols,
        figsize=(
            width,
            height,
        ),
        dpi=dpi,
        squeeze=squeeze,
        sharex=sharex,
        sharey=sharey,
        num=name,
    )


def get_extent(ppxi, dim_x, dim_y):
    lattice_site_length = 1 / ppxi
    x_lim = lattice_site_length * (dim_x - 1) / 2
    y_lim = lattice_site_length * (dim_y - 1) / 2
    return [-x_lim, x_lim, -y_lim, y_lim]


class PlotType(Enum):
    DEFAULT = 1
    MAGNITUDE = 2
    ANGLE = 3


def imshow_and_colorbar(
    fig,
    ax,
    data,
    label,
    extent,
    colormap,
    plot_type=PlotType.DEFAULT,
    epsilon=1e-6,
    origin="lower",
    interpolation="none",
    aspect=1.0,
    bad_color="dimgray",
    colorbar_fraction=0.046,
    colorbar_pad=0.04,
) -> None:
    # Get colormap and ranges from plot type.
    if plot_type == PlotType.MAGNITUDE:
        vmax = max(np.nanmax(data), epsilon)
        vmin = 0
    elif plot_type == PlotType.ANGLE:
        vmax = np.pi
        vmin = -np.pi
    else:
        tmp = max(abs(np.nanmax(data)), abs(np.nanmin(data)))
        vmax = max(tmp, epsilon)
        vmin = -vmax

    # Set NaN color.
    colormap = mpl.cm.get_cmap(colormap).copy()
    colormap.set_bad(bad_color, 1.0)

    # Plot data.
    image = ax.imshow(
        np.squeeze(data),
        vmax=vmax,
        vmin=vmin,
        extent=extent,
        origin=origin,
        cmap=colormap,
        interpolation=interpolation,
        aspect=aspect,
    )

    # Set title.
    ax.set_title(label)

    # Make colorbar.
    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        fraction=colorbar_fraction,
        pad=colorbar_pad,
    )
    colorbar.solids.set_edgecolor("face")

    # TODO(Niclas): Expose num_ticks.
    num_ticks = 5
    if plot_type == PlotType.ANGLE:
        ticks = np.linspace(vmin, vmax, num_ticks)
        colorbar.set_ticks(ticks)
        ticklabels = [r"$-\pi$", r"", r"$0$", r"", r"$\pi$"]
        colorbar.set_ticklabels(ticklabels)
    else:
        set_colorbar_ticks_and_labels(
            colorbar=colorbar, vmin=vmin, vmax=vmax, num_ticks=num_ticks
        )

    return image, colorbar


def make_plot_row(
    fig,
    axes,
    idx,
    data,
    extent,
    labels,
    colormaps,
    bad_color="dimgray",
    landscape=False,
    cartesian=False,
) -> None:
    # Extract row axes.
    axes_row = axes[:, idx] if landscape else axes[idx, :]
    num_axes = len(axes_row)

    if data.shape[0] == 1:
        # Sanity check.
        assert (num_axes == 1) or (num_axes == 2)

        # Determine which axis to turn on/off.
        ax_idx_on = int(num_axes == 2 and landscape)
        ax_idx_off = 1 - ax_idx_on

        # Plot the data.
        imshow_and_colorbar(
            fig=fig,
            ax=axes_row[ax_idx_on],
            data=np.squeeze(data),
            label=labels[0],
            extent=extent,
            plot_type=PlotType.DEFAULT,
            colormap=colormaps["default"],
            bad_color=bad_color,
        )

        # Turn off the unused axis.
        if num_axes == 2:
            axes_row[ax_idx_off].axis("off")
    else:
        # Sanity checks.
        assert data.shape[0] == 2
        assert len(axes_row) == 2

        if cartesian:
            # Plot (real, imag) or (x, y) components.
            for i, ax in enumerate(axes_row):
                imshow_and_colorbar(
                    fig=fig,
                    ax=ax,
                    data=data[i, :, :],
                    label=labels[i],
                    extent=extent,
                    plot_type=PlotType.DEFAULT,
                    colormap=colormaps["default"],
                    bad_color=bad_color,
                )
        else:
            # Plot magnitude.
            imshow_and_colorbar(
                fig=fig,
                ax=axes_row[0],
                data=np.hypot(data[0, :, :], data[1, :, :]),
                label=labels[0],
                extent=extent,
                plot_type=PlotType.MAGNITUDE,
                colormap=colormaps["magnitude"],
                bad_color=bad_color,
            )

            # Plot phase or angle.
            imshow_and_colorbar(
                fig=fig,
                ax=axes_row[1],
                data=np.arctan2(data[1, :, :], data[0, :, :]),
                label=labels[1],
                extent=extent,
                plot_type=PlotType.ANGLE,
                colormap=colormaps["angle"],
                bad_color=bad_color,
            )


def set_colorbar_ticks_and_labels(colorbar, vmin, vmax, num_ticks=5):
    ticks = np.linspace(vmin, vmax, num_ticks)
    colorbar.set_ticks(ticks)
    for label in colorbar.ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)


def set_axes_ticks_and_labels(axes, extent):
    # Make axes labels and ticks.
    _, x_lim, _, y_lim = extent
    xtick = np.floor(x_lim)
    ytick = np.floor(y_lim)
    xticks = [-xtick, -0.5 * xtick, 0, 0.5 * xtick, xtick]
    yticks = [-ytick, -0.5 * ytick, 0, 0.5 * ytick, ytick]
    xticklabels = [r"$-%d$" % xtick, r"", r"$0$", r"", r"$%d$" % xtick]
    yticklabels = [r"$-%d$" % ytick, r"", r"$0$", r"", r"$%d$" % ytick]

    # For handling single instance axis.
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
        axes = np.reshape(axes, [1, 1])

    # Get shape.
    rows, cols = axes.shape

    # X.
    for j in range(cols):
        set_xlabel = True
        for i in reversed(range(rows)):
            ax = axes[i, j]
            ax.set_xticks(xticks)
            has_data = ax.has_data()
            if set_xlabel and has_data:
                ax.set_xlabel(r"$x/\xi_0$")
                ax.set_xticklabels(xticklabels)
                set_xlabel = False
            else:
                ax.set_xticklabels([])
            set_xlabel = not has_data

    # Y.
    for i in range(rows):
        set_ylabel = True
        for j in range(cols):
            ax = axes[i, j]
            ax.set_yticks(yticks)
            has_data = ax.has_data()
            if set_ylabel and has_data:
                ax.set_ylabel(r"$y/\xi_0$")
                ax.set_yticklabels(yticklabels)
                set_ylabel = False
            else:
                ax.set_yticklabels([])
            set_ylabel = not has_data


def set_alphabetic_labels(axes):
    alphabet = "abcdefghijklmnopqrstuvwxyzåäö"
    idx = 0
    for ax in axes.flatten():
        if ax.has_data():
            title = r"$(%s)$" % alphabet[idx]
            ax.set_title(title, loc="left", fontsize="small")
            idx += 1


def set_global_parameters(fontsize, use_tex):
    # Check if TeX dependencies are installed.
    use_tex = use_tex and mpl.checkdep_usetex(True)

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": "medium",
            "text.usetex": use_tex,
            "svg.fonttype": "none",
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-0, 0),
        }
    )
