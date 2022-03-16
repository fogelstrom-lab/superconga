#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from typing import Optional

from . import default
from .arguments import make_parser

sys.path.insert(0, "../")
from common import plotting, read


def compute_kernel(blur):
    # Approx smallest value of any kernel element.
    epsilon = 1e-3

    # Compute required padding (and kernel size).
    normalization = blur * np.sqrt(2.0 * np.pi)
    pad = int(blur * np.sqrt(-2.0 * np.log(epsilon * normalization)))
    kernel_size = 2 * pad + 1

    arg = np.linspace(-pad, pad, kernel_size) / blur
    kernel = np.exp(-(arg ** 2) / 2)
    kernel /= np.sum(kernel)
    return kernel


def convolve(data, kernel):
    # Pad the data.
    pad = (len(kernel) - 1) // 2
    left = np.ones(pad) * data[0]
    right = np.ones(pad) * data[-1]
    data_padded = np.concatenate([left, data, right])

    # Convolve with kernel.
    return np.convolve(data_padded, kernel, mode="valid")


def plot(ax, x, y, label, blur, color, alpha):
    if blur > 0:
        kernel = compute_kernel(blur=blur)
        y_smoothed = convolve(data=y, kernel=kernel)
        ax.semilogy(x, y_smoothed, label=label, color=color)

        top = np.maximum(y, y_smoothed)
        bottom = np.minimum(y, y_smoothed)
        ax.fill_between(x, bottom, top, color=mpl.colors.to_rgba(color, alpha=alpha))
    else:
        ax.semilogy(x, y, label=label, color=color)


def run(
    data_dir: str,
    save_path: Optional[str] = None,
    view: Optional[str] = default.settings["view"],
    blur: Optional[int] = default.settings["blur"],
    subplot_width: Optional[int] = default.settings["subplot_width"],
    subplot_height: Optional[int] = default.settings["subplot_height"],
    fontsize: Optional[int] = default.settings["fontsize"],
    alphabet: Optional[int] = default.settings["alphabet"],
    show_path: Optional[bool] = default.settings["show_path"],
    use_tex: Optional[bool] = default.settings["use_tex"],
    alpha: Optional[float] = default.settings["alpha"],
    colors: Optional[dict] = default.settings["colors"],
) -> None:
    # Set global parameters, i.e. for all plots. Must be done before the figure is made.
    plotting.set_global_parameters(fontsize=fontsize, use_tex=use_tex)

    # Read configuration file and set posible filenames of the convergence file.
    if view == "simulation":
        config = read.get_simulation_config(data_dir=data_dir)
        # Support old filename.
        filenames = ["simulation_convergence.csv", "convergence.csv"]
    elif view == "postprocess":
        config = read.get_postprocess_config(data_dir=data_dir)
        filenames = ["postprocess_convergence.csv"]
    else:
        raise ValueError(f"-- ERROR! View, '{view}', not understood.")

    # Read convergence criterion from config.
    # Support old config files.
    groups = ["numerics", "accuracy"]
    for group in groups:
        if group in config:
            convergence_criterion = config[group]["convergence_criterion"]
            break

    # Check if file exists.
    for filename in filenames:
        csv_path = os.path.join(data_dir, filename)
        if os.path.exists(csv_path):
            break
    df = pd.read_csv(csv_path)

    rows, cols = (1, 2) if view == "simulation" else (1, 1)
    name = data_dir if show_path else None
    fig, axes = plotting.make_figure(
        rows=rows,
        cols=cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
        sharex=False,
        sharey=False,
        squeeze=False,
        name=name,
    )
    axes = axes.flatten()

    # ===================== RESIDUALS ===================== #

    # Get axis.
    ax_residual = axes[0]

    # Draw convergence criterion.
    ax_residual.axhline(
        y=convergence_criterion,
        color="black",
        linestyle=":",
        label=r"$\mathrm{Convergence}$",
    )

    indices = list(range(len(df["iteration"])))
    iterations = df["iteration"].to_numpy()

    # Loop over key in dataframe and plot the data.
    ymax = 0
    for key in df.keys():
        if (key in default.labels) and (key in colors):
            label = default.labels[key]
            color = colors[key]
        else:
            continue

        # Read data.
        data = df[key].to_numpy()

        # Update limit.
        ymax = max(ymax, np.amax(data))

        # Plot the data.
        plot(
            ax=ax_residual,
            x=indices,
            y=data,
            label=label,
            color=color,
            blur=blur,
            alpha=alpha,
        )
    ymin = 0.1 * convergence_criterion
    ax_residual.set_ylim([ymin, ymax])
    ax_residual.set_xlabel(r"$\mathrm{Iteration}$")
    ax_residual.set_ylabel(r"$\epsilon_\mathrm{G}$")
    ax_residual.legend()
    ax_residual.set_title(r"$\mathrm{Residuals}$")

    # ===================== FREE ENERGY ===================== #

    # Get axis.
    if "free_energy_per_unit_area" in df:
        ax_energy = axes[1]

        data = df["free_energy_per_unit_area"].to_numpy()
        vmin = np.amin(data)

        # Same color as residual.
        color = colors["free_energy_residual"]

        plot(
            ax=ax_energy,
            x=indices,
            y=(data - vmin),
            label=r"$\Omega$",
            color=color,
            blur=blur,
            alpha=alpha,
        )

        ax_energy.set_xlabel(r"$\mathrm{Iteration}$")
        ax_energy.set_ylabel(r"$(\Omega - \Omega_\mathrm{min})/\Omega_\mathcal{A}$")
        ax_energy.set_title(r"$\mathrm{Free\;Energy}$")

    # Set alphabetic label, e.g. (a) at the top left.
    if alphabet:
        plotting.set_alphabetic_labels(axes=axes)

    for ax in axes:
        ax.set_xlim([indices[0], indices[-1]])

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
    # Set colors.
    colors = {
        "boundary_coherence_residual": args.boundary_coherence_color,
        "order_parameter_residual": args.order_parameter_color,
        "current_density_residual": args.current_density_color,
        "vector_potential_residual": args.vector_potential_color,
        "free_energy_residual": args.free_energy_color,
        "ldos_residual": args.ldos_color,
    }

    # Run the plotter.
    run(
        data_dir=args.load_path,
        save_path=args.save_path,
        view=args.view,
        blur=args.blur,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
        fontsize=args.fontsize,
        alphabet=(not args.no_alphabet),
        show_path=(not args.no_path),
        use_tex=(not args.no_tex),
        alpha=args.alpha,
        colors=colors,
    )
