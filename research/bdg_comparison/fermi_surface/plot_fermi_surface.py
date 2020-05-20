import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy as sp

sys.path.append("../../../tools/fermi_surface/")
from compute_fermi_surface import wrapToRange

mpl.use("agg")

# Set Matplotlib parameters.
plt.rcParams.update(
    {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],}
)
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amssymb}", r"\usepackage{amsmath}"]
plt.rc("text", usetex=True)

# For the width of the plot.
# TODO(niclas): Move to Python a defines.py file?
WIDTH_IN_POINTS = 512.0
INCHES_PER_POINT = 1.0 / 72.27
ASPECT = 1.0 / np.sqrt(2.0)

LABEL_POSITION_X = 0.02
LABEL_POSITION_Y = 0.96

HEIGHT_COLORBAR = 0.2
HEIGHT_FERMISURFACE = 4
HEIGHT_DOS = 2
HEIGHT_TC = 2
WIDTH_ALL = HEIGHT_FERMISURFACE

ASPECT_DOS = 0.5 * HEIGHT_DOS / WIDTH_ALL
ASPECT_TC = 0.5 * HEIGHT_TC / WIDTH_ALL

FONTSIZE = 18
COLORMAP = mpl.cm.get_cmap("RdBu_r")
TICKS = [-1, -0.5, 0, 0.5, 1]
TICKS_LABELS = [r"$-1$", "", r"$0$", "", r"$1$"]
DPI = 100
HSPACE = 0.5

NAMES = [r"$\mathrm{\#1}$", r"$\mathrm{\#2}$"]
COLORS = [COLORMAP(0.2), COLORMAP(0.95)]
EF_COLOR = "black"
LINEWIDTH = 2


# Code taken from:
# https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        midMin = self.midpoint - self.vmin
        midMax = self.midpoint - self.vmax

        minVal = 0.5 * (1 - abs(midMin / midMax))
        normalized_min = max(0, minVal)

        maxVal = 0.5 * (1 + abs(midMax / midMin))
        normalized_max = min(1, maxVal)

        normalized_mid = 0.5
        x, y = (
            [self.vmin, self.midpoint, self.vmax],
            [normalized_min, normalized_mid, normalized_max],
        )
        return sp.ma.masked_array(np.interp(value, x, y))


def plotDispersion(
    dispersion,
    fermisurface,
    ax,
    axColorbar,
    fig,
    label,
    showYaxis,
    arrow_scale=0.2,
    arrow_head_length=0.05,
    arrow_head_width=0.05,
    aspectColorbar=1.0,
    labelPositionX=2.0 * LABEL_POSITION_X,
    labelPositionY=LABEL_POSITION_Y,
    colormap=COLORMAP,
    fontsize=FONTSIZE,
):
    # ====================== DISPERSION ====================== #
    kx = np.array(dispersion["kx"]) / np.pi
    ky = np.array(dispersion["ky"]) / np.pi
    ek = np.array(dispersion["values"])

    vmin = np.amin(ek)
    vmax = np.amax(ek)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.0)

    c = ax.imshow(
        ek,
        interpolation="none",
        cmap=colormap,
        norm=norm,
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
    )

    # Colorbar.
    cbar = fig.colorbar(c, cax=axColorbar, orientation="horizontal")
    cbar.set_label(r"$E / t$", fontsize=fontsize)
    axColorbar.set_aspect(aspectColorbar, adjustable="box")
    axColorbar.tick_params(labelsize=fontsize)

    # The contour plot of the Fermi surface, hence levels=0.
    ax.contour(kx, ky, ek, colors="k", levels=[0.0])

    # ====================== FERMI VELOCITY ====================== #
    kx = np.array(fermisurface["kx"]) / np.pi
    ky = np.array(fermisurface["ky"]) / np.pi
    vx = np.array(fermisurface["vx"]) / np.pi
    vy = np.array(fermisurface["vy"]) / np.pi

    # Plot the Fermi velocities.
    for x, y, q, p in zip(kx, ky, vx, vy):
        ax.arrow(
            x,
            y,
            q * arrow_scale,
            p * arrow_scale,
            length_includes_head=True,
            head_width=arrow_head_width,
            head_length=arrow_head_length,
            fc="k",
        )

    # Set axis title and labels.
    ax.set_xlabel(r"$k_x a / \pi$", fontsize=fontsize)
    if showYaxis:
        ax.set_ylabel(r"$k_y a / \pi$", fontsize=fontsize)

    # Set axis limits.
    xMin = -1
    xMax = 1
    yMin = xMin
    yMax = xMax
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Set ticks.
    ax.tick_params(labelsize=fontsize)
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICKS_LABELS)
    if showYaxis:
        ax.set_yticks(TICKS)
        ax.set_yticklabels(TICKS_LABELS)
    else:
        ax.set_yticks([])

    # Set aspect ratio.
    ax.set_aspect("equal", adjustable="box")

    # Add text.
    ax.text(
        labelPositionX,
        labelPositionY,
        label,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox={"boxstyle": "square", "facecolor": "white", "pad": 0.1},
    )


def plotDensityOfStates(dos, axDOS, label, color, linewidth=LINEWIDTH):
    energies = np.array(dos["energies"])
    dosValues = np.array(dos["values"])
    axDOS.plot(energies, dosValues, label=label, color=color, linewidth=linewidth)


def plotCriticalTemperature(
    criticalTemperature, chemicalPotential, ax, color, linewidth=LINEWIDTH
):
    mu = np.array(criticalTemperature["chemicalPotential"])
    dWaveTc = np.array(criticalTemperature["dWaveTc"])
    sWaveTc = np.array(criticalTemperature["sWaveTc"])

    ax.plot(
        mu, sWaveTc, label=r"$s_+$", linestyle="--", color=color, linewidth=linewidth,
    )
    ax.plot(
        mu, dWaveTc, label=r"$d$", color=color, linewidth=linewidth,
    )

    # Plot the Fermi level.
    ax.axvline(
        x=chemicalPotential,
        linestyle=":",
        label=r"$\mu$",
        color=color,
        linewidth=linewidth,
    )


def adjustPlot(
    ax,
    plotLabel,
    xLabel,
    yLabel,
    xMin,
    xMax,
    yMin,
    yMax,
    aspect,
    legendLoc,
    legendTitle=None,
    legendNumCols=1,
    labelPositionX=LABEL_POSITION_X,
    labelPositionY=LABEL_POSITION_Y,
    fontsize=FONTSIZE,
):
    # Set axis title and labels.
    ax.set_xlabel(xLabel, fontsize=fontsize)
    ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        fontsize=fontsize,
        loc=legendLoc,
        ncol=legendNumCols,
        columnspacing=0.2,
        borderaxespad=0.2,
        borderpad=0.2,
        handlelength=0.7,
        handletextpad=0.1,
        framealpha=1.0,
        title=legendTitle,
        title_fontsize=fontsize,
    )

    # Set axis limits.
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Set aspect ratio.
    xRange = xMax - xMin
    yRange = yMax - yMin
    _aspect = aspect * (xRange / yRange)
    ax.set_aspect(_aspect, adjustable="box")

    # Add text.
    ax.text(
        labelPositionX,
        labelPositionY,
        plotLabel,
        fontsize=fontsize,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="Plot the Fermi surface for a tight-binding model on a square lattice."
    )
    parser.add_argument("filenames", nargs="+", type=str, help="Name of file to load.")
    parser.add_argument(
        "-name",
        "--name",
        type=str,
        default="fermisurfaces",
        help="Name of saved file. (default: %(default)s)",
    )
    parser.add_argument(
        "-ext",
        "--extension",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="File extension of the plot. (default: %(default)s)",
    )
    args = parser.parse_args()

    numCols = len(args.filenames)
    numRows = 4

    # Setup figure and subplots.
    width = WIDTH_IN_POINTS * INCHES_PER_POINT
    height = width / ASPECT
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    # ax_cbar, ax_disp, axDOS, ax_tc
    axs = fig.subplots(
        nrows=numRows,
        ncols=numCols,
        gridspec_kw={
            "height_ratios": [
                HEIGHT_COLORBAR,
                HEIGHT_FERMISURFACE,
                HEIGHT_DOS,
                HEIGHT_TC,
            ],
            "hspace": HSPACE,
        },
    )

    gs = axs[0, 0].get_gridspec()
    for j in range(numCols):
        for i in range(numRows):
            if i < 2:
                continue
            ax = axs[i, j]
            ax.remove()

    axs_cbar = axs[0, :]
    axs_disp = axs[1, :]
    axDOS = fig.add_subplot(gs[2, :])
    ax_tc = fig.add_subplot(gs[3, :])

    for i, filename in enumerate(args.filenames):
        # Load file.
        suffix = ".json"
        if not filename.endswith(suffix):
            filename += ".json"
        file = open(filename, "r")
        data = json.load(file)

        ax_cbar = axs_cbar[i]
        ax_disp = axs_disp[i]

        # Extract stuff.
        criticalTemperature = data["criticalTemperature"]
        dispersion = data["dispersion"]
        dos = data["dos"]
        fermisurface = data["fermisurface"]
        parameters = data["parameters"]

        # ====================== FERMI SURFACE ====================== #
        first = i == 0
        if first:
            label = r"$\mathrm{(a)}$"
        else:
            label = r"$\mathrm{(b)}$"
        aspectColorbar = HEIGHT_COLORBAR / HEIGHT_FERMISURFACE
        plotDispersion(
            dispersion=dispersion,
            fermisurface=fermisurface,
            ax=ax_disp,
            axColorbar=ax_cbar,
            fig=fig,
            label=label,
            showYaxis=first,
            aspectColorbar=aspectColorbar,
            colormap=COLORMAP,
            fontsize=FONTSIZE,
        )

        # ====================== DOS ====================== #
        label = NAMES[i]
        color = COLORS[i]
        plotDensityOfStates(dos, axDOS, label=label, color=color)

        # ====================== TC ====================== #
        plotCriticalTemperature(
            criticalTemperature, parameters["mu"], ax_tc, color=color
        )

    axDOS.axvline(x=0.0, c=EF_COLOR, linestyle=":", label=r"$\epsilon_{\mathrm{F}}$")
    adjustPlot(
        axDOS,
        plotLabel=r"$\mathrm{(c)}$",
        xLabel=r"$E / t$",
        yLabel=r"$\mathcal{N}_\mathrm{N}(E) t a^2$",
        xMin=-2.0,
        xMax=2.0,
        yMin=0.0,
        yMax=1.5,
        aspect=ASPECT_DOS,
        legendLoc="upper right",
    )

    adjustPlot(
        ax_tc,
        plotLabel=r"$\mathrm{(d)}$",
        xLabel=r"$\mu/t$",
        yLabel=r"$k_\mathrm{B} T_\mathrm{c}/t$",
        xMin=-2.5,
        xMax=1.0,
        yMin=0.0,
        yMax=0.35,
        aspect=ASPECT_TC,
        legendLoc=(0.8, 0.15),
        legendNumCols=2,
        legendTitle=r"$\mathrm{\#1} \;\; \mathrm{\#2}$",
    )

    # ====================== MISC ====================== #
    # Align labels in the subplots.
    fig.align_labels()

    # Save.
    filename = args.name + "." + args.extension
    fig.savefig(filename, bbox_inches="tight", dpi=fig.dpi)
    print("Saved", filename)
    plt.close()


if __name__ == "__main__":
    main()
