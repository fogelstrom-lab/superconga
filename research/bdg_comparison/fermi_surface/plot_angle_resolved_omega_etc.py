import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy as sp

sys.path.append("../../../tools/fermi_surface/")
from compute_fermi_surface import computeFermiSurface, wrapToRange
from plot_fermi_surface import MidpointNormalize

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

LABEL_POSITION_X = 0.04
LABEL_POSITION_Y_LOW = 0.15
LABEL_POSITION_Y_HIGH = 1.0 - LABEL_POSITION_Y_LOW

HEIGHT_COLORBAR = 0.05
HEIGHT_FERMISURFACE = 1
WIDTH_ALL = HEIGHT_FERMISURFACE

FONTSIZE = 18
NUM_TICKS = 5
DPI = 100
HSPACE = 0.5
WSPACE = 0.0

NAMES = [r"$\mathrm{Circular}$", r"$\mathrm{\#1}$", r"$\mathrm{\#2}$"]
COLORS = ["k", "r", "m"]
LINESTYLES = ["-.", "-", "--"]
FS_ORDER = [1, 2, 0]

FS1 = {"t0": 1.0, "t1": -0.25, "t2": 0.0, "mu": 0.0}
FS2 = {"t0": 1.0, "t1": -0.495, "t2": 0.156, "mu": -1.267}

NUM_ANGLES = 1000
CONVERGENCE_CRITERION = 1e-6
NUM_ITERATIONS = 100

NORMAL_ANGLE = np.pi * 0.75
SURFACE_ANGLE = NORMAL_ANGLE - 0.5 * np.pi
FACTOR = NUM_ANGLES
MULT_W = True

D_X2_Y2 = True


def wrapK(kx, ky, origin):
    kx = wrapToRange(kx, -np.pi + origin, np.pi + origin)
    ky = wrapToRange(ky, -np.pi + origin, np.pi + origin)
    return kx, ky


def getBasis(kx, ky, w):
    basis = np.cos(kx) - np.cos(ky)
    basis /= np.sqrt(np.sum(basis ** 2 * w))
    return basis


def computeStuff(fermisurface, origin):
    kx = np.array(fermisurface["kx"])
    ky = np.array(fermisurface["ky"])
    vx = np.array(fermisurface["vx"])
    vy = np.array(fermisurface["vy"])
    w = np.array(fermisurface["weights"])

    # Normalize the Fermi velocity.
    avgV = np.sum(np.hypot(vx, vy) * w)
    vx /= avgV
    vy /= avgV

    # Basis function.
    if D_X2_Y2:
        basis = np.cos(kx) - np.cos(ky)
    else:
        basis = np.sin(kx) * np.sin(ky)
    basis /= np.sqrt(np.sum(basis ** 2 * w))

    # Order stuff with increasing phi.
    kx, ky = wrapK(kx, ky, origin)
    angles = np.arctan2(ky - origin, kx - origin)
    idx = np.argsort(angles)
    angles = angles[idx]
    kx = kx[idx]
    ky = ky[idx]
    vx = vx[idx]
    vy = vy[idx]
    w = w[idx]
    basis = basis[idx]

    # Fermi speed.
    v = np.hypot(vx, vy)

    if D_X2_Y2:
        normalAngle = np.pi * 0.25
    else:
        normalAngle = 0.0
    surfaceAngle = normalAngle - 0.5 * np.pi

    stuff = {}
    stuff["angles"] = angles
    stuff["basis"] = basis
    stuff["weights"] = w
    stuff["curvature"] = w * v
    stuff["fermiSpeed"] = v
    stuff["deltaExp"] = -(basis ** 2) * np.log(np.abs(basis) + 1e-16)
    stuff["delta"] = np.pi * np.exp(np.sum(stuff["deltaExp"] * w)) * np.exp(-0.57721566)
    stuff["vDotN"] = np.abs(vx * np.cos(normalAngle) + vy * np.sin(normalAngle))
    stuff["vDotP"] = np.abs(vx * np.cos(surfaceAngle) + vy * np.sin(surfaceAngle))
    return stuff


def plotWeight(stuff, ax, label, linestyle, color):
    w = stuff["weights"]
    ax.plot(
        stuff["angles"] / np.pi,
        w * FACTOR,
        label=label,
        linestyle=linestyle,
        color=color,
    )
    print("Weights=", np.sum(w))


def plotBCS(stuff, ax, label, linestyle, color):
    w = stuff["weights"]
    basis = stuff["basis"]
    d = stuff["delta"] * basis
    result = -0.5 * d ** 2
    if MULT_W:
        result *= w * FACTOR
    ax.plot(
        stuff["angles"] / np.pi, result, label=label, linestyle=linestyle, color=color,
    )
    print("Basis=", np.sum(basis ** 2 * w))
    print("Delta=", stuff["delta"])


def plotMGS(stuff, ax, label, linestyle, color):
    w = stuff["weights"]
    basis = stuff["basis"]
    vDotN = stuff["vDotN"]
    delta = stuff["delta"]
    result = vDotN * np.abs(basis) * 0.5 * np.pi ** 2 * delta
    if MULT_W:
        result *= w * FACTOR
    ax.plot(
        stuff["angles"] / np.pi, result, label=label, linestyle=linestyle, color=color
    )
    print("MGS=", np.sum(result * w))


def plotDoppler(stuff, ax, label, linestyle, color):
    w = stuff["weights"]
    basis = stuff["basis"]
    vDotN = stuff["vDotN"]
    vDotP = stuff["vDotP"]
    result = -np.pi ** 2 * vDotP * vDotN
    if MULT_W:
        result *= w * FACTOR
    ax.plot(
        stuff["angles"] / np.pi, result, label=label, linestyle=linestyle, color=color
    )
    print("Doppler=", np.sum(result * w))


def setStuff(
    ax,
    yLabel,
    plotLabel,
    xLabel=r"$\phi_\mathrm{F}/\pi$",
    doX=False,
    doLegend=False,
    highLabel=False,
):
    if D_X2_Y2:
        xMin = -0.25
        xMax = 0.75
    else:
        xMin = -0.5
        xMax = 0.5

    # Set axis title and labels.
    if doX:
        ax.set_xlabel(xLabel, fontsize=FONTSIZE)
    ax.set_ylabel(yLabel, fontsize=FONTSIZE)

    # Set axis limits.
    ax.set_xlim(xMin, xMax)

    # Set ticks.
    if doX:
        ticks = np.linspace(xMin, xMax, NUM_TICKS)
        ax.set_xticks(ticks)
    else:
        ax.set_xticks([])
    ax.tick_params(labelsize=FONTSIZE)

    if doLegend:
        ax.legend(
            mode="expand",
            ncol=3,
            fontsize=FONTSIZE,
            loc=(0.005, 1.05),
            framealpha=1.0,
            borderaxespad=0.1,
            borderpad=0.1,
        )

    if highLabel:
        labelPositionY = LABEL_POSITION_Y_HIGH
    else:
        labelPositionY = LABEL_POSITION_Y_LOW
    ax.text(
        LABEL_POSITION_X,
        labelPositionY,
        plotLabel,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="Plot the Fermi surface for a tight-binding model on a square lattice."
    )
    parser.add_argument(
        "-name",
        "--name",
        type=str,
        default="fermisurface_weight",
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
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save. (default: %(default)s)",
    )
    args = parser.parse_args()

    # Setup figure and subplots.
    width = WIDTH_IN_POINTS * INCHES_PER_POINT
    height = width
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    for i in FS_ORDER:
        if i == 0:
            angles = np.linspace(
                0.0, 2.0 * np.pi * (1.0 - 1.0 / NUM_ANGLES), NUM_ANGLES
            )
            fermisurface = {
                "kx": np.cos(angles),
                "ky": np.sin(angles),
                "vx": np.cos(angles),
                "vy": np.sin(angles),
                "weights": np.ones_like(angles) / NUM_ANGLES,
            }
            origin = 0.0
        else:
            if i == 1:
                parameters = FS1
            elif i == 2:
                parameters = FS2
            else:
                raise ValueError(f"FS index {i} not understood.")
            fermisurface = computeFermiSurface(
                parameters, NUM_ANGLES, NUM_ITERATIONS, CONVERGENCE_CRITERION
            )
            origin = np.pi

        stuff = computeStuff(fermisurface, origin)
        label = NAMES[i]
        linestyle = LINESTYLES[i]
        color = COLORS[i]

        plotWeight(stuff=stuff, ax=ax1, label=label, linestyle=linestyle, color=color)
        plotBCS(stuff=stuff, ax=ax2, label=label, linestyle=linestyle, color=color)
        plotMGS(stuff=stuff, ax=ax3, label=label, linestyle=linestyle, color=color)
        plotDoppler(stuff=stuff, ax=ax4, label=label, linestyle=linestyle, color=color)

    if MULT_W:
        setStuff(
            ax1,
            yLabel=r"$w(\phi_\mathrm{F})$",
            plotLabel=r"$\mathrm{(a)}$",
            doLegend=True,
        )
        setStuff(
            ax2,
            yLabel=r"$\Omega^\mathrm{BCS}(\phi_\mathrm{F}) w /\Omega_\mathcal{A} $",
            plotLabel=r"$\mathrm{(b)}$",
        )
        setStuff(
            ax3,
            yLabel=r"$\Omega^\mathrm{MGS}(\phi_\mathrm{F}) w /\Omega_\mathcal{L} $",
            plotLabel=r"$\mathrm{(c)}$",
            highLabel=True,
        )
        setStuff(
            ax4,
            yLabel=r"$\Omega^\mathrm{DS}(\phi_\mathrm{F}) w /\Omega_\mathcal{L}$",
            plotLabel=r"$\mathrm{(d)}$",
            doX=True,
        )
    else:
        setStuff(
            ax1,
            yLabel=r"$w(\phi_\mathrm{F})$",
            plotLabel=r"$\mathrm{(a)}$",
            doLegend=True,
        )
        setStuff(
            ax2,
            yLabel=r"$\Omega^\mathrm{BCS}(\phi_\mathrm{F}) /\Omega_\mathcal{A} $",
            plotLabel=r"$\mathrm{(b)}$",
        )
        setStuff(
            ax3,
            yLabel=r"$\Omega^\mathrm{MGS}(\phi_\mathrm{F}) /\Omega_\mathcal{L} $",
            plotLabel=r"$\mathrm{(c)}$",
            highLabel=True,
        )
        setStuff(
            ax4,
            yLabel=r"$\Omega^\mathrm{DS}(\phi_\mathrm{F}) /\Omega_\mathcal{L}$",
            plotLabel=r"$\mathrm{(d)}$",
            doX=True,
        )

    # ====================== MISC ====================== #

    # Align labels in the subplots.
    fig.align_labels()

    # Save.
    if args.save:
        filename = args.name + "." + args.extension
        fig.savefig(filename, bbox_inches="tight", dpi=fig.dpi)
        print("Saved", filename)
    plt.close()


if __name__ == "__main__":
    main()
