from __future__ import division
import sys
from enum import Enum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data structure module
sys.path.insert(0, "../../../tools/")
from pythonLibraries.dataAnalysis import dataStructs as DS

# Set Matplotlib parameters.
plt.rcParams.update(
    {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],}
)
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amssymb}", r"\usepackage{amsmath}"]
plt.rc("text", usetex=True)

pathBase = "../data/qc/fs"
pathCommon = "_R60.000_r4.000_T0.1000"

WIDTH_IN_POINTS = 512.0
INCHES_PER_POINT = 1.0 / 72.27
FONTSIZE = 18
LABEL_POSITION_X = 0.02
LABEL_POSITION_Y = 0.96
LABELS = [r"$\mathrm{(a)}$", r"$\mathrm{(b)}$", r"$\mathrm{(c)}$"]

FS_ORDER = [1, 2, 0]
NAMES = [r"$\mathrm{Circle}$", r"$\mathrm{\#1}$", r"$\mathrm{\#2}$"]
COLORMAP = mpl.cm.get_cmap("RdBu_r")
COLORS = ["black", COLORMAP(0.2), COLORMAP(0.95)]
LINESTYLES = ["--", "-", "-"]
LINEWIDTHS = [1, 2, 2]

pMin1 = np.array([0.46666667, 0.38738739])
pMin2 = np.array([0.61081081, 0.53153153])

doAverage = True
asColumn = True
numPlots = 2
axis = (1, 2)
xLim = 3.0
xLimInset = 0.3
spectralWeightLim = 0.25
yLimMS = 6.25
yLimMH = 2.0
yLimInset = 4

xLabel = r"$\epsilon/(k_\mathrm{B} T_\mathrm{c})$"
yLabel = r"$N(\epsilon)/N_\mathrm{F}$"


def computeIndices(p1, p2, numX, numY, depth=0.6):
    vSurface = p2 - p1
    vNormal = np.array([-vSurface[1], vSurface[0]])
    vNormal *= depth

    indices = []
    for j in range(numY):
        y = j / numY - p1[1]
        for i in range(numX):
            x = i / numX - p1[0]

            p = np.array([x, y])

            okOnSurface = 0 <= np.dot(vSurface, p) < np.dot(vSurface, vSurface)
            okOnNormal = 0 <= np.dot(vNormal, p) < np.dot(vNormal, vNormal)

            if okOnSurface and okOnNormal:
                indices.append((i, j))
    return indices


def main():
    width = WIDTH_IN_POINTS * INCHES_PER_POINT
    height = width
    fig = plt.figure(figsize=(width, height), dpi=100)

    # Will be computed when needed.
    indices = None

    for i, ms in enumerate([True, False]):
        print(i)
        if asColumn:
            ax = fig.add_subplot(numPlots, 1, i + 1)
        else:
            ax = fig.add_subplot(1, numPlots, i + 1)

        ax.axvline(x=spectralWeightLim, color="0.75")
        ax.axvline(x=-spectralWeightLim, color="0.75")
        if ms:
            axInset = inset_axes(
                ax, width="35%", height="55%", loc="upper right", borderpad=1.5
            )
            axInset.axvline(x=spectralWeightLim, color="0.75")
            axInset.axvline(x=-spectralWeightLim, color="0.75")

        for fs in FS_ORDER:
            label = NAMES[fs]
            color = COLORS[fs]
            linestyle = LINESTYLES[fs]
            linewidth = LINEWIDTHS[fs]

            tmp = pathBase + str(fs + 1) + pathCommon
            if ms:
                path = tmp + "_ms/"
            else:
                path = tmp + "/"

            data = DS.runData(path, fields_in=["ldos"])
            ldos = data.ldos.abs

            minEnergy = data.ldos.params["minEnergy"]
            maxEnergy = data.ldos.params["maxEnergy"]
            numEnergies = int(data.ldos.params["numEnergies"])
            energies = np.linspace(minEnergy, maxEnergy, numEnergies)

            if indices is None:
                numX, numY = ldos.shape[1:]
                indices = computeIndices(pMin1, pMin2, numX, numY)

            dos = np.zeros_like(energies)
            for p in indices:
                xIdx, yIdx = p
                dos += ldos[:, yIdx, xIdx]
            dos /= len(indices)

            cond = energies < spectralWeightLim
            _energies = energies[cond]
            _dos = dos[cond]
            spectralWeight = (
                (2.0 * np.sum(_dos) - _dos[0] - dos[-1])
                * 2.0
                * _energies[-1]
                / (2.0 * len(_energies) + 1.0)
            )
            print(spectralWeight)

            energies = np.concatenate([-np.flip(energies), energies[1:]])
            dos = np.concatenate([np.flip(dos, axis=0), dos[1:]], axis=0)

            ax.plot(
                energies,
                dos,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            if ms:
                axInset.plot(
                    energies,
                    dos,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

        ax.set_xlim(-xLim, xLim)

        if ms:
            legend_loc = "center left"
            ax.set_ylim(0.0, yLimMS)
        else:
            legend_loc = "lower left"
            ax.set_ylim(0.0, yLimMH)
        ax.tick_params(labelsize=FONTSIZE)
        ax.legend(loc=legend_loc, framealpha=1.0, fontsize=FONTSIZE)

        ax.text(
            LABEL_POSITION_X,
            LABEL_POSITION_Y,
            LABELS[i],
            fontsize=FONTSIZE,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        if ms:
            axInset.set_xlim(-xLimInset, xLimInset)
            axInset.set_ylim(yLimInset, yLimMS)
            axInset.tick_params(labelsize=FONTSIZE)
            # axInset.set_xlabel(xLabel, fontsize=FONTSIZE)
            # axInset.set_ylabel(yLabel, fontsize=FONTSIZE)
            xticks = np.linspace(-xLimInset, xLimInset, 7)
            axInset.set_xticks(xticks)
            axInset.set_xticklabels([r"$-0.3$", "", "", r"$0$", "", "", r"$0.3$"])
            axInset.set_yticks([4, 5, 6])
            axInset.set_yticklabels([r"$4$", r"$5$", r"$6$"])

        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        if i < numPlots - 1 and asColumn:
            ax.set_xticklabels(["", "", "", "", "", "", ""])
        else:
            ax.set_xlabel(xLabel, fontsize=FONTSIZE)

        if ms:
            ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
            yticklabels = [r"$0$", "", r"$2$", "", r"$4$", "", r"$6$"]
            yticklabels_empty = ["", "", "", "", "", "", ""]
        else:
            ax.set_yticks([0, 0.5, 1.0, 1.5, 2])
            yticklabels = [r"$0$", "", r"$1$", "", r"$2$"]
            yticklabels_empty = ["", "", "", "", ""]

        if i > 0 and not asColumn:
            ax.set_yticklabels(yticklabels_empty)
        else:
            ax.set_ylabel(yLabel, fontsize=FONTSIZE)
            ax.set_yticklabels(yticklabels)

    plt.tight_layout()
    # plt.show()
    filename = "figures/ldos.pdf"
    plt.savefig(filename, bbox_inches="tight", dpi=fig.dpi)
    plt.close()


if __name__ == "__main__":
    main()
