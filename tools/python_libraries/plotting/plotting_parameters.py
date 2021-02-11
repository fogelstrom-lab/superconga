#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Default plotting parameters and LaTeX plotting setup."""

# Built-in modules.
import numpy as np
import matplotlib.pyplot as plt

# Third-party libraries.

# Custom libraries.

# -----------------------------------------------------------------------------
# Plotting parameters.
# -----------------------------------------------------------------------------

# --------------------------------------------------
# FIGURE SIZES AND SCALING
# --------------------------------------------------
# Figure sizes in pt and inches, see e.g.:
# https://www.overleaf.com/learn/latex/Lengths_in_LaTeX
# Aesthetic ratio for height/width.
golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
# Convert pt to inches.
inches_per_pt = 1.0 / 72.27
# Widths and heights in pt.
fig_width_pt = 512.0
fig_height_pt = fig_width_pt * golden_ratio
# Widths and heights in inches.
fig_width_inch = fig_width_pt * inches_per_pt
fig_height_inch = fig_height_pt * inches_per_pt

# Resolution parameters.
dpi = 100

# --------------------------------------------------
# FONT SIZES
# --------------------------------------------------
# Font sizes in pt. The actual size that text objects appear in a paper depends
# on the scaling of the figure object, which means that the font size written
# in Python will usually be smaller once included in e.g. a pdf manuscript. The
# settings here account for this issue, and have been adapted to the figure
# size/golden ratio standard above, such that the figure text will end up the
# same as the main text size in e.g. APS/revtex manuscripts (12pt). This has
# been tested for both 1- and 2-column wide figures (for the latter, scale the
# above width with 2).
# Tick label font size (in pt).
tick_size = 18
# Overall font size (in pt).
font_size = 18
# Colorbar label font size (in pt).
cbar_size = 18

# --------------------------------------------------
# COLOR DEFINITIONS
# --------------------------------------------------
# Gray color, used e.g. to mask what is "outside" the grain.
gray_color = "#c0c0c0"
mask_color = gray_color

# --------------------------------------------------
# IMSHOW DEFAULT SETTINGS
# --------------------------------------------------
# 2D plot (e.g. imshow) interpolation. Useful values: none, bilinear, bicubic.
interpolation = "none"
# 2D plot aspect ratio. 1.0 means square. Default: 'auto'.
aspect = 1.0

# -----------------------------------------------------------------------------
# LaTeX support and rc parameters.
# -----------------------------------------------------------------------------
# Adjust settings for LaTeX plotting.
def init_latex_plotting():
    # Enable LaTeX rendering in plots.
    plt.rc(
        "font", **{"family": "sans-serif", "sans-serif": ["Computer Modern Sans serif"]}
    )
    # Include latex packages.
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb} \usepackage{amsmath}"
    # Enable LaTeX.
    plt.rc("text", usetex=True)
    # Avoid character/text objects to be converted to paths. This enables image
    # post-processing and editing the figure text e.g. in inkscape.
    plt.rc("svg", fonttype="none")
