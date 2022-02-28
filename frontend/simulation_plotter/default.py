#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

# Default settings.
settings = {
    "view": ["D", "J", "A", "B"],
    "fontsize": 14,
    "subplot_width": 3.5,
    "subplot_height": 3.1,
    "cartesian": False,
    "landscape": True,
    "alphabet": True,
    "show_path": True,
    "use_tex": True,
    "colormaps": {
        "default": "RdBu",
        "magnitude": "YlGnBu",
        "angle": "twilight_shifted_r",
    },
    "exterior_color": "dimgray",
}

# Labels for every object we can plot.
labels = {
    "order_parameter_dx2-y2": {
        "cartesian": [
            r"$\mathrm{Re} [ \Delta_{d_{x^2-y^2}} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\mathrm{Im} [ \Delta_{d_{x^2-y^2}} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
        ],
        "polar": [
            r"$|\Delta_{d_{x^2-y^2}}| / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\chi_{d_{x^2-y^2}}$",
        ],
    },
    "order_parameter_dxy": {
        "cartesian": [
            r"$\mathrm{Re} [ \Delta_{d_{xy}} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\mathrm{Im} [ \Delta_{d_{xy}} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
        ],
        "polar": [
            r"$|\Delta_{d_{xy}}| / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\chi_{d_{xy}}$",
        ],
    },
    "order_parameter_g": {
        "cartesian": [
            r"$\mathrm{Re} [ \Delta_{g_{xy(x^2-y^2})} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\mathrm{Im} [ \Delta_{g_{xy(x^2-y^2})} ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
        ],
        "polar": [
            r"$|\Delta_{g_{xy(x^2-y^2})}| / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\chi_{g_{xy(x^2-y^2})}$",
        ],
    },
    "order_parameter_s": {
        "cartesian": [
            r"$\mathrm{Re} [ \Delta_s ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\mathrm{Im} [ \Delta_s ] / (2\pi k_\mathrm{B} T_\mathrm{c})$",
        ],
        "polar": [
            r"$|\Delta_s| / (2\pi k_\mathrm{B} T_\mathrm{c})$",
            r"$\chi_s$",
        ],
    },
    "current_density": {
        "cartesian": [r"$j_x / j_0$", r"$j_y / j_0$"],
        "polar": [r"$|\mathbf{j}| / j_0$", r"$\angle \mathbf{j}$"],
    },
    "vector_potential": {
        "cartesian": [r"$A_x / A_0$", r"$A_y / A_0$"],
        "polar": [r"$|\mathbf{A}| / A_0$", r"$\angle \mathbf{A}$"],
    },
    "magnetic_flux_density": {
        "cartesian": [r"$B_z / B_0$"],
        "polar": [r"$B_z / B_0$"],
    },
}
