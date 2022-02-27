#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

# Default settings.
settings = {
    "view": "simulation",
    "sigma": 0,
    "subplot_width": 7.0,
    "subplot_height": 6.5,
    "fontsize": 14,
    "alphabet": True,
    "show_path": True,
    "use_tex": True,
    "alpha": 0.333,
    "colors": {
        "boundary_coherence_residual": "tab:red",
        "order_parameter_residual": "tab:orange",
        "current_density_residual": "tab:blue",
        "vector_potential_residual": "tab:pink",
        "free_energy_residual": "black",
        "ldos_residual": "black",
    },
}

# Labels for the legend.
labels = {
    "boundary_coherence_residual": r"$\gamma_{\partial D}$",
    "order_parameter_residual": r"$\Delta$",
    "current_density_residual": r"$\mathbf{j}$",
    "vector_potential_residual": r"$\mathbf{A}$",
    "free_energy_residual": r"$\Omega$",
    "ldos_residual": r"$N(\mathbf{R}, \epsilon)$",
}
