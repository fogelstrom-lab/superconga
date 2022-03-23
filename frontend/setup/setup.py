#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
The main part of the setup tool.
"""

import argparse
import os
import shutil
import sys
import subprocess

from .arguments import make_parser


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Run setup tool."""

    # Remove negatations etc.
    build_type = args.build_type
    cuda_arch = args.cuda_arch
    use_ninja = not args.no_ninja
    use_visualize = not args.no_visualize
    use_cuda_opengl_interop = args.use_cuda_opengl_interop
    use_hdf5 = not args.no_hdf5
    build_tests = not args.no_tests

    # Create paths.
    src_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
    build_dir = os.path.join(src_dir, os.path.join("build", build_type.lower()))

    # Create directory.
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Base commands.
    setup_args = [
        "cmake",
        "-S=" + src_dir,
        "-B=" + build_dir,
        "-DCMAKE_BUILD_TYPE=" + build_type,
    ]

    # Set CUDA architecture if specified.
    if cuda_arch is not None:
        setup_args.append("-DCUDA_ARCH=" + str(cuda_arch))

    # Set generator.
    if use_ninja:
        setup_args.append("-G=Ninja")

    # Set visualize.
    if use_visualize:
        setup_args.append("-DCONGA_VISUALIZE=ON")

        # Set CUDA-OpenGL interopability.
        if use_cuda_opengl_interop:
            setup_args.append("-DCONGA_USE_CUDA_OPENGL_INTEROP=ON")
        else:
            setup_args.append("-DCONGA_USE_CUDA_OPENGL_INTEROP=OFF")
    else:
        setup_args.append("-DCONGA_VISUALIZE=OFF")

    # Set HDF5.
    if use_hdf5:
        setup_args.append("-DCONGA_USE_HDF5=ON")
    else:
        setup_args.append("-DCONGA_USE_HDF5=OFF")

    # Set build tests.
    if build_tests:
        setup_args.append("-DCONGA_BUILD_TESTS=ON")
    else:
        setup_args.append("-DCONGA_BUILD_TESTS=OFF")

    # Do the setup.
    subprocess.run(setup_args)
