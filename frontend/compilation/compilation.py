#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""
The main part of the compilation tool.
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
    """Run compilation tool."""

    # Create path.
    build_type = args.build_type
    src_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
    build_dir = os.path.join(src_dir, os.path.join("build", build_type.lower()))

    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isdir(build_dir) and os.path.exists(cache_path):
        # Arguments for compilation.
        compilation_args = ["cmake", "--build", build_dir]

        # Do the compilation.
        subprocess.run(compilation_args)

        if args.test:
            # Arguments for testing.
            test_args = ["ctest", "--test-dir", build_dir]

            # Do the testing.
            subprocess.run(test_args)
    else:
        print(f"-- ERROR! Could not find {cache_path}. Have you run 'setup'?")
