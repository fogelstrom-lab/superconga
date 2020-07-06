from argparse import ArgumentParser
import json
import numpy as np
import os
import subprocess
import sys


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default="parameter_files/default_mu_sweeps.json",
        help="Path to JSON file with parameters specifying the simulation settings.",
    )

    # C++ binary.
    parser.add_argument(
        "--cpp",
        type=str,
        default="../../build/release/research/mu_sweeps_main",
        help="Path to the C++ binary.",
    )

    # Chemical potential.
    parser.add_argument(
        "--min-mu", type=float, help="Minimum chemical potential.",
    )
    parser.add_argument(
        "--max-mu", type=float, help="Maximum chemical potential.",
    )
    parser.add_argument(
        "--num-mu", type=int, help="Number of chemical potentials.",
    )

    # Temperature
    parser.add_argument(
        "--min-temp", type=float, help="Minimum temperature.",
    )
    parser.add_argument(
        "--max-temp", type=float, help="Maximum temperature.",
    )
    parser.add_argument(
        "--num-temp", type=int, help="Number of temperatures.",
    )

    # Fermi surface label.
    parser.add_argument(
        "--fs", type=int, choices=[1, 2], nargs="+", help="Fermi surface.",
    )

    # Initial guess label.
    parser.add_argument(
        "--init",
        type=int,
        choices=[1, 2, 3, 4],
        nargs="+",
        help="Initial guess label.",
    )

    # Parse.
    args = parser.parse_args()

    # Check that the relevant files exist.
    if not os.path.isfile(args.file):
        print(f"--Cannot find the JSON config file '{args.file}'.")
        sys.exit()
    if not os.path.isfile(args.cpp):
        print(
            f"--Cannot find the C++ executable '{args.cpp}'. See the README for how to compile it."
        )
        sys.exit()

    # Read the JSON file.
    with open(args.file, "r") as f:
        data = f.read()

    # Parse the JSON file.
    simparams = json.loads(data)

    # Create arrays of chemical potentials and temperatures.
    mu_arr = np.linspace(args.min_mu, args.max_mu, args.num_mu)
    temp_arr = np.linspace(args.min_temp, args.max_temp, args.num_temp)

    config_file = ".tmp_config.json"

    for fs in args.fs:
        simparams["fermi_surface_label"] = fs
        for init in args.init:
            simparams["start_guess_label"] = init
            for mu in mu_arr:
                simparams["chemical_potential"] = mu
                for temp in temp_arr:
                    simparams["temperature"] = temp

                    # Write config file.
                    with open(config_file, "w") as f:
                        json.dump(simparams, f, ensure_ascii=False, indent=4)

                    # Run the C++ program.
                    subprocess.call(["./" + args.cpp, config_file])

                    # Remove config file.
                    if os.path.isfile(config_file):
                        os.remove(config_file)


if __name__ == "__main__":
    main()
