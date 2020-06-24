import argparse
import json
import os
import subprocess
import sys
from typing import Union

VALID_SYMMETRIES = ["s", "d"]


def make_default_parser():
    parser = argparse.ArgumentParser(
        prog="SuprConGPU",
        description="SuprConGPU Main file for setting up self-consistent simulations in a confined geometry (a mesoscopic grain). Different superconducting order-paramers and geometries can be specified, as well as physical parameters such as temperature and external fields. To set these, create a .json parameter-file as in examples and give the path to this file with the -f option. All parameters can be overridden using the command-line options specified below.",
    )

    # Config file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the simulation settings.",
    )

    # C++ binary.
    parser.add_argument(
        "-b",
        "--c-binary",
        type=str,
        default="build/release/bin/main",
        help="Path to the C++ binary.",
    )

    # =============== PHYSICS =============== #

    # Temperature.
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Temperature in units of T_c.",
    )

    # Symmetry.
    group_symmetry = parser.add_mutually_exclusive_group()
    group_symmetry.add_argument(
        "--dwave-symmetry",
        action="store_true",
        help="Order parameter d-wave symmetry.",
    )
    group_symmetry.add_argument(
        "--swave-symmetry", action="store_true", help="Order parameter s-wave symmetry."
    )

    # External flux quanta.
    # TODO(niclas): Rename to "-F"?
    parser.add_argument(
        "-B",
        "--external-flux-quanta",
        type=float,
        default=None,
        help="External magnetic field, perpendicular to sample, in units of flux quanta. For non-solenoid field, it is distributed uniformly over the sample.",
    )

    # Grain side length.
    parser.add_argument(
        "-s",
        "--grain-side-length",
        type=float,
        default=None,
        help="Side length of grain in units of coherence lengths (hbar*v_F)/(2*pi*k_B*T_c).",
    )

    # Crystal axes rotation.
    parser.add_argument(
        "-r",
        "--crystal-axes-rotation",
        type=float,
        default=None,
        help="Rotation between grain and crystal ab-axes, in units of pi.",
    )

    # Geometry.
    group_geometry = parser.add_mutually_exclusive_group()
    group_geometry.add_argument(
        "--disc-geometry", action="store_true", help="Run with disc geometry"
    )
    group_geometry.add_argument(
        "--square-geometry", action="store_true", help="Run with square geometry."
    )

    # Solenoid.
    group_solenoid = parser.add_mutually_exclusive_group()
    group_solenoid.add_argument(
        "--solenoid",
        action="store_true",
        help="Set up a solenoid, and use solenoid gauge for the external field.",
    )
    group_solenoid.add_argument(
        "--no-solenoid", action="store_true", help="Do not set up a solenoid."
    )

    # =============== ACCURACY =============== #

    # Convergence criterion.
    parser.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criteron for self-consistency.",
    )

    # Lattice side length.
    parser.add_argument(
        "-N",
        "--lattice-side-length",
        type=int,
        default=None,
        help="Number of discrete lattice points in the grain. Set to odd value for best convergence. Recommended to be at least 10 times the grain size.",
    )

    # Num Fermi momenta.
    parser.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Set to "
        "odd value for best convergence. Recommended minumum: 101.",
    )

    # Num iterations max.
    parser.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run.",
    )

    # Num iterations min.
    parser.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run.",
    )

    # Num Ozaki energies.
    parser.add_argument(
        "-e",
        "--num-ozaki-energies",
        type=int,
        default=None,
        help="Number of Ozaki energies (similar to Matsubara energies, see the documentation).",
    )

    # =============== MISC =============== #

    # Data path.
    parser.add_argument(
        "-o",
        "--data-path",
        type=str,
        default=None,
        help="Base folder for output (data and parameter files).",
    )

    # Verbose.
    group_verbose = parser.add_mutually_exclusive_group()
    group_verbose.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print iteration status and other output to terminal.",
    )
    group_verbose.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print iteration status and other output to terminal.",
    )

    # Save.
    group_save = parser.add_mutually_exclusive_group()
    group_save.add_argument(
        "--save-data", action="store_true", help="Save data files to disk."
    )
    group_save.add_argument(
        "--no-save-data", action="store_true", help="Do not save data files to disk."
    )

    # Save data interval.
    parser.add_argument(
        "--save-data-interval",
        type=int,
        default=None,
        help="How frequently to save data.",
    )

    # Visualize.
    group_visualize = parser.add_mutually_exclusive_group()
    group_visualize.add_argument(
        "--visualize", action="store_true", help="Run with real-time plotting."
    )
    group_visualize.add_argument(
        "--no-visualize", action="store_true", help="Run without real-time plotting."
    )

    return parser


def set_value(simparams: dict, key: str, value: Union[int, float]) -> None:
    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        simparams[key] = value


def set_flag(simparams: dict, key: str, on: bool, off: bool) -> None:
    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        simparams[key] = on


def check_value(key: str, value: Union[bool, float, int], checks: dict) -> bool:
    # Check that the value is of the expected type.
    expected_type = checks["type"]
    if not isinstance(value, expected_type):
        print(f"--The field '{key}' must be of type {expected_type}.")
        return False

    # Check that the value is larger than 'low'.
    if "low" in checks:
        low = checks["low"]
        if not value > low:
            print(f"--The field '{key}' must be larger than {low}.")
            return False

    # No errors.
    return True


def validate(simparams: dict) -> bool:
    # A dict with the checks needed for all parameters.
    checks = {
        "convergence_criterion": {"type": float, "low": 0.0},
        "crystal_axes_rotation": {"type": float},
        "data_path": {"type": str},
        "disc_geometry": {"type": bool},
        "dwave_symmetry": {"type": bool},
        "external_flux_quanta": {"type": float},
        "grain_side_length": {"type": float, "low": 0.0},
        "lattice_side_length": {"type": int, "low": 1},
        "num_fermi_momenta": {"type": int, "low": 1},
        "num_iterations_max": {"type": int, "low": 1},
        "num_iterations_min": {"type": int, "low": 1},
        "num_ozaki_energies": {"type": int, "low": 1},
        "save_data": {"type": bool},
        "save_data_interval": {"type": int, "low": 0},
        "solenoid": {"type": bool},
        "temperature": {"type": float, "low": 0.0},
        "verbose": {"type": bool},
        "visualize": {"type": bool},
    }

    # General checks.
    success = True
    for key in checks.keys():
        if not key in simparams:
            print(f"--The field '{key}' must exist.")
            success = False
            continue
        success &= check_value(key=key, value=simparams[key], checks=checks[key])

    return success


def main() -> None:
    # Create the argument parser.
    parser = make_default_parser()

    # Parse all arguments.
    args = parser.parse_args()

    # Simulation parameters dictionary.
    simparams = {}

    # If config file is set, load it first.
    if args.file is not None:
        # Read the JSON file.
        with open(args.file, "r") as f:
            data = f.read()

        # Parse the JSON file.
        simparams = json.loads(data)

    # Override values from the command line.
    set_value(simparams, "convergence_criterion", args.convergence_criterion)
    set_value(simparams, "crystal_axes_rotation", args.crystal_axes_rotation)
    set_value(simparams, "data_path", args.data_path)
    set_flag(simparams, "disc_geometry", args.disc_geometry, args.square_geometry)
    set_flag(simparams, "dwave_symmetry", args.dwave_symmetry, args.swave_symmetry)
    set_value(simparams, "external_flux_quanta", args.external_flux_quanta)
    set_value(simparams, "grain_side_length", args.grain_side_length)
    set_value(simparams, "lattice_side_length", args.lattice_side_length)
    set_value(simparams, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(simparams, "num_iterations_max", args.num_iterations_max)
    set_value(simparams, "num_iterations_min", args.num_iterations_min)
    set_value(simparams, "num_ozaki_energies", args.num_ozaki_energies)
    set_flag(simparams, "save_data", args.save_data, args.no_save_data)
    set_value(simparams, "save_data_interval", args.save_data_interval)
    set_flag(simparams, "solenoid", args.solenoid, args.no_solenoid)
    set_value(simparams, "temperature", args.temperature)
    set_flag(simparams, "verbose", args.verbose, args.quiet)
    set_flag(simparams, "visualize", args.visualize, args.no_visualize)

    # Validate parameters, exit if there are errors.
    success = validate(simparams)
    if not success:
        sys.exit()

    # Write config file.
    config_file = ".tmp_config.json"
    with open(config_file, "w") as f:
        json.dump(simparams, f, ensure_ascii=False, indent=4)

    # Run the C++ program.
    cpp_file = args.c_binary
    if os.path.isfile(cpp_file):
        subprocess.call(["./" + cpp_file, config_file])
    else:
        print(
            f"--Cannot find the C++ executable '{cpp_file}'. See the README for how to compile it."
        )

    # Remove config file.
    if os.path.isfile(config_file):
        os.remove(config_file)


if __name__ == "__main__":
    main()
