import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Union

VALID_SYMMETRIES = ["s", "dx2_y2", "dxy"]


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

    # Symmetry.
    parser.add_argument(
        "-H",
        "--symmetry",
        type=str,
        default=None,
        choices=VALID_SYMMETRIES,
        nargs="+",
        help="Order parameter pairing symmetry. Multiple values can be passed, one for each order parameter component.",
    )

    # Relative Tc.
    parser.add_argument(
        "-C",
        "--critical-temperature",
        type=float,
        default=None,
        nargs="+",
        help="Order parameter critical temperature. Multiple values can be passed, one for each order parameter component. Note that all values will be scaled so that the maximum is 1.",
    )

    # Phase shift.
    parser.add_argument(
        "-J",
        "--initial-phase-shift",
        type=float,
        default=None,
        nargs="+",
        help="The phase shift of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Num phase windings.
    parser.add_argument(
        "-W",
        "--initial-phase-winding",
        type=float,
        default=None,
        nargs="+",
        help="The phase winding of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Random phase weight.
    parser.add_argument(
        "-R",
        "--initial-random-phase-weight",
        type=float,
        default=None,
        nargs="+",
        help="The weight of the random phase of the initial order parameter guess. Multiple values can be passed, one for each order parameter component.",
    )

    # Temperature.
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Temperature in units of T_c.",
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
        help="Rotation between grain and crystal ab-axes, in units of 2pi.",
    )

    # Geometry.
    parser.add_argument(
        "-z",
        "--num-polygon-edges",
        type=int,
        default=None,
        help="Number edges of the grain polygon. Negative values yield a circle.",
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


def set_value(
    simparams: dict, key: str, value: Union[int, float], group: Optional[str] = None
) -> None:
    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        if group is None:
            simparams[key] = value
        else:
            simparams[group][key] = value


def set_flag(simparams: dict, key: str, on: bool, off: bool) -> None:
    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        simparams[key] = on


def check_value(
    key: str, value: Union[bool, float, int, list, str], checks: dict
) -> bool:
    # Check that the value is of the expected type.
    expected_type = checks["type"]
    if not isinstance(value, expected_type):
        print(f"--The field '{key}' must be of type {expected_type}.")
        return False

    if isinstance(value, list):
        # If the expected type is a list, check that the elements are of the expected type.
        expected_type = checks["elem_type"]
        for elem in value:
            if not isinstance(elem, expected_type):
                print(
                    f"--The field '{key}' must be a list with elements of type {expected_type}."
                )
                return False
    else:
        # Convert to list for convenience.
        value = [value]

    # Check that the value is larger than 'low'.
    if "low" in checks:
        low = checks["low"]
        for elem in value:
            if not elem > low:
                print(f"--The field '{key}' must be larger than {low}.")
                return False

    # Check that the value is not in the excluded list.
    if "excluded" in checks:
        for exc in checks["excluded"]:
            if value == exc:
                print(f"--The field '{key}' cannot have value {exc}.")
                return False

    # Check that the value is in some pre-defined set.
    if "one_of" in checks:
        valid_values = checks["one_of"]
        for elem in value:
            if not elem in valid_values:
                print(f"--The field '{key}' must be one of {valid_values}.")
                return False

    # No errors.
    return True


def check_dict(simparams: dict, checks: dict) -> bool:
    success = True

    # Loop over the keys in the checks so we can spot if something is missing.
    for key in checks.keys():
        if not key in simparams:
            print(f"--The field '{key}' must exist.")
            success = False
            continue

        if isinstance(simparams[key], dict):
            # If this is a dict on its own call this function recursively.
            success &= check_dict(simparams=simparams[key], checks=checks[key])
        else:
            # If not a dict then check the values.
            success &= check_value(key=key, value=simparams[key], checks=checks[key])

    return success


def validate(simparams: dict) -> bool:
    # A dict with the checks needed for all parameters.
    checks = {
        "convergence_criterion": {"type": float, "low": 0.0},
        "crystal_axes_rotation": {"type": float},
        "data_path": {"type": str},
        "external_flux_quanta": {"type": float},
        "grain_side_length": {"type": float, "low": 0.0},
        "lattice_side_length": {"type": int, "low": 1},
        "num_fermi_momenta": {"type": int, "low": 1},
        "num_iterations_max": {"type": int, "low": 1},
        "num_iterations_min": {"type": int, "low": 1},
        "num_ozaki_energies": {"type": int, "low": 1},
        "num_polygon_edges": {"type": int, "excluded": [0, 1, 2]},
        "order_parameter": {
            "critical_temperature": {"type": list, "elem_type": float, "low": 0.0},
            "initial_phase_shift": {"type": list, "elem_type": float},
            "initial_phase_winding": {"type": list, "elem_type": float},
            "initial_random_phase_weight": {"type": list, "elem_type": float},
            "symmetry": {"type": list, "elem_type": str, "one_of": VALID_SYMMETRIES},
        },
        "save_data": {"type": bool},
        "save_data_interval": {"type": int, "low": 0},
        "solenoid": {"type": bool},
        "temperature": {"type": float, "low": 0.0},
        "verbose": {"type": bool},
        "visualize": {"type": bool},
    }

    # General checks.
    success = check_dict(simparams, checks)

    # Special checks.
    if success:
        # For convenience.
        op = simparams["order_parameter"]

        # Check that all lists are of the same length.
        tmp = [
            len(op["symmetry"]),
            len(op["critical_temperature"]),
            len(op["initial_phase_shift"]),
            len(op["initial_phase_winding"]),
            len(op["initial_random_phase_weight"]),
        ]

        if not (tmp.count(tmp[0]) == len(tmp)):
            print("--The order parameter lists must all be of the same length.")
            success = False

        # Normalize critical temperature so that the maximum is one.
        Tc_max = max(op["critical_temperature"])
        for i in range(tmp[0]):
            op["critical_temperature"][i] /= Tc_max
        assert max(op["critical_temperature"]) == 1

        # Set the order parameter.
        simparams["order_parameter"] = op

        if not (simparams["num_iterations_max"] > simparams["num_iterations_min"]):
            print("--'num_iterations_max' must be larger than 'num_iterations_min'.")
            success = False

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
    set_value(simparams, "external_flux_quanta", args.external_flux_quanta)
    set_value(simparams, "grain_side_length", args.grain_side_length)
    set_value(simparams, "lattice_side_length", args.lattice_side_length)
    set_value(simparams, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(simparams, "num_iterations_max", args.num_iterations_max)
    set_value(simparams, "num_iterations_min", args.num_iterations_min)
    set_value(simparams, "num_ozaki_energies", args.num_ozaki_energies)
    set_value(simparams, "num_polygon_edges", args.num_polygon_edges)

    group = "order_parameter"
    set_value(simparams, "critical_temperature", args.critical_temperature, group)
    set_value(simparams, "initial_phase_shift", args.initial_phase_shift, group)
    set_value(simparams, "initial_phase_winding", args.initial_phase_winding, group)
    set_value(
        simparams,
        "initial_random_phase_weight",
        args.initial_random_phase_weight,
        group,
    )
    set_value(simparams, "symmetry", args.symmetry, group)

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
