import argparse
import json
import math
import os
import pandas as pd


ENERGY_SCALE = 2.0 * math.pi
LENGTH_SCALE = 1.0 / ENERGY_SCALE
AREA_SCALE = LENGTH_SCALE ** 2


CSV_FILES = {
    "order_parameter": {"scale": 1.0 / ENERGY_SCALE, "column": None},
    "vector_potential": {"scale": math.pi * LENGTH_SCALE, "column": None},
    "magnetic_flux_density": {"scale": math.pi * AREA_SCALE, "column": None},
    "current_density": {"scale": 2.0, "column": None},
    "dos": {"scale": 1.0 / ENERGY_SCALE, "column": "energy"},
    "convergence": {
        "scale": 1 / ENERGY_SCALE ** 2,
        "column": "free_energy_per_unit_area",
    },
}

JSON_FILES = {
    "internal_parameters": {"area_compute": 1 / AREA_SCALE},
    "postprocess": {
        "energy_max": 1.0 / ENERGY_SCALE,
        "energy_min": 1.0 / ENERGY_SCALE,
        "energy_smearing": 1.0 / ENERGY_SCALE,
    },
    "results": {
        "current_density_average_magnitude": 2.0,
        "flux_density_average": math.pi * AREA_SCALE,
        "free_energy_per_unit_area": 1.0 / ENERGY_SCALE ** 2,
        "magnetic_free_energy_per_unit_area": 1.0 / ENERGY_SCALE ** 2,
        "order_parameter_average_magnitude": 1.0 / ENERGY_SCALE,
        "vector_potential_average_magnitude": math.pi * LENGTH_SCALE,
    },
}


# Get all leaf directories in this path.
def get_leaf_directories(path) -> list:
    dirs = []
    for root, sub_dirs, files in os.walk(path):
        if not sub_dirs:
            dirs.append(root)
    return dirs


# Scale a CSV file.
def scale_csv(path, scale, column) -> None:
    df = pd.read_csv(path)
    if column is None:
        column_names = df.columns.values.tolist()
        for idx in column_names:
            if (idx == "x_idx") or (idx == "y_idx"):
                # The spatial indices should always be excluded.
                continue
            df[idx] *= scale
    else:
        df[column] *= scale
    df.to_csv(path, index=False)


# Scale a JSON file.
def scale_json(path, info) -> None:
    with open(path, "r") as f:
        data = f.read()
    params = json.loads(data)
    for key, scale in info.items():
        params[key] *= scale
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    # Parse argument.
    parser = argparse.ArgumentParser(
        prog="Old Data Scaler",
        description="Scale old data files to the new internal units. This tool will be removed in a future version.",
    )
    parser.add_argument(
        dest="path",
        type=str,
        default=None,
        help="Path to a directory. All files in subdirectories will be rescaled and overwritten.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite of the files."
    )
    args = parser.parse_args()

    # Get a list of all relevant directories.
    leaf_directories = get_leaf_directories(args.path)

    # Loop over leaf directories.
    for leaf in leaf_directories:
        # Scale CSV files.
        for key, info in CSV_FILES.items():
            path = os.path.join(leaf, key + ".csv")
            if os.path.exists(path):
                if args.force:
                    print(f"Scaling {path}.")
                    scale_csv(path=path, scale=info["scale"], column=info["column"])
                else:
                    print(f"Would scale {path}.")

        # Scale JSON files.
        for key, info in JSON_FILES.items():
            path = os.path.join(leaf, key + ".json")
            if os.path.exists(path):
                if args.force:
                    print(f"Scaling {path}.")
                    scale_json(path=path, info=info)
                else:
                    print(f"Would scale {path}.")


if __name__ == "__main__":
    main()
