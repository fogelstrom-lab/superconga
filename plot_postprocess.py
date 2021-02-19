import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def get_dos(data_path):
    # Read CSV file.
    csv_path = os.path.join(data_path, "dos.csv")
    df = pd.read_csv(csv_path, header=0)

    # Convert to NumPy arrays.
    energy = np.array(df["energy"].tolist())
    dos = np.array(df["dos"].tolist())

    return energy, dos


def get_ldos(data_path):
    # Read CSV file.
    csv_path = os.path.join(data_path, "ldos.csv")
    df = pd.read_csv(csv_path, header=0)

    # Read the internal-parameters file.
    internal_parameters_path = os.path.join(data_path, "internal_parameters.json")
    with open(internal_parameters_path, "r") as f:
        data = f.read()
        internal_parameters_path = json.loads(data)

    # Get the lattice side length.
    numLatticePoints = internal_parameters_path["num_lattice_points"]
    N = int(np.sqrt(numLatticePoints))
    assert N ** 2 == numLatticePoints

    # Convert to NumPy arrays.
    x = np.reshape(df["x_idx"].tolist(), [-1, N, N])
    y = np.reshape(df["y_idx"].tolist(), [-1, N, N])
    ldos = np.reshape(df["ldos"].tolist(), [-1, N, N])

    # The x/y-coordinates are the same for every energy. So we can extract them for a single energy.
    x = x[0, :, :]
    y = y[0, :, :]

    # Read the config file.
    config_path = os.path.join(data_path, "config.json")
    with open(config_path, "r") as f:
        data = f.read()
        config = json.loads(data)

    # Move origin to center, and scale to coherence lengths.
    points_per_coherence_length = config["points_per_coherence_length"]
    offset = 0.5 * (N - 1)
    x = (x - offset) / points_per_coherence_length
    y = (y - offset) / points_per_coherence_length

    return x, y, ldos


def main():
    # Parse data path.
    parser = argparse.ArgumentParser(description="Temporary DOS/LDOS plotter.")
    parser.add_argument("-p", "--data-path", type=str, required=True, help="Data path.")
    parser.add_argument(
        "-i",
        "--energy-index",
        type=int,
        default=0,
        help="Which energy index to plot the LDOS for.",
    )
    args = parser.parse_args()
    data_path = args.data_path

    # Read files and convert to NumPy arrays.
    energy, dos = get_dos(data_path)
    x, y, ldos = get_ldos(data_path)

    # Create figure.
    fig = plt.figure(figsize=(16, 4))

    # Plot the DOS.
    ax = fig.add_subplot(121)
    ax.plot(energy, dos, color="k")
    ax.plot(-np.flip(energy), np.flip(dos), color="k")
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS")
    ax.set_title("DOS")

    # Plot the LDOS.
    ax = fig.add_subplot(122)
    index = args.energy_index
    cax = ax.pcolormesh(x, y, ldos[index, :, :], shading="auto")
    cbar_out = fig.colorbar(cax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("LDOS")

    # Show the figure.
    plt.show()


if __name__ == "__main__":
    main()
