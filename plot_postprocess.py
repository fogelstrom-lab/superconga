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
    csv_path = os.path.join(data_path, "ldos_min_energy.csv")
    df = pd.read_csv(csv_path, header=0)

    # Get the lattice side length.
    numLatticePoints = len(df)
    N = int(np.sqrt(numLatticePoints))
    assert N ** 2 == numLatticePoints

    # Convert to NumPy arrays.
    x = np.reshape(df["x_idx"].tolist(), [N, N])
    y = np.reshape(df["y_idx"].tolist(), [N, N])
    ldos = np.reshape(df["ldos"].tolist(), [N, N])

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

    # Plot the LDOS (for the lowest energy).
    ax = fig.add_subplot(122)
    cax = ax.pcolormesh(x, y, ldos, shading="auto")
    cbar_out = fig.colorbar(cax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("LDOS")

    # Show the figure.
    plt.show()


if __name__ == "__main__":
    main()
