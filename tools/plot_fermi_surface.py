import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


# Helper data class to contain the tight-binding parameters.
@dataclass
class Parameters:
    tt: float
    tp: float
    tq: float
    mu: float


# Checks whether the Fermi surface is centered around the origin,
# given some tight-binding parameters.
def isFermiSurfaceCentered(parameters):
    kx = np.array([np.pi, np.pi])
    ky = np.array([0.0, np.pi])
    ek = computeDispersion(kx, ky, parameters)
    return (np.sign(ek[0]) == np.sign(ek[1]))


# Computes the dispersion given som momenta and tight-binding parameters.
def computeDispersion(kx, ky, parameters):
    ct = -2.0 * parameters.tt * (np.cos(kx) + np.cos(ky))
    cp = -4.0 * parameters.tp * np.cos(kx) * np.cos(ky)
    cq = -2.0 * parameters.tq * (np.cos(2.0 * kx) + np.cos(2.0 * ky))
    return (ct + cp + cq - parameters.mu)


# Compute the Fermi momenta using the bisection method.
def computeFermiMomentum(parameters,
                         angles,
                         offset,
                         numIterations=100,
                         convergenceCriteration=1e-6):
    # Reshape to [numAngles, spatatial dimension].
    angles = np.reshape(angles, [-1, 1])
    offset = np.reshape(offset, [1, -1])

    # For convenience.
    cosPhi = np.cos(angles)
    sinPhi = np.sin(angles)

    # Compute dispersion at the origin.
    kA = np.zeros_like(angles)
    kAX = kA * cosPhi + offset[:, 0]
    kAY = kA * sinPhi + offset[:, 1]
    eA = computeDispersion(kAX, kAY, parameters)

    # Compute dispersion at the edge.
    kB = np.pi / np.maximum(np.abs(cosPhi), np.abs(sinPhi))
    kBX = kB * cosPhi + offset[:, 0]
    kBY = kB * sinPhi + offset[:, 1]
    eB = computeDispersion(kBX, kBY, parameters)

    # Initial guess.
    k = 0.5 * (kA + kB)

    # The main loop.
    for _ in range(numIterations):
        # Form the momentum.
        kX = k * cosPhi + offset[:, 0]
        kY = k * sinPhi + offset[:, 1]

        # Compute the dispersion.
        dispersion = computeDispersion(kX, kY, parameters)

        # The error is simply the magnitude of the dispersion.
        # We are looking for zeros.
        error = np.abs(dispersion)

        # Compare signs.
        cond = np.sign(dispersion) == np.sign(eA)

        # Update the interval dispersions.
        eA = np.where(cond, dispersion, eA)
        eB = np.where(cond, eB, dispersion)

        # Update the interval momenta.
        kA = np.where(cond, k, kA)
        kB = np.where(cond, kB, k)

        # Update the momentum.
        k = 0.5 * (kA + kB)

    # Check that everything has converged.
    assert (np.amax(error) < convergenceCriteration)

    # Form the final momenta.
    kX = k * cosPhi + offset[:, 0]
    kY = k * sinPhi + offset[:, 1]

    # For convenience.
    twoPi = 2.0 * np.pi

    # Wrap kX to [-pi, pi].
    kX = np.where(kX < -np.pi, kX + twoPi, kX)
    kX = np.where(kX > np.pi, kX - twoPi, kX)

    # Wrap kY to [-pi, pi].
    kY = np.where(kY < -np.pi, kY + twoPi, kY)
    kY = np.where(kY > np.pi, kY - twoPi, kY)

    return kX, kY


def computeFermiSurface(parameters, numGrid, numAngles):
    # Just print the tight-binding parameters.
    print(parameters)

    # Is the Fermi surface centered around (0, 0)?
    isCentered = isFermiSurfaceCentered(parameters)
    print("Is the Fermi surface centered around (0, 0)?", isCentered)

    # If it is not centered around the origin, then it is centered around (pi, pi).
    if isCentered:
        offset = np.array([0.0, 0.0])
    else:
        offset = np.array([np.pi, np.pi])

    # Create meshgrid.
    v = np.linspace(-np.pi, np.pi, numGrid)
    kxMesh, kyMesh = np.meshgrid(v, v)

    # Compute the dispersion.
    ek = computeDispersion(kxMesh, kyMesh, parameters)

    # Check the sign.
    ek = np.sign(ek)

    # Plot the Fermi surface (where the sign changes).
    plt.imshow(ek,
               cmap='hot',
               interpolation='none',
               extent=[-np.pi, np.pi, -np.pi, np.pi])

    # Compute Fermi momenta using the bisection method.
    angles = np.linspace(-np.pi, np.pi, numAngles)
    kx, ky = computeFermiMomentum(parameters, angles, offset)
    plt.scatter(kx, ky, c='r', linewidths=0.1, marker='.')

    # Set axis labels.
    plt.gca().set_xlabel('k_x')
    plt.gca().set_ylabel('k_y')

    # Show plot.
    plt.show()


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description=
        'Compute the Fermi surface for a tight-binding model on a square lattice.'
    )
    parser.add_argument(
        '-tt',
        '--hopping-nn',
        type=float,
        default=1.0,
        help='Hopping nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-tp',
        '--hopping-nnn',
        type=float,
        default=0.0,
        help='Hopping next-nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-tq',
        '--hopping-nnnn',
        type=float,
        default=0.0,
        help='Hopping next-next-nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-mu',
        '--chemical-potential',
        type=float,
        default=1.0,
        help='Hopping nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-grid',
        '--num-grid',
        type=int,
        default=1000,
        help='Number of grid points per side. (default: %(default)s)')
    parser.add_argument('-angles',
                        '--num-angles',
                        type=int,
                        default=1000,
                        help='Number of angles. (default: %(default)s)')
    args = parser.parse_args()

    parameters = Parameters(tt=args.hopping_nn,
                            tp=args.hopping_nnn,
                            tq=args.hopping_nnnn,
                            mu=args.chemical_potential)

    # Compute (and plot) the Fermi surface.
    computeFermiSurface(parameters,
                        numGrid=args.num_grid,
                        numAngles=args.num_angles)


if __name__ == '__main__':
    main()
