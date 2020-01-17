import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

NUM_ITERATIONS = 100
CONVERGENCE_CRITERION = 1e-6


# Helper data class to contain the tight-binding parameters.
@dataclass
class Parameters:
    t0: float
    t1: float
    t2: float
    mu: float


# Checks whether the Fermi surface is centered around the origin,
# given some tight-binding parameters.
def isFermiSurfaceCentered(parameters):
    kX = np.array([np.pi, np.pi])
    kY = np.array([0.0, np.pi])
    ek = computeDispersion(kX, kY, parameters)
    return (np.sign(ek[0]) == np.sign(ek[1]))


# Computes the dispersion given som momenta and tight-binding parameters.
def computeDispersion(kX, kY, parameters):
    # Nearest neighbour.
    ct = -2.0 * parameters.t0 * (np.cos(kX) + np.cos(kY))

    # Next-nearest neighbour.
    cp = -4.0 * parameters.t1 * np.cos(kX) * np.cos(kY)

    # Next-next-nearest neighbour.
    cq = -2.0 * parameters.t2 * (np.cos(2.0 * kX) + np.cos(2.0 * kY))

    # The total dispersion.
    dispersion = (ct + cp + cq - parameters.mu)
    return dispersion


# Compute the partial derivatives of the dispersion.
def computeDispersionDerivative(kX, kY, parameters):
    # Nearest neighbour.
    factorNN = 2.0 * parameters.t0
    termNNdX = factorNN * np.sin(kX)
    termNNdY = factorNN * np.sin(kY)

    # Next-nearest neighbour.
    factorNNN = 4.0 * parameters.t1
    termNNNdX = factorNNN * np.sin(kX) * np.cos(kY)
    termNNNdY = factorNNN * np.cos(kX) * np.sin(kY)

    # Next-next-nearest neighbour.
    factorNNNN = 4.0 * parameters.t2
    termNNNNdX = factorNNNN * np.sin(2.0 * kX)
    termNNNNdY = factorNNNN * np.sin(2.0 * kY)

    # The partial derivatives of the dispersion.
    dEdX = (termNNdX + termNNNdX + termNNNNdX)
    dEdY = (termNNdY + termNNNdY + termNNNNdY)
    return dEdX, dEdY


# Compute the Fermi momenta using the bisection method.
def computeFermiMomentum(parameters,
                         angles,
                         offset,
                         numIterations=NUM_ITERATIONS,
                         convergenceCriteration=CONVERGENCE_CRITERION,
                         wrapResult=True):
    # For convenience.
    cosPhi = np.cos(angles)
    sinPhi = np.sin(angles)

    # Compute dispersion at the origin.
    kA = np.zeros_like(angles)
    kAX = kA * cosPhi + offset[0]
    kAY = kA * sinPhi + offset[1]
    eA = computeDispersion(kAX, kAY, parameters)

    # Compute dispersion at the edge.
    kB = np.pi / np.maximum(np.abs(cosPhi), np.abs(sinPhi))
    kBX = kB * cosPhi + offset[0]
    kBY = kB * sinPhi + offset[1]
    eB = computeDispersion(kBX, kBY, parameters)

    # Initial guess.
    k = 0.5 * (kA + kB)

    # Start with bisection updates.
    doBisection = np.ones_like(k)

    # The main loop.
    isConverged = False
    for _ in range(numIterations):
        # Form the momentum.
        kX = k * cosPhi + offset[0]
        kY = k * sinPhi + offset[1]

        # Compute the dispersion.
        dispersion = computeDispersion(kX, kY, parameters)

        # The error is simply the magnitude of the dispersion.
        # We are looking for zeros.
        error = np.abs(dispersion)

        # Check convergence.
        isConverged = (np.amax(error) < convergenceCriteration)
        if isConverged:
            break

        # ========== BISECTION ========== #
        # Compare signs.
        cond = np.sign(dispersion) == np.sign(eA)

        # Update the interval dispersions.
        eA = np.where(cond, dispersion, eA)
        eB = np.where(cond, eB, dispersion)

        # Update the interval momenta.
        kA = np.where(cond, k, kA)
        kB = np.where(cond, kB, k)

        # Bisection update.
        kBisection = 0.5 * (kA + kB)

        # =========== NEWTON =========== #
        # Compute derivative along the radial direction.
        dEdX, dEdY = computeDispersionDerivative(kX, kY, parameters)
        dEdK = dEdX * cosPhi + dEdY * sinPhi

        # Newton update.
        kNewton = k - dispersion / dEdK

        # Switch to doing Newton steps when the error is sufficently small.
        doBisection = np.logical_and(error > np.sqrt(convergenceCriteration),
                                     doBisection)

        # Update the momentum.
        k = np.where(doBisection, kBisection, kNewton)

    # Check that everything has converged.
    if (not isConverged):
        print('WARNING! Did not converge.')

    # Form the final momenta.
    kX = k * cosPhi + offset[0]
    kY = k * sinPhi + offset[1]

    if wrapResult:
        # Wrap to [-pi, pi].
        kX = wrapToRange(kX)
        kY = wrapToRange(kY)

    return kX, kY


def computeIntegrandFactor(kX,
                           kY,
                           vx,
                           vy,
                           parameters,
                           offset,
                           numIterations=NUM_ITERATIONS,
                           convergenceCriteration=CONVERGENCE_CRITERION,
                           delta=1e-4):
    kXUnwrapped = wrapToRange(kX, offset[0])
    kYUnwrapped = wrapToRange(kY, offset[1])

    # Get parameterizing phi.
    phi = np.arctan2(kYUnwrapped - offset[1], kXUnwrapped - offset[0])
    phiPlusDelta = phi + delta
    phiMinusDelta = phi - delta

    # Compute pF(\phi + \delta).
    kXPlus, kYPlus = computeFermiMomentum(parameters,
                                          phiPlusDelta,
                                          offset,
                                          numIterations,
                                          convergenceCriteration,
                                          wrapResult=False)

    # Compute pF(\phi - \delta).
    kXMinus, kYMinus = computeFermiMomentum(parameters,
                                            phiMinusDelta,
                                            offset,
                                            numIterations,
                                            convergenceCriteration,
                                            wrapResult=False)

    # Differences.
    diffX = kXPlus - kXMinus
    diffY = kYPlus - kYMinus

    # |\frac{\partial p_F}{\partial \phi}|
    dKdPhi = np.sqrt(diffX**2 + diffY**2) / (2.0 * delta)

    # Fermi speed.
    fermiSpeed = np.sqrt(vx**2 + vy**2)

    # Integrand factor.
    factor = dKdPhi / fermiSpeed

    return factor


def computeFermiSurfaceAverage(x, factors):
    numer = np.sum(x * factors)
    denom = np.sum(factors)
    return numer / denom


def wrapToRange(x, offset=0.0):
    # For convenience.
    twoPi = 2.0 * np.pi

    # Wrap kX to [-pi, pi].
    low = offset - np.pi
    high = offset + np.pi
    y = np.where(x < low, x + twoPi, x)
    y = np.where(x > high, x - twoPi, x)
    return y


def computeFermiSurface(parameters, numGrid, numAngles):
    # Just print the tight-binding parameters.
    print(parameters)

    # Is the Fermi surface centered around (0, 0)?
    isCentered = isFermiSurfaceCentered(parameters)
    print('Is the Fermi surface centered around (0, 0)?', isCentered)

    # If it is not centered around the origin, then it is centered around (pi, pi).
    if isCentered:
        offset = np.array([0.0, 0.0])
    else:
        offset = np.array([np.pi, np.pi])

    # Compute Fermi momenta.
    # Note that K is wrapped to [-pi, pi].
    minAngle = 0.0
    maxAngle = 2.0 * np.pi * (1.0 - 1.0 / numAngles)
    angles = np.linspace(minAngle, maxAngle, numAngles)
    kX, kY = computeFermiMomentum(parameters, angles, offset)

    # Compute the Fermi velocities.
    vx, vy = computeDispersionDerivative(kX, kY, parameters)

    # Compute integrand factors.
    integrandFactors = computeIntegrandFactor(kX, kY, vx, vy, parameters,
                                              offset)

    # Some stuff to compute averages of.
    fermiSpeed = np.sqrt(vx**2 + vy**2)
    fermiMomentum = np.sqrt(kX**2 + kY**2)
    invFermiMomentum = 1.0 / fermiMomentum

    # Compute Fermi surface averages.
    avgFermiSpeed = computeFermiSurfaceAverage(fermiSpeed, integrandFactors)
    avgFermiMomentum = computeFermiSurfaceAverage(fermiMomentum,
                                                  integrandFactors)
    avgInvFermiMomentum = computeFermiSurfaceAverage(invFermiMomentum,
                                                     integrandFactors)

    # Print results (and the means).
    print('Fermi Speed: FS Avg = %f - Mean = %f' %
          (avgFermiSpeed, np.mean(fermiSpeed)))
    print('Fermi Momentum: FS Avg = %f - Mean = %f' %
          (avgFermiMomentum, np.mean(fermiMomentum)))
    print('Inv Fermi Momentum: FS Avg = %f - Mean = %f' %
          (avgInvFermiMomentum, np.mean(invFermiMomentum)))

    # ================ PLOTS ================ #

    # Create meshgrid.
    v = np.linspace(-np.pi, np.pi, numGrid)
    kxMesh, kyMesh = np.meshgrid(v, v)

    # Compute the dispersion.
    ek = computeDispersion(kxMesh, kyMesh, parameters)

    # Plot the Fermi surface (where the sign changes).
    plt.imshow(np.sign(ek),
               cmap='hot',
               interpolation='none',
               extent=[-np.pi, np.pi, -np.pi, np.pi])

    # Plot the Fermi momenta.
    plt.scatter(kX, kY, c='r', linewidths=0.1, marker='.')

    # Plot the Fermi velocities.
    plt.quiver(kX, kY, vx, vy, color='b')

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
        '-t0',
        '--hopping-nn',
        type=float,
        default=1.0,
        help='Hopping nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-t1',
        '--hopping-nnn',
        type=float,
        default=0.0,
        help='Hopping next-nearest neighbours. (default: %(default)s)')
    parser.add_argument(
        '-t2',
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
                        default=100,
                        help='Number of angles. (default: %(default)s)')
    args = parser.parse_args()

    # Set tight-binding parameters.
    parameters = Parameters(t0=args.hopping_nn,
                            t1=args.hopping_nnn,
                            t2=args.hopping_nnnn,
                            mu=args.chemical_potential)

    # Compute (and plot) the Fermi surface.
    computeFermiSurface(parameters,
                        numGrid=args.num_grid,
                        numAngles=args.num_angles)


if __name__ == '__main__':
    main()
