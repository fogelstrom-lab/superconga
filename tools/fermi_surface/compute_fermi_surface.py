import argparse
import json
import numpy as np
import os


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
    c0 = -2.0 * parameters['t0'] * (np.cos(kX) + np.cos(kY))

    # Next-nearest neighbour.
    c1 = -4.0 * parameters['t1'] * np.cos(kX) * np.cos(kY)

    # Next-next-nearest neighbour.
    c2 = -2.0 * parameters['t2'] * (np.cos(2.0 * kX) + np.cos(2.0 * kY))

    # The total dispersion.
    dispersion = (c0 + c1 + c2 - parameters['mu'])
    return dispersion


# Compute the partial derivatives of the dispersion.
def computeDispersionDerivative(kX, kY, parameters):
    # Nearest neighbour.
    factorNN = 2.0 * parameters['t0']
    termNNdX = factorNN * np.sin(kX)
    termNNdY = factorNN * np.sin(kY)

    # Next-nearest neighbour.
    factorNNN = 4.0 * parameters['t1']
    termNNNdX = factorNNN * np.sin(kX) * np.cos(kY)
    termNNNdY = factorNNN * np.cos(kX) * np.sin(kY)

    # Next-next-nearest neighbour.
    factorNNNN = 4.0 * parameters['t2']
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
                         numIterations,
                         convergenceCriterion,
                         wrapResult=True):
    print('Computing the Fermi momentum...')
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
        isConverged = (np.amax(error) < convergenceCriterion)
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
        cond = np.abs(dEdK) > 0.0
        update = np.where(cond, -dispersion / dEdK, 0.0)
        kNewton = k + update

        # Switch to doing Newton steps when the error is sufficently small.
        doBisection = np.logical_and(error > np.sqrt(convergenceCriterion),
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


def computeIntegrandFactor(inFermiMomentumX, inFermiMomentumY,
                           inFermiVelocityX, inFermiVelocityY, parameters,
                           offset):
    # Compute the partial derivatives of the dispersion.
    dEdKX, dEdKY = computeDispersionDerivative(inFermiMomentumX,
                                               inFermiMomentumY, parameters)

    # The momentum is parameterized as
    # pX = kX + offsetX
    # pY = kY + offsetY
    # with
    # kX = k * cos(phi)
    # kY = k * sin(phi)
    kXUnwrapped = wrapToRange(inFermiMomentumX, offset[0] - np.pi,
                              offset[0] + np.pi)
    kYUnwrapped = wrapToRange(inFermiMomentumY, offset[1] - np.pi,
                              offset[1] + np.pi)
    kX = kXUnwrapped - offset[0]
    kY = kYUnwrapped - offset[1]
    k = np.hypot(kX, kY)
    cosPhi = kX / k
    sinPhi = kY / k

    # The derivative of the dispersion with respect to phi is:
    # dEdPhi = dEdKX * dKXdPhi + dEdKY * dKYdPhi = 0
    # whith
    # dKXdPhi = (dKdPhi * cos(phi) - kY)
    # dKYdPhi = (dKdPhi * sin(phi) + kX)
    # which gives
    dKdPhi = (dEdKX * kY - dEdKY * kX) / (dEdKX * cosPhi + dEdKY * sinPhi)

    # Thus, the norm of the partial derivative of the Fermi momentum with respect
    # to phi is given by:
    # sqrt(dPXdPhi^2 + dPYdPhi^2) = sqrt(dKXdPhi^2 + dKYdPhi^2)
    dKXdPhi = dKdPhi * cosPhi - kY
    dKYdPhi = dKdPhi * sinPhi + kX
    dPdPhi = np.hypot(dKXdPhi, dKYdPhi)

    fermiSpeed = np.hypot(inFermiVelocityX, inFermiVelocityY)
    integrandFactor = dPdPhi / fermiSpeed
    integrandFactor /= np.sum(integrandFactor)
    return integrandFactor


def computeFermiSurfaceAverage(x, factors):
    return np.sum(x * factors)


def printPaperStuff(inFermiMomentumX, inFermiMomentumY, inFermiVelocityX,
                    inFermiVelocityY, offset, parameters):
    # Some stuff to compute averages of.
    fermiSpeed = np.hypot(inFermiVelocityX, inFermiVelocityY)
    fermiMomentum = np.hypot(inFermiMomentumX, inFermiMomentumY)
    invFermiMomentum = 1.0 / fermiMomentum

    # Compute integrand factors.
    integrandFactors = computeIntegrandFactor(inFermiMomentumX,
                                              inFermiMomentumY,
                                              inFermiVelocityX,
                                              inFermiVelocityY, parameters,
                                              offset)

    # Compute Fermi surface averages.
    avgFermiSpeed = computeFermiSurfaceAverage(fermiSpeed, integrandFactors)
    avgFermiMomentum = computeFermiSurfaceAverage(fermiMomentum,
                                                  integrandFactors)
    avgInvFermiMomentum = computeFermiSurfaceAverage(invFermiMomentum,
                                                     integrandFactors)

    # Print results (and the means).
    print('<vf> = %.8f - Mean(vf) = %.8f' %
          (avgFermiSpeed, np.mean(fermiSpeed)))
    print('<kf> = %.8f - Mean(kf) = %.8f' %
          (avgFermiMomentum, np.mean(fermiMomentum)))
    print('1/<kf> = %.8f - 1/Mean(kf) = %.8f' %
          (1.0 / avgFermiMomentum, 1.0 / np.mean(fermiMomentum)))
    print('<1/kf> = %.8f - Mean(1/kf) = %.8f' %
          (avgInvFermiMomentum, np.mean(invFermiMomentum)))


def wrapToRange(inValue, inLow=-np.pi, inHigh=np.pi):
    # Determine the range.
    r = inHigh - inLow

    # Standardize the value so that [0, 1.0) means that the value is in the
    # desired range.
    standardizedValue = (inValue - inLow) / r

    # Wrap the value. Example:
    # r = 4.0
    # inLow = 2.0
    # standardizedValue = 1.5
    # ==> wrappedValue = (1.5 - 1.0) * 4.0 + 2.0 = 4.0
    wrappedValue = (standardizedValue -
                    np.floor(standardizedValue)) * r + inLow
    return wrappedValue


def computeDensityOfStates(inEnergies,
                           inMomentumX,
                           inMomentumY,
                           inDispersion,
                           inGamma=0.01):
    print('Computing the DOS...')
    E = np.reshape(inEnergies, [-1, 1, 1])
    kx = np.expand_dims(inMomentumX, axis=0)
    ky = np.expand_dims(inMomentumY, axis=0)
    ek = np.expand_dims(inDispersion, axis=0)

    halfGamma = 0.5 * inGamma

    dos = np.zeros_like(inEnergies)
    for n, E in enumerate(inEnergies):
        x = E - ek
        lorentzian = halfGamma / (np.pi * (x**2 + halfGamma**2))
        dos[n] = np.mean(lorentzian, axis=(1, 2))
    return dos


def computeFermiSurface(parameters, numGrid, numEnergies, numAngles,
                        numIterations, convergenceCriterion):
    print('Computing Fermi surface...')

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

    # ==================== DISPERSION ==================== #
    # Create meshgrid.
    v = np.linspace(-np.pi, np.pi * (1 - 1 / numGrid), numGrid)
    kxMesh, kyMesh = np.meshgrid(v, v)

    # Compute the dispersion.
    dispersion = computeDispersion(kxMesh, kyMesh, parameters)

    dispersionDict = {
        'kx': kxMesh.tolist(),
        'ky': kyMesh.tolist(),
        'dispersion': dispersion.tolist()
    }

    # ==================== DOS ==================== #
    lim = 7.5
    energies = np.linspace(-lim, lim, numEnergies)
    dos = computeDensityOfStates(energies, kxMesh, kyMesh, dispersion)

    dosDict = {
        'energies': energies.tolist(),
        'dos': dos.tolist(),
    }

    # ==================== FERMI VELOCITIES ==================== #
    # Note that K is wrapped to [-pi, pi].
    minAngle = 0.0
    maxAngle = 2.0 * np.pi * (1.0 - 1.0 / numAngles)
    angles = np.linspace(minAngle, maxAngle, numAngles)
    kX, kY = computeFermiMomentum(parameters, angles, offset, numIterations,
                                  convergenceCriterion)

    # Compute the Fermi velocities.
    vX, vY = computeDispersionDerivative(kX, kY, parameters)

    # Just to compare with Micke.
    printPaperStuff(kX, kY, vX, vY, offset, parameters)

    fermiVelocityDict = {
        'kx': kX.tolist(),
        'ky': kY.tolist(),
        'vx': vX.tolist(),
        'vy': vY.tolist()
    }

    return dispersionDict, fermiVelocityDict, dosDict


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description=
        'Compute the Fermi surface for a tight-binding model on a square lattice.'
    )
    parser.add_argument(
        '-name',
        '--name',
        type=str,
        default=None,
        help='Name of file to save. '
        'If not passed the file will not be saved. (default: %(default)s)')
    parser.add_argument(
        '-mu',
        '--chemical-potential',
        type=float,
        default=1.0,
        help='The chemical potential, '
        'in terms of the nearest-neighbour hopping. (default: %(default)s)')
    parser.add_argument(
        '-t1',
        '--hopping-nnn',
        type=float,
        default=0.0,
        help='The next-nearest neighbour hopping, '
        'in terms of the nearest-neighbour hopping.  (default: %(default)s)')
    parser.add_argument(
        '-t2',
        '--hopping-nnnn',
        type=float,
        default=0.0,
        help='The next-next-nearest neighbour hopping, '
        'in terms of the nearest-neighbour hopping.  (default: %(default)s)')
    parser.add_argument(
        '-grid',
        '--num-grid',
        type=int,
        default=1000,
        help='Number of grid points per side. (default: %(default)s)')
    parser.add_argument(
        '-ne',
        '--num-energies',
        type=int,
        default=1000,
        help=
        'Number of energies in the density of states. (default: %(default)s)')
    parser.add_argument(
        '-na',
        '--num-angles',
        type=int,
        default=100,
        help='Number of Fermi velocity angles. (default: %(default)s)')
    parser.add_argument(
        '-ni',
        '--num-iterations',
        type=int,
        default=100,
        help='Maximum number of iterations. (default: %(default)s)')
    parser.add_argument('-cc',
                        '--convergence-criterion',
                        type=float,
                        default=1e-6,
                        help='Convergence criterion. (default: %(default)s)')
    args = parser.parse_args()

    # Set tight-binding parameters.
    parameters = {
        't0': 1.0,
        't1': args.hopping_nnn,
        't2': args.hopping_nnnn,
        'mu': args.chemical_potential
    }

    # Compute the Fermi surface.
    dispersion, fermiVelocity, dos = computeFermiSurface(
        parameters,
        numGrid=args.num_grid,
        numEnergies=args.num_energies,
        numAngles=args.num_angles,
        numIterations=args.num_iterations,
        convergenceCriterion=args.convergence_criterion)

    # Save file.
    if args.name is not None:
        filename = args.name
        suffix = '.json'
        if not filename.endswith(suffix):
            filename += '.json'
        file = open(filename, 'w')
        data = {
            'dispersion': dispersion,
            'fermiVelocity': fermiVelocity,
            'dos': dos
        }

        json.dump(data, file)


if __name__ == '__main__':
    main()
