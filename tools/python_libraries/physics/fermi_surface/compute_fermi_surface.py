#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Compute bulk Fermi surface properties based on BdG-theory."""

import argparse
import json
import numpy as np
import os
import time
import math


# Checks whether the Fermi surface is centered around the origin,
# given some tight-binding parameters.
def computeFermiSurfaceOffset(parameters):
    kX = np.array([np.pi, np.pi])
    kY = np.array([0.0, np.pi])
    eK = evalDispersion(parameters, kX, kY)
    isCentered = np.sign(eK[0]) == np.sign(eK[1])
    # If it is not centered around the origin, then it is centered around (pi, pi).
    if isCentered:
        offset = np.array([0.0, 0.0])
    else:
        offset = np.array([np.pi, np.pi])
    return offset


# Computes the dispersion given som momenta and tight-binding parameters.
def evalDispersion(parameters, kX, kY):
    # Nearest neighbour.
    c0 = -2.0 * parameters["t0"] * (np.cos(kX) + np.cos(kY))

    # Next-nearest neighbour.
    c1 = -4.0 * parameters["t1"] * np.cos(kX) * np.cos(kY)

    # Next-next-nearest neighbour.
    c2 = -2.0 * parameters["t2"] * (np.cos(2.0 * kX) + np.cos(2.0 * kY))

    # The total dispersion.
    dispersion = c0 + c1 + c2 - parameters["mu"]
    return dispersion


# Compute the partial derivatives of the dispersion.
def computeDispersionDerivative(parameters, kX, kY):
    # Nearest neighbour.
    factorNN = 2.0 * parameters["t0"]
    termNNdX = factorNN * np.sin(kX)
    termNNdY = factorNN * np.sin(kY)

    # Next-nearest neighbour.
    factorNNN = 4.0 * parameters["t1"]
    termNNNdX = factorNNN * np.sin(kX) * np.cos(kY)
    termNNNdY = factorNNN * np.cos(kX) * np.sin(kY)

    # Next-next-nearest neighbour.
    factorNNNN = 4.0 * parameters["t2"]
    termNNNNdX = factorNNNN * np.sin(2.0 * kX)
    termNNNNdY = factorNNNN * np.sin(2.0 * kY)

    # The partial derivatives of the dispersion.
    dEdX = termNNdX + termNNNdX + termNNNNdX
    dEdY = termNNdY + termNNNdY + termNNNNdY
    return dEdX, dEdY


# Compute the Fermi momenta using the bisection and Newton method.
def computeFermiMomentum(
    parameters, angles, offset, numIterations, convergenceCriterion, wrapResult=True
):
    # For convenience.
    cosPhi = np.cos(angles)
    sinPhi = np.sin(angles)

    # Compute dispersion at the origin.
    kA = np.zeros_like(angles)
    kAX = kA * cosPhi + offset[0]
    kAY = kA * sinPhi + offset[1]
    eA = evalDispersion(parameters, kAX, kAY)

    # Compute dispersion at the edge.
    kB = np.pi / np.maximum(np.abs(cosPhi), np.abs(sinPhi))
    kBX = kB * cosPhi + offset[0]
    kBY = kB * sinPhi + offset[1]
    eB = evalDispersion(parameters, kBX, kBY)

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
        dispersion = evalDispersion(parameters, kX, kY)

        # The error is simply the magnitude of the dispersion.
        # We are looking for zeros.
        error = np.abs(dispersion)

        # Check convergence.
        isConverged = np.amax(error) < convergenceCriterion
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
        dEdX, dEdY = computeDispersionDerivative(parameters, kX, kY)
        dEdK = dEdX * cosPhi + dEdY * sinPhi

        # Newton update.
        cond = np.abs(dEdK) > 0.0
        update = np.where(cond, -dispersion / dEdK, 0.0)
        kNewton = k + update

        # Switch to doing Newton steps when the error is sufficently small.
        doBisection = np.logical_and(error > np.sqrt(convergenceCriterion), doBisection)

        # Update the momentum.
        k = np.where(doBisection, kBisection, kNewton)

    # Check that everything has converged.
    if not isConverged:
        print("-- WARNING! Did not converge.")

    # Form the final momenta.
    kX = k * cosPhi + offset[0]
    kY = k * sinPhi + offset[1]

    if wrapResult:
        # Wrap to [-pi, pi].
        kX = wrapToRange(kX)
        kY = wrapToRange(kY)

    return kX, kY


# Compute the Fermi surface integrand weight factors.
def computeIntegrandWeights(
    parameters,
    inFermiMomentumX,
    inFermiMomentumY,
    inFermiVelocityX,
    inFermiVelocityY,
    offset,
):
    # The momentum is parameterized as
    # pX = kX + offsetX
    # pY = kY + offsetY
    # with
    # kX = k * cos(phi)
    # kY = k * sin(phi)
    kXUnwrapped = wrapToRange(inFermiMomentumX, offset[0] - np.pi, offset[0] + np.pi)
    kYUnwrapped = wrapToRange(inFermiMomentumY, offset[1] - np.pi, offset[1] + np.pi)
    kX = kXUnwrapped - offset[0]
    kY = kYUnwrapped - offset[1]

    # See Eq. (16) on page 7 (1105) in "The Effect of Surfaces on the Tunneling
    # Density of States of an Anisotropically Paired Superconductor" by Buchholtz
    # et. al. (1995).
    integrandFactor = (kX * kX + kY * kY) / np.abs(
        kX * inFermiVelocityX + kY * inFermiVelocityY
    )

    # Normalize.
    integrandFactor /= np.sum(integrandFactor)
    return integrandFactor


# Helper function. Wrap a value to some desired range.
def wrapToRange(inValue, inLow=-np.pi, inHigh=np.pi):
    # Determine the range.
    valueRange = inHigh - inLow

    # Standardize the value so that [0, 1.0) means that the value is in the
    # desired range.
    standardizedValue = (inValue - inLow) / valueRange

    # Wrap the value. Example:
    # valueRange = 4.0
    # inLow = 2.0
    # standardizedValue = 1.5
    # ==> wrappedValue = (1.5 - 1.0) * 4.0 + 2.0 = 4.0
    wrappedValue = (
        standardizedValue - np.floor(standardizedValue)
    ) * valueRange + inLow
    return wrappedValue


# Compute the dispersion in the first Brillouin zone.
def computeDispersion(parameters, numMomenta):
    print("-- Computing the dispersion...")
    start = time.time()

    # Create meshgrid.
    v = np.linspace(-np.pi, np.pi, numMomenta)
    kX, kY = np.meshgrid(v, v)

    # Evaluate the dispersion at every lattice point.
    eK = evalDispersion(parameters, kX, kY)

    dispersionDict = {"kx": kX.tolist(), "ky": kY.tolist(), "values": eK.tolist()}
    stop = time.time()
    print("-- Done! - Time %f [s]" % (stop - start))
    return dispersionDict


# Compute the BdG density of states.
def computeDensityOfStates(
    parameters, numMomenta, numEnergies, margin, gamma, tolerance=0.01
):
    print("-- Computing the density of states...")
    start = time.time()

    # Create meshgrid. Skip +pi to avoid double counting.
    v = np.linspace(-np.pi, np.pi * (1 - 1 / numMomenta), numMomenta)
    kX, kY = np.meshgrid(v, v)

    # Evaluate the dispersion at every lattice point.
    eK = evalDispersion(parameters, kX, kY)

    # Construct array of energies.
    energyMin = np.amin(eK)
    energyMax = np.amax(eK)
    energyMargin = margin * (energyMax - energyMin)
    energyMin -= energyMargin
    energyMax += energyMargin
    energies = np.linspace(energyMin, energyMax, numEnergies)

    halfGamma = 0.5 * gamma

    # TODO(niclas): This is done sequentially because otherwise too much memory
    # is used. But it would be faster to do the computations in batches of,
    # say, 32 energies at a time.
    dos = np.zeros_like(energies)
    for n, E in enumerate(energies):
        diff = E - eK
        lorentzian = halfGamma / (np.pi * (diff ** 2 + halfGamma ** 2))
        dos[n] = np.mean(lorentzian)

    totalDOS = np.mean(dos) * (energies[-1] - energies[0])
    expectedTotalDOS = 1.0
    assert abs(totalDOS - expectedTotalDOS) < tolerance

    # Add spin degree of freedom.
    dos *= 2.0

    dosDict = {"energies": energies.tolist(), "values": dos.tolist()}

    stop = time.time()
    print("-- Done! - Time %f [s]" % (stop - start))
    return dosDict


# Used to force the critical temperature to be positive.
# softplus(x) = log(1 + e^x)
def softplus(x):
    return np.logaddexp(0, x)


# The derivative of the softplus function.
# sigmoid(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))


# Helper convergence check for the critical temperature.
def checkConvergence(x, convergenceCriterion):
    return np.abs(x) < convergenceCriterion


# Compute the BdG critical temperature for an array of chemical potential values.
# We want to find the Tc that solves this equation:
# 1 = 0.5 * V * mean(Yk^2 / |ek| * tanh(0.5 * |ek| / Tc))
# where V is the potential, Yk is the basis function, and ek is the dispersion.
# Equivalently:
# 0 = mean(Yk^2 / |ek| * tanh(0.5 * |ek| / Tc)) - 2 / V
def computeCriticalTemperature(
    parameters,
    numMomenta,
    numChemicalPotentials,
    numIterations,
    convergenceCriterion,
    minChemicalPotential=-2.5,
    maxChemicalPotential=1.0,
    epsilon=1e-16,
):
    print("-- Computing the critical temperature...")
    start = time.time()

    # Create meshgrid. Skip +pi to avoid double counting.
    v = np.linspace(-np.pi, np.pi * (1 - 1 / numMomenta), numMomenta)
    kX, kY = np.meshgrid(v, v)

    chemicalPotentials = np.linspace(
        minChemicalPotential, maxChemicalPotential, numChemicalPotentials
    )
    dWaveTc = np.zeros_like(chemicalPotentials)
    sWaveTc = np.zeros_like(chemicalPotentials)

    for dWave in [True, False]:
        if dWave:
            basis = np.cos(kX) - np.cos(kY)
        else:
            basis = np.cos(kX) + np.cos(kY)
        basisSquared = basis ** 2
        # The natural normalization of the basis in BdG is over the
        # entire first Brillouin zone.
        basisSquared /= np.mean(basisSquared)

        # For convenience as it is a constant.
        potentialTerm = 2.0 / parameters["v"]

        for i, mu in enumerate(chemicalPotentials):
            _parameters = {
                "t0": parameters["t0"],
                "t1": parameters["t1"],
                "t2": parameters["t2"],
                "mu": mu,
            }

            # Evaluate the dispersion at every lattice point.
            dispersion = evalDispersion(_parameters, kX, kY)
            absDispersion = np.abs(dispersion)

            # This array does not change when iterating so lets pre-compute it.
            Yk2_ek = basisSquared / absDispersion

            # In order to enforce that Tc >= 0 we parameterize Tc as
            # Tc = sotfplus(x) = log(1 + e^x)
            # and solve for x, which is in the range (-\infty, +\infty).
            x = 0.0

            # The main loop.
            isConverged = False
            for n in range(numIterations):
                # Compute Tc.
                Tc = softplus(x)

                # If Tc is very small we consider ourselves done.
                isConverged = checkConvergence(Tc, convergenceCriterion)
                if isConverged:
                    break

                # The argument of the tanh factor.
                arg = absDispersion / (2.0 * Tc)
                tanhArg = np.tanh(arg)
                # This is the function we want find a zero of.
                f = np.mean(Yk2_ek * tanhArg) - potentialTerm

                # Newton's method requires the derivative of the function.
                # First, the derivative of tanh is easy:
                # \partial_x tanh(x) = 1 - tanh(x)^2
                # Second, the derivative of the argument:
                # \partial_x arg = \partial_x |ek| / (2 * Tc(x)) =
                # -|ek| / (2 * Tc(x)^2) * \partial_x (Tc(x)) =
                # -|ek| / (2 * Tc(x)^2) * sigmoid(x) =
                # -arg / Tc(x) * sigmoid(x)
                # So the derivative of the function is the following.
                argDerivative = -arg / Tc * sigmoid(x)
                dfdx = np.mean(Yk2_ek * (1.0 - tanhArg ** 2) * argDerivative)

                # If the derivative is very small we are done.
                isConverged = checkConvergence(dfdx, convergenceCriterion)
                if isConverged:
                    break

                # The Newton update.
                update = -f / dfdx

                # If the update is very small we are also done.
                isConverged = checkConvergence(update, convergenceCriterion)
                if isConverged:
                    break

                # Update x (our parameterization of Tc).
                x += update

            assert isConverged

            if dWave:
                dWaveTc[i] = Tc
            else:
                sWaveTc[i] = Tc

    crititalTemperatureDict = {
        "chemicalPotential": chemicalPotentials.tolist(),
        "dWaveTc": dWaveTc.tolist(),
        "sWaveTc": sWaveTc.tolist(),
    }

    stop = time.time()
    print("-- Done! - Time %f [s]" % (stop - start))
    return crititalTemperatureDict


# Compute Fermi momenta, Fermi velocities, and integrand weights.
def computeFermiSurface(parameters, numAngles, numIterations, convergenceCriterion):
    print("-- Computing the Fermi surface...")
    start = time.time()

    # Is the Fermi surface centered around (0, 0)?
    offset = computeFermiSurfaceOffset(parameters)

    minAngle = 0.0
    maxAngle = 2.0 * np.pi * (1.0 - 1.0 / numAngles)
    angles = np.linspace(minAngle, maxAngle, numAngles)
    kX, kY = computeFermiMomentum(
        parameters, angles, offset, numIterations, convergenceCriterion
    )

    # Compute the Fermi velocities.
    vX, vY = computeDispersionDerivative(parameters, kX, kY)

    # Compute integrand weights.
    weights = computeIntegrandWeights(parameters, kX, kY, vX, vY, offset)

    fermiSurfaceDict = {
        "kx": kX.tolist(),
        "ky": kY.tolist(),
        "vx": vX.tolist(),
        "vy": vY.tolist(),
        "weights": weights.tolist(),
    }

    stop = time.time()
    print("-- Done! - Time %f [s]" % (stop - start))

    return fermiSurfaceDict


# Helper function to compute Fermi surface average.
def computeFermiSurfaceAverage(x, weights):
    return np.sum(x * weights)


# Print some values relevant for the BdG comparison paper.
def printPaperStuff(fermisurface):
    kx = np.array(fermisurface["kx"])
    ky = np.array(fermisurface["ky"])
    vx = np.array(fermisurface["vx"])
    vy = np.array(fermisurface["vy"])
    weights = np.array(fermisurface["weights"])

    # Some stuff to compute averages of.
    fermiSpeed = np.hypot(vx, vy)
    fermiMomentum = np.hypot(kx, ky)
    invFermiMomentum = 1.0 / fermiMomentum

    # Compute Fermi surface averages.
    avgFermiSpeed = computeFermiSurfaceAverage(fermiSpeed, weights)
    avgFermiMomentum = computeFermiSurfaceAverage(fermiMomentum, weights)
    avgInvFermiMomentum = computeFermiSurfaceAverage(invFermiMomentum, weights)

    # Print results (and the means).
    print("<vf> = %.8f - Mean(vf) = %.8f" % (avgFermiSpeed, np.mean(fermiSpeed)))
    print("<kf> = %.8f - Mean(kf) = %.8f" % (avgFermiMomentum, np.mean(fermiMomentum)))
    print(
        "1/<kf> = %.8f - 1/Mean(kf) = %.8f"
        % (1.0 / avgFermiMomentum, 1.0 / np.mean(fermiMomentum))
    )
    print(
        "<1/kf> = %.8f - Mean(1/kf) = %.8f"
        % (avgInvFermiMomentum, np.mean(invFermiMomentum))
    )


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="Compute the Fermi surface for a tight-binding model on a square lattice."
    )
    parser.add_argument(
        "-name",
        "--name",
        type=str,
        default=None,
        help="Name of file to save. "
        "If not passed the file will not be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-mu",
        "--chemical-potential",
        type=float,
        default=1.0,
        help="The chemical potential, "
        "in terms of the nearest-neighbour hopping. (default: %(default)s)",
    )
    parser.add_argument(
        "-t1",
        "--hopping-nnn",
        type=float,
        default=0.0,
        help="The next-nearest neighbour hopping, "
        "in terms of the nearest-neighbour hopping.  (default: %(default)s)",
    )
    parser.add_argument(
        "-t2",
        "--hopping-nnnn",
        type=float,
        default=0.0,
        help="The next-next-nearest neighbour hopping, "
        "in terms of the nearest-neighbour hopping.  (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--interaction-potential",
        type=float,
        default=1.0,
        help="The interaction potential, "
        "in terms of the nearest-neighbour hopping. (default: %(default)s)",
    )
    parser.add_argument(
        "-nd",
        "--num-dispersion-k",
        type=int,
        default=101,
        help="Number of momenta per axis of the dispersion. (default: %(default)s)",
    )
    parser.add_argument(
        "-nk",
        "--num-dos-k",
        type=int,
        default=2001,
        help="Number of momenta per axis of the DoS. (default: %(default)s)",
    )
    parser.add_argument(
        "-ne",
        "--num-energies",
        type=int,
        default=1000,
        help="Number of energies in the DoS. (default: %(default)s)",
    )
    parser.add_argument(
        "-na",
        "--num-angles",
        type=int,
        default=50,
        help="Number of Fermi velocity angles. (default: %(default)s)",
    )
    parser.add_argument(
        "-ni",
        "--num-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations. (default: %(default)s)",
    )
    parser.add_argument(
        "-cc",
        "--convergence-criterion",
        type=float,
        default=1e-6,
        help="Convergence criterion. (default: %(default)s)",
    )
    parser.add_argument(
        "-em",
        "--energy-margin",
        type=float,
        default=0.01,
        help="Energy margin when computing the DOS. (default: %(default)s)",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.01,
        help="Value of Gamma in the Lorentzian (Used for the DOS). (default: %(default)s)",
    )
    args = parser.parse_args()

    # Set tight-binding parameters.
    parameters = {
        "t0": 1.0,
        "t1": args.hopping_nnn,
        "t2": args.hopping_nnnn,
        "mu": args.chemical_potential,
        "v": args.interaction_potential,
    }
    # Just print the tight-binding parameters.
    print(parameters)

    # ==================== DOS ==================== #
    dos = computeDensityOfStates(
        parameters=parameters,
        numMomenta=args.num_dos_k,
        numEnergies=args.num_energies,
        margin=args.energy_margin,
        gamma=args.gamma,
    )

    # ==================== TC ==================== #
    Tc = computeCriticalTemperature(
        parameters=parameters,
        numMomenta=args.num_dos_k,
        numChemicalPotentials=100,
        numIterations=args.num_iterations,
        convergenceCriterion=args.convergence_criterion,
    )

    # ==================== FERMI SURFACE ==================== #
    fermisurface = computeFermiSurface(
        parameters=parameters,
        numAngles=args.num_angles,
        numIterations=args.num_iterations,
        convergenceCriterion=args.convergence_criterion,
    )

    # ==================== DISPERSION ==================== #
    dispersion = computeDispersion(
        parameters=parameters, numMomenta=args.num_dispersion_k
    )

    # Just to compare with Micke.
    printPaperStuff(fermisurface)

    # Save file.
    if args.name is not None:
        data = {
            "criticalTemperature": Tc,
            "dispersion": dispersion,
            "fermisurface": fermisurface,
            "dos": dos,
            "parameters": parameters,
        }

        filename = args.name
        suffix = ".json"
        if not filename.endswith(suffix):
            filename += ".json"

        file = open(filename, "w")
        json.dump(data, file)
        print("-- Saved", filename)


if __name__ == "__main__":
    main()
