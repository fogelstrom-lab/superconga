//===-- OrderParameterBulk.h - Order parameter bulk calculation. ----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functionality related to computing a bulk order parameter.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_ORDER_PARAMETER_BULK_H_
#define CONGA_ORDER_PARAMETER_BULK_H_

#include "../defines.h"
#include "../utils.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace conga {
namespace bulk_internal {
/// \brief A helper function for computing the residual of the self-consistency
/// equation
///
/// \param inOrderParameter - The order parameter
/// \param inTemperature - The relative temperature T / Tc
/// \param inNumEnergies - The number of Matsubara energies
/// \param inCouplingConstant - The coupling constant
///
/// \return - The residual
template <typename T>
T computeResidual(const T inOrderParameter, const T inTemperature,
                  const int inNumEnergies, const T inCouplingConstant) {
  // Compute sum(1 / sqrt(en^2 + delta^2)).
  T integrand = static_cast<T>(0.0);
  for (int n = inNumEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        inTemperature * (static_cast<T>(n) + static_cast<T>(0.5));
    integrand +=
        static_cast<T>(1.0) / std::hypot(inOrderParameter, matsubaraEnergy);
  }
  return inOrderParameter *
         (integrand * inCouplingConstant - static_cast<T>(1.0));
}

/// \brief A helper function for computing the coupling constant of the
/// self-consistency equation
///
/// \param inTemperature - The relative temperature T / Tc
/// \param inNumEnergies - The number of Matsubara energies
///
/// \return - The coupling constant
template <typename T>
T computeCouplingConstant(const T inTemperature, const int inNumEnergies) {
  // Note, we are iterating from the largest energy to the smallest.
  // This way we are adding larger values as the partial sum increases,
  // which should be numerically preferable. Doing it the other way we would
  // be adding smaller values as the partial sum increases.
  // TODO(niclas): Use Ozaki energies.
  T energySum = static_cast<T>(0.0);
  for (int n = inNumEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        inTemperature * (static_cast<T>(n) + static_cast<T>(0.5));
    energySum += static_cast<T>(1.0) / matsubaraEnergy;
  }

  const T couplingConstant =
      static_cast<T>(1.0) /
      (std::log(inTemperature) / inTemperature + energySum);

  return couplingConstant;
}
} // namespace bulk_internal

/// \brief Compute the bulk value of an s-wave order parameter
///
/// \param inTemperature - The relative temperature T / Tc
/// \param inConvergenceCriterion - The relative change of the order parameter
///                                 at which we stop iterating
/// \param inMaxIterations - The maximum number of iterations
/// \param inEnergyCutoff - The maximum energy in the Matsubara sum
///
/// \return - The order parameter
template <typename T>
T computeOrderParameterBulkSWave(
    const T inTemperature,
    const T inConvergenceCriterion = static_cast<T>(10.0) *
                                     std::numeric_limits<T>::epsilon(),
    const int inMaxIterations = 10,
    const T inEnergyCutoff = static_cast<T>(100)) {
  // Check if the temperature is outside of the (0, 1), and return known values
  // in that case.
  if (inTemperature >= static_cast<T>(1.0)) {
    // If the temperature is above Tc then the sample is not superconducting,
    // and the order parameter is zero.
    return static_cast<T>(0.0);
  } else if (inTemperature < std::numeric_limits<T>::epsilon()) {
    // If the temperature is approximately zero then return the known
    // zero-temperature order-parameter value.
    return static_cast<T>(constants::S_WAVE_BULK);
  }

  // Comes from: cutoff = 2 * pi * T * (n + 0.5).
  // But the n is the largest n we need to include, so the number of energies is
  // one more, hence the plus one.
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / inTemperature - static_cast<T>(0.5)) +
      1;

  const T couplingConstant =
      bulk_internal::computeCouplingConstant(inTemperature, numEnergies);

  // Initial guess of the order parameter.
  T orderParameter = static_cast<T>(constants::S_WAVE_BULK) *
                     std::sqrt(static_cast<T>(1.0) -
                               inTemperature * inTemperature * inTemperature);

  // The algorithm requires two evaluated order-parameter guesses.
  vector2d<T> opVec{orderParameter, std::sqrt(orderParameter)};

  // Compute the residuals for the initial guesses.
  vector2d<T> resVec;
  for (std::size_t i = 0; i < opVec.size(); ++i) {
    resVec[i] = bulk_internal::computeResidual(opVec[i], inTemperature,
                                               numEnergies, couplingConstant);
  }

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inMaxIterations; ++iterationIdx) {
    // We want to solve the self-consistency equation
    //
    // 0 = x - f(x)
    //
    // where x is the order parameter, and f(x) is RHS of the gap equation. Lets
    // call the quantity
    //
    // r = x - f(x)
    //
    // the residual. The residual will be (close to) zero at self-concistency.
    //
    // The algorithm works by fitting a second-degree polynomial through two
    // points of the residual r0 = r(x0) and r1 = r(x1). I.e. the polynomial
    //
    // a * x^2 + b * x + c
    //
    // Only two points? Yes, because the zeroth order term, c, in the polynomial
    // is set to zero. The reason for this is because the solution, when letting
    // it vary, is close to zero anyway. So by setting it to zero we can get
    // away with only two points instead of three. And solving a system of
    // equation with two variables is really easy. We thus get a matrix equation
    //
    // | (x0)^2 , x0 | a | = | r0 |
    // | (x1)^2 , x1 | b |   | r1 |
    //
    // which has the analytical solution
    //
    // a = (r0 * x1 - r1 * x0) / (x0 * x1 * (x0 - x1))
    // b = (-r0 * (x1)^2 + r1 * (x0)^2) / (x0 * x1 * (x0 - x1))
    //
    // With a and b known, we can set the polynomial to zero and solve for x.
    // Because c = 0 the non-zero solution is
    //
    // x = - b / a
    //
    // The denominators of a and b cancel, leaving
    //
    // x = (r0 * (x1)^2 - r1 * (x0)^2) / (r0 * x1 - r1 * x0)
    //   = (1 - w) * x0 + w * x1
    //
    // where
    //
    // w = 1 / (1 - r1 * x0 / (r0 * x1))
    //
    // and x is the new order parameter value.
    // TODO(niclas): Perhaps do not set c = 0. I think this would be Muller's
    // method.

    // Update the order parameter.
    const T opRatio = opVec[0] / opVec[1];
    const T resRatio = resVec[0] / resVec[1];
    const T weight =
        static_cast<T>(1.0) / (static_cast<T>(1.0) - opRatio / resRatio);
    orderParameter =
        (static_cast<T>(1.0) - weight) * opVec[0] + weight * opVec[1];

    // Compute residual.
    const T residual = bulk_internal::computeResidual(
        orderParameter, inTemperature, numEnergies, couplingConstant);

    // Check convergence. If the order parameter is very small it makes more
    // sense to check the residual directly, instead of dividing by the order
    // parameter.
    const bool isSmall =
        orderParameter < static_cast<T>(10) * std::numeric_limits<T>::epsilon();
    const T relativeResidual =
        isSmall ? residual : std::abs(residual / orderParameter);
    isConverged = relativeResidual < inConvergenceCriterion;
    if (isConverged) {
      break;
    }

    // Get the index of the order parameter with the largest residual.
    const int idx = std::abs(resVec[0]) > std::abs(resVec[1]) ? 0 : 1;

    // Replace the element with the new order parameter and residual.
    opVec[idx] = orderParameter;
    resVec[idx] = residual;
  }

#ifndef NDEBUG
  // TODO(niclas): Raise exception instead!?
  if (!isConverged) {
    std::cerr << "-- ERROR! The bulk order parameter did NOT converge!"
              << std::endl;
  }
#endif

  return orderParameter;
}

template <typename T>
T computeFreeEnergyBulkSWave(const T inTemperature, const T inOrderParameter,
                             const T inEnergyCutoff = static_cast<T>(1e2)) {
  // Comes from: cutoff = 2 * pi * T * (n + 0.5).
  // But the n is the largest n we need to include, so the number of energies is
  // one more, hence the plus one.
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / inTemperature - static_cast<T>(0.5)) +
      1;

  T energySum = static_cast<T>(0.0);
  for (int n = numEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        inTemperature * (static_cast<T>(n) + static_cast<T>(0.5));
    const T Omega = std::hypot(inOrderParameter, matsubaraEnergy);
    // The second term is equal to i (delta^* gamma - delta gammaTilde).
    energySum += (static_cast<T>(1.0) / matsubaraEnergy -
                  static_cast<T>(2.0) / (matsubaraEnergy + Omega));
  }
  const T freeEnergy = inOrderParameter * inOrderParameter *
                       (std::log(inTemperature) + inTemperature * energySum);
  return freeEnergy;
}

/// \brief Compute the Yoshida function Y_{3/2} at a given temperature.
///
/// \param inTemperature The temperature [Tc].
/// \param inOrderParameter The bulk order-parameter [2*pi*kB*Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
///
/// \return Y_{3/2}(T).
template <typename T>
T computeYoshidaBulkSWave(const T inTemperature, const T inOrderParameter,
                          const T inEnergyCutoff) {
  if (inTemperature < std::numeric_limits<T>::epsilon()) {
    // If the temperature is approximately zero then return the known
    // zero-temperature value.
    return static_cast<T>(1);
  }

  const int numEnergies =
      static_cast<int>(inEnergyCutoff / inTemperature - static_cast<T>(0.5)) +
      1;

  T energySum = static_cast<T>(0);
  for (int n = numEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        inTemperature * (static_cast<T>(n) + static_cast<T>(0.5));
    const T Omega = std::hypot(inOrderParameter, matsubaraEnergy);
    energySum += static_cast<T>(1) / static_cast<T>(std::pow(Omega, 3));
  }

  return inTemperature * inOrderParameter * inOrderParameter * energySum;
}
} // namespace conga

#endif // CONGA_ORDER_PARAMETER_BULK_H_
