#ifndef ORDER_PARAMETER_BULK_H_
#define ORDER_PARAMETER_BULK_H_

#include "defines.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace internal {
/// \brief Compute the bulk value of an s-wave order parameter
///
/// \param inTemperature - The relative temperature T / Tc
/// \param inConvergenceCriterion - The relative change of the order parameter
///                                 at which we stop iterating
/// \param inNumIterations - The maximum number of iterations
/// \param inEnergyCutoff - The maximum energy in the Matsubara sum
///
/// \return - The order parameter
template <typename T>
T computeOrderParameterBulkSWave(
    const T inTemperature,
    const T inConvergenceCriterion = static_cast<T>(1e-8),
    const int inNumIterations = 1000,
    const T inEnergyCutoff = static_cast<T>(1e3)) {
  // If the temperature is above Tc then the sample is not superconducting, and
  // the order parameter is zero.
  if (inTemperature > static_cast<T>(1.0)) {
    return static_cast<T>(0.0);
  }

  // For convenience.
  const T twoPiT = static_cast<T>(2.0 * M_PI) * inTemperature;

  // How many energies do we need, given this cutoff and temperature?
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / twoPiT - static_cast<T>(0.5));

  // A vector of Matsubara energies.
  std::vector<T> energies;

  // Pre-compute the energies and the normalization factor.
  T energySum = static_cast<T>(0.0);
  for (int i = 0; i <= numEnergies; i++) {
    // Note, we are iterating from the largest energy to the smallest.
    // This way we are adding larger values as the partial sum increases,
    // which should be numerically preferable. Doing it the other way we would
    // be adding smaller values as the partial sum increases.
    const int n = numEnergies - i;
    const T energy = twoPiT * (static_cast<T>(n) + static_cast<T>(0.5));

    // Add the energy to the vector of energies.
    energies.push_back(energy);

    // Update the sum.
    energySum += static_cast<T>(1.0) / energy;
  }
  const T normalization =
      twoPiT / (std::log(inTemperature) + twoPiT * energySum);

  // Initial guess of the order parameter.
  T orderParameter = static_cast<T>(qcs::S_WAVE_BULK) *
                     std::sqrt(static_cast<T>(1.0) - inTemperature);

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inNumIterations; ++iterationIdx) {
    // Compute sum(1 / sqrt(en^2 + delta^2)).
    T integrand = static_cast<T>(0.0);
    for (const T &energy : energies) {
      // Note that hypot(x, y) = sqrt(x^2 + y^2) but numerically more stable.
      integrand += static_cast<T>(1.0) / std::hypot(orderParameter, energy);
    }

    // TODO(niclas): If we for some reason find this function too slow, we
    // should take Newton steps (except for the first one), and/or use Ozaki
    // energies.

    // The factor to update the order parameter with.
    const T updateFactor = integrand * normalization;

    // Update the order parameter.
    orderParameter *= updateFactor;

    // Are we done?
    isConverged =
        std::abs(updateFactor - static_cast<T>(1.0)) < inConvergenceCriterion;
    if (isConverged) {
      break;
    }
  }

  // TODO(niclas): Raise exception instead!?
  if (!isConverged) {
    std::cout << "WARNING! The bulk order parameter did NOT converge!"
              << std::endl;
  }

  return orderParameter;
}
} // namespace internal

#endif // ORDER_PARAMETER_BULK_H_
