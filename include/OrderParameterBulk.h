#ifndef ORDER_PARAMETER_BULK_H_
#define ORDER_PARAMETER_BULK_H_

#include "ComputeResidual.h"
#include "defines.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace bulk_internal {
/// \brief A helper function for updating the order parameter with a fixed-point
/// step
///
/// \param inOrderParameter - The current order parameter
/// \param inMatsubaraEnergies - An array of Matsubara energies
/// \param inNormalization - The normalization factor
///
/// \return - The new order parameter
template <typename T>
T updateOrderParameter(const T inOrderParameter,
                       const std::vector<T> &inMatsubaraEnergies,
                       const T inNormalization) {
  // Compute sum(1 / sqrt(en^2 + delta^2)).
  T integrand = static_cast<T>(0.0);
  for (const T &matsubaraEnergy : inMatsubaraEnergies) {
    // Note that hypot(x, y) = sqrt(x^2 + y^2) but numerically more stable.
    integrand +=
        static_cast<T>(1.0) / std::hypot(inOrderParameter, matsubaraEnergy);
  }
  return inOrderParameter * integrand * inNormalization;
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
    const int inMaxIterations = static_cast<int>(1e5),
    const T inEnergyCutoff = static_cast<T>(1e3)) {
  // If the temperature is above Tc then the sample is not superconducting, and
  // the order parameter is zero.
  if (inTemperature > static_cast<T>(1.0)) {
    return static_cast<T>(0.0);
  }

  // For some reason the Aitken method is not very good close to the critical
  // temperature.
  const bool useAitken = inTemperature < static_cast<T>(0.995);

  // For convenience.
  const T twoPiTemp = static_cast<T>(2.0 * M_PI) * inTemperature;

  // How many energies do we need, given this cutoff and temperature?
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / twoPiTemp - static_cast<T>(0.5));

  // A vector of Matsubara energies.
  std::vector<T> energies;

  // Pre-compute the energies and the normalization factor.
  T energySum = static_cast<T>(0.0);

  // Note, we are iterating from the largest energy to the smallest.
  // This way we are adding larger values as the partial sum increases,
  // which should be numerically preferable. Doing it the other way we would
  // be adding smaller values as the partial sum increases.
  // TODO(niclas): Use Ozaki energies.
  for (int n = numEnergies; n >= 0; --n) {
    const T matsubaraEnergy =
        twoPiTemp * (static_cast<T>(n) + static_cast<T>(0.5));

    // Add the energy to the vector of energies.
    energies.push_back(matsubaraEnergy);

    // Update the sum.
    energySum += static_cast<T>(1.0) / matsubaraEnergy;
  }
  const T normalization =
      static_cast<T>(1.0) / (std::log(inTemperature) / twoPiTemp + energySum);

  // Initial guess of the order parameter.
  // It converges faster starting with sqrt(1 - T^3).
  T orderParameter = static_cast<T>(qcs::S_WAVE_BULK) *
                     std::sqrt(static_cast<T>(1.0) -
                               inTemperature * inTemperature * inTemperature);

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inMaxIterations; ++iterationIdx) {
    // First step.
    const T orderParameterKPlusOne = bulk_internal::updateOrderParameter(
        orderParameter, energies, normalization);
    const T residualKPlusOne =
        computeResidual(orderParameter, orderParameterKPlusOne);
    isConverged = residualKPlusOne < inConvergenceCriterion;
    if (isConverged) {
      orderParameter = orderParameterKPlusOne;
      break;
    }

    // Second step.
    const T orderParameterKPlusTwo = bulk_internal::updateOrderParameter(
        orderParameterKPlusOne, energies, normalization);
    const T residualKPlusTwo =
        computeResidual(orderParameterKPlusOne, orderParameterKPlusTwo);
    isConverged = residualKPlusTwo < inConvergenceCriterion;
    if (isConverged) {
      orderParameter = orderParameterKPlusTwo;
      break;
    }

    // Update order parameter.
    if (useAitken) {
      const T diff10 = orderParameterKPlusOne - orderParameter;
      const T diff21 = orderParameterKPlusTwo - orderParameterKPlusOne;
      const T denom = diff21 - diff10;
      if (std::abs(denom) > std::numeric_limits<T>::epsilon()) {
        // Take an Aitken step.
        // https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process
        orderParameter = orderParameterKPlusTwo - diff21 * diff21 / denom;
      } else {
        // Take a normal fixed-point step.
        orderParameter = orderParameterKPlusTwo;
      }
    } else {
      // Take a rolling ball step.
      // Note, this is only used very close to the critical temperature. Hence,
      // the large weight = temperature.
      orderParameter =
          orderParameterKPlusTwo +
          inTemperature * (orderParameterKPlusOne - orderParameter);
    }
  }

  // TODO(niclas): Raise exception instead!?
  if (!isConverged) {
    std::cout << "WARNING! The bulk order parameter did NOT converge!"
              << std::endl;
  }

  return orderParameter;
}

template <typename T>
T computeFreeEnergyBulkSWave(const T inTemperature, const T inOrderParameter,
                             const T inEnergyCutoff = static_cast<T>(1e3)) {
  // For convenience.
  const T twoPiTemp = static_cast<T>(2.0 * M_PI) * inTemperature;

  // How many energies do we need, given this cutoff and temperature?
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / twoPiTemp - static_cast<T>(0.5));

  T energySum = static_cast<T>(0.0);
  for (int n = numEnergies; n >= 0; --n) {
    const T matsubaraEnergy =
        twoPiTemp * (static_cast<T>(n) + static_cast<T>(0.5));
    const T Omega = std::hypot(inOrderParameter, matsubaraEnergy);
    // The second term is equal to i (delta^* gamma - delta gammaTilde).
    energySum += (static_cast<T>(1.0) / matsubaraEnergy -
                  static_cast<T>(2.0) / (matsubaraEnergy + Omega));
  }
  const T freeEnergy = inOrderParameter * inOrderParameter *
                       (std::log(inTemperature) + twoPiTemp * energySum);
  return freeEnergy;
}

#endif // ORDER_PARAMETER_BULK_H_
