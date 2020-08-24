//===-------- bdg_utils_test.cu - Unit tests of BdG related functions -----===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file bdg_utils_test.cu
/// \brief Different tests related to the BdG theory of superconductivity.
///
//===----------------------------------------------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "OrderParameterBasis.h"
#include "bdg_utils.h"
#include "defines.h"
#include "dispersion.h"
#include "utils.h"
#include <cmath>
#include <doctest/doctest.h>
#include <limits>
#include <vector>

namespace helper {
/// \brief Test the critical temperature within BdG.
///
/// Compare it with the results in the BdG paper:
/// https://arxiv.org/abs/2006.00456
///
/// \param InUseFermiSurface1 - Whether to use Fermi surface 1 as defined in the
///                             paper.
/// \param inTolerance - The relative tolerance. It is high because the results
///                      in the paper does not have that many significant
///                      digits.
///
/// \return - Void.
template <typename T>
void testComputeCriticalTemperature(
    const bool InUseFermiSurface1, const T inTolerance = static_cast<T>(1e-3)) {
  // =================== ARRANGE ==================== //
  T hoppingNNN;
  T hoppingNNNN;
  T chemicalPotential;
  T potential;
  T expectedDWaveTc;
  T expectedSExtWaveTc;

  if (InUseFermiSurface1) {
    // Dispersion parameters.
    hoppingNNN = -static_cast<T>(0.25);
    hoppingNNNN = static_cast<T>(0.0);
    chemicalPotential = static_cast<T>(0.0);
    potential = static_cast<T>(1.4);

    // Expeted critical temperatures.
    expectedDWaveTc = static_cast<T>(0.118);
    expectedSExtWaveTc = static_cast<T>(0.0);
  } else {
    // Dispersion parameters.
    hoppingNNN = -static_cast<T>(0.495);
    hoppingNNNN = static_cast<T>(0.156);
    chemicalPotential = -static_cast<T>(1.267);
    potential = static_cast<T>(1.279);
    expectedDWaveTc = static_cast<T>(0.114);
    expectedSExtWaveTc = static_cast<T>(0.0);
  }

  // Define the dispersion.
  scg::DispersionTightBinding<T> dispersion(hoppingNNN, hoppingNNNN,
                                            chemicalPotential);

  // D-wave and extended s-wave.
  const scg::BasisFunctionSymmetry dWaveSymmetry =
      scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
  const scg::BasisFunctionSymmetry sExtSymmetry =
      scg::BasisFunctionSymmetry::SWAVE_EXT;

  // =================== ACT ==================== //
  const std::pair<bool, T> dWaveTmp = scg::bdg::computeCriticalTemperature(
      dispersion, potential, dWaveSymmetry);
  const bool dWaveSuccess = std::get<0>(dWaveTmp);
  const T dWaveTc = std::get<1>(dWaveTmp);

  const std::pair<bool, T> sExtWaveTmp =
      scg::bdg::computeCriticalTemperature(dispersion, potential, sExtSymmetry);
  const bool sExtWaveSuccess = std::get<0>(sExtWaveTmp);
  const T sExtWaveTc = std::get<1>(sExtWaveTmp);

  // =================== ASSERT ==================== //
  CHECK(dWaveSuccess);
  CHECK(dWaveTc == doctest::Approx(expectedDWaveTc).epsilon(inTolerance));
  CHECK(sExtWaveSuccess);
  CHECK(sExtWaveTc == doctest::Approx(expectedSExtWaveTc).epsilon(inTolerance));
}
} // namespace helper

// ============== Test computeCriticalTemperature ============== //

TEST_CASE("Test computeCriticalTemperature<float> - FS1 -") {
  const bool useFermiSurface1 = true;
  helper::testComputeCriticalTemperature<float>(useFermiSurface1);
}

TEST_CASE("Test computeCriticalTemperature<float><float> - FS2") {
  const bool useFermiSurface1 = false;
  helper::testComputeCriticalTemperature<float>(useFermiSurface1);
}

TEST_CASE("Test computeCriticalTemperature<double> - FS1") {
  const bool useFermiSurface1 = true;
  helper::testComputeCriticalTemperature<double>(useFermiSurface1);
}

TEST_CASE("Test computeCriticalTemperature<double> - FS2") {
  const bool useFermiSurface1 = false;
  helper::testComputeCriticalTemperature<double>(useFermiSurface1);
}
