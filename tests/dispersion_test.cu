//===--------- dispersion_test.cu - Unit tests of the dispersion  --------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Testing the dispersion, e.g. its square symmetry, analytic vs numeric
/// derivative.
///
//===----------------------------------------------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "defines.h"
#include "dispersion.h"
#include "doctest_wrapper.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace helper {
// Enum for the the different symmetries to test in testSquareSymmetry.
enum class Symmetry { mirrorX, mirrorY, rotate180, mirrorDiag };

// Apply some (square) symmetry operation to the momentum.
template <typename T>
conga::vector2d<T> applySymmetry(const conga::vector2d<T> &inMomentum,
                                 const Symmetry inSymmetry) {
  T kX = inMomentum[0];
  T kY = inMomentum[1];
  switch (inSymmetry) {
  case Symmetry::mirrorX: {
    kY = -kY;
    break;
  }
  case Symmetry::mirrorY: {
    kX = -kX;
    break;
  }
  case Symmetry::rotate180: {
    kX = -kX;
    kY = -kY;
    break;
  }
  case Symmetry::mirrorDiag: {
    const T tmp = kX;
    kX = kY;
    kY = tmp;
    break;
  }
  default:
    std::cout << "Error! Symmetry not implemented." << std::endl;
  }
  return {kX, kY};
}

// For checking the results in testSquareSymmetry.
template <typename T>
void checkResult(const T inExpected, const T inComputed, const T inTolerance,
                 const Symmetry inSymmetry) {
  // Unused.
  (void)inSymmetry;
  // This is a scalar so inSymmetry is unused.
  REQUIRE(inExpected == doctest::Approx(inComputed).epsilon(inTolerance));
}

// For checking the results in testSquareSymmetry.
template <typename T>
void checkResult(const conga::vector2d<T> &inExpected,
                 const conga::vector2d<T> &inComputed, const T inTolerance,
                 const Symmetry inSymmetry) {
  // The symmetries considered are involutory, i.e. the operation is its own
  // inverse.
  const conga::vector2d<T> computed = applySymmetry(inComputed, inSymmetry);

  REQUIRE(inExpected[0] == doctest::Approx(computed[0]).epsilon(inTolerance));
  REQUIRE(inExpected[1] == doctest::Approx(computed[1]).epsilon(inTolerance));
}

// Test that the dispersion has the symmetries of a square.
template <typename T>
void testSquareSymmetry(const T inTolerance,
                        const int inNumTestsPerParameter = 5) {
  // =================== ARRANGE ==================== //
  // Next-nearest neighbour.
  const T minHoppingNNN = static_cast<T>(-0.5);
  const T maxHoppingNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNN = conga::utils::linspace(
      minHoppingNNN, maxHoppingNNN, inNumTestsPerParameter);

  // Next-next-earest neighbour.
  const T minHoppingNNNN = static_cast<T>(-0.5);
  const T maxHoppingNNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNNN = conga::utils::linspace(
      minHoppingNNNN, maxHoppingNNNN, inNumTestsPerParameter);

  // Chemical potential.
  const T minChemicalPotential = static_cast<T>(-1.0);
  const T maxChemicalPotential = static_cast<T>(1.0);
  const std::vector<T> a_chemicalPotential = conga::utils::linspace(
      minChemicalPotential, maxChemicalPotential, inNumTestsPerParameter);

  // X-component of momentum.
  const T minKX = static_cast<T>(0.0);
  const T maxKX = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kX =
      conga::utils::linspace(minKX, maxKX, inNumTestsPerParameter);

  // Y-component of momentum.
  const T minKY = static_cast<T>(0.0);
  const T maxKY = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kY =
      conga::utils::linspace(minKY, maxKY, inNumTestsPerParameter);

  const int numSymmetries = 4;
  const std::array<Symmetry, numSymmetries> symmetries = {
      Symmetry::mirrorX, Symmetry::mirrorY, Symmetry::rotate180,
      Symmetry::mirrorDiag};

  for (const T &hoppingNNN : a_hoppingNNN) {
    for (const T &hoppingNNNN : a_hoppingNNNN) {
      for (const T &chemicalPotential : a_chemicalPotential) {
        // Create the dispersion object.
        const conga::DispersionTightBinding<T> dispersion(
            hoppingNNN, hoppingNNNN, chemicalPotential);
        for (const T &kX : a_kX) {
          for (const T &kY : a_kY) {
            const conga::vector2d<T> momentum = {kX, kY};
            const T valueExpected = dispersion.evaluate(momentum);
            const conga::vector2d<T> gradientExpected =
                dispersion.derivative(momentum);

            for (const Symmetry &symmetry : symmetries) {
              const conga::vector2d<T> momentumSym =
                  applySymmetry(momentum, symmetry);

              // =================== ACT ==================== //
              const T value = dispersion.evaluate(momentumSym);
              const conga::vector2d<T> gradient =
                  dispersion.derivative(momentumSym);

              // =================== ASSERT ==================== //
              checkResult(valueExpected, value, inTolerance, symmetry);
              checkResult(gradientExpected, gradient, inTolerance, symmetry);
            }
          }
        }
      }
    }
  }
}

// Test that the analytical derivative of the dispersion is close to the
// numerical derivative.
template <typename T>
void testDispersionDerivative(
    const T inTolerance, const int inNumTestsPerParameter = 5,
    const T inKEpsilon = static_cast<T>(1e3) *
                         std::numeric_limits<T>::epsilon()) {
  // =================== ARRANGE ==================== //
  // Next-nearest neighbour.
  const T minHoppingNNN = static_cast<T>(-0.5);
  const T maxHoppingNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNN = conga::utils::linspace(
      minHoppingNNN, maxHoppingNNN, inNumTestsPerParameter);

  // Next-next-earest neighbour.
  const T minHoppingNNNN = static_cast<T>(-0.5);
  const T maxHoppingNNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNNN = conga::utils::linspace(
      minHoppingNNNN, maxHoppingNNNN, inNumTestsPerParameter);

  // Chemical potential.
  const T minChemicalPotential = static_cast<T>(-1.0);
  const T maxChemicalPotential = static_cast<T>(1.0);
  const std::vector<T> a_chemicalPotential = conga::utils::linspace(
      minChemicalPotential, maxChemicalPotential, inNumTestsPerParameter);

  // X-component of momentum.
  const T minKX = static_cast<T>(0.0);
  const T maxKX = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kX =
      conga::utils::linspace(minKX, maxKX, inNumTestsPerParameter);

  // Y-component of momentum.
  const T minKY = static_cast<T>(0.0);
  const T maxKY = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kY =
      conga::utils::linspace(minKY, maxKY, inNumTestsPerParameter);

  // For convenience.
  const T invDenom = static_cast<T>(1.0) / (static_cast<T>(2.0) * inKEpsilon);

  for (const T &hoppingNNN : a_hoppingNNN) {
    for (const T &hoppingNNNN : a_hoppingNNNN) {
      for (const T &chemicalPotential : a_chemicalPotential) {
        // Create the dispersion object.
        const conga::DispersionTightBinding<T> dispersion(
            hoppingNNN, hoppingNNNN, chemicalPotential);
        for (const T &kX : a_kX) {
          for (const T &kY : a_kY) {
            // =================== ACT ==================== //
            // Compute the analytical derivatives.
            const conga::vector2d<T> dE = dispersion.derivative({kX, kY});
            const T dEdX = dE[0];
            const T dEdY = dE[1];

            // =================== ASSERT ==================== //
            // Compute the numerical derivatives.
            const T EXPlus = dispersion.evaluate({kX + inKEpsilon, kY});
            const T EXMinus = dispersion.evaluate({kX - inKEpsilon, kY});
            const T EYPlus = dispersion.evaluate({kX, kY + inKEpsilon});
            const T EYMinus = dispersion.evaluate({kX, kY - inKEpsilon});

            const T dEdXApprox = (EXPlus - EXMinus) * invDenom;
            const T dEdYApprox = (EYPlus - EYMinus) * invDenom;

            REQUIRE(dEdX == doctest::Approx(dEdXApprox).epsilon(inTolerance));
            REQUIRE(dEdY == doctest::Approx(dEdYApprox).epsilon(inTolerance));
          }
        }
      }
    }
  }
}
} // namespace helper

// =============== Test computeDispersion Symmetry =============== //

TEST_CASE("Test Dispersion Symmetry <float>") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testSquareSymmetry(tolerance);
}

TEST_CASE("Test Dispersion Symmetry <double>") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSquareSymmetry(tolerance);
}

// =============== Test computeDispersionDerivative Symmetry =============== //

TEST_CASE("Test Dispersion Derivative Symmetry <float>") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testSquareSymmetry(tolerance);
}

TEST_CASE("Test Dispersion Derivative Symmetry <double>") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSquareSymmetry(tolerance);
}

// ============ Test computeDispersionDerivative Numerically ============ //

TEST_CASE("Test Dispersion Derivative <float>") {
  // Relative tolerance.
  const float tolerance = 1e-3F;
  helper::testDispersionDerivative(tolerance);
}

TEST_CASE("Test Dispersion Derivative <double>") {
  // Relative tolerance.
  const double tolerance = 1e-3;
  helper::testDispersionDerivative(tolerance);
}
