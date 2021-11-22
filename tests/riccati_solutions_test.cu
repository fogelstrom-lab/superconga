#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/riccati/RiccatiSolutions.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <thrust/complex.h>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace helper {
// Define the small and large numbers use for parameter sweeps.
// TODO(niclas): These are neither very large nor small.
const double SMALL = 1e-3;
const double LARGE = 1e3;

// Factor for the tolerance of most tests.
const double TOLERANCE_FACTOR = 10.0;

// Factor for the crappy general stepping test.
const double TOLERANCE_FACTOR_BAD = 1e3;

/// \brief The stepping expression expressed with std::complex.
///
/// \param inOrderParameter The order parameter, for tilde it would be -\Delta^*
/// instead.
/// \param inEnergy The energy.
/// \param inInitial The initial value.
/// \param inStep The step length.
///
/// \return The coherence function at the step.
template <typename T>
std::complex<T> computeStep(const std::complex<T> &inOrderParameter,
                            const std::complex<T> &inEnergy,
                            const std::complex<T> &inInitial, const T inStep) {
  const std::complex<T> Omega =
      std::sqrt(std::norm(inOrderParameter) - inEnergy * inEnergy);
  const std::complex<T> iOmega(-Omega.imag(), Omega.real());
  const std::complex<T> gHomo = -inOrderParameter / (inEnergy + iOmega);

  const std::complex<T> diff = inInitial - gHomo;
  const std::complex<T> ex = std::exp((-static_cast<T>(2) * inStep) * Omega);

  const std::complex<T> correction =
      static_cast<T>(2) * iOmega * diff * ex /
      (static_cast<T>(2) * iOmega +
       std::conj(inOrderParameter) * diff * (static_cast<T>(1) - ex));

  return gHomo + correction;
}

/// \brief Compare the homogenous Riccati solutions to values computed by hand.
///
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testHomogeneousSolutionsParticular(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // Note, do not change these values. The expected result is computed by hand
  // using them.
  const T energyImag = static_cast<T>(3);
  const T orderParameterReal = static_cast<T>(4);

  // =================== ACT ==================== //
  // Construct energy and order parameter.
  const thrust::complex<T> energy(static_cast<T>(0), energyImag);
  const thrust::complex<T> orderParameter(orderParameterReal,
                                          static_cast<T>(0));

  // Compute coherence functions.
  thrust::complex<T> testGamma;
  thrust::complex<T> testGammaTilde;
  conga::riccati::computeHomogeneous(orderParameter, energy, testGamma,
                                     testGammaTilde);

  // =================== ASSERT ==================== //
  // Compute expected coherence functions.
  const std::complex<T> expectedGamma(static_cast<T>(0), static_cast<T>(0.5));
  const std::complex<T> expectedGammaTilde = std::conj(expectedGamma);

  CHECK(testGamma.real() ==
        doctest::Approx(expectedGamma.real()).epsilon(inTolerance));
  CHECK(testGamma.imag() ==
        doctest::Approx(expectedGamma.imag()).epsilon(inTolerance));
  CHECK(testGammaTilde.real() ==
        doctest::Approx(expectedGammaTilde.real()).epsilon(inTolerance));
  CHECK(testGammaTilde.imag() ==
        doctest::Approx(expectedGammaTilde.imag()).epsilon(inTolerance));
}

/// \brief Compare the homogenous Riccati solutions with an std::complex
/// implementation.
///
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testHomogeneousSolutionsGeneral(const T inTolerance) {
  // =================== ARRANGE ==================== //
  const std::vector<T> values = {-static_cast<T>(LARGE), -static_cast<T>(1),
                                 -static_cast<T>(SMALL), static_cast<T>(0),
                                 static_cast<T>(SMALL),  static_cast<T>(1),
                                 static_cast<T>(LARGE)};
  const std::vector<T> valuesPositive = {
      static_cast<T>(SMALL), static_cast<T>(1), static_cast<T>(LARGE)};

  // =================== ACT ==================== //
  for (const T energyReal : values) {
    for (const T energyImag : valuesPositive) {
      // Construct energy.
      const std::complex<T> energy(energyReal, energyImag);

      for (const T orderParameterReal : values) {
        for (const T orderParameterImag : values) {
          // Construct order parameter.
          const std::complex<T> orderParameter(orderParameterReal,
                                               orderParameterImag);

          // Compute coherence functions.
          // For some reason the inputs can not be std::complex.
          thrust::complex<T> testGamma;
          thrust::complex<T> testGammaTilde;
          conga::riccati::computeHomogeneous(thrust::complex<T>(orderParameter),
                                             thrust::complex<T>(energy),
                                             testGamma, testGammaTilde);

          // =================== ASSERT ==================== //
          // Compute expected coherence functions.
          const std::complex<T> Omega =
              std::sqrt(std::norm(orderParameter) - energy * energy);
          const std::complex<T> iOmega(-Omega.imag(), Omega.real());
          const std::complex<T> expectedGamma =
              -orderParameter / (energy + iOmega);
          const std::complex<T> expectedGammaTilde =
              std::conj(orderParameter) / (energy + iOmega);

          CHECK(testGamma.real() ==
                doctest::Approx(expectedGamma.real()).epsilon(inTolerance));
          CHECK(testGamma.imag() ==
                doctest::Approx(expectedGamma.imag()).epsilon(inTolerance));
          CHECK(
              testGammaTilde.real() ==
              doctest::Approx(expectedGammaTilde.real()).epsilon(inTolerance));
          CHECK(
              testGammaTilde.imag() ==
              doctest::Approx(expectedGammaTilde.imag()).epsilon(inTolerance));
        }
      }
    }
  }
}

/// \brief Compare the step Riccati solutions to values computed by hand.
///
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testSteppingSolutionsParticular(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // Note, do not change these values. The expected result is computed by hand
  // using them.
  const T energyImag = static_cast<T>(0);
  const T orderParameterReal = static_cast<T>(1);
  const T step = static_cast<T>(0.5);
  const T initialFactor = static_cast<T>(2);

  // =================== ACT ==================== //
  // Construct energy and order parameter.
  const thrust::complex<T> energy(static_cast<T>(0), energyImag);
  const thrust::complex<T> orderParameter(orderParameterReal,
                                          static_cast<T>(0));

  // Compute homogeneous coherence functions.
  thrust::complex<T> gammaHomo;
  thrust::complex<T> gammaTildeHomo;
  conga::riccati::computeHomogeneous(orderParameter, energy, gammaHomo,
                                     gammaTildeHomo);

  // Set initial values.
  const thrust::complex<T> gammaInit = gammaHomo * initialFactor;
  const thrust::complex<T> gammaTildeInit = gammaTildeHomo * initialFactor;

  // Compute coherence functions.
  const thrust::complex<T> testGamma = conga::riccati::Gamma<T>::computeStep(
      gammaHomo, orderParameter, energy, gammaInit, step);
  const thrust::complex<T> testGammaTilde =
      conga::riccati::GammaTilde<T>::computeStep(gammaTildeHomo, orderParameter,
                                                 energy, gammaTildeInit, step);

  // =================== ASSERT ==================== //
  // Compute expected coherence functions.
  const std::complex<T> expectedGamma(static_cast<T>(0),
                                      static_cast<T>(3.0 * M_E + 1.0) /
                                          static_cast<T>(3.0 * M_E - 1.0));
  const std::complex<T> expectedGammaTilde = std::conj(expectedGamma);

  CHECK(testGamma.real() ==
        doctest::Approx(expectedGamma.real()).epsilon(inTolerance));
  CHECK(testGamma.imag() ==
        doctest::Approx(expectedGamma.imag()).epsilon(inTolerance));
  CHECK(testGammaTilde.real() ==
        doctest::Approx(expectedGammaTilde.real()).epsilon(inTolerance));
  CHECK(testGammaTilde.imag() ==
        doctest::Approx(expectedGammaTilde.imag()).epsilon(inTolerance));
}

/// \brief Compare the step Riccati solutions with an std::complex
/// implementation.
///
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T> void testSteppingSolutionsGeneral(const T inTolerance) {
  // =================== ARRANGE ==================== //
  const std::vector<T> values = {-static_cast<T>(LARGE), -static_cast<T>(1),
                                 -static_cast<T>(SMALL), static_cast<T>(0),
                                 static_cast<T>(SMALL),  static_cast<T>(1),
                                 static_cast<T>(LARGE)};
  const std::vector<T> valuesPositive = {
      static_cast<T>(SMALL), static_cast<T>(1), static_cast<T>(LARGE)};

  // The coherence functions can never have a magnitude larger than unity.
  const int numMagnitudes = 5;
  const bool includeEndPoint = true;
  const std::vector<T> gammaMagnitudeValues = conga::utils::linspace(
      static_cast<T>(0), static_cast<T>(1), numMagnitudes, includeEndPoint);
  const std::vector<T> gammaPhaseValues = {
      -static_cast<T>(1), -static_cast<T>(SMALL), static_cast<T>(0),
      static_cast<T>(SMALL), static_cast<T>(1)};

  // =================== ACT ==================== //
  for (const T energyReal : values) {
    for (const T energyImag : valuesPositive) {
      // Construct energy.
      const std::complex<T> energy(energyReal, energyImag);

      for (const T orderParameterReal : values) {
        for (const T orderParameterImag : values) {
          // Construct order parameter.
          const std::complex<T> orderParameter(orderParameterReal,
                                               orderParameterImag);

          // Homogeneouns solutions.
          // For some reason the inputs can not be std::complex.
          thrust::complex<T> gammaHomo;
          thrust::complex<T> gammaTildeHomo;
          conga::riccati::computeHomogeneous(thrust::complex<T>(orderParameter),
                                             thrust::complex<T>(energy),
                                             gammaHomo, gammaTildeHomo);

          for (const T gammaInitMagnitude : gammaMagnitudeValues) {
            for (const T gammaInitPhase : gammaPhaseValues) {
              // Construct initial values.
              const std::complex<T> gammaInit =
                  gammaInitMagnitude *
                  std::complex<T>(std::cos(gammaInitPhase),
                                  std::sin(gammaInitPhase));

              for (const T step : valuesPositive) {
                // Compute coherence functions.
                const thrust::complex<T> testGamma =
                    conga::riccati::Gamma<T>::computeStep(
                        gammaHomo, orderParameter, energy, gammaInit, step);
                const thrust::complex<T> testGammaTilde =
                    conga::riccati::GammaTilde<T>::computeStep(
                        gammaTildeHomo, orderParameter, energy, gammaInit,
                        step);

                // =================== ASSERT ==================== //
                // Compute expected coherence functions.
                const std::complex<T> expectedGamma =
                    computeStep(orderParameter, energy, gammaInit, step);
                const std::complex<T> expectedGammaTilde = computeStep(
                    -std::conj(orderParameter), energy, gammaInit, step);

                CHECK(
                    testGamma.real() ==
                    doctest::Approx(expectedGamma.real()).epsilon(inTolerance));
                CHECK(
                    testGamma.imag() ==
                    doctest::Approx(expectedGamma.imag()).epsilon(inTolerance));
                CHECK(testGammaTilde.real() ==
                      doctest::Approx(expectedGammaTilde.real())
                          .epsilon(inTolerance));
                CHECK(testGammaTilde.imag() ==
                      doctest::Approx(expectedGammaTilde.imag())
                          .epsilon(inTolerance));
              }
            }
          }
        }
      }
    }
  }
}

/// \brief Compare the halfstep Riccati solutions with an std::complex
/// implementation.
///
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testHalfSteppingSolutionsGeneral(const T inTolerance) {
  // =================== ARRANGE ==================== //
  const std::vector<T> values = {-static_cast<T>(LARGE), -static_cast<T>(1),
                                 -static_cast<T>(SMALL), static_cast<T>(0),
                                 static_cast<T>(SMALL),  static_cast<T>(1),
                                 static_cast<T>(LARGE)};
  const std::vector<T> valuesPositive = {
      static_cast<T>(SMALL), static_cast<T>(1), static_cast<T>(LARGE)};

  // The coherence functions can never have a magnitude larger than unity.
  const int numMagnitudes = 5;
  const bool includeEndPoint = true;
  const std::vector<T> gammaMagnitudeValues = conga::utils::linspace(
      static_cast<T>(0), static_cast<T>(1), numMagnitudes, includeEndPoint);
  const std::vector<T> gammaPhaseValues = {
      -static_cast<T>(1), -static_cast<T>(SMALL), static_cast<T>(0),
      static_cast<T>(SMALL), static_cast<T>(1)};

  // =================== ACT ==================== //
  for (const T energyReal : values) {
    for (const T energyImag : valuesPositive) {
      // Construct energy.
      const std::complex<T> energy(energyReal, energyImag);

      for (const T orderParameterReal : values) {
        for (const T orderParameterImag : values) {
          // Construct order parameter.
          const std::complex<T> orderParameter(orderParameterReal,
                                               orderParameterImag);

          // Homogeneouns solutions.
          // For some reason the inputs can not be std::complex.
          thrust::complex<T> gammaHomo;
          thrust::complex<T> gammaTildeHomo;
          conga::riccati::computeHomogeneous(thrust::complex<T>(orderParameter),
                                             thrust::complex<T>(energy),
                                             gammaHomo, gammaTildeHomo);

          for (const T gammaInitMagnitude : gammaMagnitudeValues) {
            for (const T gammaInitPhase : gammaPhaseValues) {
              // Construct initial values.
              const std::complex<T> gammaInit =
                  gammaInitMagnitude *
                  std::complex<T>(std::cos(gammaInitPhase),
                                  std::sin(gammaInitPhase));

              for (const T step : valuesPositive) {
                // Compute coherence functions.
                thrust::complex<T> testGammaCenter;
                thrust::complex<T> testGammaExit;
                conga::riccati::Gamma<T>::computeHalfSteps(
                    gammaHomo, orderParameter, energy, gammaInit, step,
                    testGammaCenter, testGammaExit);

                thrust::complex<T> testGammaTildeCenter;
                thrust::complex<T> testGammaTildeExit;
                conga::riccati::GammaTilde<T>::computeHalfSteps(
                    gammaTildeHomo, orderParameter, energy, gammaInit, step,
                    testGammaTildeCenter, testGammaTildeExit);

                // =================== ASSERT ==================== //
                // Compute expected coherence functions.
                const T halfStep = static_cast<T>(0.5) * step;

                const std::complex<T> expectedGammaCenter =
                    computeStep(orderParameter, energy, gammaInit, halfStep);
                const std::complex<T> expectedGammaExit = computeStep(
                    orderParameter, energy, expectedGammaCenter, halfStep);

                const std::complex<T> expectedGammaTildeCenter = computeStep(
                    -std::conj(orderParameter), energy, gammaInit, halfStep);
                const std::complex<T> expectedGammaTildeExit =
                    computeStep(-std::conj(orderParameter), energy,
                                expectedGammaTildeCenter, halfStep);

                CHECK(testGammaCenter.real() ==
                      doctest::Approx(expectedGammaCenter.real())
                          .epsilon(inTolerance));
                CHECK(testGammaCenter.imag() ==
                      doctest::Approx(expectedGammaCenter.imag())
                          .epsilon(inTolerance));
                CHECK(testGammaExit.real() ==
                      doctest::Approx(expectedGammaExit.real())
                          .epsilon(inTolerance));
                CHECK(testGammaExit.imag() ==
                      doctest::Approx(expectedGammaExit.imag())
                          .epsilon(inTolerance));

                CHECK(testGammaTildeCenter.real() ==
                      doctest::Approx(expectedGammaTildeCenter.real())
                          .epsilon(inTolerance));
                CHECK(testGammaTildeCenter.imag() ==
                      doctest::Approx(expectedGammaTildeCenter.imag())
                          .epsilon(inTolerance));
                CHECK(testGammaTildeExit.real() ==
                      doctest::Approx(expectedGammaTildeExit.real())
                          .epsilon(inTolerance));
                CHECK(testGammaTildeExit.imag() ==
                      doctest::Approx(expectedGammaTildeExit.imag())
                          .epsilon(inTolerance));
              }
            }
          }
        }
      }
    }
  }
}
} // namespace helper

// =============== Test homogeneous solutions - particular =============== //

TEST_CASE("Test homogeneous solutions particular - double") {
  const double tolerance =
      helper::TOLERANCE_FACTOR * std::numeric_limits<double>::epsilon();
  helper::testHomogeneousSolutionsParticular(tolerance);
}

TEST_CASE("Test homogeneous solutions particular - float") {
  const float tolerance = static_cast<float>(helper::TOLERANCE_FACTOR) *
                          std::numeric_limits<float>::epsilon();
  helper::testHomogeneousSolutionsParticular(tolerance);
}

// =============== Test homogeneous solutions - general =============== //

TEST_CASE("Test homogeneous solutions general - double") {
  const double tolerance =
      helper::TOLERANCE_FACTOR * std::numeric_limits<double>::epsilon();
  helper::testHomogeneousSolutionsGeneral(tolerance);
}

TEST_CASE("Test homogeneous solutions general - float") {
  const float tolerance = static_cast<float>(helper::TOLERANCE_FACTOR) *
                          std::numeric_limits<float>::epsilon();
  helper::testHomogeneousSolutionsGeneral(tolerance);
}

// ================= Test "stepping" solutions - particular ================= //

TEST_CASE("Test stepping solutions particular - double") {
  const double tolerance =
      helper::TOLERANCE_FACTOR * std::numeric_limits<double>::epsilon();
  helper::testSteppingSolutionsParticular(tolerance);
}

TEST_CASE("Test stepping solutions particular - float") {
  const float tolerance = static_cast<float>(helper::TOLERANCE_FACTOR) *
                          std::numeric_limits<float>::epsilon();
  helper::testSteppingSolutionsParticular(tolerance);
}

// ================= Test "stepping" solutions - general ================= //

TEST_CASE("Test stepping solutions general - double") {
  const double tolerance =
      helper::TOLERANCE_FACTOR_BAD * std::numeric_limits<double>::epsilon();
  helper::testSteppingSolutionsGeneral(tolerance);
}

TEST_CASE("Test stepping solutions general - float") {
  const float tolerance = static_cast<float>(helper::TOLERANCE_FACTOR_BAD) *
                          std::numeric_limits<float>::epsilon();
  helper::testSteppingSolutionsGeneral(tolerance);
}

// ============== Test "half-stepping" solutions - general ============== //

TEST_CASE("Test half-stepping solutions general - double") {
  const double tolerance =
      helper::TOLERANCE_FACTOR_BAD * std::numeric_limits<double>::epsilon();
  helper::testHalfSteppingSolutionsGeneral(tolerance);
}

TEST_CASE("Test half-stepping solutions general - float") {
  const float tolerance = static_cast<float>(helper::TOLERANCE_FACTOR_BAD) *
                          std::numeric_limits<float>::epsilon();
  helper::testHalfSteppingSolutionsGeneral(tolerance);
}
