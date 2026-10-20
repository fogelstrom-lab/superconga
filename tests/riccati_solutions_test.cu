#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/riccati/RiccatiSolutions.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <thrust/complex.h>

#include <cmath>
#include <limits>
#include <vector>

namespace helper {
// Factor of machine precision for the tolerance of most tests.
const double TOLERANCE_FACTOR = 10.0;
const double SCALE = 0.01;

/// \brief Compare the homogenous Riccati solutions to values computed by hand.
///
/// \return Void.
template <typename T> void testHomogeneousSolutions() {
  // =================== ARRANGE ==================== //
  // Note, do not change these values. The expected result is computed by hand
  // using them.
  const thrust::complex<T> energy(static_cast<T>(0), static_cast<T>(3));
  const thrust::complex<T> orderParameter(static_cast<T>(4), static_cast<T>(0));

  // =================== ACT ==================== //
  // Compute coherence functions.
  thrust::complex<T> gammaTest;
  thrust::complex<T> gammaTildeTest;
  conga::riccati::computeHomogeneous(orderParameter, energy, gammaTest,
                                     gammaTildeTest);

  // =================== ASSERT ==================== //
  // Compute expected coherence functions.
  const thrust::complex<T> gammaExpected(static_cast<T>(0),
                                         static_cast<T>(0.5));
  const thrust::complex<T> gammaTildeExpected = thrust::conj(gammaExpected);

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(gammaTest.real() ==
        doctest::Approx(gammaExpected.real()).epsilon(tolerance).scale(scale));
  CHECK(gammaTest.imag() ==
        doctest::Approx(gammaExpected.imag()).epsilon(tolerance).scale(scale));
  CHECK(gammaTildeTest.real() == doctest::Approx(gammaTildeExpected.real())
                                     .epsilon(tolerance)
                                     .scale(scale));
  CHECK(gammaTildeTest.imag() == doctest::Approx(gammaTildeExpected.imag())
                                     .epsilon(tolerance)
                                     .scale(scale));
}

/// \brief Check that the stepping, when starting with the homogeneous solution,
/// changes nothing.
///
/// \return Void.
template <typename T> void testStepSolutionsHomogeneous() {
  // =================== ARRANGE ==================== //
  // Some arbitrary values.
  const T step = static_cast<T>(1);
  const thrust::complex<T> energy(static_cast<T>(0.5), static_cast<T>(1));
  const thrust::complex<T> orderParameter(static_cast<T>(1),
                                          static_cast<T>(0.5));

  // Compute homogeneous coherence functions.
  thrust::complex<T> gammaExpected;
  thrust::complex<T> gammaTildeExpected;
  conga::riccati::computeHomogeneous(orderParameter, energy, gammaExpected,
                                     gammaTildeExpected);

  // =================== ACT ==================== //
  const thrust::complex<T> gammaTest =
      conga::riccati::computeStep(orderParameter, energy, gammaExpected, step);
  const thrust::complex<T> gammaTildeTest = conga::riccati::computeStep(
      -thrust::conj(orderParameter), energy, gammaTildeExpected, step);

  // =================== ASSERT ==================== //
  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(gammaTest.real() ==
        doctest::Approx(gammaExpected.real()).epsilon(tolerance).scale(scale));
  CHECK(gammaTest.imag() ==
        doctest::Approx(gammaExpected.imag()).epsilon(tolerance).scale(scale));
  CHECK(gammaTildeTest.real() == doctest::Approx(gammaTildeExpected.real())
                                     .epsilon(tolerance)
                                     .scale(scale));
  CHECK(gammaTildeTest.imag() == doctest::Approx(gammaTildeExpected.imag())
                                     .epsilon(tolerance)
                                     .scale(scale));
}

/// \brief Check that the stepping, with zero step size, changes nothing.
///
/// \return Void.
template <typename T> void testStepSolutionsZero() {
  // =================== ARRANGE ==================== //
  // Some arbitrary values.
  const thrust::complex<T> energy(static_cast<T>(0), static_cast<T>(1));
  const thrust::complex<T> orderParameter(static_cast<T>(1), static_cast<T>(0));
  const thrust::complex<T> init(static_cast<T>(0), static_cast<T>(0.5));
  const T step = static_cast<T>(0);

  // =================== ACT ==================== //
  const thrust::complex<T> gammaTest =
      conga::riccati::computeStep(orderParameter, energy, init, step);
  const thrust::complex<T> gammaTildeTest = conga::riccati::computeStep(
      -thrust::conj(orderParameter), energy, init, step);

  // =================== ASSERT ==================== //
  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(gammaTest.real() ==
        doctest::Approx(init.real()).epsilon(tolerance).scale(scale));
  CHECK(gammaTest.imag() ==
        doctest::Approx(init.imag()).epsilon(tolerance).scale(scale));
  CHECK(gammaTildeTest.real() ==
        doctest::Approx(init.real()).epsilon(tolerance).scale(scale));
  CHECK(gammaTildeTest.imag() ==
        doctest::Approx(init.imag()).epsilon(tolerance).scale(scale));
}

/// \brief Check that the stepping, with a large step size, yields the
/// expected results.
///
/// \return Void.
template <typename T> void testStepSolutionsLarge() {
  // =================== ARRANGE ==================== //
  // Some arbitrary values, but large step size.
  const thrust::complex<T> energy(static_cast<T>(0.5), static_cast<T>(1));
  const thrust::complex<T> orderParameter(static_cast<T>(1),
                                          static_cast<T>(0.5));
  const thrust::complex<T> init(static_cast<T>(0.5), static_cast<T>(0.5));
  const T step = static_cast<T>(1000);

  // =================== ACT ==================== //
  const thrust::complex<T> gammaTest =
      conga::riccati::computeStep(orderParameter, energy, init, step);
  const thrust::complex<T> gammaTildeTest = conga::riccati::computeStep(
      -thrust::conj(orderParameter), energy, init, step);

  // =================== ASSERT ==================== //
  // Compute homogeneous coherence functions.
  thrust::complex<T> gammaHomo;
  thrust::complex<T> gammaTildeHomo;
  conga::riccati::computeHomogeneous(orderParameter, energy, gammaHomo,
                                     gammaTildeHomo);

  // In the limit step --> infinity this is the result. Ideally it would go to
  // the homogeneous solution like the analytical step assuming constant
  // energies, or the backward Euler method, does. But it doesn't. However, for
  // smaller steps it is great.
  const thrust::complex<T> gammaExpected = static_cast<T>(2) * gammaHomo - init;
  const thrust::complex<T> gammaTildeExpected =
      static_cast<T>(2) * gammaTildeHomo - init;

  // Check.
  const T tolerance = static_cast<float>(0.01);
  const T scale = static_cast<T>(SCALE);
  CHECK(gammaTest.real() ==
        doctest::Approx(gammaExpected.real()).epsilon(tolerance).scale(scale));
  CHECK(gammaTest.imag() ==
        doctest::Approx(gammaExpected.imag()).epsilon(tolerance).scale(scale));
  CHECK(gammaTildeTest.real() == doctest::Approx(gammaTildeExpected.real())
                                     .epsilon(tolerance)
                                     .scale(scale));
  CHECK(gammaTildeTest.imag() == doctest::Approx(gammaTildeExpected.imag())
                                     .epsilon(tolerance)
                                     .scale(scale));
}
} // namespace helper

// =============== Test homogeneous solutions =============== //

TEST_CASE("Test homogeneous solutions <double>") {
  helper::testHomogeneousSolutions<double>();
}

TEST_CASE("Test homogeneous solutions <float>") {
  helper::testHomogeneousSolutions<float>();
}

// =============== Test stepping solutions (homogeneous) =============== //

TEST_CASE("Test stepping solutions (homogeneous) <double>") {
  helper::testStepSolutionsHomogeneous<double>();
}

TEST_CASE("Test stepping solutions (homogeneous) <float>") {
  helper::testStepSolutionsHomogeneous<float>();
}

// =============== Test stepping solutions (zero step size) =============== //

TEST_CASE("Test stepping solutions (zero step size) <double>") {
  helper::testStepSolutionsZero<double>();
}

TEST_CASE("Test stepping solutions (zero step size) <float>") {
  helper::testStepSolutionsZero<float>();
}

// =============== Test stepping solutions (large step size) =============== //

TEST_CASE("Test stepping solutions (large step size) <double>") {
  helper::testStepSolutionsLarge<double>();
}

TEST_CASE("Test stepping solutions (large step size) <float>") {
  helper::testStepSolutionsLarge<float>();
}
