#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/green/LocalGreens.h"
#include "conga/riccati/RiccatiSolutions.h"
#include "conga/wrappers/doctest.h"

#include <thrust/complex.h>

#include <limits>

namespace helper {
// Factor of machine precision for the tolerance of tests.
const double TOLERANCE_FACTOR = 10.0;
const double SCALE = 0.01;

/// \brief Test Green function element 'g'.
///
/// \return Void.
template <typename T> void testGreen() {
  // =================== ARRANGE ==================== //
  const thrust::complex<T> energy(static_cast<T>(1), static_cast<T>(3));
  const thrust::complex<T> orderParameter(static_cast<T>(4),
                                          static_cast<T>(-1));
  thrust::complex<T> gamma;
  thrust::complex<T> gammaTilde;
  conga::riccati::computeHomogeneous(orderParameter, energy, gamma, gammaTilde);

  // =================== ACT ==================== //
  // Compute the Green function element 'g'.
  const thrust::complex<T> result =
      conga::Greens<T>::compute(gamma, gammaTilde);

  // =================== ASSERT ==================== //
  // The expected value with bulk coherence functions.
  const thrust::complex<T> expected =
      -energy / thrust::sqrt(thrust::norm(orderParameter) - energy * energy);

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(result.real() ==
        doctest::Approx(expected.real()).epsilon(tolerance).scale(scale));
  CHECK(result.imag() ==
        doctest::Approx(expected.imag()).epsilon(tolerance).scale(scale));
}

/// \brief Test Green function element 'f'.
///
/// \return Void.
template <typename T> void testGreenAnomalous() {
  // =================== ARRANGE ==================== //
  const thrust::complex<T> energy(static_cast<T>(1), static_cast<T>(3));
  const thrust::complex<T> orderParameter(static_cast<T>(4),
                                          static_cast<T>(-1));
  thrust::complex<T> gamma;
  thrust::complex<T> gammaTilde;
  conga::riccati::computeHomogeneous(orderParameter, energy, gamma, gammaTilde);

  // =================== ACT ==================== //
  // Compute the sum of Green function elements 'f' and '\tilde{f}^*'.
  const thrust::complex<T> result =
      conga::Greens_Delta<T>::compute(gamma, gammaTilde);

  // =================== ASSERT ==================== //
  // The expected value with bulk coherence functions.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(orderParameter) - energy * energy);
  const thrust::complex<T> expected =
      orderParameter *
      (static_cast<T>(1) / Omega + static_cast<T>(1) / thrust::conj(Omega));

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(result.real() ==
        doctest::Approx(expected.real()).epsilon(tolerance).scale(scale));
  CHECK(result.imag() ==
        doctest::Approx(expected.imag()).epsilon(tolerance).scale(scale));
}
} // namespace helper

// =============== Test Green function element g =============== //

TEST_CASE("Test Green function element g - <double>") {
  helper::testGreen<double>();
}

TEST_CASE("Test Green function element g - <float>") {
  helper::testGreen<float>();
}

// =============== Test Green function element f =============== //

TEST_CASE("Test Green function element f - <double>") {
  helper::testGreenAnomalous<double>();
}

TEST_CASE("Test Green function element f - <float>") {
  helper::testGreenAnomalous<float>();
}
