#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "cplx.h"
#include "utils.h"
#include <complex>
#include <doctest/doctest.h>
#include <limits>

namespace helper {
template <typename T> void testAddition(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T realA = static_cast<T>(5.32);
  const T imagA = static_cast<T>(-1.45);
  const T realB = static_cast<T>(3.79);
  const T imagB = static_cast<T>(0.49);

  // =================== ACT ==================== //
  const scg::cplx<T> result =
      scg::cplx<T>(realA, imagA) + scg::cplx<T>(realB, imagB);

  // =================== ASSERT ==================== //
  const std::complex<T> expected =
      std::complex<T>(realA, imagA) + std::complex<T>(realB, imagB);

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testSubtraction(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T realA = static_cast<T>(5.32);
  const T imagA = static_cast<T>(-1.45);
  const T realB = static_cast<T>(3.79);
  const T imagB = static_cast<T>(0.49);

  // =================== ACT ==================== //
  const scg::cplx<T> result =
      scg::cplx<T>(realA, imagA) - scg::cplx<T>(realB, imagB);

  // =================== ASSERT ==================== //
  const std::complex<T> expected =
      std::complex<T>(realA, imagA) - std::complex<T>(realB, imagB);

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testMultiplication(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T realA = static_cast<T>(5.32);
  const T imagA = static_cast<T>(-1.45);
  const T realB = static_cast<T>(3.79);
  const T imagB = static_cast<T>(0.49);

  // =================== ACT ==================== //
  const scg::cplx<T> result =
      scg::cplx<T>(realA, imagA) * scg::cplx<T>(realB, imagB);

  // =================== ASSERT ==================== //
  const std::complex<T> expected =
      std::complex<T>(realA, imagA) * std::complex<T>(realB, imagB);

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testDivision(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T realA = static_cast<T>(5.32);
  const T imagA = static_cast<T>(-1.45);
  const T realB = static_cast<T>(3.79);
  const T imagB = static_cast<T>(0.49);

  // =================== ACT ==================== //
  const scg::cplx<T> result =
      scg::cplx<T>(realA, imagA) / scg::cplx<T>(realB, imagB);

  // =================== ASSERT ==================== //
  const std::complex<T> expected =
      std::complex<T>(realA, imagA) / std::complex<T>(realB, imagB);

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testConj(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T real = static_cast<T>(0.32);
  const T imag = static_cast<T>(1.45);

  // =================== ACT ==================== //
  const scg::cplx<T> result = scg::cplx<T>(real, imag).conj();

  // =================== ASSERT ==================== //
  const std::complex<T> expected = std::conj(std::complex<T>(real, imag));

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testMagnitude(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T real = static_cast<T>(0.32);
  const T imag = static_cast<T>(1.45);

  // =================== ACT ==================== //
  const T result = scg::cplx<T>(real, imag).cabs();

  // =================== ASSERT ==================== //
  const T expected = std::abs(std::complex<T>(real, imag));

  CHECK(result == doctest::Approx(expected).epsilon(inTolerance));
}

template <typename T> void testNorm(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T real = static_cast<T>(0.32);
  const T imag = static_cast<T>(1.45);

  // =================== ACT ==================== //
  const T result = scg::cplx<T>(real, imag).cabsSq();

  // =================== ASSERT ==================== //
  const T expected = std::norm(std::complex<T>(real, imag));

  CHECK(result == doctest::Approx(expected).epsilon(inTolerance));
}

template <typename T> void testSqrt(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T real = static_cast<T>(0.32);
  const T imag = static_cast<T>(1.45);

  // =================== ACT ==================== //
  const scg::cplx<T> result = scg::cplx<T>::csqrt(scg::cplx<T>(real, imag));

  // =================== ASSERT ==================== //
  const std::complex<T> expected = std::sqrt(std::complex<T>(real, imag));

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}

template <typename T> void testExp(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // TODO(niclas): Test a range of values!
  const T real = static_cast<T>(0.32);
  const T imag = static_cast<T>(1.45);

  // =================== ACT ==================== //
  const scg::cplx<T> result = scg::cplx<T>::cexp(scg::cplx<T>(real, imag));

  // =================== ASSERT ==================== //
  const std::complex<T> expected = std::exp(std::complex<T>(real, imag));

  CHECK(result.x == doctest::Approx(expected.real()).epsilon(inTolerance));
  CHECK(result.y == doctest::Approx(expected.imag()).epsilon(inTolerance));
}
} // namespace helper

// ================= Test addition ================= //

TEST_CASE("Test addition - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testAddition(tolerance);
}

TEST_CASE("Test addition - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testAddition(tolerance);
}

// ================= Test subtraction ================= //

TEST_CASE("Test subtraction - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSubtraction(tolerance);
}

TEST_CASE("Test subtraction - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testSubtraction(tolerance);
}

// ================= Test multiplication ================= //

TEST_CASE("Test multiplication - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testMultiplication(tolerance);
}

TEST_CASE("Test multiplication - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testMultiplication(tolerance);
}

// ================= Test division ================= //

TEST_CASE("Test division - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testDivision(tolerance);
}

TEST_CASE("Test division - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testDivision(tolerance);
}

// ================= Test conj ================= //

TEST_CASE("Test conj - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testConj(tolerance);
}

TEST_CASE("Test conj - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testConj(tolerance);
}

// ================= Test magnitude ================= //

TEST_CASE("Test magnitude - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testMagnitude(tolerance);
}

TEST_CASE("Test magnitude - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testMagnitude(tolerance);
}

// ================= Test norm ================= //

TEST_CASE("Test norm - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testNorm(tolerance);
}

TEST_CASE("Test norm - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testNorm(tolerance);
}

// ================= Test sqrt ================= //

TEST_CASE("Test sqrt - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSqrt(tolerance);
}

TEST_CASE("Test sqrt - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testSqrt(tolerance);
}

// ================= Test exp ================= //

TEST_CASE("Test exp - double") {
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testExp(tolerance);
}

TEST_CASE("Test exp - float") {
  const float tolerance = 10.0 * std::numeric_limits<float>::epsilon();
  helper::testExp(tolerance);
}
