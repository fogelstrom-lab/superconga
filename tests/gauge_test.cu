#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/riccati/energy_shift/Gauges.h"
#include "conga/wrappers/doctest.h"

#include <limits>

namespace helper {
// These tests are not very good. But at least the gauges cannot be changed
// without the tests failing.

// Factor of machine precision for the tolerance of tests.
const double TOLERANCE_FACTOR = 10.0;
const double SCALE = 0.01;

/// \brief Test Landau gauge.
///
/// \return Void.
template <typename T> void testLandau() {
  // =================== ARRANGE ==================== //
  const T fluxDensity = static_cast<T>(1);
  const T area = static_cast<T>(1);
  const T x = static_cast<T>(1);
  const T y = static_cast<T>(1);

  // =================== ACT ==================== //
  T resultX;
  T resultY;
  conga::gauge::Landau<T>::compute(fluxDensity, area, x, y, resultX, resultY);

  // =================== ASSERT ==================== //
  const T expectedX = static_cast<T>(0);
  const T expectedY = fluxDensity * x;

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(resultX == doctest::Approx(expectedX).epsilon(tolerance).scale(scale));
  CHECK(resultY == doctest::Approx(expectedY).epsilon(tolerance).scale(scale));
}

/// \brief Test Solenoid gauge.
///
/// \return Void.
template <typename T> void testSolenoid() {
  // =================== ARRANGE ==================== //
  const T fluxDensity = static_cast<T>(1);
  const T area = static_cast<T>(1);
  const T x = static_cast<T>(1);
  const T y = static_cast<T>(1);

  // =================== ACT ==================== //
  T resultX;
  T resultY;
  conga::gauge::Solenoid<T>::compute(fluxDensity, area, x, y, resultX, resultY);

  // =================== ASSERT ==================== //
  const T factor =
      fluxDensity * area / (static_cast<T>(2.0 * M_PI) * (x * x + y * y));
  const T expectedX = -factor * y;
  const T expectedY = factor * x;

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(resultX == doctest::Approx(expectedX).epsilon(tolerance).scale(scale));
  CHECK(resultY == doctest::Approx(expectedY).epsilon(tolerance).scale(scale));
}

/// \brief Test Symmetric gauge.
///
/// \return Void.
template <typename T> void testSymmetric() {
  // =================== ARRANGE ==================== //
  const T fluxDensity = static_cast<T>(1);
  const T area = static_cast<T>(1);
  const T x = static_cast<T>(1);
  const T y = static_cast<T>(1);

  // =================== ACT ==================== //
  T resultX;
  T resultY;
  conga::gauge::Symmetric<T>::compute(fluxDensity, area, x, y, resultX,
                                      resultY);

  // =================== ASSERT ==================== //
  const T factor = static_cast<T>(0.5) * fluxDensity;
  const T expectedX = -factor * y;
  const T expectedY = factor * x;

  // Check.
  const T tolerance = static_cast<float>(TOLERANCE_FACTOR) *
                      std::numeric_limits<float>::epsilon();
  const T scale = static_cast<T>(SCALE);
  CHECK(resultX == doctest::Approx(expectedX).epsilon(tolerance).scale(scale));
  CHECK(resultY == doctest::Approx(expectedY).epsilon(tolerance).scale(scale));
}
} // namespace helper

// =============== Test Landau gauge =============== //

TEST_CASE("Test Landau gauge - <double>") { helper::testLandau<double>(); }

TEST_CASE("Test Landau gauge - <float>") { helper::testLandau<float>(); }

// =============== Test Solenoid gauge =============== //

TEST_CASE("Test Solenoid gauge - <double>") { helper::testSolenoid<double>(); }

TEST_CASE("Test Solenoid gauge - <float>") { helper::testSolenoid<float>(); }

// =============== Test Symmetric gauge =============== //

TEST_CASE("Test Symmetric gauge - <double>") {
  helper::testSymmetric<double>();
}

TEST_CASE("Test Symmetric gauge - <float>") { helper::testSymmetric<float>(); }
