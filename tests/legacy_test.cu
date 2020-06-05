#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "OrderParameterBulk.h"
#include "utils.h"
#include <cmath>
#include <doctest.h>
#include <limits>

namespace legacy {
// This legacy code generates a lot of warnings from promition to double. So,
// lets turn it off.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

// This is the implementation of atan2 that was present in the code previously.
template <typename T> T atan2(const T y, const T x) {
  T phase = (T)0.0;

  // Interior of quadrants
  if (x > (T)0.0 && y > (T)0.0)
    phase = atan(y / x);
  else if (x < (T)0.0 && y > (T)0.0)
    phase = M_PI + atan(y / x);
  else if (x < (T)0.0 && y < (T)0.0)
    phase = M_PI + atan(y / x);
  else if (x > (T)0.0 && y < (T)0.0)
    phase = (T)2.0 * M_PI + atan(y / x);
  // Boundaries
  else if (x == (T)0.0) {
    if (y > (T)0.0)
      phase = (T)0.5 * M_PI;
    if (y < (T)0.0)
      phase = (T)1.5 * M_PI;
  } else if (y == (T)0.0) {
    if (x > (T)0.0)
      phase = (T)0.0;
    if (x < (T)0.0)
      phase = M_PI;
  }
  return phase;
}

// This is the s-wave bulk solver that was present in the code previously.
template <typename T>
T computeOrderParameterBulkSWave(const T T_Tc, const T deltaConvergence,
                                 const int inNumIterations = 1000) {
  const T gMix = -0.1;
  const int numMatsubara = 1000.0 / T_Tc;

  // Delta, initial guess
  T delta = 1.76 * sqrt(1.0 - T_Tc);

  T matsubaraSum, eps, g, deltaDiff;

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inNumIterations; ++iterationIdx) {
    matsubaraSum = 0.0;

    for (int i = 0; i <= numMatsubara; ++i) {
      eps = M_PI * T_Tc * (2.0 * i + 1.0);
      matsubaraSum += 1.0 / sqrt(eps * eps + delta * delta) - 1.0 / eps;
    }
    matsubaraSum *= delta;

    g = delta * log(T_Tc) - 2.0 * M_PI * T_Tc * matsubaraSum;
    deltaDiff = g * gMix;
    delta += deltaDiff;

    isConverged = (abs(deltaDiff / delta) < deltaConvergence);
    if (isConverged) {
      break;
    }
  }

  return delta;
}

// Turn warnings back on again.
#pragma GCC diagnostic pop
} // namespace legacy

namespace helper {
// A helper function to test legacy::atan2.
template <typename T>
void testLegacyAtan2(const T inTolerance, const int inNumAngles = 32,
                     const int inNumRadii = 10,
                     const T inMinRadius = static_cast<T>(1e-4),
                     const T inMaxRadius = static_cast<T>(1e4)) {
  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);
  const T radiusStep =
      (inMaxRadius - inMinRadius) / static_cast<T>(inNumRadii - 1);

  for (int radiusIdx = 0; radiusIdx < inNumRadii; ++radiusIdx) {
    const T radius = inMinRadius + radiusStep * static_cast<T>(radiusIdx);
    for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
      const T angle = angleStep * static_cast<T>(angleIdx);

      const T x = radius * cos(angle);
      const T y = radius * sin(angle);

      const T angleOld = legacy::atan2(y, x);
      const T angleNew = utils::wrapToRangeTwoPi(atan2(y, x));
      CHECK(angle == doctest::Approx(angleNew).epsilon(inTolerance));
      CHECK(angleOld == doctest::Approx(angleNew).epsilon(inTolerance));
    }
  }
}

// A helper function to test legacy::computeOrderParameterBulkSWave.
template <typename T>
void testComputeOrderParameterBulkSWave(const T inTolerance) {
  // The temperatures to test.
  const int numTemperatures = 2;
  const std::array<T, numTemperatures> temperatures = {static_cast<T>(0.1),
                                                       static_cast<T>(0.9)};

  // The convergence criteria to test.
  const int numConvergenceCriteria = 2;
  const std::array<T, numConvergenceCriteria> convergenceCriteria = {
      static_cast<T>(1e-6), static_cast<T>(1e-8)};

  // Loop over temperatures.
  for (const T &temperature : temperatures) {
    // Loop over convergence criteria.
    for (const T &convergenceCriterion : convergenceCriteria) {
      // Compute the legacy value.
      const T oldOrderParameter = legacy::computeOrderParameterBulkSWave(
          temperature, convergenceCriterion);

      // Compute the new value.
      const T newOrderParameter =
          computeOrderParameterBulkSWave(temperature, convergenceCriterion);

      // Compare results.
      CHECK(oldOrderParameter ==
            doctest::Approx(newOrderParameter).epsilon(inTolerance));
    }
  }
}
} // namespace helper

// ================== Test computeOrderParameterBulkSWave ================== //

TEST_CASE("Test legacy::computeOrderParameterBulkSWave<float>") {
  // Relative tolerance.
  // Note that the two implementations differ in how they determine when
  // convergence has been achieved amongs other things. Also, that the legacy
  // implementation might not even converge. So we only check that they roughly
  // give the same answer.
  const float TOLERANCE = 1e-4F;
  helper::testComputeOrderParameterBulkSWave(TOLERANCE);
}

TEST_CASE("Test legacy::computeOrderParameterBulkSWave<double>") {
  // Relative tolerance.
  // Note that the two implementations differ in how they determine when
  // convergence has been achieved amongs other things. Also, that the legacy
  // implementation might not even converge. So we only check that they roughly
  // give the same answer.
  const double TOLERANCE = 1e-4;
  helper::testComputeOrderParameterBulkSWave(TOLERANCE);
}

// =============================== Test atan2 =============================== //

TEST_CASE("Test legacy::atan2<float>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testLegacyAtan2(TOLERANCE);
}

TEST_CASE("Test legacy::atan2<double>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testLegacyAtan2(TOLERANCE);
}
