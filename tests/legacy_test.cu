#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "utils.h"
#include <cmath>
#include <doctest.h>
#include <limits>

namespace legacy {
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
} // namespace helper

TEST_CASE("Test legacy::atan2<float>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testLegacyAtan2(TOLERANCE);
}

TEST_CASE("Test legacy::atan2<double>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double TOLERANCE = 10.0F * std::numeric_limits<double>::epsilon();
  helper::testLegacyAtan2(TOLERANCE);
}
