#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/fermi_surface/FermiSurface.h"
#include "conga/wrappers/doctest.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace helper {
// Helper function for testing getFractionalAngleIndex.
template <typename T>
void testGetFractionalAngleIndex(const T inTolerance,
                                 const int inNumAngles = 360) {

  // How many angle offsets to test. The truly relevant ones are 0 and pi/2, but
  // the function should be correct for any value.
  const int NUM_ANGLE_OFFSETS = 5;
  const std::array<T, NUM_ANGLE_OFFSETS> angleOffsets = {
      static_cast<T>(0.0), static_cast<T>(0.5 * M_PI), static_cast<T>(2.345),
      static_cast<T>(-0.934), static_cast<T>(7.5)};

  // Hos many Fermi surface origin offsets to test. The truly relevant ones are
  // 0 and pi (those arise naturally from the tight-binding dispersion on a
  // square lattice), but it should be correct for any value.
  const int NUM_ORIGIN_OFFSETS = 3;
  const std::array<T, NUM_ORIGIN_OFFSETS> originOffsets = {
      static_cast<T>(0.0), static_cast<T>(M_PI), static_cast<T>(0.3457)};

  // Loop over x-axis Fermi surface origin offset.
  for (const T &originOffsetX : originOffsets) {
    // Loop over y-axis Fermi surface origin offset.
    for (const T &originOffsetY : originOffsets) {
      // Loop over angle offset.
      for (const T &angleOffset : angleOffsets) {
        // Loop over angles.
        for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
          // Construct the Fermi momentum.
          const T phi = conga::internal::getPolarAngle(angleIdx, inNumAngles,
                                                       angleOffset);
          const T fermiMomentumX = cos(phi) + originOffsetX;
          const T fermiMomentumY = sin(phi) + originOffsetY;

          // Compute the fractional angle index.
          const T fractionalAngleIdx = conga::internal::getFractionalAngleIndex(
              fermiMomentumX, fermiMomentumY, originOffsetX, originOffsetY,
              angleOffset, inNumAngles);

          // Check the result.
          if (fractionalAngleIdx <
              (static_cast<T>(inNumAngles) - static_cast<T>(0.5))) {
            CHECK(angleIdx ==
                  doctest::Approx(fractionalAngleIdx).epsilon(inTolerance));
          } else {
            // If the fractional angle index is close to inNumAngles, e.g.
            // inNumAngles = 10 and  fractionalAngleIdx = 9.9999, then it should
            // be compared with angleIdx 10 and not 0.
            CHECK(angleIdx + inNumAngles ==
                  doctest::Approx(fractionalAngleIdx).epsilon(inTolerance));
          }
        }
      }
    }
  }
}

// Helper function for testing getSystemAngle.
template <typename T>
void testGetSystemAngle(const T inTolerance, const int inNumAngles = 100) {
  // How many angle offsets to test. The truly relevant ones are 0 and pi/2, but
  // the function should be correct for any value.
  const int NUM_ANGLE_OFFSETS = 5;
  const std::array<T, NUM_ANGLE_OFFSETS> angleOffsets = {
      static_cast<T>(0.0), static_cast<T>(0.5 * M_PI), static_cast<T>(2.345),
      static_cast<T>(-0.934), static_cast<T>(7.5)};

  // Loop over angle offset.
  for (const T &angleOffset : angleOffsets) {
    // Loop over angles.
    for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
      // Get phi simply fetches an angle, the value doesn't matter.
      const T fermiVelocityAngle =
          conga::internal::getPolarAngle(angleIdx, inNumAngles, angleOffset);
      const T systemAngle = conga::internal::getSystemAngle(fermiVelocityAngle);

      // Because the trajectories go along the y-axis, the Fermi velocity angle
      // and the angle we should rotate the system differ by pi/2.
      CHECK(systemAngle + static_cast<T>(0.5 * M_PI) ==
            doctest::Approx(fermiVelocityAngle).epsilon(inTolerance));
    }
  }
}
} // namespace helper

// =============== Test internal::getFractionalAngleIndex =============== //

TEST_CASE("Test getFractionalAngleIndex<float>") {
  // Relative tolerance.
  // Note, that it is not very good.
  const float TOLERANCE = 1000.0F * std::numeric_limits<float>::epsilon();
  helper::testGetFractionalAngleIndex(TOLERANCE);
}

TEST_CASE("Test getFractionalAngleIndex<double>") {
  // Relative tolerance.
  // Note, that it is not very good.
  const double TOLERANCE = 1000.0 * std::numeric_limits<double>::epsilon();
  helper::testGetFractionalAngleIndex(TOLERANCE);
}

// ================== Test internal::getSystemAngle ================== //

TEST_CASE("Test internal::getSystemAngle<float>") {
  // Relative tolerance.
  const float TOLERANCE = std::numeric_limits<float>::epsilon();
  helper::testGetSystemAngle(TOLERANCE);
}

TEST_CASE("Test internal::getSystemAngle<double>") {
  // Relative tolerance.
  const double TOLERANCE = std::numeric_limits<double>::epsilon();
  helper::testGetSystemAngle(TOLERANCE);
}
