#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/fermi_surface/FermiSurfaceCircle.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <cmath>

namespace helper {
// A helper function to test the number of angles. Instantiate the
// Fermi surface with some number of angles and see that it returns the same
// value.
template <typename T> void testGetNumAngles(const int inNumAngles = 10) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  // =================== ACT ==================== //
  const int numAngles = fermiSurface.getNumAngles();

  // =================== ASSERT ==================== //
  CHECK(numAngles == inNumAngles);
}

// A helper function to test the momentum offset. Instantiate the Fermi surface
// and check that the momentum origin offset is zero, i.e. that it is centered
// around the origin.
template <typename T> void testGetMomentumOffset(const T inTolerance) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;

  // =================== ACT ==================== //
  T offsetX;
  T offsetY;
  fermiSurface.getMomentumOffset(offsetX, offsetY);

  // =================== ASSERT ==================== //
  // Check that the origin offset is zero, i.e. the Fermi surface is centered
  // around the origin.
  const T offsetXExpected = static_cast<T>(0.0);
  const T offsetYExpected = static_cast<T>(0.0);
  CHECK(offsetX == doctest::Approx(offsetXExpected).epsilon(inTolerance));
  CHECK(offsetY == doctest::Approx(offsetYExpected).epsilon(inTolerance));
}

// A helper function test the angle offset. Instantiate the Fermi surface
// and check that the angle offset is pi/2, i.e. the angle corresponding to the
// first angle index.
template <typename T> void testGetAngleOffset(const T inTolerance) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;

  // =================== ACT ==================== //
  const T angleOffset = fermiSurface.getAngleOffset();

  // =================== ASSERT ==================== //
  // The old code always started with pF and vF along the positive y-axis, i.e.
  // an angle offset of pi/2. Here we just test that we are doing the same. In
  // the future we would like to start from an arbitrary angle, i.e. zero.
  const T angleOffsetExpected = static_cast<T>(0.5 * M_PI);
  CHECK(angleOffset ==
        doctest::Approx(angleOffsetExpected).epsilon(inTolerance));
}

// A helper function to test the system angle. The system angle and the angle
// parameterizing the Fermi surface should be offset by -pi/2. This is so
// because we are solving the Riccati equation along trajectories along the
// positive y-axis.
template <typename T>
void testGetSystemAngle(const T inTolerance, const int inNumAngles = 11) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  const T angleOffset = fermiSurface.getAngleOffset();

  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);

  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    // We need to wrap the angles to not get false negatives.
    const T systemAngle =
        conga::utils::wrapToRangeTwoPi(fermiSurface.getSystemAngle(angleIdx));

    // =================== ASSERT ==================== //
    const T angle = angleStep * static_cast<T>(angleIdx) + angleOffset;
    const T systemAngleExpected =
        conga::utils::wrapToRangeTwoPi(angle - static_cast<T>(0.5 * M_PI));

    CHECK(systemAngle ==
          doctest::Approx(systemAngleExpected).epsilon(inTolerance));
  }
}

// A helper function to test the Fermi momentum. Check that it is on the unit
// circle for all angles.
template <typename T>
void testGetFermiMomentum(const T inTolerance, const int inNumAngles = 12) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  const T angleOffset = fermiSurface.getAngleOffset();

  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);

  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    T fermiMomentumX;
    T fermiMomentumY;
    fermiSurface.getFermiMomentum(angleIdx, fermiMomentumX, fermiMomentumY);
    const T fermiMomentum = std::hypot(fermiMomentumX, fermiMomentumY);

    // =================== ASSERT ==================== //
    const T angle = angleStep * static_cast<T>(angleIdx) + angleOffset;
    const T fermiMomentumXExpected = std::cos(angle);
    const T fermiMomentumYExpected = std::sin(angle);
    const T fermiMomentumExpected = static_cast<T>(1.0);

    CHECK(fermiMomentumX ==
          doctest::Approx(fermiMomentumXExpected).epsilon(inTolerance));
    CHECK(fermiMomentumY ==
          doctest::Approx(fermiMomentumYExpected).epsilon(inTolerance));
    CHECK(fermiMomentum ==
          doctest::Approx(fermiMomentumExpected).epsilon(inTolerance));
  }
}

// A helper function to test the Fermi velocity. Check that it is on the unit
// circle for all angles.
template <typename T>
void testGetFermiVelocity(const T inTolerance, const int inNumAngles = 7) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  const T angleOffset = fermiSurface.getAngleOffset();

  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);

  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    T fermiVelocityX;
    T fermiVelocityY;
    fermiSurface.getFermiVelocity(angleIdx, fermiVelocityX, fermiVelocityY);
    const T fermiSpeed = std::hypot(fermiVelocityX, fermiVelocityY);

    // =================== ASSERT ==================== //
    const T angle = angleStep * static_cast<T>(angleIdx) + angleOffset;
    const T fermiVelocityXExpected = std::cos(angle);
    const T fermiVelocityYExpected = std::sin(angle);
    const T fermiSpeedExpected = static_cast<T>(1.0);

    CHECK(fermiVelocityX ==
          doctest::Approx(fermiVelocityXExpected).epsilon(inTolerance));
    CHECK(fermiVelocityY ==
          doctest::Approx(fermiVelocityYExpected).epsilon(inTolerance));
    CHECK(fermiSpeed ==
          doctest::Approx(fermiSpeedExpected).epsilon(inTolerance));
  }
}

// A helper function to test the integrand factor. Because this is a circular
// Fermi surface it should be a constant, independent on angle. Also, it should
// sum to one.
template <typename T>
void testGetIntegrandFactor(const T inTolerance, const int inNumAngles = 9) {
  // =================== ARRANGE ==================== //
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  // After looping over angles and summing this should be one.
  T integrandFactorSum = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    const T integrandFactor = fermiSurface.getIntegrandFactor(angleIdx);
    integrandFactorSum += integrandFactor;

    // =================== ASSERT ==================== //
    CHECK(integrandFactor > static_cast<T>(0));
  }

  // =================== ASSERT ==================== //
  const T integrandFactorSumExpected = static_cast<T>(1.0);
  CHECK(integrandFactorSum ==
        doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
}
} // namespace helper

// =============== Test FermiSurfaceCircle::getNumAngles =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getNumAngles") {
  helper::testGetNumAngles<float>();
}

TEST_CASE("Test FermiSurfaceCircle<double>::getNumAngles") {
  helper::testGetNumAngles<double>();
}

// =============== Test FermiSurfaceCircle::getMomentumOffset =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getMomentumOffset") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetMomentumOffset(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getMomentumOffset") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetMomentumOffset(TOLERANCE);
}

// =============== Test FermiSurfaceCircle::getAngleOffset =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getAngleOffset") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetAngleOffset(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getAngleOffset") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetAngleOffset(TOLERANCE);
}

// =============== Test FermiSurfaceCircle::getSystemAngle =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getSystemAngle") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetSystemAngle(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getSystemAngle") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetSystemAngle(TOLERANCE);
}

// =============== Test FermiSurfaceCircle::getFermiMumentum =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getFermiMumentum") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetFermiMomentum(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getFermiMumentum") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetFermiMomentum(TOLERANCE);
}

// =============== Test FermiSurfaceCircle::getFermiVelocity =============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getFermiVelocity") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetFermiVelocity(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getFermiVelocity") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetFermiVelocity(TOLERANCE);
}

// ============== Test FermiSurfaceCircle::getIntegrandFactor ============== //

TEST_CASE("Test FermiSurfaceCircle<float>::getIntegrandFactor") {
  // Relative tolerance.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetIntegrandFactor(TOLERANCE);
}

TEST_CASE("Test FermiSurfaceCircle<double>::getIntegrandFactor") {
  // Relative tolerance.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetIntegrandFactor(TOLERANCE);
}
