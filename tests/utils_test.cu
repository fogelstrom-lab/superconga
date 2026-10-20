#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <array>
#include <limits>

// ====================== Test conga::utils::wrapToRange ======================
// //

TEST_CASE("Test conga::utils::wrapToRange<int> - low = 0") {
  // Interval to wrap to [low, high).
  const int LOW = 0;
  const int HIGH = 10;

  // Define test values.
  const int NUM_TESTS = 9;
  const std::array<int, NUM_TESTS> valuesTest = {0,  4,   10,  20, 23,
                                                 -4, -10, -20, -23};
  const std::array<int, NUM_TESTS> valuesExpected = {0, 4, 0, 0, 3, 6, 0, 0, 7};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test conga::utils::wrapToRange<int> - low != 0") {
  // Interval to wrap to [low, high).
  const int LOW = 5;
  const int HIGH = 10;

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<int, NUM_TESTS> valuesTest = {7, 0, 11, -1};
  const std::array<int, NUM_TESTS> valuesExpected = {7, 5, 6, 9};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test conga::utils::wrapToRange<float> - low = 0") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();

  // Interval to wrap to [low, high).
  const float LOW = 0.0F;
  const float HIGH = 10.0F;

  // Define test values.
  const int NUM_TESTS = 9;
  const std::array<float, NUM_TESTS> valuesTest = {
      0.0F, 4.0F, 10.0F, 20.0F, 23.0F, -4.0F, -10.0F, -20.0F, -23.0F};
  const std::array<float, NUM_TESTS> valuesExpected = {
      0.0F, 4.0F, 0.0F, 0.0F, 3.0F, 6.0F, 0.0F, 0.0F, 7.0F};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const float value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test conga::utils::wrapToRange<float> - low != 0") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const float TOLERANCE = 10.0F * std::numeric_limits<float>::epsilon();

  // Interval to wrap to [low, high).
  const float LOW = 5.0F;
  const float HIGH = 10.0F;

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<float, NUM_TESTS> valuesTest = {7.0F, 0.0F, 11.0F, -1.0F};
  const std::array<float, NUM_TESTS> valuesExpected = {7.0F, 5.0F, 6.0F, 9.0F};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const float value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test conga::utils::wrapToRange<double> - low = 0") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();

  // Interval to wrap to [low, high).
  const double LOW = 0.0;
  const double HIGH = 10.0;

  // Define test values.
  const int NUM_TESTS = 9;
  const std::array<double, NUM_TESTS> valuesTest = {
      0.0, 4.0, 10.0, 20.0, 23.0, -4.0, -10.0, -20.0, -23.0};
  const std::array<double, NUM_TESTS> valuesExpected = {0.0, 4.0, 0.0, 0.0, 3.0,
                                                        6.0, 0.0, 0.0, 7.0};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const double value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test conga::utils::wrapToRange<double> - low != 0") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();

  // Interval to wrap to [low, high).
  const double LOW = 5.0;
  const double HIGH = 10.0;

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<double, NUM_TESTS> valuesTest = {7.0, 0.0, 11.0, -1.0};
  const std::array<double, NUM_TESTS> valuesExpected = {7.0, 5.0, 6.0, 9.0};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const double value = conga::utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

// ======================= Test conga::utils::rotate ======================= //

// A helper function for testing conga::utils::rotate.
// TODO(niclas): Should helper functions be in a separate namespace here?
template <typename T>
void rotateTest(const T inMinExponent, const T inMaxExponent,
                const T inTolerance, const int inNumPoints = 10,
                const int inNumAnglesPerPoints = 10) {
  // How much to increment the exponent each step.
  const T exponentStep = (inMaxExponent - inMinExponent) /
                         static_cast<T>(inNumAnglesPerPoints - 1);

  // Perform tests.
  for (int i = 0; i < inNumPoints; ++i) {
    // The point we will rotate.
    const T angleBase = static_cast<T>(2.0 * M_PI) * static_cast<T>(i) /
                        static_cast<T>(inNumPoints);

    // TODO(niclas): Should we test large/small magnitude as well.
    const T xBase = cos(angleBase);
    const T yBase = sin(angleBase);

    for (int j = 0; j < inNumAnglesPerPoints; ++j) {
      // Interpolate exponent to get the current angle.
      const T exponent = inMinExponent + exponentStep * static_cast<T>(j);
      const T currentAngle = pow(static_cast<T>(10.0), exponent);

      // Test both positive and negative angles.
      const std::array<T, 2> angles = {currentAngle, -currentAngle};

      // A nice, modern, C++ loop!
      for (const T &angle : angles) {
        // Rotate base point.
        T x;
        T y;
        conga::utils::rotate(angle, xBase, yBase, x, y);

        // Out expected output.
        const T xExpected = cos(angle + angleBase);
        const T yExpected = sin(angle + angleBase);

        // Check values.
        REQUIRE(x == doctest::Approx(xExpected).epsilon(inTolerance));
        REQUIRE(y == doctest::Approx(yExpected).epsilon(inTolerance));
      }
    }
  }
}

TEST_CASE("Test utils:rotate<float> - small angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const float minExponent = -4.0F;
  const float maxExponent = -2.0F;

  // Note, machine precision is too harsh.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

TEST_CASE("Test utils:rotate<float> - medium angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const float minExponent = -2.0F;
  const float maxExponent = 0.0F;

  // Note, machine precision is too harsh.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

TEST_CASE("Test utils:rotate<float> - large angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const float minExponent = 0.0F;
  const float maxExponent = 2.0F;

  // Note that the tolerance must be increased, because we get a worse loss of
  // precision for very large angles.
  const float tolerance = 100.0F * std::numeric_limits<float>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

TEST_CASE("Test utils:rotate<double> - small angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const double minExponent = -4.0;
  const double maxExponent = -2.0;

  // Note, machine precision is too harsh.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

TEST_CASE("Test utils:rotate<double> - medium angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const double minExponent = -2.0;
  const double maxExponent = 0.0;

  // Note, machine precision is too harsh.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

TEST_CASE("Test utils:rotate<double> - large angles") {
  // Set min and max value of the exponent.
  // The angles tested will be [10^min, 10^max].
  const double minExponent = 0.0;
  const double maxExponent = 2.0;

  // Note that the tolerance must be increased, because we get a worse loss of
  // precision for very large angles.
  const double tolerance = 100.0 * std::numeric_limits<double>::epsilon();

  // Perform tests.
  rotateTest(minExponent, maxExponent, tolerance);
}

// ======================== Test conga::utils::sign ======================== //

TEST_CASE("Test utils:sign<int>") {
  // Define test values.
  const int NUM_TESTS = 5;
  const std::array<int, NUM_TESTS> valuesTest = {0, 2, 17, -44, -9};
  const std::array<int, NUM_TESTS> valuesExpected = {0, 1, 1, -1, -1};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = conga::utils::sign(valuesTest[i]);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test utils:sign<float>") {
  // Define test values.
  const int NUM_TESTS = 5;
  const std::array<float, NUM_TESTS> valuesTest = {0.0F, 1e6F, 1.01F, -1e-7F,
                                                   -56.0F};
  const std::array<int, NUM_TESTS> valuesExpected = {0, 1, 1, -1, -1};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = conga::utils::sign(valuesTest[i]);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test utils:sign<double>") {
  // Define test values.
  const int NUM_TESTS = 5;
  const std::array<double, NUM_TESTS> valuesTest = {0.0, 1e12, 1e-7, -4.8,
                                                    -76.34};
  const std::array<int, NUM_TESTS> valuesExpected = {0, 1, 1, -1, -1};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = conga::utils::sign(valuesTest[i]);
    CHECK(value == valuesExpected[i]);
  }
}
