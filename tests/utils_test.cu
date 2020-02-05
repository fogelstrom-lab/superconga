#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "utils.h"
#include <array>
#include <doctest.h>
#include <limits>

// ====================== Test utils::wrapToRange ====================== //

TEST_CASE("Test utils::wrapToRange<int> - low = 0") {
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
    const int value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test utils::wrapToRange<int> - low != 0") {
  // Interval to wrap to [low, high).
  const int LOW = 5;
  const int HIGH = 10;

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<int, NUM_TESTS> valuesTest = {7, 0, 11, -1};
  const std::array<int, NUM_TESTS> valuesExpected = {7, 5, 6, 9};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == valuesExpected[i]);
  }
}

TEST_CASE("Test utils::wrapToRange<float> - low = 0") {
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
    const float value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test utils::wrapToRange<float> - low != 0") {
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
    const float value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test utils::wrapToRange<double> - low = 0") {
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
    const double value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test utils::wrapToRange<double> - low != 0") {
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
    const double value = utils::wrapToRange(valuesTest[i], LOW, HIGH);
    CHECK(value == doctest::Approx(valuesExpected[i]).epsilon(TOLERANCE));
  }
}

// ======================= Test utils::rotate ======================= //

TEST_CASE("Test utils:rotate<float>") {
  // Relative tolerance.
  const float TOLERANCE = std::numeric_limits<float>::epsilon();

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<float, NUM_TESTS> anglesTest = {0.0F, 0.6543F, -4.67F,
                                                   7.3456F};

  // The vector we will rotate.
  const float angleBase = 1.457F;
  const float xBase = cos(angleBase);
  const float yBase = sin(angleBase);

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const float angle = anglesTest[i];
    float x;
    float y;
    utils::rotate(angle, xBase, yBase, x, y);

    const float xExpected = cos(angle + angleBase);
    const float yExpected = sin(angle + angleBase);
    CHECK(x == doctest::Approx(xExpected).epsilon(TOLERANCE));
    CHECK(y == doctest::Approx(yExpected).epsilon(TOLERANCE));
  }
}

TEST_CASE("Test utils:rotate<double>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double TOLERANCE = 10.0 * std::numeric_limits<double>::epsilon();

  // Define test values.
  const int NUM_TESTS = 4;
  const std::array<double, NUM_TESTS> anglesTest = {0.0, 6.36, -1.67, 76.5};

  // The vector we will rotate.
  const double angleBase = 5.91;
  const double xBase = cos(angleBase);
  const double yBase = sin(angleBase);

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const double angle = anglesTest[i];
    double x;
    double y;
    utils::rotate(angle, xBase, yBase, x, y);

    const double xExpected = cos(angle + angleBase);
    const double yExpected = sin(angle + angleBase);
    CHECK(x == doctest::Approx(xExpected).epsilon(TOLERANCE));
    CHECK(y == doctest::Approx(yExpected).epsilon(TOLERANCE));
  }
}

// ======================== Test utils::sign ======================== //

TEST_CASE("Test utils:sign<int>") {
  // Define test values.
  const int NUM_TESTS = 5;
  const std::array<int, NUM_TESTS> valuesTest = {0, 2, 17, -44, -9};
  const std::array<int, NUM_TESTS> valuesExpected = {0, 1, 1, -1, -1};

  // Perform tests.
  for (int i = 0; i < NUM_TESTS; ++i) {
    const int value = utils::sign(valuesTest[i]);
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
    const int value = utils::sign(valuesTest[i]);
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
    const int value = utils::sign(valuesTest[i]);
    CHECK(value == valuesExpected[i]);
  }
}
