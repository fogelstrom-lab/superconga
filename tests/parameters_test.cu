#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Parameters.h"
#include "conga/wrappers/doctest.h"

#include <cmath>

namespace helper {
template <typename T> void testGridResolution() {
  // =================== ARRANGE ==================== //
  conga::Parameters<T> parameters;

  // =================== ACT ==================== //
  // Loop over some semi-random values to test.
  for (const double grainWidth : {1.0, 6.53, 25.6, 50.111}) {
    for (const double pointsPerCoherenceLength : {1.05, 3.9, 10.0, 17.8}) {
      parameters.setGrainWidth(static_cast<T>(grainWidth));
      parameters.setPointsPerCoherenceLength(
          static_cast<T>(pointsPerCoherenceLength));

      // Note, gridResolutionCoherence and gridResolutionDeltaRotated are
      // completely redundant and return the same value.
      const int testGridResolutionCoherence =
          parameters.getGridResolutionCoherence();
      const int testGridResolutionDeltaRotated =
          parameters.getGridResolutionDeltaRotated();
      const T testGridElementSize = parameters.getGridElementSize();

      // =================== ASSERT ==================== //
      const int gridResolution = parameters.getGridResolutionBase();
      const int tmp = static_cast<int>(
          std::ceil(std::sqrt(2.0) * static_cast<double>(gridResolution)));
      const int expectedGridResolutionRotated =
          (tmp - gridResolution) % 2 == 0 ? tmp : tmp + 1;
      const T expectedGridElementSize =
          static_cast<T>(1.0 / pointsPerCoherenceLength);

      CHECK(testGridResolutionCoherence == expectedGridResolutionRotated);
      CHECK(testGridResolutionDeltaRotated == expectedGridResolutionRotated);
      CHECK(testGridElementSize == doctest::Approx(expectedGridElementSize));
    }
  }
}

template <typename T> void testNumOzakiEnergies() {
  // =================== ARRANGE ==================== //
  conga::Parameters<T> parameters;
  const T temperature = static_cast<T>(0.5);
  const T cutoff = static_cast<T>(10.0 / (2.0 * M_PI));

  // =================== ACT ==================== //
  parameters.setTemperature(temperature);
  parameters.setEnergyCutoff(cutoff);
  const int numOzakiEnergiesTest = parameters.getNumOzakiEnergies();

  // =================== ASSERT ==================== //
  const int numOzakiEnergiesExpected = 3;
  CHECK(numOzakiEnergiesTest == numOzakiEnergiesExpected);
}
} // namespace helper

// ================= Test grid resolution ================= //
TEST_CASE("Test grid resolution (float)") {
  helper::testGridResolution<float>();
}

TEST_CASE("Test grid resolution (double)") {
  helper::testGridResolution<double>();
}

// ================= Test number of Ozaki energies ================= //
TEST_CASE("Test number of Ozaki energies (float)") {
  helper::testNumOzakiEnergies<float>();
}

TEST_CASE("Test number of Ozaki energies (double)") {
  helper::testNumOzakiEnergies<double>();
}
