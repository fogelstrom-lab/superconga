#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "Parameters.h"
#include <array>
#include <cmath>
#include <doctest/doctest.h>

namespace helper {
template <typename T> void testGridResolution() {
  // =================== ARRANGE ==================== //
  scg::Parameters<T> parameters;

  // Just some semi-random values to test.
  const std::array<int, 6> gridResolutions{64, 101, 130, 199, 392, 589};

  for (const int gridResolution : gridResolutions) {
    // =================== ACT ==================== //
    parameters.setGridResolution(gridResolution);

    // =================== ASSERT ==================== //
    const int tmp = static_cast<int>(
        std::ceil(std::sqrt(2.0) * static_cast<double>(gridResolution)));
    const int expectedGridResolutionRotated =
        (tmp - gridResolution) % 2 == 0 ? tmp : tmp + 1;
    // Note, gridResolutionCoherence and gridResolutionDeltaRotated are
    // completely redundant and return the same value.
    const int testGridResolutionCoherence =
        parameters.getGridResolutionCoherence();
    const int testGridResolutionDeltaRotated =
        parameters.getGridResolutionDeltaRotated();

    CHECK(testGridResolutionCoherence == expectedGridResolutionRotated);
    CHECK(testGridResolutionDeltaRotated == expectedGridResolutionRotated);
  }
}
} // namespace helper

// ================= Test grid resolution ================= //
TEST_CASE("Test grid resolution (float)") {
  helper::testGridResolution<float>();
}

TEST_CASE("Test grid resolution (double)") {
  helper::testGridResolution<double>();
}
