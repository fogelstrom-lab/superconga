#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "OrderParameterBulk.h"
#include "defines.h"
#include "utils.h"
#include <cmath>
#include <doctest.h>
#include <limits>
#include <vector>

namespace helper {
// A helper function to test that the s-wave bulk order parameter solver gives
// the expected, analytical, result at (almost) zero temperature.
template <typename T>
void testComputeOrderParameterBulkSWaveZeroTemperature(
    const T inTolerance, const T inConvergenceCriteria) {
  // =================== ARRANGE ==================== //
  const T temperature = static_cast<T>(0.001);

  // =================== ACT ==================== //
  const T orderParameter = internal::computeOrderParameterBulkSWave(
      temperature, inConvergenceCriteria);

  // =================== ASSERT ==================== //
  const T orderParameterExpected = static_cast<T>(qcs::S_WAVE_BULK);
  CHECK(orderParameter ==
        doctest::Approx(orderParameterExpected).epsilon(inTolerance));
}

// A helper function to test that the s-wave bulk order parameter solver gives
// the expected, analytical, result at the critical temperature.
template <typename T>
void testComputeOrderParameterBulkSWaveCriticalTemperature(
    const T inTolerance, const T inConvergenceCriteria) {
  // =================== ARRANGE ==================== //
  const T temperature = static_cast<T>(1.0);

  // =================== ACT ==================== //
  const T orderParameter = internal::computeOrderParameterBulkSWave(
      temperature, inConvergenceCriteria);

  // =================== ASSERT ==================== //
  const T orderParameterExpected = static_cast<T>(0.0);
  CHECK(orderParameter ==
        doctest::Approx(orderParameterExpected).epsilon(inTolerance));
}

// A helper function to test that the s-wave bulk order parameter solver gives
// approximately the right answer close to the critical temperature.
template <typename T>
void testComputeOrderParameterBulkSWaveIntermediateTemperature(
    const T inTolerance, const T inConvergenceCriteria,
    const T inMinTemperature = static_cast<T>(0.99),
    const T inMaxTemperature = static_cast<T>(0.999),
    const int inNumTemperatures = 10) {
  // =================== ARRANGE ==================== //
  const std::vector<T> temperatures =
      utils::linspace(inMinTemperature, inMaxTemperature, inNumTemperatures);

  // See page 180 of P. Holmvall lic thesis.
  const T preFactor = std::sqrt(static_cast<T>(8.0 * M_PI * M_PI) /
                                static_cast<T>(7.0 * qcs::APERY));

  for (const T &temperature : temperatures) {
    // =================== ACT ==================== //
    const T orderParameter = internal::computeOrderParameterBulkSWave(
        temperature, inConvergenceCriteria);

    // =================== ASSERT ==================== //
    // See page 180 of P. Holmvall lic thesis.
    // Note that this apprixmation is only valid very close to the critical
    // temperature.
    const T orderParameterExpected =
        preFactor * std::sqrt(static_cast<T>(1.0) - temperature);
    CHECK(orderParameter ==
          doctest::Approx(orderParameterExpected).epsilon(inTolerance));
  }
}
} // namespace helper

// ============== Test computeOrderParameterBulkSWave - T = 0 ============== //

TEST_CASE("Test computeOrderParameterBulkSWave<float> - T = 0") {
  const float tolerance = 1e-5F;

  const float convergenceCriteria =
      10.0F * std::numeric_limits<float>::epsilon();

  helper::testComputeOrderParameterBulkSWaveZeroTemperature(
      tolerance, convergenceCriteria);
}

TEST_CASE("Test computeOrderParameterBulkSWave<double> - T = 0") {
  const double tolerance = 1e-6;

  const double convergenceCriteria =
      10.0 * std::numeric_limits<double>::epsilon();

  helper::testComputeOrderParameterBulkSWaveZeroTemperature(
      tolerance, convergenceCriteria);
}

// ============== Test computeOrderParameterBulkSWave - T = 1 ============== //

TEST_CASE("Test computeOrderParameterBulkSWave<float> - T = 1") {
  const float tolerance = 1e-6F;

  const float convergenceCriteria =
      10.0F * std::numeric_limits<float>::epsilon();

  helper::testComputeOrderParameterBulkSWaveCriticalTemperature(
      tolerance, convergenceCriteria);
}

TEST_CASE("Test computeOrderParameterBulkSWave<double> - T = 1") {
  const double tolerance = 1e-6;

  const double convergenceCriteria =
      10.0 * std::numeric_limits<double>::epsilon();

  helper::testComputeOrderParameterBulkSWaveCriticalTemperature(
      tolerance, convergenceCriteria);
}

// ============== Test computeOrderParameterBulkSWave - T < 1 ============== //

TEST_CASE("Test computeOrderParameterBulkSWave<float> - T < 1") {
  const float minTemperature = 0.99F;
  const float maxTemperature = 0.999F;

  // Note, that we are testing temperatures quite far from Tc. Hence the large
  // tolerance.
  const float tolerance = 0.002F;

  const float convergenceCriteria =
      10.0F * std::numeric_limits<float>::epsilon();

  helper::testComputeOrderParameterBulkSWaveIntermediateTemperature(
      tolerance, convergenceCriteria, minTemperature, maxTemperature);
}

TEST_CASE("Test computeOrderParameterBulkSWave<double> - T < 1") {
  const double minTemperature = 0.99;
  const double maxTemperature = 0.999;

  // Note, that we are testing temperatures quite far from Tc. Hence the large
  // tolerance.
  const double tolerance = 0.001;

  const double convergenceCriteria =
      10.0 * std::numeric_limits<double>::epsilon();

  helper::testComputeOrderParameterBulkSWaveIntermediateTemperature(
      tolerance, convergenceCriteria, minTemperature, maxTemperature);
}
