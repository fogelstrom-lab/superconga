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

// A helper function to test that the s-wave bulk free energy gives
// the expected value.
template <typename T>
void testComputeFreeEnergyBulkSWave(const T inTolerance, const T inTemperature,
                                    const T inFreeEnergyExpected) {
  // =================== ARRANGE ==================== //
  const T orderParameter =
      internal::computeOrderParameterBulkSWave(inTemperature);

  // =================== ACT ==================== //
  const T freeEnergy =
      internal::computeFreeEnergyBulkSWave(inTemperature, orderParameter);

  // =================== ASSERT ==================== //
  CHECK(freeEnergy ==
        doctest::Approx(inFreeEnergyExpected).epsilon(inTolerance));
}

// A helper function to test that the s-wave bulk order parameter and free
// energy are consistent with each other. For a certain temperature, compute the
// order parameter, then check that the free energy has a minimum at that order
// parameter and temperature.
template <typename T>
void testComputeFreeEnergyBulkSWaveMinima(
    const T inMinTemperature = static_cast<T>(0.1),
    const T inMaxTemperature = static_cast<T>(0.9),
    const int inNumTemperatures = 10) {
  // =================== ARRANGE ==================== //
  const std::vector<T> temperatures =
      utils::linspace(inMinTemperature, inMaxTemperature, inNumTemperatures);

  // A small value to add to the order parameter.
  const T epsilon = static_cast<T>(1e-3);

  for (const T temperature : temperatures) {
    const T orderParameter =
        internal::computeOrderParameterBulkSWave(temperature);

    // =================== ACT ==================== //
    const T freeEnergy =
        internal::computeFreeEnergyBulkSWave(temperature, orderParameter);

    // =================== ASSERT ==================== //
    const T freeEnergyPlus = internal::computeFreeEnergyBulkSWave(
        temperature, orderParameter + epsilon);
    const T freeEnergyMinus = internal::computeFreeEnergyBulkSWave(
        temperature, orderParameter - epsilon);
    CHECK(freeEnergy < freeEnergyPlus);
    CHECK(freeEnergy < freeEnergyMinus);
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

// ============== Test computeFreeEnergyBulkSWave - T = 0 ============== //

TEST_CASE("Test computeFreeEnergyBulkSWave<float> - T = 0") {
  const float tolerance = 1e-6F;
  const float temperature = 0.001F;
  const float freeEnergyExpected = -0.5F *
                                   static_cast<float>(qcs::S_WAVE_BULK) *
                                   static_cast<float>(qcs::S_WAVE_BULK);
  helper::testComputeFreeEnergyBulkSWave(tolerance, temperature,
                                         freeEnergyExpected);
}

TEST_CASE("Test computeFreeEnergyBulkSWave<double> - T = 0") {
  const double tolerance = 1e-6;
  const double temperature = 0.001;
  const double freeEnergyExpected = -0.5 * qcs::S_WAVE_BULK * qcs::S_WAVE_BULK;
  helper::testComputeFreeEnergyBulkSWave(tolerance, temperature,
                                         freeEnergyExpected);
}

// ============== Test computeFreeEnergyBulkSWave - T = 1 ============== //

TEST_CASE("Test computeFreeEnergyBulkSWave<float> - T = 1") {
  const float tolerance = 1e-6F;
  const float temperature = 1.0F;
  const float freeEnergyExpected = 0.0F;
  helper::testComputeFreeEnergyBulkSWave(tolerance, temperature,
                                         freeEnergyExpected);
}

TEST_CASE("Test computeFreeEnergyBulkSWave<double> - T = 1") {
  const double tolerance = 1e-6;
  const double temperature = 1.0;
  const double freeEnergyExpected = 0.0;
  helper::testComputeFreeEnergyBulkSWave(tolerance, temperature,
                                         freeEnergyExpected);
}

// ============== Test computeFreeEnergyBulkSWave - Minima ============== //

TEST_CASE("Test computeFreeEnergyBulkSWave<float> - Minima") {
  helper::testComputeFreeEnergyBulkSWaveMinima<float>();
}

TEST_CASE("Test computeFreeEnergyBulkSWave<double> - Minima") {
  helper::testComputeFreeEnergyBulkSWaveMinima<double>();
}
