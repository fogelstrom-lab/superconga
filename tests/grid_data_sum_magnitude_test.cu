#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Type.h"
#include "conga/grid.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <cmath>

namespace helper {
// ============ Constants used in multiple tests ============ //
// Max dimension for real field is 2 (vector field)
const int MAX_NUM_FIELDS_REAL = 2;
// Max dimension for complex field can be anything > 0
const int MAX_NUM_FIELDS_COMPLEX = 3;
// Size for GridData objects
const int GRID_SIZE = 4;

// Compute tolerance for GridData testing, given the number of elements in a
// GridData object. Note: inToleranceBaseFactor has to be > 1.
template <typename T>
T computeTolerance(const int inNumX, const int inNumY,
                   const T inToleranceBaseFactor = static_cast<T>(5.0)) {
  return inToleranceBaseFactor * static_cast<T>(inNumX * inNumY) *
         std::numeric_limits<T>::epsilon();
}

// A helper function to test the GridData::sumMagnitude function for a complex
// field. Instantiate a GridData object and check that the grid-summed magnitude
// is correct. Every element is set to the same value, ranging from inMinValue
// to inMaxValue.
template <typename T, typename Grid>
void testSumMagnitudeComplex(const int inNumValues = 4,
                             const T inMinValue = static_cast<T>(1e-3),
                             const T inMaxValue = static_cast<T>(1e3)) {
  // =================== ARRANGE ==================== //
  // Size of GridData array is (numX*numY)*numFields
  const int numX = GRID_SIZE;
  const int numY = GRID_SIZE;

  // Calculate relative tolerance.
  const T tolerance = computeTolerance<T>(numX, numY);

  // Loop over number of fields. Each field will have the same values.
  for (int numFields = 1; numFields <= MAX_NUM_FIELDS_COMPLEX; ++numFields) {

    // Construct a GridData object of (numX*numY) complex numbers, with
    // numFields number of components
    Grid testComplexGrid(numX, numY, numFields, conga::Type::complex);

    // Stepsize between each different value to try
    const T valueStep =
        (inMaxValue - inMinValue) / static_cast<T>(inNumValues - 1);

    // Vary the real part
    for (int realPartIdx = 0; realPartIdx < inNumValues; ++realPartIdx) {
      // Update real part
      const T realPartValue =
          valueStep * static_cast<T>(realPartIdx) + inMinValue;

      for (int fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) {
        testComplexGrid.setValue(realPartValue, conga::Type::real, fieldIdx);
      }

      // Vary the imaginary part
      for (int imaginaryPartIdx = 0; imaginaryPartIdx < inNumValues;
           ++imaginaryPartIdx) {
        // Update imaginary part
        const T imaginaryPartValue =
            valueStep * static_cast<T>(imaginaryPartIdx) + inMinValue;

        for (int fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) {
          testComplexGrid.setValue(imaginaryPartValue, conga::Type::imag,
                                   fieldIdx);
        }

        // =================== ACT ==================== //
        // Calculate the grid-summed magnitude with the GridData method
        const T testResult = testComplexGrid.sumMagnitude();
        const T testResultL1Norm = testComplexGrid.L1Norm();
        // Calculate the expected value by hand
        const T expectedResult = static_cast<T>(numX * numY * numFields) *
                                 std::hypot(realPartValue, imaginaryPartValue);
        const T expectedResultL1Norm =
            static_cast<T>(numX * numY * numFields) *
            (std::abs(realPartValue) + std::abs(imaginaryPartValue));

        // =================== ASSERT ==================== //
        CHECK(testResult == doctest::Approx(expectedResult).epsilon(tolerance));
        CHECK(testResultL1Norm ==
              doctest::Approx(expectedResultL1Norm).epsilon(tolerance));
      }
    }
  }
}

// A helper function to test the GridData::sumMagnitude function for a real
// field. Instantiate a GridData object and check that the grid-summed magnitude
// is correct. Every element is set to the same value, ranging from inMinValue
// to inMaxValue.
template <typename T, typename Grid>
void testSumMagnitudeReal(const int inNumValues = 4, const T inMinValue = 1e-3,
                          const T inMaxValue = 1e3) {
  // =================== ARRANGE ==================== //
  // Size of GridData array is (numX*numY)*numFields
  const int numX = GRID_SIZE;
  const int numY = GRID_SIZE;

  // Calculate relative tolerance.
  const T tolerance = computeTolerance<T>(numX, numY);

  // Loop over number of fields. Each field will have the same values.
  for (int numFields = 1; numFields <= MAX_NUM_FIELDS_REAL; ++numFields) {

    // Construct a GridData object of (numX*numY) real numbers, with
    // numFields number of components
    Grid testRealGrid(numX, numY, numFields, conga::Type::real);

    // Stepsize between each different value to try
    const T valueStep =
        (inMaxValue - inMinValue) / static_cast<T>(inNumValues - 1);

    // Vary the test value
    for (int valueIdx = 0; valueIdx < inNumValues; ++valueIdx) {
      // Update the test value
      const T testValue = valueStep * static_cast<T>(valueIdx) + inMinValue;

      for (int fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) {
        testRealGrid.setValue(testValue, conga::Type::real, fieldIdx);
      }

      // =================== ACT ==================== //
      // Calculate the grid-summed magnitude with the GridData method
      const T testResult = testRealGrid.sumMagnitude();
      const T testResultL1Norm = testRealGrid.L1Norm();

      // Calculate the expected value by hand
      const T expectedResult =
          static_cast<T>(numX * numY) *
          std::sqrt(static_cast<T>(numFields) * testValue * testValue);
      const T expectedResultL1Norm =
          static_cast<T>(numX * numY * numFields) * std::abs(testValue);

      // =================== ASSERT ==================== //
      CHECK(testResult == doctest::Approx(expectedResult).epsilon(tolerance));
      CHECK(testResultL1Norm ==
            doctest::Approx(expectedResultL1Norm).epsilon(tolerance));
    }
  }
}

} // namespace helper

// =============== Test GridData::sumMagnitude =============== //
TEST_CASE("Test GridData<float>::sumMagnitude - Complex, GPU") {
  helper::testSumMagnitudeComplex<float, conga::GridGPU<float>>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Complex, CPU") {
  helper::testSumMagnitudeComplex<float, conga::GridCPU<float>>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Real, GPU") {
  helper::testSumMagnitudeReal<float, conga::GridGPU<float>>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Real, CPU") {
  helper::testSumMagnitudeReal<float, conga::GridCPU<float>>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Complex, GPU") {
  helper::testSumMagnitudeComplex<double, conga::GridGPU<double>>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Complex, CPU") {
  helper::testSumMagnitudeComplex<double, conga::GridCPU<double>>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Real, GPU") {
  helper::testSumMagnitudeReal<double, conga::GridGPU<double>>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Real, CPU") {
  helper::testSumMagnitudeReal<double, conga::GridCPU<double>>();
}
