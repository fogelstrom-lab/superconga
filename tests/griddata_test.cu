#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "Grid.h"
#include "GridData.h"
#include "Type.h"
#include "utils.h"
#include <cmath>
#include <doctest.h>

// Typedefs for Grid objects (wrapper for GridData objects)
typedef typename Grid<float>::GPU floatGridGPU;
typedef typename Grid<float>::CPU floatGridCPU;
typedef typename Grid<double>::GPU doubleGridGPU;
typedef typename Grid<double>::CPU doubleGridCPU;

namespace helper {
// ============ Constants used in multiple tests ============ //
// Sizes for GridData objects
const int NUM_GRID_SIZES = 3;
const std::array<int, NUM_GRID_SIZES> GRID_SIZES = {4, 100, 1000};

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
template <typename T, typename DEVICE>
void testSumMagnitudeComplex(const int inNumValues = 20,
                             const T inMinValue = static_cast<T>(1e-3),
                             const T inMaxValue = static_cast<T>(1e3),
                             const int inNumFields = 1) {

  for (const int &gridSize : GRID_SIZES) {
    // =================== ARRANGE ==================== //
    // Size of GridData array is (numX*numY)*numFields
    const int numX = gridSize;
    const int numY = gridSize;

    // Calculate relative tolerance.
    const T tolerance = computeTolerance<T>(numX, numY);

    // Construct a GridData object of (numX*numY) complex numbers, with
    // inNumFields number of components
    DEVICE testComplexGrid(numX, numY, inNumFields, Type::complex);

    // Stepsize between each different value to try
    const T valueStep =
        (inMaxValue - inMinValue) / static_cast<T>(inNumValues - 1);

    // Vary the real part
    for (int realPartIdx = 0; realPartIdx < inNumValues; ++realPartIdx) {
      // Update real part
      const T realPartValue =
          valueStep * static_cast<T>(realPartIdx) + inMinValue;
      testComplexGrid.setValue(realPartValue, Type::real);

      // Vary the imaginary part
      for (int imaginaryPartIdx = 0; imaginaryPartIdx < inNumValues;
           ++imaginaryPartIdx) {
        // Update imaginary part
        const T imaginaryPartValue =
            valueStep * static_cast<T>(imaginaryPartIdx) + inMinValue;
        testComplexGrid.setValue(imaginaryPartValue, Type::imag);

        // =================== ACT ==================== //
        // Calculate the grid-summed magnitude with the GridData method
        const T testResult = testComplexGrid.sumMagnitude();
        // Calculate the expected value by hand
        const T expectedResult = static_cast<T>(numX * numY) *
                                 std::hypot(realPartValue, imaginaryPartValue);

        // =================== ASSERT ==================== //
        CHECK(testResult == doctest::Approx(expectedResult).epsilon(tolerance));
      }
    }
  }
}

// A helper function to test the GridData::sumMagnitude function for a real
// field. Instantiate a GridData object and check that the grid-summed magnitude
// is correct. Every element is set to the same value, ranging from inMinValue
// to inMaxValue.
template <typename T, typename DEVICE>
void testSumMagnitudeReal(const int inNumValues = 20, const T inMinValue = 1e-3,
                          const T inMaxValue = 1e3, const int inNumFields = 1) {

  for (const int &gridSize : GRID_SIZES) {
    // =================== ARRANGE ==================== //
    // Size of GridData array is (numX*numY)*numFields
    const int numX = gridSize;
    const int numY = gridSize;

    // Calculate relative tolerance.
    const T tolerance = computeTolerance<T>(numX, numY);

    // Construct a GridData object of (numX*numY) real numbers, with
    // inNumFields number of components
    DEVICE testRealGrid(numX, numY, inNumFields, Type::real);

    // Stepsize between each different value to try
    const T valueStep =
        (inMaxValue - inMinValue) / static_cast<T>(inNumValues - 1);

    // Vary the test value
    for (int valueIdx = 0; valueIdx < inNumValues; ++valueIdx) {
      // Update the test value
      const T testValue = valueStep * static_cast<T>(valueIdx) + inMinValue;
      testRealGrid.setValue(testValue, Type::real);

      // =================== ACT ==================== //
      // Calculate the grid-summed magnitude with the GridData method
      const T testResult = testRealGrid.sumMagnitude();
      // Calculate the expected value by hand
      const T expectedResult =
          static_cast<T>(numX * numY) * std::abs(testValue);

      // =================== ASSERT ==================== //
      CHECK(testResult == doctest::Approx(expectedResult).epsilon(tolerance));
    }
  }
}

} // namespace helper

// =============== Test GridData::sumMagnitude =============== //
TEST_CASE("Test GridData<float>::sumMagnitude - Complex, GPU") {
  helper::testSumMagnitudeComplex<float, floatGridGPU>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Complex, CPU") {
  helper::testSumMagnitudeComplex<float, floatGridCPU>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Real, GPU") {
  helper::testSumMagnitudeReal<float, floatGridGPU>();
}

TEST_CASE("Test GridData<float>::sumMagnitude - Real, CPU") {
  helper::testSumMagnitudeReal<float, floatGridCPU>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Complex, GPU") {
  helper::testSumMagnitudeComplex<double, doubleGridGPU>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Complex, CPU") {
  helper::testSumMagnitudeComplex<double, doubleGridCPU>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Real, GPU") {
  helper::testSumMagnitudeReal<double, doubleGridGPU>();
}

TEST_CASE("Test GridData<double>::sumMagnitude - Real, CPU") {
  helper::testSumMagnitudeReal<double, doubleGridCPU>();
}
