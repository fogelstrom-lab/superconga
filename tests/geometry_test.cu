#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "ComputeCurrent.h"
#include "ComputeFreeEnergy.h"
#include "ComputeOrderParameter.h"
#include "Context.h"
#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "GridData.h"
#include "OrderParameter.h"
#include "Parameters.h"
#include "PolygonGeometry.h"
#include "Type.h"
#include <array>
#include <cmath>
#include <doctest/doctest.h>
#include <limits>

namespace helper {
template <typename T> void testZeroValuesOutsideDomain(const bool inIsDisc) {
  // =================== ARRANGE ==================== //
  // Create Context.
  scg::Context<T> *context = new scg::Context<T>();

  // Create parameters, and set the only value the geometruGroup needs.
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
  const int gridResolution = 32;
  parameters->setGridResolution(gridResolution);

  // Create a geometry group.
  scg::GeometryGroup<T> *geometryGroup = new scg::GeometryGroup<T>(context);
  if (inIsDisc) {
    // Add two disconnected discs.
    const T diameter = static_cast<T>(0.25);
    const T centerX = static_cast<T>(0.25);
    const T centerY = static_cast<T>(0.25);
    geometryGroup->add(new scg::DiscGeometry<T>(diameter, centerX, centerY),
                       scg::Type::add);
    geometryGroup->add(new scg::DiscGeometry<T>(diameter, -centerX, -centerY),
                       scg::Type::add);
  } else {
    const int numVerticesSquare = 4;
    const T holeScale = static_cast<T>(0.25);
    // Add a square.
    geometryGroup->add(new scg::PolygonGeometry<T>(numVerticesSquare),
                       scg::Type::add);
    // Remove a square in the middle.
    geometryGroup->add(
        new scg::PolygonGeometry<T>(numVerticesSquare, holeScale),
        scg::Type::remove);
  }

  // Initialize the geometry group, i.e. compute domain labels.
  context->initialize();

  // Get and set the dimensions of our grids.
  const int numFields = 1;
  const scg::Type::type representation = scg::Type::real;

  // Create the test grid (on the CPU).
  scg::GridData<T, thrust::host_vector<T>> testGridCPU(
      gridResolution, gridResolution, numFields, representation);

  // Create the expected grid (on the CPU).
  scg::GridData<T, thrust::host_vector<T>> expectedGridCPU(
      gridResolution, gridResolution, numFields, representation);

  // An extra scope so we can reuse variable names.
  {
    // Copy the domain labels to the CPU.
    const scg::GridData<char, thrust::host_vector<char>> domainLabels =
        *(geometryGroup->getDomainGrid());

    // Get the pointers to the test grid, expected grid, and the domain labels.
    T *const testPtrCPU = testGridCPU.getDataPointer();
    T *const expPtrCPU = expectedGridCPU.getDataPointer();
    const char *const domainLabelsPtr = domainLabels.getDataPointer();

    // Set the test and expected grids.
    for (int y = 0; y < gridResolution; ++y) {
      for (int x = 0; x < gridResolution; ++x) {
        const int idx = y * gridResolution + x;
        const T val = static_cast<T>(x * y);
        testPtrCPU[idx] = val;
        const char nodeType = domainLabelsPtr[idx];
        expPtrCPU[idx] = nodeType > 0 ? val : static_cast<T>(0);
      }
    }
  }

  // =================== ACT ==================== //
  // Copy test grid to the GPU.
  scg::GridData<T, thrust::device_vector<T>> gridGPU = testGridCPU;

  // Zero values outside of the grain.
  geometryGroup->zeroValuesOutsideDomainRef(&gridGPU);

  // Copy back to the CPU.
  testGridCPU = gridGPU;

  // =================== ASSERT ==================== //
  // An extra scope so we can reuse variable names.
  {
    T *const testPtrCPU = testGridCPU.getDataPointer();
    T *const expPtrCPU = expectedGridCPU.getDataPointer();
    const T tolerance = std::numeric_limits<T>::epsilon();
    for (int y = 0; y < gridResolution; ++y) {
      for (int x = 0; x < gridResolution; ++x) {
        const int idx = y * gridResolution + x;
        const T expectedValue = expPtrCPU[idx];
        const T testValue = testPtrCPU[idx];

        CHECK(testValue == doctest::Approx(expectedValue).epsilon(tolerance));
      }
    }
  }

  // Cleanup.
  delete context;
}
} // namespace helper

// ================= Test zeroValuesOutsideDomainRef disc ================= //
TEST_CASE("Test zeroValuesOutsideDomainRef disc (float)") {
  const bool isDisc = true;
  helper::testZeroValuesOutsideDomain<float>(isDisc);
}

TEST_CASE("Test zeroValuesOutsideDomainRef disc (double)") {
  const bool isDisc = true;
  helper::testZeroValuesOutsideDomain<double>(isDisc);
}

// ================= Test zeroValuesOutsideDomainRef square ================= //
TEST_CASE("Test zeroValuesOutsideDomainRef square (float)") {
  const bool isDisc = false;
  helper::testZeroValuesOutsideDomain<float>(isDisc);
}

TEST_CASE("Test zeroValuesOutsideDomainRef square (double)") {
  const bool isDisc = false;
  helper::testZeroValuesOutsideDomain<double>(isDisc);
}
