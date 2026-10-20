#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/wrappers/doctest.h"

#include <array>
#include <cmath>
#include <limits>

namespace helper {
template <typename T> void testZeroValuesOutsideDomain(const bool inIsDisc) {
  // =================== ARRANGE ==================== //
  // Create Context.
  conga::Context<T> context;

  // Create parameters, and set the only value the geometruGroup needs.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);

  // Set grain sidelength in coherence lengths: hbar*<vF>/(kB*Tc).
  const T grainWidth = static_cast<T>(10);
  parameters->setGrainWidth(grainWidth);

  // Set the number of lattice points per coherence length:
  const T pointsPerCoherenceLength = static_cast<T>(3.5);
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Create a geometry group.
  conga::GeometryGroup<T> *geometryGroup =
      new conga::GeometryGroup<T>(&context);
  const T grainFracton = parameters->getGrainFraction();
  if (inIsDisc) {
    // Add two disconnected discs.
    const T radius = static_cast<T>(0.125);
    const T centerX = static_cast<T>(0.25);
    const T centerY = static_cast<T>(0.25);
    geometryGroup->add(
        new conga::DiscGeometry<T>(grainFracton, radius, centerX, centerY),
        conga::Type::add);
    geometryGroup->add(
        new conga::DiscGeometry<T>(grainFracton, radius, -centerX, -centerY),
        conga::Type::add);
  } else {
    const int numVerticesSquare = 4;
    const T holeScale = static_cast<T>(0.25);
    // Add a square.
    geometryGroup->add(
        new conga::PolygonGeometry<T>(grainFracton, numVerticesSquare),
        conga::Type::add);
    // Remove a square in the middle.
    geometryGroup->add(new conga::PolygonGeometry<T>(
                           grainFracton, numVerticesSquare, holeScale),
                       conga::Type::remove);
  }

  // Initialize the geometry group, i.e. compute domain labels.
  context.initialize();

  // Get and set the dimensions of our grids.
  const int numFields = 1;
  const int gridResolution = parameters->getGridResolutionBase();
  const conga::Type::type representation = conga::Type::real;

  // Create the test grid (on the CPU).
  conga::GridCPU<T> testGridCPU(gridResolution, gridResolution, numFields,
                                representation);

  // Create the expected grid (on the CPU).
  conga::GridCPU<T> expectedGridCPU(gridResolution, gridResolution, numFields,
                                    representation);

  // An extra scope so we can reuse variable names.
  {
    // Copy the domain labels to the CPU.
    const conga::GridCPU<char> domainLabels = *(geometryGroup->getDomainGrid());

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
  conga::GridGPU<T> gridGPU = testGridCPU;

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
}

template <typename T> void testArea(const bool inIsDisc) {
  // =================== ARRANGE ==================== //
  // Create Context.
  conga::Context<T> context;

  // Create parameters, and set the only value the geometruGroup needs.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);

  // Set grain sidelength in coherence lengths: hbar*<vF>/(kB*Tc).
  const T grainWidth = static_cast<T>(1);
  parameters->setGrainWidth(grainWidth);

  // Set the number of lattice points per coherence length:
  const T pointsPerCoherenceLength = static_cast<T>(1e3);
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Create a geometry group.
  conga::GeometryGroup<T> *geometryGroup =
      new conga::GeometryGroup<T>(&context);
  const T grainFracton = parameters->getGrainFraction();
  if (inIsDisc) {
    // Add a disc.
    geometryGroup->add(new conga::DiscGeometry<T>(grainFracton),
                       conga::Type::add);
  } else {
    // Add a square.
    const int numVerticesSquare = 4;
    const T sideLength = grainWidth;
    geometryGroup->add(new conga::PolygonGeometry<T>(
                           grainFracton, numVerticesSquare, sideLength),
                       conga::Type::add);
  }

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Initialize the geometry group, i.e. compute domain labels.
  context.initialize();

  // =================== ACT ==================== //
  const T area = geometryGroup->area();

  // =================== ASSERT ==================== //
  const T grainArea = grainWidth * grainWidth;
  const T expectedArea =
      grainArea * (inIsDisc ? static_cast<T>(0.25 * M_PI) : static_cast<T>(1));
  // The half the ratio, circumference / area, for a square.
  const T tolerance = static_cast<T>(2) / pointsPerCoherenceLength;
  CHECK(expectedArea == doctest::Approx(area).epsilon(tolerance));
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

// ================= Test area disc ================= //

TEST_CASE("Test area disc (float)") {
  const bool isDisc = true;
  helper::testArea<float>(isDisc);
}

TEST_CASE("Test area disc (double)") {
  const bool isDisc = true;
  helper::testArea<double>(isDisc);
}

// ================= Test area square ================= //

TEST_CASE("Test area square (float)") {
  const bool isDisc = false;
  helper::testArea<float>(isDisc);
}

TEST_CASE("Test area square (double)") {
  const bool isDisc = false;
  helper::testArea<double>(isDisc);
}
