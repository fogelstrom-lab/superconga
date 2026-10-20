#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/accelerators/BarzilaiBorwein.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/defines.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_iteration.h"
#include "conga/wrappers/doctest.h"

#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>

namespace helper {
// Which polygons to test.
// TODO(niclas): We probably have some bug. Only even polygons work. FIX
// THIS!!! Add more polygons, solenoids, irregular polygons, and disjoint
// polygons.
const std::vector<int> NUM_VERTICES = {-1, 4, 6};

// ========================= Test helper objects ========================= //

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testCurrentConservation(const conga::BasisFunctionSymmetry inSymmetry,
                             const int inNumVertices) {
  // =================== ARRANGE ==================== //
  // Create Context
  conga::Context<T> context;

  // Set parameters.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(static_cast<T>(10));
  parameters->setPointsPerCoherenceLength(static_cast<T>(2));
  parameters->setAngularResolution(32);
  parameters->setConvergenceCriterion(static_cast<T>(1e-3));
  parameters->setTemperature(static_cast<T>(0.5));
  parameters->setEnergyCutoff(static_cast<T>(16));
  parameters->setNumFluxQuanta(static_cast<T>(0.01));
  parameters->setResidualNormType(conga::ResidualNorm::LInf);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  const bool isDisc = inNumVertices < 0;
  if (isDisc) {
    geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
  } else {
    // TODO(niclas): This should live within some geometry class!
    const T angleStep =
        static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumVertices);
    const T offset = angleStep * static_cast<T>(0.5);
    T xMax = std::numeric_limits<T>::min();
    T xMin = std::numeric_limits<T>::max();
    T yMax = std::numeric_limits<T>::min();
    T yMin = std::numeric_limits<T>::max();
    for (int i = 0; i < inNumVertices; ++i) {
      const T angle = angleStep * static_cast<T>(i) + offset;
      const T x = std::cos(angle);
      const T y = std::sin(angle);
      xMax = std::max(x, xMax);
      xMin = std::min(x, xMin);
      yMax = std::max(y, yMax);
      yMin = std::min(y, yMin);
    }
    const T xExtent = xMax - xMin;
    const T yExtent = yMax - yMin;
    const T extent = std::max(xExtent, yExtent);
    const T radius = static_cast<T>(1) / extent;
    const T sideLength =
        static_cast<T>(2) * radius * std::sin(static_cast<T>(0.5) * angleStep);
    geometry->add(
        new conga::PolygonGeometry<T>(grainFraction, inNumVertices, sideLength),
        conga::Type::add);
  }

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Choose a Riccati solver for a confined geometry.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::gauge::Symmetric>(&context);
  // Add specular BC
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set up order parameter
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);

  // Add an order-parameter component.
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(inSymmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  conga::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  // Order parameter compute object.
  new conga::ComputeOrderParameter<T>(&context);
  // Add current compute object.
  new conga::ComputeCurrent<T>(&context);
  // Add free energy compute object.
  new conga::ComputeFreeEnergy<T>(&context);

  // Set accelerator.
  auto accelerator =
      std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
  context.set(std::move(accelerator));

  // =================== ACT ==================== //
  context.initialize();
  const bool isConverged = conga::runCompute(1, 100, true, context);

  // Fetch the grain-summed current-density magnitude
  const conga::GridGPU<T> currentDensity =
      *(context.getComputeCurrent()->getResult());
  // Fetch the number of elements in the grain
  const int numGrainLatticeSites = geometry->getNumGrainLatticeSites();

  // Compute the mean of the components.
  const T testMeanCurrentDensityX =
      currentDensity.sumField(0) / static_cast<T>(numGrainLatticeSites);
  const T testMeanCurrentDensityY =
      currentDensity.sumField(1) / static_cast<T>(numGrainLatticeSites);
  const T sumAbsCurrent = currentDensity.sumMagnitude();
  const T freeEnergyPerUnitArea =
      context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  std::printf("J: (x, y) = (%e, %e)\n", testMeanCurrentDensityX,
              testMeanCurrentDensityY);

  // =================== ASSERT ==================== //
  const T expectedMeanCurrentDensityX = static_cast<T>(0);
  const T expectedMeanCurrentDensityY = static_cast<T>(0);

  const T tolerance = static_cast<T>(10) * std::numeric_limits<T>::epsilon();

  CHECK(testMeanCurrentDensityX ==
        doctest::Approx(expectedMeanCurrentDensityX).epsilon(tolerance));
  CHECK(testMeanCurrentDensityY ==
        doctest::Approx(expectedMeanCurrentDensityY).epsilon(tolerance));
  CHECK(sumAbsCurrent > static_cast<T>(0));
  CHECK(freeEnergyPerUnitArea < static_cast<T>(0));
  CHECK(isConverged);
}
} // namespace helper

// ================= Test current conservation ================= //

TEST_CASE("Test current conservation - s-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - s-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : helper::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}
