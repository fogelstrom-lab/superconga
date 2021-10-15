#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h"
#include "ComputeCurrent.h"
#include "ComputeFreeEnergy.h"
#include "ComputeOrderParameter.h"
#include "Context.h"
#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "OrderParameter.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h"
#include "Parameters.h"
#include "PolygonGeometry.h"
#include "RiccatiSolverConfined.h"
#include "defines.h"
#include "doctest_wrapper.h"
#include "run_utils.h"

#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.5;

// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 5.0;

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 32;

// The energy cutoff.
const double ENERGY_CUTOFF = 16.0;

// Convergence criterion for order-parameter residual
const double CONVERGENCE_CRITERION = 1e-5;

// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = true;

// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 100;

// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 1;

// The external magnetic field.
const double EXTERNAL_FLUX_QUANTA = 0.01;

// Which polygons to test.
// TODO(niclas): We probably have some bug. Only even polygons work. FIX
// THIS!!! Add more polygons, solenoids, irregular polygons, and disjoint
// polygons.
const std::vector<int> NUM_VERTICES = {-1, 4, 6};

// The tolerance is this factor times the machine precision.
const double EPSILON_TOLERANCE = 10.0;
} // namespace simParams

namespace helper {
// ========================= Test helper objects ========================= //

// Helper class for setting up simulation parameters
template <typename T>
conga::Parameters<T> *applySimParams(conga::Context<T> *inOutContext) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  conga::Parameters<T> *parameters = new conga::Parameters<T>(inOutContext);

  // Set grain sidelength in coherence lengths: hbar*<vF>/(kB*Tc).
  parameters->setGrainWidth(static_cast<T>(simParams::GRAIN_SIDE_LENGTH));

  // Set the number of lattice points per coherence length.
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(simParams::POINTS_PER_COHERENCE_LENGTH));

  // Number of discrete steps in momentum angle
  parameters->setAngularResolution(simParams::NUM_ANGLES);

  // Set convergence criterion
  parameters->setConvergenceCriterion(
      static_cast<T>(simParams::CONVERGENCE_CRITERION));

  // Set temperature
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));

  // Set number of Ozaki poles
  parameters->setEnergyCutoff(static_cast<T>(simParams::ENERGY_CUTOFF));

  // External flux-quanta.
  parameters->setNumFluxQuanta(static_cast<T>(simParams::EXTERNAL_FLUX_QUANTA));

  return parameters;
}

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testCurrentConservation(const conga::BasisFunctionSymmetry inSymmetry,
                             const int inNumVertices) {
  // =================== ARRANGE ==================== //
  // Create Context
  conga::Context<T> *context = new conga::Context<T>();

  // Set parameters to the values specified in simParams struct
  conga::Parameters<T> *parameters = applySimParams<T>(context);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(context);
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

  // Choose a Riccati solver for a confined geometry.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
          context);
  // Add specular BC
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set up order parameter
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(context);

  // Add an order-parameter component.
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(inSymmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  conga::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  // Order parameter compute object.
  new conga::ComputeOrderParameter<T>(context);
  // Add current compute object.
  new conga::ComputeCurrent<T>(context);
  // Add free energy compute object.
  new conga::ComputeFreeEnergy<T>(context);

  // Set accelerator.
  context->set(std::make_unique<conga::Accelerator<T>>());

  // =================== ACT ==================== //
  context->initialize();

  conga::run_utils::runSimulation(context, simParams::PRINT_ITERATION_STATUS,
                                  simParams::NUM_ITERATIONS_MIN,
                                  simParams::NUM_ITERATIONS_MAX);

  // Fetch the grain-summed current-density magnitude
  const conga::GridGPU<T> currentDensity =
      *(context->getComputeCurrent()->getResult());
  // Fetch the number of elements in the grain
  const int numGrainLatticeSites = geometry->getNumGrainLatticeSites();

  // Compute the mean of the components.
  const T testMeanCurrentDensityX =
      currentDensity.sumField(0) / static_cast<T>(numGrainLatticeSites);
  const T testMeanCurrentDensityY =
      currentDensity.sumField(1) / static_cast<T>(numGrainLatticeSites);
  const T sumAbsCurrent = currentDensity.sumMagnitude();
  const T freeEnergyPerUnitArea =
      context->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  std::printf("J: (x, y) = (%e, %e)\n", testMeanCurrentDensityX,
              testMeanCurrentDensityY);

  // =================== ASSERT ==================== //
  const T expectedMeanCurrentDensityX = static_cast<T>(0);
  const T expectedMeanCurrentDensityY = static_cast<T>(0);

  const T tolerance = static_cast<T>(simParams::EPSILON_TOLERANCE) *
                      std::numeric_limits<T>::epsilon();

  CHECK(testMeanCurrentDensityX ==
        doctest::Approx(expectedMeanCurrentDensityX).epsilon(tolerance));
  CHECK(testMeanCurrentDensityY ==
        doctest::Approx(expectedMeanCurrentDensityY).epsilon(tolerance));
  CHECK(sumAbsCurrent > static_cast<T>(0));
  CHECK(freeEnergyPerUnitArea < static_cast<T>(0));
  CHECK(context->isConverged());

  // Clean up context
  conga::run_utils::cleanup(context);
}
} // namespace helper

// ================= Test current conservation ================= //

TEST_CASE("Test current conservation - s-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - s-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (float)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (double)") {
  const conga::BasisFunctionSymmetry symmetry =
      conga::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}
