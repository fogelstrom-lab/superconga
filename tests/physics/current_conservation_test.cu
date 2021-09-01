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

// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0 / (2.0 * M_PI);

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 5.0 * (2.0 * M_PI);

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 32;

// Number of discrete energies (poles)
const int NUM_ENERGIES = 8;

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
scg::Parameters<T> *applySimParams(scg::Context<T> *inOutContext) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  scg::Parameters<T> *parameters = new scg::Parameters<T>(inOutContext);

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

  // Set number of Ozaki poles
  parameters->setNumMatsubaraEnergies(simParams::NUM_ENERGIES);

  // Set temperature
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));

  // External flux-quanta.
  parameters->setNumFluxQuanta(static_cast<T>(simParams::EXTERNAL_FLUX_QUANTA));

  return parameters;
}

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testCurrentConservation(const scg::BasisFunctionSymmetry inSymmetry,
                             const int inNumVertices) {
  // =================== ARRANGE ==================== //
  // Create Context
  scg::Context<T> *context = new scg::Context<T>();

  // Set parameters to the values specified in simParams struct
  scg::Parameters<T> *parameters = applySimParams<T>(context);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const T grainFraction = parameters->getGrainFraction();
  const bool isDisc = inNumVertices < 0;
  if (isDisc) {
    geometry->add(new scg::DiscGeometry<T>(grainFraction), scg::Type::add);
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
        new scg::PolygonGeometry<T>(grainFraction, inNumVertices, sideLength),
        scg::Type::add);
  }

  // Choose a Riccati solver for a confined geometry.
  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::VectorPotential::Radial>(context);
  // Add specular BC
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  // Set up order parameter
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);

  // Add an order-parameter component.
  scg::OrderParameterComponent<T> *orderParameterComponent =
      new scg::OrderParameterComponent<T>(inSymmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  scg::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  // Order parameter compute object.
  new scg::ComputeOrderParameter<T>(context);
  // Add current compute object.
  new scg::ComputeCurrent<T>(context);
  // Add free energy compute object.
  new scg::ComputeFreeEnergy<T>(context);

  // Set optimizer.
  context->set(std::make_unique<scg::Optimizer<T>>());

  // =================== ACT ==================== //
  context->initialize();

  scg::run_utils::runSimulation(context, simParams::PRINT_ITERATION_STATUS,
                                simParams::NUM_ITERATIONS_MIN,
                                simParams::NUM_ITERATIONS_MAX);

  // Fetch the grain-summed current-density magnitude
  const scg::GridGPU<T> currentDensity =
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
  scg::run_utils::cleanup(context);
}
} // namespace helper

// ================= Test current conservation ================= //

TEST_CASE("Test current conservation - s-wave (float)") {
  const scg::BasisFunctionSymmetry symmetry = scg::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - s-wave (double)") {
  const scg::BasisFunctionSymmetry symmetry = scg::BasisFunctionSymmetry::SWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (float)") {
  const scg::BasisFunctionSymmetry symmetry =
      scg::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{x^2 - y^2}-wave (double)") {
  const scg::BasisFunctionSymmetry symmetry =
      scg::BasisFunctionSymmetry::DWAVE_X2_Y2;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (float)") {
  const scg::BasisFunctionSymmetry symmetry =
      scg::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - d_{xy}-wave (double)") {
  const scg::BasisFunctionSymmetry symmetry =
      scg::BasisFunctionSymmetry::DWAVE_XY;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (float)") {
  const scg::BasisFunctionSymmetry symmetry = scg::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<float>(symmetry, numVertices);
  }
}

TEST_CASE("Test current conservation - g-wave (double)") {
  const scg::BasisFunctionSymmetry symmetry = scg::BasisFunctionSymmetry::GWAVE;

  for (const int numVertices : simParams::NUM_VERTICES) {
    std::cout << "numVertices = " << numVertices << std::endl;
    helper::testCurrentConservation<double>(symmetry, numVertices);
  }
}
