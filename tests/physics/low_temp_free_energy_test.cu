#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeCurrent.h" // Compute the current density
#include "ComputeFreeEnergy.h" // Compute the free energy (using the Eilenberger functional)
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "DiscGeometry.h" // Used to constructed simulated grain from discs
#include "FermiSurfaceCircle.h" // Standard circle Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OptimizerRollingBall.h" // Convergence accelerator for self-consistency solver
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterComponent_D.h" // d-wave
#include "OrderParameterComponent_S.h" // s-wave
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "defines.h"
#include "run_utils.h"

#include <doctest.h>
#include <iostream>
#include <stdlib.h>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.005;
// Size of superconducting grain in coherence lengths hbar*vF/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 5.0;
// Number of lattice points per grain side length
const int GRID_RESOLUTION = 128;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 64;
// Number of discrete energies (poles)
const int NUM_ENERGIES = 64;
// Convergence criterion for order-parameter residual
const double CONVERGENCE_CRITERION = 1E-6;
// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = true;
// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 100;
// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 10;
} // namespace simParams

namespace helper {
// ========================= Test helper objects ========================= //
const int NUM_VERTICES_SQUARE = 4;
const int NUM_START_VALUES = 2;
const std::array<double, NUM_START_VALUES> START_VALUES = {1.0, std::sqrt(2.0)};

// ========================= Test helper functions ========================= //
// Compute tolerance for GridData testing, given the number of elements in a
// GridData object. Note: inToleranceBaseFactor has to be > 1.
template <typename T>
T computeTolerance(const T inToleranceBaseFactor = static_cast<T>(5.0)) {
  return inToleranceBaseFactor * std::numeric_limits<T>::epsilon();
}

// Helper class for setting up simulation parameters
template <typename T> Parameters<T> *applySimParams(Context<T> *context) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set grain resolution in number of lattice sites
  parameters->setGridResolution(simParams::GRID_RESOLUTION);
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc)
  parameters->setGrainWidthCoherenceLengths(
      static_cast<T>(simParams::GRAIN_SIDE_LENGTH));
  // Number of discrete steps in momentum angle
  parameters->setAngularResolution(simParams::NUM_ANGLES);
  // Set convergence criterion
  parameters->setConvergenceCriterion(
      static_cast<T>(simParams::CONVERGENCE_CRITERION));
  // Set number of Ozaki poles
  parameters->setNumMatsubaraEnergies(simParams::NUM_ENERGIES);
  // Set temperature
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));
  return parameters;
}

// ======================== Wrappers for test cases ======================== //
// **** Test that order parameter grid is normalized properly **** /
template <typename T>
void testGridNormalization(const T inTolerance, const T inStartRealPart,
                           const T inStartImaginaryPart) {
  // Loop over s-wave and d-wave. This is an outer for loop that will create and
  // destroy a whole context. The inner loops only loop over values and call
  // context->intialize().
  for (const bool useSwave : {true, false}) {
    // =================== ARRANGE ==================== //
    // Create Context
    Context<T> *context = new Context<T>();

    // Set parameters to the values specified in simParams struct
    Parameters<T> *parameters = applySimParams<T>(context);

    // Create geometry group: add/remove geometry objects (e.g. discs,
    // polygons) from the geometry group
    GeometryGroup<T> *geometry = new GeometryGroup<T>(context);
    geometry->add(new PolygonGeometry<T>(NUM_VERTICES_SQUARE), Type::add);

    // Choose a Riccati solver for a confined geometry, without magnetic
    // fields
    RiccatiSolver<T> *riccatiSolver =
        new RiccatiSolverConfined<T, Gamma::General, VectorPotential::None>(
            context);

    // Choose FermiSurface
    FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
    context->set(fermiSurface);

    // Set up order parameter
    OrderParameter<T> *orderParameter = new OrderParameter<T>(context);

    // Use s-wave or d-wave
    OrderParameterComponent<T> *orderParameterComponent;
    if (useSwave) {
      orderParameterComponent = new OrderParameterComponent_S<T>(parameters);
    } else {
      orderParameterComponent = new OrderParameterComponent_D<T>(parameters);
    }
    Initial<T>::constant(orderParameterComponent, inStartRealPart,
                         inStartImaginaryPart);
    // Add the initialized s-wave component to the total order parameter
    orderParameter->add(orderParameterComponent);

    // Create compute objects
    ComputeOrderParameter<T> *computeOrderParameter =
        new ComputeOrderParameter<T>(context);

    // =================== ACT ==================== //
    // Initialize simulation
    context->initialize();

    // Remove all spurious values outside actual grain
    context->getGeometry()->zeroValuesOutsideDomain(
        orderParameterComponent->getOrderParameterField());

    // Fetch the grain-summed order-parameter magnitude
    const T testSumOrderParameter =
        orderParameterComponent->getOrderParameterField()->sumMagnitude();
    // Fetch the number of elements in the grain
    const int numGrainLatticeSites = geometry->getNumGrainLatticeSites();

    // =================== ASSERT ==================== //
    const T expectedSumOrderParameter =
        static_cast<T>(numGrainLatticeSites) *
        std::hypot(inStartRealPart, inStartImaginaryPart);

    REQUIRE(testSumOrderParameter ==
            doctest::Approx(expectedSumOrderParameter).epsilon(inTolerance));

    // Clean up context
    run_utils::cleanup(context);
  }
} // namespace helper

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testLowTempBulkValues(const T inTolerance, const bool inUseSwave,
                           const T inStartValue) {
  // =================== ARRANGE ==================== //
  // Create Context
  Context<T> *context = new Context<T>();

  // Set parameters to the values specified in simParams struct
  Parameters<T> *parameters = applySimParams<T>(context);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

  // Add a square to the simulation geometry.
  geometry->add(new PolygonGeometry<T>(NUM_VERTICES_SQUARE), Type::add);

  // Choose a Riccati solver for a confined geometry, without magnetic fields
  RiccatiSolver<T> *riccatiSolver =
      new RiccatiSolverConfined<T, Gamma::General, VectorPotential::None>(
          context);
  // Add specular BC
  riccatiSolver->add(new BoundaryConditionSpecular<T>);

  // Set up order parameter
  OrderParameter<T> *orderParameter = new OrderParameter<T>(context);

  // Use s-wave or d-wave
  OrderParameterComponent<T> *orderParameterComponent;
  if (inUseSwave) {
    orderParameterComponent = new OrderParameterComponent_S<T>(parameters);
  } else {
    orderParameterComponent = new OrderParameterComponent_D<T>(parameters);
  }

  // Initialize order-parameter component to a constant
  Initial<T>::constant(orderParameterComponent, inStartValue, inStartValue);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  ComputeOrderParameter<T> *computeOrderParameter =
      new ComputeOrderParameter<T>(context);
  ComputeFreeEnergy<T> *computeFreeEnergy = new ComputeFreeEnergy<T>(context);
  ComputeCurrent<T> *computeCurrent = new ComputeCurrent<T>(context);

  // Choose convergence accelerator
  OptimizerRollingBall<T> *optimizer = new OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  // Choose FermiSurface
  FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  // =================== ACT ==================== //
  // Initialize simulation
  context->initialize();

  const bool visualize = false;
  run_utils::runSimulation(
      context, visualize, simParams::PRINT_ITERATION_STATUS,
      simParams::NUM_ITERATIONS_MIN, simParams::NUM_ITERATIONS_MAX);

  // Remove all spurious values outside actual grain
  context->getGeometry()->zeroValuesOutsideDomain(
      computeOrderParameter->getResult());

  // Fetch the grain-averaged free energy
  const T testAverageFreeEnergy = computeFreeEnergy->getFreeEnergyPerUnitArea();
  // Fetch the grain-summed order-parameter magnitude
  const T sumAbsOrderParameter =
      computeOrderParameter->getResult()->sumMagnitude();
  // Fetch the grain-summed current-density magnitude
  const T sumAbsCurrent = computeCurrent->getResult()->sumMagnitude();
  // Fetch the number of elements in the grain
  const int numGrainLatticeSites = geometry->getNumGrainLatticeSites();
  // Calculate the grain-averaged order-parameter magnitude
  const T testAverageOrderParameter =
      sumAbsOrderParameter / static_cast<T>(numGrainLatticeSites);
  // Calculate the grain-averaged order-parameter magnitude
  const T testAverageCurrentDensity =
      sumAbsCurrent / static_cast<T>(numGrainLatticeSites);

  // =================== ASSERT ==================== //
  // Expected average bulk gap
  const T expectedAverageOrderParameter =
      (inUseSwave ? static_cast<T>(qcs::S_WAVE_BULK)
                  : static_cast<T>(qcs::D_WAVE_BULK));
  // Expected average current density
  const T expectedAverageCurrent = static_cast<T>(0.0);
  // The bulk free energy, see page 106 of P. Holmvall PhD Thesis
  const T expectedAverageFreeEnergy = -static_cast<T>(0.5) *
                                      expectedAverageOrderParameter *
                                      expectedAverageOrderParameter;
  const T currentDensityTolerance = std::numeric_limits<T>::epsilon();

  CHECK(testAverageOrderParameter ==
        doctest::Approx(expectedAverageOrderParameter).epsilon(inTolerance));
  CHECK(
      testAverageCurrentDensity ==
      doctest::Approx(expectedAverageCurrent).epsilon(currentDensityTolerance));
  CHECK(testAverageFreeEnergy ==
        doctest::Approx(expectedAverageFreeEnergy).epsilon(inTolerance));
  CHECK(context->isConverged());

  // Clean up context
  run_utils::cleanup(context);
}

} // namespace helper

// ================= Test grid normalization ================= //
TEST_CASE("Test Initial<float>::Constant order parameter value") {
  const float toleranceMultFactor = 10.0F;
  const float tolerance = helper::computeTolerance<float>(toleranceMultFactor);
  const float startRealPart = 1.0;
  const float startImaginaryPart = 1.0;

  helper::testGridNormalization<float>(tolerance, startRealPart,
                                       startImaginaryPart);
}

TEST_CASE("Test Initial<double>::Constant order parameter value") {
  const double toleranceMultFactor = 10.0;
  const double tolerance =
      helper::computeTolerance<double>(toleranceMultFactor);
  const double startRealPart = 1.0;
  const double startImaginaryPart = 1.0;

  helper::testGridNormalization<double>(tolerance, startRealPart,
                                        startImaginaryPart);
}

// ================= Test thermodynamics, low temp ================= //
TEST_CASE("Test low temperature thermodynamics - d-wave (double)") {
  const double tolerance = 1e-3;
  const bool useSwave = false;

  for (const double &startValue : helper::START_VALUES) {
    helper::testLowTempBulkValues<double>(tolerance, useSwave, startValue);
  }
}

TEST_CASE("Test low temperature thermodynamics - s-wave (double)") {
  const double tolerance = 1e-3;
  const bool useSwave = true;

  for (const double &startValue : helper::START_VALUES) {
    helper::testLowTempBulkValues<double>(tolerance, useSwave, startValue);
  }
}

TEST_CASE("Test low temperature thermodynamics - d-wave (float)") {
  const float tolerance = 1e-3F;
  const bool useSwave = false;

  for (const double &startValue : helper::START_VALUES) {
    helper::testLowTempBulkValues<float>(tolerance, useSwave,
                                         static_cast<float>(startValue));
  }
}

TEST_CASE("Test low temperature thermodynamics - s-wave (float)") {
  const float tolerance = 1e-3F;
  const bool useSwave = true;

  for (const double &startValue : helper::START_VALUES) {
    helper::testLowTempBulkValues<float>(tolerance, useSwave,
                                         static_cast<float>(startValue));
  }
}
