#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/defines.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
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

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.001;

// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 32.0 / GRAIN_SIDE_LENGTH;

// Number of discrete trajectory angles (on the Fermi-surface). S-wave behaves
// much nicer.
const int NUM_ANGLES_SWAVE = 32;
const int NUM_ANGLES_DWAVE = 360;

// The energy cutoff.
const double ENERGY_CUTOFF = 32.0;

// Convergence criterion for order-parameter residual
const double CONVERGENCE_CRITERION = 1E-6;

// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = true;

// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 50;

// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 1;
} // namespace simParams

namespace helper {
// ========================= Test helper objects ========================= //

// Helper class for setting up simulation parameters
template <typename T>
conga::Parameters<T> *applySimParams(const bool inUseSwave,
                                     conga::Context<T> *inOutContext) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  conga::Parameters<T> *parameters = new conga::Parameters<T>(inOutContext);

  // Set grain sidelength in coherence lengths: hbar*<vF>/(kB*Tc).
  parameters->setGrainWidth(static_cast<T>(simParams::GRAIN_SIDE_LENGTH));

  // Set the number of lattice points per coherence length.
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(simParams::POINTS_PER_COHERENCE_LENGTH));

  // Number of discrete steps in momentum angle
  const int numAngles =
      inUseSwave ? simParams::NUM_ANGLES_SWAVE : simParams::NUM_ANGLES_DWAVE;
  parameters->setAngularResolution(numAngles);

  // Set convergence criterion
  parameters->setConvergenceCriterion(
      static_cast<T>(simParams::CONVERGENCE_CRITERION));

  // Set temperature
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));

  // Set number of Ozaki poles
  parameters->setEnergyCutoff(static_cast<T>(simParams::ENERGY_CUTOFF));

  return parameters;
}

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testLowTempBulkValues(const T inTolerance, const T inToleranceFreeEnergy,
                           const bool inUseSwave, const bool inIsDisc) {
  // =================== ARRANGE ==================== //
  // Create Context
  conga::Context<T> context;

  // Set parameters to the values specified in simParams struct
  conga::Parameters<T> *parameters = applySimParams<T>(inUseSwave, &context);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  if (inIsDisc) {
    geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
  } else {
    const int numVerticesSquare = 4;
    geometry->add(
        new conga::PolygonGeometry<T>(grainFraction, numVerticesSquare),
        conga::Type::add);
  }

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Choose a Riccati solver for a confined geometry, without magnetic fields
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::None>(
          &context);
  // Add specular BC
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set up order parameter
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);

  // Add a d-wave or an s-wave order-parameter component.
  const conga::BasisFunctionSymmetry symmetry =
      inUseSwave ? conga::BasisFunctionSymmetry::SWAVE
                 : conga::BasisFunctionSymmetry::DWAVE_X2_Y2;
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(symmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  // It is interesting to have a test with non-bulk initialization. It passes
  // either way but it takes shorter time with bulk initialization. Also, if
  // everything is correct it should converge after one iteration, something
  // which we cannot see if we don't use bulk.
  conga::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  conga::ComputeOrderParameter<T> *computeOrderParameter =
      new conga::ComputeOrderParameter<T>(&context);
  conga::ComputeFreeEnergy<T> *computeFreeEnergy =
      new conga::ComputeFreeEnergy<T>(&context);
  conga::ComputeCurrent<T> *computeCurrent =
      new conga::ComputeCurrent<T>(&context);

  // =================== ACT ==================== //
  // Initialize simulation
  context.initialize();

  conga::Status status = conga::Status::kContinue;
  while (status != conga::Status::kQuit) {
    status = conga::runIteration(&context, simParams::PRINT_ITERATION_STATUS,
                                 simParams::NUM_ITERATIONS_MIN,
                                 simParams::NUM_ITERATIONS_MAX);
  }

  // Remove all spurious values outside actual grain
  context.getGeometry()->zeroValuesOutsideDomainRef(
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
      (inUseSwave ? static_cast<T>(conga::constants::S_WAVE_BULK)
                  : static_cast<T>(conga::constants::D_WAVE_BULK));
  // Expected average current density
  const T expectedAverageCurrent = static_cast<T>(0.0);
  // The bulk free energy, see page 106 of P. Holmvall PhD Thesis
  const T expectedAverageFreeEnergy = -static_cast<T>(0.5) *
                                      expectedAverageOrderParameter *
                                      expectedAverageOrderParameter;
  const T currentDensityTolerance = std::numeric_limits<T>::epsilon();

  // TODO(niclas): We are computing everything at a low temperature, but it is
  // not exactly zero. Wouldn't it be better to compare with bulk values at that
  // specific temperature? And test the bulk solver at ridiculously low
  // temperature instead? Because then we can really test that it yields exactly
  // what we expect.
  CHECK(testAverageOrderParameter ==
        doctest::Approx(expectedAverageOrderParameter).epsilon(inTolerance));
  CHECK(
      testAverageCurrentDensity ==
      doctest::Approx(expectedAverageCurrent).epsilon(currentDensityTolerance));
  CHECK(testAverageFreeEnergy == doctest::Approx(expectedAverageFreeEnergy)
                                     .epsilon(inToleranceFreeEnergy));
  CHECK(context.isConverged());
}
} // namespace helper

// ================= Test thermodynamics, low temp ================= //
TEST_CASE("Test low temperature thermodynamics - s-wave disc (float)") {
  const float tolerance = 1e-5f;
  const float toleranceFreeEnergy = 1e-4f;
  const bool useSwave = true;
  const bool isDisc = true;

  helper::testLowTempBulkValues<float>(tolerance, toleranceFreeEnergy, useSwave,
                                       isDisc);
}

TEST_CASE("Test low temperature thermodynamics - s-wave square (float)") {
  const float tolerance = 1e-5f;
  const float toleranceFreeEnergy = 1e-4f;
  const bool useSwave = true;
  const bool isDisc = false;

  helper::testLowTempBulkValues<float>(tolerance, toleranceFreeEnergy, useSwave,
                                       isDisc);
}

TEST_CASE("Test low temperature thermodynamics - d-wave square (float)") {
  const float tolerance = 1e-5f;
  const float toleranceFreeEnergy = 1e-4f;
  const bool useSwave = false;
  const bool isDisc = false;

  helper::testLowTempBulkValues<float>(tolerance, toleranceFreeEnergy, useSwave,
                                       isDisc);
}

TEST_CASE("Test low temperature thermodynamics - s-wave disc (double)") {
  const double tolerance = 1e-5;
  const double toleranceFreeEnergy = 1e-4;
  const bool useSwave = true;
  const bool isDisc = true;

  helper::testLowTempBulkValues<double>(tolerance, toleranceFreeEnergy,
                                        useSwave, isDisc);
}

TEST_CASE("Test low temperature thermodynamics - s-wave square (double)") {
  const double tolerance = 1e-5;
  const double toleranceFreeEnergy = 1e-4;
  const bool useSwave = true;
  const bool isDisc = false;

  helper::testLowTempBulkValues<double>(tolerance, toleranceFreeEnergy,
                                        useSwave, isDisc);
}

TEST_CASE("Test low temperature thermodynamics - d-wave square (double)") {
  const double tolerance = 1e-5;
  const double toleranceFreeEnergy = 1e-4;
  const bool useSwave = false;
  const bool isDisc = false;

  helper::testLowTempBulkValues<double>(tolerance, toleranceFreeEnergy,
                                        useSwave, isDisc);
}
