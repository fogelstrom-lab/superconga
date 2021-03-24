#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeCurrent.h" // Compute the current density
#include "ComputeFreeEnergy.h" // Compute the free energy (using the Eilenberger functional)
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "DiscGeometry.h"  // Used to constructed simulated grain from discs
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "defines.h"
#include "run_utils.h"

#include <doctest/doctest.h>
#include <iostream>
#include <limits>
#include <stdlib.h>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.001;

// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0 / (2.0 * M_PI);

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 32.0 / GRAIN_SIDE_LENGTH;

// Number of discrete trajectory angles (on the Fermi-surface). S-wave behaves
// much nicer.
const int NUM_ANGLES_SWAVE = 32;
const int NUM_ANGLES_DWAVE = 360;

// Number of discrete energies (poles)
const int NUM_ENERGIES = 256;

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
scg::Parameters<T> *applySimParams(const bool inUseSwave,
                                   scg::Context<T> *inOutContext) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  scg::Parameters<T> *parameters = new scg::Parameters<T>(inOutContext);

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

  // Set number of Ozaki poles
  parameters->setNumMatsubaraEnergies(simParams::NUM_ENERGIES);

  // Set temperature
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));

  return parameters;
}

// *** Test bulk free energy and order-parameter value at low temperature *** //
template <typename T>
void testLowTempBulkValues(const T inTolerance, const T inToleranceFreeEnergy,
                           const bool inUseSwave, const bool inIsDisc) {
  // =================== ARRANGE ==================== //
  // Create Context
  scg::Context<T> *context = new scg::Context<T>();

  // Set parameters to the values specified in simParams struct
  scg::Parameters<T> *parameters = applySimParams<T>(inUseSwave, context);

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const T grainFraction = parameters->getGrainFraction();
  if (inIsDisc) {
    geometry->add(new scg::DiscGeometry<T>(grainFraction), scg::Type::add);
  } else {
    const int numVerticesSquare = 4;
    geometry->add(new scg::PolygonGeometry<T>(grainFraction, numVerticesSquare),
                  scg::Type::add);
  }

  // Choose a Riccati solver for a confined geometry, without magnetic fields
  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                     scg::VectorPotential::None>(context);
  // Add specular BC
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  // Set up order parameter
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);

  // Add a d-wave or an s-wave order-parameter component.
  const scg::BasisFunctionSymmetry symmetry =
      inUseSwave ? scg::BasisFunctionSymmetry::SWAVE
                 : scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
  scg::OrderParameterComponent<T> *orderParameterComponent =
      new scg::OrderParameterComponent<T>(symmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  // It is interesting to have a test with non-bulk initialization. It passes
  // either way but it takes shorter time with bulk initialization. Also, if
  // everything is correct it should converge after one iteration, something
  // which we cannot see if we don't use bulk.
  scg::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Create compute objects
  scg::ComputeOrderParameter<T> *computeOrderParameter =
      new scg::ComputeOrderParameter<T>(context);
  scg::ComputeFreeEnergy<T> *computeFreeEnergy =
      new scg::ComputeFreeEnergy<T>(context);
  scg::ComputeCurrent<T> *computeCurrent = new scg::ComputeCurrent<T>(context);

  // =================== ACT ==================== //
  // Initialize simulation
  context->initialize();

  scg::run_utils::runSimulation(context, simParams::PRINT_ITERATION_STATUS,
                                simParams::NUM_ITERATIONS_MIN,
                                simParams::NUM_ITERATIONS_MAX);

  // Remove all spurious values outside actual grain
  context->getGeometry()->zeroValuesOutsideDomainRef(
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
      (inUseSwave ? static_cast<T>(scg::constants::S_WAVE_BULK)
                  : static_cast<T>(scg::constants::D_WAVE_BULK));
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
  CHECK(context->isConverged());

  // Clean up context
  scg::run_utils::cleanup(context);
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
