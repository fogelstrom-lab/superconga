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

namespace TEST_PARAMS {
// Temperature in units of T_c
const double TEMPERATURE = 0.5;

// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 15.0;

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 3.0;

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 30;

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

// The tolerance is this factor times the machine precision.
const double EPSILON_TOLERANCE = 10.0;

// The OP symmetries.
const conga::vector2d<conga::BasisFunctionSymmetry> SYMMETRIES = {
    conga::BasisFunctionSymmetry::SWAVE,
    conga::BasisFunctionSymmetry::DWAVE_XY};

// The OP relative TCs.
const conga::vector2d<float> RELATIVE_TCS = {0.1f, 1.0f};
} // namespace TEST_PARAMS

namespace helper {
// ========================= Test helper objects ========================= //

template <typename T> T computeFreeEnergy(const bool inSwapOrder) {
  // Create context.
  conga::Context<T> *context = new conga::Context<T>();

  // Set parameters.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(context);
  parameters->setGrainWidth(static_cast<T>(TEST_PARAMS::GRAIN_SIDE_LENGTH));
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(TEST_PARAMS::POINTS_PER_COHERENCE_LENGTH));
  parameters->setAngularResolution(TEST_PARAMS::NUM_ANGLES);
  parameters->setConvergenceCriterion(
      static_cast<T>(TEST_PARAMS::CONVERGENCE_CRITERION));
  parameters->setTemperature(static_cast<T>(TEST_PARAMS::TEMPERATURE));
  parameters->setEnergyCutoff(static_cast<T>(TEST_PARAMS::ENERGY_CUTOFF));

  // Set up a disc.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);

  // Choose a Riccati solver for a confined geometry.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
          context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set up order parameter.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(context);
  const conga::vector2d<int> indices = {static_cast<int>(inSwapOrder),
                                        static_cast<int>(!inSwapOrder)};
  for (const int i : indices) {
    conga::OrderParameterComponent<T> *orderParameterComponent =
        new conga::OrderParameterComponent<T>(
            TEST_PARAMS::SYMMETRIES[i], parameters,
            static_cast<T>(TEST_PARAMS::RELATIVE_TCS[i]));
    conga::Initial<T>::bulk(orderParameterComponent);
    orderParameter->add(orderParameterComponent);
  }

  // Create compute objects.
  new conga::ComputeOrderParameter<T>(context);
  new conga::ComputeFreeEnergy<T>(context);

  // Set optimizer.
  context->set(std::make_unique<conga::Optimizer<T>>());

  // Initialize and run simulation.
  context->initialize();
  conga::run_utils::runSimulation(context, TEST_PARAMS::PRINT_ITERATION_STATUS,
                                  TEST_PARAMS::NUM_ITERATIONS_MIN,
                                  TEST_PARAMS::NUM_ITERATIONS_MAX);

  // Get the free energy, our result.
  const T freeEnergy =
      context->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  // Clean up context.
  conga::run_utils::cleanup(context);

  return freeEnergy;
}
} // namespace helper

// ================= Test free energy ================= //

TEST_CASE("Test symmetry - (float)") {
  // =================== ARRANGE ==================== //
  // Nothing to do here...

  // =================== ACT ==================== //
  const float freeEnergy = helper::computeFreeEnergy<float>(false);
  const float freeEnergySwapped = helper::computeFreeEnergy<float>(true);

  // =================== ASSERT ==================== //
  const float tolerance = static_cast<float>(TEST_PARAMS::EPSILON_TOLERANCE) *
                          std::numeric_limits<float>::epsilon();
  CHECK(freeEnergy == doctest::Approx(freeEnergySwapped).epsilon(tolerance));
}

TEST_CASE("Test symmetry - (double)") {
  // =================== ARRANGE ==================== //
  // Nothing to do here...

  // =================== ACT ==================== //
  const double freeEnergy = helper::computeFreeEnergy<double>(false);
  const double freeEnergySwapped = helper::computeFreeEnergy<double>(true);

  // =================== ASSERT ==================== //
  const double tolerance =
      TEST_PARAMS::EPSILON_TOLERANCE * std::numeric_limits<double>::epsilon();
  CHECK(freeEnergy == doctest::Approx(freeEnergySwapped).epsilon(tolerance));
}
