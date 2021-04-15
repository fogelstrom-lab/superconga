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
#include "Parameters.h" //
#include "PolygonGeometry.h"
#include "RiccatiSolverConfined.h"
#include "defines.h"
#include "run_utils.h"

#include <doctest/doctest.h>
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

// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 15.0 / (2.0 * M_PI);

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 3.0 * (2.0 * M_PI);

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 30;

// Number of discrete energies (poles)
const int NUM_ENERGIES = 10;

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
const scg::vector2d<scg::BasisFunctionSymmetry> SYMMETRIES = {
    scg::BasisFunctionSymmetry::SWAVE, scg::BasisFunctionSymmetry::DWAVE_XY};

// The OP relative TCs.
const scg::vector2d<float> RELATIVE_TCS = {0.1f, 1.0f};
} // namespace TEST_PARAMS

namespace helper {
// ========================= Test helper objects ========================= //

template <typename T> T computeFreeEnergy(const bool inSwapOrder) {
  // Create context.
  scg::Context<T> *context = new scg::Context<T>();

  // Set parameters.
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
  parameters->setGrainWidth(static_cast<T>(TEST_PARAMS::GRAIN_SIDE_LENGTH));
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(TEST_PARAMS::POINTS_PER_COHERENCE_LENGTH));
  parameters->setAngularResolution(TEST_PARAMS::NUM_ANGLES);
  parameters->setConvergenceCriterion(
      static_cast<T>(TEST_PARAMS::CONVERGENCE_CRITERION));
  parameters->setNumMatsubaraEnergies(TEST_PARAMS::NUM_ENERGIES);
  parameters->setTemperature(static_cast<T>(TEST_PARAMS::TEMPERATURE));

  // Set up a disc.
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new scg::DiscGeometry<T>(grainFraction), scg::Type::add);

  // Choose a Riccati solver for a confined geometry.
  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                     scg::VectorPotential::Radial>(context);
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  // Set up order parameter.
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);
  const scg::vector2d<int> indices = {static_cast<int>(inSwapOrder),
                                      static_cast<int>(!inSwapOrder)};
  for (const int i : indices) {
    scg::OrderParameterComponent<T> *orderParameterComponent =
        new scg::OrderParameterComponent<T>(
            TEST_PARAMS::SYMMETRIES[i], parameters,
            static_cast<T>(TEST_PARAMS::RELATIVE_TCS[i]));
    scg::Initial<T>::bulk(orderParameterComponent);
    orderParameter->add(orderParameterComponent);
  }

  // Create compute objects.
  new scg::ComputeOrderParameter<T>(context);
  new scg::ComputeFreeEnergy<T>(context);

  // Set optimizer.
  context->set(std::make_unique<scg::Optimizer<T>>());

  // Initialize and run simulation.
  context->initialize();
  scg::run_utils::runSimulation(context, TEST_PARAMS::PRINT_ITERATION_STATUS,
                                TEST_PARAMS::NUM_ITERATIONS_MIN,
                                TEST_PARAMS::NUM_ITERATIONS_MAX);

  // Get the free energy, our result.
  const T freeEnergy =
      context->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  // Clean up context.
  scg::run_utils::cleanup(context);

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
