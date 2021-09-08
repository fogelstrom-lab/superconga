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
#include "OrderParameterBulk.h"
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

namespace helper {
//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************

// Temperature in units of T_c
const double TEMPERATURE_LOW = 0.1;
const double TEMPERATURE_HIGH = 0.9;

// Just some finite value so that the induced vector potential is used.
const double PENETRATION_DEPTH = 1.0;

// The constant values of the vector-potential components.
const double VECTOR_POTENTIAL_X = 1e-3;
const double VECTOR_POTENTIAL_Y = 1e-3;

// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 20.0 / (2.0 * M_PI);

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 10.0 * (2.0 * M_PI);

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 100;

// The energy cutoff.
const double ENERGY_CUTOFF = 100.0;

// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = false;

// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 1;

// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 1;

// The tolerance and scale.
const double TOLERANCE = 0.01;
const double SCALE = 0.0;

/// \brief Compute the current density via the linear-response expression.
///
/// \param inTemperature The temperature.
/// \param inVectorPotentialX The vector-potential x-component.
/// \param inVectorPotentialY The vector-potential y-component.
///
/// \return The current density as a 2D-vector.
template <typename T>
scg::vector2d<T> computeCurrentDensity(const T inTemperature,
                                       const T inVectorPotentialX,
                                       const T inVectorPotentialY) {
  // The bulk orderparameter at this temperature.
  const T orderParameter = scg::computeOrderParameterBulkSWave(inTemperature);

  // For convenience.
  const T twoPiTemp = static_cast<T>(2.0 * M_PI) * inTemperature;

  // Comes from: cutoff = 2 * pi * T * (n + 0.5).
  // But the n is the largest n we need to include, so the number of energies is
  // one more, hence the plus one.
  const int numEnergies =
      static_cast<int>(static_cast<T>(ENERGY_CUTOFF) / twoPiTemp -
                       static_cast<T>(0.5)) +
      1;

  // Compute the factor involving the order parameter and the energy.
  T factorEnergy = static_cast<T>(0);
  for (int n = numEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        twoPiTemp * (static_cast<T>(n) + static_cast<T>(0.5));
    const T Omega = std::hypot(orderParameter, matsubaraEnergy);
    factorEnergy += std::pow(Omega, -3);
  }
  factorEnergy *= orderParameter * orderParameter;

  // Factor due to choice of units.
  const T factorUnits =
      static_cast<T>(scg::constants::SIGN_E * M_PI) * inTemperature;

  // The total factor. The half comes from 1/(2*pi) * int cos^2(x) dx.
  const T factor = -factorEnergy * factorUnits * static_cast<T>(0.5);

  return {inVectorPotentialX * factor, inVectorPotentialY * factor};
}

/// \brief Compare the current density with the linear-response expression, for
/// a given temperature.
///
/// \param inTemperature The temperature.
///
/// \return Void.
template <typename T> void testCurrentDensity(const T inTemperature) {
  // =================== ARRANGE ==================== //
  // Create context.
  scg::Context<T> *context = new scg::Context<T>();

  // Set parameters.
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
  parameters->setGrainWidth(static_cast<T>(GRAIN_SIDE_LENGTH));
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(POINTS_PER_COHERENCE_LENGTH));
  parameters->setAngularResolution(NUM_ANGLES);
  parameters->setTemperature(inTemperature);
  parameters->setEnergyCutoff(static_cast<T>(ENERGY_CUTOFF));
  parameters->setPenetrationDepth(static_cast<T>(PENETRATION_DEPTH));

  // Set geometry.
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const T grainFraction = parameters->getGrainFraction();
  const int numVertices = 4;
  const T grainWidth = parameters->getGrainWidth();
  geometry->add(
      new scg::PolygonGeometry<T>(grainFraction, numVertices, grainWidth),
      scg::Type::add);

  // Choose a Riccati solver for a confined geometry.
  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::VectorPotential::None>(context);
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  // Set order parameter.
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);
  scg::OrderParameterComponent<T> *orderParameterComponent =
      new scg::OrderParameterComponent<T>(scg::BasisFunctionSymmetry::SWAVE,
                                          parameters);
  scg::Initial<T>::bulk(orderParameterComponent);
  orderParameter->add(orderParameterComponent);

  // Set vector potential.
  const int gridResolution = parameters->getGridResolutionBase();
  auto vectorPotential = std::make_unique<scg::GridGPU<T>>(
      gridResolution, gridResolution, 2, scg::Type::real);
  const T vectorPotentialX = static_cast<T>(VECTOR_POTENTIAL_X);
  const T vectorPotentialY = static_cast<T>(VECTOR_POTENTIAL_Y);
  vectorPotential->setValue(vectorPotentialX, scg::Type::real, 0);
  vectorPotential->setValue(vectorPotentialY, scg::Type::real, 1);
  context->set(std::move(vectorPotential));

  // Create compute objects.
  new scg::ComputeCurrent<T>(context);
  context->add(std::make_unique<scg::ComputeVectorPotential<T>>());

  // Set optimizer.
  context->set(std::make_unique<scg::Optimizer<T>>());

  // =================== ACT ==================== //
  // Initialize and run simulation.
  context->initialize();
  scg::run_utils::runSimulation(context, PRINT_ITERATION_STATUS,
                                NUM_ITERATIONS_MIN, NUM_ITERATIONS_MAX);

  // Fetch the grain-summed current-density magnitude
  const scg::GridGPU<T> currentDensity =
      *(context->getComputeCurrent()->getResult());

  // Fetch the number of elements in the grain
  const int numGrainLatticeSites = geometry->getNumGrainLatticeSites();

  scg::vector2d<T> currentDensityMean;
  for (int i = 0; i < 2; ++i) {
    currentDensityMean[i] =
        currentDensity.sumField(i) / static_cast<T>(numGrainLatticeSites);
  }

  // Clean up context.
  scg::run_utils::cleanup(context);

  // =================== ASSERT ==================== //
  const scg::vector2d<T> currentDensityExpected =
      computeCurrentDensity(inTemperature, vectorPotentialX, vectorPotentialY);

  // Note, the way doctest compares two floating point numbers, A and B, is the
  // following: |A - B| < epsilon * (scale + max(|A|, |B|))
  const T tolerance = static_cast<T>(TOLERANCE);
  const T scale = static_cast<T>(SCALE);
  CHECK(currentDensityMean[0] == doctest::Approx(currentDensityExpected[0])
                                     .epsilon(tolerance)
                                     .scale(scale));
  CHECK(currentDensityMean[1] == doctest::Approx(currentDensityExpected[1])
                                     .epsilon(tolerance)
                                     .scale(scale));
}
} // namespace helper

// ================= Test current density <float> ================= //

TEST_CASE("Test current density, low temp <float>") {
  helper::testCurrentDensity(static_cast<float>(helper::TEMPERATURE_LOW));
}

TEST_CASE("Test current density, high temp <float>") {
  helper::testCurrentDensity(static_cast<float>(helper::TEMPERATURE_HIGH));
}

// ================= Test current density <double> ================= //

TEST_CASE("Test current density, low temp <double>") {
  helper::testCurrentDensity(helper::TEMPERATURE_LOW);
}

TEST_CASE("Test current density, high temp <double>") {
  helper::testCurrentDensity(helper::TEMPERATURE_HIGH);
}
