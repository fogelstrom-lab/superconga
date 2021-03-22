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
const double TEMPERATURE = 0.5;
// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;
// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 55.0 / GRAIN_SIDE_LENGTH;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 5;
// Number of discrete energies (poles)
const int NUM_ENERGIES = 10;
// Convergence criterion for order-parameter residual
const double CONVERGENCE_CRITERION = 1E-6;
} // namespace simParams

namespace helper {
// ========================= Test helper objects ========================= //
const int NUM_VERTICES_SQUARE = 4;

// ========================= Test helper functions ========================= //
// Compute tolerance for GridData testing, given the number of elements in a
// GridData object. Note: inToleranceBaseFactor has to be > 1.
template <typename T>
T computeTolerance(const T inToleranceBaseFactor = static_cast<T>(5.0)) {
  return inToleranceBaseFactor * std::numeric_limits<T>::epsilon();
}

// Helper class for setting up simulation parameters
template <typename T>
scg::Parameters<T> *applySimParams(scg::Context<T> *context) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
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
    scg::Context<T> *context = new scg::Context<T>();

    // Set parameters to the values specified in simParams struct
    scg::Parameters<T> *parameters = applySimParams<T>(context);

    // Create geometry group: add/remove geometry objects (e.g. discs,
    // polygons) from the geometry group
    scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
    const T grainFracton = parameters->getGrainFraction();
    geometry->add(
        new scg::PolygonGeometry<T>(grainFracton, NUM_VERTICES_SQUARE),
        scg::Type::add);

    // Choose a Riccati solver for a confined geometry, without magnetic
    // fields
    scg::RiccatiSolver<T> *riccatiSolver =
        new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                       scg::VectorPotential::None>(context);

    // Set up order parameter
    scg::OrderParameter<T> *orderParameter =
        new scg::OrderParameter<T>(context);

    // Add a d-wave or an s-wave order-parameter component.
    const scg::BasisFunctionSymmetry symmetry =
        useSwave ? scg::BasisFunctionSymmetry::SWAVE
                 : scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
    scg::OrderParameterComponent<T> *orderParameterComponent =
        new scg::OrderParameterComponent<T>(symmetry, parameters);

    scg::Initial<T>::constant(orderParameterComponent, inStartRealPart,
                              inStartImaginaryPart);
    // Add the initialized s-wave component to the total order parameter
    orderParameter->add(orderParameterComponent);

    // Create compute objects
    scg::ComputeOrderParameter<T> *computeOrderParameter =
        new scg::ComputeOrderParameter<T>(context);

    // =================== ACT ==================== //
    // Initialize simulation
    context->initialize();

    // Remove all spurious values outside actual grain
    context->getGeometry()->zeroValuesOutsideDomainRef(
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

    CHECK(testSumOrderParameter ==
          doctest::Approx(expectedSumOrderParameter).epsilon(inTolerance));

    // Clean up context
    scg::run_utils::cleanup(context);
  }
}
} // namespace helper

// ================= Test grid normalization ================= //
TEST_CASE("Test Initial<float>::Constant order parameter value") {
  const float toleranceMultFactor = 10.0f;
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
