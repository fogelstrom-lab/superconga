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
#include "doctest_wrapper.h"
#include "run_utils.h"

#include <iostream>
#include <limits>
#include <stdlib.h>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.5;
// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;
// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 55.0 / GRAIN_SIDE_LENGTH;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 5;
// The energy cutoff.
const double ENERGY_CUTOFF = 16.0;
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
conga::Parameters<T> *applySimParams(conga::Context<T> *context) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  conga::Parameters<T> *parameters = new conga::Parameters<T>(context);
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
    conga::Context<T> *context = new conga::Context<T>();

    // Set parameters to the values specified in simParams struct
    conga::Parameters<T> *parameters = applySimParams<T>(context);

    // Create geometry group: add/remove geometry objects (e.g. discs,
    // polygons) from the geometry group
    conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(context);
    const T grainFracton = parameters->getGrainFraction();
    geometry->add(
        new conga::PolygonGeometry<T>(grainFracton, NUM_VERTICES_SQUARE),
        conga::Type::add);

    // Choose a Riccati solver for a confined geometry, without magnetic
    // fields
    conga::RiccatiSolver<T> *riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::None>(
            context);

    // Set up order parameter
    conga::OrderParameter<T> *orderParameter =
        new conga::OrderParameter<T>(context);

    // Add a d-wave or an s-wave order-parameter component.
    const conga::BasisFunctionSymmetry symmetry =
        useSwave ? conga::BasisFunctionSymmetry::SWAVE
                 : conga::BasisFunctionSymmetry::DWAVE_X2_Y2;
    conga::OrderParameterComponent<T> *orderParameterComponent =
        new conga::OrderParameterComponent<T>(symmetry, parameters);

    conga::Initial<T>::constant(orderParameterComponent, inStartRealPart,
                                inStartImaginaryPart);
    // Add the initialized s-wave component to the total order parameter
    orderParameter->add(orderParameterComponent);

    // Create compute objects
    conga::ComputeOrderParameter<T> *computeOrderParameter =
        new conga::ComputeOrderParameter<T>(context);

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
    conga::run_utils::cleanup(context);
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
