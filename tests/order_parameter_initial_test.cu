#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "DiscGeometry.h" // Used to constructed simulated grain from discs
#include "FermiSurfaceCircle.h" // Standard circle Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "defines.h"

#include <doctest/doctest.h>
#include <iostream>

// ========================= Simulation parameters ========================= //
namespace simParams {
// Size of superconducting grain in coherence lengths hbar*<vF>/(kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;
// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 10.0;
} // namespace simParams

namespace helper {
// ========================= Test helper objects ========================= //
enum class GrainShape { disc, square, octagon, irregular };
// Make GrainShapes iterable
const int NUM_GRAIN_SHAPES = 4;
const std::array<GrainShape, NUM_GRAIN_SHAPES> allGrainShapes = {
    GrainShape::disc, GrainShape::square, GrainShape::octagon,
    GrainShape::irregular};

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
  return parameters;
}

// Cleanup of context objects.
template <typename T> void cleanup(scg::Context<T> *context) { delete context; }

// ======================== Wrappers for test cases ======================== //
// **** Test that order parameter grid is normalized properly **** /
template <typename T>
void testIntialConstant(const T inTolerance, const int inNumValues = 2) {

  // Set up maxima for values to try. Values are sampled from
  // [-maxValue,maxValue] uniformly.
  const int numMaxValues = 2;
  const std::array<T, numMaxValues> maxValues = {static_cast<T>(1e-3),
                                                 static_cast<T>(1e3)};

  // Loop over s-wave and d-wave
  for (const bool useSwave : {true, false}) {
    // Loop over different shapes
    for (GrainShape grainShape : allGrainShapes) {
      // =================== ARRANGE ==================== //
      // Create Context
      scg::Context<T> *context = new scg::Context<T>();

      // Set parameters to the values specified in simParams struct
      scg::Parameters<T> *parameters = applySimParams<T>(context);

      // Create geometry group: add/remove geometry objects (e.g. discs,
      // polygons) from the geometry group
      scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
      const T grainFraction = parameters->getGrainFraction();
      switch (grainShape) {
      case GrainShape::disc: {
        const T discGrainScale = static_cast<T>(0.8);
        geometry->add(new scg::DiscGeometry<T>(grainFraction, discGrainScale),
                      scg::Type::add);
        break;
      }
      case GrainShape::square: {
        const int numVerticesSquare = 4;
        geometry->add(
            new scg::PolygonGeometry<T>(grainFraction, numVerticesSquare),
            scg::Type::add);
        break;
      }
      case GrainShape::octagon: {
        const int numVerticesOctagon = 8;
        const T octagonGrainScale = static_cast<T>(0.75);
        geometry->add(new scg::PolygonGeometry<T>(
                          grainFraction, numVerticesOctagon, octagonGrainScale),
                      scg::Type::add);
        break;
      }
      case GrainShape::irregular: {
        // Add irregular pentagon
        const int numVertices = 5;
        const T xVertices[numVertices] = {
            static_cast<T>(0.9 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(0.0 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(-1.0 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(-0.8 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(0.5 * scg::constants::MAXIMUM_COORDINATE)};
        const T yVertices[numVertices] = {
            static_cast<T>(0.8 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(1.0 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(0.0 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(-0.8 * scg::constants::MAXIMUM_COORDINATE),
            static_cast<T>(-0.6 * scg::constants::MAXIMUM_COORDINATE)};
        geometry->add(new scg::PolygonGeometry<T>(grainFraction, numVertices,
                                                  xVertices, yVertices),
                      scg::Type::add);
        // Remove small disc
        const T discGrainScale = static_cast<T>(0.2);
        const T discCenterX =
            static_cast<T>(0.2 * scg::constants::MAXIMUM_COORDINATE);
        const T discCenterY =
            static_cast<T>(0.1 * scg::constants::MAXIMUM_COORDINATE);
        geometry->add(new scg::DiscGeometry<T>(grainFraction, discGrainScale,
                                               discCenterX, discCenterY),
                      scg::Type::remove);
        break;
      }
      }

      // Choose a Riccati solver for a confined geometry, no magnetic fields
      scg::RiccatiSolver<T> *riccatiSolver =
          new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                         scg::VectorPotential::None>(context);

      // Choose FermiSurface
      scg::FermiSurfaceCircle<T> *fermiSurface =
          new scg::FermiSurfaceCircle<T>();
      context->set(fermiSurface);

      // Set up order parameter
      scg::OrderParameter<T> *orderParameter =
          new scg::OrderParameter<T>(context);
      const scg::BasisFunctionSymmetry symmetry =
          useSwave ? scg::BasisFunctionSymmetry::SWAVE
                   : scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
      scg::OrderParameterComponent<T> *orderParameterComponent =
          new scg::OrderParameterComponent<T>(symmetry, parameters);

      // Temporary initialization has to be done before adding it to the order
      // parameter. This is because we have to add it outside the inner for
      // loops.
      scg::Initial<T>::constant(orderParameterComponent);
      // Add the initialized s-wave component to the total order parameter
      orderParameter->add(orderParameterComponent);

      // Create compute objects
      scg::ComputeOrderParameter<T> *computeOrderParameter =
          new scg::ComputeOrderParameter<T>(context);

      // Loop over maxVales (small and large initialization values)
      for (const T &maxValue : maxValues) {
        const T minValue = -maxValue;

        // Stepsize between each different value to try
        const T valueStep =
            (maxValue - minValue) / static_cast<T>(inNumValues - 1);

        // Vary the real part
        for (int realPartIdx = 0; realPartIdx < inNumValues; ++realPartIdx) {
          // Update real part
          const T realPartValue =
              valueStep * static_cast<T>(realPartIdx) + minValue;
          // Vary the imaginary part
          for (int imaginaryPartIdx = 0; imaginaryPartIdx < inNumValues;
               ++imaginaryPartIdx) {
            // =================== ACT ==================== //
            // Update imaginary part
            const T imaginaryPartValue =
                valueStep * static_cast<T>(imaginaryPartIdx) + minValue;

            // Initialize order-parameter component to the (real) bulk value
            scg::Initial<T>::constant(orderParameterComponent, realPartValue,
                                      imaginaryPartValue);

            // Initialize simulation
            context->initialize();

            // Remove all spurious values outside actual grain
            context->getGeometry()->zeroValuesOutsideDomainRef(
                orderParameterComponent->getOrderParameterField());

            // Fetch the grain-summed order-parameter magnitude
            const T testSumOrderParameter =
                orderParameterComponent->getOrderParameterField()
                    ->sumMagnitude();
            // Fetch the number of elements in the grain
            const int numGrainLatticeSites =
                geometry->getNumGrainLatticeSites();

            // =================== ASSERT ==================== //
            const T expectedSumOrderParameter =
                static_cast<T>(numGrainLatticeSites) *
                std::hypot(realPartValue, imaginaryPartValue);

            CHECK(testSumOrderParameter ==
                  doctest::Approx(expectedSumOrderParameter)
                      .epsilon(inTolerance));
          }
        }
      }
      // Clean up context
      cleanup(context);
    }
  }
}

} // namespace helper

// ================= Test grid normalization ================= //
TEST_CASE("Test Initial<float>::Constant order parameter value") {
  const float toleranceMultFactor = 10.0F;
  const float tolerance = helper::computeTolerance<float>(toleranceMultFactor);

  helper::testIntialConstant(tolerance);
}

TEST_CASE("Test Initial<double>::Constant order parameter value") {
  const double toleranceMultFactor = 10.0;
  const double tolerance =
      helper::computeTolerance<double>(toleranceMultFactor);

  helper::testIntialConstant(tolerance);
}
