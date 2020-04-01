#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "DiscGeometry.h" // Used to constructed simulated grain from discs
#include "FermiSurfaceCircle.h" // Standard circle Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterComponent_D.h" // d-wave
#include "OrderParameterComponent_S.h" // s-wave
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "defines.h"

#include <doctest.h>
#include <iostream>

// ========================= Simulation parameters ========================= //
namespace simParams {
// Size of superconducting grain in coherence lengths hbar*vF/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 10.0;
// Number of lattice points per grain side length
const int GRID_RESOLUTION = 201;
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
template <typename T> Parameters<T> *applySimParams(Context<T> *context) {
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set grain resolution in number of lattice sites
  parameters->setGridResolution(simParams::GRID_RESOLUTION);
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc)
  parameters->setGrainWidthCoherenceLengths(
      static_cast<T>(simParams::GRAIN_SIDE_LENGTH));
  return parameters;
}

// Cleanup of context objects.
template <typename T> void cleanup(Context<T> *context) { delete context; }

// ======================== Wrappers for test cases ======================== //
// **** Test that order parameter grid is normalized properly **** /
template <typename T>
void testIntialConstant(const T inTolerance, const int inNumValues = 4) {

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
      Context<T> *context = new Context<T>();

      // Set parameters to the values specified in simParams struct
      Parameters<T> *parameters = applySimParams<T>(context);

      // Create geometry group: add/remove geometry objects (e.g. discs,
      // polygons) from the geometry group
      GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

      switch (grainShape) {
      case GrainShape::disc: {
        const T discGrainScale = 0.8;
        geometry->add(new DiscGeometry<T>(discGrainScale), Type::add);
        break;
      }
      case GrainShape::square: {
        const int numVerticesSquare = 4;
        geometry->add(new PolygonGeometry<T>(numVerticesSquare), Type::add);
        break;
      }
      case GrainShape::octagon: {
        const int numVerticesOctagon = 8;
        const T octagonGrainScale = 0.75;
        geometry->add(
            new PolygonGeometry<T>(numVerticesOctagon, octagonGrainScale),
            Type::add);
        break;
      }
      case GrainShape::irregular: {
        // Add irregular pentagon
        const int numVertices = 5;
        const T xVertices[numVertices] = {0.9, 0.0, -1.0, -0.8, 0.5};
        const T yVertices[numVertices] = {0.8, 1.0, 0.0, -0.8, -0.6};
        geometry->add(new PolygonGeometry<T>(numVertices, xVertices, yVertices),
                      Type::add);
        // Remove small disc
        const T discGrainScale = 0.2;
        const T discCenterX = 0.2;
        const T discCenterY = 0.1;
        geometry->add(
            new DiscGeometry<T>(discGrainScale, discCenterX, discCenterY),
            Type::remove);
        break;
      }
      }

      // Choose a Riccati solver for a confined geometry, no magnetic fields
      RiccatiSolver<T> *riccatiSolver =
          new RiccatiSolverConfined<T, Gamma::General, VectorPotential::None>(
              context);

      // Choose FermiSurface
      FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
      context->set(fermiSurface);

      // Set up order parameter
      OrderParameter<T> *orderParameter = new OrderParameter<T>(context);
      OrderParameterComponent<T> *orderParameterComponent;
      if (useSwave) {
        orderParameterComponent = new OrderParameterComponent_S<T>(parameters);
      } else {
        orderParameterComponent = new OrderParameterComponent_D<T>(parameters);
      }

      // Temporary initialization has to be done before adding it to the order
      // parameter. This is because we have to add it outside the inner for
      // loops.
      Initial<T>::constant(orderParameterComponent);
      // Add the initialized s-wave component to the total order parameter
      orderParameter->add(orderParameterComponent);

      // Create compute objects
      ComputeOrderParameter<T> *computeOrderParameter =
          new ComputeOrderParameter<T>(context);

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
            Initial<T>::constant(orderParameterComponent, realPartValue,
                                 imaginaryPartValue);

            // Initialize simulation
            context->initialize();

            // Remove all spurious values outside actual grain
            context->getGeometry()->zeroValuesOutsideDomain(
                orderParameterComponent->getOrderParameterField());

            // Fetch the grain-summed order-parameter magnitude
            const T testSumOrderParameter =
                orderParameterComponent->getOrderParameterField()
                    ->sumMagnitude();
            // Fetch the number of elements in the grain
            const T numElementsOrderParameter = geometry->area(1);

            // =================== ASSERT ==================== //
            const T expectedSumOrderParameter =
                numElementsOrderParameter *
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
