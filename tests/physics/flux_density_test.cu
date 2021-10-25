#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "BoundaryConditionSpecular.h"
#include "ComputeCurrent.h"
#include "ComputeFreeEnergy.h"
#include "ComputeOrderParameter.h"
#include "ComputeResidual.h"
#include "Context.h"
#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "LocalVectorPotential.h"
#include "OrderParameter.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h"
#include "Parameters.h"
#include "RiccatiSolverConfined.h"
#include "Type.h"
#include "defines.h"
#include "doctest_wrapper.h"
#include "run_utils.h"
#include "utils.h"

#include <cmath>
#include <limits>

namespace helper {
// Tolerances.
const double TOLERANCE_FLUX_DENSITY = 0.02;
const double TOLERANCE_VECTOR_POTENTIAL_RADIAL = 1e-7;
const double TOLERANCE_VECTOR_POTENTIAL_ANGULAR = 0.03;

/// \brief Test that induced vector potential comparing to analytical results.
///
/// \return Void.
template <typename T> void testFluxDensity() {
  // =================== ARRANGE ==================== //
  // The parameters entering the expression of the expected result.
  const T grainWidth = static_cast<T>(25);
  const T penetrationDepth = static_cast<T>(10);
  const T numFluxQuanta = static_cast<T>(0.01);

  conga::Context<T> context;
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(grainWidth);
  parameters->setPointsPerCoherenceLength(static_cast<T>(10));
  parameters->setAngularResolution(16);
  parameters->setConvergenceCriterion(static_cast<T>(1e-5));
  parameters->setTemperature(static_cast<T>(0.2));
  parameters->setEnergyCutoff(static_cast<T>(16));
  parameters->setNumFluxQuanta(numFluxQuanta);
  parameters->setPenetrationDepth(penetrationDepth);

  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);

  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
          &context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::bulk(orderParameterComponent);
  orderParameter->add(orderParameterComponent);

  context.set(std::make_unique<conga::Accelerator<T>>(static_cast<T>(0.01)));

  new conga::ComputeOrderParameter<T>(&context);
  new conga::ComputeCurrent<T>(&context);
  context.add(std::make_unique<conga::ComputeVectorPotential<T>>());

  // =================== ACT ==================== //
  context.initialize();
  conga::run_utils::runSimulation(&context);

  const T latticeSpacing = context.getParameters()->getGridElementSize();

  // The flux density.
  conga::GridGPU<T> testFluxDensityGPU =
      context.getInducedVectorPotential()->curl(latticeSpacing);
  context.getGeometry()->zeroValuesOutsideDomainRef(&testFluxDensityGPU);
  const conga::GridCPU<T> testFluxDensity = testFluxDensityGPU;

  // The vector potential.
  conga::GridGPU<T> testVectorPotentialGPU =
      *(context.getInducedVectorPotential());
  context.getGeometry()->zeroValuesOutsideDomainRef(&testVectorPotentialGPU);
  const conga::GridCPU<T> testVectorPotential = testVectorPotentialGPU;

  // =================== ASSERT ==================== //
  // It should have converged.
  CHECK(context.isConverged());

  // The pi factor comes from our choice of units for the flux density:
  // B_0 = \Phi_0 / (\pi \xi^2_0).
  const T area = context.getGeometry()->area();
  const T externalFluxDensity = static_cast<T>(M_PI) * numFluxQuanta / area;

  // For convenience.
  const T R = grainWidth * static_cast<T>(0.5);
  const int N = context.getParameters()->getGridResolutionBase();
  const T center = static_cast<T>(N - 1) * static_cast<T>(0.5);

  // Flux density.
  {
    // Expected flux-density grid.
    conga::GridCPU<T> expectedFluxDensity(N, N, 1, conga::Type::real);
    T *const ptr = expectedFluxDensity.getDataPointer();

    // Constant factor.
    const T factor = externalFluxDensity /
                     std::cyl_bessel_i(static_cast<T>(0), R / penetrationDepth);

    for (int j = 0; j < N; ++j) {
      // Y-coordinate relative to the center of the disc.
      const T y = latticeSpacing * (static_cast<T>(j) - center);
      for (int i = 0; i < N; ++i) {
        // X-coordinate relative to the center of the disc.
        const T x = latticeSpacing * (static_cast<T>(i) - center);

        // The distance from the center of the disc.
        const T r = std::hypot(x, y);

        // Whether a point is within the disk.
        const bool inDomain = r <= R;

        const T totalFluxDensity =
            factor * std::cyl_bessel_i(static_cast<T>(0), r / penetrationDepth);

        // Set.
        const int idx = j * N + i;
        ptr[idx] = inDomain ? (totalFluxDensity - externalFluxDensity)
                            : static_cast<T>(0);
      }
    }

    const T residual = (expectedFluxDensity - testFluxDensity).L2Norm() /
                       expectedFluxDensity.L2Norm();
    CHECK(residual < static_cast<T>(TOLERANCE_FLUX_DENSITY));
  }

  // Vector Potential
  {
    // Test vector-potential grid (\hat{r} and \hat{phi}).
    conga::GridCPU<T> testVectorPotentialR(N, N, 1, conga::Type::real);
    conga::GridCPU<T> testVectorPotentialPhi(N, N, 1, conga::Type::real);

    // Expected vector-potential grid (\hat{phi}).
    conga::GridCPU<T> expectedVectorPotentialPhi(N, N, 1, conga::Type::real);

    // Get the pointers.
    const T *const testPtrX = testVectorPotential.getDataPointer(0);
    const T *const testPtrY = testVectorPotential.getDataPointer(1);
    T *const testPtrR = testVectorPotentialR.getDataPointer();
    T *const testPtrPhi = testVectorPotentialPhi.getDataPointer();
    T *const expPtrPhi = expectedVectorPotentialPhi.getDataPointer();

    // Constant factor.
    const T factor = externalFluxDensity * penetrationDepth /
                     std::cyl_bessel_i(static_cast<T>(0), R / penetrationDepth);

    for (int j = 0; j < N; ++j) {
      // Y-coordinate relative to the center of the disc.
      const T y = latticeSpacing * (static_cast<T>(j) - center);
      for (int i = 0; i < N; ++i) {
        // X-coordinate relative to the center of the disc.
        const T x = latticeSpacing * (static_cast<T>(i) - center);

        // The distance from the center of the disc.
        const T r = std::hypot(x, y);

        // Whether a point is within the disk.
        const bool inDomain = r <= R;

        // Linear index.
        const int idx = j * N + i;

        // Expected total vector potential (\hat{phi}).
        expPtrPhi[idx] =
            inDomain ? (factor * std::cyl_bessel_i(static_cast<T>(1),
                                                   r / penetrationDepth))
                     : static_cast<T>(0);

        // External vector potential (carteesian).
        T externalVectorPotentialX;
        T externalVectorPotentialY;
        conga::VectorPotential::Radial<T>::computeVectorPotential_local(
            externalFluxDensity, area, x, y, externalVectorPotentialX,
            externalVectorPotentialY);

        // Test total vector potential (carteesian).
        const T totalVectorPotentialX =
            testPtrX[idx] + externalVectorPotentialX;
        const T totalVectorPotentialY =
            testPtrY[idx] + externalVectorPotentialY;

        // Expected total vector potential (\hat{r}).
        testPtrR[idx] =
            inDomain ? (totalVectorPotentialX * x + totalVectorPotentialY * y)
                     : static_cast<T>(0);

        // Expected total vector potential (\hat{phi}).
        testPtrPhi[idx] =
            inDomain
                ? (-totalVectorPotentialX * y + totalVectorPotentialY * x) /
                      std::max(r, static_cast<T>(1e-7))
                : static_cast<T>(0);
      }
    }

    // L-inf radial residual.
    T minR;
    T maxR;
    testVectorPotentialR.minmax(minR, maxR, 0, conga::Type::real);
    const T residualR = std::max(std::abs(minR), std::abs(maxR));

    // L2 angular residual.
    const T residualPhi =
        (expectedVectorPotentialPhi - testVectorPotentialPhi).L2Norm() /
        expectedVectorPotentialPhi.L2Norm();

    CHECK(residualR < static_cast<T>(TOLERANCE_VECTOR_POTENTIAL_RADIAL));
    CHECK(residualPhi < static_cast<T>(TOLERANCE_VECTOR_POTENTIAL_ANGULAR));
  }
}
} // namespace helper

// === Test flux density in a disc === //

TEST_CASE("Test flux density <float>") { helper::testFluxDensity<float>(); }

TEST_CASE("Test flux density <double>") { helper::testFluxDensity<double>(); }
