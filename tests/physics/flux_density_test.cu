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
/// \brief Test that induced vector potential comparing to analytical results.
///
/// \param inTolerance The acceptable tolerance.
///
/// \return Void.
template <typename T> void testFluxDensity(const T inTolerance) {
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

  context.set(std::make_unique<conga::Accelerator<T>>(static_cast<T>(0.001)));

  new conga::ComputeOrderParameter<T>(&context);
  new conga::ComputeCurrent<T>(&context);
  context.add(std::make_unique<conga::ComputeVectorPotential<T>>());

  // =================== ACT ==================== //
  context.initialize();
  conga::run_utils::runSimulation(&context);

  const T latticeSpacing = context.getParameters()->getGridElementSize();
  // Compute the magnetic field.
  conga::GridGPU<T> testFluxDensityGPU =
      context.getInducedVectorPotential()->curl(latticeSpacing);
  context.getGeometry()->zeroValuesOutsideDomainRef(&testFluxDensityGPU);
  const conga::GridCPU<T> testFluxDensity = testFluxDensityGPU;

  // =================== ASSERT ==================== //
  // The pi factor comes from our choice of units for the flux density:
  // B_0 = \Phi_0 / (\pi \xi^2_0).
  const T area = context.getGeometry()->area();
  const T externalFluxDensity = static_cast<T>(M_PI) * numFluxQuanta / area;

  const int N = context.getParameters()->getGridResolutionBase();
  conga::GridCPU<T> expectedFluxDensity(N, N, 1, conga::Type::real);

  // The radius and center coordinate of a disk.
  const T R = grainWidth * static_cast<T>(0.5);
  const T center = static_cast<T>(N - 1) * static_cast<T>(0.5);

  // Constant factor.
  const T ratio = R / penetrationDepth;
  const T numer =
      static_cast<T>(2) * externalFluxDensity * R * penetrationDepth;
  const T denom = R * std::cyl_bessel_i(static_cast<T>(0), ratio) +
                  static_cast<T>(2) * penetrationDepth *
                      std::cyl_bessel_i(static_cast<T>(1), ratio) +
                  R * std::cyl_bessel_i(static_cast<T>(2), ratio);
  const T factor = numer / denom;

  // Get the pointer.
  T *const ptr = expectedFluxDensity.getDataPointer();

  for (int j = 0; j < N; ++j) {
    // Y-coordinate relative to the center of the disc.
    const T y = latticeSpacing * (static_cast<T>(j) - center);
    for (int i = 0; i < N; ++i) {
      // Y-coordinate relative to the center of the disc.
      const T x = latticeSpacing * (static_cast<T>(i) - center);

      // The distance from the center of the disc.
      const T r = std::hypot(x, y);

      // Whether a point is within the disk.
      const bool inDomain = r <= R;

      const T v = r / penetrationDepth;
      const T val = std::cyl_bessel_i(static_cast<T>(1), v) /
                        std::max(r, static_cast<T>(1e-6)) +
                    (std::cyl_bessel_i(static_cast<T>(0), v) +
                     std::cyl_bessel_i(static_cast<T>(2), v)) /
                        (static_cast<T>(2) * penetrationDepth);
      // Set.
      const int idx = j * N + i;
      ptr[idx] =
          inDomain ? (val * factor - externalFluxDensity) : static_cast<T>(0);
    }
  }

  const T residual = (expectedFluxDensity - testFluxDensity).L2Norm() /
                     expectedFluxDensity.L2Norm();
  CHECK(residual < inTolerance);
  CHECK(context.isConverged());
}
} // namespace helper

// === Test flux density in a disc === //

TEST_CASE("Test flux density <float>") {
  const float tolerance = 0.02f;
  helper::testFluxDensity(tolerance);
}

TEST_CASE("Test flux density <double>") {
  const double tolerance = 0.02;
  helper::testFluxDensity(tolerance);
}
