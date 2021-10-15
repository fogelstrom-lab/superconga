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
#include "RiccatiSolverConfined.h"
#include "Type.h"
#include "defines.h"
#include "doctest_wrapper.h"
#include "run_utils.h"
#include "utils.h"

#include <cmath>
#include <limits>

namespace helper {
/// \brief Fit the the penetration depth to the flux via the London expression.
///
/// \param inPenetrationDepth The expected penetration depth (it is used as the
/// initial value).
/// \param inExternalFluxDensity The external flux-density.
/// \param inGrainWidth The width of the grain.
/// \param inFlux The flux through the grain.
///
/// \return The fitted penetration depth.
template <typename T>
T fitPenetrationDepth(const T inPenetrationDepth, const T inExternalFluxDensity,
                      const T inGrainWidth, const T inFlux) {
  // The radius of the disc.
  const T R = inGrainWidth * static_cast<T>(0.5);

  // Common factor. For convenience.
  const T factor = static_cast<T>(2.0 * M_PI) * inExternalFluxDensity;

  // Convergence criterion.
  const T convergenceCriterion = static_cast<T>(1e-6);

  // Compute the effective penetration depth by fitting it to the result via the
  // London expression.
  T k = inPenetrationDepth;

  bool isConverged = false;

  std::cout << "-- Fitting the penetration depth..." << std::endl;
  while (!isConverged) {
    // The flux from the London expression
    const T flux = factor * k * ((R - k) + k * std::exp(-R / k));

    // Residual.
    const T r = flux - inFlux;

    // The derivative of the residual with respect to the pentration depth.
    const T drdk = factor * ((R + static_cast<T>(2) * k) * std::exp(-R / k) +
                             R - static_cast<T>(2) * k);

    // The update of the penetration depth.
    const T dk = r * static_cast<T>(conga::utils::sign(drdk)) /
                 std::max(std::abs(drdk), std::numeric_limits<T>::epsilon());

    // Check convergence.
    isConverged = std::abs(dk / k) < convergenceCriterion;
    std::printf("|dk/k| = %e\n", std::abs(dk / k));

    // Update the penetration depth.
    k -= dk;
  }
  std::cout << "-- Done!" << std::endl;

  return k;
}

/// \brief Test that the flux density in a disc in a weak magnetic field is
/// similar to the London expression.
///
/// \param inTolerance The acceptable tolerance.
///
/// \return Void.
template <typename T> void testFluxDensity(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // The parameters entering the expression of the expected result.
  const T grainWidth = static_cast<T>(20);
  const T penetrationDepth = static_cast<T>(1);
  const T numFluxQuanta = static_cast<T>(0.5);

  conga::Context<T> context;
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(grainWidth);
  parameters->setPointsPerCoherenceLength(static_cast<T>(5));
  parameters->setAngularResolution(16);
  parameters->setConvergenceCriterion(static_cast<T>(1e-5));
  parameters->setTemperature(static_cast<T>(0.1));
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

  const T area = context.getGeometry()->area();
  // The pi factor comes from our choice of units for the flux density:
  // B_0 = \Phi_0 / (\pi \xi^2_0).
  const T externalFluxDensity = static_cast<T>(M_PI) * numFluxQuanta / area;

  const T latticeElementSize = context.getParameters()->getGridElementSize();
  const conga::GridGPU<T> vectorPotential =
      *(context.getInducedVectorPotential());
  conga::GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
  context.getGeometry()->zeroValuesOutsideDomainRef(&fluxDensity);
  const int numGrainLatticeSites =
      context.getGeometry()->getNumGrainLatticeSites();
  const T flux =
      area * (externalFluxDensity +
              fluxDensity.sumField() / static_cast<T>(numGrainLatticeSites));

  const T testPenetrationDepth = fitPenetrationDepth(
      penetrationDepth, externalFluxDensity, grainWidth, flux);

  // =================== ASSERT ==================== //
  // Note, the way doctest compares two floating point numbers, A and B, is the
  // following: |A - B| < epsilon * (scale + max(|A|, |B|))
  CHECK(penetrationDepth == doctest::Approx(testPenetrationDepth)
                                .epsilon(inTolerance)
                                .scale(static_cast<T>(0)));
  CHECK(context.isConverged());
}
} // namespace helper

// === Test flux density in a disc === //

TEST_CASE("Test flux density (float)") {
  // TODO(niclas): Is this really an acceptable tolerance?
  const float tolerance = 0.1f;
  helper::testFluxDensity(tolerance);
}

TEST_CASE("Test flux density (double)") {
  // TODO(niclas): Is this really an acceptable tolerance?
  const double tolerance = 0.1;
  helper::testFluxDensity(tolerance);
}
