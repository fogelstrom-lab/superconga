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
#include "run_utils.h"
#include "utils.h"
#include <cmath>
#include <doctest/doctest.h>
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

  std::cout << "Fitting the penetration depth..." << std::endl;
  while (!isConverged) {
    // The flux from the London expression
    const T flux = factor * k * ((R - k) + k * std::exp(-R / k));

    // Residual.
    const T r = flux - inFlux;

    // The derivative of the residual with respect to the pentration depth.
    const T drdk = factor * ((R + static_cast<T>(2) * k) * std::exp(-R / k) +
                             R - static_cast<T>(2) * k);

    // The update of the penetration depth.
    const T dk = r * static_cast<T>(scg::utils::sign(drdk)) /
                 std::max(std::abs(drdk), std::numeric_limits<T>::epsilon());

    // Check convergence.
    isConverged = std::abs(dk / k) < convergenceCriterion;
    std::printf("|dk/k| = %e\n", std::abs(dk / k));

    // Update the penetration depth.
    k -= dk;
  }
  std::cout << "Done!" << std::endl;

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
  const T grainWidth = scg::utils::toInternalLength(static_cast<T>(20));
  const T penetrationDepth = scg::utils::toInternalLength(static_cast<T>(1));
  const T numFluxQuanta = static_cast<T>(0.5);

  scg::Context<T> context;
  scg::Parameters<T> *parameters = new scg::Parameters<T>(&context);
  parameters->setGrainWidth(grainWidth);
  parameters->setPointsPerCoherenceLength(
      scg::utils::toInternalInvLength(static_cast<T>(5)));
  parameters->setAngularResolution(16);
  parameters->setConvergenceCriterion(static_cast<T>(1e-5));
  parameters->setNumMatsubaraEnergies(16);
  parameters->setTemperature(static_cast<T>(0.1));
  parameters->setNumFluxQuanta(numFluxQuanta);
  parameters->setPenetrationDepth(penetrationDepth);

  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new scg::DiscGeometry<T>(grainFraction), scg::Type::add);

  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                     scg::VectorPotential::Radial>(&context);
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(&context);
  scg::OrderParameterComponent<T> *orderParameterComponent =
      new scg::OrderParameterComponent<T>(scg::BasisFunctionSymmetry::SWAVE,
                                          parameters);
  scg::Initial<T>::bulk(orderParameterComponent);
  orderParameter->add(orderParameterComponent);

  context.set(std::make_unique<scg::Optimizer<T>>(static_cast<T>(0.001)));

  new scg::ComputeOrderParameter<T>(&context);
  new scg::ComputeCurrent<T>(&context);
  context.add(std::make_unique<scg::ComputeVectorPotential<T>>());

  // =================== ACT ==================== //
  context.initialize();
  scg::run_utils::runSimulation(&context);

  const T area = context.getGeometry()->area();
  const T externalFluxDensity = numFluxQuanta / area;

  const T latticeElementSize = context.getParameters()->getGridElementSize();
  const scg::GridGPU<T> vectorPotential =
      *(context.getInducedVectorPotential());
  scg::GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
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
