#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/accelerators/BarzilaiBorwein.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/defines.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterBulk.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <cmath>

namespace helper {
/// \brief Test that charge-current density has the correct sign with respect to
/// the phase gradient.
///
/// \return Void.
template <typename T> void testPhaseGradient() {
  // =================== ARRANGE ==================== //
  // General parameters.
  const T grainWidth = static_cast<T>(10);
  const T pointsPerCoherenceLength = static_cast<T>(2.0);
  const int numMomenta = 16;
  const T convergenceCriterion = static_cast<T>(1e-5);
  const T temperature = static_cast<T>(0.9);
  const T energyCutoff = static_cast<T>(16);
  const T phaseWinding = static_cast<T>(1);
  const T numFluxQuanta = static_cast<T>(0);
  const int chargeSign = -1;
  const conga::ResidualNorm normType = conga::ResidualNorm::LInf;

  // Create context.
  conga::Context<T> context;

  // Set parameters.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(grainWidth);
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);
  parameters->setAngularResolution(numMomenta);
  parameters->setConvergenceCriterion(convergenceCriterion);
  parameters->setTemperature(temperature);
  parameters->setEnergyCutoff(energyCutoff);
  parameters->setNumFluxQuanta(numFluxQuanta);
  parameters->setChargeSign(chargeSign);
  parameters->setResidualNormType(normType);

  // Create disc geometry.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Create solver.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::None>(
          &context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Create s-wave order parameter with phase winding.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::phaseWinding(orderParameterComponent, phaseWinding);
  orderParameter->add(orderParameterComponent);

  // Set accelerator.
  auto accelerator =
      std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
  context.set(std::move(accelerator));

  // Compute charge-current density.
  new conga::ComputeCurrent<T>(&context);

  // =================== ACT ==================== //
  // Run the simulation one iteration.
  context.initialize();
  context.compute();

  // Get the current density.
  const conga::GridGPU<T> &currentDensity =
      *(context.getComputeCurrent()->getResult());

  // Compute the phase gradient of the order parameter.
  const conga::GridGPU<T> phaseGradient =
      orderParameter->getComponentGrid(0)->complexPhaseGradient();

  // Compute the (unnormalized) overlap.
  const conga::GridGPU<T> overlap = currentDensity * phaseGradient;

  // =================== ASSERT ==================== //
  // With no vector potential the current is related to the phase gradient via:
  //
  // j \propto q * \nabla * \phi
  //
  // where q is the carrier charge.
  const int expectedSign = chargeSign * conga::utils::sign(phaseWinding);

  // Loop over xy-components.
  for (int i = 0; i < 2; ++i) {
    const auto [overlapMin, overlapMax] = overlap.minmax(i);

    // Check that the overlap does not have the wrong sign (zero is ok and
    // expected).
    CHECK(conga::utils::sign(overlapMin) != -expectedSign);
    CHECK(conga::utils::sign(overlapMax) != -expectedSign);
  }
}
} // namespace helper

TEST_CASE("Test phase gradient - <float>") {
  helper::testPhaseGradient<float>();
}

TEST_CASE("Test phase gradient - <double>") {
  helper::testPhaseGradient<double>();
}
