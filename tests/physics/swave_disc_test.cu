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
#include "conga/run_iteration.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <cmath>

namespace helper {
/// \brief Helper function for computing the expected free energy.
///
/// \param inTemperature The temperature [Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
/// \param inPenetrationDepth The penetration depth [xi0].
/// \param inNumFluxQuanta The number of external flux quanta.
/// \param inRadius The radius of the disc [xi0].
///
/// \return The free energy.
template <typename T>
T computeFreeEnergyAnalytic(const T inTemperature, const T inEnergyCutoff,
                            const T inPenetrationDepth, const T inNumFluxQuanta,
                            const T inRadius) {
  // Compute bulk free energy.
  const T orderParameterBulk =
      conga::computeOrderParameterBulkSWave(inTemperature);
  const T freeEnergyBulk =
      conga::computeFreeEnergyBulkSWave(inTemperature, orderParameterBulk);

  // Yoshida function.
  const T yoshida = conga::computeYoshidaBulkSWave(
      inTemperature, orderParameterBulk, inEnergyCutoff);

  // For convenience.
  const T R = inRadius;
  const T k = inPenetrationDepth / std::sqrt(yoshida);
  const T area = static_cast<T>(M_PI) * R * R;

  // Common factor for the magnetic and kinetic terms.
  const T exteralFluxDensity = static_cast<T>(M_PI) * inNumFluxQuanta / area;
  const T angularFactor = static_cast<T>(2.0 * M_PI);
  const T factor = angularFactor * yoshida *
                   std::pow(k * exteralFluxDensity, 2) /
                   (static_cast<T>(2) * area);

  // Relevant Bessel functions.
  const T R_k = R / k;
  const T I0 = std::cyl_bessel_i(static_cast<T>(0), R_k);
  const T I1 = std::cyl_bessel_i(static_cast<T>(1), R_k);

  // Magnetic term.
  const T tmp = R * I1 / I0;
  const T freeEnergyMagnetic = factor * (R * R - static_cast<T>(2) * k * tmp -
                                         static_cast<T>(0.5) * tmp * tmp);

  // Kinetic term.
  const T freeEnergyKinetic = factor * static_cast<T>(0.5) * R *
                              (-R * I0 * I0 + 2 * k * I0 * I1 + R * I1 * I1) /
                              (I0 * I0);

  return freeEnergyBulk + freeEnergyKinetic + freeEnergyMagnetic;
}

/// \brief Test that the free energy of an s-wave superconductor in a disc, with
/// zero phase winding, behaves as expected. The free energy can be computed
/// analytically.
///
/// \return Void.
template <typename T> void testSWaveDisc() {
  // =================== ARRANGE ==================== //
  // The accuracy of the test.
  const T tolerance = static_cast<T>(1e-3);
  const T scale = static_cast<T>(0);

  // The radius of the disc.
  const T R = static_cast<T>(20);

  // General parameters.
  const T grainWidth = static_cast<T>(2) * R;
  const T pointsPerCoherenceLength = static_cast<T>(2.7);
  const int numMomenta = 16;
  const T convergenceCriterion = static_cast<T>(1e-3);
  const T temperature = static_cast<T>(0.2);
  const T energyCutoff = static_cast<T>(16);
  const T numFluxQuanta = static_cast<T>(5);
  const T penetrationDepth = static_cast<T>(5);
  const conga::ResidualNorm normType = conga::ResidualNorm::LInf;

  // =================== ACT ==================== //
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
  parameters->setPenetrationDepth(penetrationDepth);
  parameters->setResidualNormType(normType);

  // Create disc geometry.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Create solver with radial gauge.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::gauge::Symmetric>(&context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Create s-wave order parameter with phase winding.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::bulk(orderParameterComponent);
  orderParameter->add(orderParameterComponent);

  // Compute order parameter, current density, free energy, and vector
  // potential.
  new conga::ComputeOrderParameter<T>(&context);
  new conga::ComputeCurrent<T>(&context);
  new conga::ComputeFreeEnergy<T>(&context);
  context.add(std::make_unique<conga::ComputeVectorPotential<T>>());

  // Set accelerator.
  auto accelerator =
      std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
  context.set(std::move(accelerator));

  // Run the simulation.
  context.initialize();
  const bool isConverged = conga::runCompute(0, 100, true, context);

  // Save the result.
  const T freeEnergy =
      context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  // =================== ASSERT ==================== //
  const T freeEnergyExpected = computeFreeEnergyAnalytic(
      temperature, energyCutoff, penetrationDepth, numFluxQuanta, R);

  // Compare results.
  CHECK(freeEnergy ==
        doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));
  CHECK(isConverged);
}
} // namespace helper

TEST_CASE("Test s-wave disc - <float>") { helper::testSWaveDisc<float>(); }

TEST_CASE("Test s-wave disc - <double>") { helper::testSWaveDisc<double>(); }
