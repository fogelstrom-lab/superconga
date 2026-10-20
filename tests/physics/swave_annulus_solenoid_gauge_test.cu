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
#include <limits>
#include <vector>

namespace helper {
/// \brief Test that the free energy of an s-wave superconductor in an annulus,
/// with solenoid gauge, behaves as expected. I.e. that the free energy is
/// parabolic in the phase winding with minimas located at numFluxQuanta =
/// phaseWinding.
///
/// \return Void.
template <typename T> void testSWaveAnnulusSolenoid() {
  // =================== ARRANGE ==================== //
  // The accuracy of the test.
  const T tolerance = static_cast<T>(1e-3);
  const T scale = static_cast<T>(0);

  // The outer (R) and inner (r) radii of the annulus.
  const T R = static_cast<T>(20);
  const T r = static_cast<T>(10);

  // General parameters.
  // TODO(niclas): For large phase windings a very high spatial resolution is
  // needed. This needs to be investigated! Here a small phase winding and low
  // resolution is used in order for the test to not take forever.
  const T grainWidth = static_cast<T>(2) * R;
  const T pointsPerCoherenceLength = static_cast<T>(3.5);
  const int numMomenta = 16;
  const T convergenceCriterion = static_cast<T>(1e-4);
  const T temperature = static_cast<T>(0.2);
  const T energyCutoff = static_cast<T>(16);
  const T phaseWinding = static_cast<T>(1);
  const int chargeSign = -1;
  const T stepSize = static_cast<T>(1);
  const conga::ResidualNorm normType = conga::ResidualNorm::LInf;

  // For convenience.
  const T chargePhaseWinding = static_cast<T>(chargeSign) * phaseWinding;

  // Flux quantas to loop over.
  const std::vector<T> numFluxQuantas = {
      chargePhaseWinding - static_cast<T>(0.5), chargePhaseWinding,
      chargePhaseWinding + static_cast<T>(0.5)};

  // =================== ACT ==================== //
  // For storing the results.
  std::vector<T> freeEnergies;
  std::vector<bool> convergenceSuccess;

  // Loop over flux quantas and compute the free energy.
  for (const T numFluxQuanta : numFluxQuantas) {
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

    // Create annulus geometry.
    conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
    const T grainFraction = parameters->getGrainFraction();
    geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
    const T relativeHoleRadius =
        static_cast<T>(conga::constants::MAXIMUM_COORDINATE) * r / R;
    geometry->add(new conga::DiscGeometry<T>(grainFraction, relativeHoleRadius),
                  conga::Type::remove);

    // Integrator.
    new conga::IntegrationIteratorOzaki<T>(&context);

    // Create solver with solenoid gauge.
    conga::RiccatiSolver<T> *riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::gauge::Solenoid>(&context);
    riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

    // Create s-wave order parameter with phase winding.
    conga::OrderParameter<T> *orderParameter =
        new conga::OrderParameter<T>(&context);
    conga::OrderParameterComponent<T> *orderParameterComponent =
        new conga::OrderParameterComponent<T>(
            conga::BasisFunctionSymmetry::SWAVE, parameters);
    conga::Initial<T>::phaseWinding(orderParameterComponent, phaseWinding);
    orderParameter->add(orderParameterComponent);

    // Compute order parameter and free energy.
    new conga::ComputeOrderParameter<T>(&context);
    new conga::ComputeFreeEnergy<T>(&context);

    // Set accelerator.
    auto accelerator =
        std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
    context.set(std::move(accelerator));

    // Run the simulation.
    context.initialize();
    const bool isConverged = conga::runCompute(0, 100, true, context);
    convergenceSuccess.push_back(isConverged);

    // Save the result.
    const T freeEnergy =
        context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
    freeEnergies.push_back(freeEnergy);
  }

  // =================== ASSERT ==================== //
  // For the analytical results we need the bulk free energy.
  const T orderParameterBulk =
      conga::computeOrderParameterBulkSWave(temperature);
  const T freeEnergyBulk =
      conga::computeFreeEnergyBulkSWave(temperature, orderParameterBulk);

  // Common factor of the kinetic free energy.
  const T yoshida = conga::computeYoshidaBulkSWave(
      temperature, orderParameterBulk, energyCutoff);
  const T factor =
      static_cast<T>(0.25) * yoshida * std::log(R / r) / (R * R - r * r);

  // Loop over flux quantas and compare results.
  for (std::size_t i = 0; i < numFluxQuantas.size(); ++i) {
    // Form the kinetic free energy.
    const T numFluxQuanta = numFluxQuantas[i];
    const T freeEnergyKinetic =
        factor *
        static_cast<T>(std::pow(numFluxQuanta - chargePhaseWinding, 2));
    const T freeEnergyExpected = freeEnergyBulk + freeEnergyKinetic;

    // Compare results.
    CHECK(freeEnergies[i] ==
          doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));

    // Check that it converged.
    CHECK(convergenceSuccess[i]);
  }

  // Basic sanity check, the free energy should be lower at numFluxQuanta =
  // phaseWinding. This should always be true even if we have numerical errors.
  // Otherwise something is very wrong.
  CHECK(freeEnergies[1] < freeEnergies[0]);
  CHECK(freeEnergies[1] < freeEnergies[2]);
}
} // namespace helper

TEST_CASE("Test s-wave annulus (solenoid gauge) - <float>") {
  helper::testSWaveAnnulusSolenoid<float>();
}

TEST_CASE("Test s-wave annulus (solenoid gauge) - <double>") {
  helper::testSWaveAnnulusSolenoid<double>();
}
