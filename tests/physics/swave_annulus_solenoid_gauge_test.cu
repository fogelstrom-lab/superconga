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
#include "OrderParameterBulk.h"
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
#include <vector>

namespace helper {
/// \brief Compute the Yoshida function Y_{3/2} at a given temperature.
///
/// \param inTemperature The temperature [Tc].
/// \param inOrderParameter The bulk order-parameter [2*pi*kB*Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
///
/// \return Y_{3/2}(T).
template <typename T>
T yoshida(const T inTemperature, const T inOrderParameter,
          const T inEnergyCutoff) {
  const int numEnergies =
      static_cast<int>(inEnergyCutoff / inTemperature - static_cast<T>(0.5)) +
      1;

  T energySum = static_cast<T>(0);
  for (int n = numEnergies - 1; n >= 0; --n) {
    const T matsubaraEnergy =
        inTemperature * (static_cast<T>(n) + static_cast<T>(0.5));
    const T Omega = std::hypot(inOrderParameter, matsubaraEnergy);
    energySum += static_cast<T>(1) / std::pow(Omega, 3);
  }

  return inTemperature * inOrderParameter * inOrderParameter * energySum;
}

/// \brief Test that the free energy of an s-wave superconductor in an annulus,
/// with solenoid gauge, behaves as expected. I.e. that the free energy is
/// parabolic in the phase winding with minimas located at numFluxQuanta =
/// phaseWinding.
///
/// \return Void.
template <typename T> void testFluxQuantization() {
  // =================== ARRANGE ==================== //
  // The accuracy of the test.
  const T tolerance = static_cast<T>(1e-3);
  const T scale = static_cast<T>(0);

  // The outer (R) and inner (r) radii of the annulus.
  const T R = static_cast<T>(25);
  const T r = static_cast<T>(10);

  // General parameters.
  const T grainWidth = static_cast<T>(2) * R;
  const T pointsPerCoherenceLength = static_cast<T>(3);
  const int numMomenta = 16;
  const T convergenceCriterion = static_cast<T>(1e-5);
  const T temperature = static_cast<T>(0.2);
  const T energyCutoff = static_cast<T>(16);
  const T phaseWinding = static_cast<T>(1);
  const T stepSize = static_cast<T>(0.001);

  // Flux quantas to loop over.
  const std::vector<T> numFluxQuantas = {static_cast<T>(0.5), static_cast<T>(1),
                                         static_cast<T>(1.5)};

  // =================== ACT ==================== //
  // For storing the results.
  std::vector<T> freeEnergies;

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

    // Create annulus geometry.
    conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
    const T grainFraction = parameters->getGrainFraction();
    geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
    const T holeFraction = r / R;
    geometry->add(new conga::DiscGeometry<T>(grainFraction, holeFraction),
                  conga::Type::remove);

    // Create solver with solenoid gauge.
    conga::RiccatiSolver<T> *riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Solenoid>(
            &context);
    riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

    // Create s-wave order parameter with phase winding.
    conga::OrderParameter<T> *orderParameter =
        new conga::OrderParameter<T>(&context);
    conga::OrderParameterComponent<T> *orderParameterComponent =
        new conga::OrderParameterComponent<T>(
            conga::BasisFunctionSymmetry::SWAVE, parameters);
    conga::Initial<T>::phaseWinding(orderParameterComponent, phaseWinding);
    orderParameter->add(orderParameterComponent);

    // Create accelerator.
    context.set(std::make_unique<conga::Accelerator<T>>(stepSize));

    // Compute order parameter and free energy.
    new conga::ComputeOrderParameter<T>(&context);
    new conga::ComputeFreeEnergy<T>(&context);

    // Run the simulation.
    context.initialize();
    conga::run_utils::runSimulation(&context);

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
  const T Y = yoshida(temperature, orderParameterBulk, energyCutoff);
  const T factor = static_cast<T>(0.25) * Y * std::log(R / r) / (R * R - r * r);

  // Loop over flux quantas and compare results.
  for (std::size_t i = 0; i < numFluxQuantas.size(); ++i) {
    // Form the kinetic free energy.
    const T numFluxQuanta = numFluxQuantas[i];
    const T freeEnergyKinetic =
        factor * std::pow(numFluxQuanta - phaseWinding, 2);
    const T freeEnergyExpected = freeEnergyBulk + freeEnergyKinetic;

    // Compare results.
    CHECK(freeEnergies[i] ==
          doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));
  }

  // Basic sanity check, the free energy should be lower at numFluxQuanta =
  // phaseWinding. This should always be true even if we have numerical errors.
  // Otherwise something is very wrong.
  CHECK(freeEnergies[1] < freeEnergies[0]);
  CHECK(freeEnergies[1] < freeEnergies[2]);
}
} // namespace helper

TEST_CASE("Test flux quantization <float>") {
  helper::testFluxQuantization<float>();
}

TEST_CASE("Test flux quantization <double>") {
  helper::testFluxQuantization<double>();
}
