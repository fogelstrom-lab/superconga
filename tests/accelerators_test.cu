#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/accelerators.h"
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

namespace helper {
/// \brief Enum class for the accelerators.
enum class AcceleratorType {
  ANDERSON,
  BARZILAI_BORWEIN,
  CONGACC,
  PICARD,
  POLYAK
};

/// \brief Just some simple test checking that the accelerators converge.
///
/// \param AcceleratorType Which accelerator to test.
///
/// \return Void.
template <typename T>
void testAccelerator(const AcceleratorType &inAcceleratorType) {
  // =================== ARRANGE ==================== //
  // The accuracy of the test.
  const T tolerance = static_cast<T>(1e-3);
  const T scale = static_cast<T>(0);

  // Max number of iterations.
  const int maxIterations = 100;

  // General parameters.
  const T grainWidth = static_cast<T>(10);
  const T pointsPerCoherenceLength = static_cast<T>(3);
  const int numMomenta = 16;
  const T convergenceCriterion = static_cast<T>(1e-5);
  const T temperature = static_cast<T>(0.5);
  const T energyCutoff = static_cast<T>(16);
  const T numFluxQuanta = static_cast<T>(1);
  const T penetrationDepth = static_cast<T>(10);
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
  std::unique_ptr<conga::accelerators::Accelerator<T>> accelerator;
  switch (inAcceleratorType) {
  case AcceleratorType::ANDERSON: {
    accelerator = std::make_unique<conga::accelerators::Anderson<T>>();
    break;
  }
  case AcceleratorType::BARZILAI_BORWEIN: {
    accelerator = std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
    break;
  }
  case AcceleratorType::CONGACC: {
    accelerator = std::make_unique<conga::accelerators::CongAcc<T>>();
    break;
  }
  case AcceleratorType::PICARD: {
    accelerator = std::make_unique<conga::accelerators::Picard<T>>();
    break;
  }
  case AcceleratorType::POLYAK: {
    accelerator = std::make_unique<conga::accelerators::Polyak<T>>();
  }
  }
  context.set(std::move(accelerator));

  // Run the simulation.
  context.initialize();
  const bool isConverged = conga::runCompute(0, maxIterations, true, context);

  // Save the result.
  const T freeEnergy =
      context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();

  // =================== ASSERT ==================== //
  // The expected value. Just to check that they converge to the same solution.
  const T freeEnergyExpected = static_cast<T>(-0.0185386592);

  // Compare results.
  CHECK(freeEnergy ==
        doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));

  // Check convergence.
  CHECK(isConverged);
}
} // namespace helper

// ================ CongAcc ================ //

TEST_CASE("Test CongAcc - <float>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::CONGACC;
  helper::testAccelerator<float>(acceleratorType);
}

TEST_CASE("Test CongAcc - <double>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::CONGACC;
  helper::testAccelerator<double>(acceleratorType);
}

// ================ Anderson ================ //

TEST_CASE("Test Anderson - <float>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::ANDERSON;
  helper::testAccelerator<float>(acceleratorType);
}

TEST_CASE("Test Anderson- <double>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::ANDERSON;
  helper::testAccelerator<double>(acceleratorType);
}

// ================ Barzilai-Borwein ================ //

TEST_CASE("Test Barzilai-Borwein - <float>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::BARZILAI_BORWEIN;
  helper::testAccelerator<float>(acceleratorType);
}

TEST_CASE("Test Barzilai-Borwein - <double>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::BARZILAI_BORWEIN;
  helper::testAccelerator<double>(acceleratorType);
}

// ================ Picard ================ //

TEST_CASE("Test Picard - <float>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::PICARD;
  helper::testAccelerator<float>(acceleratorType);
}

TEST_CASE("Test Picard - <double>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::PICARD;
  helper::testAccelerator<double>(acceleratorType);
}

// ================ Polyak ================ //

TEST_CASE("Test Polyak - <float>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::POLYAK;
  helper::testAccelerator<float>(acceleratorType);
}

TEST_CASE("Test Polyak - <double>") {
  const helper::AcceleratorType acceleratorType =
      helper::AcceleratorType::POLYAK;
  helper::testAccelerator<double>(acceleratorType);
}
