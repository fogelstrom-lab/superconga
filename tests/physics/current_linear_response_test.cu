#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterBulk.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/wrappers/doctest.h"

namespace helper {
//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************

// Temperature in units of T_c
const double TEMPERATURE_LOW = 0.1;
const double TEMPERATURE_HIGH = 0.9;

// Just some finite value so that the induced vector potential is used.
const double PENETRATION_DEPTH = 1.0;

// The constant values of the vector-potential components.
const double VECTOR_POTENTIAL_X = 0.01;
const double VECTOR_POTENTIAL_Y = VECTOR_POTENTIAL_X;

// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc)
const double GRAIN_SIDE_LENGTH = 20.0;

// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 1.0;

// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 32;

// The energy cutoff.
const double ENERGY_CUTOFF = 16.0;

// The tolerance and scale.
const double TOLERANCE = 0.01;
const double SCALE = 0.0;

/// \brief Compute the current density via the linear-response expression.
///
/// \param inTemperature The temperature.
/// \param inVectorPotentialX The vector-potential x-component.
/// \param inVectorPotentialY The vector-potential y-component.
///
/// \return The current density as a 2D-vector.
template <typename T>
conga::vector2d<T> computeCurrentDensity(const T inTemperature,
                                         const T inVectorPotentialX,
                                         const T inVectorPotentialY) {
  // The current is computed in linear response around a homogenous background.
  // Keeping the dimensions it is given by:
  //
  // j = 2 * e * NF * <Y_{3/2} * vF * vF^T * ps>
  // ps = hbar * nabla * chi / 2 - e A
  //
  // where e is the charge of an electron (the sign is up for debate), Y_{3/2}
  // is the Yoshida function
  //
  // Y_{3/2} = pi T sum_n |Delta|^2 / (|Delta|^2 + epsilon^2_n)^(3/2)
  //
  // and is just some number.
  //
  // We assume a constant vector potential, A, and no phase gradient, i.e.
  //
  // ps = -e * A
  //
  // Thus, the only thing in the Fermi-surface average, <...>, that depends on
  // the momentum is the Fermi velocity, vF. Fortunately, the integral is
  // trivial. Ignoring the magnitude of vF we have
  //
  // vF * vF^T = | sin^2(phi)          , sin(phi) * cos(phi) |
  //             | sin(phi) * cos(phi) , cos^2(phi)          |
  //
  // The Fermi-surface average is just |vF|^2 / 2 times the identity matrix.
  //
  // The current is therefore given by:
  //
  // j = 2 * e * NF * Y_{3/2} * (|vF|^2 / 2) * (-e * A) =
  //   = -e^2 * NF * Y_{3/2} * |vF|^2 * A
  //
  // Our internal units are:
  //
  // E_0 = 2 * pi * kB * Tc       [energy]
  // A_0 = E_0 / (|vF| * |e|)     [vector potential]
  // j_0 = E_0 * NF * |vF| * |e|  [charge-current density]
  //
  // This gives
  //
  // j / j_0 = -e^2 * NF * Y_{3/2} * |vF|^2 * A / (E_0 * NF * |vF| * |e|) =
  //         = - |e| * Y_{3/2} * |vF| * A / E_0 =
  //         = - Y_{3/2} * A / A_0

  // The bulk orderparameter at this temperature.
  const T orderParameter = conga::computeOrderParameterBulkSWave(inTemperature);

  // Yoshida function.
  const T yoshida = conga::computeYoshidaBulkSWave(
      inTemperature, orderParameter, static_cast<T>(ENERGY_CUTOFF));

  // The total factor.
  const T factor = -yoshida;

  return {inVectorPotentialX * factor, inVectorPotentialY * factor};
}

/// \brief Compare the current density with the linear-response expression, for
/// a given temperature.
///
/// \param inTemperature The temperature.
///
/// \return Void.
template <typename T> void testCurrentDensity(const T inTemperature) {
  // =================== ARRANGE ==================== //
  // Create context.
  conga::Context<T> context;

  // Set parameters.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(static_cast<T>(GRAIN_SIDE_LENGTH));
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(POINTS_PER_COHERENCE_LENGTH));
  parameters->setAngularResolution(NUM_ANGLES);
  parameters->setTemperature(inTemperature);
  parameters->setEnergyCutoff(static_cast<T>(ENERGY_CUTOFF));
  parameters->setPenetrationDepth(static_cast<T>(PENETRATION_DEPTH));

  // Set geometry.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  const int numVertices = 4;
  const T sideLength = static_cast<T>(1);
  geometry->add(
      new conga::PolygonGeometry<T>(grainFraction, numVertices, sideLength),
      conga::Type::add);

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Choose a Riccati solver for a confined geometry.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::gauge::Symmetric>(&context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set order parameter.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::bulk(orderParameterComponent);
  orderParameter->add(orderParameterComponent);

  // Set vector potential.
  const T vectorPotentialX = static_cast<T>(VECTOR_POTENTIAL_X);
  const T vectorPotentialY = static_cast<T>(VECTOR_POTENTIAL_Y);
  {
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<conga::GridGPU<T>>(
        gridResolution, gridResolution, 2, conga::Type::real);
    vectorPotential->setValue(vectorPotentialX, conga::Type::real, 0);
    vectorPotential->setValue(vectorPotentialY, conga::Type::real, 1);
    context.set(std::move(vectorPotential));
  }

  // Create compute objects.
  new conga::ComputeCurrent<T>(&context);

  // =================== ACT ==================== //
  // Initialize and run simulation.
  context.initialize();
  context.compute();

  // Fetch the grain-summed current-density magnitude
  const conga::GridCPU<T> currentDensity =
      *(context.getComputeCurrent()->getResult());

  // Fetch the current density in the middle.
  // This is done because taking the mean does not give the correct results.
  // TODO(niclas): Is this due to some boundary bug? Or is this entire test a
  // bit misguided?
  const int dimX = currentDensity.dimX();
  const int dimY = currentDensity.dimY();
  const int xMiddle = dimX / 2;
  const int yMiddle = dimY / 2;
  const int xIdx = yMiddle * dimX + xMiddle;
  const int yIdx = dimX * dimY + xIdx;
  const auto data = currentDensity.getVector();
  const conga::vector2d<T> currentDensityMiddle = {data[xIdx], data[yIdx]};

  // =================== ASSERT ==================== //
  const conga::vector2d<T> currentDensityExpected =
      computeCurrentDensity(inTemperature, vectorPotentialX, vectorPotentialY);

  // Note, the way doctest compares two floating point numbers, A and B, is the
  // following: |A - B| < epsilon * (scale + max(|A|, |B|))
  const T tolerance = static_cast<T>(TOLERANCE);
  const T scale = static_cast<T>(SCALE);
  CHECK(currentDensityMiddle[0] == doctest::Approx(currentDensityExpected[0])
                                       .epsilon(tolerance)
                                       .scale(scale));
  CHECK(currentDensityMiddle[1] == doctest::Approx(currentDensityExpected[1])
                                       .epsilon(tolerance)
                                       .scale(scale));
}
} // namespace helper

// ================= Test current density <float> ================= //

TEST_CASE("Test current density, low temp <float>") {
  helper::testCurrentDensity(static_cast<float>(helper::TEMPERATURE_LOW));
}

TEST_CASE("Test current density, high temp <float>") {
  helper::testCurrentDensity(static_cast<float>(helper::TEMPERATURE_HIGH));
}

// ================= Test current density <double> ================= //

TEST_CASE("Test current density, low temp <double>") {
  helper::testCurrentDensity(helper::TEMPERATURE_LOW);
}

TEST_CASE("Test current density, high temp <double>") {
  helper::testCurrentDensity(helper::TEMPERATURE_HIGH);
}
