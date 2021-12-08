//===--------- dost_test.cu - Unit tests of the density of states  --------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Different tests of the density of states (DOS), e.g. checking that given an
/// order parameter with a constant bulk value yields the expected DOS.
///
//===----------------------------------------------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// TODO(niclas): These should not have to be included! They are needed because
// "Context" only has forward declarations.
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeLDOS.h"
#include "conga/defines.h"
#include "conga/fermi_surface/FermiSurfaceCircle.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorKeldysh.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_iteration.h"
#include "conga/wrappers/doctest.h"

#include <thrust/complex.h>

#include <complex>
#include <iostream>
#include <stdlib.h>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.5;
// Size of superconducting grain in coherence lengths hbar*<vF>/(2*pi*kB*Tc).
const double GRAIN_SIDE_LENGTH = 5.0;
// Number of lattice points per coherence length
const double POINTS_PER_COHERENCE_LENGTH = 16.0 / GRAIN_SIDE_LENGTH;
// Number of trajectory angles (on the Fermi-surface).
// TODO(niclas): Even numbers are better than odd. Why exactly?
const int NUM_ANGLES = 720;
// Number of energies in total.
const int NUM_ENERGIES = 32;
// How many energies to compute the LDOS for in parallel.
const int NUM_ENERGIES_PER_BLOCK = NUM_ENERGIES;
// The range of the energies.
const double ENERGY_MIN = 0.0;
const double ENERGY_MAX = 0.5;
// The value of the imaginary part added to the energies.
const double ENERGY_SMEARING = 0.008;
// Maximum number of iterations.
const int NUM_ITERATIONS_MAX = 1000;
} // namespace simParams

namespace helper {
// Test that a grain, set to constant bulk value, gives the expected density of
// states.
template <typename T>
void testDensityOfStates(const T inTolerance, const bool inUseSwave,
                         const T inConvergenceCriterion) {
  // =================== ARRANGE ==================== //
  // Create Context
  conga::Context<T> context;

  // Storage for system simulation parameters (e.g. temperature, lattice size).
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  // Set grain sidelength in coherence lengths: hbar*<vF>/(kB*Tc).
  parameters->setGrainWidth(static_cast<T>(simParams::GRAIN_SIDE_LENGTH));
  // Set the number of lattice points per coherence length.
  parameters->setPointsPerCoherenceLength(
      static_cast<T>(simParams::POINTS_PER_COHERENCE_LENGTH));
  // Number of discrete steps in momentum angle.
  parameters->setAngularResolution(simParams::NUM_ANGLES);
  // Set convergence criterion.
  parameters->setConvergenceCriterion(inConvergenceCriterion);
  // Set temperature.
  parameters->setTemperature(static_cast<T>(simParams::TEMPERATURE));
  // Set new parameters relevant for a Keldysh integration iterator.
  parameters->setEnergyRange(static_cast<T>(simParams::ENERGY_MIN),
                             static_cast<T>(simParams::ENERGY_MAX),
                             simParams::NUM_ENERGIES,
                             static_cast<T>(simParams::ENERGY_SMEARING));

  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  if (inUseSwave) {
    // Add a disc to the simulation geometry.
    // Note, corners introduce artifacts. For an s-wave we can have a disc
    // instead, and lower the tolerance.
    geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
  } else {
    // Add a square to the simulation geometry.
    const int numVertices = 4;
    geometry->add(new conga::PolygonGeometry<T>(grainFraction, numVertices),
                  conga::Type::add);
  }

  // Iterator.
  new conga::IntegrationIteratorKeldysh<T>(&context,
                                           simParams::NUM_ENERGIES_PER_BLOCK);

  // Choose a Riccati solver for a confined geometry, without magnetic fields
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::None>(
          &context);
  // Add specular BC
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Set up order parameter
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);

  // Use s-wave or d-wave.
  const conga::BasisFunctionSymmetry symmetry =
      inUseSwave ? conga::BasisFunctionSymmetry::SWAVE
                 : conga::BasisFunctionSymmetry::DWAVE_X2_Y2;

  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(symmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  conga::Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Add LDOS compute object.
  new conga::ComputeLDOS<T>(&context);

  // Initialize everything.
  context.initialize();

  // =================== ACT ==================== //
  conga::Status status = conga::Status::kContinue;
  while (status != conga::Status::kQuit) {
    status =
        conga::runIteration(&context, true, 0, simParams::NUM_ITERATIONS_MAX);
  }

  // Get results.
  conga::GridCPU<T> *dosObject = context.getComputeLDOS()->getDOS();
  const T *const energies = dosObject->getDataPointer(0);
  const T *const dos = dosObject->getDataPointer(1);

  // =================== ASSERT ==================== //
  // Make circular Fermi surface.
  conga::FermiSurfaceCircle<T> fermiSurface;
  fermiSurface.initialize(simParams::NUM_ANGLES);

  // In order to compute the expected DOS we need the bulk order parameter and
  // the normalization factor for the basis function.
  // TODO(niclas): The basis functions should return normalized values.
  const T orderParameterBulk = orderParameterComponent->deltaBulk();
  conga::BasisFunction<T> basisFunction(symmetry);
  basisFunction.normalize(&fermiSurface);

  // Loop over energies and check the DOS for each.
  for (int energyIdx = 0; energyIdx < simParams::NUM_ENERGIES; ++energyIdx) {
    T expectedDOS = static_cast<T>(0.0);
    for (int angleIdx = 0; angleIdx < simParams::NUM_ANGLES; ++angleIdx) {
      T fermiMomentumX;
      T fermiMomentumY;
      fermiSurface.getFermiMomentum(angleIdx, fermiMomentumX, fermiMomentumY);

      // Construct the complex valued order parameter.
      // Strictly speaking it unnecessary to have it complex valued when we only
      // consider s- and d-wave.
      const thrust::complex<T> tmp =
          basisFunction.evaluate(fermiMomentumX, fermiMomentumY);
      const std::complex<T> eta(tmp.real(), tmp.imag());
      const std::complex<T> orderParameter = eta * orderParameterBulk;

      // Compute the angle resolved DOS.
      // See page 108 in P. Holmvall PhD Thesis.
      const std::complex<T> energy(energies[energyIdx],
                                   simParams::ENERGY_SMEARING);
      const std::complex<T> Omega =
          std::sqrt(std::norm(orderParameter) - energy * energy);
      const T dosAngle = (energy / Omega).imag();

      // Accumulate angle integral.
      const T integrandFactor = fermiSurface.getIntegrandFactor(angleIdx);
      expectedDOS += integrandFactor * dosAngle;
    }
    const T testDOS = dos[energyIdx];
    CHECK(testDOS == doctest::Approx(expectedDOS).epsilon(inTolerance));
  }
  CHECK(context.isConverged());
}
} // namespace helper

// ================= Test DOS s-wave ================= //

TEST_CASE("Test DOS - s-wave (float)") {
  const float tolerance = 1e-5f;
  const float convergenceCriterion = 1e-6f;
  const bool useSwave = true;
  helper::testDensityOfStates<float>(tolerance, useSwave, convergenceCriterion);
}

TEST_CASE("Test DOS - s-wave (double)") {
  const double tolerance = 1e-8;
  const double convergenceCriterion = 1e-10;
  const bool useSwave = true;
  helper::testDensityOfStates<double>(tolerance, useSwave,
                                      convergenceCriterion);
}

// ================= Test DOS d-wave ================= //

TEST_CASE("Test DOS - d-wave (float)") {
  // D-wave float is the odd man out. It is the only test here that does not
  // converge more or less immediately. So the tolerance is higher because there
  // is something fishy going on.
  // TODO(niclas): Find out why!
  const float tolerance = 1e-2f;
  const float convergenceCriterion = 1e-6f;
  const bool useSwave = false;
  helper::testDensityOfStates<float>(tolerance, useSwave, convergenceCriterion);
}

TEST_CASE("Test DOS - d-wave (double)") {
  const double tolerance = 1e-8;
  const double convergenceCriterion = 1e-10;
  const bool useSwave = false;
  helper::testDensityOfStates<double>(tolerance, useSwave,
                                      convergenceCriterion);
}
