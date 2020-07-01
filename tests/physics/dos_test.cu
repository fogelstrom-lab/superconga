//===--------- dost_test.cu - Unit tests of the density of states  --------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
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

#include "BoundaryConditionSpecular.h"
#include "ComputeLDOS.h"
#include "Context.h"
#include "DiscGeometry.h"
#include "FermiSurfaceCircle.h"
#include "GeometryGroup.h"
#include "OrderParameter.h"
#include "OrderParameterBasis.h"
#include "OrderParameterInitial.h"
#include "Parameters.h"
#include "PolygonGeometry.h"
#include "RiccatiSolverConfined.h"
#include "cplx.h"
#include "defines.h"
#include <complex>
#include <doctest.h>
#include <iostream>
#include <stdlib.h>

//*******************************************************
// SIMULATION PARAMETERS common between test cases
//*******************************************************

namespace simParams {
// Temperature in units of T_c
const double TEMPERATURE = 0.5;
// Size of superconducting grain in coherence lengths hbar*vF/(2*pi*kB*Tc).
const double GRAIN_SIDE_LENGTH = 5.0;
// Number of lattice points per grain side length.
const int GRID_RESOLUTION = 64;
// Number of trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 32;
// Number of energies in total.
const int NUM_ENERGIES = 32;
// How many energies to compute the LDOS for in parallel.
const int NUM_ENERGIES_PER_BLOCK = NUM_ENERGIES;
// The range of the energies.
const double ENERGY_MIN = 0.0;
const double ENERGY_MAX = 10.0;
// The value of the imaginary part added to the energies.
const double ENERGY_SMEARING = 0.02;
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
  Context<T> *context = new Context<T>();

  // Storage for system simulation parameters (e.g. temperature, lattice size).
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set grain resolution in number of lattice sites.
  parameters->setGridResolution(simParams::GRID_RESOLUTION);
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc).
  parameters->setGrainWidthCoherenceLengths(
      static_cast<T>(simParams::GRAIN_SIDE_LENGTH));
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
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

  if (inUseSwave) {
    // Add a disc to the simulation geometry.
    // Note, corners introduce artifacts. For an s-wave we can have a disc
    // instead, and lower the tolerance.
    const T discDiameter = static_cast<T>(1.0);
    geometry->add(new DiscGeometry<T>(discDiameter), Type::add);
  } else {
    // Add a square to the simulation geometry.
    geometry->add(new PolygonGeometry<T>(4), Type::add);
  }

  // Choose a Riccati solver for a confined geometry, without magnetic fields
  RiccatiSolver<T> *riccatiSolver =
      new RiccatiSolverConfined<T, Gamma::General, VectorPotential::None>(
          context);
  // Add specular BC
  riccatiSolver->add(new BoundaryConditionSpecular<T>);

  // Set up order parameter
  OrderParameter<T> *orderParameter = new OrderParameter<T>(context);

  // Use s-wave or d-wave.
  const BasisFunctionSymmetry symmetry =
      inUseSwave ? BasisFunctionSymmetry::SWAVE
                 : BasisFunctionSymmetry::DWAVE_X2_Y2;

  OrderParameterComponent<T> *orderParameterComponent =
      new OrderParameterComponent<T>(symmetry, parameters);

  // Initialize order-parameter component to the bulk value.
  Initial<T>::bulk(orderParameterComponent);
  // Add the initialized order-parameter component
  orderParameter->add(orderParameterComponent);

  // Choose FermiSurface
  FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  // Add LDOS compute object.
  new ComputeLDOS<T>(context);

  // Initialize everything.
  context->initialize(simParams::NUM_ENERGIES_PER_BLOCK);

  // =================== ACT ==================== //
  // Compute the LDOS.
  for (int i = 0; i < simParams::NUM_ITERATIONS_MAX; ++i) {
    context->compute();
    const T residual = context->getResidual(0);
    const bool isConverged = context->isConverged();
    if (isConverged) {
      break;
    }
  }

  // Get results.
  typename Grid<T>::CPU *dosObject = context->getComputeLDOS()->getDOS();
  const T *const energies = dosObject->getDataPointer(0);
  const T *const dos = dosObject->getDataPointer(1);

  // =================== ASSERT ==================== //
  // In order to compute the expected DOS we need the bulk order parameter and
  // the normalization factor for the basis function.
  // TODO(niclas): The basis functions should return normalized values.
  const T orderParameterBulk = orderParameterComponent->deltaBulk();
  BasisFunction<T> basisFunction(symmetry);
  basisFunction.normalize(fermiSurface);

  // Loop over energies and check the DOS for each.
  for (int energyIdx = 0; energyIdx < simParams::NUM_ENERGIES; ++energyIdx) {
    T expectedDOS = static_cast<T>(0.0);
    for (int angleIdx = 0; angleIdx < simParams::NUM_ANGLES; ++angleIdx) {
      T fermiMomentumX;
      T fermiMomentumY;
      fermiSurface->getFermiMomentum(angleIdx, fermiMomentumX, fermiMomentumY);

      // Construct the complex valued order parameter.
      // Strictly speaking it unnecessary to have it complex valued when we only
      // consider s- and d-wave.
      const cplx<T> tmp =
          basisFunction.evaluate(fermiMomentumX, fermiMomentumY);
      const std::complex<T> eta(tmp.x, tmp.y);
      const std::complex<T> orderParameter = eta * orderParameterBulk;

      // Compute the angle resolved DOS.
      // See page 108 in P. Holmvall PhD Thesis.
      const std::complex<T> energy(energies[energyIdx],
                                   simParams::ENERGY_SMEARING);
      const std::complex<T> Omega =
          std::sqrt(std::norm(orderParameter) - energy * energy);
      const T dosAngle = (energy / Omega).imag();

      // Accumulate angle integral.
      const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);
      expectedDOS += integrandFactor * dosAngle;
    }
    const T testDOS = dos[energyIdx];
    CHECK(testDOS == doctest::Approx(expectedDOS).epsilon(inTolerance));
  }
  CHECK(context->isConverged());

  delete context;
}
} // namespace helper

// ================= Test DOS s-wave ================= //

TEST_CASE("Test DOS - s-wave (float)") {
  const float tolerance = 1e-4f;
  const float convergenceCriterion = 1e-6f;
  const bool useSwave = true;
  helper::testDensityOfStates<float>(tolerance, useSwave, convergenceCriterion);
}

TEST_CASE("Test DOS - s-wave (double)") {
  const double tolerance = 1e-8;
  const double convergenceCriterion = 0.1 * tolerance;
  const bool useSwave = true;
  helper::testDensityOfStates<double>(tolerance, useSwave,
                                      convergenceCriterion);
}

// ================= Test DOS d-wave ================= //

TEST_CASE("Test DOS - d-wave (float)") {
  // TODO(niclas): Investigate why the tolerance must be so high. Is because of
  // artifacts from corners?
  const float tolerance = 1e-2f;
  const float convergenceCriterion = 1e-6f;
  const bool useSwave = false;
  helper::testDensityOfStates<float>(tolerance, useSwave, convergenceCriterion);
}

TEST_CASE("Test DOS - d-wave (double)") {
  // TODO(niclas): Investigate why the tolerance must be so high. Is because of
  // artifacts from corners?
  const double tolerance = 1e-2;
  const double convergenceCriterion = 1e-6;
  const bool useSwave = false;
  helper::testDensityOfStates<double>(tolerance, useSwave,
                                      convergenceCriterion);
}
