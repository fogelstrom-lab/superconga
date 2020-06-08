#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "FermiSurfaceTightBinding.h"
#include "defines.h"
#include "dispersion.h"
#include "utils.h"
#include <cmath>
#include <doctest.h>
#include <functional>
#include <iostream>
#include <vector>

namespace helper {
// The relevant tight-binding parameters for the BdG paper:
// https://arxiv.org/abs/2006.00456

// Fermi surface 1: Almost ciruclar.
const double HOPPING_NNN_FS1 = -0.25;
const double HOPPING_NNNN_FS1 = 0.0;
const double CHEMICAL_POTENTIAL_FS1 = 0.0;
const double AVG_FERMI_SPEED_EXPECTED_FS1 = 2.53562285;
const double AVG_FERMI_MOMENTUM_EXPECTED_FS1 = 2.61870038;
// Fermi surface 2: Less circular.
const double HOPPING_NNN_FS2 = -0.495;
const double HOPPING_NNNN_FS2 = 0.156;
const double CHEMICAL_POTENTIAL_FS2 = -1.267;
const double AVG_FERMI_SPEED_EXPECTED_FS2 = 2.39512296;
const double AVG_FERMI_MOMENTUM_EXPECTED_FS2 = 2.08128010;

// Test that the integrand factor behaves as expected. This is done by comparing
// the analytical value with the numerical one.
template <typename T>
void testComputeIntegrandFactor(
    const T inTolerance, const int inNumAngles = 10,
    const T inPhiEpsilon = static_cast<T>(1e4) *
                           std::numeric_limits<T>::epsilon()) {
  // =================== ARRANGE ==================== //
  const std::vector<T> angles = scg::utils::linspace(
      static_cast<T>(0.0), static_cast<T>(2.0 * M_PI), inNumAngles);

  // For convenience.
  const T invDenom = static_cast<T>(1.0) / (static_cast<T>(2.0) * inPhiEpsilon);

  // Fermi surface 1 in the BDG paper.
  const scg::DispersionTightBinding<T> dispersion1(
      static_cast<T>(HOPPING_NNN_FS1), static_cast<T>(HOPPING_NNNN_FS1),
      static_cast<T>(CHEMICAL_POTENTIAL_FS1));

  // Fermi surface 2 in the BDG paper.
  const scg::DispersionTightBinding<T> dispersion2(
      static_cast<T>(HOPPING_NNN_FS2), static_cast<T>(HOPPING_NNNN_FS2),
      static_cast<T>(CHEMICAL_POTENTIAL_FS2));

  // A vector of the tight-binding dispersions.
  const std::vector<scg::DispersionTightBinding<T>> v_dispersions = {
      dispersion1, dispersion2};

  for (const auto &dispersion : v_dispersions) {
    const scg::vector2d<T> originOffset =
        scg::tightbinding_internal::computeMomentumOffset(dispersion);

    for (const T phi : angles) {
      const scg::vector2d<T> fermiMomentum =
          scg::tightbinding_internal::computeFermiMomentum(dispersion, phi,
                                                           originOffset);
      const scg::vector2d<T> fermiVelocity =
          dispersion.derivative(fermiMomentum);

      // =================== ACT ==================== //
      // Compute the integrand factor using analytical derivatives.
      const T integrandFactor =
          scg::tightbinding_internal::computeIntegrandFactor(
              fermiMomentum, fermiVelocity, originOffset);

      // =================== ASSERT ==================== //
      // Compute the integrand factor using numerical derivatives.
      const scg::vector2d<T> fermiMomentumPlus =
          scg::tightbinding_internal::computeFermiMomentum(
              dispersion, phi + inPhiEpsilon, originOffset);
      const scg::vector2d<T> fermiMomentumMinus =
          scg::tightbinding_internal::computeFermiMomentum(
              dispersion, phi - inPhiEpsilon, originOffset);

      // Unwrap the momenta and form the derivative (dkx/dphi, dky/dphi).
      scg::vector2d<T> dKdPhi;
      for (int i = 0; i < 2; ++i) {
        const T plus = scg::utils::wrapToRange(
            fermiMomentumPlus[i], originOffset[i] - static_cast<T>(M_PI),
            originOffset[i] + static_cast<T>(M_PI));
        const T minus = scg::utils::wrapToRange(
            fermiMomentumMinus[i], originOffset[i] - static_cast<T>(M_PI),
            originOffset[i] + static_cast<T>(M_PI));
        dKdPhi[i] = (plus - minus) * invDenom;
      }

      const T integrandFactorExpected =
          scg::utils::normL2(dKdPhi) / scg::utils::normL2(fermiVelocity);

      REQUIRE(integrandFactor ==
              doctest::Approx(integrandFactorExpected).epsilon(inTolerance));
    }
  }
}

// Test the number of angles. Instantiate the Fermi surface with some number of
// angles and see that it returns the same value.
template <typename T> void testGetNumAngles(const int inNumAngles = 10) {
  // =================== ARRANGE ==================== //
  scg::FermiSurfaceTightBinding<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  // =================== ACT ==================== //
  const int numAngles = fermiSurface.getNumAngles();

  // =================== ASSERT ==================== //
  CHECK(numAngles == inNumAngles);
}

// Test the momentum offset. Instantiate the Fermi surface and check that the
// momentum origin offset is either (0,0) or (pi,pi), depending on the
// tight-binding parameters.
template <typename T>
void testGetMomentumOffset(const T inTolerance, const bool inCenteredOrigin,
                           const int inNumAngles = 19) {
  // =================== ARRANGE ==================== //
  const T hoppingNNN = static_cast<T>(0.0);
  const T hoppingNNNN = static_cast<T>(0.0);
  const T chemicalPotential =
      inCenteredOrigin ? static_cast<T>(-1.0) : static_cast<T>(1.0);

  scg::FermiSurfaceTightBinding<T> fermiSurface(hoppingNNN, hoppingNNNN,
                                                chemicalPotential);
  fermiSurface.initialize(inNumAngles);

  // =================== ACT ==================== //
  const scg::vector2d<T> offset = fermiSurface.getMomentumOffset();

  // =================== ASSERT ==================== //
  const T offsetExpected =
      inCenteredOrigin ? static_cast<T>(0.0) : static_cast<T>(M_PI);

  CHECK(offset[0] == doctest::Approx(offsetExpected).epsilon(inTolerance));
  CHECK(offset[1] == doctest::Approx(offsetExpected).epsilon(inTolerance));
}

// Test the angle offset. Instantiate the Fermi surface and check that the angle
// offset is pi/2, i.e. the angle corresponding to the first angle index.
template <typename T> void testGetAngleOffset(const T inTolerance) {
  // =================== ARRANGE ==================== //
  scg::FermiSurfaceTightBinding<T> fermiSurface;

  // =================== ACT ==================== //
  const T angleOffset = fermiSurface.getAngleOffset();

  // =================== ASSERT ==================== //
  // The old code always started with pF and vF along the positive y-axis, i.e.
  // an angle offset of pi/2. Here we just test that we are doing the same. In
  // the future we would like to start from an arbitrary angle, i.e. zero.
  const T angleOffsetExpected = static_cast<T>(0.5 * M_PI);
  CHECK(angleOffset ==
        doctest::Approx(angleOffsetExpected).epsilon(inTolerance));
}

// Test the Fermi momentum. Check that for certain tight-binding parameters it
// becomes a square.
template <typename T>
void testGetFermiMomentum(const T inTolerance, const int inNumAngles = 12) {
  // =================== ARRANGE ==================== //
  // These parameters will produce a square Fermi surface given by
  // cos(kx) + cos(ky) = 0
  // with the vertices positioned on the xy-axes,
  // i.e. (\pm pi, 0) and (0, \pm pi).
  const T hoppingNNN = static_cast<T>(0.0);
  const T hoppingNNNN = static_cast<T>(0.0);
  const T chemicalPotential = static_cast<T>(0.0);

  scg::FermiSurfaceTightBinding<T> fermiSurface(hoppingNNN, hoppingNNNN,
                                                chemicalPotential);
  fermiSurface.initialize(inNumAngles);

  const T angleOffset = fermiSurface.getAngleOffset();

  const scg::vector2d<T> offset = fermiSurface.getMomentumOffset();

  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);

  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    const scg::vector2d<T> fermiMomentum =
        fermiSurface.getFermiMomentum(angleIdx);

    // =================== ASSERT ==================== //
    const T angle = angleStep * static_cast<T>(angleIdx) + angleOffset;
    // For convenience.
    const scg::vector2d<T> unitVector = {std::cos(angle), std::sin(angle)};

    // The magnitude is given by |x| + |y| = pi, i.e. a square with a diagonal
    // length of 2pi.
    const T magnitude = static_cast<T>(M_PI) / scg::utils::normL1(unitVector);
    const scg::vector2d<T> fermiMomentumExpected =
        scg::utils::multAdd(magnitude, unitVector, offset);

    CHECK(std::cos(fermiMomentum[0]) ==
          doctest::Approx(std::cos(fermiMomentumExpected[0]))
              .epsilon(inTolerance));
    CHECK(std::cos(fermiMomentum[1]) ==
          doctest::Approx(std::cos(fermiMomentumExpected[1]))
              .epsilon(inTolerance));
  }
}

// Test the integrand factor. It should sum to one.
template <typename T>
void testGetIntegrandFactor(const T inTolerance, const int inNumAngles = 9) {
  // =================== ARRANGE ==================== //
  scg::FermiSurfaceTightBinding<T> fermiSurface;
  fermiSurface.initialize(inNumAngles);

  // After looping over angles and summing this should be one.
  T integrandFactorSum = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    const T integrandFactor = fermiSurface.getIntegrandFactor(angleIdx);
    integrandFactorSum += integrandFactor;
  }

  // =================== ASSERT ==================== //
  const T integrandFactorSumExpected = static_cast<T>(1.0);
  CHECK(integrandFactorSum ==
        doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
}

// Test that the Fermi surface averages of the Fermi speed and momentum are like
// the ones computed in Python.
template <typename T>
void testCompareWithPython(const T inTolerance, const T inHoppingNNN,
                           const T inHoppingNNNN, const T inChemicalPotential,
                           const T inAvgFermiMomentumExpected,
                           const T inAvgFermiSpeedExpected,
                           const int inNumAngles = 100) {
  // =================== ARRANGE ==================== //
  scg::FermiSurfaceTightBinding<T> fermiSurface(inHoppingNNN, inHoppingNNNN,
                                                inChemicalPotential);
  fermiSurface.initialize(inNumAngles);

  const T avgFermiMomentum = fermiSurface.getAvgFermiMomentum();
  const T avgFermiSpeed = fermiSurface.getAvgFermiSpeed();

  // =================== ASSERT ==================== //
  CHECK(avgFermiSpeed ==
        doctest::Approx(inAvgFermiSpeedExpected).epsilon(inTolerance));
  CHECK(avgFermiMomentum ==
        doctest::Approx(inAvgFermiMomentumExpected).epsilon(inTolerance));
}
} // namespace helper

// ============ Test computeIntegrandFactor Numerically ============ //

TEST_CASE("Test Integrand Factor <float>") {
  // Relative tolerance.
  const float tolerance = 1e-3F;
  helper::testComputeIntegrandFactor(tolerance);
}

TEST_CASE("Test Integrand Factor <double>") {
  // Relative tolerance.
  const double tolerance = 1e-3;
  helper::testComputeIntegrandFactor(tolerance);
}

// =========== Test FermiSurfaceTightBinding::getNumAngles =========== //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getNumAngles") {
  helper::testGetNumAngles<float>();
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getNumAngles") {
  helper::testGetNumAngles<double>();
}

// ======= Test FermiSurfaceTightBinding::getMomentumOffset (0,0) ======= //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getMomentumOffset (0,0)") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  const bool centeredOrigin = true;
  helper::testGetMomentumOffset(tolerance, centeredOrigin);
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getMomentumOffset (0,0)") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  const bool centeredOrigin = true;
  helper::testGetMomentumOffset(tolerance, centeredOrigin);
}

// ======= Test FermiSurfaceTightBinding::getMomentumOffset (pi,pi) ======= //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getMomentumOffset (pi,pi)") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  const bool centeredOrigin = false;
  helper::testGetMomentumOffset(tolerance, centeredOrigin);
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getMomentumOffset (pi,pi)") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  const bool centeredOrigin = false;
  helper::testGetMomentumOffset(tolerance, centeredOrigin);
}

// ============ Test FermiSurfaceTightBinding::getAngleOffset ============ //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getAngleOffset") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetAngleOffset(tolerance);
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getAngleOffset") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetAngleOffset(tolerance);
}

// ========== Test FermiSurfaceTightBinding::getFermiMumentum ========== //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getFermiMumentum") {
  // Relative tolerance.
  const float tolerance = 100.0F * std::numeric_limits<float>::epsilon();
  helper::testGetFermiMomentum(tolerance);
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getFermiMumentum") {
  // Relative tolerance.
  const double tolerance = 100.0 * std::numeric_limits<double>::epsilon();
  helper::testGetFermiMomentum(tolerance);
}

// ========== Test FermiSurfaceTightBinding::getIntegrandFactor ========== //

TEST_CASE("Test FermiSurfaceTightBinding<float>::getIntegrandFactor") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testGetIntegrandFactor(tolerance);
}

TEST_CASE("Test FermiSurfaceTightBinding<double>::getIntegrandFactor") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testGetIntegrandFactor(tolerance);
}

// ========== Test Compare Python FS1 ========== //

TEST_CASE("Test Compare Python FS1 <float>") {
  const float tolerance = 1e-6f;
  helper::testCompareWithPython(
      tolerance, static_cast<float>(helper::HOPPING_NNN_FS1),
      static_cast<float>(helper::HOPPING_NNNN_FS1),
      static_cast<float>(helper::CHEMICAL_POTENTIAL_FS1),
      static_cast<float>(helper::AVG_FERMI_MOMENTUM_EXPECTED_FS1),
      static_cast<float>(helper::AVG_FERMI_SPEED_EXPECTED_FS1));
}

TEST_CASE("Test Compare Python FS1 <double>") {
  const double tolerance = 1e-8;
  helper::testCompareWithPython(
      tolerance, helper::HOPPING_NNN_FS1, helper::HOPPING_NNNN_FS1,
      helper::CHEMICAL_POTENTIAL_FS1, helper::AVG_FERMI_MOMENTUM_EXPECTED_FS1,
      helper::AVG_FERMI_SPEED_EXPECTED_FS1);
}

// ========== Test Compare Python FS2 ========== //

TEST_CASE("Test Compare Python FS2 <float>") {
  const float tolerance = 1e-6f;
  helper::testCompareWithPython(
      tolerance, static_cast<float>(helper::HOPPING_NNN_FS2),
      static_cast<float>(helper::HOPPING_NNNN_FS2),
      static_cast<float>(helper::CHEMICAL_POTENTIAL_FS2),
      static_cast<float>(helper::AVG_FERMI_MOMENTUM_EXPECTED_FS2),
      static_cast<float>(helper::AVG_FERMI_SPEED_EXPECTED_FS2));
}

TEST_CASE("Test Compare Python FS2 <double>") {
  const double tolerance = 1e-8;
  helper::testCompareWithPython(
      tolerance, helper::HOPPING_NNN_FS2, helper::HOPPING_NNNN_FS2,
      helper::CHEMICAL_POTENTIAL_FS2, helper::AVG_FERMI_MOMENTUM_EXPECTED_FS2,
      helper::AVG_FERMI_SPEED_EXPECTED_FS2);
}
