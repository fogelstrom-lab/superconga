#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "FermiSurfaceTightBinding.h"
#include "utils.h"
#include <cmath>
#include <doctest.h>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

namespace helper {
// The relevant tight-binding parameters for the BdG paper.
// Almost ciruclar.
const double HOPPING_NNN_FS1 = -0.25;
const double HOPPING_NNNN_FS1 = 0.0;
const double CHEMICAL_POTENTIAL_FS1 = 0.0;
const double AVG_FERMI_SPEED_EXPECTED_FS1 = 2.53562285;
const double AVG_FERMI_MOMENTUM_EXPECTED_FS1 = 2.61870038;
// Less circular.
const double HOPPING_NNN_FS2 = -0.477;
const double HOPPING_NNNN_FS2 = 0.156;
const double CHEMICAL_POTENTIAL_FS2 = -1.267;
const double AVG_FERMI_SPEED_EXPECTED_FS2 = 2.38288777;
const double AVG_FERMI_MOMENTUM_EXPECTED_FS2 = 2.09334829;

// Enum for the the different symmetries to test in testSquareSymmetry.
enum class Symmetry { mirrorX, mirrorY, rotate180, mirrorDiag };

// Apply some (square) symmetry operation to the momentum.
template <typename T>
std::pair<T, T> applySymmetry(const std::pair<T, T> &inMomentum,
                              const Symmetry inSymmetry) {
  T kX = inMomentum.first;
  T kY = inMomentum.second;
  switch (inSymmetry) {
  case Symmetry::mirrorX: {
    kY = -kY;
    break;
  }
  case Symmetry::mirrorY: {
    kX = -kX;
    break;
  }
  case Symmetry::rotate180: {
    kX = -kX;
    kY = -kY;
    break;
  }
  case Symmetry::mirrorDiag: {
    const T tmp = kX;
    kX = kY;
    kY = tmp;
    break;
  }
  default:
    std::cout << "Error! Symmetry not implemented." << std::endl;
  }
  return {kX, kY};
}

// For checking the results in testSquareSymmetry.
template <typename T>
void checkResult(const T inExpected, const T inComputed, const T inTolerance,
                 const Symmetry inSymmetry) {
  // This is a scalar so inSymmetry is unused.
  REQUIRE(inExpected == doctest::Approx(inComputed).epsilon(inTolerance));
}

// For checking the results in testSquareSymmetry.
template <typename T>
void checkResult(const std::pair<T, T> &inExpected,
                 const std::pair<T, T> &inComputed, const T inTolerance,
                 const Symmetry inSymmetry) {
  // The symmetries considered are involutory, i.e. the operation is its own
  // inverse.
  const std::pair<T, T> computed = applySymmetry(inComputed, inSymmetry);

  REQUIRE(inExpected.first ==
          doctest::Approx(computed.first).epsilon(inTolerance));
  REQUIRE(inExpected.second ==
          doctest::Approx(computed.second).epsilon(inTolerance));
}

// Test that the function inFunc has the symmetries of a square.
template <typename T, class F>
void testSquareSymmetry(const T inTolerance, const F inFunc,
                        const int inNumTestsPerParameter = 5) {
  // =================== ARRANGE ==================== //
  // Next-nearest neighbour.
  const T minHoppingNNN = static_cast<T>(-0.5);
  const T maxHoppingNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNN =
      utils::linspace(minHoppingNNN, maxHoppingNNN, inNumTestsPerParameter);

  // Next-next-earest neighbour.
  const T minHoppingNNNN = static_cast<T>(-0.5);
  const T maxHoppingNNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNNN =
      utils::linspace(minHoppingNNNN, maxHoppingNNNN, inNumTestsPerParameter);

  // Chemical potential.
  const T minChemicalPotential = static_cast<T>(-1.0);
  const T maxChemicalPotential = static_cast<T>(1.0);
  const std::vector<T> a_chemicalPotential = utils::linspace(
      minChemicalPotential, maxChemicalPotential, inNumTestsPerParameter);

  // X-component of momentum.
  const T minKX = static_cast<T>(0.0);
  const T maxKX = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kX =
      utils::linspace(minKX, maxKX, inNumTestsPerParameter);

  // Y-component of momentum.
  const T minKY = static_cast<T>(0.0);
  const T maxKY = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kY =
      utils::linspace(minKY, maxKY, inNumTestsPerParameter);

  const int numSymmetries = 4;
  const std::array<Symmetry, numSymmetries> symmetries = {
      Symmetry::mirrorX, Symmetry::mirrorY, Symmetry::rotate180,
      Symmetry::mirrorDiag};

  for (const T &hoppingNNN : a_hoppingNNN) {
    for (const T &hoppingNNNN : a_hoppingNNNN) {
      for (const T &chemicalPotential : a_chemicalPotential) {
        const TightBindingParameters<T> parameters(hoppingNNN, hoppingNNNN,
                                                   chemicalPotential);
        for (const T &kX : a_kX) {
          for (const T &kY : a_kY) {
            const auto momentum = std::make_pair(kX, kY);
            const auto funcValExpected = inFunc(parameters, momentum);
            for (const Symmetry &symmetry : symmetries) {
              const auto momentumSym = applySymmetry(momentum, symmetry);
              // =================== ACT ==================== //
              const auto funcVal = inFunc(parameters, momentumSym);

              // =================== ASSERT ==================== //
              checkResult(funcValExpected, funcVal, inTolerance, symmetry);
            }
          }
        }
      }
    }
  }
}

// Test that the analytical derivative of the dispersion is close to the
// numerical derivative.
template <typename T>
void testDispersionDerivative(
    const T inTolerance, const int inNumTestsPerParameter = 5,
    const T inKEpsilon = static_cast<T>(1e3) *
                         std::numeric_limits<T>::epsilon()) {
  // =================== ARRANGE ==================== //
  // Next-nearest neighbour.
  const T minHoppingNNN = static_cast<T>(-0.5);
  const T maxHoppingNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNN =
      utils::linspace(minHoppingNNN, maxHoppingNNN, inNumTestsPerParameter);

  // Next-next-earest neighbour.
  const T minHoppingNNNN = static_cast<T>(-0.5);
  const T maxHoppingNNNN = static_cast<T>(0.5);
  const std::vector<T> a_hoppingNNNN =
      utils::linspace(minHoppingNNNN, maxHoppingNNNN, inNumTestsPerParameter);

  // Chemical potential.
  const T minChemicalPotential = static_cast<T>(-1.0);
  const T maxChemicalPotential = static_cast<T>(1.0);
  const std::vector<T> a_chemicalPotential = utils::linspace(
      minChemicalPotential, maxChemicalPotential, inNumTestsPerParameter);

  // X-component of momentum.
  const T minKX = static_cast<T>(0.0);
  const T maxKX = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kX =
      utils::linspace(minKX, maxKX, inNumTestsPerParameter);

  // Y-component of momentum.
  const T minKY = static_cast<T>(0.0);
  const T maxKY = static_cast<T>(2.0 * M_PI);
  const std::vector<T> a_kY =
      utils::linspace(minKY, maxKY, inNumTestsPerParameter);

  // For convenience.
  const T invDenom = static_cast<T>(1.0) / (static_cast<T>(2.0) * inKEpsilon);

  for (const T &hoppingNNN : a_hoppingNNN) {
    for (const T &hoppingNNNN : a_hoppingNNNN) {
      for (const T &chemicalPotential : a_chemicalPotential) {
        const TightBindingParameters<T> parameters(hoppingNNN, hoppingNNNN,
                                                   chemicalPotential);
        for (const T &kX : a_kX) {
          for (const T &kY : a_kY) {
            // =================== ACT ==================== //
            // Compute the analytical derivatives.
            const std::pair<T, T> dE =
                tightbinding_internal::computeDispersionDerivative(parameters,
                                                                   {kX, kY});
            const T dEdX = dE.first;
            const T dEdY = dE.second;

            // =================== ASSERT ==================== //
            // Compute the numerical derivatives.
            const T EXPlus = tightbinding_internal::computeDispersion(
                parameters, {kX + inKEpsilon, kY});
            const T EXMinus = tightbinding_internal::computeDispersion(
                parameters, {kX - inKEpsilon, kY});
            const T EYPlus = tightbinding_internal::computeDispersion(
                parameters, {kX, kY + inKEpsilon});
            const T EYMinus = tightbinding_internal::computeDispersion(
                parameters, {kX, kY - inKEpsilon});

            const T dEdXApprox = (EXPlus - EXMinus) * invDenom;
            const T dEdYApprox = (EYPlus - EYMinus) * invDenom;

            REQUIRE(dEdX == doctest::Approx(dEdXApprox).epsilon(inTolerance));
            REQUIRE(dEdY == doctest::Approx(dEdYApprox).epsilon(inTolerance));
          }
        }
      }
    }
  }
}

// Test that the integrand factor behaves as expected. This is done by comparing
// the analytical value with the numerical one.
template <typename T>
void testComputeIntegrandFactor(
    const T inTolerance, const int inNumAngles = 10,
    const T inPhiEpsilon = static_cast<T>(1e3) *
                           std::numeric_limits<T>::epsilon()) {
  // =================== ARRANGE ==================== //
  const std::vector<T> angles = utils::linspace(
      static_cast<T>(0.0), static_cast<T>(2.0 * M_PI), inNumAngles);

  // For convenience.
  const T invDenom = static_cast<T>(1.0) / (static_cast<T>(2.0) * inPhiEpsilon);

  // Fermi surface 1 in the BDG paper.
  const TightBindingParameters<T> parameters1(
      static_cast<T>(HOPPING_NNN_FS1), static_cast<T>(HOPPING_NNNN_FS1),
      static_cast<T>(CHEMICAL_POTENTIAL_FS1));

  // Fermi surface 2 in the BDG paper.
  const TightBindingParameters<T> parameters2(
      static_cast<T>(HOPPING_NNN_FS2), static_cast<T>(HOPPING_NNNN_FS2),
      static_cast<T>(CHEMICAL_POTENTIAL_FS2));

  // A vector of the tight-binding parameters.
  const std::vector<TightBindingParameters<T>> v_parameters = {parameters1,
                                                               parameters2};

  for (const auto &parameters : v_parameters) {
    const std::pair<T, T> originOffset =
        tightbinding_internal::computeMomentumOffset(parameters);

    for (const T phi : angles) {
      const std::pair<T, T> fermiMomentum =
          tightbinding_internal::computeFermiMomentum(parameters, phi,
                                                      originOffset);
      const std::pair<T, T> fermiVelocity =
          tightbinding_internal::computeDispersionDerivative(parameters,
                                                             fermiMomentum);

      // =================== ACT ==================== //
      // Compute the integrand factor using analytical derivatives.
      const T integrandFactor = tightbinding_internal::computeIntegrandFactor(
          parameters, fermiMomentum, fermiVelocity, originOffset);

      // =================== ASSERT ==================== //
      // Compute the integrand factor using numerical derivatives.
      const std::pair<T, T> fermiMomentumPlus =
          tightbinding_internal::computeFermiMomentum(
              parameters, phi + inPhiEpsilon, originOffset);
      const std::pair<T, T> fermiMomentumMinus =
          tightbinding_internal::computeFermiMomentum(
              parameters, phi - inPhiEpsilon, originOffset);

      // Unwrap everything.
      // TODO(niclas): Perhaps write a function that does this for pairs!?
      const T fermiMomentumPlusX = utils::wrapToRange(
          fermiMomentumPlus.first, originOffset.first - static_cast<T>(M_PI),
          originOffset.first + static_cast<T>(M_PI));
      const T fermiMomentumPlusY = utils::wrapToRange(
          fermiMomentumPlus.second, originOffset.second - static_cast<T>(M_PI),
          originOffset.second + static_cast<T>(M_PI));
      const T fermiMomentumMinusX = utils::wrapToRange(
          fermiMomentumMinus.first, originOffset.first - static_cast<T>(M_PI),
          originOffset.first + static_cast<T>(M_PI));
      const T fermiMomentumMinusY = utils::wrapToRange(
          fermiMomentumMinus.second, originOffset.second - static_cast<T>(M_PI),
          originOffset.second + static_cast<T>(M_PI));

      const T dKXdPhi = (fermiMomentumPlusX - fermiMomentumMinusX) * invDenom;
      const T dKYdPhi = (fermiMomentumPlusY - fermiMomentumMinusY) * invDenom;

      const T integrandFactorExpected =
          std::hypot(dKXdPhi, dKYdPhi) /
          std::hypot(fermiVelocity.first, fermiVelocity.second);

      REQUIRE(integrandFactor ==
              doctest::Approx(integrandFactorExpected).epsilon(inTolerance));
    }
  }
}

// Test the number of angles. Instantiate the Fermi surface with some number of
// angles and see that it returns the same value.
template <typename T> void testGetNumAngles(const int inNumAngles = 10) {
  // =================== ARRANGE ==================== //
  FermiSurfaceTightBinding<T> fermiSurface;
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

  FermiSurfaceTightBinding<T> fermiSurface(hoppingNNN, hoppingNNNN,
                                           chemicalPotential);
  fermiSurface.initialize(inNumAngles);

  // =================== ACT ==================== //
  T offsetX;
  T offsetY;
  fermiSurface.getMomentumOffset(offsetX, offsetY);

  // =================== ASSERT ==================== //
  const T offsetXExpected =
      inCenteredOrigin ? static_cast<T>(0.0) : static_cast<T>(M_PI);
  const T offsetYExpected =
      inCenteredOrigin ? static_cast<T>(0.0) : static_cast<T>(M_PI);
  CHECK(offsetX == doctest::Approx(offsetXExpected).epsilon(inTolerance));
  CHECK(offsetY == doctest::Approx(offsetYExpected).epsilon(inTolerance));
}

// Test the angle offset. Instantiate the Fermi surface and check that the angle
// offset is pi/2, i.e. the angle corresponding to the first angle index.
template <typename T>
void testGetAngleOffset(const T inTolerance, const int inNumAngles = 1) {
  // =================== ARRANGE ==================== //
  FermiSurfaceTightBinding<T> fermiSurface;

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

  FermiSurfaceTightBinding<T> fermiSurface(hoppingNNN, hoppingNNNN,
                                           chemicalPotential);
  fermiSurface.initialize(inNumAngles);

  const T angleOffset = fermiSurface.getAngleOffset();

  T offsetX;
  T offsetY;
  fermiSurface.getMomentumOffset(offsetX, offsetY);

  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);

  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // =================== ACT ==================== //
    T fermiMomentumX;
    T fermiMomentumY;
    fermiSurface.getFermiMomentum(angleIdx, fermiMomentumX, fermiMomentumY);

    // =================== ASSERT ==================== //
    const T angle = angleStep * static_cast<T>(angleIdx) + angleOffset;
    const T cosAngle = std::cos(angle);
    const T sinAngle = std::sin(angle);
    // The magnitude is given by |x| + |y| = pi, i.e. a square with a diagonal
    // length of 2pi.
    const T magnitude =
        static_cast<T>(M_PI) / (std::abs(cosAngle) + std::abs(sinAngle));
    const T fermiMomentumXExpected = magnitude * std::cos(angle) + offsetX;
    const T fermiMomentumYExpected = magnitude * std::sin(angle) + offsetY;

    CHECK(
        std::cos(fermiMomentumX) ==
        doctest::Approx(std::cos(fermiMomentumXExpected)).epsilon(inTolerance));
    CHECK(
        std::cos(fermiMomentumY) ==
        doctest::Approx(std::cos(fermiMomentumYExpected)).epsilon(inTolerance));
  }
}

// Test the integrand factor. It should sum to one.
template <typename T>
void testGetIntegrandFactor(const T inTolerance, const int inNumAngles = 9) {
  // =================== ARRANGE ==================== //
  FermiSurfaceTightBinding<T> fermiSurface;
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
  FermiSurfaceTightBinding<T> fermiSurface(inHoppingNNN, inHoppingNNNN,
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

// =============== Test computeDispersion Symmetry =============== //

TEST_CASE("Test Dispersion Symmetry <float>") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testSquareSymmetry(tolerance,
                             tightbinding_internal::computeDispersion<float>);
}

TEST_CASE("Test Dispersion Symmetry <double>") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSquareSymmetry(tolerance,
                             tightbinding_internal::computeDispersion<double>);
}

// =============== Test computeDispersionDerivative Symmetry =============== //

TEST_CASE("Test Dispersion Derivative Symmetry <float>") {
  // Relative tolerance.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testSquareSymmetry(
      tolerance, tightbinding_internal::computeDispersionDerivative<float>);
}

TEST_CASE("Test Dispersion Derivative Symmetry <double>") {
  // Relative tolerance.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testSquareSymmetry(
      tolerance, tightbinding_internal::computeDispersionDerivative<double>);
}

// ============ Test computeDispersionDerivative Numerically ============ //

TEST_CASE("Test Dispersion Derivative <float>") {
  // Relative tolerance.
  const float tolerance = 1e-3F;
  helper::testDispersionDerivative(tolerance);
}

TEST_CASE("Test Dispersion Derivative <double>") {
  // Relative tolerance.
  const double tolerance = 1e-3;
  helper::testDispersionDerivative(tolerance);
}

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
