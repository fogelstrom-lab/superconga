#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "FermiSurface.h"
#include "FermiSurfaceCircle.h"
#include "FermiSurfaceTightBinding.h"
#include "OrderParameterBasis.h"
#include "cplx.h"
#include "utils.h"

#include <cmath>
#include <doctest.h>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace helper {
// Enum for the the different Fermi-surfaces: fs1 and fs2 refer to the
// tight-binding models of the BdG-comparison paper, with parameters given below
enum class FermiSurfaceModels { circular, fs1, fs2 };
const int numFermiSurfaceModels = 3;
const std::array<FermiSurfaceModels, numFermiSurfaceModels>
    allFermiSurfaceModels = {FermiSurfaceModels::circular,
                             FermiSurfaceModels::fs1, FermiSurfaceModels::fs2};

// The relevant tight-binding parameters for the BdG paper.
// Almost ciruclar.
const double HOPPING_NNN_FS1 = -0.25;
const double HOPPING_NNNN_FS1 = 0.0;
const double CHEMICAL_POTENTIAL_FS1 = 0.0;
// Less circular.
const double HOPPING_NNN_FS2 = -0.495;
const double HOPPING_NNNN_FS2 = 0.156;
const double CHEMICAL_POTENTIAL_FS2 = -1.267;

// Check that the order-parameter basis function is normalized, <|eta|^2 > = 1,
// for different order parameter symmetries and different fermi surface types.
template <typename T>
void testBasisFunctionNormaliztation(const T inTolerance,
                                     const bool inNumNormalizations = 2) {
  // =================== ARRANGE ==================== //
  // Set up Fermi surface angles
  const int numAngles = 100;
  const std::vector<T> angles = scg::utils::linspace(
      static_cast<T>(0.0), static_cast<T>(2.0 * M_PI), numAngles);

  // Create Fermi surface object
  std::unique_ptr<scg::FermiSurface<T>> fermiSurface;

  // Loop over Fermi surface types
  for (FermiSurfaceModels fermiSurfaceModel : allFermiSurfaceModels) {
    // Construct appropriate fermi surface
    if (fermiSurfaceModel == FermiSurfaceModels::circular) {
      fermiSurface = std::make_unique<scg::FermiSurfaceCircle<T>>();
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs1) {
      fermiSurface = std::make_unique<scg::FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS1), static_cast<T>(HOPPING_NNNN_FS1),
          static_cast<T>(CHEMICAL_POTENTIAL_FS1));
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs2) {
      fermiSurface = std::make_unique<scg::FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS2), static_cast<T>(HOPPING_NNNN_FS2),
          static_cast<T>(CHEMICAL_POTENTIAL_FS2));
    } else {
      std::cout << "Error! Unknown FermiSurface!" << std::endl;
    }

    // Initialize Fermi surface
    fermiSurface->initialize(numAngles);

    // Basis functions.
    scg::BasisFunction<T> basisFunctionS(scg::BasisFunctionSymmetry::SWAVE);
    scg::BasisFunction<T> basisFunctionD(scg::BasisFunctionSymmetry::DWAVE_X2_Y2);

    for (int i = 0; i < inNumNormalizations; ++i) {
      // Normalize basis function. It should not matter whether we have called
      // 'normalize' before, hence the loop.
      basisFunctionS.normalize(fermiSurface.get());
      basisFunctionD.normalize(fermiSurface.get());

      T integrandFactorSumS = static_cast<T>(0.0);
      T integrandFactorSumD = static_cast<T>(0.0);

      for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
        // =================== ACT ==================== //
        const T angle = angles[angleIdx];
        const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

        T momentumX;
        T momentumY;
        fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);
        const scg::cplx<T> etaS = basisFunctionS.evaluate(momentumX, momentumY);
        const scg::cplx<T> etaD = basisFunctionD.evaluate(momentumX, momentumY);

        // Compute the integrand of <|eta|^2 >
        integrandFactorSumS += integrandFactor * etaS.cabsSq();
        integrandFactorSumD += integrandFactor * etaD.cabsSq();
      }

      // =================== ASSERT ==================== //
      const T integrandFactorSumExpected = static_cast<T>(1.0);

      CHECK(integrandFactorSumS ==
            doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
      CHECK(integrandFactorSumD ==
            doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
    }
  }
}

} // namespace helper

// =============== Test basis function normalization =============== //
TEST_CASE("Test Basis Function Normalization: s-wave, <float>") {
  // Relative tolerance.
  const float tolerance = 5.0F * std::numeric_limits<float>::epsilon();
  helper::testBasisFunctionNormaliztation(tolerance);
}

TEST_CASE("Test Basis Function Normalization: s-wave, <double>") {
  // Relative tolerance.
  const double tolerance = 5.0 * std::numeric_limits<double>::epsilon();
  helper::testBasisFunctionNormaliztation(tolerance);
}
