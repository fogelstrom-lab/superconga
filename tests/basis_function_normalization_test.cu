#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "FermiSurface.h"
#include "FermiSurfaceCircle.h"
#include "FermiSurfaceTightBinding.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterComponent_D.h"
#include "OrderParameterComponent_S.h"
#include "cplx.h"
#include "utils.h"

#include <cmath>
#include <doctest.h>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
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
void testBasisFunctionNormaliztation(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // Set up Fermi surface angles
  const int numAngles = 100;
  const std::vector<T> angles = utils::linspace(
      static_cast<T>(0.0), static_cast<T>(2.0 * M_PI), numAngles);

  // Create order parameter and Fermi surface objects
  OrderParameterComponent_S<T> orderParameterComponentS;
  OrderParameterComponent_D<T> orderParameterComponentD;
  std::unique_ptr<FermiSurface<T>> fermiSurface;

  // Loop over Fermi surface types
  for (FermiSurfaceModels fermiSurfaceModel : allFermiSurfaceModels) {
    // Construct appropriate fermi surface
    if (fermiSurfaceModel == FermiSurfaceModels::circular) {
      fermiSurface = std::make_unique<FermiSurfaceCircle<T>>();
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs1) {
      fermiSurface = std::make_unique<FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS1), static_cast<T>(HOPPING_NNNN_FS1),
          static_cast<T>(CHEMICAL_POTENTIAL_FS1));
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs2) {
      fermiSurface = std::make_unique<FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS2), static_cast<T>(HOPPING_NNNN_FS2),
          static_cast<T>(CHEMICAL_POTENTIAL_FS2));
    } else {
      std::cout << "Error! Unknown FermiSurface!" << std::endl;
    }

    // Initialize Fermi surface
    fermiSurface->initialize(numAngles);

    // Normalize basis function
    orderParameterComponentS.initBasisFunctionNormalization(fermiSurface.get());
    orderParameterComponentD.initBasisFunctionNormalization(fermiSurface.get());
    const T basisNormalizationS =
        orderParameterComponentS.getBasisFunctionNormalization();
    const T basisNormalizationD =
        orderParameterComponentD.getBasisFunctionNormalization();

    T integrandFactorSumS = static_cast<T>(0.0);
    T integrandFactorSumD = static_cast<T>(0.0);

    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
      // =================== ACT ==================== //
      const T angle = angles[angleIdx];
      const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

      T momentumX;
      T momentumY;
      fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);
      const cplx<T> basisFunctionS =
          basisNormalizationS *
          BasisFunction_S<T>::compute(momentumX, momentumY);
      const cplx<T> basisFunctionD =
          basisNormalizationD *
          BasisFunction_D<T>::compute(momentumX, momentumY);

      // Compute the integrand of <|eta|^2 >
      integrandFactorSumS += integrandFactor * basisFunctionS.cabsSq();
      integrandFactorSumD += integrandFactor * basisFunctionD.cabsSq();
    }

    // =================== ASSERT ==================== //
    const T integrandFactorSumExpected = static_cast<T>(1.0);

    std::cout << integrandFactorSumS << std::endl;
    std::cout << integrandFactorSumD << std::endl;

    CHECK(integrandFactorSumS ==
          doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
    CHECK(integrandFactorSumD ==
          doctest::Approx(integrandFactorSumExpected).epsilon(inTolerance));
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
