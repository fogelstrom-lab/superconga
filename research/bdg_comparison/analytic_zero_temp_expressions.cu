//=== analytic_zero_temp_expressions.cu - Evaluate zero temp. expressions. ===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Evaluate some analytically derived bulk expressions at zero temperature for
/// the order parameter and free energies. In particular, the BCS free energy,
/// the cost of midgap states and the gain of doppler shifts. These expressions
/// are taken from P. Holmvall, PhD Thesis (2019):
/// Order parameter: Eq. (D.11) on p. 104
/// BCS free energy: Eq. (D.26) on p. 106
/// MGS free energy: Eq. (E.104) on p. 131
/// Doppler free energy: second term in Eq. (E.117) on p. 132
///
//===----------------------------------------------------------------------===//

#include "FermiSurface.h"
#include "FermiSurfaceCircle.h"
#include "FermiSurfaceTightBinding.h"
#include "OrderParameterBasis.h"
#include "cplx.h"
#include "defines.h"
#include "utils.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

//*******************************************************
// TYPEDEFS
//*******************************************************
// Run with double or single floating point precision, double is highly
// recommended
typedef double T;

//*******************************************************
// MAIN
//*******************************************************
int main() {
  // Enum for the the different Fermi-surfaces: fs1 and fs2 refer to the
  // tight-binding models of the BdG-comparison paper, with parameters given
  // below
  enum class FermiSurfaceModels { circular, fs1, fs2 };
  const int numFermiSurfaceModels = 3;
  const std::array<FermiSurfaceModels, numFermiSurfaceModels>
      allFermiSurfaceModels = {FermiSurfaceModels::circular,
                               FermiSurfaceModels::fs1,
                               FermiSurfaceModels::fs2};

  // The relevant tight-binding parameters for the BdG paper.
  // Almost ciruclar.
  const double HOPPING_NNN_FS1 = -0.25;
  const double HOPPING_NNNN_FS1 = 0.0;
  const double CHEMICAL_POTENTIAL_FS1 = 0.0;
  const double TSTAR_FS1 = 0.15;
  // Less circular.
  const double HOPPING_NNN_FS2 = -0.495;
  const double HOPPING_NNNN_FS2 = 0.156;
  const double CHEMICAL_POTENTIAL_FS2 = -1.267;
  const double TSTAR_FS2 = 0.25;

  const double TSTAR_CIRC = 0.18;

  // Set up Fermi surface angles
  const int numAngles = 100;

  // Epsilon
  const T epsilon =
      static_cast<T>(1e2) * std::numeric_limits<double>::epsilon();

  // Rotation of ab-axes wrt pairbreaking interface (for MGS and Doppler
  // energies): perfect pairbreaking for pi/4
  const T latticeRotation = static_cast<T>(M_PI * 0.25);
  // Angle between surface normal and a-axis
  const T surfNormalAngle = static_cast<T>(M_PI * 0.5) - latticeRotation;
  // Angle between surface parallel and a-axis
  const T surfParallelAngle = -latticeRotation;

  // Set up the surface normal and parallel vectors. The components vFy (vFx) in
  // the equations will be obtained by taking the dot product between the
  // surface normal (parallel) and the Fermi velocity.
  const T cosNormal = std::cos(surfNormalAngle);
  const T sinNormal = std::sin(surfNormalAngle);
  const T cosParallel = std::cos(surfParallelAngle);
  const T sinParallel = std::sin(surfParallelAngle);
  const std::vector<T> surfaceNormal = {cosNormal, sinNormal};
  const std::vector<T> surfaceParallel = {cosParallel, sinParallel};

  // Set up Fermi surfaces
  std::unique_ptr<FermiSurface<T>> fermiSurface;

  T TStar;

  // Loop over different Fermi surfaces
  for (FermiSurfaceModels fermiSurfaceModel : allFermiSurfaceModels) {
    // Construct appropriate fermi surface
    if (fermiSurfaceModel == FermiSurfaceModels::circular) {
      fermiSurface = std::make_unique<FermiSurfaceCircle<T>>();
      TStar = TSTAR_CIRC;
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs1) {
      fermiSurface = std::make_unique<FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS1), static_cast<T>(HOPPING_NNNN_FS1),
          static_cast<T>(CHEMICAL_POTENTIAL_FS1));
      TStar = TSTAR_FS1;
    } else if (fermiSurfaceModel == FermiSurfaceModels::fs2) {
      fermiSurface = std::make_unique<FermiSurfaceTightBinding<T>>(
          static_cast<T>(HOPPING_NNN_FS2), static_cast<T>(HOPPING_NNNN_FS2),
          static_cast<T>(CHEMICAL_POTENTIAL_FS2));
      TStar = TSTAR_FS2;
    } else {
      std::cout << "Error! Unknown FermiSurface!" << std::endl;
    }

    // Initialize Fermi surface
    fermiSurface->initialize(numAngles);

    BasisFunction<T> basisFunctionD(BasisFunctionSymmetry::DWAVE);
    basisFunctionD.normalize(fermiSurface.get());

    // Storage for integrand sums
    T integrandBasisFunctionSquared = static_cast<T>(0.0);
    T integrandOrderParameter = static_cast<T>(0.0);
    T integrandFreeEnergyMGS = static_cast<T>(0.0);
    T integrandFreeEnergyDoppler = static_cast<T>(0.0);

    // Loop over angles
    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
      // Fetch the integrand factor of the Fermi integral at current angle
      const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

      // Fermi momentum in rotated coordinate system (angleIdx=0 => theta=pi/2)
      T momentumX;
      T momentumY;
      fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

      // Basis function: eta
      const cplx<T> basisFunction =
          basisFunctionD.evaluate(momentumX, momentumY);
      // |eta|
      const T basisFunctionAbs = basisFunction.cabs();
      // |eta|^2
      const T basisFunctionAbs2 = basisFunctionAbs * basisFunctionAbs;

      // Compute the integrand of <|eta|^2 >
      integrandBasisFunctionSquared += integrandFactor * basisFunctionAbs2;

      // Avoid potential issues with 0*log(0)
      if (basisFunctionAbs > epsilon) {
        // OP: compute the integrand of <|eta|^2 * ln(|eta|)>
        integrandOrderParameter +=
            integrandFactor * basisFunctionAbs2 * std::log(basisFunctionAbs);
      }

      // Fermi momentum in rotated coordinate system (angleIdx=0 => theta=pi/2)
      T velocityX;
      T velocityY;
      fermiSurface->getFermiVelocity(angleIdx, velocityX, velocityY);

      // Calculate the projection of vF on the surface normal and parallel
      const T velocityFactorY =
          std::abs(velocityX * surfaceNormal[0] + velocityY * surfaceNormal[1]);
      const T velocityFactorX = std::abs(velocityX * surfaceParallel[0] +
                                         velocityY * surfaceParallel[1]);

      // MGS energy: compute the integrand of <|dot(vf, n)|*|eta|> =
      // <|vFy|*|eta|> where n=(0,1) is the surface normal.
      integrandFreeEnergyMGS +=
          integrandFactor * velocityFactorY * basisFunctionAbs;

      // Doppler energy: integrand of <|dot(vf,n)|*|dot(vF,ps)|>
      integrandFreeEnergyDoppler +=
          integrandFactor * velocityFactorY * velocityFactorX;
    }

    // Bulk OP: Delta = pi*exp(-EulerGamma)*exp(-<|eta|^2 * ln(|eta|)>),
    // see Eq. (D11) on p. 104 in P. Holmvall PhD Thesis (2019)
    const T bulkOrderParameter =
        static_cast<T>(M_PI) *
        std::exp(-static_cast<T>(qcs::EULER_MASCHERONI)) *
        std::exp(-integrandOrderParameter);
    // BCS energy: Omega = -<|Delta|^2>/2 = -|Delta_0|^2 * <|eta|^2>,
    // see Eq. (D.26) on p. 106 in P. Holmvall PhD Thesis (2019)
    const T freeEnergyBCS = -static_cast<T>(0.5) *
                            (bulkOrderParameter * bulkOrderParameter) *
                            integrandBasisFunctionSquared;
    // MGS energy: Omega = (pi^2/2)*<|Delta|*|dot(vf, n)|> with surface normal
    // n=(0,1), see Eq. (E.104) on p. 131 in P. Holmvall PhD Thesis (2019)
    const T freeEnergyMGS = static_cast<T>(M_PI * M_PI * 0.5) *
                            bulkOrderParameter * integrandFreeEnergyMGS;
    // Doppler energy: Omega = -(pi^2)<|dot(vf,n)|*dot(vF,ps)>,
    // with surface normal n=(0,1), see second term in Eq. (E.117) on p. 132 in
    // P. Holmvall PhD Thesis (2019)
    // NOTE !!! In the thesis, the integral is over half the FS: dot(vF,ps)>0
    const T freeEnergyDoppler =
        -static_cast<T>(M_PI * M_PI) * integrandFreeEnergyDoppler;

    // Expected bulk OP at T=0, circular FS
    const T expecteOrderParameter = qcs::D_WAVE_BULK;
    // Expected BCS energy at T=0, circular FS
    const T expecteFreeEnergyBCS = -0.5 * qcs::D_WAVE_BULK * qcs::D_WAVE_BULK;
    // Expected values for circular FS:
    // MGS cost: Eq. (E.106) on p. 131 in P. Holmvall PhD Thesis (2019) - note
    // that there is a typo in the thesis, since it should be sin(2*theta) in
    // instead of cos(2*theta) in Eq. (E.105)
    const T expectedFreeEnergyMGS =
        static_cast<T>(std::sqrt(8.0) * M_PI / 3.0) * qcs::D_WAVE_BULK;
    // Doppler gain: slightly rewritten Eq. (E.121) on p. 132 in P. Holmvall PhD
    // Thesis (2019)
    const T expectedFreeEnergyDoppler = -static_cast<T>(M_PI);

    std::cout << "====================== RESULTS =============================";
    std::cout << std::endl;
    std::cout << "Norm: " << integrandBasisFunctionSquared << " (should be 1)."
              << std::endl;
    std::cout << "Delta/(kB*Tc) = " << bulkOrderParameter
              << " \t(circular should be " << expecteOrderParameter << ")."
              << std::endl;
    std::cout << "OmegaBCS/(V*NF*(kB*Tc)^2) = " << freeEnergyBCS
              << " \t(circular should be " << expecteFreeEnergyBCS << ")."
              << std::endl;
    std::cout << "OmegaMGS/(Lx*xi0*NF*(kB*Tc)^2) = " << freeEnergyMGS
              << " \t(circular should be " << expectedFreeEnergyMGS << ")."
              << std::endl;
    std::cout << "OmegaDoppler/(Lx*xi0*NF*(kB*Tc)^2) = " << freeEnergyDoppler
              << " \t(circular should be " << expectedFreeEnergyDoppler << ")."
              << std::endl;
    std::cout << "____________________________________________________________";
    std::cout << std::endl;
    std::cout << "T^*/T^*_circ = " << TStar / TSTAR_CIRC << std::endl;
    std::cout << "Delta/Delta_circ = "
              << bulkOrderParameter / expecteOrderParameter << std::endl;
    std::cout << "OmegaBCS/OmegaBCS_circ = "
              << freeEnergyBCS / expecteFreeEnergyBCS << std::endl;
    std::cout << "OmegaMGS/OmegaMGS_circ = "
              << freeEnergyMGS / expectedFreeEnergyMGS << std::endl;
    std::cout << "OmegaDoppler/OmegaDoppler_circ = "
              << freeEnergyDoppler / expectedFreeEnergyDoppler << std::endl;
    std::cout << "============================================================";
    std::cout << "\n" << std::endl;
  }

  return 0;
}
