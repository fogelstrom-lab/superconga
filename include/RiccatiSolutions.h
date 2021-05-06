//===- RiccatiSolutions.h - Coherence func. solutions to the Riccati eq. -===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This contains the analytic solutions to the Riccati equation (with respect
/// to the coherence functions), namely the homogeneous (bulk) solution, and the
/// stepping solution when assuming a piecewise constant order parameter. See
/// e.g. Appendix B.4 on p. 90 of the PhD thesis of P. Holmvall (2019):
/// https://research.chalmers.se/en/publication/513562. Please note that the
/// implementation uses another definition of the coherence length internally,
/// namely xi_0 = hbar*v_F/k_BT_c, i.e. without the usual factor of 2*pi in the
/// denominator.
///
//===----------------------------------------------------------------------===//

#ifndef RICCATI_SOLUTIONS_H
#define RICCATI_SOLUTIONS_H

#include "cplx.h"
#include <cuda.h>
#include <iostream>

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

namespace scg {
namespace Gamma {
// Function declarations: use template kernel for both gamma and gammaTilde.
template <typename T> struct General {
  struct Gamma {
    /// \brief Riccati equation stepping solution: gamma.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    /// \param inGammaInitial Gamma at start of step.
    /// \param inStep Step size.
    ///
    /// \return complex-valued solution.
    _HD_ static cplx<T> computeStep(const cplx<T> &inOrderParameter,
                                    const cplx<T> &inEnergy,
                                    const cplx<T> &inGammaInitial,
                                    const T inStep);
    /// \brief Riccati equation homogeneous solution: gamma.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    ///
    /// \return complex-valued solution.
    _HD_ static cplx<T> computeHomogeneous(const cplx<T> &inOrderParameter,
                                           const cplx<T> &inEnergy);
  };

  struct GammaTilde {
    /// \brief Riccati equation stepping solution: gammaTilde.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    /// \param inGammaTildeInitial GammaTilde at start of step.
    /// \param inStep Step size.
    ///
    /// \return complex-valued solution.
    _HD_ static cplx<T> computeStep(const cplx<T> &inOrderParameter,
                                    const cplx<T> &inEnergy,
                                    const cplx<T> &inGammaTildeInitial,
                                    const T inStep);
    /// \brief Riccati equation stepping solution: gammaTilde.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    ///
    /// \return complex-valued solution.
    _HD_ static cplx<T> computeHomogeneous(const cplx<T> &inOrderParameter,
                                           const cplx<T> &inEnergy);
  };
};
} // namespace Gamma

template <typename T>
_HD_ cplx<T> Gamma::General<T>::Gamma::computeStep(
    const cplx<T> &inOrderParameter, const cplx<T> &inEnergy,
    const cplx<T> &inGammaInitial, const T inStep) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = Omega.imult();

  const cplx<T> gammaHomogeneous = -D / (z + iOmega);

  const cplx<T> diff = (inGammaInitial - gammaHomogeneous);

  const cplx<T> ex = cplx<T>::cexp((-static_cast<T>(2) * inStep) * Omega);

  const cplx<T> gammaStep =
      gammaHomogeneous + (static_cast<T>(2) * iOmega * diff * ex) /
                             (static_cast<T>(2) * iOmega +
                              D.conj() * diff * (static_cast<T>(1) - ex));

  return gammaStep;
}

template <typename T>
_HD_ cplx<T>
Gamma::General<T>::Gamma::computeHomogeneous(const cplx<T> &inOrderParameter,
                                             const cplx<T> &inEnergy) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = Omega.imult();

  const cplx<T> gammaHomogeneous = -D / (z + iOmega);
  return gammaHomogeneous;
}

template <typename T>
_HD_ cplx<T> Gamma::General<T>::GammaTilde::computeStep(
    const cplx<T> &inOrderParameter, const cplx<T> &inEnergy,
    const cplx<T> &inGammaTildeInitial, const T inStep) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = Omega.imult();

  const cplx<T> gammaTildeHomogeneous = D.conj() / (z + iOmega);

  const cplx<T> diff = (inGammaTildeInitial - gammaTildeHomogeneous);

  const cplx<T> ex = cplx<T>::cexp((-static_cast<T>(2) * inStep) * Omega);

  const cplx<T> gammaTildeStep =
      gammaTildeHomogeneous +
      (static_cast<T>(2) * iOmega * diff * ex) /
          (static_cast<T>(2) * iOmega - D * diff * (static_cast<T>(1) - ex));
  return gammaTildeStep;
}

template <typename T>
_HD_ cplx<T> Gamma::General<T>::GammaTilde::computeHomogeneous(
    const cplx<T> &inOrderParameter, const cplx<T> &inEnergy) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = Omega.imult();

  const cplx<T> gammaTildeHomogeneous = D.conj() / (z + iOmega);
  return gammaTildeHomogeneous;
}
} // namespace scg

#endif // RICCATI_SOLUTIONS_H
