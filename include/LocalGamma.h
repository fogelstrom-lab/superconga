//////////////////////////////////////////////
// Functions to compute local value of Gammas
//////////////////////////////////////////////

#ifndef LOCAL_GAMMA_H_
#define LOCAL_GAMMA_H_

#include "cplx.h"
#include <cuda.h>

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

namespace scg {
namespace Gamma {
// New method with unified template kernel for both gamma and gammabar
template <typename T> struct General {
  struct Gamma {
    _HD_ static cplx<T> computeStep(const cplx<T> &inOrderParameter,
                                    const cplx<T> &inEnergy,
                                    const cplx<T> &inGammaInitial,
                                    const T inStep);
    _HD_ static cplx<T> computeHomogeneous(const cplx<T> &inOrderParameter,
                                           const cplx<T> &inEnergy);
  };

  struct Gammabar {
    _HD_ static cplx<T> computeStep(const cplx<T> &inOrderParameter,
                                    const cplx<T> &inEnergy,
                                    const cplx<T> &inGammaTildeInitial,
                                    const T inStep);
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
  const cplx<T> iOmega = cplx<T>::i() * Omega;

  const cplx<T> gammaHomogeneous = -D / (z + iOmega);

  const cplx<T> diff = (inGammaInitial - gammaHomogeneous);

  const cplx<T> C = diff / (static_cast<T>(2) * iOmega + D.conj() * diff);
  const cplx<T> Cexp = C * cplx<T>::cexp((-static_cast<T>(2) * inStep) * Omega);

  const cplx<T> gammaStep =
      gammaHomogeneous + (static_cast<T>(2) * iOmega * Cexp) /
                             (static_cast<T>(1) - D.conj() * Cexp);
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
  const cplx<T> iOmega = cplx<T>::i() * Omega;

  const cplx<T> gammaHomogeneous = -D / (z + iOmega);
  return gammaHomogeneous;
}

template <typename T>
_HD_ cplx<T> Gamma::General<T>::Gammabar::computeStep(
    const cplx<T> &inOrderParameter, const cplx<T> &inEnergy,
    const cplx<T> &inGammaTildeInitial, const T inStep) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = cplx<T>::i() * Omega;

  const cplx<T> gammaTildeHomogeneous = D.conj() / (z + iOmega);

  const cplx<T> diff = (inGammaTildeInitial - gammaTildeHomogeneous);

  const cplx<T> C = diff / (static_cast<T>(2) * iOmega - D * diff);
  const cplx<T> Cexp = C * cplx<T>::cexp((-static_cast<T>(2) * inStep) * Omega);

  const cplx<T> gammaTildeStep =
      gammaTildeHomogeneous +
      (static_cast<T>(2) * iOmega * Cexp) / (static_cast<T>(1) + D * Cexp);
  return gammaTildeStep;
}

template <typename T>
_HD_ cplx<T>
Gamma::General<T>::Gammabar::computeHomogeneous(const cplx<T> &inOrderParameter,
                                                const cplx<T> &inEnergy) {
  // References to make the math more readable.
  const cplx<T> &D = inOrderParameter;
  const cplx<T> &z = inEnergy;

  const cplx<T> Omega = cplx<T>::csqrt(D.cabsSq() - z * z);
  const cplx<T> iOmega = cplx<T>::i() * Omega;

  const cplx<T> gammaTildeHomogeneous = D.conj() / (z + iOmega);
  return gammaTildeHomogeneous;
}
} // namespace scg

#endif // LOCAL_GAMMA_H_
