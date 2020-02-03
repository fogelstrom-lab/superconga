//////////////////////////////////////////////
// Functions to compute local value of Gammas
//////////////////////////////////////////////

#ifndef LOCAL_GAMMA_H_
#define LOCAL_GAMMA_H_

#include <cplx.h>
#include <cuda.h>

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

namespace Gamma {
// New method with unified template kernel for both gamma and gammabar
template <typename T> struct General {
  struct Gamma {
    _HD_ static void computeGamma_local(const cplx<T> &D, const cplx<T> &g0,
                                        const cplx<T> &e, const T h,
                                        const int y, cplx<T> &g);
  };

  struct Gammabar {
    _HD_ static void computeGamma_local(const cplx<T> &D, const cplx<T> &g0,
                                        const cplx<T> &e, const T h,
                                        const int y, cplx<T> &g);
  };
};

template <typename T> struct Debug {
  struct Gamma {
    _HD_ static void computeGamma_local(const cplx<T> &D, const cplx<T> &g0,
                                        const cplx<T> &e, const T h,
                                        const int y, cplx<T> &g);
  };

  struct Gammabar {
    _HD_ static void computeGamma_local(const cplx<T> &D, const cplx<T> &g0,
                                        const cplx<T> &e, const T h,
                                        const int y, cplx<T> &g);
  };
};

} // namespace Gamma

template <typename T>
_HD_ void Gamma::General<T>::Gamma::computeGamma_local(const cplx<T> &D,
                                                       const cplx<T> &g0,
                                                       const cplx<T> &e,
                                                       const T h, const int y,
                                                       cplx<T> &g) {
  // Create complex variables
  cplx<T> Omega, iOmega, C, Cexp;

  // Compute gamma
  Omega = cplx<T>::csqrt(D.cabsSq() - e * e);
  iOmega = cplx<T>::i() * Omega;

  g = -D / (e + iOmega);

  if (y > 0) {
    C = (g0 - g) / ((T)2.0 * iOmega + D.conj() * (g0 - g));
    Cexp = C * cplx<T>::cexp((-(T)2.0 * h) * Omega);

    g += ((T)2.0 * iOmega * Cexp) / ((T)1.0 - D.conj() * Cexp);
  }
}

template <typename T>
_HD_ void Gamma::General<T>::Gammabar::computeGamma_local(
    const cplx<T> &D, const cplx<T> &g0, const cplx<T> &e, const T h,
    const int y, cplx<T> &g) {
  // Create complex variables
  cplx<T> Omega, iOmega, C, Cexp;

  // Compute gammabar
  Omega = cplx<T>::csqrt(D.cabsSq() - e * e);
  iOmega = cplx<T>::i() * Omega;

  g = D.conj() / (e + iOmega);

  if (y > 0) {
    C = (g0 - g) / ((T)2.0 * iOmega - D * (g0 - g));
    Cexp = C * cplx<T>::cexp((-(T)2.0 * h) * Omega);

    g += ((T)2.0 * iOmega * Cexp) / ((T)1.0 + D * Cexp);
  }
}

template <typename T>
_HD_ void
Gamma::Debug<T>::Gamma::computeGamma_local(const cplx<T> &D, const cplx<T> &g0,
                                           const cplx<T> &e, const T h,
                                           const int y, cplx<T> &g) {
  g = g0 * 0.999;
}

template <typename T>
_HD_ void Gamma::Debug<T>::Gammabar::computeGamma_local(const cplx<T> &D,
                                                        const cplx<T> &g0,
                                                        const cplx<T> &e,
                                                        const T h, const int y,
                                                        cplx<T> &g) {
  Gamma::computeGamma_local(D, g0, e, h, y, g);
}

#endif // LOCAL_GAMMA_H_
