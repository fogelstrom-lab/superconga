//////////////////////////////////////////////
// Functions to compute local value of Gammas
//////////////////////////////////////////////

#ifndef _LOCALGAMMA_H
#define _LOCALGAMMA_H

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

template <typename T> struct Matsubara {
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
_HD_ void Gamma::Matsubara<T>::Gamma::computeGamma_local(const cplx<T> &D,
                                                         const cplx<T> &g0,
                                                         const cplx<T> &e,
                                                         const T h, const int y,
                                                         cplx<T> &g) {
  cplx<T> i((T)0.0, (T)1.0);

  cplx<T> g01, g1, C, Cexp, iTwoOm;

  T Omega, cExp, twoOm, tmp;

  Omega = sqrt(D.cabsSq() + e.y * e.y);

  tmp = (T)1.0 / (e.y + Omega);

  // i*Delta / (e + Omega)
  g1 = D.imult() * tmp;

  if (y > 0) {
    twoOm = (T)2.0 * Omega;
    iTwoOm = cplx<T>((T)0.0, twoOm); // 2i*omega

    cExp = exp(-twoOm * h);

    g01 = g0 - g1;

    C = g01 / (iTwoOm + D.conj() * g01);
    C *= cExp;

    g1 += (iTwoOm * C) / ((T)1.0 - D.conj() * C);
  }

  g.x = g1.x;
  g.y = g1.y;
}

template <typename T>
_HD_ void Gamma::Matsubara<T>::Gammabar::computeGamma_local(
    const cplx<T> &D, const cplx<T> &g0, const cplx<T> &e, const T h,
    const int y, cplx<T> &g) {
  const cplx<T> D2(-D.x, D.y);
  Gamma::computeGamma_local(D2, g0, e, h, y, g);
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

#endif
