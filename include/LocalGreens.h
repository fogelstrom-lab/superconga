////////////////////////////////////////////////////////
// Functions to compute local value of Greens functions
///////////////////////////////////////////////////////

#ifndef LOCAL_GREENS_H_
#define LOCAL_GREENS_H_

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

#include "cplx.h"

namespace scg {
template <typename T> struct Greens {
  _HD_ static void computeGreens_local(const cplx<T> &g, const cplx<T> &gb,
                                       cplx<T> &greens) {
    const cplx<T> g_gb = g * gb;
    greens = -cplx<T>::i() * (static_cast<T>(1.0) - g_gb) /
             (static_cast<T>(1.0) + g_gb);
  }
};

template <typename T> struct GreensAnomalous {
  _HD_ static void computeGreens_local(const cplx<T> &g, const cplx<T> &gb,
                                       cplx<T> &greens) {
    // Calc anomalous greens function (f), -2i*gamma / (1 + gamma*gamma_bar)
    greens = -static_cast<T>(2.0) * cplx<T>::i() * g /
             (static_cast<T>(1.0) + g * gb);
  }
};

template <typename T> struct GreensAnomalousTilde {
  _HD_ static void computeGreens_local(const cplx<T> &g, const cplx<T> &gb,
                                       cplx<T> &greens) {
    // Calc f_tilde, 2i*gammabar / (1 + gamma*gamma_bar)
    greens = static_cast<T>(2.0) * cplx<T>::i() * gb /
             (static_cast<T>(1.0) + g * gb);
  }
};

template <typename T> struct Greens_Delta {
  _HD_ static void computeGreens_local(const cplx<T> &g, const cplx<T> &gb,
                                       cplx<T> &greens) {
    // Calc total greens function expression for order parameter: f +
    // conj(f_tilde)
    cplx<T> mult =
        static_cast<T>(2.0) * cplx<T>::i() / (static_cast<T>(1.0) + g * gb);

    const cplx<T> f = -mult * g;
    const cplx<T> f_tilde_conj = (mult * gb).conj();

    greens = f + f_tilde_conj;
  }
};
} // namespace scg

#endif // LOCAL_GREENS_H_
