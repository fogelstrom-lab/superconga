//===-- LocalGreens.h - Compute Green functions. --------------------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute Green functions (quasiparticle propagator and anomalous pair
/// propagator) from coherence functions.
///
//===----------------------------------------------------------------------===//
#ifndef LOCAL_GREENS_H_
#define LOCAL_GREENS_H_

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

#include <thrust/complex.h>

namespace scg {
template <typename T> struct Greens {
  _HD_ static void computeGreens_local(const thrust::complex<T> &g,
                                       const thrust::complex<T> &gb,
                                       thrust::complex<T> &greens) {
    const thrust::complex<T> g_gb = g * gb;
    const thrust::complex<T> tmp =
        -((static_cast<T>(1.0) - g_gb) / (static_cast<T>(1.0) + g_gb));
    greens = thrust::complex<T>(-tmp.imag(), tmp.real());
  }
};

template <typename T> struct GreensAnomalous {
  _HD_ static void computeGreens_local(const thrust::complex<T> &g,
                                       const thrust::complex<T> &gb,
                                       thrust::complex<T> &greens) {
    // Calc anomalous greens function (f), -2i*gamma / (1 + gamma*gamma_bar)
    const thrust::complex<T> tmp =
        -static_cast<T>(2.0) * g / (static_cast<T>(1.0) + g * gb);
    greens = thrust::complex<T>(-tmp.imag(), tmp.real());
  }
};

template <typename T> struct GreensAnomalousTilde {
  _HD_ static void computeGreens_local(const thrust::complex<T> &g,
                                       const thrust::complex<T> &gb,
                                       thrust::complex<T> &greens) {
    // Calc f_tilde, 2i*gammabar / (1 + gamma*gamma_bar)
    const thrust::complex<T> tmp =
        static_cast<T>(2.0) * gb / (static_cast<T>(1.0) + g * gb);
    greens = thrust::complex<T>(-tmp.imag(), tmp.real());
  }
};

template <typename T> struct Greens_Delta {
  _HD_ static void computeGreens_local(const thrust::complex<T> &g,
                                       const thrust::complex<T> &gb,
                                       thrust::complex<T> &greens) {
    // Calc total greens function expression for order parameter: f +
    // conj(f_tilde)
    const thrust::complex<T> tmp =
        static_cast<T>(2.0) / (static_cast<T>(1.0) + g * gb);
    const thrust::complex<T> factor(-tmp.imag(), tmp.real());

    const thrust::complex<T> f = -factor * g;
    const thrust::complex<T> f_tilde_conj = thrust::conj(factor * gb);

    greens = f + f_tilde_conj;
  }
};
} // namespace scg

#endif // LOCAL_GREENS_H_
