//===-- LocalGreens.h - Compute Green functions. --------------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute Green functions (quasiparticle propagator and anomalous pair
/// propagator) from coherence functions.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_LOCAL_GREENS_H_
#define CONGA_LOCAL_GREENS_H_

#include "conga/backend.h"
#ifndef _HD_
#define _HD_ __host__ __device__
#endif

#include <thrust/complex.h>

namespace conga {
/// \brief Compute the Green function "g".
///
/// g = -i (1 - \gamma \tilde{gamma}) / (1 + \gamma \tilde{gamma})
///
/// \param inGamma The coherence function "gamma".
/// \param inGammaTilde The coherence function "gammaTilde".
///
/// \return The Green function "g".
template <typename T> struct Greens {
  _HD_ static thrust::complex<T>
  compute(const thrust::complex<T> &inGamma,
          const thrust::complex<T> &inGammaTilde) {
    const thrust::complex<T> prod = inGamma * inGammaTilde;
    const thrust::complex<T> tmp =
        -((static_cast<T>(1.0) - prod) / (static_cast<T>(1.0) + prod));

    // Multiply by I.
    return thrust::complex<T>(-tmp.imag(), tmp.real());
  }
};

/// \brief Compute sum of the the Green functions "f + \tilde{f}^*".
///
/// f = -2i \gamma / (1 + \gamma \tilde{gamma})
/// \tilde{f} = 2i \tilde{gamma} / (1 + \gamma \tilde{gamma})
///
/// \param inGamma The coherence function "gamma".
/// \param inGammaTilde The coherence function "gammaTilde".
///
/// \return The sum "f + \tilde{f}^*".
template <typename T> struct Greens_Delta {
  _HD_ static thrust::complex<T>
  compute(const thrust::complex<T> &inGamma,
          const thrust::complex<T> &inGammaTilde) {
    // Common factor of f and \tilde{f} (except I).
    const thrust::complex<T> tmp =
        static_cast<T>(2.0) / (static_cast<T>(1.0) + inGamma * inGammaTilde);

    // Multiply by I.
    const thrust::complex<T> factor(-tmp.imag(), tmp.real());

    const thrust::complex<T> f = -factor * inGamma;
    const thrust::complex<T> f_tilde_conj = thrust::conj(factor * inGammaTilde);

    return f + f_tilde_conj;
  }
};
} // namespace conga

#endif // CONGA_LOCAL_GREENS_H_
