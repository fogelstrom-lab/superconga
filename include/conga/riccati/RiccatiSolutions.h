//===- RiccatiSolutions.h - Coherence func. solutions to the Riccati eq. -===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This contains the analytic solutions to the Riccati equation (with respect
/// to the coherence functions), namely the homogeneous (bulk) solution, and the
/// stepping solution using the mid-point method.
///
//===----------------------------------------------------------------------===//

#ifndef CONGA_RICCATI_SOLUTIONS_H
#define CONGA_RICCATI_SOLUTIONS_H

#include <cuda.h>
#include <thrust/complex.h>

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

namespace conga {
namespace riccati {
/// \brief Riccati equation homogeneous: gamma and gammaTilde.
///
/// \param inOrderParameter Order parameter.
/// \param inEnergy Energy.
/// \param outGamma The homogeneous solution for gamma.
/// \param outGammaTilde The homogeneous solution for gammaTilde.
///
/// \return Void.
template <typename T>
_HD_ void computeHomogeneous(const thrust::complex<T> &inOrderParameter,
                             const thrust::complex<T> &inEnergy,
                             thrust::complex<T> &outGamma,
                             thrust::complex<T> &outGammaTilde) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // Inverse denominator.
  const thrust::complex<T> invDenom =
      thrust::complex<T>(static_cast<T>(1), static_cast<T>(0)) /
      (inEnergy + iOmega);

  // The homogenous solutions.
  outGamma = -inOrderParameter * invDenom;
  outGammaTilde = thrust::conj(inOrderParameter) * invDenom;
}

/// \brief Mid-point stepping solution: gamma. Input the negative conjugate of
/// the order parameter in order to compute gammaTilde.
///
/// \param inOrderParameter The order parameter at the middle of the step.
/// \param inEnergy The energy at the middle of the step.
/// \param inInitial Value at the start of the step.
/// \param inStep Step size.
///
/// \return Value at the end of the step.
template <typename T>
_HD_ thrust::complex<T> computeStep(const thrust::complex<T> &inOrderParameter,
                                    const thrust::complex<T> &inEnergy,
                                    const thrust::complex<T> &inInitial,
                                    const T inStep) {
  // For convenience.
  const thrust::complex<T> step(static_cast<T>(0), inStep);

  // These coefficients come from the mid-point method:
  //
  // y_n = y_{n-1} + h f((s_n + s_{n-1}) / 2 , (y_n + y_{n-1}) / 2)
  //
  // It yields a quadratic equation to solve:
  //
  // 0 = a * y^2_n + b * y_n + c
  //
  // with the following coefficients.
  const thrust::complex<T> a =
      static_cast<T>(0.25) * step * thrust::conj(inOrderParameter);
  const thrust::complex<T> b =
      static_cast<T>(2) * a * inInitial + step * inEnergy - static_cast<T>(1);
  const thrust::complex<T> c =
      (a * inInitial + step * inEnergy + static_cast<T>(1)) * inInitial +
      inOrderParameter * step;

  // Solution.
  const thrust::complex<T> discriminant =
      thrust::sqrt(b * b - static_cast<T>(4) * a * c);
  return static_cast<T>(2) * c / (-b + discriminant);
}
} // namespace riccati
} // namespace conga

#endif // CONGA_RICCATI_SOLUTIONS_H
