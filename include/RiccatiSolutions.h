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

#include <cuda.h>
#include <thrust/complex.h>

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

namespace scg {
namespace riccati {
/// \brief Riccati equation homogeneous: gamma and gammaTilde.
///
/// \param inOrderParameter Order parameter.
/// \param inEnergy Complex energy (imaginary pole).
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

template <typename T> struct Gamma {
  /// \brief Riccati equation stepping solution: gamma.
  ///
  /// \param inGammaHomogeneous The homogenous solution.
  /// \param inOrderParameter Order parameter.
  /// \param inEnergy Complex energy (imaginary pole).
  /// \param inGammaInitial Gamma at start of step.
  /// \param inStep Step size.
  ///
  /// \return complex-valued solution.
  _HD_ static thrust::complex<T>
  computeStep(const thrust::complex<T> &inGammaHomogeneous,
              const thrust::complex<T> &inOrderParameter,
              const thrust::complex<T> &inEnergy,
              const thrust::complex<T> &inGammaInitial, const T inStep);

  /// \brief Riccati equation half-stepping solution: gamma.
  ///
  /// \param inGammaHomogeneous The homogenous solution.
  /// \param inOrderParameter Order parameter.
  /// \param inEnergy Complex energy (imaginary pole).
  /// \param inGammaInitial Gamma at start of step.
  /// \param inStep Step size.
  /// \param outGammaCenter The value of gamma at the center of the step.
  /// \param outGammaExit The value of gamma at the end of the step.
  _HD_ static void
  computeHalfSteps(const thrust::complex<T> &inGammaHomogeneous,
                   const thrust::complex<T> &inOrderParameter,
                   const thrust::complex<T> &inEnergy,
                   const thrust::complex<T> &inGammaInitial, const T inStep,
                   thrust::complex<T> &outGammaCenter,
                   thrust::complex<T> &outGammaExit);
};

template <typename T> struct GammaTilde {
  /// \brief Riccati equation stepping solution: gammaTilde.
  ///
  /// \param inGammaTildeHomogeneous The homogenous solution.
  /// \param inOrderParameter Order parameter.
  /// \param inEnergy Complex energy (imaginary pole).
  /// \param inGammaTildeInitial GammaTilde at start of step.
  /// \param inStep Step size.
  ///
  /// \return complex-valued solution.
  _HD_ static thrust::complex<T>
  computeStep(const thrust::complex<T> &inGammaTildeHomogeneous,
              const thrust::complex<T> &inOrderParameter,
              const thrust::complex<T> &inEnergy,
              const thrust::complex<T> &inGammaTildeInitial, const T inStep);

  /// \brief Riccati equation half-stepping solution: gammaTilde.
  ///
  /// \param inGammaTildeHomogeneous The homogenous solution.
  /// \param inOrderParameter Order parameter.
  /// \param inEnergy Complex energy (imaginary pole).
  /// \param inGammaInitial GammaTilde at start of step.
  /// \param inStep Step size.
  /// \param outGammaTildeCenter The value of gammaTilde at the center of the
  /// step.
  /// \param outGammaTildeExit The value of gammaTilde at the end of the
  /// step.
  _HD_ static void
  computeHalfSteps(const thrust::complex<T> &inGammaTildeHomogeneous,
                   const thrust::complex<T> &inOrderParameter,
                   const thrust::complex<T> &inEnergy,
                   const thrust::complex<T> &inGammaTildeInitial,
                   const T inStep, thrust::complex<T> &outGammaTildeCenter,
                   thrust::complex<T> &outGammaTildeExit);
};

template <typename T>
_HD_ thrust::complex<T>
Gamma<T>::computeStep(const thrust::complex<T> &inGammaHomogeneous,
                      const thrust::complex<T> &inOrderParameter,
                      const thrust::complex<T> &inEnergy,
                      const thrust::complex<T> &inGammaInitial,
                      const T inStep) {
  // Form Omega and i*Omega.
  const thrust::complex<T> iOmega =
      inEnergy + inGammaHomogeneous * thrust::conj(inOrderParameter);
  const thrust::complex<T> Omega(iOmega.imag(), -iOmega.real());

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaInitial - inGammaHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay =
      thrust::exp((-static_cast<T>(2) * inStep) * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega + thrust::conj(inOrderParameter) * diff;
  const thrust::complex<T> C = -thrust::conj(inOrderParameter) * diff * decay;

  // The value at the end of the step.
  return inGammaHomogeneous + A / (B + C);
}

template <typename T>
_HD_ void
Gamma<T>::computeHalfSteps(const thrust::complex<T> &inGammaHomogeneous,
                           const thrust::complex<T> &inOrderParameter,
                           const thrust::complex<T> &inEnergy,
                           const thrust::complex<T> &inGammaInitial,
                           const T inStep, thrust::complex<T> &outGammaCenter,
                           thrust::complex<T> &outGammaExit) {
  // Form Omega and i*Omega.
  const thrust::complex<T> iOmega =
      inEnergy + inGammaHomogeneous * thrust::conj(inOrderParameter);
  const thrust::complex<T> Omega(iOmega.imag(), -iOmega.real());

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaInitial - inGammaHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay = thrust::exp(-inStep * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega + thrust::conj(inOrderParameter) * diff;
  const thrust::complex<T> C = -thrust::conj(inOrderParameter) * diff * decay;

  // Take first half-step.
  outGammaCenter = inGammaHomogeneous + A / (B + C);

  // Take second half-step.
  outGammaExit = inGammaHomogeneous + A * decay / (B + C * decay);
}

template <typename T>
_HD_ thrust::complex<T>
GammaTilde<T>::computeStep(const thrust::complex<T> &inGammaTildeHomogeneous,
                           const thrust::complex<T> &inOrderParameter,
                           const thrust::complex<T> &inEnergy,
                           const thrust::complex<T> &inGammaTildeInitial,
                           const T inStep) {
  // Form Omega and i*Omega.
  const thrust::complex<T> iOmega =
      inEnergy - inGammaTildeHomogeneous * inOrderParameter;
  const thrust::complex<T> Omega(iOmega.imag(), -iOmega.real());

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaTildeInitial - inGammaTildeHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay =
      thrust::exp((-static_cast<T>(2) * inStep) * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega - inOrderParameter * diff;
  const thrust::complex<T> C = inOrderParameter * diff * decay;

  // The value at the end of the step.
  return inGammaTildeHomogeneous + A / (B + C);
}

template <typename T>
_HD_ void GammaTilde<T>::computeHalfSteps(
    const thrust::complex<T> &inGammaTildeHomogeneous,
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inGammaTildeInitial, const T inStep,
    thrust::complex<T> &outGammaTildeCenter,
    thrust::complex<T> &outGammaTildeExit) {
  // Form Omega and i*Omega.
  const thrust::complex<T> iOmega =
      inEnergy - inGammaTildeHomogeneous * inOrderParameter;
  const thrust::complex<T> Omega(iOmega.imag(), -iOmega.real());

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaTildeInitial - inGammaTildeHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay = thrust::exp(-inStep * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega - inOrderParameter * diff;
  const thrust::complex<T> C = inOrderParameter * diff * decay;

  // Take first half-step.
  outGammaTildeCenter = inGammaTildeHomogeneous + A / (B + C);

  // Take second half-step.
  outGammaTildeExit = inGammaTildeHomogeneous + A * decay / (B + C * decay);
}
} // namespace riccati
} // namespace scg

#endif // RICCATI_SOLUTIONS_H
