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
    _HD_ static thrust::complex<T>
    computeStep(const thrust::complex<T> &inOrderParameter,
                const thrust::complex<T> &inEnergy,
                const thrust::complex<T> &inGammaInitial, const T inStep);

    /// \brief Riccati equation half-stepping solution: gamma.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    /// \param inGammaInitial Gamma at start of step.
    /// \param inStep Step size.
    /// \param outGammaCenter The value of gamma at the center of the step.
    /// \param outGammaExit The value of gamma at the end of the step.
    _HD_ static void
    computeHalfSteps(const thrust::complex<T> &inOrderParameter,
                     const thrust::complex<T> &inEnergy,
                     const thrust::complex<T> &inGammaInitial, const T inStep,
                     thrust::complex<T> &outGammaCenter,
                     thrust::complex<T> &outGammaExit);

    /// \brief Riccati equation homogeneous solution: gamma.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    ///
    /// \return complex-valued solution.
    _HD_ static thrust::complex<T>
    computeHomogeneous(const thrust::complex<T> &inOrderParameter,
                       const thrust::complex<T> &inEnergy);
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
    _HD_ static thrust::complex<T>
    computeStep(const thrust::complex<T> &inOrderParameter,
                const thrust::complex<T> &inEnergy,
                const thrust::complex<T> &inGammaTildeInitial, const T inStep);

    /// \brief Riccati equation half-stepping solution: gammaTilde.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    /// \param inGammaInitial GammaTilde at start of step.
    /// \param inStep Step size.
    /// \param outGammaTildeCenter The value of gammaTilde at the center of the
    /// step. \param outGammaTildeExit The value of gammaTilde at the end of the
    /// step.
    _HD_ static void
    computeHalfSteps(const thrust::complex<T> &inOrderParameter,
                     const thrust::complex<T> &inEnergy,
                     const thrust::complex<T> &inGammaTildeInitial,
                     const T inStep, thrust::complex<T> &outGammaTildeCenter,
                     thrust::complex<T> &outGammaTildeExit);

    /// \brief Riccati equation stepping solution: gammaTilde.
    ///
    /// \param inOrderParameter Order parameter.
    /// \param inEnergy Complex energy (imaginary pole).
    ///
    /// \return complex-valued solution.
    _HD_ static thrust::complex<T>
    computeHomogeneous(const thrust::complex<T> &inOrderParameter,
                       const thrust::complex<T> &inEnergy);
  };
};
} // namespace Gamma

template <typename T>
_HD_ thrust::complex<T> Gamma::General<T>::Gamma::computeStep(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inGammaInitial, const T inStep) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  const thrust::complex<T> gammaHomogeneous =
      -inOrderParameter / (inEnergy + iOmega);

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaInitial - gammaHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay =
      thrust::exp((-static_cast<T>(2) * inStep) * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega + thrust::conj(inOrderParameter) * diff;
  const thrust::complex<T> C = -thrust::conj(inOrderParameter) * diff * decay;

  // The value at the end of the step.
  return gammaHomogeneous + A / (B + C);
}

template <typename T>
_HD_ void Gamma::General<T>::Gamma::computeHalfSteps(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inGammaInitial, const T inStep,
    thrust::complex<T> &outGammaCenter, thrust::complex<T> &outGammaExit) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  const thrust::complex<T> gammaHomogeneous =
      -inOrderParameter / (inEnergy + iOmega);

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaInitial - gammaHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay = thrust::exp(-inStep * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega + thrust::conj(inOrderParameter) * diff;
  const thrust::complex<T> C = -thrust::conj(inOrderParameter) * diff * decay;

  // Take first half-step.
  outGammaCenter = gammaHomogeneous + A / (B + C);

  // Take second half-step.
  outGammaExit = gammaHomogeneous + A * decay / (B + C * decay);
}

template <typename T>
_HD_ thrust::complex<T> Gamma::General<T>::Gamma::computeHomogeneous(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  return -inOrderParameter / (inEnergy + iOmega);
}

template <typename T>
_HD_ thrust::complex<T> Gamma::General<T>::GammaTilde::computeStep(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inGammaTildeInitial, const T inStep) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  const thrust::complex<T> gammaTildeHomogeneous =
      thrust::conj(inOrderParameter) / (inEnergy + iOmega);

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaTildeInitial - gammaTildeHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay =
      thrust::exp((-static_cast<T>(2) * inStep) * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega - inOrderParameter * diff;
  const thrust::complex<T> C = inOrderParameter * diff * decay;

  // The value at the end of the step.
  return gammaTildeHomogeneous + A / (B + C);
}

template <typename T>
_HD_ void Gamma::General<T>::GammaTilde::computeHalfSteps(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inGammaTildeInitial, const T inStep,
    thrust::complex<T> &outGammaTildeCenter,
    thrust::complex<T> &outGammaTildeExit) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  const thrust::complex<T> gammaTildeHomogeneous =
      thrust::conj(inOrderParameter) / (inEnergy + iOmega);

  // The difference between the initial and homogenous values.
  const thrust::complex<T> diff = inGammaTildeInitial - gammaTildeHomogeneous;

  // The exponential decay factor.
  const thrust::complex<T> decay = thrust::exp(-inStep * Omega);

  // For convenience.
  const thrust::complex<T> A = static_cast<T>(2) * iOmega * diff * decay;
  const thrust::complex<T> B =
      static_cast<T>(2) * iOmega - inOrderParameter * diff;
  const thrust::complex<T> C = inOrderParameter * diff * decay;

  // Take first half-step.
  outGammaTildeCenter = gammaTildeHomogeneous + A / (B + C);

  // Take second half-step.
  outGammaTildeExit = gammaTildeHomogeneous + A * decay / (B + C * decay);
}

template <typename T>
_HD_ thrust::complex<T> Gamma::General<T>::GammaTilde::computeHomogeneous(
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inEnergy) {
  // Form Omega and i*Omega.
  const thrust::complex<T> Omega =
      thrust::sqrt(thrust::norm(inOrderParameter) - inEnergy * inEnergy);
  const thrust::complex<T> iOmega(-Omega.imag(), Omega.real());

  // The homogenous solution.
  return thrust::conj(inOrderParameter) / (inEnergy + iOmega);
}
} // namespace scg

#endif // RICCATI_SOLUTIONS_H
