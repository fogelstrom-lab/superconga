//===-- GreensFunctionStatic.h - Green function computation wrapper. ------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernels and wrappers to compute Green functions (quasiparticle propagator
/// and anomalous pair propagator).
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GREENS_FUNCTION_STATIC_H_
#define CONGA_GREENS_FUNCTION_STATIC_H_

#include "../ParameterIterator.h"
#include "../Parameters.h"
#include "../grid.h"
#include "LocalGreens.h"

#include <thrust/complex.h>

#include <cassert>

namespace conga {
template <typename T, template <typename> class GREENS>
struct GreensFunctionStatic {
  /// \brief Compute Green function for all energies.
  ///
  /// \param inGamma The coherence function "gamma".
  /// \param inGammaTilde The coherence function "gammaTilde".
  /// \param outResult The result.
  ///
  /// \return Void.
  static void computeGreens(const GridGPU<T> *const inGamma,
                            const GridGPU<T> *const inGammaTilde,
                            GridGPU<T> *outResult);

  /// \brief Compute Green function for all energies and sum over energies.
  ///
  /// \param inGamma The coherence function "gamma".
  /// \param inGammaTilde The coherence function "gammaTilde".
  /// \param outResult The result.
  ///
  /// \return Void.
  static void computeGreensSumEnergy(const GridGPU<T> *const inGamma,
                                     const GridGPU<T> *const inGammaTilde,
                                     GridGPU<T> *outResult);

  /// \brief Compute Green function for all energies and sum over energies.
  ///
  /// \param inGamma The coherence function "gamma".
  /// \param inGammaTilde The coherence function "gammaTilde".
  /// \param inResidues Residues to multiply the Green function (for each
  /// energy) before summing.
  /// \param outResult The result.
  ///
  /// \return Void.
  static void computeGreensSumEnergy(const GridGPU<T> *const inGamma,
                                     const GridGPU<T> *const inGammaTilde,
                                     const GridGPU<T> *const inResidues,
                                     GridGPU<T> *outResult);
};

namespace internal {
template <typename T, template <typename> class GREENS>
__global__ void
computeGreens_kernel(const int inDimX, const int inDimY, const int inDimZ,
                     const T *const inGammaReal, const T *const inGammaImag,
                     const T *const inGammaTildeReal,
                     const T *const inGammaTildeImag, T *outResultReal,
                     T *outResultImag) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x < inDimX) && (y < inDimY) && (z < inDimZ)) {
    // TODO(niclas): Make this a macro.
    const int idx = z * 2 * inDimX * inDimY + y * inDimX + x;

    const thrust::complex<T> gamma(inGammaReal[idx], inGammaImag[idx]);
    const thrust::complex<T> gammaTilde(inGammaTildeReal[idx],
                                        inGammaTildeImag[idx]);

    const thrust::complex<T> val = GREENS<T>::compute(gamma, gammaTilde);

    outResultReal[idx] = val.real();
    outResultImag[idx] = val.imag();
  }
}

template <typename T, template <typename> class GREENS>
__global__ void computeGreensSumEnergy_kernel(
    const int inDimX, const int inDimY, const int inDimZ,
    const T *const inGammaReal, const T *const inGammaImag,
    const T *const inGammaTildeReal, const T *const inGammaTildeImag,
    T *const outResultReal, T *const outResultImag) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if ((x < inDimX) && (y < inDimY)) {
    // TODO(niclas): Make this a macro.
    const int greens_idx = y * inDimX + x;

    thrust::complex<T> sum(static_cast<T>(0), static_cast<T>(0));

    for (int z = 0; z < inDimZ; ++z) {
      // TODO(niclas): Make this a macro.
      const int gamma_idx = z * 2 * inDimX * inDimY + greens_idx;

      const thrust::complex<T> gamma(inGammaReal[gamma_idx],
                                     inGammaImag[gamma_idx]);
      const thrust::complex<T> gammaTilde(inGammaTildeReal[gamma_idx],
                                          inGammaTildeImag[gamma_idx]);

      const thrust::complex<T> val = GREENS<T>::compute(gamma, gammaTilde);
      sum += val;
    }

    outResultReal[greens_idx] += sum.real();
    outResultImag[greens_idx] += sum.imag();
  }
}

template <typename T, template <typename> class GREENS>
__global__ void computeGreensSumEnergy_kernel(
    const int inDimX, const int inDimY, const int inDimZ,
    const T *const inGammaReal, const T *const inGammaImag,
    const T *const inGammaTildeReal, const T *const inGammaTildeImag,
    const T *const inSummandFactor, T *const outResultReal,
    T *const outResultImag) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if ((x < inDimX) && (y < inDimY)) {
    // TODO(niclas): Make this a macro.
    const int greens_idx = y * inDimX + x;

    thrust::complex<T> sum(static_cast<T>(0), static_cast<T>(0));

    for (int z = 0; z < inDimZ; ++z) {
      // TODO(niclas): Make this a macro.
      const int gamma_idx = z * 2 * inDimX * inDimY + greens_idx;

      const thrust::complex<T> gamma(inGammaReal[gamma_idx],
                                     inGammaImag[gamma_idx]);
      const thrust::complex<T> gammaTilde(inGammaTildeReal[gamma_idx],
                                          inGammaTildeImag[gamma_idx]);

      const thrust::complex<T> val = GREENS<T>::compute(gamma, gammaTilde);
      sum += inSummandFactor[z] * val;
    }

    outResultReal[greens_idx] += sum.real();
    outResultImag[greens_idx] += sum.imag();
  }
}
} // namespace internal

// Note, all these kernel calls can be optimized by using
// getKernelDimensionsOptim().

template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreens(
    const GridGPU<T> *const inGamma, const GridGPU<T> *const inGammaTilde,
    GridGPU<T> *const outResult) {
  // Sanity checks.
  assert(inGamma->dimX() == inGammaTilde->dimX());
  assert(inGamma->dimY() == inGammaTilde->dimY());
  assert(inGamma->numFields() == inGammaTilde->numFields());
  assert(inGamma->dimX() == outResult->dimX());
  assert(inGamma->dimY() == outResult->dimY());
  assert(inGamma->numFields() == outResult->numFields());
  assert(inGamma->isComplex());
  assert(inGammaTilde->isComplex());
  assert(outResult->isComplex());

  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] =
      outResult->getKernelDimensionsFields();

  // Dimensions.
  const int dimX = inGamma->dimX();
  const int dimY = inGamma->dimY();
  const int numEnergies = inGamma->numFields();

  // Compute Greens function from gamma and gammaTilde
  internal::computeGreens_kernel<T, GREENS><<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, numEnergies, inGamma->getDataPointer(0, Type::real),
      inGamma->getDataPointer(0, Type::imag),
      inGammaTilde->getDataPointer(0, Type::real),
      inGammaTilde->getDataPointer(0, Type::imag),
      outResult->getDataPointer(0, Type::real),
      outResult->getDataPointer(0, Type::imag));
}

template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    const GridGPU<T> *const inGamma, const GridGPU<T> *const inGammaTilde,
    GridGPU<T> *const outResult) {
  // Sanity checks.
  assert(inGamma->dimX() == inGammaTilde->dimX());
  assert(inGamma->dimY() == inGammaTilde->dimY());
  assert(inGamma->numFields() == inGammaTilde->numFields());
  assert(inGamma->dimX() == outResult->dimX());
  assert(inGamma->dimY() == outResult->dimY());
  assert(outResult->numFields() == 1);
  assert(inGamma->isComplex());
  assert(inGammaTilde->isComplex());
  assert(outResult->isComplex());

  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = outResult->getKernelDimensions();

  // Dimensions.
  const int dimX = inGamma->dimX();
  const int dimY = inGamma->dimY();
  const int numEnergies = inGamma->numFields();

  // Compute Greens function from gamma and gammaTilde.
  internal::computeGreensSumEnergy_kernel<T, GREENS>
      <<<numBlocks, numThreadsPerBlock>>>(
          dimX, dimY, numEnergies, inGamma->getDataPointer(0, Type::real),
          inGamma->getDataPointer(0, Type::imag),
          inGammaTilde->getDataPointer(0, Type::real),
          inGammaTilde->getDataPointer(0, Type::imag),
          outResult->getDataPointer(0, Type::real),
          outResult->getDataPointer(0, Type::imag));
}

template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    const GridGPU<T> *const inGamma, const GridGPU<T> *const inGammaTilde,
    const GridGPU<T> *const inResidues, GridGPU<T> *const outResult) {
  // Sanity checks.
  assert(inGamma->dimX() == inGammaTilde->dimX());
  assert(inGamma->dimY() == inGammaTilde->dimY());
  assert(inGamma->numFields() == inGammaTilde->numFields());
  assert(inGamma->dimX() == outResult->dimX());
  assert(inGamma->dimY() == outResult->dimY());
  assert(inGamma->numFields() == inResidues->numElements());
  assert(outResult->numFields() == 1);
  assert(inGamma->isComplex());
  assert(inGammaTilde->isComplex());
  assert(outResult->isComplex());

  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = outResult->getKernelDimensions();

  // Dimensions.
  const int dimX = inGamma->dimX();
  const int dimY = inGamma->dimY();
  const int numEnergies = inGamma->numFields();

  // Compute Greens function from gamma and gammaTilde.
  internal::computeGreensSumEnergy_kernel<T, GREENS>
      <<<numBlocks, numThreadsPerBlock>>>(
          dimX, dimY, numEnergies, inGamma->getDataPointer(0, Type::real),
          inGamma->getDataPointer(0, Type::imag),
          inGammaTilde->getDataPointer(0, Type::real),
          inGammaTilde->getDataPointer(0, Type::imag),
          inResidues->getDataPointer(0, Type::real),
          outResult->getDataPointer(0, Type::real),
          outResult->getDataPointer(0, Type::imag));
}
} // namespace conga

#endif // CONGA_GREENS_FUNCTION_STATIC_H_
