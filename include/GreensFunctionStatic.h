#ifndef GREENS_FUNCTION_STATIC_H_
#define GREENS_FUNCTION_STATIC_H_

#include "Grid.h"
#include "LocalGreens.h"
#include "ParameterIterator.h"
#include "Parameters.h"
#include <thrust/complex.h>

namespace scg {
template <typename> class GeometryGroup;

template <typename T, template <typename> class GREENS>
struct GreensFunctionStatic {
  // Greens function functions
  static void computeGreens(GridGPU<T> *result,
                            const ParameterIterator<int> *const iter,
                            const GridGPU<T> *const gamma,
                            const GridGPU<T> *const gammabar);

  static void computeGreensSumEnergy(GridGPU<T> *result,
                                     const ParameterIterator<int> *const iter,
                                     const GridGPU<T> *const gamma,
                                     const GridGPU<T> *const gammabar);

  static void computeGreensSumEnergy(GridGPU<T> *result,
                                     const ParameterIterator<int> *const iter,
                                     const GridGPU<T> *const summandFactor,
                                     const GridGPU<T> *const gamma,
                                     const GridGPU<T> *const gammabar);

  // Helper functions
  static void getKernelDimensions(dim3 &blocksPerGrid, dim3 &threadsPerBlock,
                                  const GridGPU<T> *grid, int numEnergies = 0);
};

namespace internal {
template <typename T, template <typename> class GREENS>
__global__ void
computeGreens_kernel(const int N, T *greens_re, T *greens_im,
                     const T *const gamma_re, const T *const gamma_im,
                     const T *const gammabar_re, const T *const gammabar_im) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x; // x index
  const int y = blockIdx.y * blockDim.y + threadIdx.y; // y index
  const int e_idx = blockIdx.z;                        // energy index

  if (x < N) {
    const int gamma_idx = e_idx * 2 * N * N + y * N + x;

    const thrust::complex<T> gamma(gamma_re[gamma_idx], gamma_im[gamma_idx]);
    const thrust::complex<T> gammabar(gammabar_re[gamma_idx],
                                      gammabar_im[gamma_idx]);

    thrust::complex<T> greens_local;
    GREENS<T>::computeGreens_local(gamma, gammabar, greens_local);

    greens_re[gamma_idx] = greens_local.real();
    greens_im[gamma_idx] = greens_local.imag();
  }
}

template <typename T, template <typename> class GREENS>
__global__ void computeGreensSumEnergy_kernel(const int N, const int Ne,
                                              T *greens_re, T *greens_im,
                                              const T *const gamma_re,
                                              const T *const gamma_im,
                                              const T *const gammabar_re,
                                              const T *const gammabar_im) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  if (x < N) {
    const int greens_idx = y * N + x;

    thrust::complex<T> greens_sum(static_cast<T>(0), static_cast<T>(0));

    for (int e_idx = 0; e_idx < Ne; ++e_idx) {
      const int gamma_idx = e_idx * 2 * N * N + greens_idx;

      const thrust::complex<T> gamma(gamma_re[gamma_idx], gamma_im[gamma_idx]);
      const thrust::complex<T> gammabar(gammabar_re[gamma_idx],
                                        gammabar_im[gamma_idx]);

      thrust::complex<T> greens_local;
      GREENS<T>::computeGreens_local(gamma, gammabar, greens_local);

      greens_sum += greens_local;
    }

    greens_re[greens_idx] += greens_sum.real();
    greens_im[greens_idx] += greens_sum.imag();
  }
}

template <typename T, template <typename> class GREENS>
__global__ void computeGreensSumEnergy_kernel(
    const int N, const int Ne, const T *const summandFactor, T *greens_re,
    T *greens_im, const T *const gamma_re, const T *const gamma_im,
    const T *const gammabar_re, const T *const gammabar_im) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  if (x < N) {
    const int greens_idx = y * N + x;

    thrust::complex<T> greens_sum(static_cast<T>(0), static_cast<T>(0));

    for (int e_idx = 0; e_idx < Ne; ++e_idx) {
      const int gamma_idx = e_idx * 2 * N * N + greens_idx;

      const thrust::complex<T> gamma(gamma_re[gamma_idx], gamma_im[gamma_idx]);
      const thrust::complex<T> gammabar(gammabar_re[gamma_idx],
                                        gammabar_im[gamma_idx]);

      thrust::complex<T> greens_local;
      GREENS<T>::computeGreens_local(gamma, gammabar, greens_local);

      greens_sum += summandFactor[e_idx] * greens_local;
    }

    greens_re[greens_idx] += greens_sum.real();
    greens_im[greens_idx] += greens_sum.imag();
  }
}
} // namespace internal

// All these kernels can be optimized by using getKernelDimensionsOptim() !!!!

// Compute greens function for all energies, without integrating (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreens(
    GridGPU<T> *result, const ParameterIterator<int> *iter,
    const GridGPU<T> *gamma, const GridGPU<T> *gammabar) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GreensFunctionStatic<T, GREENS>::getKernelDimensions(bpg, tpb, result,
                                                       numEnergies);

  // Compute Greens function from gamma and gammabar
  internal::computeGreens_kernel<T, GREENS>
      <<<bpg, tpb>>>(result->dimX(), result->getDataPointer(0, Type::real),
                     result->getDataPointer(0, Type::imag),
                     gamma->getDataPointer(0, Type::real),
                     gamma->getDataPointer(0, Type::imag),
                     gammabar->getDataPointer(0, Type::real),
                     gammabar->getDataPointer(0, Type::imag));
}

// Compute greens function for all energies and integrate (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    GridGPU<T> *result, const ParameterIterator<int> *iter,
    const GridGPU<T> *gamma, const GridGPU<T> *gammabar) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GridGPU<T>::getKernelDimensions(bpg, tpb, result->dimX(), result->dimY(), 1);

  // Compute Greens function from gamma and gammabar
  internal::computeGreensSumEnergy_kernel<T, GREENS><<<bpg, tpb>>>(
      result->dimX(), numEnergies, result->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::imag),
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      gammabar->getDataPointer(0, Type::real),
      gammabar->getDataPointer(0, Type::imag));
}

// Compute greens function for all energies and integrate (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    GridGPU<T> *result, const ParameterIterator<int> *const iter,
    const GridGPU<T> *const summandFactor, const GridGPU<T> *const gamma,
    const GridGPU<T> *const gammabar) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GridGPU<T>::getKernelDimensions(bpg, tpb, result->dimX(), result->dimY(), 1);

  // Compute Greens function from gamma and gammabar
  internal::computeGreensSumEnergy_kernel<T, GREENS><<<bpg, tpb>>>(
      result->dimX(), numEnergies, summandFactor->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::imag),
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      gammabar->getDataPointer(0, Type::real),
      gammabar->getDataPointer(0, Type::imag));
}

template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::getKernelDimensions(
    dim3 &blocksPerGrid, dim3 &threadsPerBlock, const GridGPU<T> *grid,
    int numEnergies) {
  const int N = grid->dimX();

  if (numEnergies == 0)
    numEnergies = grid->numFields();

  int tpb_x, tpb_y, bpg_x, bpg_y;

  tpb_x = N;
  tpb_y = 1;

  bpg_x = 1;

  if (tpb_x > 1024) {
    tpb_x = N / 2 + 1;
    bpg_x = 2;
  }

  bpg_y = N;

  threadsPerBlock = dim3(tpb_x, tpb_y, 1);
  blocksPerGrid = dim3(bpg_x, bpg_y, numEnergies);
}
} // namespace scg

#endif // GREENS_FUNCTION_STATIC_H_
