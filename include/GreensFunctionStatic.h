#ifndef GREENS_FUNCTION_STATIC_H_
#define GREENS_FUNCTION_STATIC_H_

#include "LocalGreens.h"
#include "Grid.h"
#include "ParameterIterator.h"
#include "Parameters.h"
#include "cplx.h"

template <typename> class GeometryGroup;

template <typename T, template <typename> class GREENS>
struct GreensFunctionStatic {
  typedef typename Grid<T>::GPU GridGPU;

  // Greens function functions
  static void computeGreens(GridGPU *result, const ParameterIterator<int> *iter,
                            const GridGPU *gamma, const GridGPU *gammabar,
                            const GeometryGroup<T> *geom);

  static void computeGreensSumEnergy(GridGPU *result,
                                     const ParameterIterator<int> *iter,
                                     const GridGPU *gamma,
                                     const GridGPU *gammabar,
                                     const GeometryGroup<T> *geom);

  static void
  computeGreensSumEnergy(GridGPU *result, const ParameterIterator<int> *iter,
                         const GridGPU *summandFactor, const GridGPU *gamma,
                         const GridGPU *gammabar, const GeometryGroup<T> *geom);

  // Helper functions
  static void getKernelDimensions(dim3 &blocksPerGrid, dim3 &threadsPerBlock,
                                  const GridGPU *grid, int numEnergies = 0);
};

namespace internal {
template <typename T, template <typename> class GREENS>
__global__ void computeGreens_kernel(const int N, T *greens_re, T *greens_im,
                                     T *gamma_re, T *gamma_im, T *gammabar_re,
                                     T *gammabar_im, char *domainLabels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x; // x index
  const int y = blockIdx.y * blockDim.y + threadIdx.y; // y index
  const int e_idx = blockIdx.z;                        // energy index

  if (x < N) {
    const int domain_idx = y * N + x;
    const int gamma_idx = e_idx * 2 * N * N + y * N + x;

    cplx<T> greens_local, gamma(gamma_re[gamma_idx], gamma_im[gamma_idx]),
        gammabar(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

    if (domainLabels[domain_idx] > 0) {
      GREENS<T>::computeGreens_local(gamma, gammabar, greens_local);
    }

    greens_re[gamma_idx] = greens_local.x;
    greens_im[gamma_idx] = greens_local.y;
  }
}

template <typename T, template <typename> class GREENS>
__global__ void computeGreensSumEnergy_kernel(const int N, const int Ne,
                                              T *greens_re, T *greens_im,
                                              T *gamma_re, T *gamma_im,
                                              T *gammabar_re, T *gammabar_im,
                                              char *domainLabels) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  int gamma_idx, greens_idx;
  int e_idx;

  if (x < N) {
    greens_idx = y * N + x;

    cplx<T> greensTmp, greens_local, gamma, gammabar,
        greens_sum((T)0.0, (T)0.0);

    if (domainLabels[greens_idx] > 0) {
      for (e_idx = 0; e_idx < Ne; ++e_idx) {
        gamma_idx = e_idx * 2 * N * N + greens_idx;

        gamma = cplx<T>(gamma_re[gamma_idx], gamma_im[gamma_idx]);
        gammabar = cplx<T>(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

        GREENS<T>::computeGreens_local(gamma, gammabar, greensTmp);

        greens_sum += greensTmp;
      }
    }

    greens_re[greens_idx] += greens_sum.x;
    greens_im[greens_idx] += greens_sum.y;
  }
}

template <typename T, template <typename> class GREENS>
__global__ void
computeGreensSumEnergy_kernel(const int N, const int Ne, const T *summandFactor,
                              T *greens_re, T *greens_im, T *gamma_re,
                              T *gamma_im, T *gammabar_re, T *gammabar_im,
                              char *domainLabels) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  int gamma_idx, greens_idx;
  int e_idx;

  if (x < N) {
    greens_idx = y * N + x;

    cplx<T> greensTmp, greens_local, gamma, gammabar,
        greens_sum((T)0.0, (T)0.0);

    if (domainLabels[greens_idx] > 0) {
      for (e_idx = 0; e_idx < Ne; ++e_idx) {
        gamma_idx = e_idx * 2 * N * N + greens_idx;

        gamma = cplx<T>(gamma_re[gamma_idx], gamma_im[gamma_idx]);
        gammabar = cplx<T>(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

        GREENS<T>::computeGreens_local(gamma, gammabar, greensTmp);

        greens_sum += summandFactor[e_idx] * greensTmp;
      }
    }

    greens_re[greens_idx] += greens_sum.x;
    greens_im[greens_idx] += greens_sum.y;
  }
}
} // namespace internal

// All these kernels can be optimized by using getKernelDimensionsOptim() !!!!

// Compute greens function for all energies, without integrating (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreens(
    GridGPU *result, const ParameterIterator<int> *iter, const GridGPU *gamma,
    const GridGPU *gammabar, const GeometryGroup<T> *geom) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GreensFunctionStatic<T, GREENS>::getKernelDimensions(bpg, tpb, result,
                                                       numEnergies);

  // Compute Greens function from gamma and gammabar
  internal::computeGreens_kernel<T, GREENS><<<bpg, tpb>>>(
      result->dimX(), result->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::imag),
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      gammabar->getDataPointer(0, Type::real),
      gammabar->getDataPointer(0, Type::imag),
      geom->getDomainLabelsGrid()->getDataPointer(0, Type::real));
}

// Compute greens function for all energies and integrate (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    GridGPU *result, const ParameterIterator<int> *iter, const GridGPU *gamma,
    const GridGPU *gammabar, const GeometryGroup<T> *geom) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GridGPU::getKernelDimensions(bpg, tpb, result->dimX(), result->dimY(), 1);

  // Compute Greens function from gamma and gammabar
  internal::computeGreensSumEnergy_kernel<T, GREENS><<<bpg, tpb>>>(
      result->dimX(), numEnergies, result->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::imag),
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      gammabar->getDataPointer(0, Type::real),
      gammabar->getDataPointer(0, Type::imag),
      geom->getDomainLabelsGrid()->getDataPointer(0, Type::real));
}

// Compute greens function for all energies and integrate (over energies)
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::computeGreensSumEnergy(
    GridGPU *result, const ParameterIterator<int> *iter,
    const GridGPU *summandFactor, const GridGPU *gamma, const GridGPU *gammabar,
    const GeometryGroup<T> *geom) {
  int numEnergies = iter->getNumInterval();

  dim3 bpg, tpb;
  GridGPU::getKernelDimensions(bpg, tpb, result->dimX(), result->dimY(), 1);

  // Compute Greens function from gamma and gammabar
  internal::computeGreensSumEnergy_kernel<T, GREENS><<<bpg, tpb>>>(
      result->dimX(), numEnergies, summandFactor->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::imag),
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      gammabar->getDataPointer(0, Type::real),
      gammabar->getDataPointer(0, Type::imag),
      geom->getDomainLabelsGrid()->getDataPointer(0, Type::real));
}

template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T, GREENS>::getKernelDimensions(dim3 &blocksPerGrid,
                                                          dim3 &threadsPerBlock,
                                                          const GridGPU *grid,
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

#endif // GREENS_FUNCTION_STATIC_H_
