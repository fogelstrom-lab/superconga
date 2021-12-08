#ifndef CONGA_DOMAIN_OPS_H_
#define CONGA_DOMAIN_OPS_H_

#include "../grid.h"

#include <cuda.h>

namespace conga {
namespace geometry {
namespace internal {
// Slow operation, so we have two explicit kernels, one for complex and one for
// real valued.
template <typename T>
__global__ void extendValuesAtBoundary_kernel( // complex
    const char *const dom_ptr, T *grid_real, T *grid_imag, int Nx_geom,
    int Ny_geom, int Nx_target, int Ny_target, int Nx_geom_offset,
    int Ny_geom_offset, const int numExtend) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tf = blockIdx.z;

  if (tx < Nx_target && ty < Ny_target) {
    int idx_geom = (ty + Ny_geom_offset) * Nx_geom + (tx + Nx_geom_offset);

    int i, j, idx_neighbor, idx_geom_neighbor;

    T interp_real = static_cast<T>(0.0);
    T interp_imag = static_cast<T>(0.0);

    int numNeighbors = 0;

    if (dom_ptr[idx_geom] == 0) {
      for (j = -numExtend; j <= numExtend; ++j) {
        for (i = -numExtend; i <= numExtend; ++i) {
          if ((tx + i) > 0 && (tx + i) < Nx_target && (ty + j) > 0 &&
              (ty + j) < Ny_target) {
            idx_geom_neighbor =
                (ty + j + Ny_geom_offset) * Nx_geom + (tx + i + Nx_geom_offset);

            if (dom_ptr[idx_geom_neighbor] > 0) {
              idx_neighbor = tf * 2 * Nx_target * Ny_target +
                             (ty + j) * Nx_target + (tx + i);

              interp_real += grid_real[idx_neighbor];
              interp_imag += grid_imag[idx_neighbor];

              numNeighbors++;
            }
          }
        }
      }
    }

    if (numNeighbors) {
      const int idx = tf * 2 * Nx_target * Ny_target + ty * Nx_target + tx;
      grid_real[idx] = interp_real / (T)numNeighbors;
      grid_imag[idx] = interp_imag / (T)numNeighbors;
    }
  }
}

template <typename T>
__global__ void extendValuesAtBoundary_kernel( // real
    const char *const dom_ptr, T *grid_real, int Nx_geom, int Ny_geom,
    int Nx_target, int Ny_target, int Nx_geom_offset, int Ny_geom_offset,
    const int numExtend) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tf = blockIdx.z;

  if (tx < Nx_target && ty < Ny_target) {
    int idx_geom = (ty + Ny_geom_offset) * Nx_geom + (tx + Nx_geom_offset);

    int i, j, idx_neighbor, idx_geom_neighbor;

    T interp_real = static_cast<T>(0.0);

    int numNeighbors = 0;

    if (dom_ptr[idx_geom] == 0) {
      for (j = -numExtend; j <= numExtend; ++j) {
        for (i = -numExtend; i <= numExtend; ++i) {
          if ((tx + i) > 0 && (tx + i) < Nx_target && (ty + j) > 0 &&
              (ty + j) < Ny_target) {
            idx_geom_neighbor =
                (ty + j + Ny_geom_offset) * Nx_geom + (tx + i + Nx_geom_offset);

            if (dom_ptr[idx_geom_neighbor] > 0) {
              idx_neighbor =
                  tf * Nx_target * Ny_target + (ty + j) * Nx_target + (tx + i);

              interp_real += grid_real[idx_neighbor];

              numNeighbors++;
            }
          }
        }
      }
    }

    if (numNeighbors) {
      const int idx = tf * Nx_target * Ny_target + ty * Nx_target + tx;
      grid_real[idx] = interp_real / (T)numNeighbors;
    }
  }
}
} // namespace internal

template <typename T>
void extendValuesAtBoundary(const GridGPU<char> &inDomainLabels,
                            GridGPU<T> &inOutTarget) {
  const int Nx_geom = inDomainLabels.dimX();
  const int Ny_geom = inDomainLabels.dimY();

  const int Nx_target = inOutTarget.dimX();
  const int Ny_target = inOutTarget.dimY();

  const int Nx_geom_offset = (Nx_geom - Nx_target) / 2;
  const int Ny_geom_offset = (Ny_geom - Ny_target) / 2;

  const int Nf = inOutTarget.numFields();

  // Set kernel dimensions
  dim3 bpg, tpb;
  inOutTarget.getKernelDimensionsFields(bpg, tpb);

  // This is a sensible value. It probably does not need to be exposed to the
  // user. Note that 2 is chosen due to corner artifacts otherwise.
  // TODO(niclas): Have a separate domain label for corners, and handle them
  // separately.
  const int numExtend = 2;

  if (inOutTarget.isComplex()) {
    internal::extendValuesAtBoundary_kernel<T><<<bpg, tpb>>>(
        inDomainLabels.getDataPointer(),
        inOutTarget.getDataPointer(0, Type::real),
        inOutTarget.getDataPointer(0, Type::imag), Nx_geom, Ny_geom, Nx_target,
        Ny_target, Nx_geom_offset, Ny_geom_offset, numExtend);
  } else {
    internal::extendValuesAtBoundary_kernel<T><<<bpg, tpb>>>(
        inDomainLabels.getDataPointer(),
        inOutTarget.getDataPointer(0, Type::real), Nx_geom, Ny_geom, Nx_target,
        Ny_target, Nx_geom_offset, Ny_geom_offset, numExtend);
  }
}
} // namespace geometry
} // namespace conga

#endif // CONGA_DOMAIN_OPS_H_
