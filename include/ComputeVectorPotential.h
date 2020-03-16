#ifndef COMPUTE_VECTOR_POTENTIAL_H_
#define COMPUTE_VECTOR_POTENTIAL_H_

#include "GeometryGroup.h"
#include "Grid.h"
#include <cuda.h>

namespace internal {
// Compute vector potential by integrating greens function kernel
template <typename T>
__global__ void
computeVectorPotential2d_kernel(T *Ax, T *Ay, const T *jx, const T *jy,
                                const char *geom, const int px, const int py,
                                const T h, const T c, const int Nx) {
  // Coordinates to grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = y * Nx + x;
  const int idxp = py * Nx + px;

  if (x < Nx && y < Nx) {
    T mult, h2 = h * h;

    if (x == px && y == py) {
      mult = c * 0.25 * h2 * (M_PI - 6.0 + 2.0 * log(h2 * 0.5));
    } else {
      const T dx = px - x;
      const T dy = py - y;

      mult = c * h2 * log(h * sqrt(dx * dx + dy * dy));
    }

    Ax[idx] += mult * jx[idxp];
    Ay[idx] += mult * jy[idxp];
  }
}
} // namespace internal

template <typename T>
void computeVectorPotential(GridData<T, thrust::device_vector<T>> *A,
                            const GridData<T, thrust::device_vector<T>> *j,
                            GridData<char, thrust::device_vector<char>> *geom,
                            const T h, const T c) {
  A->setZero();

  dim3 bpg, tpb;
  A->getKernelDimensions(bpg, tpb);

  const int N = A->dimX();

  GridData<char, thrust::host_vector<char>> geom_host = *geom;
  char *g = geom_host.getDataPointer(0);

  // Note, the vector potential is computed for all lattice sites, not just the
  // ones within the grain.
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      if (g[y * N + x]) {
        internal::computeVectorPotential2d_kernel<<<bpg, tpb>>>(
            A->getDataPointer(0), A->getDataPointer(1), j->getDataPointer(0),
            j->getDataPointer(1), geom->getDataPointer(0), x, y, h, c, N);
      }
    }
  }
  *A *= (-1.0 / (2.0 * M_PI));
}

#endif // COMPUTE_VECTOR_POTENTIAL_H_
