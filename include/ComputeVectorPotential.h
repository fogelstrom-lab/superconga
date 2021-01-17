#ifndef COMPUTE_VECTOR_POTENTIAL_H_
#define COMPUTE_VECTOR_POTENTIAL_H_

#include "GeometryGroup.h"
#include "Grid.h"
#include <cuda.h>

namespace scg {
namespace internal {
// Compute vector potential by integrating greens function kernel
template <typename T>
__global__ void computeVectorPotential2d_kernel(
    T *const inOutVectorPotentialX, T *const inOutVectorPotentialY,
    const T *const inCurrentDensityX, const T *const inCurrentDensityY,
    const int px, const int py, const T inLatticeSpacing, const int Nx) {
  // Coordinates to grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < Nx && y < Nx) {
    // For convenience.
    const T latticeArea = inLatticeSpacing * inLatticeSpacing;

    T factor;
    if (x == px && y == py) {
      factor = static_cast<T>(0.25) * latticeArea *
               (static_cast<T>(M_PI - 6.0) +
                static_cast<T>(2.0) * log(latticeArea * static_cast<T>(0.5)));
    } else {
      const T dx = static_cast<T>(px - x);
      const T dy = static_cast<T>(py - y);
      factor = latticeArea * log(inLatticeSpacing * sqrt(dx * dx + dy * dy));
    }

    // Vector-potential index.
    const int idx = y * Nx + x;

    // Current-density index.
    const int idxp = py * Nx + px;

    inOutVectorPotentialX[idx] += factor * inCurrentDensityX[idxp];
    inOutVectorPotentialY[idx] += factor * inCurrentDensityY[idxp];
  }
}
} // namespace internal

template <typename T>
void computeVectorPotential(GridGPU<T> &inOutVectorPotential,
                            const GridGPU<T> &inCurrentDensity,
                            const GridGPU<char> &inDomainLabels,
                            const T inLatticeSpacing, const T inKappa) {
  // Set to zero because we'll add stuff to it.
  inOutVectorPotential.setZero();

  // Get the required kernel dimensions.
  dim3 bpg, tpb;
  inOutVectorPotential.getKernelDimensions(bpg, tpb);

  const int N = inOutVectorPotential.dimX();

  // Get the thrust-vector from the domain labels, and copy to the CPU.
  const thrust::host_vector<char> domainLabelsCPU = *inDomainLabels.getVector();
  const char *const domainLabelsPtr =
      thrust::raw_pointer_cast(domainLabelsCPU.data());

  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      if (domainLabelsPtr[y * N + x] > 0) {
        internal::computeVectorPotential2d_kernel<<<bpg, tpb>>>(
            inOutVectorPotential.getDataPointer(0),
            inOutVectorPotential.getDataPointer(1),
            inCurrentDensity.getDataPointer(0),
            inCurrentDensity.getDataPointer(1), x, y, inLatticeSpacing, N);
      }
    }
  }

  // Factor from the equation itself.
  const T equationFactor = -static_cast<T>(4.0) / (inKappa * inKappa);

  // Factor from the Green function.
  const T greenFactor = static_cast<T>(1) / static_cast<T>(2.0 * M_PI);

  // Scale the result appropriately.
  inOutVectorPotential *= equationFactor * greenFactor;
}
} // namespace scg

#endif // COMPUTE_VECTOR_POTENTIAL_H_
