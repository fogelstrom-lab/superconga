#ifndef CONGA_DOMAIN_OPS_H_
#define CONGA_DOMAIN_OPS_H_

#include "../grid.h"

#include "conga/backend.h"

namespace conga {
namespace geometry {
namespace internal {
template <typename T>
__global__ void extendValuesAtBoundary_kernel(
    const int inDimX, const int inDimY, const int inDimZ, const int inNumExtend,
    const char *const inDomainLabels, T *const inOutTarget) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    const int idxDomain = y * inDimX + x;
    const char domainLabel = inDomainLabels[idxDomain];

    if (domainLabel == 0) {
      // Extent of the window along the x-axis.
      const int xMin = max(x - inNumExtend, 0);
      const int xMax = min(x + inNumExtend, inDimX - 1);

      // Extent of the window along the y-axis.
      const int yMin = max(y - inNumExtend, 0);
      const int yMax = min(y + inNumExtend, inDimY - 1);

      T val = static_cast<T>(0);
      int numPoints = 0;
      int minSquaredDistance = 1000;

      for (int ky = yMin; ky <= yMax; ++ky) {
        for (int kx = xMin; kx <= xMax; ++kx) {
          const int idxDomain = ky * inDimX + kx;
          const char domainLabel = inDomainLabels[idxDomain];

          if (domainLabel > 0) {
            // The squared distance to the pixel from the window center.
            const int dx = x - kx;
            const int dy = y - ky;
            const int squaredDistance = dx * dx + dy * dy;

            // The index of this point.
            const int idxNeighbour = (z * inDimY + ky) * inDimX + kx;

            // Update if better or equal.
            if (squaredDistance < minSquaredDistance) {
              val = inOutTarget[idxNeighbour];
              numPoints = 1;
              minSquaredDistance = squaredDistance;
            } else if (squaredDistance == minSquaredDistance) {
              val += inOutTarget[idxNeighbour];
              numPoints++;
            }
          }
        }
      }

      if (numPoints > 0) {
        const int idx = (z * inDimY + y) * inDimX + x;
        inOutTarget[idx] = val / static_cast<T>(numPoints);
      }
    }
  }
}
} // namespace internal

template <typename T>
void extendValuesAtBoundary(const GridGPU<char> &inDomainLabels,
                            GridGPU<T> &inOutTarget) {
  // Sanity checks.
  assert(inDomainLabels.dimX() == inOutTarget.dimX());
  assert(inDomainLabels.dimY() == inOutTarget.dimY());

  // Dimensions.
  const int dimX = inOutTarget.dimX();
  const int dimY = inOutTarget.dimY();
  const int complexFactor = inOutTarget.isComplex() ? 2 : 1;
  const int dimZ = complexFactor * inOutTarget.numFields();

  // This is a sensible value. It probably does not need to be exposed to the
  // user. Note that 2 is chosen due to corner artifacts otherwise.
  // TODO(niclas): Have a separate domain label for corners, and handle them
  // separately.
  const int numExtend = 2;

  // Kernel parameters.
  const auto [blocksPerGrid, threadsPerBlock] =
      inOutTarget.getKernelDimensions(dimX, dimY, dimZ);

  // Launch kernel.
  CONGA_LAUNCH((internal::extendValuesAtBoundary_kernel<T>), blocksPerGrid,
               threadsPerBlock, dimX, dimY, dimZ, numExtend,
               inDomainLabels.getDataPointer(), inOutTarget.getDataPointer());
}
} // namespace geometry
} // namespace conga

#endif // CONGA_DOMAIN_OPS_H_
