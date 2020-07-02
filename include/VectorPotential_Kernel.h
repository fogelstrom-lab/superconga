#ifndef VECTOR_POTENTIAL_KERNEL_H_
#define VECTOR_POTENTIAL_KERNEL_H_

#include <cuda.h>

namespace scg {
template <typename T, template <typename> class POTENTIAL>
__global__ void
computeEnergyShift_external_kernel(T *eShift, const T systemAngle, const T B,
                                   const int N, const T h) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  const int idx = y * N + x;

  if (x < N && y < N) {
    eShift[idx] = T(M_PI) * POTENTIAL<T>::computeVectorPotential_local(
                                (T)x, (T)y, B, systemAngle, N, h);
  }
}

template <typename T>
__global__ void
computeEnergyShift_internal_kernel(T *eShift, const T *jx, const T *jy,
                                   const T systemAngle, const int N) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  const int idx = y * N + x;

  if (x < N && y < N)
    eShift[idx] +=
        T(M_PI) * (cos(systemAngle) * jx[idx] + sin(systemAngle) * jy[idx]);
}
} // namespace scg

#endif // VECTOR_POTENTIAL_KERNEL_H_
