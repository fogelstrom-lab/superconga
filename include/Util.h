//
// Useful functions not belonging to any particular part of the framework of the
// self-consistent iterations.
//
#ifndef UTIL_H
#define UTIL_H

#include <Grid.h>
#include <cuda.h>
#include <math.h>

// TODO(niclas): Is it ever used? If so, remove!
template <typename T> struct Util {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  static void mirror_diag_lower_to_upper(typename Grid<T>::GPU *grid);
};

namespace {
template <typename T>
__global__ void mirror_diag_lower_to_upper_kernel(T *data, const int N) {
  // Coordinates to grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = y * N + x;

  if (y < x && x < N) {
    const int idx_dst = x * N + y;
    data[idx_dst] = data[idx];
  }
}
} // namespace

template <typename T>
void Util<T>::mirror_diag_lower_to_upper(typename Grid<T>::GPU *grid) {
  const int N = grid->dimX();

  // GPU init
  dim3 blocksPerGrid, threadsPerBlock;
  grid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  // Kernel launch
  for (int c = 0; c < grid->numFields(); ++c) {
    mirror_diag_lower_to_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        grid->getDataPointer(c, Type::real), N);

    if (grid->isComplex())
      mirror_diag_lower_to_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::imag), N);
  }
}

#endif // UTIL_H
