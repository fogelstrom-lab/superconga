#ifndef ORDERPARAMETERCOMPONENT_KERNEL_H
#define ORDERPARAMETERCOMPONENT_KERNEL_H

#include <cplx.h>
#include <cuda.h>

template <typename T> struct Basis_S {
  __host__ __device__ static cplx<T> compute(const T total_angle) {
    return cplx<T>(T(1.0), T(0));
  }
};

template <typename T> struct Basis_D {
  __host__ __device__ static cplx<T> compute(const T total_angle) {
    return cplx<T>(sqrt((T)2.0) * cos((T)2.0 * (total_angle)), T(0));
  }
};

template <typename T> struct Basis_P {
  __host__ __device__ static cplx<T> compute(const T total_angle) {
    return cplx<T>(cos(total_angle), sin(total_angle));
  }
};

template <typename T, template <typename> class BASIS>
__global__ void
computeMomentumDependence_kernel(T *target_re, T *target_im, T *orientation,
                                 const T orientation_offset, const T angle,
                                 const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = ty * Nx + tx;

  if (tx < Nx && ty < Ny) {
    const T localAngle = orientation[idx];

    const T total_angle =
        angle + localAngle + orientation_offset + (T)0.5 * (T)M_PI;
    cplx<T> tmp(target_re[idx], target_im[idx]);

    tmp *= BASIS<T>::compute(total_angle);

    target_re[idx] = tmp.x;
    target_im[idx] = tmp.y;
  }
}

#endif // ORDERPARAMETERCOMPONENT_KERNEL_H
