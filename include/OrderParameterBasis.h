#ifndef ORDER_PARAMETER_BASIS_H_
#define ORDER_PARAMETER_BASIS_H_

#include "cplx.h"
#include <cuda.h>
#include <math.h>

template <typename T> struct Basis_S {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = static_cast<T>(1.0);
    const T etaImag = static_cast<T>(0.0);
    return cplx<T>(etaReal, etaImag);
  }
};

template <typename T> struct Basis_D {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = cos(pX) - cos(pY);
    const T etaImag = static_cast<T>(0.0);
    return cplx<T>(etaReal, etaImag);
  }
};

template <typename T> struct Basis_P {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = sin(pX);
    const T etaImag = sin(pY);
    return cplx<T>(etaReal, etaImag);
  }
};

#endif // ORDER_PARAMETER_BASIS_H_
