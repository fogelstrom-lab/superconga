#ifndef ORDERPARAMETERBASIS_H_
#define ORDERPARAMETERBASIS_H_

#include <cmath>
#include <cplx.h>
#include <cuda.h>

template <typename T> struct Basis_S {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = static_cast<T>(1.0);
    const T etaImag = static_cast<T>(0.0);
    return cplx<T>(etaReal, etaImag);
  }
};

template <typename T> struct Basis_D {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = cosf(pX) - cosf(pY);
    const T etaImag = static_cast<T>(0.0);
    return cplx<T>(etaReal, etaImag);
  }
};

template <typename T> struct Basis_P {
  __host__ __device__ static cplx<T> compute(const T pX, const T pY) {
    const T etaReal = sinf(pX);
    const T etaImag = sinf(pY);
    return cplx<T>(etaReal, etaImag);
  }
};

#endif // ORDERPARAMETERBASIS_H_
