#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <cuda.h>
#include <math.h>

namespace utils {
template <typename T>
__host__ __device__ T wrapToRange(const T value, const T low, const T high) {
  const T diff = high - low;
  const T tmp = value < low ? value + diff : value;
  const T wrappedValue = fmod(tmp - low, diff) + low;
  return wrappedValue;
}

template <>
__host__ __device__ int wrapToRange<int>(const int value, const int low,
                                         const int high) {
  const int diff = high - low;
  const int tmp = value < low ? value + diff : value;
  const int wrappedValue = ((tmp - low) % diff) + low;
  return wrappedValue;
}

template <typename T> __host__ __device__ T wrapToRangeTwoPi(const T value) {
  return wrapToRange(value, static_cast<T>(0.0), static_cast<T>(2.0 * M_PI));
}

template <typename T>
__host__ __device__ void rotate(const T inAngle, const T &inX, const T &inY,
                                T &outX, T &outY) {
  const T c = cos(inAngle);
  const T s = sin(inAngle);
  outX = (inX * c - inY * s);
  outY = (inX * s + inY * c);
}

template <typename T>
__host__ __device__ void rotate(const T inAngle, T &inOutX, T &inOutY) {
  T x;
  T y;
  rotate(inAngle, inOutX, inOutY, x, y);
  inOutX = x;
  inOutY = y;
}

template <typename T> __host__ __device__ int sign(const T inValue) {
  const bool isNegative = (inValue < static_cast<T>(0));
  const bool isPositive = (static_cast<T>(0) < inValue);
  return (static_cast<int>(isPositive) - static_cast<int>(isNegative));
}
} // namespace utils
#endif // UTILS_H_
