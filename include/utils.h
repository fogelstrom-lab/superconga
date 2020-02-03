#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <cuda.h>
#include <math.h>

namespace utils {
template <typename T>
__host__ __device__ T wrapToRange(const T inValue, const T inLow,
                                  const T inHigh) {
  // TODO(niclas): Add check that low < high.
  const T range = inHigh - inLow;
  const T standardizedValue = (inValue - inLow) / range;
  const T wrappedValue =
      (standardizedValue - floor(standardizedValue)) * range + inLow;
  return wrappedValue;
}

template <>
__host__ __device__ int wrapToRange<int>(const int inValue, const int inLow,
                                         const int inHigh) {
  const float value =
      wrapToRange<float>(static_cast<float>(inValue), static_cast<float>(inLow),
                         static_cast<float>(inHigh));
  const int wrappedValue = static_cast<int>(round(value));
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
