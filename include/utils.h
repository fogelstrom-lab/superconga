//===-- utils.h - Collection of useful functions --------------------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains useful functions used all over the project.
///
//===----------------------------------------------------------------------===//
#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <cuda.h>

namespace utils {
/// \brief Wrap value to range
///
/// Wrap inValue periodically to lie in the range [inLow, inHigh).
///
/// \param inValue - The value to wrap
/// \param inLow - The lower bound of range
/// \param inHigh - The upper bound of range.
///
/// \return - The wrapped value
template <typename T>
__host__ __device__ T wrapToRange(const T inValue, const T inLow,
                                  const T inHigh) {
  // TODO(niclas): Add check that low < high.

  // Determine the range.
  const T range = inHigh - inLow;

  // Standardize the value so that [0, 1.0) means that the value is in the
  // desired range.
  const T standardizedValue = (inValue - inLow) / range;

  // Wrap the value. Example:
  // range = 4.0
  // inLow = 2.0
  // standardizedValue = 1.5
  // ==> wrappedValue = (1.5 - 1.0) * 4.0 + 2.0 = 4.0
  const T wrappedValue =
      (standardizedValue - floor(standardizedValue)) * range + inLow;
  return wrappedValue;
}

/// \brief Wrap value to range
///
/// Wrap inValue periodically to lie in the range [inLow, inHigh). A
/// specialization for integer inputs.
///
/// \param inValue - The value to wrap
/// \param inLow - The lower bound of range
/// \param inHigh - The upper bound of range.
///
/// \return - The wrapped value
template <>
__host__ __device__ int wrapToRange<int>(const int inValue, const int inLow,
                                         const int inHigh) {
  const float value =
      wrapToRange<float>(static_cast<float>(inValue), static_cast<float>(inLow),
                         static_cast<float>(inHigh));
  const int wrappedValue = static_cast<int>(round(value));
  return wrappedValue;
}

/// \brief Wrap value to range [0, 2 pi)
///
/// Wrap inValue periodically to lie in the range [0, 2 pi)
///
/// \param inValue - The value to wrap
///
/// \return - The wrapped value
template <typename T> __host__ __device__ T wrapToRangeTwoPi(const T value) {
  return wrapToRange(value, static_cast<T>(0.0), static_cast<T>(2.0 * M_PI));
}

/// \brief Rotate a point counter-clockwise.
///
/// Rotate a point (inX, inY) an angle (inAngle) counter-clockwise.
///
/// \param inAngle - The rotation angle in radians
/// \param inX - The x-component of the point to rotate
/// \param inY - The y-component of the point to rotate
/// \param outX - The x-component of the rotated point
/// \param outY - The y-component of the rotated point
///
/// \return - void
template <typename T>
__host__ __device__ void rotate(const T inAngle, const T inX, const T inY,
                                T &outX, T &outY) {
  const T c = cos(inAngle);
  const T s = sin(inAngle);
  outX = (inX * c - inY * s);
  outY = (inX * s + inY * c);
}

/// \brief Rotate a point counter-clockwise inplace.
///
/// Rotate a point (inOutX, inOutY) an angle (inAngle) counter-clockwise.
///
/// \param inAngle - The rotation angle in radians
/// \param inOutX - The x-component of the point to rotate, it will be
/// overwritten by the x-component of the rotated point
/// \param inOutY - The y-component of the point to rotate, it will be
/// overwritten by the y-component of the rotated point
///
/// \return - void
template <typename T>
__host__ __device__ void rotate(const T inAngle, T &inOutX, T &inOutY) {
  T x;
  T y;
  rotate(inAngle, inOutX, inOutY, x, y);
  inOutX = x;
  inOutY = y;
}

/// \brief Compute the sign of a value
///
/// \param inValue - The value to compute the sign of
///
/// \return - The sign as an integer {-1,0,1}
template <typename T> __host__ __device__ int sign(const T inValue) {
  const bool isNegative = (inValue < static_cast<T>(0));
  const bool isPositive = (static_cast<T>(0) < inValue);
  return (static_cast<int>(isPositive) - static_cast<int>(isNegative));
}
} // namespace utils
#endif // UTILS_H_
