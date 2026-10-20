//===-- utils.h - Collection of useful functions --------------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains useful functions used all over the project.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_UTILS_H_
#define CONGA_UTILS_H_

#include "defines.h"

#include "conga/backend.h"
#define HOST_DEVICE __host__ __device__

#include <cmath>
#include <vector>

namespace conga {
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
HOST_DEVICE T wrapToRange(const T inValue, const T inLow, const T inHigh) {
  // TODO(niclas): Add check that low < high.

  // If the value is within the bounds then do an early exit.
  if ((inLow <= inValue) && (inValue < inHigh)) {
    return inValue;
  }

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
HOST_DEVICE int wrapToRange<int>(const int inValue, const int inLow,
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
template <typename T> HOST_DEVICE T wrapToRangeTwoPi(const T inValue) {
  return wrapToRange(inValue, static_cast<T>(0.0), static_cast<T>(2.0 * M_PI));
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
HOST_DEVICE void rotate(const T inAngle, const T inX, const T inY, T &outX,
                        T &outY) {
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
HOST_DEVICE void rotate(const T inAngle, T &inOutX, T &inOutY) {
  T x;
  T y;
  rotate(inAngle, inOutX, inOutY, x, y);
  inOutX = x;
  inOutY = y;
}

/// \brief Compute the dot product between two 2D vectors.
///
/// \param inVectorA - The first vector.
/// \param inVectorB - The second vector.
///
/// \return - The dot product between the two vectors.
template <typename T>
T dot(const vector2d<T> &inVectorA, const vector2d<T> &inVectorB) {
  return (inVectorA[0] * inVectorB[0] + inVectorA[1] * inVectorB[1]);
}

/// \brief Compute the L1 norm of a 2D vector.
///
/// \param inVector - The vector.
///
/// \return - The L1 norm.
template <typename T> T normL1(const vector2d<T> &inVector) {
  return std::abs(inVector[0]) + std::abs(inVector[1]);
}

/// \brief Compute the L2 norm of a 2D vector.
///
/// \param inVector - The vector.
///
/// \return - The L2 norm.
template <typename T> T normL2(const vector2d<T> &inVector) {
  return std::hypot(inVector[0], inVector[1]);
}

/// \brief Compute c*X+Y where c is a scalar, and X and Y are 2D vectors.
///
/// \param inFactor - The scalar factor.
/// \param inVectorA - The first vector.
/// \param inVectorB - The second vector.
///
/// \return - The result of c*X+Y.
template <typename T>
vector2d<T> multAdd(const T inFactor, const vector2d<T> &inVectorA,
                    const vector2d<T> &inVectorB) {
  return {inFactor * inVectorA[0] + inVectorB[0],
          inFactor * inVectorA[1] + inVectorB[1]};
}

/// \brief Compute the sign of a value
///
/// \param inValue - The value to compute the sign of
///
/// \return - The sign as an integer {-1,0,1}
template <typename T> HOST_DEVICE int sign(const T inValue) {
  const bool isNegative = (inValue < static_cast<T>(0));
  const bool isPositive = (static_cast<T>(0) < inValue);
  return (static_cast<int>(isPositive) - static_cast<int>(isNegative));
}

/// \brief Create a vector with equidistant values
///
/// Works just like linspace in Python. It creates a vector with inNumValues
/// values in the range [inMinVal, inMaxVal]. Or [inMinVal, inMaxVal) if not
/// including the end point.
///
/// \param inMinVal - The minimum value
/// \param inMaxVal - The maximum value
/// \param inNumValues - The number of values
/// \param InIncludeEndPoint - If true inMaxVal is included
///
/// \return - The resulting vector
template <typename T>
std::vector<T> linspace(const T inMinVal, const T inMaxVal,
                        const int inNumValues,
                        const bool inIncludeEndPoint = true) {
  const int N = inIncludeEndPoint ? inNumValues - 1 : inNumValues;
  const T step = (inMaxVal - inMinVal) / static_cast<T>(N);
  std::vector<T> values;
  for (int i = 0; i < inNumValues; ++i) {
    const T value = inMinVal + static_cast<T>(i) * step;
    values.push_back(value);
  }
  return values;
}
} // namespace utils
} // namespace conga

#endif // CONGA_UTILS_H_
