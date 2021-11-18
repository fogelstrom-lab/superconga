//===----- ComputeResidual.h - Functionality for computing residual. ------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functionality for computing the residual between different self-consistent
/// iterations.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_COMPUTE_RESIDUAL_H_
#define CONGA_COMPUTE_RESIDUAL_H_

#include "../grid.h"

#include <cmath>
#include <limits>

namespace conga {
namespace compute_residual_internal {
template <typename T> T getGoodEpsilon(const int inNumElements) {
  const T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()) *
                    static_cast<T>(inNumElements);
  return epsilon;
}
} // namespace compute_residual_internal

template <typename T> T computeResidual(const T inOld, const T inNew) {
  const int numElements = 1;
  const T epsilon = compute_residual_internal::getGoodEpsilon<T>(numElements);
  return std::abs(inNew - inOld) / std::max(std::abs(inOld), epsilon);
}

template <typename T, typename DEV>
T computeResidual(const GridData<T, DEV> &inOld,
                  const GridData<T, DEV> &inNew) {
  if (inOld != inNew) {
#ifndef NDEBUG
    std::cout << "computeResidual: Error: Grids have different "
                 "dimensions.\n";
#endif
    // TODO(niclas): Raise exception instead!?
    return static_cast<T>(1e6);
  }

  const T diff = (inOld - inNew).L2Norm();
  const T old = inOld.L2Norm();

  const int numElements = inOld.numElements();
  const T epsilon = compute_residual_internal::getGoodEpsilon<T>(numElements);

  // We are interested in the relative error, hence the division.
  return diff / std::max(old, epsilon);
}
} // namespace conga

#endif // CONGA_COMPUTE_RESIDUAL_H_
