#ifndef COMPUTE_RESIDUAL_H_
#define COMPUTE_RESIDUAL_H_

#include "Grid.h"
#include <cmath>
#include <limits>

template <typename T>
T computeResidual(const T inOld, const T inNew,
                  const T inEpsilon = std::numeric_limits<T>::epsilon()) {
  return std::abs(inNew - inOld) / std::max(std::abs(inOld), inEpsilon);
}

template <typename T, typename DEV>
T computeResidual(const GridData<T, DEV> &inOld, const GridData<T, DEV> &inNew,
                  const T inEpsilon = std::numeric_limits<T>::epsilon()) {
  if (inOld != inNew) {
    std::cout << "GridData::differenceMagnitude(): Error: Grids have different "
                 "dimensions.\n";
    // TODO(niclas): Raise exception instead!?
    return static_cast<T>(1e6);
  }

  const T diff = (inOld - inNew).L2Norm();
  const T old = inOld.L2Norm();

  // We are interested in the relative error, hence the division.
  return diff / std::max(old, inEpsilon);
}

#endif // COMPUTE_RESIDUAL_H_
