//===---------- GridFunctors.h - Functors used by Grid data class. --------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functors used by Grid data class, in the iterator-algorithm transformation
/// functions: these functors are used in operations between GridData objects
/// (and therefore thrust vectors).
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GRID_FUNCTORS_H_
#define CONGA_GRID_FUNCTORS_H_

#include <cmath>

#include "conga/backend.h"
#define __HD__ __host__ __device__

namespace conga {
namespace GridFunctors {

template <typename T> struct squareRoot {
  __HD__
  void operator()(T &val) { val = std::sqrt(val); }
};

template <typename T> struct absVal {
  __HD__
  void operator()(T &val) { val = std::abs(val); }
};
} // namespace GridFunctors
} // namespace conga

#endif // CONGA_GRID_FUNCTORS_H_
