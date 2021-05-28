//===---------- GridFunctors.h - Functors used by Grid data class. --------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
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
#ifndef GRID_FUNCTORS_H_
#define GRID_FUNCTORS_H_

#include <cmath>

#ifdef __CUDACC__
#define __HD__ __host__ __device__
#else
#define __HD__
#endif

namespace scg {
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
} // namespace scg

#endif // GRID_FUNCTORS_H_
