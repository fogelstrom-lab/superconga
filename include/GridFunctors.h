#ifndef GRID_FUNCTORS_H_
#define GRID_FUNCTORS_H_

#include <cmath>

#ifdef __CUDACC__
#define __HD__ __host__ __device__
#else
#define __HD__
#endif

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

#endif // GRID_FUNCTORS_H_
