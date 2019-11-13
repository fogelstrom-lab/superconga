#ifndef GRIDFUNCTORS_H
#define GRIDFUNCTORS_H

#include <math.h>

#ifdef __CUDACC__
#define __HD__ __host__ __device__
#else
#define __HD__
#endif

namespace GridFunctors {

template <typename T> struct square_root {
  __HD__
  void operator()(T &val) { val = sqrt(val); }
};

template <typename T> struct absVal {
  __HD__
  void operator()(T &val) { val = abs(val); }
};
} // namespace GridFunctors

#endif // GRIDFUNCTORS_H
