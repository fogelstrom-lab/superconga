#ifndef GRID_FUNCTIONS_H_
#define GRID_FUNCTIONS_H_

// Declaring the fully templated version, so we can specialize below.
template <typename T, typename DEV> struct GridFunctions {};

#include <GridFunctions_CPU.h>
#ifdef __CUDACC__
#include <GridFunctions_GPU.h>
#endif

#endif // GRID_FUNCTIONS_H_