#ifndef ARMADILLO_WRAPPER_H_
#define ARMADILLO_WRAPPER_H_

// For some reason the armadillo lib has warnings when compiling with
// nvcc. This wrapper turns them off.
#define ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_CLANG
#define ARMA_DONT_PRINT_ERRORS
#pragma GCC system_header

#include <armadillo>

#endif // ARMADILLO_WRAPPER_H_
