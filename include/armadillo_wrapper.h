//===------- armadillo_wrapper.h - wrapper for armadillo inclusion. -------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Wrapper for including armadillo, such that pragmas and definitions only have
/// to be written once, rather than in every file that includes armadillo.
///
//===----------------------------------------------------------------------===//
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
