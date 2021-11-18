//===------- GridFunctions.h - GridFunctions wrapper for CPU or GPU. ------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Wrapper for GridFunctions file, choosing either CPU or GPU implementation.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GRID_FUNCTIONS_H_
#define CONGA_GRID_FUNCTIONS_H_

// Declaring the fully templated version, so we can specialize below.
namespace conga {
template <typename T, typename DEV> struct GridFunctions {};
} // namespace conga

#include "GridFunctions_CPU.h"
#ifdef __CUDACC__
#include "GridFunctions_GPU.h"
#endif

#endif // CONGA_GRID_FUNCTIONS_H_
