//===------------- Grid.h - GridData wrapper for CPU or GPU. -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Wrapper for GridData class type, choosing either host or device for CPU or
/// GPU. Also has convenient shortened definitions for data structures.
///
//===----------------------------------------------------------------------===//
#ifndef GRID_H_
#define GRID_H_

#ifdef __CUDACC__
#define _DEV_ __host__ __device__
#else
#define _DEV_
#endif

#include "GridData.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace conga {
#ifdef __CUDACC__
template <typename T> using VectorGPU = thrust::device_vector<T>;
template <typename T> using GridGPU = GridData<T, thrust::device_vector<T>>;
#else
template <typename T> using VectorGPU = thrust::host_vector<T>;
template <typename T> using GridGPU = GridData<T, thrust::host_vector<T>>;
#endif

template <typename T> using VectorCPU = thrust::host_vector<T>;
template <typename T> using GridCPU = GridData<T, thrust::host_vector<T>>;
} // namespace conga

#endif // GRID_H_
