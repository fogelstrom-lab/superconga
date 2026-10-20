//===------------- grid.h - GridData wrapper for CPU or GPU. -------------===//
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
#ifndef CONGA_GRID_H_
#define CONGA_GRID_H_

#include "conga/backend.h"
#define _DEV_ __host__ __device__

#include "grid/GridData.h"

#include "conga/backend.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace conga {
// The "GPU" containers are thrust::device_vector on both backends; on the CPU
// backend thrust uses its OpenMP device system, so these live in host memory.
template <typename T> using VectorGPU = thrust::device_vector<T>;
template <typename T> using GridGPU = GridData<T, thrust::device_vector<T>>;

template <typename T> using VectorCPU = thrust::host_vector<T>;
template <typename T> using GridCPU = GridData<T, thrust::host_vector<T>>;
} // namespace conga

#endif // CONGA_GRID_H_
