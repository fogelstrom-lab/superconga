//===------------ backend.h - Execution backend: CUDA or CPU. -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Single point of contact with the execution backend.
///
/// Under nvcc (CONGA_BACKEND_CUDA) this is a thin wrapper around the CUDA
/// runtime. Under a plain C++ compiler (CONGA_BACKEND_CPU) it emulates the
/// CUDA kernel-launch model on the host: a kernel is an ordinary function, a
/// launch is a loop over blocks that is parallelised with OpenMP, and the
/// built-in index variables blockIdx/threadIdx/blockDim/gridDim are
/// thread-local globals set by the launcher. Thrust is used with its OpenMP
/// device system in that case, so thrust::device_vector lives in host memory.
///
/// Kernels are launched through the macros
///
///   CONGA_LAUNCH((kernel), blocksPerGrid, threadsPerBlock, args...)
///   CONGA_LAUNCH_STREAM((kernel), bpg, tpb, sharedMem, stream, args...)
///
/// The kernel name must be wrapped in parentheses so that template argument
/// lists containing commas survive macro expansion.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_BACKEND_H_
#define CONGA_BACKEND_H_

// Strip one level of parentheses from a macro argument.
#define CONGA_UNPAREN(x) CONGA_UNPAREN_ESC(CONGA_UNPAREN_ISH x)
#define CONGA_UNPAREN_ISH(...) CONGA_UNPAREN_ISH __VA_ARGS__
#define CONGA_UNPAREN_ESC(...) CONGA_UNPAREN_ESC_(__VA_ARGS__)
#define CONGA_UNPAREN_ESC_(...) CONGA_UNPAREN_VAN##__VA_ARGS__
#define CONGA_UNPAREN_VANCONGA_UNPAREN_ISH

#if defined(__CUDACC__)
// ========================================================================== //
//                                  CUDA                                      //
// ========================================================================== //
#define CONGA_BACKEND_CUDA 1
#include "conga/backend.h"
#include <cuda_runtime.h>

#define CONGA_LAUNCH(kernel, bpg, tpb, ...)                                    \
  CONGA_UNPAREN(kernel)<<<(bpg), (tpb)>>>(__VA_ARGS__)
#define CONGA_LAUNCH_STREAM(kernel, bpg, tpb, shmem, stream, ...)              \
  CONGA_UNPAREN(kernel)<<<(bpg), (tpb), (shmem), (stream)>>>(__VA_ARGS__)

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

namespace conga {
namespace backend {
/// Reductions: on the CUDA backend these are plain Thrust calls.
template <typename It, typename T, typename Op>
inline T reduce(It first, It last, T init, Op op) {
  return thrust::reduce(first, last, init, op);
}
template <typename It, typename T> inline T reduce(It first, It last, T init) {
  return thrust::reduce(first, last, init);
}
template <typename It, typename Unary, typename T, typename Binary>
inline T transform_reduce(It first, It last, Unary unary, T init,
                          Binary binary) {
  return thrust::transform_reduce(first, last, unary, init, binary);
}

inline const char *name() { return "cuda"; }
inline int numThreads() { return 1; }
/// Wait for all launched kernels to finish (no-op on CPU).
inline void synchronize() { cudaDeviceSynchronize(); }
} // namespace backend
} // namespace conga

#else
// ========================================================================== //
//                               CPU / OpenMP                                 //
// ========================================================================== //
#define CONGA_BACKEND_CPU 1

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

// Device code may call the math functions unqualified, as CUDA allows. Make
// the overloaded std versions visible at global scope so that e.g. abs(double)
// does not silently resolve to the C integer abs().
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::log;
using std::max;
using std::min;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

#if defined(_OPENMP)
#include <omp.h>
#endif

// CUDA function-space qualifiers become no-ops.
#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

/// Minimal dim3 with the CUDA constructor semantics.
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

/// CUDA runtime stand-ins used by the code base.
typedef int cudaStream_t;
enum cudaError_t { cudaSuccess = 0 };
inline cudaError_t cudaStreamCreate(cudaStream_t *s) {
  *s = 0;
  return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char *cudaGetErrorString(cudaError_t) { return "no error"; }

namespace conga {
namespace backend {

struct Index3 {
  unsigned int x, y, z;
};

/// Built-in kernel index variables, one copy per OpenMP thread.
inline thread_local Index3 cpu_blockIdx{0, 0, 0};
inline thread_local Index3 cpu_threadIdx{0, 0, 0};
inline thread_local dim3 cpu_blockDim{1, 1, 1};
inline thread_local dim3 cpu_gridDim{1, 1, 1};

inline const char *name() {
#if defined(_OPENMP)
  return "cpu-openmp";
#else
  return "cpu-serial";
#endif
}
inline int numThreads() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}
inline void synchronize() {}

/// \brief Emulate a kernel launch on the host.
///
/// Blocks are distributed over OpenMP threads; the threads of one block are
/// executed sequentially by the same OpenMP thread. This is correct for
/// kernels without __syncthreads()/__shared__ memory, which is the case for
/// the whole SuperConga code base.
template <typename Kernel, typename... Args>
inline void launch(const dim3 grid, const dim3 block, Kernel &&kernel,
                   Args &&...args) {
  const std::int64_t numBlocks = static_cast<std::int64_t>(grid.x) *
                                 static_cast<std::int64_t>(grid.y) *
                                 static_cast<std::int64_t>(grid.z);
  const std::int64_t gxy =
      static_cast<std::int64_t>(grid.x) * static_cast<std::int64_t>(grid.y);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::int64_t b = 0; b < numBlocks; ++b) {
    cpu_gridDim = grid;
    cpu_blockDim = block;
    cpu_blockIdx.x = static_cast<unsigned int>(b % grid.x);
    cpu_blockIdx.y = static_cast<unsigned int>((b / grid.x) % grid.y);
    cpu_blockIdx.z = static_cast<unsigned int>(b / gxy);
    for (unsigned int tz = 0; tz < block.z; ++tz) {
      cpu_threadIdx.z = tz;
      for (unsigned int ty = 0; ty < block.y; ++ty) {
        cpu_threadIdx.y = ty;
        for (unsigned int tx = 0; tx < block.x; ++tx) {
          cpu_threadIdx.x = tx;
          kernel(args...);
        }
      }
    }
  }
}

/// \brief Blocked pairwise reduction.
///
/// Thrust's OpenMP reduction sums each thread's chunk sequentially, which in
/// single precision is markedly less accurate than the tree reduction CUDA
/// performs. This routine reduces fixed-size blocks in parallel and then
/// combines the partial results recursively, giving O(log N) rounding depth
/// like the GPU and making f32 results comparable between the backends.
template <typename It, typename Unary, typename T, typename Binary>
inline T transform_reduce(It first, It last, Unary unary, T init,
                          Binary binary) {
  const std::int64_t n = static_cast<std::int64_t>(last - first);
  if (n <= 0) {
    return init;
  }
  constexpr std::int64_t blockSize = 256;
  std::int64_t numBlocks = (n + blockSize - 1) / blockSize;
  std::vector<T> partial(static_cast<std::size_t>(numBlocks));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t b = 0; b < numBlocks; ++b) {
    const std::int64_t begin = b * blockSize;
    const std::int64_t end = (begin + blockSize < n) ? begin + blockSize : n;
    // Pairwise tree within the block as well.
    T buf[blockSize];
    std::int64_t m = end - begin;
    for (std::int64_t i = 0; i < m; ++i) {
      buf[i] = unary(*(first + begin + i));
    }
    while (m > 1) {
      const std::int64_t half = m / 2;
      for (std::int64_t i = 0; i < half; ++i) {
        buf[i] = binary(buf[2 * i], buf[2 * i + 1]);
      }
      if (m % 2 == 1) {
        buf[half] = buf[m - 1];
        m = half + 1;
      } else {
        m = half;
      }
    }
    partial[static_cast<std::size_t>(b)] = buf[0];
  }

  // Combine partial results pairwise until one remains.
  while (numBlocks > 1) {
    const std::int64_t half = (numBlocks + 1) / 2;
    for (std::int64_t i = 0; i < numBlocks / 2; ++i) {
      partial[static_cast<std::size_t>(i)] =
          binary(partial[static_cast<std::size_t>(2 * i)],
                 partial[static_cast<std::size_t>(2 * i + 1)]);
    }
    if (numBlocks % 2 == 1) {
      partial[static_cast<std::size_t>(half - 1)] =
          partial[static_cast<std::size_t>(numBlocks - 1)];
    }
    numBlocks = half;
  }
  return binary(init, partial[0]);
}

struct Identity {
  template <typename V> V operator()(const V &v) const { return v; }
};

template <typename It, typename T, typename Op>
inline T reduce(It first, It last, T init, Op op) {
  return ::conga::backend::transform_reduce(first, last, Identity(), init, op);
}
template <typename It, typename T> inline T reduce(It first, It last, T init) {
  return ::conga::backend::transform_reduce(
      first, last, Identity(), init,
      [](const T &a, const T &b) { return a + b; });
}

} // namespace backend
} // namespace conga

// Expose the CUDA built-in names at global scope so kernel bodies compile
// unchanged.
#define blockIdx (::conga::backend::cpu_blockIdx)
#define threadIdx (::conga::backend::cpu_threadIdx)
#define blockDim (::conga::backend::cpu_blockDim)
#define gridDim (::conga::backend::cpu_gridDim)

#define CONGA_LAUNCH(kernel, bpg, tpb, ...)                                    \
  ::conga::backend::launch((bpg), (tpb),                                       \
                           [&](auto &&...conga_launch_args) {                  \
                             CONGA_UNPAREN(kernel)(conga_launch_args...);      \
                           },                                                  \
                           __VA_ARGS__)
#define CONGA_LAUNCH_STREAM(kernel, bpg, tpb, shmem, stream, ...)              \
  CONGA_LAUNCH(kernel, bpg, tpb, __VA_ARGS__)

#endif // __CUDACC__

#endif // CONGA_BACKEND_H_
