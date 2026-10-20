//===-------------- GridFunctions_GPU.h - GPU grid functions. -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU-specific implementation of functions to operate on Grid functions, i.e.
/// thrust device vectors.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GRID_FUNCTIONS_GPU_H_
#define CONGA_GRID_FUNCTIONS_GPU_H_

#include "../Type.h"
#include "../defines.h"
#include "../utils.h"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cmath>
#include <utility>

namespace conga {
///////////////////////////////////////////
// Code path for GPU (device) specific code
///////////////////////////////////////////
/*
// Declaring the fully templated version, so we can specialize below.
template <typename T, typename DEV>
struct GridFunctions
{};
*/
// Forward declaration.
template <typename, typename> class GridData;

template <typename T> struct GridFunctions<T, thrust::device_vector<T>> {
  GridData<T, thrust::device_vector<T>> *grid;

  GridFunctions(GridData<T, thrust::device_vector<T>> *ptr) : grid(ptr) {}

  template <typename T2>
  void typeConvert(GridData<T2, thrust::device_vector<T2>> &other) const;

  // IN place transponate
  GridData<T, thrust::device_vector<T>> &transpose(const int field);

  // Compute the complex magnitude squared, of specified field.
  void complexMagnitudeSq(GridData<T, thrust::device_vector<T>> *result) const;

  // Compute complex phase of this (complex) grid.
  void complexPhase(GridData<T, thrust::device_vector<T>> *result) const;

  // Compute gradient of complex phase of this grid.
  void complexPhaseGradient(GridData<T, thrust::device_vector<T>> *result,
                            const T h) const;

  /// \brief Compute the moment: 0.5 * sum(cross(r, grid)). Note that the only
  /// the z-component is returned.
  ///
  /// \param h The side length of a lattice site.
  ///
  /// \return The result.
  T moment(const T h) const;

  // Compute the magnitude of the real part of this vector field.
  void fieldMagnitude(GridData<T, thrust::device_vector<T>> *result) const;

  // Compute divergence
  void div(GridData<T, thrust::device_vector<T>> *result, const T h) const;

  // Compute curl
  void curl(GridData<T, thrust::device_vector<T>> *result, const T h) const;

  // Compute Laplacian
  void del2(GridData<T, thrust::device_vector<T>> *result, const T h) const;

  // Sum all fields.
  void sumFieldComponents(GridData<T, thrust::device_vector<T>> *result) const;

  // Insert other grid into this. <x,y>_offset are offset in discrete elements
  // from center. x_offset = y_offset = 0 results in a centered grid.
  void insertGrid(const GridData<T, thrust::device_vector<T>> *other,
                  const int x_offset, const int y_offset);

  // Insert other grid into this, with applied rotation (to other grid).
  void insertRotatedGrid(const GridData<T, thrust::device_vector<T>> *other,
                         const T phi, const int x_offset, const int y_offset,
                         Type::type part);

  // Add other grid to this, with applied rotation (to other grid).
  void
  insertRotatedGridAdditive(const GridData<T, thrust::device_vector<T>> *other,
                            const T phi, int field, const int x_offset,
                            const int y_offset, Type::type part);

  // Change resolution of grid, with linear interpolation of existing data.
  void resample(const int inDimX, const int inDimY);

  // Find min and max value of range
  void minmax(const GridData<T, thrust::device_vector<T>> &grid, T &minv,
              T &maxv, int field, Type::type part) const;

  /// \brief Get the CUDA kernel dimensions spanning the spatial elements.
  ///
  /// \return The pair {number of blocks per grid, number of threads per block}.
  std::pair<dim3, dim3> getKernelDimensions() const;

  /// \brief Get the CUDA kernel dimensions spanning the spatial elements and
  /// all fields.
  ///
  /// \return The pair {number of blocks per grid, number of threads per block}.
  std::pair<dim3, dim3> getKernelDimensionsFields() const;

  /// \brief Get the CUDA kernel dimensions from requested number of threads.
  ///
  /// \param inDimX The number of requested threads along the x-axis.
  /// \param inDimY The number of requested threads along the y-axis.
  /// \param inDimZ The number of requested threads along the z-axis (fields,
  /// real/imag etc).
  ///
  /// \return The pair {number of blocks per grid, number of threads per block}.
  static std::pair<dim3, dim3>
  getKernelDimensions(const int inDimX, const int inDimY, const int inDimZ = 1);

  // Check for latest error
  static int checkCudaError(std::string msg = "");
};

namespace internal {
template <typename T, typename T2>
__global__ void typeConvert_kernel(const int inDimX, const int inDimY,
                                   const int inDimZ, const T *const inSrc,
                                   T2 *const outDst) {
  // Thread indices.
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  if (tx < inDimX && ty < inDimY && tz < inDimZ) {
    const int idx = (tz * inDimY + ty) * inDimX + tx;
    outDst[idx] = static_cast<T2>(inSrc[idx]);
  }
}

template <typename T> __global__ void transpose_kernel(T *data, const int Nx) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx_ij = ty * Nx + tx;
  const int idx_ji = tx * Nx + ty;

  if (tx < Nx && tx > ty) {
    const T tmp = data[idx_ji];
    data[idx_ji] = data[idx_ij];
    data[idx_ij] = tmp;
  }
}

template <typename T>
__global__ void
complexMagnitudeSq_kernel(const int inDimX, const int inDimY, const int inDimZ,
                          const T *const inSrcReal, const T *const inSrcImag,
                          T *const outDst) {
  // Thread indices.
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  if (tx < inDimX && ty < inDimY && tz < inDimZ) {
    // Indices.
    const int annoyingComplexFactor = 2;
    const int srcIdx =
        tz * annoyingComplexFactor * inDimX * inDimY + ty * inDimX + tx;
    const int dstIdx = tz * inDimX * inDimY + ty * inDimX + tx;

    const T real = inSrcReal[srcIdx];
    const T imag = inSrcImag[srcIdx];
    outDst[dstIdx] = real * real + imag * imag;
  }
}

template <typename T>
__global__ void complexPhase_kernel(const int inDimX, const int inDimY,
                                    const int inDimZ, const T *const inSrcReal,
                                    const T *const inSrcImag, T *const outDst) {
  // Thread indices.
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  if (tx < inDimX && ty < inDimY && tz < inDimZ) {
    // Indices.
    const int annoyingComplexFactor = 2;
    const int srcIdx =
        tz * annoyingComplexFactor * inDimX * inDimY + ty * inDimX + tx;
    const int dstIdx = tz * inDimX * inDimY + ty * inDimX + tx;

    const T real = inSrcReal[srcIdx];
    const T imag = inSrcImag[srcIdx];
    outDst[dstIdx] = atan2(imag, real);
  }
}

template <typename T>
__global__ void complexPhaseGradient_kernel(const T *data_src_re,
                                            const T *data_src_im, T *data_dst_x,
                                            T *data_dst_y, const T h,
                                            const int Nx, const int Ny) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  T f0, f1, f2, g0, g1, g2, denom, dPhase_dx, dPhase_dy;

  // Grid index
  const int idx = ty * Nx + tx;

  const int idx_x0 = ty * Nx + (tx - 1);
  const int idx_x2 = ty * Nx + (tx + 1);

  const int idx_y0 = (ty - 1) * Nx + tx;
  const int idx_y2 = (ty + 1) * Nx + tx;

  // Set gradient to zero at borders for now. Implement other gradient methods
  // later!
  dPhase_dx = (T)0.0;
  dPhase_dy = (T)0.0;

  if (tx >= 0 && tx <= Nx - 1 && ty >= 0 && ty <= Ny - 1) {
    if (tx > 0 && tx < Nx - 1 && ty > 0 && ty < Ny - 1) {
      // Read elements at current grid point
      f1 = data_src_im[idx];
      g1 = data_src_re[idx];

      // Some precomputed quantities
      denom = (T)2.0 * h * (f1 * f1 + g1 * g1);

      if (denom > (T)0.00001) {
        denom = (T)1.0 / denom;

        // Gradient in x
        f0 = data_src_im[idx_x0];
        g0 = data_src_re[idx_x0];

        f2 = data_src_im[idx_x2];
        g2 = data_src_re[idx_x2];

        dPhase_dx = ((f2 - f0) * g1 - f1 * (g2 - g0)) * denom;

        // Gradient in y
        f0 = data_src_im[idx_y0];
        f2 = data_src_im[idx_y2];

        g0 = data_src_re[idx_y0];
        g2 = data_src_re[idx_y2];

        dPhase_dy = ((f2 - f0) * g1 - f1 * (g2 - g0)) * denom;
      }
    }

    data_dst_x[idx] = dPhase_dx;
    data_dst_y[idx] = dPhase_dy;
  }
}

template <typename T>
__global__ void fieldMagnitude_kernel(const T *data, T *data_dst, const int Nx,
                                      const int Ny, const int numFields,
                                      const int isComplex) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    // Array index to grid_sum
    const int idx = ty * Nx + tx;

    // Loop over components
    int i, idx_c;

    T val, sum = (T)0.0;

    for (i = 0; i < numFields; ++i) {
      idx_c = i * Nx * Ny * (1 + isComplex) + idx;
      val = data[idx_c];
      sum += val * val;
    }

    data_dst[idx] = sqrt(sum);
  }
}

template <typename T>
__global__ void div_kernel(const int inDimX, const int inDimY,
                           const T inInvDenom, const T *const inSrcX,
                           const T *const inSrcY, T *const outDst) {
  // Spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < inDimX && y < inDimY) {
    // Next and previous x-coordinate making sure we're inside bounds.
    const int xNext = min(x + 1, inDimX - 1);
    const int xPrev = max(x - 1, 0);

    // Next and previous y-coordinate making sure we're inside bounds.
    const int yNext = min(y + 1, inDimY - 1);
    const int yPrev = max(y - 1, 0);

    // The derivatives.
    const T dAxdx = (inSrcX[y * inDimX + xNext] - inSrcX[y * inDimX + xPrev]) /
                    static_cast<T>(xNext - xPrev);
    const T dAydy = (inSrcY[yNext * inDimX + x] - inSrcY[yPrev * inDimX + x]) /
                    static_cast<T>(yNext - yPrev);

    // The actual divergence.
    const int idx = y * inDimX + x;
    outDst[idx] = (dAxdx + dAydy) * inInvDenom;
  }
}

template <typename T>
__global__ void curl_kernel(const int inDimX, const int inDimY,
                            const T inInvDenom, const T *const inSrcX,
                            const T *const inSrcY, T *const outDst) {
  // Spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < inDimX && y < inDimY) {
    // Next and previous x-coordinate making sure we're inside bounds.
    const int xNext = min(x + 1, inDimX - 1);
    const int xPrev = max(x - 1, 0);

    // Next and previous y-coordinate making sure we're inside bounds.
    const int yNext = min(y + 1, inDimY - 1);
    const int yPrev = max(y - 1, 0);

    // The derivatives.
    const T dAydx = (inSrcY[y * inDimX + xNext] - inSrcY[y * inDimX + xPrev]) /
                    static_cast<T>(xNext - xPrev);
    const T dAxdy = (inSrcX[yNext * inDimX + x] - inSrcX[yPrev * inDimX + x]) /
                    static_cast<T>(yNext - yPrev);

    // The actual curl.
    const int idx = y * inDimX + x;
    outDst[idx] = (dAydx - dAxdy) * inInvDenom;
  }
}

template <typename T>
__global__ void del2_kernel(const int inDimX, const int inDimY,
                            const int inDimZ, const T inInvDenom,
                            const T *const inSrc, T *const outDst) {
  // Spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Internal coordinate.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Next and previous x-coordinate making sure we're inside bounds.
    const int xNext = min(x + 1, inDimX - 1);
    const int xPrev = max(x - 1, 0);

    // Next and previous y-coordinate making sure we're inside bounds.
    const int yNext = min(y + 1, inDimY - 1);
    const int yPrev = max(y - 1, 0);

    // Note that we extrapolate outside of the domain using the (implicit)
    // nearest-neighbour extrapolation.
    const int idx = y * inDimX + x;
    outDst[idx] = (outDst[y * inDimX + xNext] + outDst[y * inDimX + xPrev] +
                   outDst[yNext * inDimX + x] + outDst[yPrev * inDimX + x] -
                   static_cast<T>(4) * outDst[idx]) *
                  inInvDenom;
  }
}

template <typename T>
__global__ void sumFieldComponents_kernel(const T *data, T *data_dst,
                                          const int Nx, const int Ny,
                                          const int numFields,
                                          const int isComplex) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    // Array index to grid_sum
    const int idx_result = ty * Nx + tx;

    // Loop over components
    int i, idx_source;

    T sum = (T)0.0;

    for (i = 0; i < numFields; ++i) {
      idx_source = i * Nx * Ny * (1 + isComplex) + idx_result;
      sum += data[idx_source];
    }

    data_dst[idx_result] = sum;
  }
}

template <typename T>
__global__ void
insertGrid_kernel(T *data_this, const T *data_other, const int Nx_this,
                  const int Ny_this, const int Nx_other, const int Ny_other,
                  const int x_start_this, const int y_start_this,
                  const int x_start_other, const int y_start_other) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int tx_this = tx + x_start_this;
  const int ty_this = ty + y_start_this;

  const int tx_other = tx + x_start_other;
  const int ty_other = ty + y_start_other;

  if (tx_this < Nx_this && ty_this < Ny_this && tx_other < Nx_other &&
      ty_other < Ny_other) {
    const int idx_this = ty_this * Nx_this + tx_this;
    const int idx_other = ty_other * Nx_other + tx_other;

    data_this[idx_this] = data_other[idx_other];
  }
}

template <typename T>
__global__ void insertRotatedGrid_kernel(T *data, const T *data_other,
                                         const int Nx, const int Ny,
                                         const int Nx_other, const int Ny_other,
                                         const float phi, const int x_offset,
                                         const int y_offset) {
  // --> offset not implemented yet! <-- //

  // Coordinate transforms from current grid to lookup grid
  const T x_transform = (Nx_other - Nx) / (T)2.0;
  const T y_transform = (Ny_other - Ny) / (T)2.0;

  // Shifts lookup grid center to origin
  const T grid_origin_shift_x = (Nx_other - 1) / (T)2.0;
  const T grid_origin_shift_y = (Ny_other - 1) / (T)2.0;

  // Coordinates to current grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    // Array index to current grid
    const int idx_grid_out = y * Nx + x;

    // Coordinates to lookup grid
    const T X = x + x_transform - grid_origin_shift_x;
    const T Y = y + y_transform - grid_origin_shift_y;

    // Rotate centered coordinates, and put grid corner at origin
    const T Xr = (T)X * cos(-phi) - (T)Y * sin(-phi) + grid_origin_shift_x;
    const T Yr = (T)X * sin(-phi) + (T)Y * cos(-phi) + grid_origin_shift_y;

    // Bilinear interpolation
    const int X_int = Xr;
    const int Y_int = Yr;

    if ((X_int >= 0) && (X_int + 1 < Nx_other) && (Y_int >= 0) &&
        (Y_int + 1 < Ny_other)) {
      const T X_frac = Xr - (T)X_int;
      const T Y_frac = Yr - (T)Y_int;

      T X1_interp, X2_interp;

      X1_interp = ((T)1.0 - X_frac) * data_other[Y_int * Nx_other + X_int] +
                  X_frac * data_other[Y_int * Nx_other + (X_int + 1)];

      X2_interp =
          ((T)1.0 - X_frac) * data_other[(Y_int + 1) * Nx_other + X_int] +
          X_frac * data_other[(Y_int + 1) * Nx_other + (X_int + 1)];

      data[idx_grid_out] = ((T)1.0 - Y_frac) * X1_interp + Y_frac * X2_interp;
    }
  }
}

template <typename T>
__global__ void
insertRotatedGridAdditive_kernel(T *data, const T *data_other, const int Nx,
                                 const int Ny, const int Nx_other,
                                 const int Ny_other, const float phi,
                                 const int x_offset, const int y_offset) {
  // --> offset not implemented yet! <-- //

  // Coordinate transforms from current grid to lookup grid
  const T x_transform = (Nx_other - Nx) / (T)2.0;
  const T y_transform = (Ny_other - Ny) / (T)2.0;

  // Shifts lookup grid center to origin
  const T grid_origin_shift_x = (Nx_other - 1) / (T)2.0;
  const T grid_origin_shift_y = (Ny_other - 1) / (T)2.0;

  // Coordinates to current grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < Nx && y < Ny) {
    // Array index to current grid
    const int idx_grid_out = y * Nx + x;

    // Coordinates to lookup grid
    const T X = x + x_transform - grid_origin_shift_x;
    const T Y = y + y_transform - grid_origin_shift_y;

    // Rotate centered coordinates, and put grid corner at origin
    const T Xr = (T)X * cos(-phi) - (T)Y * sin(-phi) + grid_origin_shift_x;
    const T Yr = (T)X * sin(-phi) + (T)Y * cos(-phi) + grid_origin_shift_y;

    // Bilinear interpolation
    const int X_int = Xr;
    const int Y_int = Yr;

    if ((X_int >= 0) && (X_int + 1 < Nx_other) && (Y_int >= 0) &&
        (Y_int + 1 < Ny_other)) {
      const T X_frac = Xr - (T)X_int;
      const T Y_frac = Yr - (T)Y_int;

      T X1_interp, X2_interp;

      X1_interp = ((T)1.0 - X_frac) * data_other[Y_int * Nx_other + X_int] +
                  X_frac * data_other[Y_int * Nx_other + (X_int + 1)];

      X2_interp =
          ((T)1.0 - X_frac) * data_other[(Y_int + 1) * Nx_other + X_int] +
          X_frac * data_other[(Y_int + 1) * Nx_other + (X_int + 1)];

      data[idx_grid_out] += ((T)1.0 - Y_frac) * X1_interp + Y_frac * X2_interp;
    }
  }
}

template <typename T>
__global__ void resample_kernel(const int inDimX, const int inDimY,
                                const int inDimZ, const int inOldDimX,
                                const int inOldDimY, const int inMargin,
                                const T *const inSrc, T *const outDst) {
  // Spatial coordinates of new grid.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Real/imag or xy-components.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Scaling factor between grids.
    // The safety margin should not be included in the scales.
    const int twoMargin = 2 * inMargin;
    const T xScale = static_cast<T>(inOldDimX - 1 - twoMargin) /
                     static_cast<T>(inDimX - 1 - twoMargin);
    const T yScale = static_cast<T>(inOldDimY - 1 - twoMargin) /
                     static_cast<T>(inDimY - 1 - twoMargin);

    // Coordinates in old grid.
    const T xOld = xScale * static_cast<T>(x - inMargin) + inMargin;
    const T yOld = yScale * static_cast<T>(y - inMargin) + inMargin;

    // Floor coordinates.
    const int xFloor = static_cast<int>(xOld);
    const int yFloor = static_cast<int>(yOld);

    // Fractions.
    const T xFrac = xOld - static_cast<T>(xFloor);
    const T yFrac = yOld - static_cast<T>(yFloor);

    // What to add to the index to get the next pixel. If there is no next pixel
    // the value is zero.
    const int xNext = x < inDimX - 1 ? 1 : 0;
    const int yNext = y < inDimY - 1 ? inOldDimX : 0;

    // Indices to interpolate with.
    const int idx00 = (z * inOldDimY + yFloor) * inOldDimX + xFloor;
    const int idx01 = idx00 + xNext;
    const int idx10 = idx00 + yNext;
    const int idx11 = idx10 + xNext;

    const T x1_interp =
        (static_cast<T>(1) - xFrac) * inSrc[idx00] + xFrac * inSrc[idx01];

    const T x2_interp =
        (static_cast<T>(1) - xFrac) * inSrc[idx10] + xFrac * inSrc[idx11];

    const int idx = (z * inDimY + y) * inDimX + x;
    outDst[idx] = (static_cast<T>(1) - yFrac) * x1_interp + yFrac * x2_interp;
  }
}

template <typename T>
__global__ void momentKernel(const int inDimX, const int inDimY,
                             const T *const inSrcX, const T *const inSrcY,
                             T *const outDst) {
  const int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

  if ((xIdx < inDimX) && (yIdx < inDimY)) {
    const int idx = yIdx * inDimY + xIdx;
    const T x =
        static_cast<T>(xIdx) - static_cast<T>(0.5) * static_cast<T>(inDimX - 1);
    const T y =
        static_cast<T>(yIdx) - static_cast<T>(0.5) * static_cast<T>(inDimY - 1);
    outDst[idx] = x * inSrcY[idx] - y * inSrcX[idx];
  }
}
} // namespace internal

template <typename T>
template <typename T2>
void GridFunctions<T, thrust::device_vector<T>>::typeConvert(
    GridData<T2, thrust::device_vector<T2>> &other) const {
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();
  const int numFields = grid->numFields();
  const bool isComplex = grid->isComplex();
  const int dimZ = numFields * (isComplex ? 2 : 1);

  const auto [blocksPerGrid, threadsPerBlock] =
      getKernelDimensions(dimX, dimY, dimZ);

  CONGA_LAUNCH((internal::typeConvert_kernel<T, T2>), blocksPerGrid,
               threadsPerBlock, dimX, dimY, dimZ, grid->getDataPointer(),
               other.getDataPointer());
}

template <typename T>
GridData<T, thrust::device_vector<T>> &
GridFunctions<T, thrust::device_vector<T>>::transpose(const int field) {
  if (grid->_dimX != grid->_dimY) {
#ifndef NDEBUG
    std::cout << "GridFunctions<T,thrust::device_vector<T> >::transpose(): "
                 "Error, dimensions not equal\n";
#endif
    return *grid;
  }

  const auto [bpg, tpb] = getKernelDimensions();

  int f_start = 0, f_end = grid->_numFields - 1;

  if (field != -1) {
    f_start = field;
    f_end = field;
  }

  for (int i = f_start; i <= f_end; ++i) {
    CONGA_LAUNCH((internal::transpose_kernel), bpg, tpb,
                 grid->getDataPointer(i, Type::real), grid->_dimX);
    if (grid->isComplex())
      CONGA_LAUNCH((internal::transpose_kernel), bpg, tpb,
                   grid->getDataPointer(i, Type::imag), grid->_dimX);
  }

  return *grid;
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexMagnitudeSq(
    GridData<T, thrust::device_vector<T>> *result) const {
  const auto [blocksPerGrid, threadsPerBlock] = getKernelDimensionsFields();

  const int dimX = grid->dimX();
  const int dimY = grid->dimY();
  const int numFields = grid->numFields();
  const bool isComplex = grid->isComplex();

  // Sanity check.
  assert(isComplex);

  CONGA_LAUNCH((internal::complexMagnitudeSq_kernel), blocksPerGrid,
               threadsPerBlock, dimX, dimY, numFields,
               grid->getDataPointer(0, Type::real),
               grid->getDataPointer(0, Type::imag),
               result->getDataPointer(0, Type::real));

  checkCudaError("GridData::complexMagnitudeSq()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexPhase(
    GridData<T, thrust::device_vector<T>> *result) const {
  const auto [blocksPerGrid, threadsPerBlock] = getKernelDimensionsFields();

  const int dimX = grid->dimX();
  const int dimY = grid->dimY();
  const int numFields = grid->numFields();
  const bool isComplex = grid->isComplex();

  // Sanity check.
  assert(isComplex);

  CONGA_LAUNCH((internal::complexPhase_kernel), blocksPerGrid, threadsPerBlock,
               dimX, dimY, numFields, grid->getDataPointer(0, Type::real),
               grid->getDataPointer(0, Type::imag),
               result->getDataPointer(0, Type::real));

  checkCudaError("GridData::complexPhase()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexPhaseGradient(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  const auto [bpg, tpb] = getKernelDimensions();

  CONGA_LAUNCH(
      (internal::complexPhaseGradient_kernel), bpg, tpb,
      grid->getDataPointer(0, Type::real), grid->getDataPointer(0, Type::imag),
      result->getDataPointer(0, Type::real),
      result->getDataPointer(1, Type::real), h, grid->_dimX, grid->_dimY);

  checkCudaError("GridData<T>::complexPhaseGradient()");
}

template <typename T>
T GridFunctions<T, thrust::device_vector<T>>::moment(const T h) const {
  // Sanity check.
  assert(grid->numFields() == 2);
  assert(grid->representation() == Type::real);

  // Kernel parameters.
  const auto [bpg, tpb] = getKernelDimensions();

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();

  thrust::device_vector<T> dst(dimX * dimY);
  CONGA_LAUNCH((internal::momentKernel), bpg, tpb, dimX, dimY,
               grid->getDataPointer(0, Type::real),
               grid->getDataPointer(1, Type::real),
               thrust::raw_pointer_cast(dst.data()));

  checkCudaError("GridData<T>::moment()");

  // Two factors of h comes from the integral, the other from the position
  // vector.
  const T factor = static_cast<T>(0.5) * h * h * h;
  const T init = static_cast<T>(0);
  const T result =
      factor * ::conga::backend::reduce(dst.begin(), dst.end(), init);

  return result;
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::fieldMagnitude(
    GridData<T, thrust::device_vector<T>> *result) const {
  const auto [bpg, tpb] = getKernelDimensions();

  CONGA_LAUNCH((internal::fieldMagnitude_kernel), bpg, tpb,
               grid->getDataPointer(0, Type::real),
               result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY,
               grid->_numFields, int(grid->isComplex()));

  checkCudaError("GridData<T>::fieldMagnitude()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::div(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  // Sanity checks.
  assert(grid->dimX() == result->dimX());
  assert(grid->dimY() == result->dimY());
  assert(grid->numFields() == 2);
  assert(grid->representation() == Type::real);
  assert(result->numFields() == 1);
  assert(result->representation() == Type::real);
  assert(h > static_cast<T>(0));

  // Do division outside of the kernel.
  const T invLatticeSiteSideLength = static_cast<T>(1) / h;

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();

  // Kernel parameters.
  const auto [bpg, tpb] = getKernelDimensions();

  // Launch kernel.
  CONGA_LAUNCH((internal::div_kernel), bpg, tpb, dimX, dimY,
               invLatticeSiteSideLength, grid->getDataPointer(0, Type::real),
               grid->getDataPointer(1, Type::real),
               result->getDataPointer(0, Type::real));

  checkCudaError("GridData<T>::div()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::curl(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  // Sanity checks.
  assert(grid->dimX() == result->dimX());
  assert(grid->dimY() == result->dimY());
  assert(grid->numFields() == 2);
  assert(grid->representation() == Type::real);
  assert(result->numFields() == 1);
  assert(result->representation() == Type::real);
  assert(h > static_cast<T>(0));

  // Do division outside of the kernel.
  const T invLatticeSiteSideLength = static_cast<T>(1) / h;

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();

  // Kernel parameters.
  const auto [bpg, tpb] = getKernelDimensions();

  // Launch kernel.
  CONGA_LAUNCH((internal::curl_kernel), bpg, tpb, dimX, dimY,
               invLatticeSiteSideLength, grid->getDataPointer(0, Type::real),
               grid->getDataPointer(1, Type::real),
               result->getDataPointer(0, Type::real));

  checkCudaError("GridData<T>::curl()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::del2(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  // Sanity checks.
  assert(grid->dimX() == result->dimX());
  assert(grid->dimY() == result->dimY());
  assert(grid->isComplex() == result->isComplex());
  assert(grid->representation() == result->representation());
  assert(h > static_cast<T>(0));

  // Do division outside of the kernel.
  const T invLatticeSiteArea = static_cast<T>(1) / (h * h);

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();
  const int numFields = grid->numFields();
  const bool isComplex = grid->isComplex();
  const int dimZ = numFields * (isComplex ? 2 : 1);

  // Kernel parameters.
  const auto [bpg, tpb] = getKernelDimensions(dimX, dimY, dimZ);

  // Launch kernel.
  CONGA_LAUNCH((internal::del2_kernel), bpg, tpb, dimX, dimY, dimZ,
               invLatticeSiteArea, grid->getDataPointer(),
               result->getDataPointer());

  checkCudaError("GridData<T>::del2()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::sumFieldComponents(
    GridData<T, thrust::device_vector<T>> *result) const {
  const auto [bpg, tpb] = getKernelDimensions();

  CONGA_LAUNCH((internal::sumFieldComponents_kernel), bpg, tpb,
               grid->getDataPointer(0, Type::real),
               result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY,
               grid->_numFields, int(grid->isComplex()));

  if (grid->isComplex())
    CONGA_LAUNCH((internal::sumFieldComponents_kernel), bpg, tpb,
                 grid->getDataPointer(0, Type::imag),
                 result->getDataPointer(0, Type::imag), grid->_dimX,
                 grid->_dimY, grid->_numFields, int(grid->isComplex()));

  checkCudaError("GridData<T>::sumFieldComponents()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::insertGrid(
    const GridData<T, thrust::device_vector<T>> *other, const int x_offset,
    const int y_offset) {
  // insert 'other' grid into 'this'
  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();
  const int Nf_other = other->numFields();

  int x_start_this = 0, y_start_this = 0, x_start_other = 0, y_start_other = 0,
      tpb, bpg;

  int x_diff = (grid->_dimX - Nx_other) / 2;
  int y_diff = (grid->_dimY - Ny_other) / 2;

  if (x_diff < 0) // Other grid bigger than this
  {
    x_start_other = -x_diff + x_offset;
    tpb = grid->_dimX;
  } else // Other grid smaller (or equal) than this
  {
    x_start_this = x_diff + x_offset;
    tpb = Nx_other;
  }

  if (y_diff < 0) {
    y_start_other = -y_diff + y_offset;
    bpg = grid->_dimY;
  } else {
    y_start_this = y_diff + y_offset;
    bpg = Ny_other;
  }

  const auto [blocksPerGrid, threadsPerBlock] = getKernelDimensions(tpb, bpg);

  for (int c = 0; c < grid->_numFields; ++c) {
    int c_other = c >= Nf_other ? Nf_other - 1 : c;

    CONGA_LAUNCH((internal::insertGrid_kernel), blocksPerGrid, threadsPerBlock,
                 grid->getDataPointer(c, Type::real),
                 other->getDataPointer(c_other, Type::real), grid->_dimX,
                 grid->_dimY, Nx_other, Ny_other, x_start_this, y_start_this,
                 x_start_other, y_start_other);

    if (grid->isComplex()) {
      CONGA_LAUNCH((internal::insertGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::imag),
                   other->getDataPointer(c_other, Type::imag), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, x_start_this, y_start_this,
                   x_start_other, y_start_other);
    }
  }

  checkCudaError("GridData<T>::insertGrid()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::insertRotatedGrid(
    const GridData<T, thrust::device_vector<T>> *other, const T phi,
    const int x_offset, const int y_offset, Type::type part) {
  checkCudaError("GridData<T>::insertRotatedGrid() b4 ");

  // Grid dimensions of grid to rotate and read from (lookup grid)
  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();

  const bool complex_this = grid->isComplex();
  const bool complex_other = other->isComplex();

  // Kernel launch
  const auto [blocksPerGrid, threadsPerBlock] = getKernelDimensions();

  for (int c = 0; c < grid->_numFields; ++c) {
    int field_other = c;

    if (other->numFields() <= c)
      field_other = 0;

    if (complex_this && complex_other) {
      CONGA_LAUNCH((internal::insertRotatedGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);

      CONGA_LAUNCH((internal::insertRotatedGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::imag),
                   other->getDataPointer(field_other, Type::imag), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this && !complex_other) {
      CONGA_LAUNCH((internal::insertRotatedGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this) {
      CONGA_LAUNCH((internal::insertRotatedGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, part), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else {
      CONGA_LAUNCH((internal::insertRotatedGrid_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, part),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    }
  }

  checkCudaError("GridData<T>::insertRotatedGrid()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::insertRotatedGridAdditive(
    const GridData<T, thrust::device_vector<T>> *other, const T phi, int field,
    const int x_offset, const int y_offset, Type::type part) {
  int fieldStart, fieldEnd;

  if (field == -1) {
    fieldStart = 0;
    fieldEnd = grid->_numFields;
  } else if (field < grid->_numFields) {
    fieldStart = field;
    fieldEnd = field + 1;
  } else {
#ifndef NDEBUG
    std::cout << "GridData() : Field out of bounds, operation terminated.\n";
#endif
    return;
  }

  // Grid dimensions of grid to rotate and read from (lookup grid)
  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();

  const bool complex_this = grid->isComplex();
  const bool complex_other = other->isComplex();

  // Kernel launch
  const auto [blocksPerGrid, threadsPerBlock] = getKernelDimensions();

  for (int c = fieldStart; c < fieldEnd; ++c) {
    int field_other = c;

    if (other->numFields() <= c)
      field_other = 0; // Maybe put this to other->numFields()-1 ??

    if (complex_this && complex_other) {
      CONGA_LAUNCH((internal::insertRotatedGridAdditive_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);

      CONGA_LAUNCH((internal::insertRotatedGridAdditive_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::imag),
                   other->getDataPointer(field_other, Type::imag), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this && !complex_other) {
      CONGA_LAUNCH((internal::insertRotatedGridAdditive_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this) {
      CONGA_LAUNCH((internal::insertRotatedGridAdditive_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, Type::real),
                   other->getDataPointer(field_other, part), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else {
      CONGA_LAUNCH((internal::insertRotatedGridAdditive_kernel), blocksPerGrid,
                   threadsPerBlock, grid->getDataPointer(c, part),
                   other->getDataPointer(field_other, Type::real), grid->_dimX,
                   grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    }
  }

  checkCudaError("GridData<T>::insertRotatedGridAdditive()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::resample(const int inDimX,
                                                          const int inDimY) {
  const int isComplex = int(grid->isComplex());
  const int dimZ = grid->numFields() * (1 + isComplex);
  const int numElements = inDimX * inDimY * dimZ;

  // Allocate vector for new grid
  thrust::device_vector<T> tmp(numElements);

  // Get the old spatial dimensions.
  const int oldDimX = grid->dimX();
  const int oldDimY = grid->dimY();

  // We have a number of safety pixels along each edge of the simulation space.
  // These pixels should not be a part of the interpolation.
  // TODO(Niclas): If we change the margin size then we need to read the margin
  // from file. And have two inputs to this function: marginOld and marginNew.
  const int margin = constants::GRAIN_SIZE_MARGIN;

  // Kernel parameters.
  // TODO(Niclas): This can be slightly optimized by not spawning threads for
  // the safety margin.
  const auto [blocksPerGrid, threadsPerBlock] =
      getKernelDimensions(inDimX, inDimY, dimZ);

  // Launch kernel.
  CONGA_LAUNCH((internal::resample_kernel), blocksPerGrid, threadsPerBlock,
               inDimX, inDimY, dimZ, oldDimX, oldDimY, margin,
               grid->getDataPointer(0), thrust::raw_pointer_cast(tmp.data()));

  // Change affected class member variables.
  grid->_dimX = inDimX;
  grid->_dimY = inDimY;
  grid->m_vector = tmp;
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::minmax(
    const GridData<T, thrust::device_vector<T>> &grid, T &minv, T &maxv,
    int field, Type::type part) const {
  const thrust::device_vector<T> &vec = grid.getVector();

  const size_t start = grid.getFieldComponentOffset(field, part);
  const size_t end = start + grid.dimX() * grid.dimY();

  const auto result =
      thrust::minmax_element(vec.begin() + start, vec.begin() + end);

  minv = *result.first;
  maxv = *result.second;
}

template <typename T>
std::pair<dim3, dim3>
GridFunctions<T, thrust::device_vector<T>>::getKernelDimensions() const {
  return getKernelDimensions(grid->_dimX, grid->_dimY);
}

template <typename T>
std::pair<dim3, dim3>
GridFunctions<T, thrust::device_vector<T>>::getKernelDimensionsFields() const {
  return getKernelDimensions(grid->_dimX, grid->_dimY, grid->_numFields);
}

template <typename T>
std::pair<dim3, dim3>
GridFunctions<T, thrust::device_vector<T>>::getKernelDimensions(
    const int inDimX, const int inDimY, const int inDimZ) {
  // The maximum number of threads per block.
  // What this number should be is up for debate. The tests are the fastest
  // with 32. but large simulations are faster with 256 or 512 for float, and
  // 128 for double.
  // TODO(Niclas): Investigate this.
  const int maxThreadsPerBlock = 128;

  // The maximum number of threads per block along the x-axis.
  const int maxThreadsX = maxThreadsPerBlock;

  // The maximum number of threads per block along the y-axis.
  const int maxThreadsY = maxThreadsPerBlock;

  // The maximum number of threads per block along the z-axis.
  // Note that 64 is the maximum along z accordint to the CUDA specs.
  const int maxThreadsZ = std::min(maxThreadsPerBlock, 64);

  // Set the number of threads per block along the x-axis.
  const int exponentX = static_cast<int>(std::ceil(std::log2(inDimX)));
  int nThreadsX =
      std::min(static_cast<int>(std::pow(2, exponentX)), maxThreadsX);

  // Set the number of threads per block along the y-axis.
  const int exponentY = static_cast<int>(std::ceil(std::log2(inDimY)));
  int nThreadsY =
      std::min(static_cast<int>(std::pow(2, exponentY)), maxThreadsY);

  // Set the number of threads per block along the z-axis.
  const int exponentZ = static_cast<int>(std::ceil(std::log2(inDimZ)));
  int nThreadsZ =
      std::min(static_cast<int>(std::pow(2, exponentZ)), maxThreadsZ);

  // Reduce the number of threads per block until less than the maximum.
  while ((nThreadsX * nThreadsY * nThreadsZ) > maxThreadsPerBlock) {
    // Check if the largest dim is one of y and z.
    const bool yLargest = (nThreadsY >= nThreadsZ) && (nThreadsY >= nThreadsX);
    const bool zLargest = (nThreadsZ >= nThreadsY) && (nThreadsZ >= nThreadsX);

    // Reduce the largest dimension, starting with z as it has the lowest bound
    // on the number of threads per block.
    if (zLargest) {
      nThreadsZ = std::max(nThreadsZ / 2, 1);
    } else if (yLargest) {
      nThreadsY = std::max(nThreadsY / 2, 1);
    } else {
      nThreadsX = std::max(nThreadsX / 2, 1);
    }
  }

  // Set the number of blocks along the x-axis.
  const int nBlocksX = static_cast<int>(
      std::ceil(static_cast<double>(inDimX) / static_cast<double>(nThreadsX)));

  // Set the number of blocks along the y-axis.
  const int nBlocksY = static_cast<int>(
      std::ceil(static_cast<double>(inDimY) / static_cast<double>(nThreadsY)));

  // Set the number of blocks along the z-axis.
  const int nBlocksZ = static_cast<int>(
      std::ceil(static_cast<double>(inDimZ) / static_cast<double>(nThreadsZ)));

  const dim3 blocksPerGrid(nBlocksX, nBlocksY, nBlocksZ);
  const dim3 threadsPerBlock(nThreadsX, nThreadsY, nThreadsZ);
  return {blocksPerGrid, threadsPerBlock};
}

template <typename T>
int GridFunctions<T, thrust::device_vector<T>>::checkCudaError(
    std::string msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << msg << ": " << cudaGetErrorString(err) << std::endl;
    return 1;
  } else
    return 0;
}
} // namespace conga

#endif // CONGA_GRID_FUNCTIONS_GPU_H_
