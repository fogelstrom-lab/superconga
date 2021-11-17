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
#ifndef GRID_FUNCTIONS_GPU_H_
#define GRID_FUNCTIONS_GPU_H_

#include "Type.h"
#include "utils.h"
#include <cmath>
#include <thrust/device_vector.h>

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
  void resample(const int nx_new, const int ny_new);
  void resample(GridData<T, thrust::device_vector<T>> *&target,
                const int nx_new, const int ny_new) const;

  // Find min and max value of range
  void minmax(const GridData<T, thrust::device_vector<T>> &grid, T &minv,
              T &maxv, int field, Type::type part) const;

  // Helper functions to create kernel dimensions
  void getKernelDimensions(dim3 &blocksPerGrid, dim3 &threadsPerBlock) const;
  void getKernelDimensionsFields(dim3 &blocksPerGrid,
                                 dim3 &threadsPerBlock) const;
  static void getKernelDimensions(dim3 &bpg, dim3 &tpb, int nx, int ny,
                                  int nf = 1);

  // Check for latest error
  static int checkCudaError(std::string msg = "");
};

namespace internal {
template <typename T, typename T2>
__global__ void typeConvert_kernel(const T *data_src, T2 *data_dst,
                                   const int Nx, const int Ny,
                                   const int complex) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tf = blockIdx.z;

  const int idx = tf * (1 + complex) * Nx * Ny + ty * Nx + tx;

  if (tx < Nx && ty < Ny)
    data_dst[idx] = T2(data_src[idx]);
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
__global__ void complexMagnitudeSq_kernel(const T *data_re, const T *data_im,
                                          T *result, const int Nx,
                                          const int Ny) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tf = blockIdx.z;

  const int idx_src = tf * 2 * Nx * Ny + ty * Nx + tx;
  const int idx_dst = tf * Nx * Ny + ty * Nx + tx;

  if (tx < Nx && ty < Ny) {
    T re = data_re[idx_src];
    T im = data_im[idx_src];

    result[idx_dst] = re * re + im * im;
  }
}

template <typename T>
__global__ void complexPhase_kernel(const T *const inReal,
                                    const T *const inImag, T *const outPhase,
                                    const int inNumX, const int inNumY) {
  // Coordinates to grid.
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tf = blockIdx.z;

  // TODO(niclas): This requires some explanation.
  const int idx_src = tf * 2 * inNumX * inNumY + ty * inNumX + tx;
  const int idx_dst = tf * inNumX * inNumY + ty * inNumX + tx;

  // TODO(niclas): We should check that tf < numFields or something. We souldn't
  // trust that the kernel call was made with exactly the right number of
  // threads.
  if (tx < inNumX && ty < inNumY) {
    const T real = inReal[idx_src];
    const T imag = inImag[idx_src];

    outPhase[idx_dst] = atan2(imag, real);
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
__global__ void div_kernel(const T *data_x, const T *data_y, T *data_dst,
                           const T h, const int Nx, const int Ny) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx > 0 && tx < Nx - 1 && ty > 0 && ty < Ny - 1) {
    // Grid index
    const int idx = ty * Nx + tx;

    data_dst[idx] =
        (data_x[idx + 1] - data_x[idx]) + (data_y[idx + Nx] - data_y[idx]) / h;
  }
}

template <typename T>
__global__ void curl_kernel(const T *data_x, const T *data_y, T *data_dst,
                            const T h, const int Nx, const int Ny) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  // Grid index
  const int idx = ty * Nx + tx;

  if (tx > 0 && tx < Nx - 1 && ty > 0 && ty < Ny - 1) {
    const T Ay = data_y[ty * Nx + tx + 1] - data_y[idx];
    const T Ax = data_x[(ty + 1) * Nx + tx] - data_x[idx];

    data_dst[idx] = (Ay - Ax) / h;
  } else if (tx == 0 || ty == 0 || tx == Nx - 1 || ty == Ny - 1)
    data_dst[idx] = T(0.0);
}

template <typename T>
__global__ void del2_kernel(T *del2_f, const T *f, const T inv_h2, const int Nx,
                            const int Ny) {
  // Coordinates to grid
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = ty * Nx + tx;

  if (tx > 0 && tx < Nx - 1 && ty > 0 && ty < Ny - 1)
    del2_f[idx] = (f[idx + 1] + f[idx - 1] + f[idx + Nx] + f[idx - Nx] -
                   T(4.0) * f[idx]) *
                  inv_h2;
  else if (tx == 0 || ty == 0 || tx == Nx - 1 || ty == Ny - 1)
    del2_f[idx] = T(0.0);
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
__global__ void resample_kernel(const T *data, T *data_tmp, const int Nx,
                                const int Ny, const int Nx_new,
                                const int Ny_new) {
  // Coordinates to new resampling grid
  const int x_new = blockIdx.x * blockDim.x + threadIdx.x;
  const int y_new = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_new < Nx_new && y_new < Ny_new) {
    const int idx_new = y_new * Nx_new + x_new;

    // Scaling factor between grids
    const T sx = (T)(Nx - 1) / (T)(Nx_new - 1);
    const T sy = (T)(Ny - 1) / (T)(Ny_new - 1);

    // Coordinates to old grid
    const T x = sx * x_new;
    const T y = sy * y_new;

    const int x_int = x;
    const int y_int = y;

    const T x_frac = x - (T)x_int;
    const T y_frac = y - (T)y_int;

    // Bilinear interpolation
    int x_add = 1, y_add = 1;

    if (x_new == Nx_new - 1)
      x_add = 0;
    if (y_new == Ny_new - 1)
      y_add = 0;

    T x1_interp, x2_interp;

    x1_interp = ((T)1.0 - x_frac) * data[y_int * Nx + x_int] +
                x_frac * data[y_int * Nx + (x_int + x_add)];

    x2_interp = ((T)1.0 - x_frac) * data[(y_int + y_add) * Nx + x_int] +
                x_frac * data[(y_int + y_add) * Nx + (x_int + x_add)];

    data_tmp[idx_new] = ((T)1.0 - y_frac) * x1_interp + y_frac * x2_interp;
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
  dim3 tpb, bpg;
  getKernelDimensionsFields(bpg, tpb);

  int complex = int(grid->isComplex());

  internal::typeConvert_kernel<T, T2><<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real), other.getDataPointer(0, Type::real),
      grid->_dimX, grid->_dimY, complex);

  if (complex)
    internal::typeConvert_kernel<T, T2><<<bpg, tpb>>>(
        grid->getDataPointer(0, Type::imag),
        other.getDataPointer(0, Type::imag), grid->_dimX, grid->_dimY, complex);
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

  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  int f_start = 0, f_end = grid->_numFields - 1;

  if (field != -1) {
    f_start = field;
    f_end = field;
  }

  for (int i = f_start; i <= f_end; ++i) {
    internal::transpose_kernel<<<bpg, tpb>>>(
        grid->getDataPointer(i, Type::real), grid->_dimX);
    if (grid->isComplex())
      internal::transpose_kernel<<<bpg, tpb>>>(
          grid->getDataPointer(i, Type::imag), grid->_dimX);
  }

  return *grid;
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexMagnitudeSq(
    GridData<T, thrust::device_vector<T>> *result) const {
  dim3 tpb, bpg;
  getKernelDimensionsFields(bpg, tpb);

  internal::complexMagnitudeSq_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real), grid->getDataPointer(0, Type::imag),
      result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY);

  checkCudaError("GridData::complexMagnitudeSq()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexPhase(
    GridData<T, thrust::device_vector<T>> *result) const {
  dim3 tpb, bpg;
  getKernelDimensionsFields(bpg, tpb);

  internal::complexPhase_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real), grid->getDataPointer(0, Type::imag),
      result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY);

  checkCudaError("GridData::complexPhase()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::complexPhaseGradient(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  internal::complexPhaseGradient_kernel<<<bpg, tpb>>>(
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
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();

  thrust::device_vector<T> dst(dimX * dimY);
  internal::momentKernel<<<bpg, tpb>>>(dimX, dimY,
                                       grid->getDataPointer(0, Type::real),
                                       grid->getDataPointer(1, Type::real),
                                       thrust::raw_pointer_cast(dst.data()));

  checkCudaError("GridData<T>::moment()");

  // Two factors of h comes from the integral, the other from the position
  // vector.
  const T factor = static_cast<T>(0.5) * h * h * h;
  const T init = static_cast<T>(0);
  const T result = factor * thrust::reduce(dst.begin(), dst.end(), init);

  return result;
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::fieldMagnitude(
    GridData<T, thrust::device_vector<T>> *result) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  internal::fieldMagnitude_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY,
      grid->_numFields, int(grid->isComplex()));

  checkCudaError("GridData<T>::fieldMagnitude()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::div(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  internal::div_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real), grid->getDataPointer(1, Type::real),
      result->getDataPointer(0, Type::real), h, grid->dimX(), grid->dimY());

  checkCudaError("GridData<T>::div()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::curl(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  internal::curl_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real), grid->getDataPointer(1, Type::real),
      result->getDataPointer(0, Type::real), h, grid->dimX(), grid->dimY());

  checkCudaError("GridData<T>::curl()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::del2(
    GridData<T, thrust::device_vector<T>> *result, const T h) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  const T inv_h2 = 1.0 / (h * h);

  for (int c = 0; c < result->numFields(); ++c) {
    internal::del2_kernel<<<bpg, tpb>>>(result->getDataPointer(c, Type::real),
                                        grid->getDataPointer(c, Type::real),
                                        inv_h2, grid->dimX(), grid->dimY());
  }

  checkCudaError("GridData<T>::del2()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::sumFieldComponents(
    GridData<T, thrust::device_vector<T>> *result) const {
  dim3 tpb, bpg;
  getKernelDimensions(bpg, tpb);

  internal::sumFieldComponents_kernel<<<bpg, tpb>>>(
      grid->getDataPointer(0, Type::real),
      result->getDataPointer(0, Type::real), grid->_dimX, grid->_dimY,
      grid->_numFields, int(grid->isComplex()));

  if (grid->isComplex())
    internal::sumFieldComponents_kernel<<<bpg, tpb>>>(
        grid->getDataPointer(0, Type::imag),
        result->getDataPointer(0, Type::imag), grid->_dimX, grid->_dimY,
        grid->_numFields, int(grid->isComplex()));

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

  dim3 threadsPerBlock, blocksPerGrid;
  getKernelDimensions(blocksPerGrid, threadsPerBlock, tpb, bpg);

  for (int c = 0; c < grid->_numFields; ++c) {
    int c_other = c >= Nf_other ? Nf_other - 1 : c;

    internal::insertGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        grid->getDataPointer(c, Type::real),
        other->getDataPointer(c_other, Type::real), grid->_dimX, grid->_dimY,
        Nx_other, Ny_other, x_start_this, y_start_this, x_start_other,
        y_start_other);

    if (grid->isComplex()) {
      internal::insertGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::imag),
          other->getDataPointer(c_other, Type::imag), grid->_dimX, grid->_dimY,
          Nx_other, Ny_other, x_start_this, y_start_this, x_start_other,
          y_start_other);
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
  dim3 threadsPerBlock, blocksPerGrid;
  getKernelDimensions(blocksPerGrid, threadsPerBlock);

  for (int c = 0; c < grid->_numFields; ++c) {
    int field_other = c;

    if (other->numFields() <= c)
      field_other = 0;

    if (complex_this && complex_other) {
      internal::insertRotatedGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::real),
          other->getDataPointer(field_other, Type::real), grid->_dimX,
          grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);

      internal::insertRotatedGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::imag),
          other->getDataPointer(field_other, Type::imag), grid->_dimX,
          grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this && !complex_other) {
      internal::insertRotatedGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::real),
          other->getDataPointer(field_other, Type::real), grid->_dimX,
          grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this) {
      internal::insertRotatedGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::real),
          other->getDataPointer(field_other, part), grid->_dimX, grid->_dimY,
          Nx_other, Ny_other, phi, x_offset, y_offset);
    } else {
      internal::insertRotatedGrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, part),
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
  dim3 threadsPerBlock, blocksPerGrid;
  getKernelDimensions(blocksPerGrid, threadsPerBlock);

  for (int c = fieldStart; c < fieldEnd; ++c) {
    int field_other = c;

    if (other->numFields() <= c)
      field_other = 0; // Maybe put this to other->numFields()-1 ??

    if (complex_this && complex_other) {
      internal::
          insertRotatedGridAdditive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
              grid->getDataPointer(c, Type::real),
              other->getDataPointer(field_other, Type::real), grid->_dimX,
              grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);

      internal::
          insertRotatedGridAdditive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
              grid->getDataPointer(c, Type::imag),
              other->getDataPointer(field_other, Type::imag), grid->_dimX,
              grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this && !complex_other) {
      internal::
          insertRotatedGridAdditive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
              grid->getDataPointer(c, Type::real),
              other->getDataPointer(field_other, Type::real), grid->_dimX,
              grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else if (!complex_this) {
      internal::
          insertRotatedGridAdditive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
              grid->getDataPointer(c, Type::real),
              other->getDataPointer(field_other, part), grid->_dimX,
              grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    } else {
      internal::
          insertRotatedGridAdditive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
              grid->getDataPointer(c, part),
              other->getDataPointer(field_other, Type::real), grid->_dimX,
              grid->_dimY, Nx_other, Ny_other, phi, x_offset, y_offset);
    }
  }

  checkCudaError("GridData<T>::insertRotatedGridAdditive()");
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::resample(const int nx_new,
                                                          const int ny_new) {
  const int isComplex = int(grid->isComplex());
  const int numFields = grid->numFields();
  const int numElements = nx_new * ny_new * numFields * (1 + isComplex);

  // Allocate vector for new grid
  thrust::device_vector<T> vec_tmp(numElements);

  const int nx = grid->dimX();
  const int ny = grid->dimY();

  T *data_tmp = thrust::raw_pointer_cast(vec_tmp.data());

  // Kernel launch
  dim3 threadsPerBlock, blocksPerGrid;
  getKernelDimensions(blocksPerGrid, threadsPerBlock, nx_new, ny_new);

  for (int c = 0; c < numFields; ++c) {
    int fieldOffset = (nx_new * ny_new * (1 + isComplex)) * c;

    internal::resample_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        grid->getDataPointer(c, Type::real), data_tmp + fieldOffset,
        grid->_dimX, grid->_dimY, nx_new, ny_new);

    if (isComplex) {
      internal::resample_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::imag),
          data_tmp + fieldOffset + (nx_new * ny_new), grid->_dimX, grid->_dimY,
          nx_new, ny_new);
    }
  }

  grid->m_vector = vec_tmp;

  // Change affected class member variables
  grid->_dimX = nx_new;
  grid->_dimY = ny_new;

  grid->_allocatedMemTotal -= nx * ny * numFields * (1 + isComplex) * sizeof(T);
  grid->_allocatedMemTotal += numElements * sizeof(T);
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::resample(
    GridData<T, thrust::device_vector<T>> *&target, const int nx_new,
    const int ny_new) const {
  GridData<T, thrust::device_vector<T>>::initializeGrid(
      target, nx_new, ny_new, grid->numFields(), grid->representation());

  const int isComplex = int(grid->isComplex());
  const int numFields = grid->numFields();

  const int nx = grid->dimX();
  const int ny = grid->dimY();

  // Kernel launch
  dim3 threadsPerBlock, blocksPerGrid;
  getKernelDimensions(blocksPerGrid, threadsPerBlock, nx_new, ny_new);

  for (int c = 0; c < numFields; ++c) {
    internal::resample_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        grid->getDataPointer(c, Type::real),
        target->getDataPointer(c, Type::real), nx, ny, nx_new, ny_new);

    if (isComplex) {
      internal::resample_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          grid->getDataPointer(c, Type::imag),
          target->getDataPointer(c, Type::imag), nx, ny, nx_new, ny_new);
    }
  }
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
void GridFunctions<T, thrust::device_vector<T>>::getKernelDimensions(
    dim3 &blocksPerGrid, dim3 &threadsPerBlock) const {
  getKernelDimensions(blocksPerGrid, threadsPerBlock, grid->_dimX, grid->_dimY,
                      1);
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::getKernelDimensionsFields(
    dim3 &blocksPerGrid, dim3 &threadsPerBlock) const {
  getKernelDimensions(blocksPerGrid, threadsPerBlock, grid->_dimX, grid->_dimY,
                      grid->_numFields);
}

template <typename T>
void GridFunctions<T, thrust::device_vector<T>>::getKernelDimensions(
    dim3 &blocksPerGrid, dim3 &threadsPerBlock, int Nx, int Ny, int Nf) {
  int tpb_x, bpg_x;

  const int threadmax = 64;

  if (Nx <= threadmax) {
    tpb_x = Nx;
    bpg_x = 1;
  } else {
    tpb_x = threadmax;
    bpg_x = Nx / threadmax;

    if (tpb_x * bpg_x != Nx)
      bpg_x++;
  }

  threadsPerBlock = dim3(tpb_x, 1, 1);
  blocksPerGrid = dim3(bpg_x, Ny, Nf);
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

#endif // GRID_FUNCTIONS_GPU_H_
