//===-------------- GridFunctions_CPU.h - CPU grid functions. -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU-specific implementation of functions to operate on Grid functions, i.e.
/// thrust host vectors.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GRID_FUNCTIONS_CPU_H_
#define CONGA_GRID_FUNCTIONS_CPU_H_

#include "../Type.h"
#include "../utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>
#include <cmath>

namespace conga {
///////////////////////////////////////////
// Code path for CPU (host) specific code
///////////////////////////////////////////
/*
// Declaring the fully templated version, so we can specialize below.
template <typename T, typename DEV>
struct GridFunctions
{};
*/
// Forward declaration.
template <typename, typename> class GridData;

template <typename T> struct GridFunctions<T, thrust::host_vector<T>> {
  GridData<T, thrust::host_vector<T>> *grid;

  GridFunctions(GridData<T, thrust::host_vector<T>> *ptr) : grid(ptr) {}

  template <typename T2>
  void typeConvert(GridData<T2, thrust::device_vector<T2>> &other) const;

  // Compute the complex magnitude squared, of specified field.
  void complexMagnitudeSq(GridData<T, thrust::host_vector<T>> *result) const;

  // Compute complex phase of this (complex) grid.
  void complexPhase(GridData<T, thrust::host_vector<T>> *result) const;

  // Compute gradient of complex phase of this grid.
  void complexPhaseGradient(GridData<T, thrust::host_vector<T>> *result,
                            const T h) const;

  /// \brief Compute the moment: 0.5 * sum(cross(r, grid)). Note that the only
  /// the z-component is returned.
  ///
  /// \param h The side length of a lattice site.
  ///
  /// \return The result.
  T moment(const T h) const;

  // Compute the magnitude of the real part of this vector field.
  void fieldMagnitude(GridData<T, thrust::host_vector<T>> *result) const;

  // Sum all fields.
  void sumFieldComponents(GridData<T, thrust::host_vector<T>> *result) const;

  // Insert other grid into this. <x,y>_offset are offset in discrete elements
  // from center. x_offset = y_offset = 0 results in a centered grid.
  void insertGrid(const GridData<T, thrust::host_vector<T>> *other,
                  const int x_offset, const int y_offset);

  // Insert other grid into this, with applied rotation (to other grid).
  void insertRotatedGrid(const GridData<T, thrust::host_vector<T>> *other,
                         const T phi, const int x_offset, const int y_offset,
                         Type::type part);

  // Add other grid to this, with applied rotation (to other grid).
  void
  insertRotatedGridAdditive(const GridData<T, thrust::host_vector<T>> *other,
                            const T phi, int field, const int x_offset,
                            const int y_offset, Type::type part);

  // Change resolution of grid, with linear interpolation of existing data.
  void resample(const int nx_new, const int ny_new);

  // Find min and max value of range
  void minmax(const GridData<T, thrust::host_vector<T>> &grid, T &minv, T &maxv,
              int field, Type::type part) const;
};

namespace internal {
// Iterators
//
template <typename T, typename DEV, template <typename> class F>
void iterOne(const GridData<T, DEV> &dst, F<T> functor, int field = -1,
             Type::type part = Type::complex) {
  const int Nx = dst.dimX();
  const int Ny = dst.dimY();

  const int doReal = int(part == Type::real || part == Type::complex);
  const int doImag =
      int((part == Type::imag || part == Type::complex) && dst.isComplex());

  const Type::type parts[2] = {Type::real, Type::imag};

  const int c_begin = field == -1 ? 0 : field;
  const int c_end = field == -1 ? dst.numFields() : field + 1;

  int i, j, p, c, idx;
  T *data_dst;

  for (c = c_begin; c < c_end; ++c) {
    for (p = 1 - doReal; p < 1 + doImag; ++p) {
      data_dst = dst.getDataPointer(c, parts[p]);

      for (j = 0; j < Ny; ++j) {
        for (i = 0; i < Nx; ++i) {
          idx = j * Nx + i;
          data_dst[idx] = functor(data_dst[idx]);
        }
      }
    }
  }
}

template <typename T, typename DEV, template <typename> class F>
void iterTwo(const GridData<T, DEV> &src, GridData<T, DEV> &dst, F<T> functor,
             Type::type part = Type::complex, int fieldSrc = -1,
             int fieldDst = -1) {
  const int Nx = dst.dimX();
  const int Ny = dst.dimY();

  const int numFieldsSrc = src.numFields();
  const int numFieldsDst = dst.numFields();

  const Type::type parts[2] = {Type::real, Type::imag};

  const int doReal = int(part == Type::real || part == Type::complex);
  const int doImag =
      int((part == Type::imag || part == Type::complex) && src.isComplex());

  int c_end = 1, doAllFields = 0;

  if (fieldSrc == -1) {
    c_end = numFieldsSrc;
    doAllFields = 1;
  }

  int i, j, p, c, idx;
  T *data_src, *data_dst;

  for (c = 0; c < c_end; ++c) {
    fieldSrc = doAllFields ? c : fieldSrc;
    fieldDst = doAllFields ? c : fieldDst;

    fieldDst = fieldDst >= numFieldsDst ? numFieldsDst - 1 : fieldDst;

    for (p = 1 - doReal; p < 1 + doImag; ++p) {
      data_src = src.getDataPointer(fieldSrc, parts[p]);
      data_dst = dst.getDataPointer(fieldDst, parts[p]);

      for (j = 0; j < Ny; ++j) {
        for (i = 0; i < Nx; ++i) {
          idx = j * Nx + i;
          data_dst[idx] = functor(data_src[idx], data_dst[idx]);
        }
      }
    }
  }
}

template <typename T, typename DEV, template <typename> class F>
void iterTwoComplexToReal(const GridData<T, DEV> &src, GridData<T, DEV> &dst,
                          F<T> functor) {
  const int Nx = dst.dimX();
  const int Ny = dst.dimY();

  int i, j, c, idx;
  T *data_src_re, *data_src_im, *data_dst;

  for (c = 0; c < dst.numFields(); ++c) {
    data_src_re = src.getDataPointer(c, Type::real);
    data_src_im = src.getDataPointer(c, Type::imag);

    data_dst = dst.getDataPointer(c, Type::real);

    for (j = 0; j < Ny; ++j) {
      for (i = 0; i < Nx; ++i) {
        idx = j * Nx + i;
        data_dst[idx] = functor(data_src_re[idx], data_src_im[idx]);
      }
    }
  }
}

// Functors
//
template <typename T> struct EqFuncScalar {
  const T a;
  EqFuncScalar(T _a) : a(_a) {}
  T operator()(const T &dst) { return a; }
};

template <typename T> struct EqFuncGrid {
  EqFuncGrid() {}
  T operator()(const T &src, const T &dst) { return src; }
};

template <typename T> struct MultFuncOne {
  const T a;
  MultFuncOne(T _a) : a(_a) {}
  T operator()(const T &dst) { return a * dst; }
};

template <typename T> struct MultFuncTwo {
  MultFuncTwo() {}
  T operator()(const T &src, const T &dst) { return src * dst; }
};

template <typename T> struct AddFuncOne {
  const T a;
  AddFuncOne(T _a) : a(_a) {}
  T operator()(const T &dst) { return a + dst; }
};

template <typename T> struct AddFuncTwo {
  AddFuncTwo() {}
  T operator()(const T &src, const T &dst) { return src + dst; }
};

template <typename T> struct SubFuncTwo {
  SubFuncTwo() {}
  T operator()(const T &src, const T &dst) { return dst - src; }
};

template <typename T> struct ComplexMagnitudeSqFunc {
  ComplexMagnitudeSqFunc() {}
  T operator()(const T inReal, const T inImag) {
    return inReal * inReal + inImag * inImag;
  }
};

template <typename T> struct ComplexPhaseFunc {
  ComplexPhaseFunc() {}
  T operator()(const T inReal, const T inImag) { return atan2(inImag, inReal); }
};

template <typename T> struct FieldMagnitudeFunc {
  FieldMagnitudeFunc() {}
  T operator()(const T &src, const T &dst) { return dst + src * src; }
};

template <typename T> struct SqrtFunc {
  SqrtFunc() {}
  T operator()(const T &dst) { return std::sqrt(dst); }
};

template <typename T> struct LogFunc {
  const T a;
  LogFunc(T _a) : a(_a) {}
  T operator()(const T &dst) { return std::log(dst + a); }
};
} // namespace internal

template <typename T>
template <typename T2>
void GridFunctions<T, thrust::host_vector<T>>::typeConvert(
    GridData<T2, thrust::device_vector<T2>> &other) const {
  other.getVector() = grid->getVector();
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::complexMagnitudeSq(
    GridData<T, thrust::host_vector<T>> *result) const {
  iterTwoComplexToReal(*grid, *result, internal::ComplexMagnitudeSqFunc<T>());
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::complexPhase(
    GridData<T, thrust::host_vector<T>> *result) const {
  iterTwoComplexToReal(*grid, *result, internal::ComplexPhaseFunc<T>());
}

template <typename T>
T GridFunctions<T, thrust::host_vector<T>>::moment(const T h) const {
  // Sanity check.
  assert(grid->numFields() == 2);
  assert(grid->representation() == Type::real);

  // Get pointers.
  const T *const srcX = grid->getDataPointer(0, Type::real);
  const T *const srcY = grid->getDataPointer(1, Type::real);

  // Dimensions.
  const int dimX = grid->dimX();
  const int dimY = grid->dimY();

  // Coordinate of the center.
  const T centerX = static_cast<T>(0.5) * static_cast<T>(dimX - 1);
  const T centerY = static_cast<T>(0.5) * static_cast<T>(dimY - 1);

  // Do cross product and sum.
  T sum = static_cast<T>(0);
  for (int yIdx = 0; yIdx < dimY; ++yIdx) {
    for (int xIdx = 0; xIdx < dimX; ++xIdx) {
      const int idx = yIdx * dimY + xIdx;
      const T x = static_cast<T>(xIdx) - centerX;
      const T y = static_cast<T>(yIdx) - centerY;
      sum += x * srcY[idx] - y * srcX[idx];
    }
  }

  // Two factors of h comes from the integral, the other from the position
  // vector.
  const T factor = static_cast<T>(0.5) * h * h * h;
  const T result = factor * sum;

  return result;
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::fieldMagnitude(
    GridData<T, thrust::host_vector<T>> *result) const {
  result->setZero();
  iterTwo(*grid, *result, internal::FieldMagnitudeFunc<T>(), Type::real);
  result->sqrt();
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::sumFieldComponents(
    GridData<T, thrust::host_vector<T>> *result) const {
  result->setZero();
  iterTwo(*grid, *result, internal::AddFuncTwo<T>(), grid->representation());
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::complexPhaseGradient(
    GridData<T, thrust::host_vector<T>> *result, const T h) const {
  const int Nx = grid->_dimX;
  const int Ny = grid->_dimY;

  T *data_src_re = grid->getDataPointer(0, Type::real);
  T *data_src_im = grid->getDataPointer(0, Type::imag);
  T *data_dst_x = result->getDataPointer(0, Type::real);
  T *data_dst_y = result->getDataPointer(1, Type::real);

  T f0, f1, f2, g0, g1, g2, denom, dPhase_dx, dPhase_dy;

  // Grid index
  int idx, idx_x0, idx_x2, idx_y0, idx_y2;

  // Set gradient to zero at borders for now. Implement other gradient methods
  // later!
  dPhase_dx = (T)0.0;
  dPhase_dy = (T)0.0;

  for (int ty = 0; ty < Ny; ++ty) {
    for (int tx = 0; tx < Nx; ++tx) {
      dPhase_dx = (T)0.0;
      dPhase_dy = (T)0.0;

      idx = ty * Nx + tx;

      if (tx > 0 && tx < Nx - 1 && ty > 0 && ty < Ny - 1) {
        idx_x0 = ty * Nx + (tx - 1);
        idx_x2 = ty * Nx + (tx + 1);

        idx_y0 = (ty - 1) * Nx + tx;
        idx_y2 = (ty + 1) * Nx + tx;

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
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::insertGrid(
    const GridData<T, thrust::host_vector<T>> *other, const int x_offset,
    const int y_offset) {
  // insert 'other' grid into 'this'
  const int Nx_this = grid->dimX();
  const int Ny_this = grid->dimY();

  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();
  const int Nf_other = other->numFields();

  const Type::type part[2] = {Type::real, Type::imag};

  int x_start_this = 0, y_start_this = 0, x_start_other = 0, y_start_other = 0,
      x_max, y_max, c_other;
  int tx_this, ty_this, tx_other, ty_other, idx_this, idx_other;

  int x_diff = (grid->_dimX - Nx_other) / 2;
  int y_diff = (grid->_dimY - Ny_other) / 2;

  if (x_diff < 0) // Other grid bigger than this
  {
    x_start_other = -x_diff + x_offset;
    x_max = grid->_dimX;
  } else // Other grid smaller (or equal) than this
  {
    x_start_this = x_diff + x_offset;
    x_max = Nx_other;
  }

  if (y_diff < 0) {
    y_start_other = -y_diff + y_offset;
    y_max = grid->_dimY;
  } else {
    y_start_this = y_diff + y_offset;
    y_max = Ny_other;
  }

  T *data_this, *data_other;

  for (int c = 0; c < grid->numFields(); ++c) {
    c_other = c >= Nf_other ? Nf_other - 1 : c;

    for (int p = 0; p < 1 + int(grid->isComplex()); ++p) {
      data_this = grid->getDataPointer(c, part[p]);
      data_other = other->getDataPointer(c_other, part[p]);

      for (int ty = 0; ty < y_max; ++ty) {
        for (int tx = 0; tx < x_max; ++tx) {
          tx_this = tx + x_start_this;
          ty_this = ty + y_start_this;

          tx_other = tx + x_start_other;
          ty_other = ty + y_start_other;

          if (tx_this < Nx_this && ty_this < Ny_this && tx_other < Nx_other &&
              ty_other < Ny_other) {
            idx_this = ty_this * Nx_this + tx_this;
            idx_other = ty_other * Nx_other + tx_other;

            data_this[idx_this] = data_other[idx_other];
          }
        }
      }
    }
  }
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::insertRotatedGrid(
    const GridData<T, thrust::host_vector<T>> *other, const T phi,
    const int x_offset, const int y_offset, Type::type representation) {
  // Grid dimensions of grid to rotate and read from (lookup grid)
  const int Nx_this = grid->dimX();
  const int Ny_this = grid->dimY();

  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();
  const int Nf_other = other->numFields();

  const int complex = int(grid->isComplex());

  const Type::type part[2] = {Type::real, Type::imag};

  // Coordinate transforms from current grid to lookup grid
  const T x_transform = (Nx_other - Nx_this) / (T)2.0;
  const T y_transform = (Ny_other - Ny_this) / (T)2.0;

  // Shifts lookup grid center to origin
  const T grid_origin_shift_x = (Nx_other - 1) / (T)2.0;
  const T grid_origin_shift_y = (Ny_other - 1) / (T)2.0;

  int idx_grid_out, c_other, X_int, Y_int;
  T X, Y, Xr, Yr, X_frac, Y_frac, X1_interp, X2_interp;

  T *data_this, *data_other;

  // Loop over elements
  for (int c = 0; c < grid->numFields(); ++c) {
    c_other = c >= Nf_other ? Nf_other - 1 : c;

    for (int p = 0; p < 1 + complex; ++p) {
      data_this = grid->getDataPointer(c, part[p]);
      data_other = other->getDataPointer(c_other, part[p]);

      for (int y = 0; y < Ny_this; ++y) {
        for (int x = 0; x < Nx_this; ++x) {
          // Array index to current grid
          idx_grid_out = y * Nx_this + x;

          // Coordinates to lookup grid
          X = x + x_transform - grid_origin_shift_x;
          Y = y + y_transform - grid_origin_shift_y;

          // Rotate centered coordinates, and put grid corner at origin
          Xr = (T)X * std::cos(-phi) - (T)Y * std::sin(-phi) +
               grid_origin_shift_x;
          Yr = (T)X * std::sin(-phi) + (T)Y * std::cos(-phi) +
               grid_origin_shift_y;

          // Bilinear interpolation
          X_int = Xr;
          Y_int = Yr;

          if ((X_int >= 0) && (X_int + 1 < Nx_other) && (Y_int >= 0) &&
              (Y_int + 1 < Ny_other)) {
            X_frac = Xr - (T)X_int;
            Y_frac = Yr - (T)Y_int;

            X1_interp =
                ((T)1.0 - X_frac) * data_other[Y_int * Nx_other + X_int] +
                X_frac * data_other[Y_int * Nx_other + (X_int + 1)];

            X2_interp =
                ((T)1.0 - X_frac) * data_other[(Y_int + 1) * Nx_other + X_int] +
                X_frac * data_other[(Y_int + 1) * Nx_other + (X_int + 1)];

            data_this[idx_grid_out] =
                ((T)1.0 - Y_frac) * X1_interp + Y_frac * X2_interp;
          }
        }
      }
    }
  }
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::insertRotatedGridAdditive(
    const GridData<T, thrust::host_vector<T>> *other, const T phi, int field,
    const int x_offset, const int y_offset, Type::type representation) {
  int fieldStart, fieldEnd;

  if (field == -1) {
    fieldStart = 0;
    fieldEnd = grid->_numFields;
  } else if (field < grid->_numFields) {
    fieldStart = field;
    fieldEnd = field + 1;
  } else {
#ifndef NDEBUG
    std::cout << "GridData()::insertRotatedGridAdditive : Field out of bounds, "
                 "operation terminated.\n";
#endif
    return;
  }

  // Grid dimensions of grid to rotate and read from (lookup grid)
  const int Nx_this = grid->dimX();
  const int Ny_this = grid->dimY();

  const int Nx_other = other->dimX();
  const int Ny_other = other->dimY();
  const int Nf_other = other->numFields();

  const int complex = int(grid->isComplex());

  const Type::type part[2] = {Type::real, Type::imag};

  // Coordinate transforms from current grid to lookup grid
  const T x_transform = (Nx_other - Nx_this) / (T)2.0;
  const T y_transform = (Ny_other - Ny_this) / (T)2.0;

  // Shifts lookup grid center to origin
  const T grid_origin_shift_x = (Nx_other - 1) / (T)2.0;
  const T grid_origin_shift_y = (Ny_other - 1) / (T)2.0;

  int idx_grid_out, c_other, X_int, Y_int;
  T X, Y, Xr, Yr, X_frac, Y_frac, X1_interp, X2_interp;

  T *data_this, *data_other;

  // Loop over elements
  for (int c = fieldStart; c < fieldEnd; ++c) {
    c_other = c >= Nf_other ? Nf_other - 1 : c;

    for (int p = 0; p < 1 + complex; ++p) {
      data_this = grid->getDataPointer(c, part[p]);
      data_other = other->getDataPointer(c_other, part[p]);

      for (int y = 0; y < Ny_this; ++y) {
        for (int x = 0; x < Nx_this; ++x) {
          // Array index to current grid
          idx_grid_out = y * Nx_this + x;

          // Coordinates to lookup grid
          X = x + x_transform - grid_origin_shift_x;
          Y = y + y_transform - grid_origin_shift_y;

          // Rotate centered coordinates, and put grid corner at origin
          Xr = (T)X * std::cos(-phi) - (T)Y * std::sin(-phi) +
               grid_origin_shift_x;
          Yr = (T)X * std::sin(-phi) + (T)Y * std::cos(-phi) +
               grid_origin_shift_y;

          // Bilinear interpolation
          X_int = Xr;
          Y_int = Yr;

          if ((X_int >= 0) && (X_int + 1 < Nx_other) && (Y_int >= 0) &&
              (Y_int + 1 < Ny_other)) {
            X_frac = Xr - (T)X_int;
            Y_frac = Yr - (T)Y_int;

            X1_interp =
                ((T)1.0 - X_frac) * data_other[Y_int * Nx_other + X_int] +
                X_frac * data_other[Y_int * Nx_other + (X_int + 1)];

            X2_interp =
                ((T)1.0 - X_frac) * data_other[(Y_int + 1) * Nx_other + X_int] +
                X_frac * data_other[(Y_int + 1) * Nx_other + (X_int + 1)];

            data_this[idx_grid_out] +=
                ((T)1.0 - Y_frac) * X1_interp + Y_frac * X2_interp;
          }
        }
      }
    }
  }
}

template <typename T>
void GridFunctions<T, thrust::host_vector<T>>::resample(const int nx_new,
                                                        const int ny_new) {
  const int isComplex = int(grid->isComplex());
  const int numFields = grid->numFields();
  const int numElements = nx_new * ny_new * numFields * (1 + isComplex);

  // Allocate vector for new grid
  thrust::host_vector<T> *vec_tmp = new thrust::host_vector<T>(numElements);

  T *data, *data_tmp;

  const int nx = grid->dimX();
  const int ny = grid->dimY();

  int i, j, c, p, idx_new;

  int x_int, y_int, x_add, y_add;
  T sx, sy, x, y, x_frac, y_frac, x1_interp, x2_interp;
  size_t ptr_offset;

  const Type::type part[2] = {Type::real, Type::imag};

  for (c = 0; c < grid->numFields(); ++c) {
    for (p = 0; p < 1 + isComplex; ++p) {
      data = grid->getDataPointer(c, part[p]);

      ptr_offset =
          nx_new * ny_new *
          (c * (1 + isComplex) + int(part[p] == Type::imag && isComplex));
      data_tmp = thrust::raw_pointer_cast(vec_tmp->data()) + ptr_offset;

      for (j = 0; j < ny_new; ++j) {
        for (i = 0; i < nx_new; ++i) {
          idx_new = j * nx_new + i;

          // Scaling factor between grids
          sx = (T)(nx - 1) / (T)(nx_new - 1);
          sy = (T)(ny - 1) / (T)(ny_new - 1);

          // Coordinates to old grid
          x = sx * i;
          y = sy * j;

          x_int = x;
          y_int = y;

          x_frac = x - (T)x_int;
          y_frac = y - (T)y_int;

          // Bilinear interpolation
          if (i == nx_new - 1)
            x_add = 0;
          else
            x_add = 1;
          if (j == ny_new - 1)
            y_add = 0;
          else
            y_add = 1;

          x1_interp = ((T)1.0 - x_frac) * data[y_int * nx + x_int] +
                      x_frac * data[y_int * nx + (x_int + x_add)];

          x2_interp = ((T)1.0 - x_frac) * data[(y_int + y_add) * nx + x_int] +
                      x_frac * data[(y_int + y_add) * nx + (x_int + x_add)];

          data_tmp[idx_new] =
              ((T)1.0 - y_frac) * x1_interp + y_frac * x2_interp;
        }
      }
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
void GridFunctions<T, thrust::host_vector<T>>::minmax(
    const GridData<T, thrust::host_vector<T>> &grid, T &minv, T &maxv,
    int field, Type::type part) const {
  const thrust::host_vector<T> &vec = grid.getVector();

  const size_t start = grid.getFieldComponentOffset(field, part);
  const size_t end = start + grid.dimX() * grid.dimY();

  const auto result =
      thrust::minmax_element(vec.begin() + start, vec.begin() + end);

  minv = *result.first;
  maxv = *result.second;
}
} // namespace conga

#endif // CONGA_GRID_FUNCTIONS_CPU_H_
