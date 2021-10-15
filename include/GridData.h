//===-- GridData.h - GridData class. --------------------------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GridData class and functionality. GridData is a data type that wraps thrust
/// host (device) vectors for CPU (GPU) implementation. Different kinds of
/// GridData objects are defined, like 2D real and complex fields, vector fields
/// etcetera.
///
//===----------------------------------------------------------------------===//
#ifndef GRID_DATA_H_
#define GRID_DATA_H_

#ifdef __CUDACC__
#define _DEV_ __host__ __device__
#else
#define _DEV_
#endif

#include "GridFunctions.h"
#include "GridFunctors.h"
#include "Type.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace conga {
// Forward declarations. These can be used when we want to refer to another
// class without including the .h file.
template <typename> class Parameters;

// Diverging codepaths (CPU and GPU) are put in GridFunctions class, with
// template specialization telling them apart
template <typename T, typename DEV>
class GridData : public GridFunctions<T, DEV> {
public:
  GridData();
  ~GridData();

  // Copy constructor
  GridData(const GridData &other); // Copies entire object

  // Allocates memory only, no field data values are set
  GridData(const GridData &other,
           int numFields); // Allocates with field dimensions from other, but
                           // number of fields as specified
  GridData(int nx, int ny, int numFields = 1,
           Type::type representation = Type::complex);

  // Operators
  bool operator==(const GridData &right) const;
  bool operator!=(const GridData &right) const;

  GridData &operator=(GridData right);
  GridData &operator=(const thrust::complex<T> right);
  GridData &operator=(const T right);

  // Try thrust::for_each instead of thrust::transform for better performance
  // for operators?

  // IMPORTANT NOTE on operators:
  // GridData does NOT represent matrices. As such the operator* does NOT
  // perform a matrix-matrix multiplication, but rather an element-element
  // multiplication; for all elements in the array: c_i = a_i * b_i

  void operator+=(const GridData &right);
  void operator+=(const thrust::complex<T> right);
  void operator+=(const T right);
  void operator-=(const GridData &right);
  void operator-=(const thrust::complex<T> right);
  void operator-=(const T right);
  void operator*=(const GridData &right);
  void operator*=(const thrust::complex<T> right);
  void operator*=(const T right);

  GridData operator+(GridData right) const;
  GridData operator+(const thrust::complex<T> right) const;
  GridData operator+(const T right) const;
  GridData operator-(GridData right) const;
  GridData operator-(const thrust::complex<T> right) const;
  GridData operator-(const T right) const;
  GridData operator*(GridData right) const;
  GridData operator*(const thrust::complex<T> right) const;
  GridData operator*(const T right) const;

  GridData operator-() const;

  // Returns a copy of the specified field (somewhat dubious use of operator[])
  GridData operator[](const int field) const;

  // operator() is identical to getDataPointer()
  T *operator()(const int field = 0,
                Type::type representation = Type::real) const {
    return thrust::raw_pointer_cast(m_vector.data()) +
           getFieldComponentOffset(field, representation);
  }

  // Type conversion operator
  template <typename T2, typename DEV2> operator GridData<T2, DEV2>();

  template <typename T2> operator GridData<T2, thrust::device_vector<T2>>();

  //// Query functions ////

  int dimX() const { return _dimX; }
  int dimY() const { return _dimY; }
  int numFields() const { return _numFields; }
  Type::type representation() const { return _representation; }

  bool isComplex() const { return _representation == Type::complex; }

  const DEV &getVector() const { return m_vector; }
  DEV &getVector() { return m_vector; }

  const T *getDataPointer(const int field = 0,
                          Type::type representation = Type::real) const {
    return thrust::raw_pointer_cast(m_vector.data()) +
           getFieldComponentOffset(field, representation);
  }
  T *getDataPointer(const int field = 0,
                    Type::type representation = Type::real) {
    return thrust::raw_pointer_cast(m_vector.data()) +
           getFieldComponentOffset(field, representation);
  }

  //// Modifying basic functions ////

  // Set elements of all or specified field(s) to scalar value.
  void setValue(const T val, Type::type representation = Type::complex,
                const int field = -1);

  // Set all elements to zero.
  void setZero();

  // Copy raw data from p_data to this grid.
  int setData(const T *p_host_data, size_t numElements);

  // In-place complex conjugation of field data.
  GridData &conjugate();

  // In-place transponate of field data. (Not implemented for CPU!!!)
  GridData &transpose(const int field = -1);

  // In-place square root (Only per-element. Does NOT perform complex-valued
  // square-root!)
  GridData &sqrt();

  //// Non-modifying utility functions ////

  // In all cases where the argument is of type GridData*, the object to which
  // the pointer refers must be initialized and have the correct dimensions.

  // Compute absolute value for grid (works for complex as well)
  GridData abs() const;

  // Compute the grain-summed absolute value
  T sumMagnitude() const;

  // Compute the grain-summed absolute square
  T sumSquare() const;

  // Compute the L1 norm
  T L1Norm() const;

  // Compute the L2 norm
  T L2Norm() const;

  // Compute the complex magnitude squared
  void complexMagnitudeSq(GridData *result) const;
  GridData complexMagnitudeSq() const;

  // Compute complex phase of this (complex) grid
  void complexPhase(GridData *result) const;
  GridData complexPhase() const;

  // Compute complex phase gradient of this grid
  void complexPhaseGradient(GridData *result, const T h = 1.0) const;
  GridData complexPhaseGradient(const T h = 1.0) const;

  // Compute the magnitude of this (real valued) vector field
  void fieldMagnitude(GridData *result) const;
  GridData fieldMagnitude() const;

  // Compute divergence of vector field (Not implemented for CPU!!!)
  void div(GridData *result, const T h = 1.0) const;
  GridData div(const T h = 1.0) const;

  // Compute curl of vector field (Not implemented for CPU!!!)
  // Only returns z-component, since x- and y-components will be zero for a
  // 2d-field
  void curl(GridData *result, const T h = 1.0) const;
  GridData curl(const T h = 1.0) const;

  // Compute laplacian (div(grad)) of field (Not implemented for CPU!!!)
  void del2(GridData *result, const T h = 1.0) const;
  GridData del2(const T h = 1.0) const;

  // Sum all field components.
  void sumFieldComponents(GridData *result) const;
  GridData sumFieldComponents() const;

  // Sum all elements of specified fields. Result returned (real) or put in
  // reference variables (complex).
  T sumField(const int field = 0) const; // Use for real valued
  void sumField(T &sum_re, T &sum_im,
                const int field = 0) const; // Use for real or complex valued

  // Sum all elements in a field, for all fields
  GridData sumFields() const;

  // Copy single field from this to target
  void copyFieldTo(GridData *target, int thisField, int otherField) const;

  void minmax(T &minv, T &maxv, int field = 0,
              Type::type part = Type::real) const;

  //// Modifying utility functions ////

  // Insert other grid into this. <x,y>_offset are offset in discrete elements
  // from center. x_offset = y_offset = 0 results in a centered grid.
  void insertGrid(const GridData *other, const int x_offset = 0,
                  const int y_offset = 0);

  // Insert other grid into this, with applied rotation (to other grid).
  void insertRotatedGrid(const GridData *other, const T phi,
                         const int x_offset = 0, const int y_offset = 0,
                         Type::type part = Type::complex);

  // Add other grid to this, with applied rotation (to other grid).
  void insertRotatedGridAdditive(const GridData *grid_insert, const T phi,
                                 int field = -1, const int x_offset = 0,
                                 const int y_offset = 0,
                                 Type::type part = Type::complex);

  // Change resolution of grid, with linear interpolation of existing data.
  void resample(const int nx_new, const int ny_new);
  void resample(GridData *&target, const int nx_new,
                const int ny_new) const; //(CPU NOT IMPLEMENTED)

  //// Helper query functions ////

  // Returns the pointer offset (==number of data elements) to the specified
  // field.
  size_t getFieldComponentOffset(int field,
                                 const Type::type representation = Type::real,
                                 Type::location pos = Type::start) const {
    const int complex = int(isComplex());
    return _dimX * _dimY *
           (field * (1 + complex) +
            int(representation == Type::imag && complex) +
            int(pos == Type::end));
  }

  // Returns the number of data elements in the entire array.
  size_t numElements() const { return numElementsField() * _numFields; }

  // Returns the number of elements of one field.
  size_t numElementsField() const {
    return size_t(_dimX) * size_t(_dimY) * size_t(1 + int(isComplex()));
  }

  //// File I/O ////

  int read(const std::filesystem::path &filename, Parameters<T> *param = 0);
  int read(const std::filesystem::path &filename, int num,
           Parameters<T> *param = 0);

  int write(const std::filesystem::path &filename,
            Parameters<T> *param = 0) const;
  int write(const std::filesystem::path &filename, int num,
            Parameters<T> *param = 0) const;

  int readNumpyArray(const std::filesystem::path &filename,
                     int numFields = 1); // Can only read real valued data
  int writeNumpyArray(const std::filesystem::path &filename,
                      Type::type representation = Type::complex) const;

  // Print data to terminal
  void print(int field, Type::type representation) const;
  void print(int field) const;
  void print() const;

  void printDimensions() const;

  //// Static functions ////
  static int readParameters(const std::filesystem::path &filename,
                            Parameters<T> *param);
  static int readParameters(const std::filesystem::path &filename, int num,
                            Parameters<T> *param);

  // Safely initialize or delete GridData pointers, without copying values.
  // *grid must point to properly initialized GridData, or == 0.
  static int initializeGrid(GridData *&grid, int Nx, int Ny, int numFields = 1,
                            Type::type representation = Type::complex);

  template <typename T2, typename DEV2>
  static int initializeGrid(GridData<T, DEV> *&grid,
                            const GridData<T2, DEV2> *other);

  static void deleteGrid(GridData *&grid);

  // Returns True if A.dimX * A.dimY == B.dimX * B.dimY
  static bool dimEq(const GridData &A, const GridData &B);

  // Used to compute residual
  static T differenceMagnitude(const GridData &A, const GridData &B);

  static size_t allocatedMemTotal() { return _allocatedMemTotal; }
  static void printAllocatedMemTotal() {
    std::cout << _allocatedMemTotal << std::endl;
  }

  // GridFunctions is a class used to contain the code which is CPU or GPU
  // specific. By declaring it a friend class it can access the private members
  // of this class.
  friend class GridFunctions<T, DEV>;

  // Functors for complex valued multiplication
  //

  // Complex vector dot product
  struct complexVectorProductFunctor {
    template <typename Tuple> _DEV_ void operator()(Tuple inOutValues) {
      const T temp_re = thrust::get<0>(inOutValues);
      const T temp_im = thrust::get<1>(inOutValues);
      thrust::get<0>(inOutValues) = (temp_re * thrust::get<2>(inOutValues)) -
                                    (temp_im * thrust::get<3>(inOutValues));
      thrust::get<1>(inOutValues) = (temp_re * thrust::get<3>(inOutValues)) +
                                    (temp_im * thrust::get<2>(inOutValues));
    }
  };

  // Scale complex vector with complex scalar
  struct scaleComplexVectorFunctor {
    const thrust::complex<T> scalar;

    scaleComplexVectorFunctor(const thrust::complex<T> _scalar)
        : scalar(_scalar) {}

    template <typename Tuple> _DEV_ void operator()(Tuple inOutValues) {
      const T temp_re = thrust::get<0>(inOutValues);
      const T temp_im = thrust::get<1>(inOutValues);

      thrust::get<0>(inOutValues) =
          (temp_re * scalar.real()) - (temp_im * scalar.imag());
      thrust::get<1>(inOutValues) =
          (temp_re * scalar.imag()) + (temp_im * scalar.real());
    }
  };

  // Scale complex vector with complex scalar
  struct addComplexScalarFunctor {
    const thrust::complex<T> scalar;

    addComplexScalarFunctor(const thrust::complex<T> _scalar)
        : scalar(_scalar) {}

    template <typename Tuple> _DEV_ void operator()(Tuple inOutValues) {

      thrust::get<0>(inOutValues) += scalar.real();
      thrust::get<1>(inOutValues) += scalar.imag();
    }
  };

  // Calculate square root of inner product: sqrt(x*x + y*y)
  struct hypotFunctor {
    template <typename Tuple> _DEV_ T operator()(const Tuple inValues) {
      return std::hypot(thrust::get<0>(inValues), thrust::get<1>(inValues));
    }
  };

  // Calculate absolute value: sqrt(x*x)
  struct absFunctor {
    _DEV_ T operator()(const T inValue) const { return std::abs(inValue); }
  };

private:
  // Specifies if GridData contains real (Type::real) or complex (Type::complex)
  // valued data
  Type::type _representation;

  // Number of elements in x and y
  int _dimX;
  int _dimY;

  // Number of fields
  int _numFields;

  // DEV can be thrust::host_vector<T> or thrust::device_vector<T>. Contains
  // data array.
  DEV m_vector;

  // The total memory for ALL grid objects.
  static size_t _allocatedMemTotal;

  // Swap function implemented according to the copy-and-swap c++ idiom (very
  // useful!)
  template <typename S, typename DEVS>
  friend void swap(GridData<S, DEVS> &grid1, GridData<S, DEVS> &grid2);
};

// Initialize allocated memory to zero.
// TODO(niclas): This is a bit ugly!
template <typename T, typename DEV>
size_t GridData<T, DEV>::_allocatedMemTotal = 0;

template <typename T, typename DEV>
GridData<T, DEV>::GridData()
    : GridFunctions<T, DEV>(this), _dimX(0), _dimY(0), _numFields(0) {}

template <typename T, typename DEV>
GridData<T, DEV>::GridData(const GridData<T, DEV> &other)
    : GridFunctions<T, DEV>(this), _representation(other._representation),
      _dimX(other._dimX), _dimY(other._dimY), _numFields(other._numFields),
      m_vector(other.m_vector) {
  _allocatedMemTotal += numElements() * sizeof(T);
}

template <typename T, typename DEV>
GridData<T, DEV>::GridData(const GridData<T, DEV> &other, const int numFields)
    : GridFunctions<T, DEV>(this), _representation(other._representation),
      _dimX(other._dimX), _dimY(other._dimY), _numFields(numFields),
      m_vector(numElements()) {
  _allocatedMemTotal += numElements() * sizeof(T);
}

template <typename T, typename DEV>
GridData<T, DEV>::GridData(int nx, int ny, int numFields,
                           Type::type representation)
    : GridFunctions<T, DEV>(this), _representation(representation), _dimX(nx),
      _dimY(ny), _numFields(numFields), m_vector(numElements()) {
  _allocatedMemTotal += numElements() * sizeof(T);
}

template <typename T, typename DEV> GridData<T, DEV>::~GridData() {
  _allocatedMemTotal -= numElements() * sizeof(T);
}

template <typename T, typename DEV>
bool GridData<T, DEV>::operator==(const GridData<T, DEV> &right) const {
  return (_dimX == right._dimX && _dimY == right._dimY &&
          _numFields == right._numFields &&
          _representation == right._representation);
}

template <typename T, typename DEV>
bool GridData<T, DEV>::operator!=(const GridData<T, DEV> &right) const {
  return !operator==(right);
}

template <typename T, typename DEV>
GridData<T, DEV> &GridData<T, DEV>::operator=(GridData<T, DEV> right) {
  swap(*this, right);
  return *this;
}

template <typename T, typename DEV>
GridData<T, DEV> &GridData<T, DEV>::operator=(const thrust::complex<T> right) {
  if (isComplex()) {
    setValue(right.real(), Type::real);
    setValue(right.imag(), Type::imag);
  } else {
#ifndef NDEBUG
    std::cout << "Warning: Attempted to fill a real valued grid to a complex "
                 "valued constant! Ignored.\n";
#endif
  }
  return *this;
}

template <typename T, typename DEV>
GridData<T, DEV> &GridData<T, DEV>::operator=(const T right) {
  setValue(right);
  return *this;
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator+=(const GridData<T, DEV> &right) {
  if (*this == right) {
    thrust::transform(m_vector.begin(), m_vector.end(), right.m_vector.begin(),
                      m_vector.begin(), thrust::plus<T>());
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator+=(const thrust::complex<T> right) {
  // This assumes that the calling object is complex, rightwise new memory for
  // the imaginary part would have to be allocated, and it would not be
  // coallesced properly
  if (isComplex()) {
    for (int c = 0; c < numFields(); ++c) {
      thrust::for_each(
          thrust::make_zip_iterator(thrust::make_tuple(
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::real, Type::begin),
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::imag, Type::begin))),
          thrust::make_zip_iterator(thrust::make_tuple(
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::real, Type::end),
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::imag, Type::end))),
          addComplexScalarFunctor(right));
    }
  } else {
#ifndef NDEBUG
    std::cout << "Warning: Attempted to add real valued grid with complex "
                 "valued constant! Ignored.\n";
#endif
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator+=(const T right) {
  if (!isComplex()) {
    thrust::constant_iterator<T> value(right);
    thrust::transform(value, value + m_vector.size(), m_vector.begin(),
                      m_vector.begin(), thrust::plus<T>());
  } else {
    const thrust::complex<T> temp_cplx(right, static_cast<T>(0));
    operator+=(temp_cplx);
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator-=(const GridData<T, DEV> &right) {
  if (*this == right) {
    thrust::transform(m_vector.begin(), m_vector.end(), right.m_vector.begin(),
                      m_vector.begin(), thrust::minus<T>());
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator-=(const T right) {
  operator+=(-right);
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator-=(const thrust::complex<T> right) {
  operator+=(-right);
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator*=(const GridData<T, DEV> &right) {
  if (*this == right) {
    // Check if both vectors are complex
    if (isComplex()) {
      // Complex valued multiplication
      for (int c = 0; c < numFields(); ++c) {
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                m_vector.begin() +
                    getFieldComponentOffset(c, Type::real, Type::begin),
                m_vector.begin() +
                    getFieldComponentOffset(c, Type::imag, Type::begin),
                right.m_vector.begin() +
                    right.getFieldComponentOffset(c, Type::real, Type::begin),
                right.m_vector.begin() +
                    right.getFieldComponentOffset(c, Type::imag, Type::begin))),
            thrust::make_zip_iterator(thrust::make_tuple(
                m_vector.begin() +
                    getFieldComponentOffset(c, Type::real, Type::end),
                m_vector.begin() +
                    getFieldComponentOffset(c, Type::imag, Type::end),
                right.m_vector.begin() +
                    right.getFieldComponentOffset(c, Type::real, Type::end),
                right.m_vector.begin() +
                    right.getFieldComponentOffset(c, Type::imag, Type::end))),
            complexVectorProductFunctor());
      }
    } else {
      // Real valued multiplication
      thrust::transform(m_vector.begin(), m_vector.end(),
                        right.m_vector.begin(), m_vector.begin(),
                        thrust::multiplies<T>());
    }
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator*=(const thrust::complex<T> right) {
  // Only callable if this grid is complex valued
  if (isComplex()) {
    for (int c = 0; c < numFields(); ++c) {
      thrust::for_each(
          thrust::make_zip_iterator(thrust::make_tuple(
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::real, Type::begin),
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::imag, Type::begin))),
          thrust::make_zip_iterator(thrust::make_tuple(
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::real, Type::end),
              m_vector.begin() +
                  getFieldComponentOffset(c, Type::imag, Type::end))),
          scaleComplexVectorFunctor(right));
    }
  } else {
#ifndef NDEBUG
    std::cout << "Warning: Attempted to multiply real valued grid with complex "
                 "valued constant! Ignored.\n";
#endif
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::operator*=(const T right) {
  // This will be valid even if the calling object is complex, because re and im
  // scale the same with real scalar
  thrust::constant_iterator<T> value(right);
  thrust::transform(value, value + m_vector.size(), m_vector.begin(),
                    m_vector.begin(), thrust::multiplies<T>());
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator+(GridData<T, DEV> right) const {
  right += *this;
  return right;
}

template <typename T, typename DEV>
GridData<T, DEV>
GridData<T, DEV>::operator+(const thrust::complex<T> right) const {
  GridData<T, DEV> tmp(*this);
  tmp += right;
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator+(const T right) const {
  GridData<T, DEV> tmp(*this);
  tmp += right;
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator-(GridData<T, DEV> right) const {
  right -= *this;
  return -right;
}

template <typename T, typename DEV>
GridData<T, DEV>
GridData<T, DEV>::operator-(const thrust::complex<T> right) const {
  GridData<T, DEV> tmp(*this);
  tmp -= right;
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator-(const T right) const {
  GridData<T, DEV> tmp(*this);
  tmp -= right;
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator*(GridData<T, DEV> right) const {
  right *= *this;
  return right;
}

template <typename T, typename DEV>
GridData<T, DEV>
GridData<T, DEV>::operator*(const thrust::complex<T> right) const {
  // Not optimal, but arithmetic operators are typically not performance
  // critical:
  GridData<T, DEV> tmp(*this); // One read (this) and one write (tmp)
  tmp *= right;                // One read (tmp) and one write (tmp)
  // --> Total two reads and two writes

  // Optimal implentation would be something like:

  // Allocate tmp grid with correct dimensions
  // GridData<T,DEV> tmp( dimX(), dimY(), numFields(), representation() ); // No
  // read or write
  // Then:
  //"some op that will read data from *this and multiplies with right,
  // then writes to tmp. All in one go." // One read (this) and one write (tmp)
  // --> Total one read and one write of an entire grid.

  // The above comment goes for all arithmetic operators starting with a call
  // to the copy constructor, i.e.: GridData<T,DEV> tmp( *this );

  // Return result
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator*(const T right) const {
  GridData<T, DEV> tmp(*this);
  tmp *= right;
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator-() const {
  GridData<T, DEV> tmp(*this);
  tmp *= T(-1.0);
  return tmp;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::operator[](const int field) const {
  GridData<T, DEV> tmp(*this, 1);
  copyFieldTo(&tmp, field, 0);
  return tmp;
}

template <typename T, typename DEV>
template <typename T2, typename DEV2>
GridData<T, DEV>::operator GridData<T2, DEV2>() {
  GridData<T2, DEV2> other(_dimX, _dimY, _numFields, _representation);
  other.getVector() = m_vector;
  return other;
}

template <typename T, typename DEV>
template <typename T2>
GridData<T, DEV>::operator GridData<T2, thrust::device_vector<T2>>() {
  GridData<T2, thrust::device_vector<T2>> other(_dimX, _dimY, _numFields,
                                                _representation);
  GridFunctions<T, DEV>::typeConvert(other);
  return other;
}

template <typename T, typename DEV>
void swap(GridData<T, DEV> &grid1, GridData<T, DEV> &grid2) {
  std::swap(grid1._representation, grid2._representation);
  std::swap(grid1._dimX, grid2._dimX);
  std::swap(grid1._dimY, grid2._dimY);
  std::swap(grid1._numFields, grid2._numFields);
  std::swap(grid1.m_vector, grid2.m_vector);
}

template <typename T, typename DEV>
void GridData<T, DEV>::setValue(const T val, Type::type part, const int field) {
  // Make sure we don't try to set elements when no memory is allocated.
  if (!m_vector.empty()) {
    if (field == -1) {
      // Set all elements
      if (part == Type::complex)
        thrust::fill(m_vector.begin(), m_vector.end(), val);
      else {
        // Set real or imag parts for all fields
        for (int c = 0; c < _numFields; ++c) {
          thrust::fill(
              m_vector.begin() + getFieldComponentOffset(c, part, Type::begin),
              m_vector.begin() + getFieldComponentOffset(c, part, Type::end),
              val);
        }
      }
    } else {
      // Set real part, if requested, for specified field
      if (part == Type::real || part == Type::complex)
        thrust::fill(m_vector.begin() + getFieldComponentOffset(
                                            field, Type::real, Type::begin),
                     m_vector.begin() +
                         getFieldComponentOffset(field, Type::real, Type::end),
                     val);

      // Set imag part, if requested and if existing, for specified field
      if (isComplex() && (part == Type::imag || part == Type::complex))
        thrust::fill(m_vector.begin() + getFieldComponentOffset(
                                            field, Type::imag, Type::begin),
                     m_vector.begin() +
                         getFieldComponentOffset(field, Type::imag, Type::end),
                     val);
    }
  }
}

template <typename T, typename DEV> void GridData<T, DEV>::setZero() {
  setValue(T(0.0));
}

template <typename T, typename DEV>
int GridData<T, DEV>::setData(const T *p_host_data, size_t numElements) {
  size_t numElements_this = this->numElements();

  if (numElements > numElements_this)
    numElements = numElements_this;

  thrust::host_vector<T> data_tmp(numElements);
  T *data_ptr = thrust::raw_pointer_cast(data_tmp.data());

  for (size_t i = 0; i < numElements; ++i) {
    data_ptr[i] = p_host_data[i];
  }

  m_vector = data_tmp;

  return 0;
}

template <typename T, typename DEV>
GridData<T, DEV> &GridData<T, DEV>::conjugate() {
  if (isComplex()) {
    for (int i = 0; i < numFields(); ++i) {
      thrust::transform(m_vector.begin() +
                            getFieldComponentOffset(i, Type::imag, Type::begin),
                        m_vector.begin() +
                            getFieldComponentOffset(i, Type::imag, Type::end),
                        m_vector.begin() +
                            getFieldComponentOffset(i, Type::imag, Type::begin),
                        thrust::negate<T>());
    }
  }
  return *this;
}

template <typename T, typename DEV>
GridData<T, DEV> &GridData<T, DEV>::transpose(const int field) {
  return GridFunctions<T, DEV>::transpose(field);
}

template <typename T, typename DEV> GridData<T, DEV> &GridData<T, DEV>::sqrt() {
  thrust::for_each_n(m_vector.begin(), numElements(),
                     GridFunctors::squareRoot<T>());
  return *this;
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::abs() const {
  GridData<T, DEV> result;

  if (isComplex())
    result = complexMagnitudeSq().sqrt();
  else {
    result = *this;
    thrust::for_each_n(result.getVector().begin(), result.numElements(),
                       GridFunctors::absVal<T>());
  }

  return result;
}

// Calculate the sum of all magnitudes of a GridData object. This amounts to
// first calculating the element-wise absolute value (hypotenuse), then
// performing a reduction to sum the absolute value in each coordinate into a
// scalar. Note that the first two steps are done in a single
// thrust::transform_reduce call.
template <typename T, typename DEV> T GridData<T, DEV>::sumMagnitude() const {
  // Initial value for the reduction: sumMagnitude = initValue + sum(...)
  const T initValue = static_cast<T>(0.0);

  // The binary operator to sum (reduce) all the magnitudes
  thrust::plus<T> binaryOperatorPlus;

  // Check the kind of field: complex or real
  if (isComplex()) {
    T resultSum = static_cast<T>(0.0);

    for (int fieldIdx = 0; fieldIdx < _numFields; ++fieldIdx) {
      // Pointers to iterator start and end for real part
      const auto realPartIteratorBegin =
          m_vector.begin() +
          getFieldComponentOffset(fieldIdx, Type::real, Type::begin);
      const auto realPartIteratorEnd =
          m_vector.begin() +
          getFieldComponentOffset(fieldIdx, Type::real, Type::end);

      // Pointers to iterator start and end points for imaginary part
      const auto imagPartIteratorBegin =
          m_vector.begin() +
          getFieldComponentOffset(fieldIdx, Type::imag, Type::begin);
      const auto imagPartIteratorEnd =
          m_vector.begin() +
          getFieldComponentOffset(fieldIdx, Type::imag, Type::end);

      // Make zip iterators of real/imaginary parts (transform_reduce takes a
      // single start/end pointer, but we want to send two: zip_iterator solves
      // this by combining two into one)
      const auto zipIteratorBegin = thrust::make_zip_iterator(
          thrust::make_tuple(realPartIteratorBegin, imagPartIteratorBegin));
      const auto zipIteratorEnd = thrust::make_zip_iterator(
          thrust::make_tuple(realPartIteratorEnd, imagPartIteratorEnd));

      // Sum the magnitude in each coordinate of the calling GridData object
      resultSum += thrust::transform_reduce(zipIteratorBegin, zipIteratorEnd,
                                            hypotFunctor(), initValue,
                                            binaryOperatorPlus);
    }

    return resultSum;
  } else {
    // Pointers to iterator start and end for the first component
    const int xComponentOffsetIdx = 0;
    const auto xComponentIteratorBegin =
        m_vector.begin() +
        getFieldComponentOffset(xComponentOffsetIdx, Type::real, Type::begin);
    const auto xComponentIteratorEnd =
        m_vector.begin() +
        getFieldComponentOffset(xComponentOffsetIdx, Type::real, Type::end);

    // Check the kind of field: scalar, vector, unknown
    if (_numFields == 1) {
      // Scalar field: Sum the absolute value in each coordinate
      const T resultSum = thrust::transform_reduce(
          xComponentIteratorBegin, xComponentIteratorEnd, absFunctor(),
          initValue, binaryOperatorPlus);
      return resultSum;
    } else if (_numFields == 2) {
      // Pointer to start and end points for y-component
      const int yComponentOffsetIdx = 1;
      const auto yComponentIteratorBegin =
          m_vector.begin() +
          getFieldComponentOffset(yComponentOffsetIdx, Type::real, Type::begin);
      const auto yComponentIteratorEnd =
          m_vector.begin() +
          getFieldComponentOffset(yComponentOffsetIdx, Type::real, Type::end);

      // Make zip iterators of (x,y)-components (transform_reduce takes a
      // single start/end pointer, but we want to send two: zip_iterator solves
      // this by combining two into one)
      const auto zipIteratorBegin = thrust::make_zip_iterator(
          thrust::make_tuple(xComponentIteratorBegin, yComponentIteratorBegin));
      const auto zipIteratorEnd = thrust::make_zip_iterator(
          thrust::make_tuple(xComponentIteratorEnd, yComponentIteratorEnd));

      // Vector field: Sum the hypotenuse in each coordinate
      const T resultSum = thrust::transform_reduce(
          zipIteratorBegin, zipIteratorEnd, hypotFunctor(), initValue,
          binaryOperatorPlus);

      return resultSum;
    } else {
#ifndef NDEBUG
      // Unknown field: Print warning and return 0.
      // TODO(patric): Raise exception
      std::cout << "-- WARNING! No defined behavior for sumMagnitude for real "
                   "field with "
                << _numFields << " number of fields. Function call ignored."
                << std::endl;
#endif
      return static_cast<T>(0.0);
    }
  }
}

// Calculate the sum of all squares of a GridData object. This amounts to
// first calculating the element-wise absolute square, then performing a
// reduction to sum the square in each coordinate into a scalar. Note that the
// first two steps are done in a single thrust::transform_reduce call.
template <typename T, typename DEV> T GridData<T, DEV>::sumSquare() const {
  // Initial value for the reduction: sumSquare = initValue + sum(...)
  const T initValue = static_cast<T>(0.0);

  // The unary operator to square each element
  thrust::square<T> unaryOperatorSquare;

  // The binary operator to sum (reduce) all the squares
  thrust::plus<T> binaryOperatorPlus;

  // Start iterator: pointer to the first element of the first field
  const auto startIterator = m_vector.begin();

  // Start iterator: pointer to the last element of the last field
  const auto endIterator = m_vector.end();

  const T resultSum =
      thrust::transform_reduce(startIterator, endIterator, unaryOperatorSquare,
                               initValue, binaryOperatorPlus);

  return resultSum;
}

// Calculate the L1 Norm of a GridData object.
template <typename T, typename DEV> T GridData<T, DEV>::L1Norm() const {
  // Initial value for the reduction
  const T initValue = static_cast<T>(0.0);

  // The binary operator to sum (reduce) all the squares
  thrust::plus<T> binaryOperatorPlus;

  // Start iterator: pointer to the first element of the first field
  const auto startIterator = m_vector.begin();

  // Start iterator: pointer to the last element of the last field
  const auto endIterator = m_vector.end();

  const T resultNorm = thrust::transform_reduce(
      startIterator, endIterator, absFunctor(), initValue, binaryOperatorPlus);
  return resultNorm;
}

// Calculate the L2 Norm of a GridData object.
template <typename T, typename DEV> T GridData<T, DEV>::L2Norm() const {
  // Calculate the L2 Norm using the elementwise sum of squares
  const T resultNorm = std::sqrt(this->sumSquare());
  return resultNorm;
}

template <typename T, typename DEV>
void GridData<T, DEV>::complexMagnitudeSq(GridData<T, DEV> *result) const {
  if (GridData<T, DEV>::dimEq(*this, *result) && isComplex()) {
    GridFunctions<T, DEV>::complexMagnitudeSq(result);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::complexMagnitudeSq(): Grids not of equal "
                 "dimensions, or source field not complex.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::complexMagnitudeSq() const {
  GridData<T, DEV> result(_dimX, _dimY, _numFields, Type::real);
  complexMagnitudeSq(&result);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::complexPhase(GridData<T, DEV> *result) const {
  if (GridData<T, DEV>::dimEq(*this, *result) && isComplex()) {
    GridFunctions<T, DEV>::complexPhase(result);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::complexPhase(): Grids not of equal "
                 "dimensions, or source grid not complex\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::complexPhase() const {
  GridData<T, DEV> result(_dimX, _dimY, _numFields, Type::real);
  complexPhase(&result);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::complexPhaseGradient(GridData<T, DEV> *result,
                                            const T h) const {
  if (GridData<T, DEV>::dimEq(*this, *result) && isComplex() &&
      result->numFields() == 2) {
    GridFunctions<T, DEV>::complexPhaseGradient(result, h);
  } else {
#ifndef NDEBUG
    std::cout
        << "-- ERROR! GridData::complexPhaseGradient(): Grids not of equal "
           "dimensions, or source grid not complex\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::complexPhaseGradient(const T h) const {
  GridData<T, DEV> result(_dimX, _dimY, _numFields * 2, Type::real);
  complexPhaseGradient(&result, h);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::fieldMagnitude(GridData<T, DEV> *result) const {
  if (GridData<T, DEV>::dimEq(*this, *result)) {
    GridFunctions<T, DEV>::fieldMagnitude(result);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::fieldMagnitude(): Grids not of equal "
                 "dimensions.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::fieldMagnitude() const {
  GridData<T, DEV> result(_dimX, _dimY, 1, Type::real);
  fieldMagnitude(&result);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::div(GridData<T, DEV> *result, const T h) const {
  if (GridData<T, DEV>::dimEq(*this, *result) && _numFields == 2) {
    GridFunctions<T, DEV>::div(result, h);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::fieldMagnitude(): Grids not of equal "
                 "dimensions.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::div(const T h) const {
  GridData<T, DEV> result(_dimX, _dimY, 1, Type::real);
  result.setZero();
  div(&result, h);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::curl(GridData<T, DEV> *result, const T h) const {
  if (GridData<T, DEV>::dimEq(*this, *result) && _numFields == 2) {
    GridFunctions<T, DEV>::curl(result, h);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::fieldMagnitude(): Grids not of equal "
                 "dimensions.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::curl(const T h) const {
  GridData<T, DEV> result(_dimX, _dimY, 1, Type::real);
  curl(&result, h);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::del2(GridData<T, DEV> *result, const T h) const {
  if (GridData<T, DEV>::dimEq(*this, *result)) {
    GridFunctions<T, DEV>::del2(result, h);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::fieldMagnitude(): Grids not of equal "
                 "dimensions.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::del2(const T h) const {
  GridData<T, DEV> result(_dimX, _dimY, _numFields, Type::real);
  del2(&result, h);
  return result;
}

template <typename T, typename DEV>
void GridData<T, DEV>::sumFieldComponents(GridData<T, DEV> *result) const {
  if (GridData<T, DEV>::dimEq(*this, *result) &&
      isComplex() == result->isComplex()) {
    GridFunctions<T, DEV>::sumFieldComponents(result);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData::sumFieldComponents(): Grids not of equal "
                 "dimensions.\n";
#endif
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::sumFieldComponents() const {
  GridData<T, DEV> result(*this, 1);
  sumFieldComponents(&result);
  return result;
}

template <typename T, typename DEV>
T GridData<T, DEV>::sumField(const int field) const {
  T sum = thrust::reduce(
      m_vector.begin() +
          getFieldComponentOffset(field, Type::real, Type::begin),
      m_vector.begin() + getFieldComponentOffset(field, Type::real, Type::end),
      (T)0, thrust::plus<T>());
  return sum;
}

template <typename T, typename DEV>
void GridData<T, DEV>::sumField(T &sum_re, T &sum_im, const int field) const {
  sum_re = thrust::reduce(
      m_vector.begin() +
          getFieldComponentOffset(field, Type::real, Type::begin),
      m_vector.begin() + getFieldComponentOffset(field, Type::real, Type::end),
      (T)0, thrust::plus<T>());

  if (isComplex()) {
    sum_im = thrust::reduce(
        m_vector.begin() +
            getFieldComponentOffset(field, Type::imag, Type::begin),
        m_vector.begin() +
            getFieldComponentOffset(field, Type::imag, Type::end),
        (T)0, thrust::plus<T>());
  }
}

template <typename T, typename DEV>
GridData<T, DEV> GridData<T, DEV>::sumFields() const {
  GridData<T, DEV> result(numFields(), 1, 1, representation());

  T re_tmp, im_tmp, *re, *im;

  re = result.getDataPointer(0, Type::real);

  if (isComplex())
    im = result.getDataPointer(0, Type::imag);

  for (int c = 0; c < numFields(); ++c) {
    sumField(re_tmp, im_tmp, c);

    re[c] = re_tmp;

    if (isComplex())
      im[c] = im_tmp;
  }

  return GridData(result);
}

template <typename T, typename DEV>
void GridData<T, DEV>::insertGrid(const GridData<T, DEV> *other,
                                  const int x_offset, const int y_offset) {
  GridFunctions<T, DEV>::insertGrid(other, x_offset, y_offset);
}

template <typename T, typename DEV>
void GridData<T, DEV>::insertRotatedGrid(const GridData *other, const T phi,
                                         const int x_offset, const int y_offset,
                                         Type::type part) {
  GridFunctions<T, DEV>::insertRotatedGrid(other, phi, x_offset, y_offset,
                                           part);
}

template <typename T, typename DEV>
void GridData<T, DEV>::insertRotatedGridAdditive(const GridData *other,
                                                 const T phi, int field,
                                                 const int x_offset,
                                                 const int y_offset,
                                                 Type::type part) {
  GridFunctions<T, DEV>::insertRotatedGridAdditive(other, phi, field, x_offset,
                                                   y_offset, part);
}

template <typename T, typename DEV>
void GridData<T, DEV>::resample(const int nx_new, const int ny_new) {
  GridFunctions<T, DEV>::resample(nx_new, ny_new);
}

template <typename T, typename DEV>
void GridData<T, DEV>::resample(GridData *&target, const int nx_new,
                                const int ny_new) const {
  GridFunctions<T, DEV>::resample(target, nx_new, ny_new);
}

template <typename T, typename DEV>
void GridData<T, DEV>::copyFieldTo(GridData<T, DEV> *target, int thisField,
                                   int otherField) const {
  if (GridData<T, DEV>::dimEq(*this, *target) && thisField < numFields() &&
      otherField < target->numFields()) {
    size_t src_begin = getFieldComponentOffset(thisField);
    size_t src_end = getFieldComponentOffset(thisField + 1);
    size_t dst_begin = getFieldComponentOffset(otherField);

    thrust::copy(m_vector.begin() + src_begin, m_vector.begin() + src_end,
                 target->getVector().begin() + dst_begin);
  } else {
#ifndef NDEBUG
    std::cout << "-- ERROR! GridData<T,DEV>::copyFieldTo(): x and y dimensions "
                 "are not equal, or field out of range.\n";
#endif
  }
}

template <typename T, typename DEV>
int GridData<T, DEV>::initializeGrid(GridData<T, DEV> *&grid, int nx, int ny,
                                     int numFields, Type::type representation) {
  int init = 0;

  if (grid && (grid->dimX() != nx || grid->dimY() != ny ||
               grid->numFields() != numFields ||
               grid->representation() != representation)) {
    delete grid;
    grid = 0;
  }

  if (grid == 0) {
    grid = new GridData<T, DEV>(nx, ny, numFields, representation);
    init = 1;
  }

  return init;
}

template <typename T, typename DEV>
template <typename T2, typename DEV2>
int GridData<T, DEV>::initializeGrid(GridData<T, DEV> *&grid,
                                     const GridData<T2, DEV2> *other) {
  return initializeGrid(grid, other->dimX(), other->dimY(), other->numFields(),
                        other->representation());
}

template <typename T, typename DEV>
void GridData<T, DEV>::deleteGrid(GridData<T, DEV> *&grid) {
  if (grid) {
    delete grid;
    grid = 0;
  }
}

template <typename T, typename DEV>
bool GridData<T, DEV>::dimEq(const GridData<T, DEV> &A,
                             const GridData<T, DEV> &B) {
  return (A._dimX == B._dimX && A._dimY == B._dimY);
}

template <typename T, typename DEV>
T GridData<T, DEV>::differenceMagnitude(const GridData<T, DEV> &A,
                                        const GridData<T, DEV> &B) {
  if (A != B) {
#ifndef NDEBUG
    std::cout << "GridData::differenceMagnitude(): Error: Grids have different "
                 "dimensions.\n";
#endif
    return 1000000.0;
  }

  T sumDiff, sumOld;

  if (A.isComplex()) {
    sumDiff = (A - B).complexMagnitudeSq().sumField();
    sumOld = A.complexMagnitudeSq().sumField();
  } else if (A.numFields() > 1) {
    sumDiff = (A - B).fieldMagnitude().sumField();
    sumOld = A.fieldMagnitude().sumField();
  } else {
    sumDiff = (A - B).sumField();
    sumOld = A.sumField();
  }

  if (sumOld < 0.000001)
    sumOld = 0.000001;

  // Compiler dependent namespace issues?
  return std::sqrt(sumDiff) / std::sqrt(sumOld);
}

template <typename T, typename DEV>
void GridData<T, DEV>::minmax(T &minv, T &maxv, int field,
                              Type::type part) const {
  GridFunctions<T, DEV>::minmax(*this, minv, maxv, field, part);
}

template <typename T, typename DEV>
void GridData<T, DEV>::print(int field, Type::type representation) const {
  thrust::host_vector<T> vec = m_vector;

  T *p_data = thrust::raw_pointer_cast(vec.data()) +
              getFieldComponentOffset(field, representation);

  const int Nx = dimX();
  const int Ny = dimY();

  int i, j, idx;

  for (j = 0; j < Ny; ++j) {
    for (i = 0; i < Nx; ++i) {
      idx = j * Nx + i;
      if (i < Nx - 1)
        std::cout << p_data[idx] << ", ";
      else
        std::cout << p_data[idx];
    }
    std::cout << std::endl;
  }
}

template <typename T, typename DEV>
void GridData<T, DEV>::print(int field) const {
  std::cout << "Field = " << field << "\nReal:\n";
  print(field, Type::real);

  if (representation() == Type::complex) {
    std::cout << "\nImaginary:\n";
    print(field, Type::imag);
  }
  std::cout << std::endl;
}

template <typename T, typename DEV> void GridData<T, DEV>::print() const {
  for (int i = 0; i < numFields(); ++i)
    print(i);
}

template <typename T, typename DEV>
void GridData<T, DEV>::printDimensions() const {
  std::cout << "Nx = " << dimX() << ", Ny = " << dimY();
  std::cout << ", Num fields = " << numFields()
            << ", Complex = " << int(isComplex()) << std::endl;
}

/* ==================== FILE ====================*/

template <typename T, typename DEV>
int GridData<T, DEV>::read(const std::filesystem::path &filename, int num,
                           Parameters<T> *param) {
  char buf[256];
  sprintf(buf, "%s.%d.grd", filename, num);

  return read(buf, param);
}

template <typename T, typename DEV>
int GridData<T, DEV>::write(const std::filesystem::path &filename, int num,
                            Parameters<T> *param) const {
  char buf[256];
  sprintf(buf, "%s.%d.grd", filename, num);

  return write(buf, param);
}

template <typename T, typename DEV>
int GridData<T, DEV>::read(const std::filesystem::path &filename,
                           Parameters<T> *param) {
  // Data header structure: (all int datatype)
  // size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
  size_t numBytesHeader = 32 * sizeof(int);
  size_t numBytesData_old = numElements() * sizeof(T);
  size_t numBytesData;

  Parameters<T> param_read;

  int storeParameters, parametersVersion;

  std::ifstream file(filename, std::ios::in | std::ios::binary);

  if (file.is_open()) {
    int *header = new int[32];

    file.read((char *)header, 32 * sizeof(int));

    int sizeOfInputDatatype = header[0];

    if (sizeOfInputDatatype != sizeof(T)) {
      std::cout << "Error: datatype to be read is not the same as instantiated "
                   "class\n";
      file.close();
      return 1;
    }

    int isComplex = header[1];
    _dimX = header[2];
    _dimY = header[3];
    _numFields = header[5];

    storeParameters = header[6];
    parametersVersion = header[7];

    if (isComplex == 0)
      _representation = Type::real;
    if (isComplex == 1)
      _representation = Type::complex;

    numBytesData = numElements() * sizeof(T);

    GridData<T, DEV>::_allocatedMemTotal -= numBytesData_old;
    GridData<T, DEV>::_allocatedMemTotal += numBytesData;

    // Read data to host memory
    thrust::host_vector<T> data_tmp(numElements());
    T *data_ptr = thrust::raw_pointer_cast(data_tmp.data());

    // Go past header
    file.seekg(numBytesHeader);

    // Read parameters
    param_read.setInvalid();

    if (storeParameters == 1) {
      param_read.read(file, parametersVersion);
      param_read.setValid();
    }

    file.read((char *)data_ptr, numBytesData);

    file.close();
    std::cout << "Read: " << filename << "\n";

    if (param) {
      *param = param_read;
    }

    m_vector = data_tmp;

    return 0;
  } else {
    std::cout << "Could not open file: " << filename << "\n";
    return 1;
  }
}

template <typename T, typename DEV>
int GridData<T, DEV>::write(const std::filesystem::path &filename,
                            Parameters<T> *param) const {
  // Data header structure: (all int datatype)
  // size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
  size_t numBytesHeader = 32 * sizeof(int);

  size_t numBytesData, numBytesParameters; //, numBytesPerField;
  numBytesData = numElements() * sizeof(T);
  numBytesParameters = sizeof(Parameters<T>);

  int storeParameters = 0, parametersVersion = 0;
  if (param != 0) {
    storeParameters = 1;
    parametersVersion = param->getVersion();
  }

  int isComplex_out = int(isComplex());

  // Open file for writing
  std::ofstream file(filename,
                     std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open()) {
    // copyDataToCPU();
    thrust::host_vector<T> data_tmp = m_vector;
    T *data_ptr = thrust::raw_pointer_cast(data_tmp.data());

    int sizeOfDatatype = sizeof(T);
    int _dimZ = 1;
    // Write header info
    file.write((char *)(&sizeOfDatatype), sizeof(int));
    file.write((char *)(&isComplex_out), sizeof(int));
    file.write((char *)(&_dimX), sizeof(int));
    file.write((char *)(&_dimY), sizeof(int));
    file.write((char *)(&_dimZ), sizeof(int)); // legacy stuff
    file.write((char *)(&_numFields), sizeof(int));

    file.write((char *)(&storeParameters), sizeof(int));
    file.write((char *)(&parametersVersion), sizeof(int));
    file.write((char *)(&numBytesParameters), sizeof(int));

    file.seekp(numBytesHeader);

    // Write parameters
    if (storeParameters)
      file.write((char *)param, numBytesParameters);

    // Write grid data
    file.write((char *)data_ptr, numBytesData);

    file.close();
    return 0;
  } else {
    std::cout << "Could not open file: " << filename << "\n";
    return 1;
  }
}

template <typename T, typename DEV>
int GridData<T, DEV>::readNumpyArray(const std::filesystem::path &filename,
                                     int numFields) {
  const int MAX_TOKENS_PER_LINE = 16384;

  // create a file-reading object
  std::ifstream fin;
  fin.open(filename);

  if (!fin.good())
    return 1;

  // Determine grid dimensions (Nx)
  int Nx, Ny, Nf;

  int n_delimit = 0, found = 0, found_last = 1;

  std::string line;
  getline(fin, line);

  while (line.find('#', 0) <= line.length() || line.length() == 0)
    getline(fin, line);

  while (found != found_last) {
    found_last = found;
    found = line.find(' ', found_last + 1);

    if (found >= line.length())
      break;

    n_delimit++;
  }

  Nx = n_delimit;

  // Determine grid dimensions (Ny,Nf)
  int lines_data = 0;

  fin.clear();
  fin.seekg(0);

  while (!fin.eof()) {
    getline(fin, line);

    if (line.find('#', 0) >= line.length()) {
      if ((int)line[0] > 0)
        lines_data++;
    }
  }

  if (numFields > 0) {
    Ny = lines_data / numFields;
    Nf = numFields;
  } else {
    Ny = 1;
    Nf = lines_data;
  }

  // Create grid
  size_t numBytesData_old = numElements() * sizeof(T);

  _dimX = Nx;
  _dimY = Ny;
  _numFields = Nf;
  _representation = Type::real;

  thrust::host_vector<T> data_tmp(numElements());

  // Rewind
  fin.clear();
  fin.seekg(0);

  // Read file contents
  T *data_ptr = thrust::raw_pointer_cast(data_tmp.data());
  int j = 0;

  while (!fin.eof()) {
    getline(fin, line);

    int idx_delimit[MAX_TOKENS_PER_LINE];
    n_delimit = 0;

    found = 0;
    found_last = 1;

    if (line.find('#', 0) >= line.length() && line.length() > 0) {
      idx_delimit[n_delimit++] = -1;

      while (found != found_last) {
        found_last = found;
        found = line.find(' ', found_last + 1);

        if (found >= line.length())
          break;

        idx_delimit[n_delimit++] = found;
      }

      T subfloat;
      for (int i = 0; i < n_delimit - 1; ++i) {
        int start = idx_delimit[i] + 1;
        int num = idx_delimit[i + 1] - start;
        std::string sub = line.substr(start, num);

        std::stringstream tmp;
        tmp << sub;
        tmp >> subfloat;
        data_ptr[j++] = subfloat;
      }
    }
  }

  fin.close();

  m_vector = data_tmp;

  GridData<T, DEV>::_allocatedMemTotal -= numBytesData_old;
  GridData<T, DEV>::_allocatedMemTotal += numElements() * sizeof(T);

  return 0;
}

template <typename T, typename DEV>
int GridData<T, DEV>::writeNumpyArray(const std::filesystem::path &filename,
                                      Type::type representation) const {
  // Open file for writing
  std::ofstream file(filename,
                     std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open()) {
    GridData<T, thrust::host_vector<T>> grid_tmp = *this;

    T *data_re, *data_im, val_re, val_im;

    int n, idx;
    char tmp[64];
    char *sig;

    int writeComplex = int(isComplex() && representation == Type::complex);

    n = sprintf(tmp, "# Nx=%d, Ny=%d, Nf=%d, complex=%d\n", _dimX, _dimY,
                _numFields, writeComplex);
    file.write(tmp, n);

    for (int c = 0; c < _numFields; ++c) {
      if (writeComplex) {
        data_re = grid_tmp.getDataPointer(c, Type::real);
        data_im = grid_tmp.getDataPointer(c, Type::imag);

        for (int j = 0; j < _dimY; ++j) {
          for (int i = 0; i < _dimX; ++i) {
            idx = j * _dimX + i;

            val_re = data_re[idx];
            val_im = data_im[idx];

            if (val_im >= (T)0.0)
              sig = (char *)"+";
            else
              sig = (char *)"";

            n = sprintf(tmp, "(%.10e%s%.10ej) ", double(val_re), sig,
                        double(val_im));
            file.write(tmp, n);
          }
          file.write((char *)"\n", 1);
        }
      } else {
        data_re = grid_tmp.getDataPointer(c, Type::real);

        for (int j = 0; j < _dimY; ++j) {
          for (int i = 0; i < _dimX; ++i) {
            idx = j * _dimX + i;

            val_re = data_re[idx];

            n = sprintf(tmp, "%.10e ", double(val_re));
            file.write(tmp, n);
          }
          file.write((char *)"\n", 1);
        }
      }
      file.write((char *)"\n", 1);
    }

    file.close();
    std::cout << "Wrote Ascii in numpy format: " << filename << "\n";
    return 0;
  } else {
    std::cout << "Could not open file: " << filename << "\n";
    return 1;
  }
}

template <typename T, typename DEV>
int GridData<T, DEV>::readParameters(const std::filesystem::path &filename,
                                     int num, Parameters<T> *param) {
  char buf[256];
  sprintf(buf, "%s.%d.grd", filename, num);

  return readParameters(buf, param);
}

template <typename T, typename DEV>
int GridData<T, DEV>::readParameters(const std::filesystem::path &filename,
                                     Parameters<T> *param) {
  // Data header structure: (all int datatype)
  // size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
  size_t numBytesHeader = 32 * sizeof(int);
  int storeParameters, parametersVersion; //,numBytesParameters;

  Parameters<T> param_read;

  std::ifstream file(filename, std::ios::in | std::ios::binary);

  if (file.is_open()) {
    int *header = new int[32];

    file.read((char *)header, 32 * sizeof(int));

    int sizeOfInputDatatype = header[0];

    if (sizeOfInputDatatype != sizeof(T)) {
      std::cout << "Error: datatype to be read is not the same as instantiated "
                   "class\n";
      file.close();
      return 1;
    }

    storeParameters = header[6];
    parametersVersion = header[7];

    // Go past header
    file.seekg(numBytesHeader);

    // Read parameters
    param_read.setInvalid();

    if (storeParameters == 1) {
      param_read.read(file, parametersVersion);
      param_read.setValid();
    }

    file.close();

    if (param)
      *param = param_read;

    return 0;
  } else {
    std::cout << "Could not open file: " << filename << "\n";
    return 1;
  }
}
} // namespace conga

#endif // GRID_DATA_H_
