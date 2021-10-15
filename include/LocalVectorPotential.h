//===-- LocalVectorPotential.h - Compute energy from vector potential. ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functions to compute local contribution to energy from a vector potential.
///
//===----------------------------------------------------------------------===//
#ifndef LOCAL_VECTOR_POTENTIAL_H_
#define LOCAL_VECTOR_POTENTIAL_H_

#include <cuda.h>
#include <string>

#ifndef _HD_
#define _HD_ __host__ __device__
#endif

namespace conga {
namespace VectorPotential {
template <typename T> struct None {
  _HD_ static void computeVectorPotential_local(const T inFluxDensity,
                                                const T inArea, const T inX,
                                                const T inY,
                                                T &outVectorPotentialX,
                                                T &outVectorPotentialY);
  static std::string type() { return std::string("none"); }
};

template <typename T> struct LinearCentered {
  _HD_ static void computeVectorPotential_local(const T inFluxDensity,
                                                const T inArea, const T inX,
                                                const T inY,
                                                T &outVectorPotentialX,
                                                T &outVectorPotentialY);
  static std::string type() { return std::string("linear_centered"); }
};

template <typename T> struct Radial {
  _HD_ static void computeVectorPotential_local(const T inFluxDensity,
                                                const T inArea, const T inX,
                                                const T inY,
                                                T &outVectorPotentialX,
                                                T &outVectorPotentialY);
  static std::string type() { return std::string("radial"); }
};

template <typename T> struct Solenoid {
  _HD_ static void computeVectorPotential_local(const T inFluxDensity,
                                                const T inArea, const T inX,
                                                const T inY,
                                                T &outVectorPotentialX,
                                                T &outVectorPotentialY);
  static std::string type() { return std::string("solenoid"); }
};
} // namespace VectorPotential

template <typename T>
_HD_ void VectorPotential::None<T>::computeVectorPotential_local(
    const T inFluxDensity, const T inArea, const T inX, const T inY,
    T &outVectorPotentialX, T &outVectorPotentialY) {
  outVectorPotentialX = static_cast<T>(0.0);
  outVectorPotentialY = static_cast<T>(0.0);
}

template <typename T>
_HD_ void VectorPotential::LinearCentered<T>::computeVectorPotential_local(
    const T inFluxDensity, const T inArea, const T inX, const T inY,
    T &outVectorPotentialX, T &outVectorPotentialY) {
  outVectorPotentialX = static_cast<T>(0.0);
  outVectorPotentialY = inX * inFluxDensity;
}

template <typename T>
_HD_ void VectorPotential::Radial<T>::computeVectorPotential_local(
    const T inFluxDensity, const T inArea, const T inX, const T inY,
    T &outVectorPotentialX, T &outVectorPotentialY) {
  const T factor = static_cast<T>(0.5) * inFluxDensity;
  outVectorPotentialX = -inY * factor;
  outVectorPotentialY = inX * factor;
}

template <typename T>
_HD_ void VectorPotential::Solenoid<T>::computeVectorPotential_local(
    const T inFluxDensity, const T inArea, const T inX, const T inY,
    T &outVectorPotentialX, T &outVectorPotentialY) {
  // TODO(niclas): Where does this come from?

  // Don't allow it to be zero.
  const T epsilon = static_cast<T>(1e-7);
  const T r2 = max(inX * inX + inY * inY, epsilon);
  const T factor = inFluxDensity * inArea / (static_cast<T>(2.0 * M_PI) * r2);

  outVectorPotentialX = -inY * factor;
  outVectorPotentialY = inX * factor;
}
} // namespace conga

#endif // LOCAL_VECTOR_POTENTIAL_H_
