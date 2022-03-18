//===------------------ Gauges.h - Vector potential gauges. ---------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementations of different vector-potential gauges.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GAUGES_H_
#define CONGA_GAUGES_H_

#include <cuda.h>

#include <string>

#ifndef _HD_
#define _HD_ __host__ __device__
#endif

namespace conga {
namespace gauge {
template <typename T> struct Landau {
  _HD_ static void compute(const T inFluxDensity, const T inArea, const T inX,
                           const T inY, T &outVectorPotentialX,
                           T &outVectorPotentialY) {
    outVectorPotentialX = static_cast<T>(0.0);
    outVectorPotentialY = inX * inFluxDensity;
  }

  static std::string type() { return std::string("landau"); }
};

template <typename T> struct Symmetric {
  _HD_ static void compute(const T inFluxDensity, const T inArea, const T inX,
                           const T inY, T &outVectorPotentialX,
                           T &outVectorPotentialY) {
    const T factor = static_cast<T>(0.5) * inFluxDensity;
    outVectorPotentialX = -inY * factor;
    outVectorPotentialY = inX * factor;
  }

  static std::string type() { return std::string("symmetric"); }
};

template <typename T> struct Solenoid {
  _HD_ static void compute(const T inFluxDensity, const T inArea, const T inX,
                           const T inY, T &outVectorPotentialX,
                           T &outVectorPotentialY) {
    // Don't allow it to be zero.
    const T epsilon = static_cast<T>(1e-7);
    const T r2 = max(inX * inX + inY * inY, epsilon);
    const T factor = inFluxDensity * inArea / (static_cast<T>(2.0 * M_PI) * r2);

    outVectorPotentialX = -inY * factor;
    outVectorPotentialY = inX * factor;
  }

  static std::string type() { return std::string("solenoid"); }
};
} // namespace gauge
} // namespace conga

#endif // CONGA_GAUGES_H_
