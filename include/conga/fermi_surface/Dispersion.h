//===------------------ dispersion.h - The dispersion class --------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// It contains the dispersion base class, and the classes derived from it, e.g.
/// a tight-binding dispersion on a square lattice.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_DISPERSION_H_
#define CONGA_DISPERSION_H_

#include "../defines.h"

#include <cmath>
#include <iostream>

namespace conga {
/// \brief Dispersion base class.
template <typename T> class Dispersion {
public:
  /// \brief Constructor.
  Dispersion() {}
  virtual ~Dispersion() {}

  /// \brief Evaluate the dispersion.
  ///
  /// \param inMomentum - The Fermi momentum (kx, ky).
  ///
  /// \return - The dispersion value.
  virtual T evaluate(const vector2d<T> &inMomentum) const = 0;

  /// \brief Evaluate the dispersion.
  ///
  /// \param inMomentumX - The Fermi momentum along the x-axis.
  /// \param inMomentumY - The Fermi momentum along the y-axis.
  ///
  /// \return - The dispersion value.
  virtual T evaluate(const T inMomentumX, const T inMomentumY) const = 0;

  /// \brief Analytical derivative of the dispersion.
  ///
  /// I.e. the Fermi velocity.
  ///
  /// \param inMomentum - The Fermi momentum (kx, ky).
  ///
  /// \return - The partial derivatives (de/dkx, de/dky).
  virtual vector2d<T> derivative(const vector2d<T> &inMomentum) const = 0;

  /// \brief Analytical derivative of the dispersion.
  ///
  /// I.e. the Fermi velocity.
  ///
  /// \param inMomentumX - The Fermi momentum along the x-axis.
  /// \param inMomentumY - The Fermi momentum along the y-axis.
  /// \param outVelocityX - The Fermi velocity along the x-axis.
  /// \param outVelocityY - The Fermi velocity along the y-axis.
  ///
  /// \return - Void.
  virtual void derivative(const T inMomentumX, const T inMomentumY,
                          T &outVelocityX, T &outVelocityY) const = 0;
};
} // namespace conga

#endif // CONGA_DISPERSION_H_
