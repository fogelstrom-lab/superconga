//===------------------ dispersion.h - The dispersion class --------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// It contains the dispersion base class, and the classes derived from it, e.g.
/// a tight-binding dispersion on a square lattice.
///
//===----------------------------------------------------------------------===//
#ifndef DISPERSION_H_
#define DISPERSION_H_

#include "defines.h"
#include <cmath>
#include <iostream>

namespace scg {
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

/// \brief Tight-binding dispersion class.
template <typename T> class DispersionTightBinding : Dispersion<T> {
private:
  // Nearest neighbour hopping. It is always 1, all other hopping terms, and the
  // chemical potential, is expressed in terms of it. It is our energy scale.
  const T m_hoppingNN = static_cast<T>(1.0);

  // Next-nearest neighbour.
  const T m_hoppingNNN;

  // Next-next-nearest neighbour.
  const T m_hoppingNNNN;

  // Chemical potential (or rather, Fermi level).
  const T m_chemicalPotential;

public:
  /// \brief Constructor.
  ///
  /// \param inHoppingNNN - The next-nearest neighbour hopping.
  /// \param inHoppingNNNN - The next-next-nearest neighbour hopping.
  /// \param inChemicalPotential - The chemical potential.
  DispersionTightBinding(const T inHoppingNNN, const T inHoppingNNNN,
                         const T inChemicalPotential)
      : Dispersion<T>(), m_hoppingNN(static_cast<T>(1.0)),
        m_hoppingNNN(inHoppingNNN), m_hoppingNNNN(inHoppingNNNN),
        m_chemicalPotential(inChemicalPotential) {}

  /// \brief Evaluate the dispersion.
  ///
  /// \param inMomentum - The Fermi momentum (kx, ky).
  ///
  /// \return - The dispersion value.
  T evaluate(const vector2d<T> &inMomentum) const override {
    return evaluate(inMomentum[0], inMomentum[1]);
  }

  /// \brief Evaluate the dispersion.
  ///
  /// \param inMomentumX - The Fermi momentum along the x-axis.
  /// \param inMomentumY - The Fermi momentum along the y-axis.
  ///
  /// \return - The dispersion value.
  T evaluate(const T inMomentumX, const T inMomentumY) const override {
    // Nearest neighbour.
    const T factorNN = -static_cast<T>(2.0) * m_hoppingNN;
    const T termNN = factorNN * (std::cos(inMomentumX) + std::cos(inMomentumY));

    // Next-nearest neighbour.
    const T factorNNN = -static_cast<T>(4.0) * m_hoppingNNN;
    const T termNNN = factorNNN * std::cos(inMomentumX) * std::cos(inMomentumY);

    // Next-next-nearest neighbour.
    const T factorNNNN = -static_cast<T>(2.0) * m_hoppingNNNN;
    const T termNNNN =
        factorNNNN * (std::cos(static_cast<T>(2.0) * inMomentumX) +
                      std::cos(static_cast<T>(2.0) * inMomentumY));

    return (termNN + termNNN + termNNNN - m_chemicalPotential);
  }

  /// \brief Analytical derivative of the dispersion.
  ///
  /// \param inMomentum - The Fermi momentum (kx, ky).
  ///
  /// \return - The partial derivatives (de/dkx, de/dky).
  vector2d<T> derivative(const vector2d<T> &inMomentum) const override {
    T velocityX;
    T velocityY;
    derivative(inMomentum[0], inMomentum[1], velocityX, velocityY);
    return {velocityX, velocityY};
  }

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
  void derivative(const T inMomentumX, const T inMomentumY, T &outVelocityX,
                  T &outVelocityY) const override {
    // Nearest neighbour.
    const T factorNN = static_cast<T>(2.0) * m_hoppingNN;
    const T termNNdX = factorNN * std::sin(inMomentumX);
    const T termNNdY = factorNN * std::sin(inMomentumY);

    // Next-nearest neighbour.
    const T factorNNN = static_cast<T>(4.0) * m_hoppingNNN;
    const T termNNNdX =
        factorNNN * std::sin(inMomentumX) * std::cos(inMomentumY);
    const T termNNNdY =
        factorNNN * std::cos(inMomentumX) * std::sin(inMomentumY);

    // Next-next-nearest neighbour.
    const T factorNNNN = static_cast<T>(4.0) * m_hoppingNNNN;
    const T termNNNNdX =
        factorNNNN * std::sin(static_cast<T>(2.0) * inMomentumX);
    const T termNNNNdY =
        factorNNNN * std::sin(static_cast<T>(2.0) * inMomentumY);

    outVelocityX = (termNNdX + termNNNdX + termNNNNdX);
    outVelocityY = (termNNdY + termNNNdY + termNNNNdY);
  }
};
} // namespace scg

#endif // DISPERSION_H_
