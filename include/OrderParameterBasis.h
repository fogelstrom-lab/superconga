#ifndef ORDER_PARAMETER_BASIS_H_
#define ORDER_PARAMETER_BASIS_H_

#include "FermiSurface.h"
#include "cplx.h"
#include "defines.h"
#include <cmath>
#include <iostream>
#include <string>

/// \brief Enum class with the supported basis-function symmetries.
///
/// The following values are supported:
/// SWAVE, SWAVE_EXT, DWAVE, PWAVE
enum class BasisFunctionSymmetry { SWAVE, SWAVE_EXT, DWAVE, PWAVE };

/// \brief Basis-function class.
template <typename T> class BasisFunction {
private:
  /// The symmetry of the basis function.
  const BasisFunctionSymmetry m_symmetry;

  /// Factor to ensure that the Fermi-surface average of the squared basis
  /// function is one.
  T m_normalizationFactor;

public:
  /// \brief Constructor.
  ///
  /// \param inSymmetry - BasisFunctionSymmetry enum
  BasisFunction(const BasisFunctionSymmetry inSymmetry);

  /// \brief Get the symmetry as an enum.
  ///
  /// \return - A symmetry enum.
  BasisFunctionSymmetry getSymmetry() const { return m_symmetry; }

  /// \brief Get the symmetry as a string.
  ///
  /// \return - A string e.g. "s"
  std::string getSymmetryStr() const;

  /// \brief Evaluate the basis function.
  ///
  /// \param inMomentum - The Fermi momentum (x,y).
  ///
  /// \return - The basis function value
  cplx<T> evaluate(const vector2d<T> &inMomentum) const;

  /// \brief Evaluate the basis function.
  ///
  /// \param inMomentumX - The x-component of the Fermi momentum
  /// \param inMomentumY - The y-component of the Fermi momentum
  ///
  /// \return - The basis function value
  cplx<T> evaluate(const T inMomentumX, const T inMomentumY) const;

  /// \brief Normalize the basis function on the Fermi surface.
  ///
  /// \param inFermiSurface - The Fermi surface to normalize the basis function
  ///                         over
  ///
  /// \return - void
  void normalize(const FermiSurface<T> *const inFermiSurface);

  /// \brief Normalize the basis function in the first Brillouin zone.
  ///
  /// \param inMomentum - All momenta in the first Brillouin zone.
  ///
  /// \return - void.
  void normalize(const std::vector<vector2d<T>> &inMomentum);
};

template <typename T>
BasisFunction<T>::BasisFunction(const BasisFunctionSymmetry inSymmetry)
    : m_symmetry(inSymmetry), m_normalizationFactor(static_cast<T>(1.0)) {}

template <typename T> std::string BasisFunction<T>::getSymmetryStr() const {
  std::string str;

  switch (m_symmetry) {
  case BasisFunctionSymmetry::SWAVE: {
    str = "s";
    break;
  }
  case BasisFunctionSymmetry::SWAVE_EXT: {
    str = "s_ext";
    break;
  }
  case BasisFunctionSymmetry::DWAVE: {
    str = "d";
    break;
  }
  case BasisFunctionSymmetry::PWAVE: {
    str = "p";
    break;
  }
  default: {
    // TODO(niclas): Raise exception instead!?
    std::cout << "ERROR! Basis-function symmetry not implemented!" << std::endl;
  }
  }
  return str;
}

template <typename T>
cplx<T> BasisFunction<T>::evaluate(const vector2d<T> &inMomentum) const {
  return evaluate(inMomentum[0], inMomentum[1]);
}

template <typename T>
cplx<T> BasisFunction<T>::evaluate(const T inMomentumX,
                                   const T inMomentumY) const {
  cplx<T> basisFunction;

  switch (m_symmetry) {
  case BasisFunctionSymmetry::SWAVE: {
    basisFunction = cplx<T>(static_cast<T>(1.0), static_cast<T>(0.0));
    break;
  }
  case BasisFunctionSymmetry::SWAVE_EXT: {
    basisFunction = cplx<T>(std::cos(inMomentumX) + std::cos(inMomentumY),
                            static_cast<T>(0.0));
    break;
  }
  case BasisFunctionSymmetry::DWAVE: {
    basisFunction = cplx<T>(std::cos(inMomentumX) - std::cos(inMomentumY),
                            static_cast<T>(0.0));
    break;
  }
  case BasisFunctionSymmetry::PWAVE: {
    basisFunction = cplx<T>(std::sin(inMomentumX), std::sin(inMomentumY));
    break;
  }
  default: {
    // TODO(niclas): Raise exception instead!?
    std::cout << "ERROR! Basis-function symmetry not implemented!" << std::endl;
    basisFunction = cplx<T>(static_cast<T>(0.0), static_cast<T>(0.0));
  }
  }

  return basisFunction * m_normalizationFactor;
}

template <typename T>
void BasisFunction<T>::normalize(const FermiSurface<T> *const inFermiSurface) {
  const int numAngles = inFermiSurface->getNumAngles();

  T avgBasisFunctionSquared = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Get Fermi momentum.
    const vector2d<T> momentum = inFermiSurface->getFermiMomentum(angleIdx);

    // Evaluate the basis function.
    const cplx<T> basisFunction = evaluate(momentum);

    // Get integrand factor.
    const T integrandFactor = inFermiSurface->getIntegrandFactor(angleIdx);

    // Accumulate.
    avgBasisFunctionSquared += basisFunction.cabsSq() * integrandFactor;
  }

  // The normalization needs to be updated multiplicatively because the previous
  // value was used in the for loop above (via the 'evaluate' call).
  m_normalizationFactor *=
      static_cast<T>(1.0) / std::sqrt(avgBasisFunctionSquared);
}

template <typename T>
void BasisFunction<T>::normalize(const std::vector<vector2d<T>> &inMomenta) {
  T avgBasisFunctionSquared = static_cast<T>(0.0);
  for (const auto &momentum : inMomenta) {
    // Evaluate the basis function.
    const cplx<T> basisFunction = evaluate(momentum);

    // Accumulate.
    avgBasisFunctionSquared += basisFunction.cabsSq();
  }
  avgBasisFunctionSquared /= static_cast<T>(inMomenta.size());

  // The normalization needs to be updated multiplicatively because the previous
  // value was used in the for loop above (via the 'evaluate' call).
  m_normalizationFactor *=
      static_cast<T>(1.0) / std::sqrt(avgBasisFunctionSquared);
}

#endif // ORDER_PARAMETER_BASIS_H_
