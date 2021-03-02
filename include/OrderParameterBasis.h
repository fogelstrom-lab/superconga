#ifndef ORDER_PARAMETER_BASIS_H_
#define ORDER_PARAMETER_BASIS_H_

#include "FermiSurface.h"
#include "FermiSurfaceCircle.h"
#include "FermiSurfaceTightBinding.h"
#include "cplx.h"
#include "defines.h"
#include <cmath>
#include <iostream>
#include <string>

namespace scg {
/// \brief Enum class with the supported basis-function symmetries.
///
/// The following values are supported:
///
/// SWAVE
/// DWAVE_X2_Y2
/// DWAVE_XY
/// GWAVE
enum class BasisFunctionSymmetry { SWAVE, DWAVE_X2_Y2, DWAVE_XY, GWAVE };

/// \brief Basis-function class.
template <typename T> class BasisFunction {
private:
  /// The symmetry of the basis function.
  const BasisFunctionSymmetry m_symmetry;

  /// Factor to ensure that the Fermi-surface average of the squared basis
  /// function is one.
  T m_normalizationFactor;

  // Whether we should use the circular basis-function representation.
  bool m_circularFermiSurface;

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
  /// The appropriate basis-function representation will be deduced from the
  /// Fermi surface.  E.g. d_xy = x * y, will be used for a circular Fermi
  /// surface.
  ///
  /// \param inFermiSurface - The Fermi surface to normalize the basis function
  ///                         over
  ///
  /// \return - void
  void normalize(const FermiSurface<T> *const inFermiSurface);

  /// \brief Normalize the basis function in the first Brillouin zone.
  ///
  /// The general basis-function representation will be used.
  ///
  /// \param inMomentum - All momenta in the first Brillouin zone.
  ///
  /// \return - void.
  void normalize(const std::vector<vector2d<T>> &inMomentum);
};

template <typename T>
BasisFunction<T>::BasisFunction(const BasisFunctionSymmetry inSymmetry)
    : m_symmetry(inSymmetry), m_normalizationFactor(static_cast<T>(1.0)),
      m_circularFermiSurface(true) {}

template <typename T> std::string BasisFunction<T>::getSymmetryStr() const {
  std::string str;

  switch (m_symmetry) {
  case BasisFunctionSymmetry::SWAVE: {
    str = "s";
    break;
  }
  case BasisFunctionSymmetry::DWAVE_X2_Y2: {
    str = "dx2-y2";
    break;
  }
  case BasisFunctionSymmetry::DWAVE_XY: {
    str = "dxy";
    break;
  }
  case BasisFunctionSymmetry::GWAVE: {
    str = "g";
    break;
  }
  default: {
#ifndef NDEBUG
    // TODO(niclas): Raise exception instead!?
    std::cout << "ERROR! Basis-function symmetry not implemented!" << std::endl;
#endif
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

  if (m_circularFermiSurface) {
    switch (m_symmetry) {
    case BasisFunctionSymmetry::SWAVE: {
      basisFunction = cplx<T>(static_cast<T>(1.0), static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::DWAVE_X2_Y2: {
      basisFunction =
          cplx<T>(inMomentumX * inMomentumX - inMomentumY * inMomentumY,
                  static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::DWAVE_XY: {
      basisFunction = cplx<T>(inMomentumX * inMomentumY, static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::GWAVE: {
      basisFunction =
          cplx<T>(inMomentumX * inMomentumY *
                      (inMomentumX * inMomentumX - inMomentumY * inMomentumY),
                  static_cast<T>(0.0));
      break;
    }
    default: {
#ifndef NDEBUG
      // TODO(niclas): Raise exception instead!?
      std::cout << "ERROR! Basis-function symmetry not implemented!"
                << std::endl;
#endif
      basisFunction = cplx<T>(static_cast<T>(0.0), static_cast<T>(0.0));
    }
    }
  } else {
    switch (m_symmetry) {
    case BasisFunctionSymmetry::SWAVE: {
      basisFunction = cplx<T>(std::cos(inMomentumX) + std::cos(inMomentumY),
                              static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::DWAVE_X2_Y2: {
      basisFunction = cplx<T>(std::cos(inMomentumX) - std::cos(inMomentumY),
                              static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::DWAVE_XY: {
      basisFunction = cplx<T>(std::sin(inMomentumX) * std::sin(inMomentumY),
                              static_cast<T>(0.0));
      break;
    }
    case BasisFunctionSymmetry::GWAVE: {
      basisFunction =
          cplx<T>(std::sin(inMomentumX) * std::sin(inMomentumY) *
                      (std::cos(inMomentumX) - std::cos(inMomentumY)),
                  static_cast<T>(0.0));
      break;
    }
    default: {
#ifndef NDEBUG
      // TODO(niclas): Raise exception instead!?
      std::cout << "ERROR! Basis-function symmetry not implemented!"
                << std::endl;
#endif
      basisFunction = cplx<T>(static_cast<T>(0.0), static_cast<T>(0.0));
    }
    }
  }

  return basisFunction * m_normalizationFactor;
}

template <typename T>
void BasisFunction<T>::normalize(const FermiSurface<T> *const inFermiSurface) {
  // Check if this is a circular Fermi surface.
  m_circularFermiSurface =
      inFermiSurface->getType() == FermiSurfaceType::CIRCLE;

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
  // This method is only used for BdG computations, i.e. the circular
  // basis-function representation is not appropriate.
  m_circularFermiSurface = false;

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
} // namespace scg

#endif // ORDER_PARAMETER_BASIS_H_
