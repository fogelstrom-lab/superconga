//===------------------ bdg.h - BdG related functions ------------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// BdG related functions, e.g. computing the critical temperature within BdG.
///
//===----------------------------------------------------------------------===//
#ifndef BDG_H_
#define BDG_H_

#include "ComputeResidual.h"
#include "OrderParameterBasis.h"
#include "cplx.h"
#include "defines.h"
#include "dispersion.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace bdg {
namespace internal {
/// \brief Compute softplus(x).
///
/// The definition is softplus(x) = log(1 + exp(x)). It is useful for
/// transforming a constrained optimization problem to an unconstrained one. If
/// given the problem 'minimize f(y) given that y >= 0' one can use the
/// parameterization y = softplus(x), and solve for x instead, which is
/// unconstrained. This function is used to constrain the critical temperature
/// to be positive.
///
/// \param inValue - The softplus argument.
///
/// \return - softplus(inValue).
template <typename T> T softplus(const T inValue) {
  // The following is numerically more stable than just using the mathematical
  // defition, log(1 + exp(x), for large positive values.
  return std::log(static_cast<T>(1) + std::exp(-std::abs(inValue))) +
         std::max(inValue, static_cast<T>(0));
}

/// \brief Compute sigmoid(x).
///
/// The definition is sigmoid(x) = 1/(1 + exp(-x)). The sigmoid is the
/// derivative of the softplus function. The derivative is needed because the
/// critical temperature is computed using Newton's method.
///
/// \param inValue - The sigmoid argument.
///
/// \return - sigmoid(inValue).
template <typename T> T sigmoid(const T inValue) {
  // The following is numerically more stable than just using the mathematical
  // defition, 1/(1 + exp(-x)), for large negative values.
  const T tmp = std::exp(-std::abs(inValue));
  const T denom = static_cast<T>(1) + tmp;
  const T numer = inValue > static_cast<T>(0) ? static_cast<T>(1) : tmp;
  return numer / denom;
}

/// \brief Compute the residual of the critical temperature equation (and its
/// derivative).
///
/// See Eq.(15) in Tomas BdG note. It gives an equation for the critical
/// temperature (Tc):
///
/// 0 = mean(Yk^2 / |ek| * tanh(0.5*|ek|/Tc)) - 2/Vd
///
/// where the mean is over all momenta in the Brillouin zone, Yk is the basis
/// function (normalized over the entire Brillouin zone), ek is the dispersion,
/// and Vd is the pairing potential.
///
/// This function computes the residual r
///
/// r = mean(Yk^2 / |ek| * tanh(0.5*|ek|/Tc)) - 2/Vd
///
/// and the derivative of r with respect to Tc, drdTc. The derivative is needed
/// because the critical temperature is solved using Newton's method.
///
/// \param inCriticalTemperature - The Tc to evaluate at.
/// \param inMomenta - All momenta in the first Brillouin zone.
/// \param inDispersion - The tight-binding dispersion.
/// \param inBasisFunction - The basis function.
/// \param inPotential - The pairing potential.
///
/// \return - The residual and its derivative (r, drdTc).
template <typename T>
std::pair<T, T> computeCriticalTempResidual(
    const T inCriticalTemperature, const std::vector<vector2d<T>> &inMomenta,
    const DispersionTightBinding<T> &inDispersion,
    const BasisFunction<T> &inBasisFunction, const T inPotential) {
  // The residual of the critical temperature equation.
  T r = static_cast<T>(0);

  // The derivative of the residual with respect to the critical temperature.
  T drdTc = static_cast<T>(0);

  // Loop over all momenta in order to compute the mean over the Brillouin zone.
  for (const auto &momentum : inMomenta) {
    // Evaluate the basis function.
    const cplx<T> basisFunction = inBasisFunction.evaluate(momentum);
    const T basisFunctionSquared = basisFunction.cabsSq();

    // Evaluate the dispersion.
    const T absDispersion = std::abs(inDispersion.evaluate(momentum));

    // Argument to tanh.
    const T arg = static_cast<T>(0.5) * absDispersion / inCriticalTemperature;

    // For convenience.
    const T tanh = std::tanh(arg);

    // Add the residual for this particual momentum.
    r += tanh * basisFunctionSquared / absDispersion;

    // The derivative of tanh(arg(Tc)) with respect to Tc.
    const T tanhDerivative =
        (tanh * tanh - static_cast<T>(1)) * arg / inCriticalTemperature;

    // Add the derivative of the residual with respect to the critical
    // temperature for this particular momentum.
    drdTc += tanhDerivative * basisFunctionSquared / absDispersion;
  }

  // Divide by the number of momenta to get the means.
  r /= static_cast<T>(inMomenta.size());
  drdTc /= static_cast<T>(inMomenta.size());

  // Subtract the potential term.
  r -= static_cast<T>(2.0) / inPotential;

  return {r, drdTc};
}
} // namespace internal

/// \brief Compute the BdG bulk critical temperature.
///
/// \param inDispersion - A tight-binding dispersion.
/// \param inPotential - The pairing potential.
/// \param inSymmetry - The basis-function symmetry.
/// \param inNumMomentaPerAxis - The number of momenta along one axis.
/// \param inConvergenceCriterion - Stop when the critical temperature, the
///                                 gradient, or the update, is smaller than
///                                 this.
/// \param inMaxIterations - The maximum number of iterations.
///
/// \return - The critical temperature.
template <typename T>
T computeCriticalTemperature(
    const DispersionTightBinding<T> &inDispersion, const T inPotential,
    const BasisFunctionSymmetry &inSymmetry,
    const int inNumMomentaPerAxis = 100,
    const T inConvergenceCriterion = static_cast<T>(10.0) *
                                     std::numeric_limits<T>::epsilon(),
    const int inMaxIterations = static_cast<int>(1e3)) {

  // The momenta along on axis of the first Brillouin zone.
  const T minVal = -static_cast<T>(M_PI);
  const T maxVal = static_cast<T>(M_PI);
  const bool includeEndPoint = false;
  const std::vector<T> momentaAxis =
      utils::linspace(minVal, maxVal, inNumMomentaPerAxis, includeEndPoint);

  // All the momenta in the first Brillouin zone.
  std::vector<vector2d<T>> momenta;
  for (const T ky : momentaAxis) {
    for (const T kx : momentaAxis) {
      momenta.push_back({kx, ky});
    }
  }

  // Basis function normalized over the Brillouin zone.
  BasisFunction<T> basisFunction(inSymmetry);
  basisFunction.normalize(momenta);

  // In order to enforce that Tc >= 0 we parameterize Tc as
  // Tc = softplus(x) = log(1 + e^x)
  // and solve for x, which is in the range (-\infty, +\infty).
  T x = static_cast<T>(0);

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inMaxIterations; ++iterationIdx) {
    // Form the critical temperature.
    const T Tc = internal::softplus(x);

    //  If Tc is very small we consider ourselves done.
    isConverged = Tc < inConvergenceCriterion;
    if (isConverged) {
      break;
    }

    // Evaluate the residual, and its derivative with respect to Tc.
    const auto tmp = internal::computeCriticalTempResidual(
        Tc, momenta, inDispersion, basisFunction, inPotential);
    // The residual.
    const T r = std::get<0>(tmp);
    // The derivative.
    const T drdTc = std::get<1>(tmp);

    // Compute the derivative with respect to the parameterization of Tc.
    const T dTcdx = internal::sigmoid(x);
    const T drdx = drdTc * dTcdx;

    // If the derivative is very small we are done.
    isConverged = std::abs(drdx) < inConvergenceCriterion;
    if (isConverged) {
      break;
    }

    // The Newton update.
    const T update = -r / drdx;

    // If the update is very small we are also done.
    isConverged = std::abs(update) < inConvergenceCriterion;
    if (isConverged) {
      break;
    }

    // Update x (our parameterization of Tc).
    x += update;
  }

  // TODO(niclas): Raise exception instead!?
  if (!isConverged) {
    std::cout << "WARNING! The BdG bulk critical temperature did not converge!"
              << std::endl;
  }

  // Form the critical temperature.
  return internal::softplus(x);
}
} // namespace bdg

#endif // BDG_H_
