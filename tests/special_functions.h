//===------ special_functions.h - Portable special functions for tests ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The C++17 mathematical special functions (std::cyl_bessel_i, std::riemann_zeta,
/// ...) are provided by libstdc++ but not by libc++ (macOS). The tests only need
/// integer-order modified Bessel functions and zeta(3), so these are implemented
/// here to full double precision with spectrally convergent trapezoidal rules:
///
///   I_n(x) = (1/pi) int_0^pi exp(x cos t) cos(n t) dt
///   K_n(x) = int_0^inf exp(-x cosh t) cosh(n t) dt,  x > 0
///
/// The first integrand is periodic and analytic, the second decays double
/// exponentially, so the trapezoidal rule converges geometrically in both cases.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_TESTS_SPECIAL_FUNCTIONS_H_
#define CONGA_TESTS_SPECIAL_FUNCTIONS_H_

#include <cmath>
#include <stdexcept>

namespace conga {
namespace special {

/// Modified Bessel function of the first kind, integer order n >= 0.
inline double cyl_bessel_i(const int n, const double x) {
  if (n < 0) {
    throw std::invalid_argument("cyl_bessel_i: order must be non-negative");
  }
  if (x == 0.0) {
    return n == 0 ? 1.0 : 0.0;
  }
  // For moderate |x| the ascending series has only positive terms and is
  // free of cancellation; the trapezoidal rule below suffers from it for
  // n > 0 at small x.
  if (std::fabs(x) < 25.0) {
    const double half = 0.5 * x;
    double term = 1.0;
    for (int k = 1; k <= n; ++k) {
      term *= half / k; // (x/2)^n / n!
    }
    double sum = term;
    const double quarterSq = half * half;
    for (int k = 1; k < 500; ++k) {
      term *= quarterSq / (static_cast<double>(k) * (k + n));
      sum += term;
      if (term < 1e-17 * sum) {
        break;
      }
    }
    return sum;
  }
  // Trapezoidal rule on [0, pi]; the number of nodes is chosen so that the
  // rule is converged to double precision for |x| up to a few hundred.
  const int numIntervals = 64 + 4 * static_cast<int>(std::fabs(x));
  const double pi = 3.14159265358979323846;
  const double h = pi / numIntervals;
  double sum = 0.5 * (std::exp(x) + std::exp(-x) * (n % 2 == 0 ? 1.0 : -1.0));
  for (int k = 1; k < numIntervals; ++k) {
    const double t = k * h;
    sum += std::exp(x * std::cos(t)) * std::cos(n * t);
  }
  return sum * h / pi;
}

/// Modified Bessel function of the second kind, integer order n >= 0, x > 0.
inline double cyl_bessel_k(const int n, const double x) {
  if (n < 0) {
    throw std::invalid_argument("cyl_bessel_k: order must be non-negative");
  }
  if (x <= 0.0) {
    throw std::domain_error("cyl_bessel_k: argument must be positive");
  }
  // Trapezoidal rule on [0, T] where exp(-x cosh T) is negligible.
  const double h = 0.02;
  double sum = 0.5 * std::exp(-x); // t = 0 term, cosh(0) = 1.
  for (int k = 1;; ++k) {
    const double t = k * h;
    const double term = std::exp(-x * std::cosh(t)) * std::cosh(n * t);
    sum += term;
    if (term < 1e-18 * sum) {
      break;
    }
  }
  return sum * h;
}

/// Riemann zeta function for real s > 1 (Euler-Maclaurin tail correction).
inline double riemann_zeta(const double s) {
  if (s <= 1.0) {
    throw std::domain_error("riemann_zeta: only s > 1 is implemented");
  }
  const int N = 1000;
  double sum = 0.0;
  for (int k = 1; k < N; ++k) {
    sum += std::pow(static_cast<double>(k), -s);
  }
  const double Nd = static_cast<double>(N);
  sum += std::pow(Nd, 1.0 - s) / (s - 1.0) + 0.5 * std::pow(Nd, -s) +
         s * std::pow(Nd, -s - 1.0) / 12.0 -
         s * (s + 1.0) * (s + 2.0) * std::pow(Nd, -s - 3.0) / 720.0;
  return sum;
}

} // namespace special
} // namespace conga

#endif // CONGA_TESTS_SPECIAL_FUNCTIONS_H_
