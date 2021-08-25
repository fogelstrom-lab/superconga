//===------ defines.h - Application-level (global) definitions. -----------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains definitions (constants, functions etcetera) that should
/// be accessible on an application level (global level).
///
//===----------------------------------------------------------------------===//
#ifndef DEFINES_H_
#define DEFINES_H_

#include <array>
#include <cmath>

namespace scg {
// Convenient 2D vector definition.
template <typename T> using vector2d = std::array<T, 2>;

namespace constants {
// Margin around superconducting grain in simulation space, in pixels.
const int GRAIN_SIZE_MARGIN = 3;

// Size of coordinate system where grain is allowed to be, as seen by the user:
// [-MAXIMUM_COORDINATE, +MAXIMUM_COORDINATE], used when specifying size
// and coordinates of of discs/polygons.
// Maximum grain "radius": MAXIMUM_COORDINATE
// Maximum grain "diameter": 2*MAXIMUM_COORDINATE
const double MAXIMUM_COORDINATE = 0.5;

// The Euler-Mascheroni constant, gamma. Value taken from Wikipedia:
// https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant
const double EULER_MASCHERONI = 0.57721566490153286060651209008240243;

// The bulk s-wave order parameter at zero temperature has a nice analytical
// expression. See page 104 of P. Holmvall PhD Thesis.
// Note that we are using 2*pi*kB*Tc as our energy scale.
const double S_WAVE_BULK = 0.5 * std::exp(-EULER_MASCHERONI);

// The bulk d-wave order parameter at zero temperature has a nice analytical
// expression. See page 104 of P. Holmvall PhD Thesis.
// Note that we are using 2*pi*kB*Tc as our energy scale.
const double D_WAVE_BULK = std::exp(-EULER_MASCHERONI - 0.5) / std::sqrt(2.0);

// The value of the Riemann zeto function evaluated at 3, i.e. Apery's constant
// = zeta(3).
const double APERY = std::riemann_zeta(3);

// The sign of the elementary charge 'e', i.e. SIGN_E = e / |e|.
const double SIGN_E = 1.0;
} // namespace constants
} // namespace scg

#endif // DEFINES_H_
