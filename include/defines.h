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

// Convenient 2D vector definition.
template <typename T> using vector2d = std::array<T, 2>;

namespace qcs {
// Margin around superconducting grain in simulation space. A margin
// of 0.04 means that the grain covers 1.0-0.04 = 96 percent of the total
// simulation space.
const double GRAIN_SIZE_MARGIN = 0.04;

// Return grain size fraction of simulation domain. Should never be
// larger than 96 percent! This is to allow for padding of the order parameter
// for treatment of boundary conditions. The grain size percent is used when
// setting up a new simulation grain to calculate the lattice spacing in
// setGridElementSize(), and in the classes PolygonGeometry and DiscGeometry.
const double GRAIN_SIZE_FRACTION = 1.0 - GRAIN_SIZE_MARGIN;

// Size of coordinate system where grain is allowed to be, as seen by the user:
// [-MAXIMUM_COORDINATE, +MAXIMUM_COORDINATE], used when specifying size
// and coordinates of of discs/polygons.
// Maximum grain "radius": MAXIMUM_COORDINATE
// Maximum grain "diameter": 2*MAXIMUM_COORDINATE
const double MAXIMUM_COORDINATE = 0.5;

// The maximum coordinate expected by the internal Disc and Polygon classes.
// They are different, because polygons (n-gons) are defined via the square that
// bounds the circle they lie on, while the discs are defined straight up.
// DO NOT MODIFY!
const double INTERNAL_MAXIMUM_COORDINATE_POLY = 0.5 / std::sqrt(2.0);
const double INTERNAL_MAXIMUM_COORDINATE_DISC = 0.5;

// The Euler-Mascheroni constant, gamma. Value taken from Wikipedia:
// https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant
const double EULER_MASCHERONI = 0.57721566490153286060651209008240243;

// The bulk s-wave order parameter at zero temperature has a nice analytical
// expression, Delta = pi * exp(-gamma), see page 104 of P. Holmvall PhD Thesis.
const double S_WAVE_BULK = M_PI * std::exp(-EULER_MASCHERONI);

// The bulk d-wave order parameter at zero temperature has a nice analytical
// expression, Delta = pi * exp(-gamma), see page 104 of P. Holmvall PhD Thesis.
const double D_WAVE_BULK =
    M_PI * std::sqrt(2.0) * std::exp(-EULER_MASCHERONI - 0.5);

// The value of the Riemann zeto function evaluated at 3, i.e. Apery's constant
// = zeta(3). Value taken from Wikipedia:
// https://en.wikipedia.org/wiki/Ap%C3%A9ry%27s_constant
// TODO(niclas): When moving to C++17 we can use std::riemann_zeta(3) instead.
const double APERY = 1.202056903159594285399738161511;
} // namespace qcs

#endif // DEFINES_H_
