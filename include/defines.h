//===-- defines.h - Application-level (global) definitions.
//----------------------------------===//
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

namespace qcs {

// Margin around superconducting grain in simulation space. A margin
// of 0.04 means that the grain covers 1.0-0.04 = 96 percent of the total
// simulation space.
const double GRAIN_SIZE_MARGIN = static_cast<double>(0.04);

// Return grain size fraction of simulation domain. Should never be
// larger than 96 percent! This is to allow for padding of the order parameter
// for treatment of boundary conditions. The grain size percent is used when
// setting up a new simulation grain to calculate the lattice spacing in
// setGridElementSize(), and in the classes PolygonGeometry and DiscGeometry.
const double GRAIN_SIZE_FRACTION = static_cast<double>(1.0 - GRAIN_SIZE_MARGIN);

} // namespace qcs

#endif // DEFINES_H_