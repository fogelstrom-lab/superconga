//===---------- BoundaryCondition.h - Boundary condition class. -----------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Boundary condition class. Parent class to inherit from for different
/// boundary conditions.
///
//===----------------------------------------------------------------------===//
#ifndef BOUNDARY_CONDITION_H_
#define BOUNDARY_CONDITION_H_

#include "Context.h"

namespace scg {
template <typename T> class BoundaryCondition {
public:
  BoundaryCondition() {}
  virtual ~BoundaryCondition() {}

  virtual void computeBoundaryCondition(Context<T> *ctx) = 0;
};
} // namespace scg

#endif // BOUNDARY_CONDITION_H_
