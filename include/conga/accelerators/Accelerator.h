//===----------------- Accelerator.h - Accelerator base class.
//----------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Accelerator base class, i.e. for different kinds of self-consistency
/// accelerators.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_ACCELERATOR_H_
#define CONGA_ACCELERATOR_H_

#include "../grid.h"

namespace conga {
// We need to forward declare the order parameter so it can compile.
// TODO(niclas): Disentangle the class dependencies.
template <typename> class OrderParameter;

namespace accelerators {
template <typename T> class Accelerator {
public:
  /// \brief Constructor of the accelerator base class.
  Accelerator() {}

  /// \brief Destructor of the accelerator base class.
  virtual ~Accelerator() {}

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  /// \param inOutVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  ///
  /// \return Void.
  virtual void compute(const OrderParameter<T> &inOrderParameterG,
                       const GridGPU<T> &inVectorPotentialG,
                       OrderParameter<T> &inOutOrderParameterX,
                       GridGPU<T> &inOutVectorPotentialX) = 0;

  /// \brief Compute the next order parameter.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  ///
  /// \return Void.
  virtual void compute(const OrderParameter<T> &inOrderParameterG,
                       OrderParameter<T> &inOutOrderParameterX) = 0;

  /// \brief Compute the next vector potential.
  ///
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  ///
  /// \return Void.
  virtual void compute(const GridGPU<T> &inVectorPotentialG,
                       GridGPU<T> &inOutVectorPotentialX) = 0;
};
} // namespace accelerators
} // namespace conga

#endif // CONGA_ACCELERATOR_H_
