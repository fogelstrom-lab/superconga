//===---------------------- Picard.h - Picard accelerator. ----------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Picard accelerator. It is not really an accelerator, it is the default.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_PICARD_H_
#define CONGA_PICARD_H_

#include "Accelerator.h"

#include <json/json.h>

#include <cassert>

namespace conga {
namespace accelerators {
template <typename T> class Picard : public Accelerator<T> {
public:
  /// \brief Constructor of the Picard accelerator.
  ///
  /// \param inStepSize The step size.
  explicit Picard(const T inStepSize = m_stepSizeDefault);

  /// \brief Constructor of the Picard accelerator.
  ///
  /// \param inJSON JSON with parameters. Absent keys get default values.
  explicit Picard(const Json::Value &inJSON);

  /// \brief Destructor of the Picard accelerator.
  ~Picard(){};

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
  void compute(const OrderParameter<T> &inOrderParameterG,
               const GridGPU<T> &inVectorPotentialG,
               OrderParameter<T> &inOutOrderParameterX,
               GridGPU<T> &inOutVectorPotentialX) override;

  /// \brief Compute the next order parameter.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  ///
  /// \return Void.
  void compute(const OrderParameter<T> &inOrderParameterG,
               OrderParameter<T> &inOutOrderParameterX) override;

  /// \brief Compute the next vector potential.
  ///
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  ///
  /// \return Void.
  void compute(const GridGPU<T> &inVectorPotentialG,
               GridGPU<T> &inOutVectorPotentialX) override;

private:
  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inG The right-hand side of the self-consistency equation, 'g(x)'.
  /// \param inOutX The left-hand side of the self-consistency equation, 'x'.
  ///
  /// \return Void.
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX) const;

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(1);

  // The step size.
  const T m_stepSize;
};

template <typename T>
Picard<T>::Picard(const T inStepSize)
    : Accelerator<T>(), m_stepSize(inStepSize) {
  assert(m_stepSize > static_cast<T>(0));
}

template <typename T>
Picard<T>::Picard(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSize(inJSON.isMember("step_size")
                     ? static_cast<T>(inJSON["step_size"].asDouble())
                     : m_stepSizeDefault) {
  assert(m_stepSize > static_cast<T>(0));
}

template <typename T>
void Picard<T>::compute(const OrderParameter<T> &inOrderParameterG,
                        const GridGPU<T> &inVectorPotentialG,
                        OrderParameter<T> &inOutOrderParameterX,
                        GridGPU<T> &inOutVectorPotentialX) {
  compute(inOrderParameterG, inOutOrderParameterX);
  compute(inVectorPotentialG, inOutVectorPotentialX);
}

template <typename T>
void Picard<T>::compute(const OrderParameter<T> &inOrderParameterG,
                        OrderParameter<T> &inOutOrderParameterX) {
  const T numComponents = inOrderParameterG.getNumComponents();
  for (int i = 0; i < numComponents; ++i) {
    computeInternal(*(inOrderParameterG.getComponentGrid(i)),
                    *(inOutOrderParameterX.getComponentGrid(i)));
  }
}

template <typename T>
void Picard<T>::compute(const GridGPU<T> &inVectorPotentialG,
                        GridGPU<T> &inOutVectorPotentialX) {
  computeInternal(inVectorPotentialG, inOutVectorPotentialX);
}

template <typename T>
void Picard<T>::computeInternal(const GridGPU<T> &inG,
                                GridGPU<T> &inOutX) const {
  // Take a step.
  inOutX += (inG - inOutX) * m_stepSize;
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_PICARD_H_
