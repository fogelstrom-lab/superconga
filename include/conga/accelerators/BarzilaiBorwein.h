//===----------- BarzilaiBorwein.h - Barzilai-Borwein accelerator. --------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Barzilai-Borwein accelerator.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_BARZILAI_BORWEIN_H_
#define CONGA_BARZILAI_BORWEIN_H_

#include "../Type.h"
#include "Accelerator.h"
#include "utils.h"

#include <cuda.h>
#include <json/json.h>

#include <cassert>
#include <cmath>

namespace conga {
namespace accelerators {
template <typename T> class BarzilaiBorwein : public Accelerator<T> {
public:
  /// \brief Constructor of the Barzilai-Borwein accelerator.
  ///
  /// \param inStepSize The initial step size.
  explicit BarzilaiBorwein(const T inStepSize = m_stepSizeDefault);

  /// \brief Constructor of the Barzilai-Borwein accelerator.
  ///
  /// \param inJSON JSON with parameters. Absent keys get default values.
  explicit BarzilaiBorwein(const Json::Value &inJSON);

  /// \brief Destructor of the Barzilai-Borwein accelerator.
  ~BarzilaiBorwein(){};

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
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX);

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(1);

  // The initial step size.
  const T m_stepSizeInit;

  // If initialized.
  bool m_initialized;

  // The previous order parameter and vector potential.
  GridGPU<T> m_xPrevious;
  GridGPU<T> m_gPrevious;
};

template <typename T>
BarzilaiBorwein<T>::BarzilaiBorwein(const T inStepSize)
    : Accelerator<T>(), m_stepSizeInit(inStepSize), m_initialized(false) {
  assert(m_stepSize > static_cast<T>(0));
}

template <typename T>
BarzilaiBorwein<T>::BarzilaiBorwein(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSizeInit(inJSON.isMember("step_size")
                         ? static_cast<T>(inJSON["step_size"].asDouble())
                         : m_stepSizeDefault),
      m_initialized(false) {
  assert(m_stepSize > static_cast<T>(0));
}

template <typename T>
void BarzilaiBorwein<T>::compute(const OrderParameter<T> &inOrderParameterG,
                                 const GridGPU<T> &inVectorPotentialG,
                                 OrderParameter<T> &inOutOrderParameterX,
                                 GridGPU<T> &inOutVectorPotentialX) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inVectorPotentialG.dimX();
  const int dimY = inVectorPotentialG.dimY();
  const T numComponents = inOrderParameterG.getNumComponents();
  const int complexFactor = 2;
  const int numFields =
      complexFactor * numComponents + inVectorPotentialG.numFields();
  const Type::type representation = Type::real;

  // Initialize grids. The names come from the self-consistency equation:
  // x = g(x).
  GridGPU<T> X(dimX, dimY, numFields, representation);
  GridGPU<T> G(dimX, dimY, numFields, representation);

  // Concatenate the order parameter and the vector potential to real-valued
  // grids.
  concatenate(inOutOrderParameterX, inOutVectorPotentialX, X);
  concatenate(inOrderParameterG, inVectorPotentialG, G);

  // Take step.
  computeInternal(G, X);

  // Split the grid into its constituent order parameter and vector potential.
  split(X, inOutOrderParameterX, inOutVectorPotentialX);
}

template <typename T>
void BarzilaiBorwein<T>::compute(const OrderParameter<T> &inOrderParameterG,
                                 OrderParameter<T> &inOutOrderParameterX) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inOrderParameterG.getComponentGrid(0)->dimX();
  const int dimY = inOrderParameterG.getComponentGrid(0)->dimY();
  const T numComponents = inOrderParameterG.getNumComponents();
  const int complexFactor = 2;
  const int numFields = complexFactor * numComponents;
  const Type::type representation = Type::real;

  // Initialize grids. The names come from the self-consistency equation:
  // x = g(x).
  GridGPU<T> X(dimX, dimY, numFields, representation);
  GridGPU<T> G(dimX, dimY, numFields, representation);

  // Concatenate the order-parameter components to real-valued grids.
  concatenate(inOutOrderParameterX, X);
  concatenate(inOrderParameterG, G);

  // Take step.
  computeInternal(G, X);

  // Split the grid into its constituent order parameter.
  split(X, inOutOrderParameterX);
}

template <typename T>
void BarzilaiBorwein<T>::compute(const GridGPU<T> &inVectorPotentialG,
                                 GridGPU<T> &inOutVectorPotentialX) {
  // Take step.
  computeInternal(inVectorPotentialG, inOutVectorPotentialX);
}

template <typename T>
void BarzilaiBorwein<T>::computeInternal(const GridGPU<T> &inG,
                                         GridGPU<T> &inOutX) {
  // This is, of course, not truly the gradient. But when comparing with e.g.
  // gradient descent, it takes the gradients place.
  const GridGPU<T> gradient = inOutX - inG;

  // Set step size.
  T stepSize = m_stepSizeInit;

  // See:
  // https://www.caam.rice.edu/~yzhang/caam565/L1_reg/BB-step.pdf
  if (m_initialized) {
    const GridGPU<T> dG = gradient - m_gPrevious;
    const GridGPU<T> dX = inOutX - m_xPrevious;

    // For small BB step size use:
    // m_stepSizeInit = dot / dG.sumSquare();
    const T dot = std::abs(sumGrid(dX * dG));
    stepSize = dX.sumSquare() / dot;
    // Should this minimum step size be exposed?
    stepSize = std::max(stepSize, static_cast<T>(1e-3));
  }

  // Update previous.
  m_xPrevious = inOutX;
  m_gPrevious = gradient;

  // Take step.
  inOutX -= gradient * stepSize;

  // First iteration is done.
  m_initialized = true;
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_BARZILAI_BORWEIN_H_
