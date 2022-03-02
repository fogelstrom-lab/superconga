//===------------------- Polyak.h - Polyak accelerator. -------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Polyak's "small heavy ball" accelerator.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_POLYAK_H_
#define CONGA_POLYAK_H_

#include "../Type.h"
#include "Accelerator.h"
#include "utils.h"

#include <cuda.h>
#include <json/json.h>

namespace conga {
namespace accelerators {
template <typename T> class Polyak : public Accelerator<T> {
public:
  /// \brief Constructor of the Polyak accelerator.
  ///
  /// \param inStepSize The step size.
  /// \param inDrag The drag.
  explicit Polyak(const T inStepSize = m_stepSizeDefault,
                  const T inDrag = m_dragDefault);

  /// \brief Constructor of the Polyak accelerator.
  ///
  /// \param inJSON JSON with parameters. Absent keys get default values.
  explicit Polyak(const Json::Value &inJSON);

  /// \brief Destructor of the adaptive-ball accelerator.
  ~Polyak(){};

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
  /// \brief Initialize the running mean.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  ///
  /// \return Void.
  void initialize(const GridGPU<T> &inX);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inG The right-hand side of the self-consistency equation, 'g(x)'.
  /// \param inOutX The left-hand side of the self-consistency equation, 'x'.
  ///
  /// \return Void.
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX);

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(2);
  static constexpr T m_dragDefault = static_cast<T>(0.5);

  // The step size.
  const T m_stepSize;

  // The drag.
  const T m_drag;

  // If initialized.
  bool m_initialized;

  // The velocity.
  GridGPU<T> m_velocity;
};

template <typename T>
Polyak<T>::Polyak(const T inStepSize, const T inDrag)
    : Accelerator<T>(), m_stepSize(inStepSize), m_drag(inDrag),
      m_initialized(false) {
  assert(m_stepSize > static_cast<T>(0));
  assert(m_drag > static_cast<T>(0));
  assert(m_drag < static_cast<T>(1));
}

template <typename T>
Polyak<T>::Polyak(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSize(inJSON.isMember("step_size")
                     ? static_cast<T>(inJSON["step_size"].asDouble())
                     : m_stepSizeDefault),
      m_drag(inJSON.isMember("drag") ? static_cast<T>(inJSON["drag"].asDouble())
                                     : m_dragDefault),
      m_initialized(false) {
  assert(m_stepSize > static_cast<T>(0));
  assert(m_drag > static_cast<T>(0));
  assert(m_drag < static_cast<T>(1));
}

template <typename T> void Polyak<T>::initialize(const GridGPU<T> &inX) {
  const int dimX = inX.dimX();
  const int dimY = inX.dimY();
  const int numFields = inX.numFields();
  const Type::type representation = inX.representation();

  m_velocity = GridGPU<T>(dimX, dimY, numFields, representation);
  m_velocity.setZero();

  m_initialized = true;
}

template <typename T>
void Polyak<T>::compute(const OrderParameter<T> &inOrderParameterG,
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
void Polyak<T>::compute(const OrderParameter<T> &inOrderParameterG,
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

  // Concatenate the order parameter and the vector potential to real-valued
  // grids.
  concatenate(inOutOrderParameterX, X);
  concatenate(inOrderParameterG, G);

  // Take step.
  computeInternal(G, X);

  // Split the grid into its constituent order parameter and vector potential.
  split(X, inOutOrderParameterX);
}

template <typename T>
void Polyak<T>::compute(const GridGPU<T> &inVectorPotentialG,
                        GridGPU<T> &inOutVectorPotentialX) {
  // Take step.
  computeInternal(inVectorPotentialG, inOutVectorPotentialX);
}

namespace polyak_internal {
template <typename T>
__global__ void computeInternal(const int inDimX, const int inDimY,
                                const int inDimZ, const T inStepSize,
                                const T inDrag, const T *const inDiff,
                                T *const inOutVelocity, T *const inOutX) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Field.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Linear index.
    const int idx = inDimX * (z * inDimY + y) + x;

    // Update velocity.
    const T velocity = inOutVelocity[idx] * (static_cast<T>(1) - inDrag) +
                       inDiff[idx] * inStepSize;
    inOutVelocity[idx] = velocity;

    // Compute the result.
    inOutX[idx] += velocity;
  }
}
} // namespace polyak_internal

template <typename T>
void Polyak<T>::computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX) {
  // Initialize grids.
  if (!m_initialized) {
    initialize(inOutX);
  }

  const int dimX = inOutX.dimX();
  const int dimY = inOutX.dimY();
  const int numFields = inOutX.numFields();

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOutX.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // The difference.
  const GridGPU<T> diff = inG - inOutX;

  // Take a step.
  polyak_internal::computeInternal<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, numFields, m_stepSize, m_drag, diff.getDataPointer(),
      m_velocity.getDataPointer(), inOutX.getDataPointer());
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_POLYAK_H_
