//===-------------- utils.h - Accelerator utility functions. --------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Accelerator utility functions.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_ACCELERATOR_UTILS_H_
#define CONGA_ACCELERATOR_UTILS_H_

#include "../grid.h"
#include "../order_parameter/OrderParameter.h"

#include <cuda.h>

namespace conga {
namespace accelerators {
namespace internal {
template <typename T>
__global__ void copy(const int inDimX, const int inDimY, const T *const inSrc,
                     T *const outDst) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < inDimX && y < inDimY) {
    const int idx = y * inDimX + x;
    outDst[idx] = inSrc[idx];
  }
}
} // namespace internal

/// \brief Sum all elements of a grid.
///
/// \param inGrid The grid to sum.
///
/// \return The sum.
template <typename T> T sumGrid(const GridGPU<T> &inGrid) {
  T sum = static_cast<T>(0);
  for (int i = 0; i < inGrid.numFields(); ++i) {
    sum += inGrid.sumField(i);
  }
  return sum;
}

/// \brief Concatenate the order.parameter components.
///
/// The real and imaginary parts are treated as separate fields.
///
/// \param inOrderParameter The order parameter.
/// \param inOutDst The concatenated result.
///
/// \return Void.
template <typename T>
void concatenate(const OrderParameter<T> &inOrderParameter,
                 GridGPU<T> &inOutDst) {
  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = inOutDst.getKernelDimensions();

  // Dimensions.
  const int dimX = inOutDst.dimX();
  const int dimY = inOutDst.dimY();
  const int numComponents = inOrderParameter.getNumComponents();

  // Copy order parameter.
  for (int i = 0; i < numComponents; ++i) {
    const int realIdx = 2 * i;
    const int imagIdx = realIdx + 1;

    // Get the order parameter component.
    const GridGPU<T> *const component = inOrderParameter.getComponentGrid(i);

    internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, component->getDataPointer(0, Type::real),
        inOutDst.getDataPointer(realIdx));
    internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, component->getDataPointer(0, Type::imag),
        inOutDst.getDataPointer(imagIdx));
  }
}

/// \brief Concatenate the order-parameter components and the vector potential.
///
/// The real and imaginary parts are treated as separate fields.
///
/// \param inOrderParameter The order parameter.
/// \param inVectorPotential The vector potential.
/// \param inOutDst The concatenated result.
///
/// \return Void.
template <typename T>
void concatenate(const OrderParameter<T> &inOrderParameter,
                 const GridGPU<T> &inVectorPotential, GridGPU<T> &inOutDst) {
  // Copy order parameter.
  concatenate(inOrderParameter, inOutDst);

  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = inOutDst.getKernelDimensions();

  // Dimensions.
  const int dimX = inOutDst.dimX();
  const int dimY = inOutDst.dimY();
  const int numComponents = inOrderParameter.getNumComponents();

  // Copy vector potential.
  const int xIdx = 2 * numComponents;
  const int yIdx = xIdx + 1;
  internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inVectorPotential.getDataPointer(0),
      inOutDst.getDataPointer(xIdx));
  internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inVectorPotential.getDataPointer(1),
      inOutDst.getDataPointer(yIdx));
}

/// \brief Split the concatenated order parameter and the vector potential.
///
/// \param inSrc The order parameter and vector potential concatenated.
/// \param inOutOrderParameter The order parameter.
/// \param inOutVectorPotential The vector potential.
///
/// \return Void.
template <typename T>
void split(const GridGPU<T> &inSrc, OrderParameter<T> &inOutOrderParameter) {
  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = inSrc.getKernelDimensions();

  // Dimensions.
  const int dimX = inSrc.dimX();
  const int dimY = inSrc.dimY();
  const int numComponents = inOutOrderParameter.getNumComponents();

  // Copy order parameter.
  for (int i = 0; i < numComponents; ++i) {
    const int realIdx = 2 * i;
    const int imagIdx = realIdx + 1;

    // Get the order parameter component.
    GridGPU<T> *const component = inOutOrderParameter.getComponentGrid(i);

    internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, inSrc.getDataPointer(realIdx),
        component->getDataPointer(0, Type::real));
    internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, inSrc.getDataPointer(imagIdx),
        component->getDataPointer(0, Type::imag));
  }
}

/// \brief Split the concatenated order-parameter components and the vector
/// potential.
///
/// \param inSrc The order-parameter components and vector potential
/// concatenated.
/// \param inOutOrderParameter The order parameter.
/// \param inOutVectorPotential The vector potential.
///
/// \return Void.
template <typename T>
void split(const GridGPU<T> &inSrc, OrderParameter<T> &inOutOrderParameter,
           GridGPU<T> &inOutVectorPotential) {
  // Copy order parameter.
  split(inSrc, inOutOrderParameter);

  // Kernel parameters.
  const auto [numBlocks, numThreadsPerBlock] = inSrc.getKernelDimensions();

  // Dimensions.
  const int dimX = inSrc.dimX();
  const int dimY = inSrc.dimY();
  const int numComponents = inOutOrderParameter.getNumComponents();

  // Copy vector potential.
  const int xIdx = 2 * numComponents;
  const int yIdx = xIdx + 1;
  internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inSrc.getDataPointer(xIdx),
      inOutVectorPotential.getDataPointer(0));
  internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inSrc.getDataPointer(yIdx),
      inOutVectorPotential.getDataPointer(1));
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_ACCELERATOR_UTILS_H_
