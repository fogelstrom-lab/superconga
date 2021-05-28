//===- VectorPotential_Kernel.h - CUDA kernel for computing energy shift. -===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA kernel for computing the energy shift caused by vector potentials (both
/// external vector potentials, and vector potentials induced by currents).
///
//===----------------------------------------------------------------------===//
#ifndef VECTOR_POTENTIAL_KERNEL_H_
#define VECTOR_POTENTIAL_KERNEL_H_

#include "defines.h"
#include <cuda.h>
#include <math.h>

namespace scg {
template <typename T, template <typename> class POTENTIAL>
__global__ void computeEnergyShiftOnlyExternalKernel(
    const T inFermiVelocityX, const T inFermiVelocityY, const T inSystemAngle,
    const T inFluxDensity, const T inArea, const int inGridResolution,
    const T inGridElementSize, T *const outEnergyShift) {
  // XY-coordinate indices.
  const int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

  if (xIdx < inGridResolution && yIdx < inGridResolution) {
    // Center system coordinates at origin.
    const T offset =
        -static_cast<T>(0.5) * static_cast<T>(inGridResolution - 1);

    // Go to local, physical coordinates.
    T x = (static_cast<T>(xIdx) + offset) * inGridElementSize;
    T y = (static_cast<T>(yIdx) + offset) * inGridElementSize;

    // Rotate local coordinates to global frame.
    utils::rotate(inSystemAngle, x, y);

    T vectorPotentialX;
    T vectorPotentialY;
    POTENTIAL<T>::computeVectorPotential_local(
        inFluxDensity, inArea, x, y, vectorPotentialX, vectorPotentialY);

    // Linear index.
    const int idx = yIdx * inGridResolution + xIdx;

    // Compute energy shift.
    // The factor pi comes from our choice of coordinates.
    outEnergyShift[idx] = -static_cast<T>(constants::SIGN_E * M_PI) *
                          (inFermiVelocityX * vectorPotentialX +
                           inFermiVelocityY * vectorPotentialY);
  }
}

template <typename T, template <typename> class POTENTIAL>
__global__ void computeEnergyShiftKernel(
    const T *const inVectorPotentialX, const T *const inVectorPotentialY,
    const T inFermiVelocityX, const T inFermiVelocityY, const T inSystemAngle,
    const T inFluxDensity, const T inArea, const int inGridResolution,
    const T inGridElementSize, T *const outEnergyShift) {
  // XY-coordinate indices.
  const int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

  if (xIdx < inGridResolution && yIdx < inGridResolution) {
    // Center system coordinates at origin.
    const T offset =
        -static_cast<T>(0.5) * static_cast<T>(inGridResolution - 1);

    // Go to local, physical coordinates.
    T x = (static_cast<T>(xIdx) + offset) * inGridElementSize;
    T y = (static_cast<T>(yIdx) + offset) * inGridElementSize;

    // Rotate local coordinates to global frame.
    utils::rotate(inSystemAngle, x, y);

    // External vector potential.
    T extVectorPotentialX;
    T extVectorPotentialY;
    POTENTIAL<T>::computeVectorPotential_local(
        inFluxDensity, inArea, x, y, extVectorPotentialX, extVectorPotentialY);

    // Linear index.
    const int idx = yIdx * inGridResolution + xIdx;

    // The total vector potential.
    const T vectorPotentialX = inVectorPotentialX[idx] + extVectorPotentialX;
    const T vectorPotentialY = inVectorPotentialY[idx] + extVectorPotentialY;

    // Compute energy shift.
    // The factor pi comes from our choice of coordinates.
    outEnergyShift[idx] = -static_cast<T>(constants::SIGN_E * M_PI) *
                          (inFermiVelocityX * vectorPotentialX +
                           inFermiVelocityY * vectorPotentialY);
  }
}
} // namespace scg

#endif // VECTOR_POTENTIAL_KERNEL_H_
