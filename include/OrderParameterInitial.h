//===-------------- OrderParameterInitial.h - Polygon class. --------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Collection of functions to set the initial condition of the order parameter.
/// All member functions are static so this struct should not be instantiated.
///
//===----------------------------------------------------------------------===//
#ifndef ORDER_PARAMETER_INITIAL_H_
#define ORDER_PARAMETER_INITIAL_H_

#include "Grid.h"
#include "OrderParameterComponent.h"
#include "defines.h"
#include "utils.h"
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <limits>

namespace conga {
template <typename T> struct Initial {
public:
  /// \brief Initialize order parameter to bulk value.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inPhase - Order parameter phase.
  ///
  /// \return - void.
  static void bulk(OrderParameterComponent<T> *inOutDeltaComponent,
                   const T inPhase = 0.0);

  /// \brief Initialize order parameter to a constant.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inRealPart - Order parameter real part in k_B*T_c.
  /// \param inImaginaryPart - Order parameter imaginary part in k_B*T_c.
  ///
  /// \return - void.
  static void constant(OrderParameterComponent<T> *inOutDeltaComponent,
                       const T inRealPart = 1.0, const T inImaginaryPart = 0.0);

  /// \brief Initialize order parameter from an existing one.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inGrid - Order parameter to initialize from.
  ///
  /// \return - void.
  static void grid(OrderParameterComponent<T> *inOutDeltaComponent,
                   const GridGPU<T> &inGrid);

  /// \brief Initialize order parameter to bulk with phase winding.
  ///
  /// Set the order parameter to bulk value with a phase winding, good for e.g.
  /// Abrikosov vortices or flux quantization in a solenoid: n flux quanta =
  /// n*2pi phase winding. Note that negative phase winding is reasonable, and
  /// leads to e.g. opposite current circulation in a solenoid or around an
  /// Abrikosov vortex.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inNumWinding - Phase winding.
  /// \param inPhaseShift - Arbitrary shift to phase.
  /// \param inCenterX - Center of phase winding: x-coordinate in [-1,1].
  /// \param inCenterY - Center of phase winding: y-coordinate in [-1,1].
  /// \param inCutoffRadius - Radius below which order parameter = 0, which is
  /// appropriate since a phase winding corresponds to a topological defect in
  /// the order parameter (the order parameter usually goes smoothly to zero in
  /// an Abrikosov vortex core). Note that coordinates are scaled by a factor
  /// sqrt(2) between discs and polygons: the user has to divide center with
  /// sqrt(2) when using polygon.
  ///
  /// \return - void.
  static void
  phaseWinding(OrderParameterComponent<T> *const inOutDeltaComponent,
               const T inNumWinding, const T inPhaseShift = static_cast<T>(0.0),
               const T inCenterX = static_cast<T>(0.0),
               const T inCenterY = static_cast<T>(0.0),
               const T inCutoffRadius = static_cast<T>(0.0));

  /// \brief Initialize order parameter to an "edge phase crystal".
  ///
  /// Set order parameter to an "edge phase crystal", following Eq. (12) of
  /// Phys. Rev. Research 2, 013104 (2020):
  /// https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013104.
  /// This form has a constant order parameter magnitude, with a phase (chi)
  /// which oscillates sinusoidally along an edge (x) and decays exponentially
  /// away from it (y): chi(x,y) = -cos(q_x*x)*(1+y/y_0)*exp(-y/y_0). Here, q_x
  /// is the wave number and y_0 the decay length.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inPeriod - Period of phase oscillation.
  /// \param inDecayLength - Decay length of phase, in coherence lengths.
  /// \param inCenterX - x origin of phase crystal (coordinate on edge).
  /// \param inCenterY - y origin of phase crystal (coordinate on edge).
  /// \param inRotAngle - Rotation angle (in radians) of edge.
  ///
  /// \return - void.
  static void
  edgePhaseCrystal(OrderParameterComponent<T> *const inOutDeltaComponent,
                   const T inPeriod, const T inDecayLength, const T inCenterX,
                   const T inCenterY, const T inRotAngle);

  /// \brief Initialize order parameter to bulk with random phase perturbation.
  ///
  /// Set the order parameter to bulk value with a spatially random phase
  /// perturbation between inRandomPhaseMix*[0,2pi].
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inRandomPhaseMix - Mixing term for (spatially) random phase factor.
  /// \param inPhase - Overall phase of order parameter.
  ///
  /// \return - void.
  static void random(OrderParameterComponent<T> *inOutDeltaComponent,
                     const T inRandomPhaseMix = 0.5, const T inPhase = 0.0);

  /// \brief Initialize order parameter to bulk w. phase winding + random phase.
  ///
  /// Set the order parameter to bulk value with a phase winding, good for e.g.
  /// Abrikosov vortices or flux quantization in a solenoid: n flux quanta =
  /// n*2pi phase winding. Note that negative phase winding is reasonable, and
  /// leads to e.g. opposite current circulation in a solenoid or around an
  /// Abrikosov vortex.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inNumWinding - Phase winding.
  /// \param inPhaseShift - Arbitrary shift to phase.
  /// \param inRandomPhaseMix - Mixing term for (spatially) random phase factor.
  /// \param inCenterX - Center of phase winding: x-coordinate in [-1,1].
  /// \param inCenterY - Center of phase winding: y-coordinate in [-1,1].
  /// \param inCutoffRadius - Radius below which order parameter = 0, which is
  /// appropriate since a phase winding corresponds to a topological defect in
  /// the order parameter (the order parameter usually goes smoothly to zero in
  /// an Abrikosov vortex core). Note that coordinates are scaled by a factor
  /// sqrt(2) between discs and polygons: the user has to divide center with
  /// sqrt(2) when using polygon.
  ///
  /// \return - void.
  static void
  perturbedPhaseWinding(OrderParameterComponent<T> *const inOutDeltaComponent,
                        const T inNumWinding,
                        const T inRandomPhaseMix = static_cast<T>(0.5),
                        const T inPhaseShift = static_cast<T>(0.0),
                        const T inCenterX = static_cast<T>(0.0),
                        const T inCenterY = static_cast<T>(0.0),
                        const T inCutoffRadius = static_cast<T>(0.0));

  /// \brief Initialize order parameter from existing one via file.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inFilename - Filename (full path) of existing order parameter.
  ///
  /// \return - void.
  static int read(OrderParameterComponent<T> *inOutDeltaComponent,
                  const char *const inFilename);

  /// \brief Initialize order parameter from existing one via file.
  ///
  /// \param inOutDeltaComponent - Pointer to order parameter.
  /// \param inFilename - Filename (full path) of existing order parameter.
  /// \param inFileNumber - Number to append to file name.
  ///
  /// \return - void.
  static int read(OrderParameterComponent<T> *inOutDeltaComponent,
                  const char *const inFilename, const int inFileNumber);
};

namespace internal {
template <typename T>
__global__ void perturbedPhaseWindingKernel(
    T *const outReal, T *const outImag, const int inNumX, const int inNumY,
    const T inNumWinding, const T inRandomPhaseMix, const T inPhaseShift,
    const T inCenterX, const T inCenterY, const T inCutoffRadius,
    const T inDeltaBulk, const T inEpsilon) {
  // Discrete coordinates in discretized order parameter lattice.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Check that coordinates lie within geometry.
  if (i < inNumX && j < inNumY) {
    // Normalize coordinates to [-1, 1], and offset with center.
    const T x = static_cast<T>(2 * i) / static_cast<T>(inNumX - 1) -
                static_cast<T>(1.0) - inCenterX;
    const T y = static_cast<T>(2 * j) / static_cast<T>(inNumY - 1) -
                static_cast<T>(1.0) - inCenterY;
    const T radius = sqrt(x * x + y * y);

    // Check if point is within cutoff, set to zero in that case. Otherwise,
    // compute angle and add phase shift.
    const bool isPointWithinCutoff = (radius <= inCutoffRadius);
    const T angle = isPointWithinCutoff ? static_cast<T>(0.0) : atan2(y, x);

    // Check if point is at the center. If so, set to zero, otherwise use
    // the bulk value.
    const bool isPointAtCenter = (radius < inEpsilon);
    const T magnitude =
        (isPointAtCenter && (std::abs(inNumWinding) > inEpsilon))
            ? static_cast<T>(0.0)
            : inDeltaBulk;

    // Compute unique coordinate index (for storage and seed).
    const int idx = j * inNumX + i;

    // Generate state and seed for random number:
    // https://docs.nvidia.com/cuda/curand/device-api-overview.html.
    curandState state;
    const unsigned int seed = static_cast<unsigned int>(clock());
    const int seedSubSequence = idx;
    const int seedOffsetPerCall = 0;
    curand_init(seed, seedSubSequence, seedOffsetPerCall, &state);

    // Compute random phase shift from [-pi,pi].
    const T randomPhase = static_cast<T>(2.0 * M_PI) *
                          static_cast<T>(curand_uniform_double(&state) - 0.5);

    // Compute the total phase.
    const T phase =
        angle * inNumWinding + randomPhase * inRandomPhaseMix + inPhaseShift;

    // Store result.
    outReal[idx] = magnitude * cos(phase);
    outImag[idx] = magnitude * sin(phase);
  }
}

template <typename T>
__global__ void edgePhaseCrystalKernel(
    T *const outReal, T *const outImag, const int inNumX, const int inNumY,
    const T inWaveNumber, const T inInvDecayLength, const T inCenterX,
    const T inCenterY, const T inRotAngle, const T inMagnitude,
    const T inPhaseMagnitude = static_cast<T>(M_PI / 8.0)) {
  // Coordinates to new resampling grid
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < inNumX && j < inNumY) {
    const int idx = j * inNumX + i;

    // Normalize coordinates to -1.0 <= x,y <= +1.0, and offset with center
    const T xOrigin = (static_cast<T>(2 * i) / static_cast<T>(inNumX - 1)) -
                      static_cast<T>(1.0);
    const T yOrigin = (static_cast<T>(2 * j) / static_cast<T>(inNumY - 1)) -
                      static_cast<T>(1.0);
    // Scale with margin so that physical grain is within [-1,1]
    const T xOriginMargin =
        xOrigin / (static_cast<T>(1.0) - constants::GRAIN_SIZE_MARGIN);
    const T yOriginMargin =
        yOrigin / (static_cast<T>(1.0) - constants::GRAIN_SIZE_MARGIN);

    T xRotated;
    T yRotated;

    // Rotate coordinate system with angle given by argument
    utils::rotate(inRotAngle, xOriginMargin, yOriginMargin, xRotated, yRotated);

    // Translate coordinate system to origin given by arguments
    const T xTranslated = xRotated - inCenterX;
    const T yTranslated = yRotated - inCenterY;

    // Calculate the edge phase crystal, see Eq. (12) of Phys. Rev. Research 2,
    // 013104 (2020)
    const T phase = cos(inWaveNumber * xTranslated) *
                    (static_cast<T>(1.0) + yTranslated * inInvDecayLength) *
                    exp(-yTranslated * inInvDecayLength) * inPhaseMagnitude;

    // Set the edge phase crystal
    outReal[idx] = inMagnitude * cos(phase);
    outImag[idx] = inMagnitude * sin(phase);
  }
}
} // namespace internal

template <typename T>
void Initial<T>::bulk(OrderParameterComponent<T> *inOutDeltaComponent,
                      const T inPhase) {

  const T deltaBulk = inOutDeltaComponent->deltaBulk();
  GridGPU<T> *deltaGrid = inOutDeltaComponent->getGrid();
  deltaGrid->setValue(deltaBulk * std::cos(inPhase), Type::real);
  deltaGrid->setValue(deltaBulk * std::sin(inPhase), Type::imag);
}

template <typename T>
void Initial<T>::constant(OrderParameterComponent<T> *inOutDeltaComponent,
                          const T inRealPart, const T inImaginaryPart) {
  GridGPU<T> *deltaGrid = inOutDeltaComponent->getGrid();

  deltaGrid->setValue(inRealPart, Type::real);
  deltaGrid->setValue(inImaginaryPart, Type::imag);
}

template <typename T>
void Initial<T>::grid(OrderParameterComponent<T> *inOutDeltaComponent,
                      const GridGPU<T> &inGrid) {
  GridGPU<T> tmp = inGrid;

  GridGPU<T> *delta = inOutDeltaComponent->getGrid();

  if (tmp.dimX() != delta->dimX()) {
    tmp.resample(delta->dimX(), delta->dimY());
  }

  *delta = tmp;
}

template <typename T>
void Initial<T>::perturbedPhaseWinding(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inNumWinding,
    const T inRandomPhaseMix, const T inPhaseShift, const T inCenterX,
    const T inCenterY, const T inCutoffRadius) {
  // Get pointer to order parameter data.
  GridGPU<T> *deltaGrid = inOutDeltaComponent->getGrid();

  // Compute order parameter bulk value.
  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  // Used to compare floating-point numbers with "zero".
  const T epsilon = std::numeric_limits<T>::epsilon();

  // Number of lattice points in order parameter lattice.
  const int numLatticePoints = deltaGrid->dimX();

  // Calculate GPU thread blocks.
  dim3 blocksPerGrid, threadsPerBlock;
  deltaGrid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  // Initialize all order parameter components.
  for (int c = 0; c < deltaGrid->numFields(); ++c) {
    // Apply phase winding via GPU kernel.
    internal::perturbedPhaseWindingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deltaGrid->getDataPointer(c, Type::real),
        deltaGrid->getDataPointer(c, Type::imag), numLatticePoints,
        numLatticePoints, inNumWinding, inRandomPhaseMix, inPhaseShift,
        inCenterX, inCenterY, inCutoffRadius, deltaBulk, epsilon);
  }
}

template <typename T>
void Initial<T>::phaseWinding(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inNumWinding,
    const T inPhaseShift, const T inCenterX, const T inCenterY,
    const T inCutoffRadius) {
  const T randomPhaseMix = static_cast<T>(0);
  perturbedPhaseWinding(inOutDeltaComponent, inNumWinding, randomPhaseMix,
                        inPhaseShift, inCenterX, inCenterY, inCutoffRadius);
}

// Set the order parameter to a 'phase crystal', at the coordinates specified by
// the input parameters. See Eq. (12) of Phys. Rev. Research 2, 013104 (2020)
// for a definition of an edge phase crystal.
template <typename T>
void Initial<T>::edgePhaseCrystal(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inPeriod,
    const T inDecayLength, const T inCenterX, const T inCenterY,
    const T inRotAngle) {
  // Get pointer to order parameter data.
  GridGPU<T> *deltaGrid = inOutDeltaComponent->getGrid();

  // Number of lattice points in order parameter lattice.
  const int numLatticePoints = deltaGrid->dimX();

  // Compute order parameter bulk value.
  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  // Calculate phase crystal parameters
  const T inverseDecayLength = static_cast<T>(1.0) / inDecayLength;
  const T waveNumber = static_cast<T>(2.0 * M_PI) / inPeriod;

  // Thread dimensions
  dim3 blocksPerGrid, threadsPerBlock;
  deltaGrid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  // Initialize all order parameter components.
  for (int c = 0; c < deltaGrid->numFields(); ++c) {
    // Calculate and set order parameter component.
    internal::edgePhaseCrystalKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deltaGrid->getDataPointer(c, Type::real),
        deltaGrid->getDataPointer(c, Type::imag), numLatticePoints,
        numLatticePoints, waveNumber, inverseDecayLength, inCenterX, inCenterY,
        inRotAngle, deltaBulk);
  }
}

template <typename T>
void Initial<T>::random(OrderParameterComponent<T> *inOutDeltaComponent,
                        const T inRandomPhaseMix, const T inPhase) {
  const T phaseWinding = static_cast<T>(0);
  const T centerX = static_cast<T>(0);
  const T centerY = static_cast<T>(0);
  const T cutoffRadius = static_cast<T>(0);
  perturbedPhaseWinding(inOutDeltaComponent, phaseWinding, inRandomPhaseMix,
                        inPhase, centerX, centerY, cutoffRadius);
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *inOutDeltaComponent,
                     const char *const inFilename) {
  GridCPU<T> tmp;

  if (tmp.read(inFilename))
    return 1;

  GridGPU<T> *delta = inOutDeltaComponent->getGrid();

  if (tmp.dimX() != delta->dimX())
    tmp.resample(delta->dimX(), delta->dimY());

  *delta = tmp;

  return 0;
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *inOutDeltaComponent,
                     const char *const inFilename, const int inFileNumber) {
  char fn[512];
  sprintf(fn, "%s.%d.grd", inFilename, inFileNumber);

  return read(inOutDeltaComponent, fn);
}
} // namespace conga

#endif // ORDER_PARAMETER_INITIAL_H_
