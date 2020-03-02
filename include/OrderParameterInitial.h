//
// Collection of functions to set the initial condition of the order parameter.
// All member functions are static so this struct should not be instantiated.
//
#ifndef ORDER_PARAMETER_INITIAL_H_
#define ORDER_PARAMETER_INITIAL_H_

#include "OrderParameterComponent.h"
#include "defines.h"
#include "utils.h"
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <limits>

template <typename T> struct Initial {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // The following functions set the inOutDeltaComponent field to...

  // ...delta bulk
  static void bulk(OrderParameterComponent<T> *inOutDeltaComponent,
                   const T phase = 0.0);

  // ...specified complex valued constant
  static void constant(OrderParameterComponent<T> *inOutDeltaComponent,
                       const T re = 1.0, const T im = 0.0);

  // ...the input grid argument
  static void grid(OrderParameterComponent<T> *inOutDeltaComponent,
                   const GridGPU &grid);

  // ...a given phase winding (one flux vortex has a phase winding of 2pi)
  static void
  phaseWinding(OrderParameterComponent<T> *const inOutDeltaComponent,
               const T inNumWinding,
               const T inCutoffRadius = static_cast<T>(0.0),
               const T inPhaseShift = static_cast<T>(0.0),
               const T inCenterX = static_cast<T>(0.0),
               const T inCenterY = static_cast<T>(0.0));

  // ...an edge phase crystal.
  static void
  edgePhaseCrystal(OrderParameterComponent<T> *const inOutDeltaComponent,
                   const T inPeriod, const T inDecayLength, const T inCenterX,
                   const T inCenterY, const T inRotAngle);

  // ...delta bulk + a perturbation in the phase (to push solution out of
  // meta-stable initial state)
  static void random(OrderParameterComponent<T> *inOutDeltaComponent,
                     const T mix = 0.5, const T phase = 0.0);

  // Mixture between random() and phaseWinding():
  //     phaseWinding with an additional random phase factor
  static void
  perturbedPhaseWinding(OrderParameterComponent<T> *const inOutDeltaComponent,
                        const T numWinding,
                        const T inRandomPhaseMix = static_cast<T>(0.5),
                        const T inCutoffRadius = static_cast<T>(0.0),
                        const T inPhaseShift = static_cast<T>(0.0),
                        const T inCenterX = static_cast<T>(0.0),
                        const T inCenterY = static_cast<T>(0.0));

  // ...the grid in the input file argument
  static int read(OrderParameterComponent<T> *inOutDeltaComponent,
                  const char *filename);
  static int read(OrderParameterComponent<T> *inOutDeltaComponent,
                  const char *filename, int filenum);
};

namespace internal {
template <typename T>
__global__ void
phaseWinding_radial_Kernel(T *const outReal, T *const outImag, const int inNumX,
                           const int inNumY, const T inNumWinding,
                           const T inCutoffRadius, const T inPhaseShift,
                           const T inCenterX, const T inCenterY,
                           const T inDeltaBulk, const T inEpsilon) {
  // Coordinates to new resampling grid
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < inNumX && j < inNumY) {
    const int idx = j * inNumX + i;

    // Normalize coordinates to -0.5 <= x,y <= +0.5, and offset with center
    const T x = ((T)i / (T)(inNumX - 1) - (T)0.5) - inCenterX;
    const T y = ((T)j / (T)(inNumY - 1) - (T)0.5) - inCenterY;
    const T radius = sqrt(x * x + y * y);

    // Check if point is within cutoff, set to zero in that case. Otherwise,
    // compute angle and add phase shift.
    const bool isPointWithinCutoff = (radius < inCutoffRadius);
    const T phase =
        isPointWithinCutoff ? static_cast<T>(0.0) : atan2(y, x) + inPhaseShift;

    // Check if point is at the center. If so, set to zero, otherwise use
    // the bulk value.
    const bool isPointAtCenter = (radius < inEpsilon);
    const T magnitude = isPointAtCenter ? static_cast<T>(0.0) : inDeltaBulk;

    outReal[idx] = magnitude * cos(phase * inNumWinding);
    outImag[idx] = magnitude * sin(phase * inNumWinding);
  }
}

template <typename T>
__global__ void edgePhaseCrystal_kernel(
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
        xOrigin / (static_cast<T>(1.0) - qcs::GRAIN_SIZE_MARGIN);
    const T yOriginMargin =
        yOrigin / (static_cast<T>(1.0) - qcs::GRAIN_SIZE_MARGIN);

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
                      const T phase) {

  const T deltaBulk = inOutDeltaComponent->deltaBulk();
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();
  deltaGrid->setValue(deltaBulk * std::cos(phase), Type::real);
  deltaGrid->setValue(deltaBulk * std::sin(phase), Type::imag);
}

template <typename T>
void Initial<T>::constant(OrderParameterComponent<T> *inOutDeltaComponent,
                          const T re, const T im) {
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();

  deltaGrid->setValue(re, Type::real);
  deltaGrid->setValue(im, Type::imag);
}

template <typename T>
void Initial<T>::grid(OrderParameterComponent<T> *inOutDeltaComponent,
                      const GridGPU &grid) {
  GridGPU tmp = grid;

  GridGPU *delta = inOutDeltaComponent->getOrderParameterField();

  if (tmp.dimX() != delta->dimX()) {
    tmp.resample(delta->dimX(), delta->dimY());
  }

  *delta = tmp;
}

template <typename T>
void Initial<T>::phaseWinding(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inNumWinding,
    const T inCutoffRadius, const T inPhaseShift, const T inCenterX,
    const T inCenterY) {
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();

  const int numLatticePoints = deltaGrid->dimX();
  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  // GPU init
  dim3 blocksPerGrid, threadsPerBlock;
  deltaGrid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  // For checking if a point is at the center. The limit cannot be fetched from
  // within the kernel.
  const T epsilon = std::numeric_limits<T>::epsilon();

  if (inNumWinding > static_cast<T>(0.0)) {
    // Kernel launch
    for (int c = 0; c < deltaGrid->numFields(); ++c) {
      // Calc and set boundary grid
      internal::phaseWinding_radial_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
          deltaGrid->getDataPointer(c, Type::real),
          deltaGrid->getDataPointer(c, Type::imag), numLatticePoints,
          numLatticePoints, inNumWinding, inCutoffRadius, inPhaseShift,
          inCenterX, inCenterY, deltaBulk, epsilon);
    }
  } else {
    deltaGrid->setValue(deltaBulk, Type::real);
    deltaGrid->setValue(static_cast<T>(0.0), Type::imag);
  }
}

// Set the order parameter to a 'phase crystal', at the coordinates specified by
// the input parameters. See Eq. (12) of Phys. Rev. Research 2, 013104 (2020)
// for a definition of an edge phase crystal.
template <typename T>
void Initial<T>::edgePhaseCrystal(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inPeriod,
    const T inDecayLength, const T inCenterX, const T inCenterY,
    const T inRotAngle) {
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();

  const int numLatticePoints = deltaGrid->dimX();

  // Calculate bulk value
  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  // Calculate phase crystal parameters
  const T inverseDecayLength = static_cast<T>(1.0) / inDecayLength;
  const T waveNumber = static_cast<T>(2.0 * M_PI) / inPeriod;

  // Thread dimensions
  dim3 blocksPerGrid, threadsPerBlock;
  deltaGrid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  for (int c = 0; c < deltaGrid->numFields(); ++c) {
    // Calculate and set order parameter component
    internal::edgePhaseCrystal_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        deltaGrid->getDataPointer(c, Type::real),
        deltaGrid->getDataPointer(c, Type::imag), numLatticePoints,
        numLatticePoints, waveNumber, inverseDecayLength, inCenterX, inCenterY,
        inRotAngle, deltaBulk);
  }
}

template <typename T>
void Initial<T>::random(OrderParameterComponent<T> *inOutDeltaComponent,
                        const T mix, const T phase) {
  Parameters<T> *params = inOutDeltaComponent->getParameters();
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();

  const int N = deltaGrid->dimX();
  const T h = params->getGridElementSize();

  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  T *data_re, *data_im, rand_phase;
  int idx;

  // Kernel launch
  GridCPU *delta = 0;
  Grid<T>::CPU::initializeGrid(delta, deltaGrid);

  for (int c = 0; c < delta->numFields(); ++c) {
    data_re = delta->getDataPointer(c, Type::real);
    data_im = delta->getDataPointer(c, Type::imag);

    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        idx = j * N + i;

        rand_phase = ((T)rand() / (T)RAND_MAX - 0.5) * (T)2.0 * M_PI * mix;

        data_re[idx] = deltaBulk * std::cos(phase + rand_phase);
        data_im[idx] = deltaBulk * std::sin(phase + rand_phase);
      }
    }
  }
  *deltaGrid = *delta;
}

template <typename T>
void Initial<T>::perturbedPhaseWinding(
    OrderParameterComponent<T> *const inOutDeltaComponent, const T inNumWinding,
    const T inRandomPhaseMix, const T inCutoffRadius, const T inPhaseShift,
    const T inCenterX, const T inCenterY) {
  const Parameters<T> *const params = inOutDeltaComponent->getParameters();
  GridGPU *deltaGrid = inOutDeltaComponent->getOrderParameterField();

  const int N = deltaGrid->dimX();
  const T h = params->getGridElementSize();

  const T deltaBulk = inOutDeltaComponent->deltaBulk();

  // Kernel launch
  GridCPU *delta = 0;
  Grid<T>::CPU::initializeGrid(delta, deltaGrid);

  for (int c = 0; c < delta->numFields(); ++c) {
    T *const data_re = delta->getDataPointer(c, Type::real);
    T *const data_im = delta->getDataPointer(c, Type::imag);

    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        const int idx = j * N + i;

        // Normalize coordinates to -0.5 <= x,y <= +0.5, and offset with center
        const T x = ((T)i / (T)(N - 1) - (T)0.5) - inCenterX;
        const T y = ((T)j / (T)(N - 1) - (T)0.5) - inCenterY;
        const T radius = std::sqrt(x * x + y * y);

        // Check if point is within cutoff, set to zero in that case. Otherwise,
        // compute angle and add phase shift.
        const bool isPointWithinCutoff = (radius < inCutoffRadius);
        const T phase = isPointWithinCutoff ? static_cast<T>(0.0)
                                            : atan2(y, x) + inPhaseShift;

        // Check if point is at the center. If so, set to zero, otherwise use
        // the bulk value.
        const bool isPointAtCenter =
            (radius < std::numeric_limits<T>::epsilon());
        const T magnitude = isPointAtCenter ? static_cast<T>(0.0) : deltaBulk;

        // A random phase.
        const T rand_phase =
            ((T)rand() / (T)RAND_MAX - 0.5) * (T)2.0 * M_PI * inRandomPhaseMix;

        data_re[idx] = magnitude * std::cos(phase * inNumWinding + rand_phase);
        data_im[idx] = magnitude * std::sin(phase * inNumWinding + rand_phase);
      }
    }
  }
  *deltaGrid = *delta;
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *inOutDeltaComponent,
                     const char *filename) {
  GridCPU tmp;

  if (tmp.read(filename))
    return 1;

  GridGPU *delta = inOutDeltaComponent->getOrderParameterField();

  if (tmp.dimX() != delta->dimX())
    tmp.resample(delta->dimX(), delta->dimY());

  *delta = tmp;

  return 0;
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *inOutDeltaComponent,
                     const char *filename, int num) {
  char fn[512];
  sprintf(fn, "%s.%d.grd", filename, num);

  return read(inOutDeltaComponent, fn);
}

#endif // ORDER_PARAMETER_INITIAL_H_
