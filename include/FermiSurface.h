//===-- FermiSurface.h - Abstract Fermi surface class ---------------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the abstract implementation of Fermi surface.
///
//===----------------------------------------------------------------------===//
#ifndef FERMI_SURFACE_H_
#define FERMI_SURFACE_H_

#include "Context.h"
#include "ContextModule.h"
#include "utils.h"
#include <cmath>

template <typename T> class FermiSurface : public ContextModule<T> {
public:
  // Constructor.
  FermiSurface(const int inNumAngles);
  FermiSurface(Context<T> *const inContext);

  // Destructor.
  virtual ~FermiSurface(){};

  // Initialize Fermi surface.
  virtual int initialize() = 0;

  // Get the polar angle.
  T getPhi(const int inAngleIdx);

  // Get the angle we should rotate the system for a given angle of the Fermi
  // velocity.
  T getSystemAngle(const T inFermiVelocityAngle);

  // Get the angle offset, i.e. the angle for index 0.
  T getAngleOffset() { return m_angleOffset; }

  // Get the number of angles.
  int getNumAngles() { return m_numAngles; }

  // The wrapped angle index, i.e. in [0, numAngles).
  int getWrappedAngleIndex(const int inAngleIdx);

  // Get the Fermi momentum.
  virtual void getFermiMomentum(const int inAngleIdx, T &outMomentumX,
                                T &outMomentumY) = 0;

  // Get the Fermi velocity.
  virtual void getFermiVelocity(const int inAngleIdx, T &outVelocityX,
                                T &outVelocityY) = 0;

  // Get the Fermi speed.
  virtual T getFermiSpeed(const int inAngledx) = 0;

  // Get the integrand factor.
  virtual T getIntegrandFactor(const int inAngleIdx) = 0;

  // Get system angle.
  virtual T getSystemAngle(const int inAngleIdx) = 0;
  virtual T *getSystemAnglePointer() = 0;

  // Get momentum offset.
  virtual void getMomentumOffset(T &outOffsetX, T &outOffsetY) = 0;

  // This function is called from within a CUDA kernel. Thus it has to be a
  // static method. Note, it completely relies on the Fermi surface being
  // parameterized by the polar angle of pF.
  __host__ __device__ static T getFractionalAngleIndex(const T inMomentumX,
                                                       const T inMomentumY,
                                                       const T inOffsetX,
                                                       const T inOffsetY,
                                                       const int inNumAngles);

private:
  // This is the angular offset of phi (the Fermi momentum angle) of the
  // momentum with index zero (i.e. the first momentum). One would think that
  // this offset is completely unnecessary, which it should be, but it is not.
  // The original code had this offset implicitly. Probabably something else
  // depends on this offset being there.
  // TODO(niclas): Investigate this! It would be nice to have it set to zero.
  static constexpr T m_angleOffset = static_cast<T>(0.5 * M_PI);

  // The number of angles.
  int m_numAngles;
};

namespace internal {
/// \brief Get the angle phi corresponding to some angle index
///
/// \param inAngleIdx - The angle index to wrap
/// \param inNumAngles - The total number of angles
/// \param inAngleOffset - The angle which has index 0
///
/// \return - The angle phi corresponding to inAngleIdx
template <typename T>
T getPhi(const int inAngleIdx, const int inNumAngles, const T inAngleOffset) {
  const T phiUnwrapped = static_cast<T>(M_PI) * static_cast<T>(2 * inAngleIdx) /
                             static_cast<T>(inNumAngles) +
                         inAngleOffset;
  return utils::wrapToRangeTwoPi(phiUnwrapped);
}

/// \brief Get the system angle corresponding to a Fermi velocity angle
///
/// Because the trajectories go along the positive y-axis, the Fermi velocity
/// angle and the system angle just differ by pi/2.
///
/// \param inFermiVelocityAngle - The polar angle of the Fermi velocity
///
/// \return - The system angle
template <typename T> T getSystemAngle(const T inFermiVelocityAngle) {
  return inFermiVelocityAngle - static_cast<T>(0.5 * M_PI);
}

/// \brief Compute the fractional angle index
///
/// Compute the fractional index a momentum corresponds to
///
/// \param inMomentumX - The x-component of the momentum
/// \param inMomentumY - The y-component of the momentum
/// \param inOffsetX - The offset of the Fermi surface origin along the x-axis
/// \param inOffsetY - The offset of the Fermi surface origin along the y-axis
/// \param inAngleOffset - The angle which has index 0
/// \param inNumAngles - The total number of angles
///
/// \return - The fractional angle index in the range [0.0, inNumAngles)
template <typename T>
__host__ __device__ T getFractionalAngleIndex(
    const T inMomentumX, const T inMomentumY, const T inOffsetX,
    const T inOffsetY, const T inAngleOffset, const int inNumAngles) {
  // Wrap x-component.
  const T xLow = inOffsetX - static_cast<T>(M_PI);
  const T xHigh = inOffsetX + static_cast<T>(M_PI);
  const T kX = utils::wrapToRange(inMomentumX, xLow, xHigh);

  // Wrap y-component.
  const T yLow = inOffsetY - static_cast<T>(M_PI);
  const T yHigh = inOffsetY + static_cast<T>(M_PI);
  const T kY = utils::wrapToRange(inMomentumY, yLow, yHigh);

  // Compute unwrapped angle.
  const T phiUnwrapped = atan2(kY - inOffsetY, kX - inOffsetX) - inAngleOffset;

  // Wrap angle.
  const T phi = utils::wrapToRangeTwoPi(phiUnwrapped);

  // Convert to index.
  return static_cast<T>(inNumAngles) * phi / static_cast<T>(2.0 * M_PI);
}
} // namespace internal

template <typename T>
FermiSurface<T>::FermiSurface(const int inNumAngles)
    : ContextModule<T>(), m_numAngles(inNumAngles) {}

template <typename T>
FermiSurface<T>::FermiSurface(Context<T> *const inContext)
    : ContextModule<T>(inContext) {
  inContext->set(this);
  m_numAngles = ContextModule<T>::getParameters()->getAngularResolution();
}

template <typename T> T FermiSurface<T>::getPhi(const int inAngleIdx) {
  return internal::getPhi(inAngleIdx, m_numAngles, m_angleOffset);
}

template <typename T>
T FermiSurface<T>::getSystemAngle(const T inFermiVelocityAngle) {
  return internal::getSystemAngle(inFermiVelocityAngle);
}

template <typename T>
int FermiSurface<T>::getWrappedAngleIndex(const int inAngleIdx) {
  return utils::wrapToRange(inAngleIdx, 0, m_numAngles);
}

template <typename T>
__host__ __device__ T FermiSurface<T>::getFractionalAngleIndex(
    const T inMomentumX, const T inMomentumY, const T inOffsetX,
    const T inOffsetY, const int inNumAngles) {
  return internal::getFractionalAngleIndex(inMomentumX, inMomentumY, inOffsetX,
                                           inOffsetY, m_angleOffset,
                                           inNumAngles);
}

#endif // FERMI_SURFACE_H_
