//===-- FermiSurfaceCircular.h - Circular Fermi surface class -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of a circular Fermi surface.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_FERMI_SURFACE_CIRCLE_H_
#define CONGA_FERMI_SURFACE_CIRCLE_H_

#include "../defines.h"
#include "../utils.h"
#include "FermiSurface.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <iostream>

namespace conga {
template <typename T> class FermiSurfaceCircle : public FermiSurface<T> {
public:
  // Constructor.
  FermiSurfaceCircle();

  // Destructor.
  ~FermiSurfaceCircle();

  // Initialize the Fermi surface.
  void initialize(const int inNumAngles) override;

  // Get the type.
  FermiSurfaceType getType() const override { return FermiSurfaceType::CIRCLE; }

  // Get the Fermi momentum.
  vector2d<T> getFermiMomentum(const int inAngleIdx) const override;
  void getFermiMomentum(const int inAngleIdx, T &outMomentumX,
                        T &outMomentumY) const override;

  // Get the Fermi velocity.
  vector2d<T> getFermiVelocity(const int inAngleIdx) const override;
  void getFermiVelocity(const int inAngleIdx, T &outVelocityX,
                        T &outVelocityY) const override;

  // Get the Fermi speed.
  // This is a circle so it is a constant.
  T getFermiSpeed(const int inAngleIdx) const override;

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx) const override;

  // Get system angle.
  T getSystemAngle(const int inAngleIdx) const override;
  const T *getSystemAnglePointer() const override;

  // Get momentum offset.
  vector2d<T> getMomentumOffset() const override;
  void getMomentumOffset(T &outOffsetX, T &outOffsetY) const override;

private:
  // The system angles, basically angle(vF) - pi/2.
  thrust::host_vector<T> m_systemAngles;
  // The system angles need to be on the GPU as well because the entire array is
  // passed to a CUDA kernel.
  thrust::device_vector<T> m_systemAnglesGPU;
};

template <typename T>
FermiSurfaceCircle<T>::FermiSurfaceCircle()
    : FermiSurface<T>(), m_systemAngles(0), m_systemAnglesGPU(0) {}

template <typename T> FermiSurfaceCircle<T>::~FermiSurfaceCircle() {
#ifndef NDEBUG
  std::cout << "~FermiSurfaceCircle<T>\n";
#endif
}

template <typename T>
void FermiSurfaceCircle<T>::initialize(const int inNumAngles) {
#ifndef NDEBUG
  std::cout << "FermiSurfaceCircle<T>::initialize()\n";
#endif

  // The base class needs to know the number of angles.
  FermiSurface<T>::initialize(inNumAngles);

  // System angle array on the CPU.
  m_systemAngles.clear();

  // Fill the array.
  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    const T phi = FermiSurface<T>::getPolarAngle(angleIdx);
    const T systemAngle = FermiSurface<T>::getSystemAngle(phi);
    m_systemAngles.push_back(systemAngle);
  }

  // Copy system angles to the GPU.
  m_systemAnglesGPU = m_systemAngles;
}

template <typename T>
vector2d<T>
FermiSurfaceCircle<T>::getFermiMomentum(const int inAngleIdx) const {
  // Get phi.
  const T phi = FermiSurface<T>::getPolarAngle(inAngleIdx);

  // Remember, this is a circle.
  return {std::cos(phi), std::sin(phi)};
}

template <typename T>
void FermiSurfaceCircle<T>::getFermiMomentum(const int inAngleIdx,
                                             T &outMomentumX,
                                             T &outMomentumY) const {
  const vector2d<T> fermiMomentum = getFermiMomentum(inAngleIdx);
  outMomentumX = fermiMomentum[0];
  outMomentumY = fermiMomentum[1];
}

template <typename T>
vector2d<T>
FermiSurfaceCircle<T>::getFermiVelocity(const int inAngleIdx) const {
  // Momentum and velocity are equal.
  return getFermiMomentum(inAngleIdx);
}

template <typename T>
void FermiSurfaceCircle<T>::getFermiVelocity(const int inAngleIdx,
                                             T &outVelocityX,
                                             T &outVelocityY) const {
  // Momentum and velocity are equal.
  getFermiMomentum(inAngleIdx, outVelocityX, outVelocityY);
}

template <typename T>
T FermiSurfaceCircle<T>::getSystemAngle(const int inAngleIdx) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  return m_systemAngles[idx];
}

template <typename T>
const T *FermiSurfaceCircle<T>::getSystemAnglePointer() const {
  return thrust::raw_pointer_cast(m_systemAnglesGPU.data());
}

template <typename T>
T FermiSurfaceCircle<T>::getFermiSpeed(const int inAngleIdx) const {
  // Unused.
  (void)inAngleIdx;
  return static_cast<T>(1.0);
}

template <typename T>
T FermiSurfaceCircle<T>::getIntegrandFactor(const int inAngleIdx) const {
  // Unused.
  (void)inAngleIdx;

  // Because this is a circle, and we are taking equidistant angular steps, the
  // only thing left to do is dividing by the number of angles.
  const int numAngles = FermiSurface<T>::getNumAngles();
  return static_cast<T>(1.0) / static_cast<T>(numAngles);
}

template <typename T>
vector2d<T> FermiSurfaceCircle<T>::getMomentumOffset() const {
  // This is a circle centered around the origin.
  return {static_cast<T>(0.0), static_cast<T>(0.0)};
}

template <typename T>
void FermiSurfaceCircle<T>::getMomentumOffset(T &outOffsetX,
                                              T &outOffsetY) const {
  const vector2d<T> offset = getMomentumOffset();
  outOffsetX = offset[0];
  outOffsetY = offset[1];
}
} // namespace conga

#endif // CONGA_FERMI_SURFACE_CIRCLE_H_
