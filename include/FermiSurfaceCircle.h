#ifndef FERMI_SURFACE_CIRCLE_H_
#define FERMI_SURFACE_CIRCLE_H_

#include "FermiSurface.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>

// TODO: (Patric) Choose a circular fermisurface by default if the user does not
// specify any in the run file.
template <typename T> class FermiSurfaceCircle : public FermiSurface<T> {
public:
  // Constructor.
  FermiSurfaceCircle();

  // Destructor.
  ~FermiSurfaceCircle();

  // Initialize the Fermi surface.
  void initialize(const int inNumAngles);

  // Get the Fermi momentum.
  void getFermiMomentum(const int inAngleIdx, T &outMomentumX,
                        T &outMomentumY) const;

  // Get the Fermi velocity.
  void getFermiVelocity(const int inAngleIdx, T &outVelocityX,
                        T &outVelocityY) const;

  // Get the Fermi speed.
  // This is a circle so it is a constant.
  T getFermiSpeed(const int inAngleIdx) const { return static_cast<T>(1.0); }

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx) const;

  // Get system angle.
  T getSystemAngle(const int inAngleIdx) const;
  const T *getSystemAnglePointer() const;

  // Get momentum offset.
  void getMomentumOffset(T &outOffsetX, T &outOffsetY) const;

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
  std::cout << "~FermiSurfaceCircle<T>\n";
}

template <typename T>
void FermiSurfaceCircle<T>::initialize(const int inNumAngles) {
  std::cout << "FermiSurfaceCircle<T>::initialize()\n";

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
void FermiSurfaceCircle<T>::getFermiMomentum(const int inAngleIdx,
                                             T &outMomentumX,
                                             T &outMomentumY) const {
  // Get phi.
  const T phi = FermiSurface<T>::getPolarAngle(inAngleIdx);

  // Remember, this is a circle.
  outMomentumX = std::cos(phi);
  outMomentumY = std::sin(phi);
}

template <typename T>
void FermiSurfaceCircle<T>::getFermiVelocity(const int inAngleIdx,
                                             T &outVelocityX,
                                             T &outVelocityY) const {
  // Get phi.
  const T phi = FermiSurface<T>::getPolarAngle(inAngleIdx);

  // Once again, a boring circle.
  outVelocityX = std::cos(phi);
  outVelocityY = std::sin(phi);
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
T FermiSurfaceCircle<T>::getIntegrandFactor(const int inAngleIdx) const {
  // Because this is a circle, and we are taking equidistant angular steps, the
  // only thing left to do is dividing by the number of angles.
  const int numAngles = FermiSurface<T>::getNumAngles();
  return static_cast<T>(1.0) / static_cast<T>(numAngles);
}

template <typename T>
void FermiSurfaceCircle<T>::getMomentumOffset(T &outOffsetX,
                                              T &outOffsetY) const {
  // This is a circle centered around the origin.
  outOffsetX = static_cast<T>(0.0);
  outOffsetY = static_cast<T>(0.0);
}

#endif // FERMI_SURFACE_CIRCLE_H_
