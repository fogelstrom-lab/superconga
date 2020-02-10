#ifndef FERMI_SURFACE_CIRCLE_H_
#define FERMI_SURFACE_CIRCLE_H_

#include "FermiSurface.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <math.h>
#include <thrust/device_vector.h>

// TODO: (Patric) Choose a circular fermisurface by default if the user does not
// specify any in the run file.
template <typename T> class FermiSurfaceCircle : public FermiSurface<T> {
public:
  // Constructor.
  FermiSurfaceCircle();
  FermiSurfaceCircle(Context<T> *ctx);

  // Destructor.
  ~FermiSurfaceCircle();

  // Initialize the Fermi surface.
  int initialize();

  // Get the Fermi momentum.
  void getFermiMomentum(const int inAngleIdx, T &outMomentumX, T &outMomentumY);

  // Get the Fermi velocity.
  void getFermiVelocity(const int inAngleIdx, T &outVelocityX, T &outVelocityY);

  // Get the Fermi speed.
  // This is a circle so it is a constant.
  T getFermiSpeed(const int inAngleIdx) { return static_cast<T>(1.0); };

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx);

  // Get system angle.
  T getSystemAngle(const int inAngleIdx);
  T *getSystemAnglePointer();

  // Get momentum offset.
  void getMomentumOffset(T &outOffsetX, T &outOffsetY);

private:
  // The system angle, basically angle(vF) - pi/2.
  // We need an array of all values because we need to pass it to a CUDA kernel
  // implementing the boundary condition.
  thrust::device_vector<T> m_systemAngle;
};

template <typename T>
FermiSurfaceCircle<T>::FermiSurfaceCircle()
    : FermiSurface<T>(), m_systemAngle(0) {}

template <typename T>
FermiSurfaceCircle<T>::FermiSurfaceCircle(Context<T> *ctx)
    : FermiSurface<T>(ctx), m_systemAngle(0) {}

template <typename T> FermiSurfaceCircle<T>::~FermiSurfaceCircle() {
  std::cout << "~FermiSurfaceCircle<T>\n";
}

template <typename T> int FermiSurfaceCircle<T>::initialize() {
  std::cout << "FermiSurfaceCircle<T>::initialize()\n";

  // Get the number of angles.
  const int numAngles = FermiSurface<T>::getNumAngles();

  // System angle array on the CPU.
  thrust::host_vector<T> systemAngle(numAngles);

  // Fill the array.
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    const T phi = FermiSurface<T>::getPhi(angleIdx);
    systemAngle[angleIdx] = FermiSurface<T>::getSystemAngle(phi);
  }

  // Copy to the GPU.
  m_systemAngle =
      thrust::device_vector<T>(systemAngle.begin(), systemAngle.end());
  return 0;
}

template <typename T>
void FermiSurfaceCircle<T>::getFermiMomentum(const int inAngleIdx,
                                             T &outMomentumX, T &outMomentumY) {
  // Get phi.
  const T phi = FermiSurface<T>::getPhi(inAngleIdx);

  // Remember, this is a circle.
  outMomentumX = std::cos(phi);
  outMomentumY = std::sin(phi);
}

template <typename T>
void FermiSurfaceCircle<T>::getFermiVelocity(const int inAngleIdx,
                                             T &outVelocityX, T &outVelocityY) {
  // Get phi.
  const T phi = FermiSurface<T>::getPhi(inAngleIdx);

  // Once again, a boring circle.
  outVelocityX = std::cos(phi);
  outVelocityY = std::sin(phi);
}

template <typename T>
T FermiSurfaceCircle<T>::getSystemAngle(const int inAngleIdx) {
  // Get phi.
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);

  // Set the system angle.
  return m_systemAngle[idx];
}

template <typename T> T *FermiSurfaceCircle<T>::getSystemAnglePointer() {
  return thrust::raw_pointer_cast(m_systemAngle.data());
}

template <typename T>
T FermiSurfaceCircle<T>::getIntegrandFactor(const int inAngleIdx) {
  // Because this is a circle, and we are taking equidistant angular steps, the
  // only thing left to do is dividing by the number of angles.
  const int numAngles = FermiSurface<T>::getNumAngles();
  return static_cast<T>(1.0) / static_cast<T>(numAngles);
}

template <typename T>
void FermiSurfaceCircle<T>::getMomentumOffset(T &outOffsetX, T &outOffsetY) {
  // This is a circle centered around the origin.
  outOffsetX = static_cast<T>(0.0);
  outOffsetY = static_cast<T>(0.0);
}

#endif // FERMI_SURFACE_CIRCLE_H_
