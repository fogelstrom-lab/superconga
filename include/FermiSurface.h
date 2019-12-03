#ifndef FERMISURFACE_H_
#define FERMISURFACE_H_

#include "Context.h"
#include "ContextModule.h"
#include "utils.h"
#include <cmath>
#include <math.h>

template <typename T> class FermiSurface : public ContextModule<T> {
public:
  // Constructor.
  FermiSurface();
  FermiSurface(Context<T> *ctx);

  // Destructor.
  virtual ~FermiSurface(){};

  // Initialize Fermi surface.
  virtual int initialize() = 0;

  // Get the polar angle.
  T getPhi(const int inAngleIdx);

  // Get the number of angles.
  int getNumAngles() { return m_numAngles; }

  // The wrapped angle index, i.e. in [0, numAngles - 1].
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

private:
  // The number of angles.
  int m_numAngles;
};

template <typename T>
FermiSurface<T>::FermiSurface() : ContextModule<T>(), m_numAngles(0) {}

template <typename T>
FermiSurface<T>::FermiSurface(Context<T> *ctx) : ContextModule<T>(ctx) {
  ctx->set(this);
  m_numAngles = ContextModule<T>::getParameters()->getAngularResolution();
}

template <typename T> T FermiSurface<T>::getPhi(const int inAngleIdx) {
  const int idx = getWrappedAngleIndex(inAngleIdx);
  return static_cast<T>(2.0 * M_PI) * static_cast<T>(idx) /
         static_cast<T>(m_numAngles);
}

template <typename T>
int FermiSurface<T>::getWrappedAngleIndex(const int inAngleIdx) {
  return utils::wrapToRange(inAngleIdx, 0, m_numAngles);
}

namespace fermisurface {
/*
This function is called from within a CUDA kernel. Thus it has to be a separate
function. Note, it completely relies on the Fermi surface being parameterized by
the polar angle of pF.
*/
template <typename T>
__host__ __device__ T getFractionalAngleIndex(const T inMomentumX,
                                              const T inMomentumY,
                                              const T inOffsetX,
                                              const T inOffsetY,
                                              const int inNumAngles) {
  // Wrap x-element.
  const T xLow = inOffsetX - static_cast<T>(M_PI);
  const T xHigh = inOffsetX + static_cast<T>(M_PI);
  const T kX = utils::wrapToRange(inMomentumX, xLow, xHigh);

  // Wrap y-element.
  const T yLow = inOffsetY - static_cast<T>(M_PI);
  const T yHigh = inOffsetY + static_cast<T>(M_PI);
  const T kY = utils::wrapToRange(inMomentumY, yLow, yHigh);

  // Compute unwrapped angle.
  const T phiUnwrapped = std::atan2(kY - inOffsetY, kX - inOffsetX);

  // Wrap angle.
  const T phi = utils::wrapToRangeTwoPi(phiUnwrapped);

  // Convert to index.
  return static_cast<T>(inNumAngles) * phi / static_cast<T>(2.0 * M_PI);
}
} // namespace fermisurface

#endif // FERMISURFACE_H_
