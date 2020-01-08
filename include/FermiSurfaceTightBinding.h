#ifndef FERMISURFACETIGHTBINDING_H_
#define FERMISURFACETIGHTBINDING_H_

#include "FermiSurface.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <thrust/device_vector.h>

// Forward declaration, it is defined below.
template <typename> struct TightBindingParameters;

// Fermi surface from a tight-binding dispersion.
// Note, there are tacit assumptions:
// 1. A single species of atom
// 2. A Square lattice
template <typename T> class FermiSurfaceTightBinding : public FermiSurface<T> {
public:
  // Constructor.
  FermiSurfaceTightBinding();
  FermiSurfaceTightBinding(Context<T> *ctx);

  // Destructor.
  ~FermiSurfaceTightBinding();

  // Set the tight-binding parameters.
  void setTightBindingParameters(const T inHoppingNN, const T inHoppingNNN,
                                 const T inHoppingNNNN,
                                 const T inChemicalPotential);

  // Initialize the Fermi surface.
  int initialize();

  // Get the Fermi momentum.
  void getFermiMomentum(const int inAngleIdx, T &outMomentumX, T &outMomentumY);

  // Get the Fermi velocity.
  void getFermiVelocity(const int inAngleIdx, T &outVelocityX, T &outVelocityY);

  // Get the Fermi speed.
  T getFermiSpeed(const int inAngleIdx);

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx);

  // Get system angle.
  T getSystemAngle(const int inAngleIdx);
  T *getSystemAnglePointer();

  // Get momentum offset.
  void getMomentumOffset(T &outOffsetX, T &outOffsetY);

private:
  // We find the Fermi surface with an iterative method (bisection), so we
  // need to set the maximum number of iterations, and when we consider
  // ourselves done.
  const int m_numIterations = 100;
  const T m_convergence =
      static_cast<T>(10.0) * std::numeric_limits<T>::epsilon();

  // For integrand factor, phi derivative.
  // Note, this is radians.
  const T m_angleDelta = std::sqrt(std::numeric_limits<T>::epsilon());

  // Tight-binding parameters.
  TightBindingParameters<T> m_parameters;

  // For doped materials the Fermi surface can encircle the point (pi, pi)
  // instead of (0, 0), this is simply that offset.
  T m_offsetX;
  T m_offsetY;

  // The Fermi momentum.
  thrust::device_vector<T> m_momentumX;
  thrust::device_vector<T> m_momentumY;

  // The Fermi velocity.
  thrust::device_vector<T> m_velocityX;
  thrust::device_vector<T> m_velocityY;

  // The system angle, basically angle(vF) - pi/2.
  thrust::device_vector<T> m_systemAngle;

  // The momentum dependent factor in the integral.
  thrust::device_vector<T> m_integrandFactor;
};

// A helper struct.
template <typename T> struct TightBindingParameters {
  void print() const {
    std::cout << "Tight-Binding Parameters:\n";
    std::cout << "hoppingNN = " << hoppingNN << ", hoppingNNN = " << hoppingNNN
              << ", hoppingNNNN = " << hoppingNNNN
              << ", chemicalPotential = " << chemicalPotential << "\n";
  }

  // Nearest neighbour.
  T hoppingNN;

  // Next-nearest neighbour.
  T hoppingNNN;

  // Next-next-nearest neighbour.
  T hoppingNNNN;

  // Chemical potential (or rather, Fermi level).
  T chemicalPotential;
};

namespace detail {
template <typename T>
T computeDispersion(const TightBindingParameters<T> &inParameters,
                    const T inMomentumX, const T inMomentumY) {

  // Nearest neighbour.
  const T factorNN = -static_cast<T>(2.0) * inParameters.hoppingNN;
  const T termNN = factorNN * (std::cos(inMomentumX) + std::cos(inMomentumY));

  // Next-nearest neighbour.
  const T factorNNN = -static_cast<T>(4.0) * inParameters.hoppingNNN;
  const T termNNN = factorNNN * std::cos(inMomentumX) * std::cos(inMomentumY);

  // Next-next-nearest neighbour.
  const T factorNNNN = -static_cast<T>(2.0) * inParameters.hoppingNNNN;
  const T termNNNN = factorNNNN * (std::cos(static_cast<T>(2.0) * inMomentumX) +
                                   std::cos(static_cast<T>(2.0) * inMomentumY));

  return (termNN + termNNN + termNNNN - inParameters.chemicalPotential);
}

template <typename T>
void computeDispersionDerivative(const TightBindingParameters<T> &inParameters,
                                 const T inMomentumX, const T inMomentumY,
                                 T &outDerivativeX, T &outDerivativeY) {

  // Nearest neighbour.
  const T factorNN = static_cast<T>(2.0) * inParameters.hoppingNN;
  const T termNNdX = factorNN * std::sin(inMomentumX);
  const T termNNdY = factorNN * std::sin(inMomentumY);

  // Next-nearest neighbour.
  const T factorNNN = static_cast<T>(4.0) * inParameters.hoppingNNN;
  const T termNNNdX = factorNNN * std::sin(inMomentumX) * std::cos(inMomentumY);
  const T termNNNdY = factorNNN * std::cos(inMomentumX) * std::sin(inMomentumY);

  // Next-next-nearest neighbour.
  const T factorNNNN = static_cast<T>(4.0) * inParameters.hoppingNNNN;
  const T termNNNNdX = factorNNNN * std::sin(static_cast<T>(2.0) * inMomentumX);
  const T termNNNNdY = factorNNNN * std::sin(static_cast<T>(2.0) * inMomentumY);

  outDerivativeX = (termNNdX + termNNNdX + termNNNNdX);
  outDerivativeY = (termNNdY + termNNNdY + termNNNNdY);
}

template <typename T>
void computeMomentumOffset(const TightBindingParameters<T> &inParameters,
                           T &outOffsetX, T &outOffsetY) {
  // Compute the dispersion at k=(0,pi).
  const T kAX = static_cast<T>(0.0);
  const T kAY = static_cast<T>(M_PI);
  const T eA = computeDispersion(inParameters, kAX, kAY);

  // Compute the dispersion at k=(pi,pi).
  const T kBX = static_cast<T>(M_PI);
  const T kBY = static_cast<T>(M_PI);
  const T eB = computeDispersion(inParameters, kBX, kBY);

  // Check if they have the same sign.
  const bool sameSign = utils::sign(eA) == utils::sign(eB);

  if (sameSign) {
    // If they have the same sign then the Fermi surface encircles k=(0,0).
    outOffsetX = static_cast<T>(0.0);
    outOffsetY = static_cast<T>(0.0);
  } else {
    // If they don't have the same sign then the Fermi surface encircles
    // k=(pi,pi).
    outOffsetX = static_cast<T>(M_PI);
    outOffsetY = static_cast<T>(M_PI);
  }
}

template <typename T>
bool computeFermiMomentum(const TightBindingParameters<T> &inParameters,
                          const T inAngle, const T inOffsetX, const T inOffsetY,
                          const int inNumIterations, const T inConvergence,
                          T &outMomentumX, T &outMomentumY) {
  // For convenience. Sets the direction of the momentum.
  const T cosPhi = std::cos(inAngle);
  const T sinPhi = std::sin(inAngle);

  // For the bisection method we need to two end-points of the interval.
  // 1st point: The origin (which could be (pi, pi)).
  T kA = static_cast<T>(0.0);
  const T kAX = kA * cosPhi + inOffsetX;
  const T kAY = kA * sinPhi + inOffsetY;
  T eA = computeDispersion(inParameters, kAX, kAY);

  // 2nd point: The BZ "edge".
  T kB = static_cast<T>(M_PI) / std::max(std::abs(cosPhi), std::abs(sinPhi));
  const T kBX = kB * cosPhi + inOffsetX;
  const T kBY = kB * sinPhi + inOffsetY;
  T eB = computeDispersion(inParameters, kBX, kBY);

  // TODO(niclas): Make sure that the signs of eA and eB differ. Exit gracefully
  // otherwise.
  if (utils::sign(eA) == utils::sign(eB)) {
    printf("BISECTION ERROR! Signs equal.\n");
  }

  // Initial guess.
  T k = static_cast<T>(0.5) * (kA + kB);

  // Have we converged?
  bool isConverged = false;

  for (int i = 0; i < inNumIterations; ++i) {
    // Form the momentum.
    const T kX = k * cosPhi + inOffsetX;
    const T kY = k * sinPhi + inOffsetY;

    // Compute the dispersion.
    const T dispersion = computeDispersion(inParameters, kX, kY);

    // The error is simply the magnitude of the dispersion. We are looking for
    // zeros.
    const T error = std::abs(dispersion);

    // Do a bisection step or decide that we're done.
    if (error > inConvergence) {
      // Update the interval.
      if (utils::sign(dispersion) == utils::sign(eA)) {
        eA = dispersion;
        kA = k;
      } else {
        eB = dispersion;
        kB = k;
      }

      // Update the momentum.
      k = static_cast<T>(0.5) * (kA + kB);
    } else {
      // Success!!
      isConverged = true;
      break;
    }
  }

  // The momentum magnitude has to be positive. Otherwise we are implicitly
  // computing the momentum for another angle.
  const bool isSane = k > static_cast<T>(0.0);

  // The momentum has to be within the Brillouin zone (BZ). Otherwise we are
  // implicitly computing the momentum for another angle.
  const bool isWithinBZ =
      k * std::max(std::abs(cosPhi), std::abs(sinPhi)) <= static_cast<T>(M_PI);

  // Print info if failure.
  if (!isConverged) {
    printf("The Fermi momentum did not converge!\n");
  }
  if (!isSane) {
    printf("The Fermi momentum magnitude is negative!\n");
  }
  if (!isWithinBZ) {
    printf("The Fermi momentum is outside of BZ!\n");
  }

  // Form the momentum.
  outMomentumX = k * cosPhi + inOffsetX;
  outMomentumY = k * sinPhi + inOffsetY;

  // Return whether it succeeded or not.
  const bool success = (isConverged && isSane && isWithinBZ);
  return success;
}

template <typename T>
T computeIntegrandFactor(const TightBindingParameters<T> &inParameters,
                         const T inMomentumX, const T inMomentumY,
                         const T inVelocityX, const T inVelocityY,
                         const T inOffsetX, const T inOffsetY,
                         const int inNumIterations, const T inConvergence,
                         const T inDelta) {
  // Get the angle parameterizing the momentum.
  const T phi = std::atan2(inMomentumY - inOffsetY, inMomentumX - inOffsetX);

  // For the derivative we need to apply two small perturbations (in a three
  // stencil).
  const T phiPlusDelta = phi + inDelta;
  const T phiMinusDelta = phi - inDelta;

  // Compute pF(\phi + \delta).
  T pXPlusDelta;
  T pYPlusDelta;
  const bool successPlus = computeFermiMomentum(
      inParameters, phiPlusDelta, inOffsetX, inOffsetY, inNumIterations,
      inConvergence, pXPlusDelta, pYPlusDelta);

  // Compute pF(\phi - \delta).
  T pXMinusDelta;
  T pYMinusDelta;
  const bool successMinus = computeFermiMomentum(
      inParameters, phiMinusDelta, inOffsetX, inOffsetY, inNumIterations,
      inConvergence, pXMinusDelta, pYMinusDelta);

  const T pXDiff = pXPlusDelta - pXMinusDelta;
  const T pYDiff = pYPlusDelta - pYMinusDelta;

  // |\frac{\partial p_F}{\partial \phi}|
  const T dPdPhi =
      std::sqrt(pXDiff * pXDiff + pYDiff * pYDiff) / (2.0 * std::abs(inDelta));

  // Fermi speed.
  const T speed =
      std::sqrt(inVelocityX * inVelocityX + inVelocityY * inVelocityY);
  return dPdPhi / speed;
}
} // namespace detail

template <typename T>
FermiSurfaceTightBinding<T>::FermiSurfaceTightBinding()
    : FermiSurface<T>(), m_momentumX(0), m_momentumY(0), m_velocityX(0),
      m_velocityY(0), m_systemAngle(0), m_integrandFactor(0) {}

template <typename T>
FermiSurfaceTightBinding<T>::FermiSurfaceTightBinding(Context<T> *ctx)
    : FermiSurface<T>(ctx), m_momentumX(0), m_momentumY(0), m_velocityX(0),
      m_velocityY(0), m_systemAngle(0), m_integrandFactor(0) {}

template <typename T> FermiSurfaceTightBinding<T>::~FermiSurfaceTightBinding() {
  std::cout << "~FermiSurfaceTightBinding<T>\n";
}

template <typename T>
void FermiSurfaceTightBinding<T>::setTightBindingParameters(
    const T inHoppingNN, const T inHoppingNNN, const T inHoppingNNNN,
    const T inChemicalPotential) {
  m_parameters.hoppingNN = inHoppingNN;
  m_parameters.hoppingNNN = inHoppingNNN;
  m_parameters.hoppingNNNN = inHoppingNNNN;
  m_parameters.chemicalPotential = inChemicalPotential;
}

template <typename T> int FermiSurfaceTightBinding<T>::initialize() {
  std::cout << "FermiSurfaceTightBinding<T>::initialize()\n";

  // Print the tight-binding parameters.
  m_parameters.print();

  // Compute momentum offset.
  detail::computeMomentumOffset(m_parameters, m_offsetX, m_offsetY);
  printf("FS Offset: (x, y) = (%f, %f)\n", m_offsetX, m_offsetY);

  // Get the number of angles.
  const int numAngles = FermiSurface<T>::getNumAngles();

  // Fermi momentum array on the CPU.
  thrust::host_vector<T> momentumX(numAngles);
  thrust::host_vector<T> momentumY(numAngles);

  // Fermi velocity array on the CPU.
  thrust::host_vector<T> velocityX(numAngles);
  thrust::host_vector<T> velocityY(numAngles);

  // System angle array on the CPU.
  thrust::host_vector<T> systemAngle(numAngles);

  // Integrand factor array on the CPU.
  thrust::host_vector<T> integrandFactor(numAngles);

  T integrandFactorSum = static_cast<T>(0.0);
  T fermiSpeedSum = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // The momentum angle.
    const T phi = FermiSurface<T>::getPhi(angleIdx);

    // Compute the Fermi momentum.
    T pX;
    T pY;
    const bool success =
        detail::computeFermiMomentum(m_parameters, phi, m_offsetX, m_offsetY,
                                     m_numIterations, m_convergence, pX, pY);

    // Print stuff in case of failure.
    if (!success) {
      printf("ERROR! Fermi momentum failed at angleIdx = %d\n", angleIdx);
    }

    // The Fermi velocity is the partial derivative of the dispersion at the
    // Fermi surface.
    T vX;
    T vY;
    detail::computeDispersionDerivative(m_parameters, pX, pY, vX, vY);

    // Update the Fermi momentum arrays.
    momentumX[angleIdx] = pX;
    momentumY[angleIdx] = pY;

    // Update the Fermi velocity arrays.
    velocityX[angleIdx] = vX;
    velocityY[angleIdx] = vY;

    // Update the system angle array.
    systemAngle[angleIdx] = std::atan2(vY, vX) - static_cast<T>(0.5 * M_PI);

    // Compute integrand factor.
    const T factor = detail::computeIntegrandFactor(
        m_parameters, pX, pY, vX, vY, m_offsetX, m_offsetY, m_numIterations,
        m_convergence, m_angleDelta);
    integrandFactor[angleIdx] = factor;

    // For normalizing the integrand factor.
    integrandFactorSum += factor;

    // For computing the mean Fermi speed.
    fermiSpeedSum += std::sqrt(vX * vX + vY * vY) * factor;
  }

  // The average Fermi speed.
  const T avgFermiSpeed = fermiSpeedSum / integrandFactorSum;

  // Normalize the Fermi velocity and integrand factor.
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    velocityX[angleIdx] /= avgFermiSpeed;
    velocityY[angleIdx] /= avgFermiSpeed;
    integrandFactor[angleIdx] /= integrandFactorSum;
  }

  // Create device vectors.
  m_momentumX = thrust::device_vector<T>(momentumX.begin(), momentumX.end());
  m_momentumY = thrust::device_vector<T>(momentumY.begin(), momentumY.end());
  m_velocityX = thrust::device_vector<T>(velocityX.begin(), velocityX.end());
  m_velocityY = thrust::device_vector<T>(velocityY.begin(), velocityY.end());
  m_systemAngle =
      thrust::device_vector<T>(systemAngle.begin(), systemAngle.end());
  m_integrandFactor =
      thrust::device_vector<T>(integrandFactor.begin(), integrandFactor.end());
  return 0;
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiMomentum(const int inAngleIdx,
                                                   T &outMomentumX,
                                                   T &outMomentumY) {
  // Get the wrapped angle index.
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);

  outMomentumX = m_momentumX[idx];
  outMomentumY = m_momentumY[idx];
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiVelocity(const int inAngleIdx,
                                                   T &outVelocityX,
                                                   T &outVelocityY) {
  // Get the wrapped angle index.
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);

  // Get the velocity.
  outVelocityX = m_velocityX[idx];
  outVelocityY = m_velocityY[idx];
}

template <typename T>
T FermiSurfaceTightBinding<T>::getFermiSpeed(const int inAngleIdx) {
  T vX;
  T vY;
  getFermiVelocity(inAngleIdx, vX, vY);
  return std::sqrt(vX * vX + vY * vY);
}

template <typename T>
T FermiSurfaceTightBinding<T>::getSystemAngle(const int inAngleIdx) {
  // Get the wrapped angle index.
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);

  // Get the system angle.
  return m_systemAngle[idx];
}

template <typename T> T *FermiSurfaceTightBinding<T>::getSystemAnglePointer() {
  return thrust::raw_pointer_cast(m_systemAngle.data());
}

template <typename T>
T FermiSurfaceTightBinding<T>::getIntegrandFactor(const int inAngleIdx) {
  // Get the wrapped angle index.
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);

  // Get the integrand factor.
  return m_integrandFactor[idx];
}

template <typename T>
void FermiSurfaceTightBinding<T>::getMomentumOffset(T &outOffsetX,
                                                    T &outOffsetY) {
  outOffsetX = m_offsetX;
  outOffsetY = m_offsetY;
}

#endif // FERMISURFACETIGHTBINDING_H_
