#ifndef FERMI_SURFACE_TIGHT_BINDING_H_
#define FERMI_SURFACE_TIGHT_BINDING_H_

#include "FermiSurface.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <utility>

// Forward declaration, it is defined below.
template <typename> struct TightBindingParameters;

// Fermi surface from a tight-binding dispersion.
// Note, there are tacit assumptions:
// 1. A single species of atom.
// 2. A square lattice.
template <typename T> class FermiSurfaceTightBinding : public FermiSurface<T> {
public:
  // Constructor.
  FermiSurfaceTightBinding(const T inHoppingNNN = static_cast<T>(0.0),
                           const T inHoppingNNNN = static_cast<T>(0.0),
                           const T inChemicalPotential = static_cast<T>(1.0));

  // Destructor.
  ~FermiSurfaceTightBinding();

  // Initialize the Fermi surface.
  void initialize(const int inNumAngles);

  // Get the Fermi momentum.
  void getFermiMomentum(const int inAngleIdx, T &outFermiMomentumX,
                        T &outFermiMomentumY) const;

  // Get the Fermi velocity.
  void getFermiVelocity(const int inAngleIdx, T &outFermiVelocityX,
                        T &outFermiVelocityY) const;

  // Get the Fermi speed.
  T getFermiSpeed(const int inAngleIdx) const;

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx) const;

  // Get system angle.
  T getSystemAngle(const int inAngleIdx) const;
  const T *getSystemAnglePointer() const;

  // Get momentum offset.
  void getMomentumOffset(T &outOffsetX, T &outOffsetY) const;

  // Only used for tests.
  T getAvgFermiMomentum() const { return m_avgFermiMomentum; }
  T getAvgFermiSpeed() const { return m_avgFermiSpeed; }

private:
  // We find the Fermi surface with an iterative method (bisection and Newton),
  // so we need to set the maximum number of iterations, and when we consider
  // ourselves done.
  const int m_maxIterations = 100;

  // The convergence criterion could probably be descreased, but not by much.
  const T m_convergence =
      static_cast<T>(1e2) * std::numeric_limits<T>::epsilon();

  // Tight-binding parameters.
  const TightBindingParameters<T> m_parameters;

  // For doped materials the Fermi surface can encircle the point (pi, pi)
  // instead of (0, 0), this is simply that offset.
  std::pair<T, T> m_originOffset;

  // The Fermi momentum.
  thrust::host_vector<T> m_fermiMomentumX;
  thrust::host_vector<T> m_fermiMomentumY;

  // The Fermi velocity.
  thrust::host_vector<T> m_fermiVelocityX;
  thrust::host_vector<T> m_fermiVelocityY;

  // The system angles, basically angle(vF) - pi/2.
  thrust::host_vector<T> m_systemAngles;
  // The system angles need to be on the GPU as well because the entire array is
  // passed to a CUDA kernel.
  thrust::device_vector<T> m_systemAnglesGPU;

  // The momentum dependent factor in the integral.
  thrust::host_vector<T> m_integrandFactor;

  // Only used for tests.
  T m_avgFermiMomentum;
  T m_avgFermiSpeed;
};

// A helper struct.
template <typename T> struct TightBindingParameters {
  TightBindingParameters(){};
  TightBindingParameters(const T inHoppingNNN, const T inHoppingNNNN,
                         const T inChemicalPotential)
      : hoppingNNN(inHoppingNNN), hoppingNNNN(inHoppingNNNN),
        chemicalPotential(inChemicalPotential){};

  void print() const {
    std::cout << "Tight-Binding Parameters:" << std::endl;
    std::cout << "hoppingNN = " << hoppingNN << ", hoppingNNN = " << hoppingNNN
              << ", hoppingNNNN = " << hoppingNNNN
              << ", chemicalPotential = " << chemicalPotential << std::endl;
  }

  // Nearest neighbour hopping. It is always 1, all other hopping terms, and the
  // chemical potential, is expressed in terms of it. It is out energy scale.
  const T hoppingNN = static_cast<T>(1.0);

  // Next-nearest neighbour.
  T hoppingNNN;

  // Next-next-nearest neighbour.
  T hoppingNNNN;

  // Chemical potential (or rather, Fermi level).
  T chemicalPotential;
};

namespace tightbinding_internal {
/// \brief Compute the value of the dispersion at a given momentum
///
/// Evaluates the dispersion, which is parameterized by inParameters, at the
/// momentum inMomentum.
///
/// \param inParameters - The tight-binding parameters for a square lattice
/// \param inMomentum - The momentum as a pair(x, y)
///
/// \return - The value of the dispersion at inMomentum
template <typename T>
T computeDispersion(const TightBindingParameters<T> &inParameters,
                    const std::pair<T, T> &inMomentum) {
  const T momentumX = inMomentum.first;
  const T momentumY = inMomentum.second;

  // Nearest neighbour.
  const T factorNN = -static_cast<T>(2.0) * inParameters.hoppingNN;
  const T termNN = factorNN * (std::cos(momentumX) + std::cos(momentumY));

  // Next-nearest neighbour.
  const T factorNNN = -static_cast<T>(4.0) * inParameters.hoppingNNN;
  const T termNNN = factorNNN * std::cos(momentumX) * std::cos(momentumY);

  // Next-next-nearest neighbour.
  const T factorNNNN = -static_cast<T>(2.0) * inParameters.hoppingNNNN;
  const T termNNNN = factorNNNN * (std::cos(static_cast<T>(2.0) * momentumX) +
                                   std::cos(static_cast<T>(2.0) * momentumY));

  return (termNN + termNNN + termNNNN - inParameters.chemicalPotential);
}

/// \brief Compute the partial derivatives of the dispersion at a given momentum
///
/// \param inParameters - The tight-binding parameters for a square lattice
/// \param inMomentum - The momentum as a pair (x, y)
///
/// \return - The partial derivatives (dx, dy) of the dispersion at
/// inMomentum
template <typename T>
std::pair<T, T>
computeDispersionDerivative(const TightBindingParameters<T> &inParameters,
                            const std::pair<T, T> &inMomentum) {
  const T momentumX = inMomentum.first;
  const T momentumY = inMomentum.second;

  // Nearest neighbour.
  const T factorNN = static_cast<T>(2.0) * inParameters.hoppingNN;
  const T termNNdX = factorNN * std::sin(momentumX);
  const T termNNdY = factorNN * std::sin(momentumY);

  // Next-nearest neighbour.
  const T factorNNN = static_cast<T>(4.0) * inParameters.hoppingNNN;
  const T termNNNdX = factorNNN * std::sin(momentumX) * std::cos(momentumY);
  const T termNNNdY = factorNNN * std::cos(momentumX) * std::sin(momentumY);

  // Next-next-nearest neighbour.
  const T factorNNNN = static_cast<T>(4.0) * inParameters.hoppingNNNN;
  const T termNNNNdX = factorNNNN * std::sin(static_cast<T>(2.0) * momentumX);
  const T termNNNNdY = factorNNNN * std::sin(static_cast<T>(2.0) * momentumY);

  const T dX = (termNNdX + termNNNdX + termNNNNdX);
  const T dY = (termNNdY + termNNNdY + termNNNNdY);
  return {dX, dY};
}

/// \brief Compute the momentum offset for this dispersion
///
/// The Fermi surface (given reasonable parameters) is either centered around
/// (0,0) or (pi,pi), this function computes that offset.
///
/// \param inParameters - The tight-binding parameters for a square lattice
///
/// \return - The offset, either (0,0) or (pi, pi)
template <typename T>
std::pair<T, T>
computeMomentumOffset(const TightBindingParameters<T> &inParameters) {
  // Compute the dispersion at k=(0,pi).
  const std::pair<T, T> kA =
      std::make_pair(static_cast<T>(0.0), static_cast<T>(M_PI));
  const T eA = computeDispersion(inParameters, kA);

  // Compute the dispersion at k=(pi,pi).
  const std::pair<T, T> kB =
      std::make_pair(static_cast<T>(M_PI), static_cast<T>(M_PI));
  const T eB = computeDispersion(inParameters, kB);

  // Check if they have the same sign.
  const bool sameSign = utils::sign(eA) == utils::sign(eB);

  if (sameSign) {
    // If they have the same sign then the Fermi surface encircles k=(0,0).
    return {static_cast<T>(0.0), static_cast<T>(0.0)};
  } else {
    // If they don't have the same sign then the Fermi surface encircles
    // k=(pi,pi).
    return {static_cast<T>(M_PI), static_cast<T>(M_PI)};
  }
}

/// \brief Make momentum pair (x,y) from magnitude, angle, and offset
///
/// The momentum is parameterized by:
/// kx = magnitude * cos(angle) + offsetX
/// ky = magnitude * sin(angle) + offsetY
///
/// \param inMagnitude - The magnitude
/// \param inAngle - The angle in radians
/// \param inOffset - The offset (x, y)
///
/// \return - The momentum as a pair (x, y)
template <typename T>
std::pair<T, T> makeMomentum(const T inMagnitude, const T inAngle,
                             const std::pair<T, T> &inOffset,
                             const bool inWrapToBrillouinZone = false) {
  T x = inMagnitude * std::cos(inAngle) + inOffset.first;
  T y = inMagnitude * std::sin(inAngle) + inOffset.second;
  if (inWrapToBrillouinZone) {
    x = utils::wrapToRange(x, -static_cast<T>(M_PI), static_cast<T>(M_PI));
    y = utils::wrapToRange(y, -static_cast<T>(M_PI), static_cast<T>(M_PI));
  }
  return {x, y};
}

/// \brief Compute the Fermi momentum for a given dispersion and direction
///
/// Essentially it computes k in
/// kx = k * cos(angle) + offsetX
/// ky = k * sin(angle) + offsetY
/// where k is given as the root of the dispersion going from the offset along a
/// ray with the given angle.
///
/// \param inParameters - The tight-binding parameters for a square lattice
/// \param inAngle - The angle in radians
/// \param inOffset - The offset (x, y)
/// \param inNumIterations - The maximum number of iterations
/// \param inConvergenCriterion - The convergence criterion. We stop when the
///                               error is smaller than this
///
/// \return - The Fermi momentum as a pair (x, y)
template <typename T>
std::pair<T, T> computeFermiMomentum(
    const TightBindingParameters<T> &inParameters, const T inAngle,
    const std::pair<T, T> inOriginOffset, const int inNumIterations = 100,
    const T inConvergenCriterion = static_cast<T>(1e2) *
                                   std::numeric_limits<T>::epsilon(),
    const bool inWrapToBrillouinZone = true) {
  // For the bisection method we need to two end-points of the interval.
  // 1st point: The origin (which could be (pi, pi)).
  T kA = static_cast<T>(0.0);
  const std::pair<T, T> momentumA = makeMomentum(kA, inAngle, inOriginOffset);
  T eA = computeDispersion(inParameters, momentumA);

  // 2nd point: The BZ "edge".
  const T maxAbsComponent =
      std::max(std::abs(std::cos(inAngle)), std::abs(std::sin(inAngle)));
  T kB = static_cast<T>(M_PI) / maxAbsComponent;
  const std::pair<T, T> momentumB = makeMomentum(kB, inAngle, inOriginOffset);
  T eB = computeDispersion(inParameters, momentumB);

  // TODO(niclas): Make sure that the signs of eA and eB differ. Exit gracefully
  // otherwise.
  if (utils::sign(eA) == utils::sign(eB)) {
    std::cout << "ERROR! Failure of the bisection method." << std::endl;
  }

  // Initial guess.
  T k = static_cast<T>(0.5) * (kA + kB);

  // Have we converged?
  bool isConverged = false;

  // Should we do a bisection or Newton step?
  bool doBisectionStep = true;

  for (int i = 0; i < inNumIterations; ++i) {
    // Form the momentum.
    const std::pair<T, T> momentum = makeMomentum(k, inAngle, inOriginOffset);

    // Compute the dispersion.
    const T dispersion = computeDispersion(inParameters, momentum);

    // The error is simply the magnitude of the dispersion.
    // We are looking for zeros.
    const T error = std::abs(dispersion);

    // Decide what to do depending on the error.
    if ((error > std::sqrt(inConvergenCriterion)) && doBisectionStep) {
      // Do a bisection step.

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
    } else if (error > inConvergenCriterion) {
      // Do a newton step.

      // Continue with Newton steps.
      doBisectionStep = false;

      // Compute the gradient along the radial direction.
      const std::pair<T, T> dE =
          computeDispersionDerivative(inParameters, momentum);
      // dXdK = cos(phi), dYdK = sin(phi)
      const T dEdK =
          dE.first * std::cos(inAngle) + dE.second * std::sin(inAngle);

      // Update the momentum.
      if (std::abs(dEdK) > static_cast<T>(0.0)) {
        k -= dispersion / dEdK;
      } else {
        // We cannot go any further.
        break;
      }
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
  const bool isWithinBZ = k * maxAbsComponent <= static_cast<T>(M_PI);

  // TODO(niclas): Raise exception instead!
  if (!isConverged) {
    std::cout << "ERROR! The Fermi momentum did not converge!" << std::endl;
  }
  if (!isSane) {
    std::cout << "ERROR! The Fermi momentum magnitude is negative!"
              << std::endl;
  }
  if (!isWithinBZ) {
    std::cout << "ERROR! The Fermi momentum is outside of Brillouin zone!"
              << std::endl;
  }

  // Form the momentum.
  return makeMomentum(k, inAngle, inOriginOffset, inWrapToBrillouinZone);
}

/// \brief Compute the integrand factor for a Fermi momentum
///
/// The factor is given by the norm of the partial derivative of the Fermi
/// momentum with the respect to the angle phi, divided by the norm of the Fermi
/// velocity.
///
/// \param inParameters - The tight-binding parameters for a square lattice
/// \param inFermiMomentum - The Fermi momentum (x, y)
/// \param inFermiVelocity - The Fermi velocity (x, y)
/// \param inOffset - The momentum offset (x, y)
///
/// \return - The integrand factor
template <typename T>
T computeIntegrandFactor(const TightBindingParameters<T> &inParameters,
                         const std::pair<T, T> &inFermiMomentum,
                         const std::pair<T, T> &inFermiVelocity,
                         const std::pair<T, T> &inOffset) {
  // Compute the partial derivatives of the dispersion.
  const std::pair<T, T> partialDerivatives =
      computeDispersionDerivative(inParameters, inFermiMomentum);
  const T dEdKX = partialDerivatives.first;
  const T dEdKY = partialDerivatives.second;

  // The momentum is parameterized as
  // pX = kX + offsetX
  // pY = kY + offsetY
  // with
  // kX = k * cos(phi)
  // kY = k * sin(phi)
  const T pXUnwrapped = utils::wrapToRange(
      inFermiMomentum.first, inOffset.first - static_cast<T>(M_PI),
      inOffset.first + static_cast<T>(M_PI));
  const T pYUnwrapped = utils::wrapToRange(
      inFermiMomentum.second, inOffset.second - static_cast<T>(M_PI),
      inOffset.first + static_cast<T>(M_PI));
  const T kX = pXUnwrapped - inOffset.first;
  const T kY = pYUnwrapped - inOffset.second;
  const T k = std::hypot(kX, kY);
  const T cosPhi = kX / k;
  const T sinPhi = kY / k;

  // The derivative of the dispersion with respect to phi is:
  // dEdPhi = dEdKX * dKXdPhi + dEdKY * dKYdPhi = 0
  // whith
  // dKXdPhi = (dKdPhi * cos(phi) - kY)
  // dKYdPhi = (dKdPhi * sin(phi) + kX)
  // which gives
  const T dKdPhi =
      (dEdKX * kY - dEdKY * kX) / (dEdKX * cosPhi + dEdKY * sinPhi);

  // Thus, the norm of the partial derivative of the Fermi momentum with respect
  // to phi is given by:
  // sqrt(dPXdPhi^2 + dPYdPhi^2) = sqrt(dKXdPhi^2 + dKYdPhi^2)
  const T dKXdPhi = dKdPhi * cosPhi - kY;
  const T dKYdPhi = dKdPhi * sinPhi + kX;
  const T dPdPhi = std::hypot(dKXdPhi, dKYdPhi);

  const T fermiSpeed =
      std::hypot(inFermiVelocity.first, inFermiVelocity.second);
  const T integrandFactor = dPdPhi / fermiSpeed;
  return integrandFactor;
}
} // namespace tightbinding_internal

template <typename T>
FermiSurfaceTightBinding<T>::FermiSurfaceTightBinding(
    const T inHoppingNNN, const T inHoppingNNNN, const T inChemicalPotential)
    : FermiSurface<T>(), m_fermiMomentumX(0), m_fermiMomentumY(0),
      m_fermiVelocityX(0), m_fermiVelocityY(0), m_systemAngles(0),
      m_systemAnglesGPU(0), m_integrandFactor(0), m_avgFermiMomentum(0.0),
      m_avgFermiSpeed(0.0),
      m_parameters(inHoppingNNN, inHoppingNNNN, inChemicalPotential) {}

template <typename T> FermiSurfaceTightBinding<T>::~FermiSurfaceTightBinding() {
  std::cout << "~FermiSurfaceTightBinding<T>\n";
}

template <typename T>
void FermiSurfaceTightBinding<T>::initialize(const int inNumAngles) {
  std::cout << "FermiSurfaceTightBinding<T>::initialize()" << std::endl;

  // The base class needs to know the number of angles.
  FermiSurface<T>::initialize(inNumAngles);

  // Print the tight-binding parameters.
  m_parameters.print();

  // Compute momentum offset.
  m_originOffset = tightbinding_internal::computeMomentumOffset(m_parameters);
  std::cout << "Fermi Surface Origin: (x, y) = (" << m_originOffset.first
            << ", " << m_originOffset.second << ")" << std::endl;

  // Clear all arrays on the CPU.
  m_fermiMomentumX.clear();
  m_fermiMomentumY.clear();
  m_fermiVelocityX.clear();
  m_fermiVelocityY.clear();
  m_systemAngles.clear();
  m_integrandFactor.clear();

  T integrandFactorSum = static_cast<T>(0.0);
  T fermiMomentumSum = static_cast<T>(0.0);
  T fermiSpeedSum = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    // The momentum angle.
    const T phi = FermiSurface<T>::getPolarAngle(angleIdx);

    // Compute the Fermi momentum.
    const std::pair<T, T> fermiMomentum =
        tightbinding_internal::computeFermiMomentum(
            m_parameters, phi, m_originOffset, m_maxIterations, m_convergence);

    // The Fermi velocity is the partial derivative of the dispersion at the
    // Fermi surface.
    const std::pair<T, T> fermiVelocity =
        tightbinding_internal::computeDispersionDerivative(m_parameters,
                                                           fermiMomentum);

    // Update the Fermi momentum arrays.
    m_fermiMomentumX.push_back(fermiMomentum.first);
    m_fermiMomentumY.push_back(fermiMomentum.second);

    // Update the Fermi velocity arrays.
    m_fermiVelocityX.push_back(fermiVelocity.first);
    m_fermiVelocityY.push_back(fermiVelocity.second);

    // Update the system angle array.
    const T fermiVelocityAngle =
        std::atan2(fermiVelocity.second, fermiVelocity.first);
    const T systemAngle = FermiSurface<T>::getSystemAngle(fermiVelocityAngle);
    m_systemAngles.push_back(systemAngle);

    // Compute integrand factor.
    const T factor = tightbinding_internal::computeIntegrandFactor(
        m_parameters, fermiMomentum, fermiVelocity, m_originOffset);
    m_integrandFactor.push_back(factor);

    // For normalizing the integrand factor.
    integrandFactorSum += factor;

    // For computing averages.
    fermiMomentumSum +=
        std::hypot(fermiMomentum.first, fermiMomentum.second) * factor;
    fermiSpeedSum +=
        std::hypot(fermiVelocity.first, fermiVelocity.second) * factor;
  }

  // The average Fermi momentum and speed.
  m_avgFermiMomentum = fermiMomentumSum / integrandFactorSum;
  m_avgFermiSpeed = fermiSpeedSum / integrandFactorSum;
  std::cout << "<|vF|>/(ta) = " << m_avgFermiSpeed
            << ", 1/(a<|kF|>) = " << static_cast<T>(1.0) / m_avgFermiMomentum
            << std::endl;

  // Normalize the Fermi velocity and integrand factor.
  for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
    m_fermiVelocityX[angleIdx] /= m_avgFermiSpeed;
    m_fermiVelocityY[angleIdx] /= m_avgFermiSpeed;
    m_integrandFactor[angleIdx] /= integrandFactorSum;
  }

  // Copy system angles to the GPU.
  m_systemAnglesGPU = m_systemAngles;
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiMomentum(const int inAngleIdx,
                                                   T &outFermiMomentumX,
                                                   T &outFermiMomentumY) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  outFermiMomentumX = m_fermiMomentumX[idx];
  outFermiMomentumY = m_fermiMomentumY[idx];
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiVelocity(const int inAngleIdx,
                                                   T &outFermiVelocityX,
                                                   T &outFermiVelocityY) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  outFermiVelocityX = m_fermiVelocityX[idx];
  outFermiVelocityY = m_fermiVelocityY[idx];
}

template <typename T>
T FermiSurfaceTightBinding<T>::getFermiSpeed(const int inAngleIdx) const {
  T fermiVelocityX;
  T fermiVelocityY;
  getFermiVelocity(inAngleIdx, fermiVelocityX, fermiVelocityY);
  return std::hypot(fermiVelocityX, fermiVelocityY);
}

template <typename T>
T FermiSurfaceTightBinding<T>::getSystemAngle(const int inAngleIdx) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  return m_systemAngles[idx];
}

template <typename T>
const T *FermiSurfaceTightBinding<T>::getSystemAnglePointer() const {
  return thrust::raw_pointer_cast(m_systemAnglesGPU.data());
}

template <typename T>
T FermiSurfaceTightBinding<T>::getIntegrandFactor(const int inAngleIdx) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  return m_integrandFactor[idx];
}

template <typename T>
void FermiSurfaceTightBinding<T>::getMomentumOffset(T &outOffsetX,
                                                    T &outOffsetY) const {
  outOffsetX = m_originOffset.first;
  outOffsetY = m_originOffset.second;
}

#endif // FERMI_SURFACE_TIGHT_BINDING_H_
