#ifndef FERMI_SURFACE_TIGHT_BINDING_H_
#define FERMI_SURFACE_TIGHT_BINDING_H_

#include "FermiSurface.h"
#include "defines.h"
#include "dispersion.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Fermi surface from a tight-binding dispersion.
// Note, there are tacit assumptions:
// 1. A single species of atom.
// 2. A square lattice.
template <typename T> class FermiSurfaceTightBinding : public FermiSurface<T> {
public:
  // Constructors.
  FermiSurfaceTightBinding(const T inHoppingNNN = static_cast<T>(0.0),
                           const T inHoppingNNNN = static_cast<T>(0.0),
                           const T inChemicalPotential = static_cast<T>(1.0));
  FermiSurfaceTightBinding(const DispersionTightBinding<T> &inDispersion);

  // Destructor.
  ~FermiSurfaceTightBinding();

  // Initialize the Fermi surface.
  void initialize(const int inNumAngles);

  // Get the Fermi momentum.
  vector2d<T> getFermiMomentum(const int inAngleIdx) const;
  void getFermiMomentum(const int inAngleIdx, T &outMomentumX,
                        T &outMomentumY) const;

  // Get the Fermi velocity.
  vector2d<T> getFermiVelocity(const int inAngleIdx) const;
  void getFermiVelocity(const int inAngleIdx, T &outVelocityX,
                        T &outVelocityY) const;

  // Get the Fermi speed.
  T getFermiSpeed(const int inAngleIdx) const;

  // Get the integrand factor.
  T getIntegrandFactor(const int inAngleIdx) const;

  // Get system angle.
  T getSystemAngle(const int inAngleIdx) const;
  const T *getSystemAnglePointer() const;

  // Get momentum offset.
  vector2d<T> getMomentumOffset() const;
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

  // The dispersion.
  const DispersionTightBinding<T> m_dispersion;

  // For doped materials the Fermi surface can encircle the point (pi, pi)
  // instead of (0, 0), this is simply that offset.
  vector2d<T> m_originOffset;

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

namespace tightbinding_internal {
/// \brief Compute the momentum offset for this dispersion
///
/// The Fermi surface (given reasonable parameters) is either centered around
/// (0,0) or (pi,pi), this function computes that offset.
///
/// \param inDispersion - The tight-binding dispersion for a square lattice
///
/// \return - The offset, either (0,0) or (pi, pi)
template <typename T>
vector2d<T>
computeMomentumOffset(const DispersionTightBinding<T> &inDispersion) {
  // Compute the dispersion at k=(0,pi).
  const vector2d<T> kA = {static_cast<T>(0.0), static_cast<T>(M_PI)};
  const T eA = inDispersion.evaluate(kA);

  // Compute the dispersion at k=(pi,pi).
  const vector2d<T> kB = {static_cast<T>(M_PI), static_cast<T>(M_PI)};
  const T eB = inDispersion.evaluate(kB);

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

/// \brief Compute the Fermi momentum for a given dispersion and direction
///
/// Essentially it computes k in
/// kx = k * cos(angle) + offsetX
/// ky = k * sin(angle) + offsetY
/// where k is given as the root of the dispersion going from the offset along a
/// ray with the given angle.
///
/// \param inDispersion - The tight-binding dispersion for a square lattice
/// \param inAngle - The angle in radians
/// \param inOffset - The offset (x, y)
/// \param inNumIterations - The maximum number of iterations
/// \param inConvergenCriterion - The convergence criterion. We stop when the
///                               error is smaller than this
///
/// \return - The Fermi momentum (x, y)
template <typename T>
vector2d<T> computeFermiMomentum(
    const DispersionTightBinding<T> &inDispersion, const T inAngle,
    const vector2d<T> inOriginOffset, const int inNumIterations = 100,
    const T inConvergenCriterion = static_cast<T>(1e2) *
                                   std::numeric_limits<T>::epsilon(),
    const bool inWrapToBrillouinZone = true) {
  // For convenience.
  const vector2d<T> unitVector = {std::cos(inAngle), std::sin(inAngle)};

  // For the bisection method we need to two end-points of the interval.
  // 1st point: The origin (which could be (pi, pi)).
  T kA = static_cast<T>(0.0);
  const vector2d<T> momentumA = utils::multAdd(kA, unitVector, inOriginOffset);
  T eA = inDispersion.evaluate(momentumA);

  // 2nd point: The BZ "edge".
  const T maxAbsComponent =
      std::max(std::abs(unitVector[0]), std::abs(unitVector[1]));
  T kB = static_cast<T>(M_PI) / maxAbsComponent;
  const vector2d<T> momentumB = utils::multAdd(kB, unitVector, inOriginOffset);
  T eB = inDispersion.evaluate(momentumB);

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
    const vector2d<T> momentum = utils::multAdd(k, unitVector, inOriginOffset);

    // Compute the dispersion.
    const T dispersion = inDispersion.evaluate(momentum);

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
      const vector2d<T> dE = inDispersion.derivative(momentum);
      // dXdK = cos(phi), dYdK = sin(phi).
      const T dEdK = utils::dot(dE, unitVector);

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
  vector2d<T> momentum = utils::multAdd(k, unitVector, inOriginOffset);
  if (inWrapToBrillouinZone) {
    for (int i = 0; i < 2; ++i) {
      momentum[i] = utils::wrapToRange(momentum[i], -static_cast<T>(M_PI),
                                       static_cast<T>(M_PI));
    }
  }
  return momentum;
}

/// \brief Compute the integrand factor for a Fermi momentum
///
/// The factor is given by the norm of the partial derivative of the Fermi
/// momentum with the respect to the angle phi, divided by the norm of the Fermi
/// velocity. But it can be simplified a lot. See Eq. (16) on page 7 (1105) in
/// "The Effect of Surfaces on the Tunneling Density of States of an
/// Anisotropically Paired Superconductor" by Buchholtz et. al. (1995).
///
/// \param inFermiMomentum - The Fermi momentum (x, y)
/// \param inFermiVelocity - The Fermi velocity (x, y)
/// \param inOffset - The momentum offset (x, y)
///
/// \return - The integrand factor
template <typename T>
T computeIntegrandFactor(const vector2d<T> &inFermiMomentum,
                         const vector2d<T> &inFermiVelocity,
                         const vector2d<T> &inOffset) {
  // The momentum is parameterized as: p = k + offset.
  // To get k we must unwrap p at the offset, and then subtract it.
  vector2d<T> k;
  for (int i = 0; i < 2; ++i) {
    const T pUnwrapped = utils::wrapToRange(inFermiMomentum[i],
                                            inOffset[i] - static_cast<T>(M_PI),
                                            inOffset[i] + static_cast<T>(M_PI));
    k[i] = pUnwrapped - inOffset[i];
  }

  // Buchholtz simplified expression.
  const T integrandFactor =
      std::abs(utils::dot(k, k) / utils::dot(k, inFermiVelocity));

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
      m_dispersion(inHoppingNNN, inHoppingNNNN, inChemicalPotential) {}

template <typename T>
FermiSurfaceTightBinding<T>::FermiSurfaceTightBinding(
    const DispersionTightBinding<T> &inDispersion)
    : FermiSurface<T>(), m_fermiMomentumX(0), m_fermiMomentumY(0),
      m_fermiVelocityX(0), m_fermiVelocityY(0), m_systemAngles(0),
      m_systemAnglesGPU(0), m_integrandFactor(0), m_avgFermiMomentum(0.0),
      m_avgFermiSpeed(0.0), m_dispersion(inDispersion) {}

template <typename T> FermiSurfaceTightBinding<T>::~FermiSurfaceTightBinding() {
  std::cout << "~FermiSurfaceTightBinding<T>\n";
}

template <typename T>
void FermiSurfaceTightBinding<T>::initialize(const int inNumAngles) {
  std::cout << "FermiSurfaceTightBinding<T>::initialize()" << std::endl;

  // The base class needs to know the number of angles.
  FermiSurface<T>::initialize(inNumAngles);

  // Compute momentum offset.
  m_originOffset = tightbinding_internal::computeMomentumOffset(m_dispersion);
  std::cout << "Fermi Surface Origin: (x, y) = (" << m_originOffset[0] << ", "
            << m_originOffset[1] << ")" << std::endl;

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
    const vector2d<T> fermiMomentum =
        tightbinding_internal::computeFermiMomentum(
            m_dispersion, phi, m_originOffset, m_maxIterations, m_convergence);

    // The Fermi velocity is the partial derivative of the dispersion at the
    // Fermi surface.
    const vector2d<T> fermiVelocity = m_dispersion.derivative(fermiMomentum);

    // Update the Fermi momentum arrays.
    m_fermiMomentumX.push_back(fermiMomentum[0]);
    m_fermiMomentumY.push_back(fermiMomentum[1]);

    // Update the Fermi velocity arrays.
    m_fermiVelocityX.push_back(fermiVelocity[0]);
    m_fermiVelocityY.push_back(fermiVelocity[1]);

    // Update the system angle array.
    const T fermiVelocityAngle = std::atan2(fermiVelocity[1], fermiVelocity[0]);
    const T systemAngle = FermiSurface<T>::getSystemAngle(fermiVelocityAngle);
    m_systemAngles.push_back(systemAngle);

    // Compute integrand factor.
    const T factor = tightbinding_internal::computeIntegrandFactor(
        fermiMomentum, fermiVelocity, m_originOffset);
    m_integrandFactor.push_back(factor);

    // For normalizing the integrand factor.
    integrandFactorSum += factor;

    // For computing averages.
    fermiMomentumSum += utils::normL2(fermiMomentum) * factor;
    fermiSpeedSum += utils::normL2(fermiVelocity) * factor;
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
vector2d<T>
FermiSurfaceTightBinding<T>::getFermiMomentum(const int inAngleIdx) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  return {m_fermiMomentumX[idx], m_fermiMomentumY[idx]};
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiMomentum(const int inAngleIdx,
                                                   T &outFermiMomentumX,
                                                   T &outFermiMomentumY) const {
  const vector2d<T> fermiMomentum = getFermiMomentum(inAngleIdx);
  outFermiMomentumX = fermiMomentum[0];
  outFermiMomentumY = fermiMomentum[1];
}

template <typename T>
vector2d<T>
FermiSurfaceTightBinding<T>::getFermiVelocity(const int inAngleIdx) const {
  const int idx = FermiSurface<T>::getWrappedAngleIndex(inAngleIdx);
  return {m_fermiVelocityX[idx], m_fermiVelocityY[idx]};
}

template <typename T>
void FermiSurfaceTightBinding<T>::getFermiVelocity(const int inAngleIdx,
                                                   T &outFermiVelocityX,
                                                   T &outFermiVelocityY) const {
  const vector2d<T> fermiVelocity = getFermiVelocity(inAngleIdx);
  outFermiVelocityX = fermiVelocity[0];
  outFermiVelocityY = fermiVelocity[1];
}

template <typename T>
T FermiSurfaceTightBinding<T>::getFermiSpeed(const int inAngleIdx) const {
  const vector2d<T> fermiVelocity = getFermiVelocity(inAngleIdx);
  return utils::normL2(fermiVelocity);
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
vector2d<T> FermiSurfaceTightBinding<T>::getMomentumOffset() const {
  return m_originOffset;
}

template <typename T>
void FermiSurfaceTightBinding<T>::getMomentumOffset(T &outOffsetX,
                                                    T &outOffsetY) const {
  outOffsetX = m_originOffset[0];
  outOffsetY = m_originOffset[1];
}

#endif // FERMI_SURFACE_TIGHT_BINDING_H_
