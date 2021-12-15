//===-- ComputeLDOS.h - Class for local density of states compute object. -===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute object class for local density of states.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_COMPUTE_LDOS_H_
#define CONGA_COMPUTE_LDOS_H_

#include "../Type.h"
#include "../grid.h"
#include "ComputeProperty.h"
#include "ComputeResidual.h"

namespace conga {
template <typename T> class ComputeLDOS : public ComputeProperty<T> {
public:
  // Constructors and Destructors
  ComputeLDOS();
  ~ComputeLDOS();

  ComputeLDOS(Context<T> *ctx);

  GridGPU<T> *getResult() override { return m_ldosAll; }
  GridGPU<T> *getResultInterval() { return m_ldosInterval; }

  int initialize() override;

  // This should be called once for every iteration, before any integration
  // takes place.
  void initializeIteration(int numFields = 1) override;

  // This should be called once for every angle, before any integration takes
  // place.
  void initializeGreensAngle() override;

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  void computeGreensAngle(const GridGPU<T> *const inGamma,
                          const GridGPU<T> *const inGammaTilde) override;

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  void integrateGreensAngle(GeometryGroup<T> *geom,
                            const int angleIdx) override;

  // Compute physical property from greens function.
  void computePropertyFromGreens() override;

private:
  // Computed LDOS will be stored here at end of IntegrationIterator::compute()
  // function.
  GridGPU<T> *m_ldosInterval;

  // The result when every energy level has converged.
  GridGPU<T> *m_ldosAll;
};

template <typename T>
ComputeLDOS<T>::ComputeLDOS()
    : ComputeProperty<T>(), m_ldosInterval(0), m_ldosAll(0) {}

template <typename T>
ComputeLDOS<T>::ComputeLDOS(Context<T> *ctx)
    : ComputeProperty<T>(ctx), m_ldosInterval(0), m_ldosAll(0) {
  ctx->add(this);
}

template <typename T> ComputeLDOS<T>::~ComputeLDOS() {
  if (m_ldosInterval) {
    delete m_ldosInterval;
  }
  if (m_ldosAll) {
    delete m_ldosAll;
  }
}

template <typename T> int ComputeLDOS<T>::initialize() {
  typedef ContextModule<T> C;

  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  const int numEnergies = C::getParameters()->getNumEnergies();
  const int numEnergiesInterval =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU<T>::initializeGrid(m_ldosInterval, N0, N0, numEnergiesInterval,
                             Type::real);
  GridGPU<T>::initializeGrid(m_ldosAll, N0, N0, numEnergies, Type::real);

#ifndef NDEBUG
  const int memoryLDOS =
      (m_ldosAll->numElements() + m_ldosInterval->numElements()) *
      static_cast<int>(sizeof(T));
  std::cout << "LDOS memory: " << memoryLDOS << "\n";
#endif

  ComputeProperty<T>::initGreensAngle(N1, numEnergiesInterval);
  return 0;
}

template <typename T> void ComputeLDOS<T>::initializeIteration(int numFields) {
  // Unused.
  (void)numFields;

  if (ComputeProperty<T>::doCompute()) {
    m_ldosInterval->setZero();
  }
}

template <typename T> void ComputeLDOS<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeLDOS<T>::computeGreensAngle(const GridGPU<T> *const inGamma,
                                        const GridGPU<T> *const inGammaTilde) {
  if (ComputeProperty<T>::doCompute()) {
    GreensFunctionStatic<T, Greens>::computeGreens(
        inGamma, inGammaTilde, ComputeProperty<T>::getGreensCurrentAngle());
  }
}

template <typename T>
void ComputeLDOS<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                          const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU<T> *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens);

    const T integrandFactor =
        ContextModule<T>::getFermiSurface()->getIntegrandFactor(angleIdx);
    *greens *= integrandFactor;

    const T systemAngle =
        ContextModule<T>::getFermiSurface()->getSystemAngle(angleIdx);
    m_ldosInterval->insertRotatedGridAdditive(greens, systemAngle, -1, 0, 0,
                                              Type::imag);
  }
}

template <typename T> void ComputeLDOS<T>::computePropertyFromGreens() {
  if (ComputeProperty<T>::doCompute()) {
    typedef ContextModule<T> C;

    // Get some info
    const ParameterIterator<int> *iter =
        C::getIntegrationIterator()->getEnergyIterator();

    const int numEnergiesCurrentBlock = iter->getNumInterval();
    const int energyIdx_offset = iter->getCurrentValue();

    // Compute LDOS
    *m_ldosInterval *= -1.0;

    C::getGeometry()->zeroValuesOutsideDomainRef(m_ldosInterval);

    const int dimX = m_ldosInterval->dimX();
    const int dimY = m_ldosInterval->dimY();
    const Type::type representation = m_ldosInterval->representation();
    GridGPU<T> ldosIntervalOld(dimX, dimY, numEnergiesCurrentBlock,
                               representation);

    // Copy temp LDOS to full grid.
    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      m_ldosAll->copyFieldTo(&ldosIntervalOld, i + energyIdx_offset, i);
      m_ldosInterval->copyFieldTo(m_ldosAll, i, i + energyIdx_offset);
    }

    // Compute residual.
    const T residual = computeResidual(ldosIntervalOld, *m_ldosInterval);
    ComputeProperty<T>::setResidual(residual);

    // Check convergence.
    const bool isPartiallyConverged =
        residual < C::getParameters()->getConvergenceCriterion();
    ComputeProperty<T>::setPartiallyConverged(isPartiallyConverged);
  }
}
} // namespace conga

#endif // CONGA_COMPUTE_LDOS_H_
