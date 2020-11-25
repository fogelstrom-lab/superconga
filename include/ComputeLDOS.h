#ifndef COMPUTE_LDOS_H_
#define COMPUTE_LDOS_H_

#include "ComputeProperty.h"
#include "ComputeResidual.h"
#include "IntegrationIteratorKeldysh.h"

namespace scg {
template <typename T> class ComputeLDOS : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors
  ComputeLDOS();
  virtual ~ComputeLDOS();

  ComputeLDOS(Context<T> *ctx);

  virtual GridGPU *getResult() { return m_ldosAll; }
  GridGPU *getResultInterval() { return m_ldosInterval; }
  GridCPU *getDOS() { return m_dos; }

  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1);

  virtual int initialize();

  // This should be called once for every iteration, before any integration
  // takes place.
  virtual void initializeIteration(int numFields = 1);

  // This should be called once for every angle, before any integration takes
  // place.
  virtual void initializeGreensAngle();

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  virtual void computeGreensAngle(ParameterIterator<int> *iter, GridGPU *gamma,
                                  GridGPU *gammabar);

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const int angleIdx);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

  // Set whether to multiply result with Fermi function or not.
  bool getMultFermiFunction() const { return m_multFermiFunction; }
  void setMultFermiFunction(const bool mult) { m_multFermiFunction = mult; }

private:
  // Computed LDOS will be stored here at end of IntegrationIterator::compute()
  // function.
  GridGPU *m_ldosInterval;

  // The result when every energy level has converged.
  GridGPU *m_ldosAll;

  // Integrated DOS for system, for each energy level.
  GridCPU *m_dos;

  bool m_multFermiFunction;
};

template <typename T>
ComputeLDOS<T>::ComputeLDOS()
    : ComputeProperty<T>(), m_ldosInterval(0), m_ldosAll(0), m_dos(0),
      m_multFermiFunction(false) {}

template <typename T>
ComputeLDOS<T>::ComputeLDOS(Context<T> *ctx)
    : ComputeProperty<T>(ctx), m_ldosInterval(0), m_ldosAll(0), m_dos(0),
      m_multFermiFunction(false) {
  ctx->add(this);
}

template <typename T> ComputeLDOS<T>::~ComputeLDOS() {
  if (m_ldosInterval) {
    delete m_ldosInterval;
  }
  if (m_ldosAll) {
    delete m_ldosAll;
  }
  if (m_dos) {
    delete m_dos;
  }
}

template <typename T> int ComputeLDOS<T>::initialize() {
  typedef ContextModule<T> C;

  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  const int numEnergies = C::getParameters()->getNumEnergies();
  const int numEnergiesInterval =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU::initializeGrid(m_ldosInterval, N0, N0, numEnergiesInterval,
                          Type::real);
  GridGPU::initializeGrid(m_ldosAll, N0, N0, numEnergies, Type::real);
  // Field 0 of m_dos contains E, field 1 contains DOS.
  GridCPU::initializeGrid(m_dos, numEnergies, 1, 2, Type::real);

  const int memoryLDOS =
      (m_ldosAll->numElements() + m_ldosInterval->numElements()) *
      static_cast<int>(sizeof(T));
  std::cout << "LDOS memory: " << memoryLDOS << "\n";

  m_dos->setZero();

  ComputeProperty<T>::initGreensAngle(N1, numEnergiesInterval);
  return 0;
}

template <typename T>
IntegrationIterator<T> *
ComputeLDOS<T>::createIntegrationIterator(int numEnergiesPerBlock) {
  return new IntegrationIteratorKeldysh<T>(ContextModule<T>::getContext(),
                                           numEnergiesPerBlock);
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
void ComputeLDOS<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                        GridGPU *gamma, GridGPU *gammabar) {
  if (ComputeProperty<T>::doCompute()) {
    GreensFunctionStatic<T, Greens>::computeGreens(
        ComputeProperty<T>::getGreensCurrentAngle(), iter, gamma, gammabar);
  }
}

template <typename T>
void ComputeLDOS<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                          const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU *greens = ComputeProperty<T>::getGreensCurrentAngle();

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

    const int numAngles = C::getParameters()->getAngularResolution();
    const T T_Tc = C::getParameters()->getTemperature();

    GridGPU *energies = C::getIntegrationIterator()->getEnergies();

    // Compute LDOS
    *m_ldosInterval *= -1.0;

    // Add fermifunction
    if (m_multFermiFunction) {
      dim3 bpg, tpb;
      m_ldosInterval->getKernelDimensionsFields(bpg, tpb);

      compute_property_internal::addFermiFunction_kernel<<<bpg, tpb>>>(
          m_ldosInterval->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, m_ldosInterval->dimX());
    }

    C::getGeometry()->zeroValuesOutsideDomain(m_ldosInterval);

    // Copy temp LDOS to full grid.
    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      m_ldosInterval->copyFieldTo(m_ldosAll, i, i + energyIdx_offset);
    }

    // Compute DOS
    T *E_ptr = m_dos->getDataPointer(0) + energyIdx_offset;
    T *DOS_ptr = m_dos->getDataPointer(1) + energyIdx_offset;

    GridCPU energies_cpu = *energies;
    T *energies_ptr =
        energies_cpu.getDataPointer(0, Type::real) + energyIdx_offset;

    const int numGrainLatticeSites =
        C::getGeometry()->getNumGrainLatticeSites();
    const T invNumGrainLatticeSites =
        static_cast<T>(1.0) / numGrainLatticeSites;

    T diffSquaredSum = static_cast<T>(0.0);
    T oldSquaredSum = static_cast<T>(0.0);
    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      const T sumLDOS = m_ldosInterval->sumField(i);
      const T newDOS = sumLDOS * invNumGrainLatticeSites;
      const T oldDOS = DOS_ptr[i];

      const T diff = newDOS - oldDOS;
      diffSquaredSum += diff * diff;
      oldSquaredSum += oldDOS * oldDOS;

      // Store energy and DOS values.
      E_ptr[i] = energies_ptr[i];
      DOS_ptr[i] = newDOS;
    }

    // TODO(niclas): Use computeResidual.
    const T epsilon =
        compute_residual_internal::getGoodEpsilon<T>(numEnergiesCurrentBlock);
    const T diffL2Norm = std::sqrt(diffSquaredSum);
    const T oldL2Norm = std::sqrt(oldSquaredSum);
    const T residual = diffL2Norm / std::max(oldL2Norm, epsilon);

    ComputeProperty<T>::setResidual(residual);

    const bool isPartiallyConverged =
        residual < C::getParameters()->getConvergenceCriterion();
    ComputeProperty<T>::setPartiallyConverged(isPartiallyConverged);
  }
}
} // namespace scg

#endif // COMPUTE_LDOS_H_
