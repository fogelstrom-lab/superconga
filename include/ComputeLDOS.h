#ifndef COMPUTE_LDOS_H_
#define COMPUTE_LDOS_H_

#include "ComputeProperty.h"
#include "IntegrationIteratorKeldysh.h"

template <typename T> class ComputeLDOS : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors
  ComputeLDOS();
  virtual ~ComputeLDOS();

  ComputeLDOS(Context<T> *ctx);

  virtual GridGPU *getResult() { return _ldos_all; }
  GridGPU *getResult_interval() { return _ldos; }
  GridCPU *getDOS() { return _dos; }

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
                                  GridGPU *gammabar, GeometryGroup<T> *geom);

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const int angleIdx);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

  // Set wheter to multiply result with fermi function or not.
  int getMultFermiFunction() const { return _multFermiFunction; }
  void setMultFermiFunction(const int mult) { _multFermiFunction = mult; }

private:
  // Computed LDOS will be stored here at end of IntegrationIterator::compute()
  // function.
  GridGPU *_ldos;
  GridGPU *_ldos_all;

  // Integrated DOS for system, for each energy level
  GridCPU *_dos;

  int _multFermiFunction;
};

template <typename T>
ComputeLDOS<T>::ComputeLDOS()
    : ComputeProperty<T>(), _ldos(0), _ldos_all(0), _dos(0),
      _multFermiFunction(0) {}

template <typename T>
ComputeLDOS<T>::ComputeLDOS(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _ldos(0), _ldos_all(0), _dos(0),
      _multFermiFunction(0) {
  ctx->add(this);
}

template <typename T> ComputeLDOS<T>::~ComputeLDOS() {
  if (_ldos)
    delete _ldos;
  if (_ldos_all)
    delete _ldos_all;
  if (_dos)
    delete _dos;
}

template <typename T> int ComputeLDOS<T>::initialize() {
  typedef ContextModule<T> C;

  int N0 = C::getParameters()->getGridResolutionBase();
  int N1 = C::getParameters()->getGridResolutionCoherence();

  int numEnergies = C::getParameters()->getNumEnergies();
  int numEnergiesInterval =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU::initializeGrid(_ldos, N0, N0, numEnergiesInterval, Type::real);
  GridGPU::initializeGrid(_ldos_all, N0, N0, numEnergies, Type::real);
  GridCPU::initializeGrid(_dos, numEnergies, 1, 2, Type::real);

  std::cout << "LDOS mem: "
            << (_ldos_all->numElements() + _ldos->numElements()) * sizeof(T)
            << "\n";

  // Field 0 of _dos contains E, field 1 contains DOS
  _dos->setZero();

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
  if (ComputeProperty<T>::doCompute())
    _ldos->setZero();
}

template <typename T> void ComputeLDOS<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeLDOS<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                        GridGPU *gamma, GridGPU *gammabar,
                                        GeometryGroup<T> *geom) {
  if (ComputeProperty<T>::doCompute()) {
    GreensFunctionStatic<T, Greens>::computeGreens(
        ComputeProperty<T>::getGreensCurrentAngle(), iter, gamma, gammabar,
        geom);
  }
}

template <typename T>
void ComputeLDOS<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                          const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens, 1);

    const T integrandFactor =
        ContextModule<T>::getFermiSurface()->getIntegrandFactor(angleIdx);
    *greens *= integrandFactor;

    const T systemAngle =
        ContextModule<T>::getFermiSurface()->getSystemAngle(angleIdx);
    _ldos->insertRotatedGridAdditive(greens, systemAngle, -1, 0, 0, Type::imag);
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
    const T h = C::getParameters()->getGridElementSize();

    GridGPU *energies = C::getIntegrationIterator()->getEnergies();

    // Compute LDOS
    *_ldos *= -1.0;

    // Add fermifunction
    if (_multFermiFunction) {
      dim3 bpg, tpb;
      _ldos->getKernelDimensionsFields(bpg, tpb);

      addFermiFunction_kernel<<<bpg, tpb>>>(
          _ldos->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, _ldos->dimX());
    }

    // Set values on border (representing fractional lattice sites) to zero.
    C::getGeometry()->zeroValuesOutsideDomain(_ldos);

    // Copy temp LDOS to full grid
    for (int i = 0; i < numEnergiesCurrentBlock; ++i)
      _ldos->copyFieldTo(_ldos_all, i, i + energyIdx_offset);

    // Compute DOS
    T *E_ptr = _dos->getDataPointer(0) + energyIdx_offset;
    T *DOS_ptr = _dos->getDataPointer(1) + energyIdx_offset;

    GridCPU energies_cpu = *energies;
    T *energies_ptr =
        energies_cpu.getDataPointer(0, Type::real) + energyIdx_offset;

    T sum_re, old, tmp, sumdiff = 0.0, sumold = 0.0;

    const int numGrainLatticeSites =
        C::getGeometry()->getNumGrainLatticeSites();
    const T invNumGrainLatticeSites =
        static_cast<T>(1.0) / numGrainLatticeSites;

    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      // Sum for energy level
      sum_re = _ldos->sumField(i);
      old = DOS_ptr[i];
      DOS_ptr[i] = sum_re * invNumGrainLatticeSites;

      tmp = DOS_ptr[i] - old;
      sumdiff += tmp * tmp;
      sumold += old * old;

      // Store energy value
      E_ptr[i] = energies_ptr[i];
    }

    // TODO(niclas): Use computeResidual instead!
    sumdiff = sqrt(sumdiff) / sqrt(sumold + 0.000001);
    ComputeProperty<T>::setResidual(sumdiff);

    if (sumdiff < C::getParameters()->getConvergenceCriterion())
      ComputeProperty<T>::setConverged();
  }
}

#endif // COMPUTE_LDOS_H_
