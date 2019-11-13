#ifndef _COMPUTECURRENTDENSITY_H
#define _COMPUTECURRENTDENSITY_H

#include <ComputeProperty.h>
#include <IntegrationIteratorKeldysh.h>

template <typename T> class ComputeCurrentDensity : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors
  ComputeCurrentDensity();
  virtual ~ComputeCurrentDensity();

  ComputeCurrentDensity(Context<T> *ctx);

  virtual GridGPU *getResult() { return _lcurrent_all; }
  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1);

  // Context initialize function
  virtual int initialize();

  // This should be called once for every iteration, before any integration
  // takes place.
  virtual void initializeIteration(int numFields = 1);

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  virtual void computeGreensAngle(ParameterIterator<int> *iter, GridGPU *gamma,
                                  GridGPU *gammabar, GeometryGroup<T> *geom);

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const T angle);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

  GridGPU *getResult_interval(int field = 0) {
    if (field == 0)
      return _lcurrent_x;
    else
      return _lcurrent_y;
  }

  GridCPU *getCurrentDensity() const { return _current; }

  int getMultFermiFunction() const { return _multFermiFunction; }
  void setMultFermiFunction(const int mult) { _multFermiFunction = mult; }

private:
  // Computed local current density will be stored here at end of
  // IntegrationIterator::compute() function.
  GridGPU *_lcurrent_x;
  GridGPU *_lcurrent_y;

  // Total current density for system, for each energy level
  GridGPU *_lcurrent_all;

  // Integrated current, per energy level
  GridCPU *_current;

  // Temporary grid.
  GridGPU *_tmp;

  // Multiply with fermi function
  int _multFermiFunction;
};

template <typename T>
ComputeCurrentDensity<T>::ComputeCurrentDensity()
    : ComputeProperty<T>(), _lcurrent_x(0), _lcurrent_y(0), _lcurrent_all(0),
      _current(0), _tmp(0), _multFermiFunction(0) {}

template <typename T>
ComputeCurrentDensity<T>::ComputeCurrentDensity(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _lcurrent_x(0), _lcurrent_y(0), _lcurrent_all(0),
      _current(0), _tmp(0), _multFermiFunction(0) {
  ctx->add(this);
}

template <typename T> ComputeCurrentDensity<T>::~ComputeCurrentDensity() {
  if (_lcurrent_x)
    delete _lcurrent_x;
  if (_lcurrent_y)
    delete _lcurrent_y;
  if (_lcurrent_all)
    delete _lcurrent_all;
  if (_current)
    delete _current;
  if (_tmp)
    delete _tmp;
}

template <typename T>
IntegrationIterator<T> *
ComputeCurrentDensity<T>::createIntegrationIterator(int numEnergiesPerBlock) {
  return new IntegrationIteratorKeldysh<T>(ContextModule<T>::getContext(),
                                           numEnergiesPerBlock);
}

template <typename T> int ComputeCurrentDensity<T>::initialize() {
  typedef ContextModule<T> C;

  int N0 = C::getParameters()->getGridResolutionBase();
  int N1 = C::getParameters()->getGridResolutionCoherence();

  int numEnergies = C::getParameters()->getNumEnergies();
  int numEnergiesInterval =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU::initializeGrid(_lcurrent_all, N0, N0, 2 * numEnergies, Type::real);
  GridGPU::initializeGrid(_lcurrent_x, N0, N0, numEnergiesInterval, Type::real);
  GridGPU::initializeGrid(_lcurrent_y, N0, N0, numEnergiesInterval, Type::real);
  GridGPU::initializeGrid(_tmp, N0, N0, numEnergiesInterval, Type::real);
  GridCPU::initializeGrid(_current, numEnergies, 1, 2, Type::real);

  _current->setZero();

  ComputeProperty<T>::initGreensAngle(N1, numEnergiesInterval);

  std::cout << "CurrentDensity mem: "
            << (_lcurrent_all->numElements() + 3 * _lcurrent_x->numElements()) *
                   sizeof(T)
            << "\n";

  return 0;
}

template <typename T>
void ComputeCurrentDensity<T>::initializeIteration(int numFields) {
  if (ComputeProperty<T>::doCompute()) {
    _lcurrent_x->setZero();
    _lcurrent_y->setZero();
    _tmp->setZero();
  }
}

template <typename T>
void ComputeCurrentDensity<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                                  GridGPU *gamma,
                                                  GridGPU *gammabar,
                                                  GeometryGroup<T> *geom) {
  if (ComputeProperty<T>::doCompute()) {
    GreensFunctionStatic<T, Greens>::computeGreens(
        ComputeProperty<T>::getGreensCurrentAngle(), iter, gamma, gammabar,
        geom);
  }
}

template <typename T>
void ComputeCurrentDensity<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                                    const T angle) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens, 1);

    const T momentum_angle = angle + 0.5 * M_PI;

    _tmp->insertRotatedGrid(greens, angle, 0, 0, Type::imag);

    *_lcurrent_x += *_tmp * cos(momentum_angle);
    *_lcurrent_y += *_tmp * sin(momentum_angle);
  }
}

template <typename T>
void ComputeCurrentDensity<T>::computePropertyFromGreens() {
  if (ComputeProperty<T>::doCompute()) {
    typedef ContextModule<T> C;

    // Get some info
    const T h = C::getParameters()->getGridElementSize();
    const int numAngles = C::getParameters()->getAngularResolution();
    const T T_Tc = C::getParameters()->getTemperature();

    const ParameterIterator<int> *iter =
        C::getIntegrationIterator()->getEnergyIterator();

    int numEnergiesCurrentBlock = iter->getNumInterval();
    int energyIdx_offset = iter->getCurrentValue();

    GridGPU *energies = C::getIntegrationIterator()->getEnergies();

    // Add fermifunction
    if (_multFermiFunction) {
      dim3 bpg, tpb;
      _lcurrent_x->getKernelDimensionsFields(bpg, tpb);

      addFermiFunction_kernel<<<bpg, tpb>>>(
          _lcurrent_x->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, _lcurrent_x->dimX());

      addFermiFunction_kernel<<<bpg, tpb>>>(
          _lcurrent_y->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, _lcurrent_y->dimX());
    }

    *_lcurrent_x *= -1.0 / (2.0 * M_PI * (T)numAngles);
    *_lcurrent_y *= -1.0 / (2.0 * M_PI * (T)numAngles);

    // Copy currents to full storage
    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      _lcurrent_x->copyFieldTo(_lcurrent_all, i, 2 * (i + energyIdx_offset));
      _lcurrent_y->copyFieldTo(_lcurrent_all, i,
                               2 * (i + energyIdx_offset) + 1);
    }

    // Integrate local current density (used to compute residual)
    GridCPU energies_cpu = *energies;
    T *energies_ptr =
        energies_cpu.getDataPointer(0, Type::real) + energyIdx_offset;

    T *E_ptr = _current->getDataPointer(0) + energyIdx_offset;
    T *j_ptr = _current->getDataPointer(1) + energyIdx_offset;

    T sum_x, sum_y, old, tmp, sumdiff = 0.0, sumold = 0.0;

    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      // Sum for energy level
      sum_x = _lcurrent_x->sumField(i);
      sum_y = _lcurrent_y->sumField(i);

      old = j_ptr[i];
      j_ptr[i] = sqrt(sum_x * sum_x + sum_y * sum_y) * h * h;

      tmp = j_ptr[i] - old;
      sumdiff += tmp * tmp;
      sumold += old * old;

      // Store energy value
      E_ptr[i] = energies_ptr[i];
    }

    sumdiff = sqrt(sumdiff) / sqrt(sumold + 0.000001);
    ComputeProperty<T>::setResidual(sumdiff);

    if (sumdiff < C::getParameters()->getConvergenceCriterion())
      ComputeProperty<T>::setConverged();
  }
}
#endif
